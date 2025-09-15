"""PubMed E-utilities client for evidence retrieval."""

import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class PubMedClient:
    """Client for PubMed E-utilities API."""
    
    def __init__(self, base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"):
        self.base_url = base_url
        self.email = "ai-wellness@example.com"  # Required by NCBI
        self.tool = "ai-wellness-assistant"
        self.api_key = None  # Optional API key for higher rate limits
        
        # Rate limiting: 3 requests per second without API key, 10 with API key
        self.rate_limit = 0.34 if not self.api_key else 0.1
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> str:
        """Make rate-limited request to PubMed API."""
        await self._rate_limit()
        
        # Add common parameters
        common_params = {
            'email': self.email,
            'tool': self.tool
        }
        if self.api_key:
            common_params['api_key'] = self.api_key
        
        params.update(common_params)
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.text
                
        except httpx.RequestError as e:
            logger.error("PubMed API request failed", error=str(e), url=url)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("PubMed API error", status=e.response.status_code, url=url)
            raise
    
    async def search(self, query: str, max_results: int = 20, 
                    date_limit_years: int = 10) -> List[str]:
        """Search PubMed and return PMIDs."""
        try:
            # Add date filter to get recent publications
            date_filter = datetime.now() - timedelta(days=365 * date_limit_years)
            date_str = date_filter.strftime("%Y/%m/%d")
            
            # Enhance query with date and study type filters
            enhanced_query = f"({query}) AND ({date_str}[PDAT]:3000[PDAT])"
            
            params = {
                'db': 'pubmed',
                'term': enhanced_query,
                'retmax': max_results,
                'retmode': 'xml',
                'sort': 'relevance',
                'field': 'title/abstract'
            }
            
            logger.info("Searching PubMed", query=query, max_results=max_results)
            
            response_text = await self._make_request('esearch.fcgi', params)
            
            # Parse XML response
            root = ET.fromstring(response_text)
            
            # Extract PMIDs
            pmids = []
            for id_elem in root.findall('.//Id'):
                pmids.append(id_elem.text)
            
            logger.info("PubMed search completed", 
                       query=query, 
                       results_found=len(pmids))
            
            return pmids
            
        except Exception as e:
            logger.error("PubMed search failed", query=query, error=str(e))
            return []
    
    async def get_summaries(self, pmids: List[str]) -> List[Dict[str, Any]]:
        """Get article summaries for given PMIDs."""
        if not pmids:
            return []
        
        try:
            # Process in batches to avoid URL length limits
            batch_size = 50
            all_summaries = []
            
            for i in range(0, len(pmids), batch_size):
                batch_pmids = pmids[i:i + batch_size]
                
                params = {
                    'db': 'pubmed',
                    'id': ','.join(batch_pmids),
                    'retmode': 'xml'
                }
                
                response_text = await self._make_request('esummary.fcgi', params)
                
                # Parse summaries
                summaries = self._parse_summaries(response_text)
                all_summaries.extend(summaries)
            
            logger.info("Retrieved PubMed summaries", count=len(all_summaries))
            return all_summaries
            
        except Exception as e:
            logger.error("Failed to get PubMed summaries", error=str(e))
            return []
    
    def _parse_summaries(self, xml_text: str) -> List[Dict[str, Any]]:
        """Parse esummary XML response."""
        summaries = []
        
        try:
            root = ET.fromstring(xml_text)
            
            for doc_sum in root.findall('.//DocSum'):
                pmid = None
                title = ""
                authors = []
                journal = ""
                pub_date = ""
                study_type = "unknown"
                
                # Extract PMID
                pmid_elem = doc_sum.find('.//Id')
                if pmid_elem is not None:
                    pmid = pmid_elem.text
                
                # Extract fields
                for item in doc_sum.findall('.//Item'):
                    name = item.get('Name', '')
                    
                    if name == 'Title':
                        title = item.text or ""
                    elif name == 'AuthorList':
                        for author in item.findall('.//Item'):
                            authors.append(author.text or "")
                    elif name == 'Source':
                        journal = item.text or ""
                    elif name == 'PubDate':
                        pub_date = item.text or ""
                    elif name == 'PublicationTypeList':
                        # Determine study type from publication types
                        pub_types = []
                        for pub_type in item.findall('.//Item'):
                            pub_types.append((pub_type.text or "").lower())
                        
                        study_type = self._classify_study_type(pub_types)
                
                if pmid:
                    summary = {
                        'pmid': pmid,
                        'title': title,
                        'authors': authors[:5],  # Limit to first 5 authors
                        'journal': journal,
                        'pub_date': pub_date,
                        'study_type': study_type,
                        'url': f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        'quality_score': self._calculate_study_quality(study_type, journal)
                    }
                    summaries.append(summary)
            
        except ET.ParseError as e:
            logger.error("Failed to parse PubMed XML", error=str(e))
        
        return summaries
    
    def _classify_study_type(self, pub_types: List[str]) -> str:
        """Classify study type based on publication types."""
        pub_types_str = ' '.join(pub_types)
        
        if any(term in pub_types_str for term in ['randomized controlled trial', 'clinical trial']):
            return 'rct'
        elif any(term in pub_types_str for term in ['meta-analysis', 'systematic review']):
            return 'meta_analysis'
        elif 'cohort study' in pub_types_str:
            return 'cohort'
        elif 'case-control study' in pub_types_str:
            return 'case_control'
        elif any(term in pub_types_str for term in ['case report', 'case series']):
            return 'case_series'
        elif any(term in pub_types_str for term in ['review', 'practice guideline']):
            return 'review'
        else:
            return 'observational'
    
    def _calculate_study_quality(self, study_type: str, journal: str) -> float:
        """Calculate quality score based on study type and journal."""
        # Base scores by study type
        type_scores = {
            'meta_analysis': 0.95,
            'rct': 0.90,
            'cohort': 0.75,
            'case_control': 0.65,
            'observational': 0.55,
            'case_series': 0.35,
            'review': 0.60,
            'unknown': 0.50
        }
        
        base_score = type_scores.get(study_type, 0.50)
        
        # Journal impact factor approximation (simplified)
        high_impact_journals = [
            'new england journal of medicine', 'lancet', 'jama', 'nature medicine',
            'diabetes care', 'circulation', 'american journal of cardiology',
            'journal of the american college of cardiology', 'european heart journal'
        ]
        
        journal_lower = journal.lower()
        if any(hj in journal_lower for hj in high_impact_journals):
            journal_bonus = 0.1
        else:
            journal_bonus = 0.0
        
        return min(1.0, base_score + journal_bonus)
    
    async def search_and_summarize(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform search and get summaries in one call."""
        try:
            # Search for PMIDs
            pmids = await self.search(query, max_results)
            
            if not pmids:
                return []
            
            # Get summaries
            summaries = await self.get_summaries(pmids)
            
            # Sort by quality score
            summaries.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            return summaries
            
        except Exception as e:
            logger.error("PubMed search and summarize failed", query=query, error=str(e))
            return []
    
    def build_clinical_query(self, condition: str, intervention: str, 
                           outcome: str = None) -> str:
        """Build PICO-formatted clinical query."""
        query_parts = []
        
        # Population/Problem
        query_parts.append(f'"{condition}"[MeSH Terms] OR "{condition}"[Title/Abstract]')
        
        # Intervention
        query_parts.append(f'"{intervention}"[MeSH Terms] OR "{intervention}"[Title/Abstract]')
        
        # Outcome
        if outcome:
            query_parts.append(f'"{outcome}"[Title/Abstract]')
        
        # Add study type filters for higher quality evidence
        study_filters = [
            '"randomized controlled trial"[Publication Type]',
            '"meta-analysis"[Publication Type]',
            '"systematic review"[Publication Type]',
            '"clinical trial"[Publication Type]'
        ]
        
        query = f"({' AND '.join(query_parts)}) AND ({' OR '.join(study_filters)})"
        
        return query
