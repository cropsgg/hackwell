"""PubMed E-utilities client for evidence retrieval."""

import asyncio
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import structlog
import httpx
from pydantic import BaseModel

logger = structlog.get_logger()


class PubMedArticle(BaseModel):
    """PubMed article model."""
    pmid: str
    title: str
    abstract: Optional[str] = None
    authors: List[str] = []
    journal: Optional[str] = None
    pub_date: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    study_type: Optional[str] = None
    quality_score: Optional[float] = None


class PubMedClient:
    """PubMed E-utilities client."""
    
    def __init__(
        self,
        base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        api_key: Optional[str] = None,
        email: str = "your-email@example.com",
        tool: str = "hackwell-evidence-verifier",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.email = email
        self.tool = tool
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Rate limiting
        self.requests_per_second = 3
        self.last_request_time = 0
    
    async def search_articles(
        self, 
        query: str, 
        max_results: int = 20,
        date_range_days: int = 3650  # 10 years
    ) -> List[PubMedArticle]:
        """Search PubMed for articles matching query."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Search for PMIDs
            pmids = await self._search_pmids(query, max_results, date_range_days)
            
            if not pmids:
                logger.info("No PMIDs found for query", query=query)
                return []
            
            # Fetch article details
            articles = await self._fetch_article_details(pmids)
            
            logger.info("PubMed search completed",
                       query=query,
                       pmids_found=len(pmids),
                       articles_retrieved=len(articles))
            
            return articles
            
        except Exception as e:
            logger.error("PubMed search failed", query=query, error=str(e))
            return []
    
    async def _search_pmids(
        self, 
        query: str, 
        max_results: int,
        date_range_days: int
    ) -> List[str]:
        """Search for PMIDs using ESearch."""
        # Build date filter
        end_date = datetime.now()
        start_date = end_date - timedelta(days=date_range_days)
        date_filter = f"{start_date.strftime('%Y/%m/%d')}:{end_date.strftime('%Y/%m/%d')}[pdat]"
        
        # Build search query
        search_query = f"({query}) AND {date_filter}"
        
        params = {
            "db": "pubmed",
            "term": search_query,
            "retmax": min(max_results, 100),  # ESearch limit
            "retmode": "json",
            "sort": "relevance",
            "email": self.email,
            "tool": self.tool
        }
        
        if self.api_key:
            params["api_key"] = self.api_key
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/esearch.fcgi", params=params)
            response.raise_for_status()
            
            data = response.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            return pmids[:max_results]
    
    async def _fetch_article_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch detailed article information using EFetch."""
        if not pmids:
            return []
        
        # Process in batches to avoid URL length limits
        batch_size = 200
        all_articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            batch_articles = await self._fetch_batch_details(batch_pmids)
            all_articles.extend(batch_articles)
            
            # Rate limiting between batches
            if i + batch_size < len(pmids):
                await asyncio.sleep(0.1)
        
        return all_articles
    
    async def _fetch_batch_details(self, pmids: List[str]) -> List[PubMedArticle]:
        """Fetch details for a batch of PMIDs."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
                "rettype": "abstract",
                "email": self.email,
                "tool": self.tool
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/efetch.fcgi", params=params)
                response.raise_for_status()
                
                # Parse XML response
                root = ET.fromstring(response.text)
                articles = self._parse_articles_xml(root)
                
                return articles
                
        except Exception as e:
            logger.error("Failed to fetch article details", pmids=pmids[:5], error=str(e))
            return []
    
    def _parse_articles_xml(self, root: ET.Element) -> List[PubMedArticle]:
        """Parse PubMed XML response."""
        articles = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                article_data = self._parse_single_article(article)
                if article_data:
                    articles.append(article_data)
            except Exception as e:
                logger.warning("Failed to parse article", error=str(e))
                continue
        
        return articles
    
    def _parse_single_article(self, article_elem: ET.Element) -> Optional[PubMedArticle]:
        """Parse a single article from XML."""
        try:
            # Extract PMID
            pmid_elem = article_elem.find(".//PMID")
            if pmid_elem is None:
                return None
            pmid = pmid_elem.text
            
            # Extract title
            title_elem = article_elem.find(".//ArticleTitle")
            title = title_elem.text if title_elem is not None else ""
            
            # Extract abstract
            abstract_elem = article_elem.find(".//AbstractText")
            abstract = abstract_elem.text if abstract_elem is not None else None
            
            # Extract authors
            authors = []
            for author in article_elem.findall(".//Author"):
                last_name = author.find("LastName")
                first_name = author.find("ForeName")
                if last_name is not None:
                    name = last_name.text
                    if first_name is not None:
                        name = f"{first_name.text} {name}"
                    authors.append(name)
            
            # Extract journal
            journal_elem = article_elem.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else None
            
            # Extract publication date
            pub_date = self._extract_publication_date(article_elem)
            
            # Extract DOI
            doi = self._extract_doi(article_elem)
            
            # Determine study type
            study_type = self._determine_study_type(article_elem)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(article_elem)
            
            return PubMedArticle(
                pmid=pmid,
                title=title,
                abstract=abstract,
                authors=authors,
                journal=journal,
                pub_date=pub_date,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                study_type=study_type,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.warning("Failed to parse article", error=str(e))
            return None
    
    def _extract_publication_date(self, article_elem: ET.Element) -> Optional[str]:
        """Extract publication date from article."""
        try:
            # Try PubDate first
            pub_date_elem = article_elem.find(".//PubDate")
            if pub_date_elem is not None:
                year = pub_date_elem.find("Year")
                month = pub_date_elem.find("Month")
                day = pub_date_elem.find("Day")
                
                if year is not None:
                    date_parts = [year.text]
                    if month is not None:
                        date_parts.append(month.text.zfill(2))
                    if day is not None:
                        date_parts.append(day.text.zfill(2))
                    
                    return "-".join(date_parts)
            
            # Try ArticleDate
            article_date_elem = article_elem.find(".//ArticleDate")
            if article_date_elem is not None:
                year = article_date_elem.find("Year")
                month = article_date_elem.find("Month")
                day = article_date_elem.find("Day")
                
                if year is not None:
                    date_parts = [year.text]
                    if month is not None:
                        date_parts.append(month.text.zfill(2))
                    if day is not None:
                        date_parts.append(day.text.zfill(2))
                    
                    return "-".join(date_parts)
            
            return None
            
        except:
            return None
    
    def _extract_doi(self, article_elem: ET.Element) -> Optional[str]:
        """Extract DOI from article."""
        try:
            for article_id in article_elem.findall(".//ArticleId"):
                if article_id.get("IdType") == "doi":
                    return article_id.text
            return None
        except:
            return None
    
    def _determine_study_type(self, article_elem: ET.Element) -> Optional[str]:
        """Determine study type from article metadata."""
        try:
            # Check publication type
            for pub_type in article_elem.findall(".//PublicationType"):
                pub_type_text = pub_type.text.lower()
                
                if "randomized controlled trial" in pub_type_text or "rct" in pub_type_text:
                    return "rct"
                elif "meta-analysis" in pub_type_text:
                    return "meta_analysis"
                elif "systematic review" in pub_type_text:
                    return "systematic_review"
                elif "cohort" in pub_type_text:
                    return "cohort"
                elif "case-control" in pub_type_text:
                    return "case_control"
                elif "case report" in pub_type_text:
                    return "case_report"
                elif "guideline" in pub_type_text:
                    return "guideline"
            
            # Check MeSH terms
            for mesh in article_elem.findall(".//MeshHeading/DescriptorName"):
                mesh_text = mesh.text.lower()
                if "randomized controlled trial" in mesh_text:
                    return "rct"
                elif "cohort studies" in mesh_text:
                    return "cohort"
                elif "case-control studies" in mesh_text:
                    return "case_control"
            
            return "observational"
            
        except:
            return "observational"
    
    def _calculate_quality_score(self, article_elem: ET.Element) -> float:
        """Calculate quality score based on article characteristics."""
        score = 0.5  # Base score
        
        try:
            # Check for abstract
            if article_elem.find(".//AbstractText") is not None:
                score += 0.1
            
            # Check for DOI
            if self._extract_doi(article_elem) is not None:
                score += 0.1
            
            # Check for multiple authors
            authors = article_elem.findall(".//Author")
            if len(authors) > 3:
                score += 0.1
            
            # Check for recent publication
            pub_date = self._extract_publication_date(article_elem)
            if pub_date:
                try:
                    pub_year = int(pub_date.split("-")[0])
                    current_year = datetime.now().year
                    if pub_year >= current_year - 5:
                        score += 0.1
                except:
                    pass
            
            # Check study type
            study_type = self._determine_study_type(article_elem)
            if study_type == "rct":
                score += 0.2
            elif study_type == "meta_analysis":
                score += 0.15
            elif study_type == "cohort":
                score += 0.1
            
            return min(1.0, score)
            
        except:
            return 0.5
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def get_article_by_pmid(self, pmid: str) -> Optional[PubMedArticle]:
        """Get a specific article by PMID."""
        articles = await self._fetch_article_details([pmid])
        return articles[0] if articles else None


def create_pubmed_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> PubMedClient:
    """Factory function to create PubMed client."""
    return PubMedClient(
        base_url=base_url or "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        api_key=api_key,
        **kwargs
    )
