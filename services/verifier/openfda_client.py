"""OpenFDA API client for drug safety information."""

import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus
from datetime import datetime, timedelta

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class OpenFDAClient:
    """Client for OpenFDA API drug safety information."""
    
    def __init__(self, base_url: str = "https://api.fda.gov"):
        self.base_url = base_url
        self.api_key = None  # Optional API key for higher rate limits
        
        # Rate limiting: 1000 requests per hour, 40 per minute without API key
        self.rate_limit = 1.5  # seconds between requests
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make rate-limited request to OpenFDA API."""
        await self._rate_limit()
        
        if self.api_key:
            params['api_key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            logger.error("OpenFDA API request failed", error=str(e), url=url)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("OpenFDA API error", status=e.response.status_code, url=url)
            if e.response.status_code == 404:
                return {'results': []}  # No data found
            raise
    
    async def get_drug_label(self, drug_name: str) -> Dict[str, Any]:
        """Get FDA drug label information."""
        try:
            # Clean and format drug name for search
            clean_name = self._clean_drug_name(drug_name)
            
            params = {
                'search': f'openfda.generic_name:"{clean_name}" OR openfda.brand_name:"{clean_name}"',
                'limit': 5
            }
            
            logger.info("Searching OpenFDA drug labels", drug_name=drug_name)
            
            response = await self._make_request('drug/label.json', params)
            
            if not response.get('results'):
                logger.info("No FDA label found", drug_name=drug_name)
                return self._empty_drug_response(drug_name)
            
            # Process the first result
            label_data = response['results'][0]
            processed_data = self._process_drug_label(label_data, drug_name)
            
            logger.info("OpenFDA drug label retrieved", 
                       drug_name=drug_name,
                       warnings_count=len(processed_data.get('warnings', [])))
            
            return processed_data
            
        except Exception as e:
            logger.error("Failed to get drug label", drug_name=drug_name, error=str(e))
            return self._empty_drug_response(drug_name)
    
    async def get_adverse_events(self, drug_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get adverse event reports for a drug."""
        try:
            clean_name = self._clean_drug_name(drug_name)
            
            # Search in both generic and brand names
            search_query = f'patient.drug.medicinalproduct:"{clean_name}"'
            
            params = {
                'search': search_query,
                'count': 'patient.reaction.reactionmeddrapt.exact',
                'limit': limit
            }
            
            logger.info("Searching OpenFDA adverse events", drug_name=drug_name)
            
            response = await self._make_request('drug/event.json', params)
            
            if not response.get('results'):
                logger.info("No adverse events found", drug_name=drug_name)
                return []
            
            # Process adverse event counts
            adverse_events = []
            for result in response['results']:
                event = {
                    'reaction': result.get('term', 'Unknown'),
                    'count': result.get('count', 0),
                    'severity': self._classify_adverse_event_severity(result.get('term', ''))
                }
                adverse_events.append(event)
            
            # Sort by count (most frequent first)
            adverse_events.sort(key=lambda x: x['count'], reverse=True)
            
            logger.info("OpenFDA adverse events retrieved", 
                       drug_name=drug_name,
                       events_count=len(adverse_events))
            
            return adverse_events
            
        except Exception as e:
            logger.error("Failed to get adverse events", drug_name=drug_name, error=str(e))
            return []
    
    async def get_drug_recalls(self, drug_name: str) -> List[Dict[str, Any]]:
        """Get drug recall information."""
        try:
            clean_name = self._clean_drug_name(drug_name)
            
            params = {
                'search': f'product_description:"{clean_name}"',
                'limit': 10
            }
            
            logger.info("Searching OpenFDA drug recalls", drug_name=drug_name)
            
            response = await self._make_request('drug/enforcement.json', params)
            
            if not response.get('results'):
                return []
            
            recalls = []
            for result in response['results']:
                recall = {
                    'recall_number': result.get('recall_number', ''),
                    'product_description': result.get('product_description', ''),
                    'reason_for_recall': result.get('reason_for_recall', ''),
                    'recall_initiation_date': result.get('recall_initiation_date', ''),
                    'classification': result.get('classification', ''),
                    'status': result.get('status', ''),
                    'severity': self._classify_recall_severity(result.get('classification', ''))
                }
                recalls.append(recall)
            
            logger.info("OpenFDA drug recalls retrieved", 
                       drug_name=drug_name,
                       recalls_count=len(recalls))
            
            return recalls
            
        except Exception as e:
            logger.error("Failed to get drug recalls", drug_name=drug_name, error=str(e))
            return []
    
    def _clean_drug_name(self, drug_name: str) -> str:
        """Clean drug name for API search."""
        # Remove common suffixes and dosage information
        clean_name = drug_name.lower()
        
        # Remove dosage information
        clean_name = clean_name.split(' ')[0]  # Take first word
        
        # Remove common suffixes
        suffixes = ['er', 'xr', 'sr', 'cr', 'la', 'xl', 'hcl', 'hydrochloride']
        for suffix in suffixes:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)].strip()
        
        return clean_name
    
    def _process_drug_label(self, label_data: Dict[str, Any], drug_name: str) -> Dict[str, Any]:
        """Process FDA drug label data."""
        processed = {
            'drug_name': drug_name,
            'generic_name': '',
            'brand_names': [],
            'warnings': [],
            'contraindications': [],
            'adverse_reactions': [],
            'drug_interactions': [],
            'pregnancy_category': '',
            'controlled_substance': False,
            'black_box_warning': False
        }
        
        # Extract OpenFDA standardized fields
        openfda = label_data.get('openfda', {})
        processed['generic_name'] = openfda.get('generic_name', [''])[0] if openfda.get('generic_name') else ''
        processed['brand_names'] = openfda.get('brand_name', [])
        
        # Extract warnings
        warnings = label_data.get('warnings', [])
        if warnings:
            processed['warnings'] = [self._clean_text(w) for w in warnings]
        
        # Extract contraindications
        contraindications = label_data.get('contraindications', [])
        if contraindications:
            processed['contraindications'] = [self._clean_text(c) for c in contraindications]
        
        # Extract adverse reactions
        adverse_reactions = label_data.get('adverse_reactions', [])
        if adverse_reactions:
            processed['adverse_reactions'] = [self._clean_text(ar) for ar in adverse_reactions]
        
        # Extract drug interactions
        drug_interactions = label_data.get('drug_interactions', [])
        if drug_interactions:
            processed['drug_interactions'] = [self._clean_text(di) for di in drug_interactions]
        
        # Check for black box warning
        boxed_warning = label_data.get('boxed_warning', [])
        processed['black_box_warning'] = len(boxed_warning) > 0
        
        # Pregnancy category
        pregnancy_info = label_data.get('pregnancy', [])
        if pregnancy_info:
            processed['pregnancy_category'] = self._extract_pregnancy_category(pregnancy_info[0])
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and truncate text for display."""
        if not text:
            return ""
        
        # Remove excessive whitespace and truncate
        cleaned = ' '.join(text.split())
        
        # Truncate to reasonable length
        if len(cleaned) > 500:
            cleaned = cleaned[:497] + "..."
        
        return cleaned
    
    def _extract_pregnancy_category(self, pregnancy_text: str) -> str:
        """Extract pregnancy category from text."""
        categories = ['A', 'B', 'C', 'D', 'X']
        pregnancy_lower = pregnancy_text.lower()
        
        for category in categories:
            if f'category {category.lower()}' in pregnancy_lower:
                return category
        
        return 'Unknown'
    
    def _classify_adverse_event_severity(self, reaction: str) -> str:
        """Classify adverse event severity."""
        reaction_lower = reaction.lower()
        
        severe_terms = [
            'death', 'fatal', 'cardiac arrest', 'myocardial infarction',
            'stroke', 'seizure', 'anaphylaxis', 'liver failure', 'kidney failure'
        ]
        
        moderate_terms = [
            'hypertension', 'hypotension', 'arrhythmia', 'chest pain',
            'dyspnea', 'syncope', 'confusion', 'depression'
        ]
        
        if any(term in reaction_lower for term in severe_terms):
            return 'severe'
        elif any(term in reaction_lower for term in moderate_terms):
            return 'moderate'
        else:
            return 'mild'
    
    def _classify_recall_severity(self, classification: str) -> str:
        """Classify FDA recall severity."""
        if classification == 'Class I':
            return 'high'
        elif classification == 'Class II':
            return 'moderate'
        elif classification == 'Class III':
            return 'low'
        else:
            return 'unknown'
    
    def _empty_drug_response(self, drug_name: str) -> Dict[str, Any]:
        """Return empty response structure."""
        return {
            'drug_name': drug_name,
            'generic_name': '',
            'brand_names': [],
            'warnings': [],
            'contraindications': [],
            'adverse_reactions': [],
            'drug_interactions': [],
            'pregnancy_category': '',
            'controlled_substance': False,
            'black_box_warning': False,
            'data_available': False
        }
    
    async def get_comprehensive_drug_info(self, drug_name: str) -> Dict[str, Any]:
        """Get comprehensive drug information from multiple endpoints."""
        try:
            # Get all information in parallel
            label_task = self.get_drug_label(drug_name)
            adverse_events_task = self.get_adverse_events(drug_name)
            recalls_task = self.get_drug_recalls(drug_name)
            
            label_info, adverse_events, recalls = await asyncio.gather(
                label_task, adverse_events_task, recalls_task,
                return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(label_info, Exception):
                label_info = self._empty_drug_response(drug_name)
            if isinstance(adverse_events, Exception):
                adverse_events = []
            if isinstance(recalls, Exception):
                recalls = []
            
            # Combine information
            comprehensive_info = {
                **label_info,
                'recent_adverse_events': adverse_events[:5],  # Top 5 most frequent
                'recalls': recalls,
                'safety_score': self._calculate_safety_score(label_info, adverse_events, recalls),
                'risk_level': self._determine_risk_level(label_info, adverse_events, recalls)
            }
            
            return comprehensive_info
            
        except Exception as e:
            logger.error("Failed to get comprehensive drug info", drug_name=drug_name, error=str(e))
            return self._empty_drug_response(drug_name)
    
    def _calculate_safety_score(self, label_info: Dict, adverse_events: List, recalls: List) -> float:
        """Calculate safety score (0-1, higher is safer)."""
        score = 1.0
        
        # Black box warning significantly reduces score
        if label_info.get('black_box_warning'):
            score -= 0.4
        
        # Warnings reduce score
        warnings_count = len(label_info.get('warnings', []))
        score -= min(0.3, warnings_count * 0.1)
        
        # Recent recalls reduce score
        recent_recalls = [r for r in recalls if self._is_recent_recall(r.get('recall_initiation_date', ''))]
        if recent_recalls:
            score -= min(0.2, len(recent_recalls) * 0.1)
        
        # High frequency adverse events reduce score
        if adverse_events:
            high_frequency_events = [e for e in adverse_events if e.get('count', 0) > 100]
            score -= min(0.1, len(high_frequency_events) * 0.02)
        
        return max(0.0, score)
    
    def _determine_risk_level(self, label_info: Dict, adverse_events: List, recalls: List) -> str:
        """Determine overall risk level."""
        safety_score = self._calculate_safety_score(label_info, adverse_events, recalls)
        
        if safety_score >= 0.8:
            return 'low'
        elif safety_score >= 0.6:
            return 'moderate'
        else:
            return 'high'
    
    def _is_recent_recall(self, recall_date: str) -> bool:
        """Check if recall is recent (within 2 years)."""
        if not recall_date:
            return False
        
        try:
            recall_dt = datetime.strptime(recall_date, '%Y%m%d')
            two_years_ago = datetime.now() - timedelta(days=730)
            return recall_dt > two_years_ago
        except ValueError:
            return False
