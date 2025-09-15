"""openFDA client for drug safety information."""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import structlog
import httpx
from pydantic import BaseModel

logger = structlog.get_logger()


class DrugLabel(BaseModel):
    """Drug label information from openFDA."""
    drug_name: str
    generic_name: Optional[str] = None
    brand_name: Optional[str] = None
    manufacturer: Optional[str] = None
    warnings: List[str] = []
    contraindications: List[str] = []
    adverse_reactions: List[str] = []
    boxed_warnings: List[str] = []
    drug_interactions: List[str] = []
    pregnancy_category: Optional[str] = None
    pediatric_use: Optional[str] = None
    geriatric_use: Optional[str] = None
    url: Optional[str] = None
    last_updated: Optional[str] = None


class OpenFDAClient:
    """openFDA API client for drug safety information."""
    
    def __init__(
        self,
        base_url: str = "https://api.fda.gov",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Rate limiting
        self.requests_per_second = 1
        self.last_request_time = 0
    
    async def search_drug(
        self, 
        drug_name: str, 
        search_type: str = "name"
    ) -> List[DrugLabel]:
        """Search for drug information by name."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            # Build search query
            if search_type == "name":
                query = f"openfda.brand_name:\"{drug_name}\" OR openfda.generic_name:\"{drug_name}\""
            elif search_type == "generic":
                query = f"openfda.generic_name:\"{drug_name}\""
            else:
                query = f"openfda.brand_name:\"{drug_name}\""
            
            params = {
                "search": query,
                "limit": 10,
                "format": "json"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/drug/label.json",
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                drug_labels = []
                for result in results:
                    drug_label = self._parse_drug_label(result, drug_name)
                    if drug_label:
                        drug_labels.append(drug_label)
                
                logger.info("openFDA search completed",
                           drug_name=drug_name,
                           results_found=len(drug_labels))
                
                return drug_labels
                
        except Exception as e:
            logger.error("openFDA search failed", drug_name=drug_name, error=str(e))
            return []
    
    def _parse_drug_label(self, data: Dict[str, Any], search_name: str) -> Optional[DrugLabel]:
        """Parse drug label data from openFDA response."""
        try:
            # Extract basic information
            openfda = data.get("openfda", {})
            brand_names = openfda.get("brand_name", [])
            generic_names = openfda.get("generic_name", [])
            manufacturer = openfda.get("manufacturer_name", [None])[0]
            
            # Extract warnings and safety information
            warnings = self._extract_safety_info(data, "warnings")
            contraindications = self._extract_safety_info(data, "contraindications")
            adverse_reactions = self._extract_safety_info(data, "adverse_reactions")
            boxed_warnings = self._extract_safety_info(data, "boxed_warnings")
            drug_interactions = self._extract_safety_info(data, "drug_interactions")
            
            # Extract pregnancy category
            pregnancy_category = self._extract_pregnancy_category(data)
            
            # Extract pediatric and geriatric use
            pediatric_use = self._extract_use_info(data, "pediatric_use")
            geriatric_use = self._extract_use_info(data, "geriatric_use")
            
            # Get URL and last updated
            url = data.get("openfda", {}).get("package_ndc", [None])[0]
            if url:
                url = f"https://www.accessdata.fda.gov/scripts/cder/daf/index.cfm?event=overview.process&ApplNo={url}"
            
            last_updated = data.get("effective_time", "")
            if last_updated:
                try:
                    # Parse date format YYYYMMDD
                    date_obj = datetime.strptime(last_updated[:8], "%Y%m%d")
                    last_updated = date_obj.isoformat()
                except:
                    pass
            
            return DrugLabel(
                drug_name=search_name,
                generic_name=generic_names[0] if generic_names else None,
                brand_name=brand_names[0] if brand_names else None,
                manufacturer=manufacturer,
                warnings=warnings,
                contraindications=contraindications,
                adverse_reactions=adverse_reactions,
                boxed_warnings=boxed_warnings,
                drug_interactions=drug_interactions,
                pregnancy_category=pregnancy_category,
                pediatric_use=pediatric_use,
                geriatric_use=geriatric_use,
                url=url,
                last_updated=last_updated
            )
            
        except Exception as e:
            logger.warning("Failed to parse drug label", error=str(e))
            return None
    
    def _extract_safety_info(self, data: Dict[str, Any], field: str) -> List[str]:
        """Extract safety information from drug label data."""
        safety_info = []
        
        try:
            # Look for the field in various locations
            if field in data:
                field_data = data[field]
                if isinstance(field_data, list):
                    for item in field_data:
                        if isinstance(item, str):
                            safety_info.append(item.strip())
                        elif isinstance(item, dict) and "text" in item:
                            safety_info.append(item["text"].strip())
                elif isinstance(field_data, str):
                    safety_info.append(field_data.strip())
            
            # Also check in nested structures
            for key, value in data.items():
                if isinstance(value, dict) and field in value:
                    nested_info = self._extract_safety_info(value, field)
                    safety_info.extend(nested_info)
        
        except Exception as e:
            logger.warning(f"Failed to extract {field}", error=str(e))
        
        return safety_info
    
    def _extract_pregnancy_category(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract pregnancy category from drug label."""
        try:
            # Look for pregnancy information
            pregnancy_fields = ["pregnancy", "pregnancy_category", "pregnancy_warning"]
            
            for field in pregnancy_fields:
                if field in data:
                    value = data[field]
                    if isinstance(value, str):
                        return value.strip()
                    elif isinstance(value, list) and value:
                        return str(value[0]).strip()
            
            return None
            
        except:
            return None
    
    def _extract_use_info(self, data: Dict[str, Any], field: str) -> Optional[str]:
        """Extract use information (pediatric/geriatric) from drug label."""
        try:
            if field in data:
                value = data[field]
                if isinstance(value, str):
                    return value.strip()
                elif isinstance(value, list) and value:
                    return str(value[0]).strip()
            
            return None
            
        except:
            return None
    
    async def get_drug_interactions(
        self, 
        drug_names: List[str]
    ) -> List[Dict[str, Any]]:
        """Get drug interaction information for multiple drugs."""
        interactions = []
        
        try:
            for drug_name in drug_names:
                drug_labels = await self.search_drug(drug_name)
                
                for label in drug_labels:
                    if label.drug_interactions:
                        interactions.append({
                            "drug": drug_name,
                            "interactions": label.drug_interactions,
                            "source": "openFDA",
                            "url": label.url
                        })
            
            logger.info("Drug interactions retrieved",
                       drugs=len(drug_names),
                       interactions_found=len(interactions))
            
            return interactions
            
        except Exception as e:
            logger.error("Failed to get drug interactions", error=str(e))
            return []
    
    async def search_by_ndc(self, ndc: str) -> Optional[DrugLabel]:
        """Search for drug by NDC (National Drug Code)."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            params = {
                "search": f"openfda.package_ndc:\"{ndc}\"",
                "limit": 1,
                "format": "json"
            }
            
            if self.api_key:
                params["api_key"] = self.api_key
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/drug/label.json",
                    params=params
                )
                response.raise_for_status()
                
                data = response.json()
                results = data.get("results", [])
                
                if results:
                    return self._parse_drug_label(results[0], ndc)
                
                return None
                
        except Exception as e:
            logger.error("openFDA NDC search failed", ndc=ndc, error=str(e))
            return None
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    async def get_safety_alerts(self, drug_name: str) -> List[str]:
        """Get safety alerts for a specific drug."""
        try:
            drug_labels = await self.search_drug(drug_name)
            
            safety_alerts = []
            for label in drug_labels:
                safety_alerts.extend(label.boxed_warnings)
                safety_alerts.extend(label.warnings)
                safety_alerts.extend(label.contraindications)
            
            # Remove duplicates and filter for significant alerts
            unique_alerts = list(set(safety_alerts))
            significant_alerts = [
                alert for alert in unique_alerts 
                if any(keyword in alert.lower() for keyword in [
                    "warning", "caution", "contraindication", "adverse", 
                    "serious", "severe", "fatal", "life-threatening"
                ])
            ]
            
            return significant_alerts
            
        except Exception as e:
            logger.error("Failed to get safety alerts", drug_name=drug_name, error=str(e))
            return []


def create_openfda_client(
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> OpenFDAClient:
    """Factory function to create openFDA client."""
    return OpenFDAClient(
        base_url=base_url or "https://api.fda.gov",
        api_key=api_key,
        **kwargs
    )
