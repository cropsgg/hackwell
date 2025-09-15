"""RxNorm client for drug normalization and interaction checking."""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import structlog
import httpx
from pydantic import BaseModel

logger = structlog.get_logger()


class RxNormConcept(BaseModel):
    """RxNorm concept model."""
    rxcui: str
    name: str
    tty: str  # Term Type (IN, PIN, BN, etc.)
    synonym: bool = False
    language: str = "ENG"
    suppress: bool = False
    umlscui: Optional[str] = None


class DrugInteraction(BaseModel):
    """Drug interaction model."""
    drug1_rxcui: str
    drug1_name: str
    drug2_rxcui: str
    drug2_name: str
    severity: str  # minor, moderate, major, contraindicated
    description: str
    clinical_effects: List[str] = []
    management: Optional[str] = None


class RxNormClient:
    """RxNorm REST API client."""
    
    def __init__(
        self,
        base_url: str = "https://rxnav.nlm.nih.gov/REST",
        max_retries: int = 3,
        timeout: float = 30.0
    ):
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Rate limiting
        self.requests_per_second = 2
        self.last_request_time = 0
    
    async def find_rxcui_by_name(
        self, 
        drug_name: str, 
        search_type: str = "exact"
    ) -> Optional[str]:
        """Find RxCUI by drug name."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            if search_type == "exact":
                endpoint = f"{self.base_url}/rxcui"
                params = {"name": drug_name}
            else:
                endpoint = f"{self.base_url}/drugs"
                params = {"name": drug_name}
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(endpoint, params=params)
                response.raise_for_status()
                
                data = response.json()
                
                if search_type == "exact":
                    # Direct lookup
                    if "idGroup" in data and "rxnormId" in data["idGroup"]:
                        rxnorm_ids = data["idGroup"]["rxnormId"]
                        if rxnorm_ids:
                            return rxnorm_ids[0]
                else:
                    # Search results
                    if "drugGroup" in data and "conceptGroup" in data["drugGroup"]:
                        for concept_group in data["drugGroup"]["conceptGroup"]:
                            if "conceptProperties" in concept_group:
                                for concept in concept_group["conceptProperties"]:
                                    if concept.get("name", "").lower() == drug_name.lower():
                                        return concept.get("rxcui")
                
                return None
                
        except Exception as e:
            logger.error("RxNorm lookup failed", drug_name=drug_name, error=str(e))
            return None
    
    async def get_drug_info(self, rxcui: str) -> Optional[RxNormConcept]:
        """Get detailed drug information by RxCUI."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/rxcui/{rxcui}/properties")
                response.raise_for_status()
                
                data = response.json()
                
                if "properties" in data:
                    props = data["properties"]
                    return RxNormConcept(
                        rxcui=props.get("rxcui", rxcui),
                        name=props.get("name", ""),
                        tty=props.get("tty", ""),
                        synonym=props.get("synonym", False),
                        language=props.get("language", "ENG"),
                        suppress=props.get("suppress", False),
                        umlscui=props.get("umlscui")
                    )
                
                return None
                
        except Exception as e:
            logger.error("Failed to get drug info", rxcui=rxcui, error=str(e))
            return None
    
    async def get_drug_synonyms(self, rxcui: str) -> List[str]:
        """Get synonyms for a drug by RxCUI."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/rxcui/{rxcui}/related")
                response.raise_for_status()
                
                data = response.json()
                synonyms = []
                
                if "relatedGroup" in data and "conceptGroup" in data["relatedGroup"]:
                    for concept_group in data["relatedGroup"]["conceptGroup"]:
                        if "conceptProperties" in concept_group:
                            for concept in concept_group["conceptProperties"]:
                                name = concept.get("name", "")
                                if name:
                                    synonyms.append(name)
                
                return synonyms
                
        except Exception as e:
            logger.error("Failed to get drug synonyms", rxcui=rxcui, error=str(e))
            return []
    
    async def normalize_drug_names(self, drug_names: List[str]) -> Dict[str, Optional[str]]:
        """Normalize multiple drug names to RxCUIs."""
        normalized = {}
        
        for drug_name in drug_names:
            try:
                rxcui = await self.find_rxcui_by_name(drug_name)
                normalized[drug_name] = rxcui
                
                if rxcui:
                    logger.debug("Drug normalized", 
                               drug_name=drug_name, 
                               rxcui=rxcui)
                else:
                    logger.warning("Drug not found in RxNorm", drug_name=drug_name)
                    
            except Exception as e:
                logger.error("Drug normalization failed", 
                           drug_name=drug_name, 
                           error=str(e))
                normalized[drug_name] = None
        
        return normalized
    
    async def check_drug_interactions(
        self, 
        drug_rxcuis: List[str]
    ) -> List[DrugInteraction]:
        """Check for drug interactions between multiple drugs."""
        interactions = []
        
        try:
            if len(drug_rxcuis) < 2:
                return interactions
            
            # Check pairwise interactions
            for i in range(len(drug_rxcuis)):
                for j in range(i + 1, len(drug_rxcuis)):
                    drug1_rxcui = drug_rxcuis[i]
                    drug2_rxcui = drug_rxcuis[j]
                    
                    interaction = await self._check_pairwise_interaction(
                        drug1_rxcui, drug2_rxcui
                    )
                    if interaction:
                        interactions.append(interaction)
            
            logger.info("Drug interaction check completed",
                       drugs=len(drug_rxcuis),
                       interactions_found=len(interactions))
            
            return interactions
            
        except Exception as e:
            logger.error("Drug interaction check failed", error=str(e))
            return []
    
    async def _check_pairwise_interaction(
        self, 
        drug1_rxcui: str, 
        drug2_rxcui: str
    ) -> Optional[DrugInteraction]:
        """Check interaction between two specific drugs."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/interaction/list",
                    params={"rxcuis": f"{drug1_rxcui}+{drug2_rxcui}"}
                )
                response.raise_for_status()
                
                data = response.json()
                
                if "interactionTypeGroup" in data:
                    for interaction_group in data["interactionTypeGroup"]:
                        if "interactionType" in interaction_group:
                            for interaction_type in interaction_group["interactionType"]:
                                if "interactionPair" in interaction_type:
                                    for pair in interaction_type["interactionPair"]:
                                        return self._parse_interaction_pair(
                                            pair, drug1_rxcui, drug2_rxcui
                                        )
                
                return None
                
        except Exception as e:
            logger.warning("Pairwise interaction check failed",
                         drug1=drug1_rxcui,
                         drug2=drug2_rxcui,
                         error=str(e))
            return None
    
    def _parse_interaction_pair(
        self, 
        pair: Dict[str, Any], 
        drug1_rxcui: str, 
        drug2_rxcui: str
    ) -> Optional[DrugInteraction]:
        """Parse interaction pair data."""
        try:
            # Extract drug information
            drug1_info = pair.get("interactionConcept", [{}])[0]
            drug2_info = pair.get("interactionConcept", [{}])[1] if len(pair.get("interactionConcept", [])) > 1 else {}
            
            drug1_name = drug1_info.get("minConceptItem", {}).get("name", "")
            drug2_name = drug2_info.get("minConceptItem", {}).get("name", "")
            
            # Extract severity and description
            severity = pair.get("severity", "unknown")
            description = pair.get("description", "")
            
            # Extract clinical effects
            clinical_effects = []
            if "clinicalEffect" in pair:
                for effect in pair["clinicalEffect"]:
                    if isinstance(effect, dict) and "clinicalEffectItem" in effect:
                        clinical_effects.append(effect["clinicalEffectItem"])
                    elif isinstance(effect, str):
                        clinical_effects.append(effect)
            
            return DrugInteraction(
                drug1_rxcui=drug1_rxcui,
                drug1_name=drug1_name,
                drug2_rxcui=drug2_rxcui,
                drug2_name=drug2_name,
                severity=severity,
                description=description,
                clinical_effects=clinical_effects
            )
            
        except Exception as e:
            logger.warning("Failed to parse interaction pair", error=str(e))
            return None
    
    async def get_ingredient_by_rxcui(self, rxcui: str) -> Optional[str]:
        """Get ingredient name for a drug by RxCUI."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/rxcui/{rxcui}/related")
                response.raise_for_status()
                
                data = response.json()
                
                if "relatedGroup" in data and "conceptGroup" in data["relatedGroup"]:
                    for concept_group in data["relatedGroup"]["conceptGroup"]:
                        if concept_group.get("tty") == "IN":  # Ingredient
                            if "conceptProperties" in concept_group:
                                for concept in concept_group["conceptProperties"]:
                                    return concept.get("name")
                
                return None
                
        except Exception as e:
            logger.error("Failed to get ingredient", rxcui=rxcui, error=str(e))
            return None
    
    async def search_drugs_by_ingredient(self, ingredient: str) -> List[RxNormConcept]:
        """Search for drugs containing a specific ingredient."""
        try:
            # Rate limiting
            await self._rate_limit()
            
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/drugs",
                    params={"name": ingredient}
                )
                response.raise_for_status()
                
                data = response.json()
                concepts = []
                
                if "drugGroup" in data and "conceptGroup" in data["drugGroup"]:
                    for concept_group in data["drugGroup"]["conceptGroup"]:
                        if "conceptProperties" in concept_group:
                            for concept in concept_group["conceptProperties"]:
                                concepts.append(RxNormConcept(
                                    rxcui=concept.get("rxcui", ""),
                                    name=concept.get("name", ""),
                                    tty=concept.get("tty", ""),
                                    synonym=concept.get("synonym", False),
                                    language=concept.get("language", "ENG"),
                                    suppress=concept.get("suppress", False)
                                ))
                
                return concepts
                
        except Exception as e:
            logger.error("Failed to search drugs by ingredient", 
                        ingredient=ingredient, 
                        error=str(e))
            return []
    
    async def _rate_limit(self):
        """Implement rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.requests_per_second
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()


def create_rxnorm_client(
    base_url: Optional[str] = None,
    **kwargs
) -> RxNormClient:
    """Factory function to create RxNorm client."""
    return RxNormClient(
        base_url=base_url or "https://rxnav.nlm.nih.gov/REST",
        **kwargs
    )
