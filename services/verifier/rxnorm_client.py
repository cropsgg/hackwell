"""RxNorm API client for drug normalization and interaction checking."""

import asyncio
from typing import Dict, List, Any, Optional
from urllib.parse import quote_plus

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()


class RxNormClient:
    """Client for RxNorm API drug information and normalization."""
    
    def __init__(self, base_url: str = "https://rxnav.nlm.nih.gov/REST"):
        self.base_url = base_url
        
        # No explicit rate limiting required for RxNorm, but be respectful
        self.rate_limit = 0.1  # 100ms between requests
        self.last_request_time = 0
    
    async def _rate_limit(self):
        """Enforce rate limiting."""
        current_time = asyncio.get_event_loop().time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.rate_limit:
            await asyncio.sleep(self.rate_limit - time_since_last)
        
        self.last_request_time = asyncio.get_event_loop().time()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make rate-limited request to RxNorm API."""
        await self._rate_limit()
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, params=params or {})
                response.raise_for_status()
                return response.json()
                
        except httpx.RequestError as e:
            logger.error("RxNorm API request failed", error=str(e), url=url)
            raise
        except httpx.HTTPStatusError as e:
            logger.error("RxNorm API error", status=e.response.status_code, url=url)
            raise
    
    async def search_drug(self, drug_name: str) -> List[Dict[str, Any]]:
        """Search for drug by name and return normalized results."""
        try:
            # Clean drug name
            clean_name = self._clean_drug_name(drug_name)
            
            endpoint = f"drugs.json"
            params = {'name': clean_name}
            
            logger.info("Searching RxNorm", drug_name=drug_name)
            
            response = await self._make_request(endpoint, params)
            
            drugs_group = response.get('drugGroup', {})
            concept_group = drugs_group.get('conceptGroup')
            
            if not concept_group:
                logger.info("No RxNorm results found", drug_name=drug_name)
                return []
            
            # Extract drug concepts
            results = []
            for group in concept_group:
                concept_properties = group.get('conceptProperties', [])
                for concept in concept_properties:
                    result = {
                        'rxcui': concept.get('rxcui'),
                        'name': concept.get('name'),
                        'synonym': concept.get('synonym'),
                        'tty': concept.get('tty'),  # Term type
                        'language': concept.get('language'),
                        'suppress': concept.get('suppress'),
                        'umlscui': concept.get('umlscui')
                    }
                    results.append(result)
            
            # Filter for active ingredients and branded drugs
            filtered_results = [r for r in results if r['tty'] in ['IN', 'MIN', 'BN', 'SBD', 'SCD']]
            
            logger.info("RxNorm search completed", 
                       drug_name=drug_name,
                       results_found=len(filtered_results))
            
            return filtered_results[:10]  # Limit results
            
        except Exception as e:
            logger.error("RxNorm search failed", drug_name=drug_name, error=str(e))
            return []
    
    async def get_rxcui_by_name(self, drug_name: str) -> Optional[str]:
        """Get RxCUI for a drug name."""
        try:
            results = await self.search_drug(drug_name)
            
            # Prefer active ingredients (IN) or branded names (BN)
            for result in results:
                if result['tty'] in ['IN', 'BN'] and result['rxcui']:
                    return result['rxcui']
            
            # Fall back to any available RxCUI
            for result in results:
                if result['rxcui']:
                    return result['rxcui']
            
            return None
            
        except Exception as e:
            logger.error("Failed to get RxCUI", drug_name=drug_name, error=str(e))
            return None
    
    async def get_drug_properties(self, rxcui: str) -> Dict[str, Any]:
        """Get comprehensive drug properties by RxCUI."""
        try:
            endpoint = f"rxcui/{rxcui}/properties.json"
            
            response = await self._make_request(endpoint)
            
            properties = response.get('properties', {})
            
            if not properties:
                return {}
            
            return {
                'rxcui': properties.get('rxcui'),
                'name': properties.get('name'),
                'synonym': properties.get('synonym'),
                'tty': properties.get('tty'),
                'language': properties.get('language'),
                'suppress': properties.get('suppress'),
                'umlscui': properties.get('umlscui')
            }
            
        except Exception as e:
            logger.error("Failed to get drug properties", rxcui=rxcui, error=str(e))
            return {}
    
    async def get_drug_interactions(self, rxcui_list: List[str]) -> List[Dict[str, Any]]:
        """Get drug-drug interactions for a list of RxCUIs."""
        try:
            if len(rxcui_list) < 2:
                return []
            
            # RxNorm interaction API works with pairs
            interactions = []
            
            for i in range(len(rxcui_list)):
                for j in range(i + 1, len(rxcui_list)):
                    rxcui1, rxcui2 = rxcui_list[i], rxcui_list[j]
                    
                    endpoint = f"interaction/interaction.json"
                    params = {'rxcui': rxcui1, 'sources': 'DrugBank'}
                    
                    response = await self._make_request(endpoint, params)
                    
                    interaction_type_group = response.get('interactionTypeGroup', [])
                    
                    for type_group in interaction_type_group:
                        interaction_type = type_group.get('interactionType', [])
                        
                        for interaction in interaction_type:
                            interaction_pair = interaction.get('interactionPair', [])
                            
                            for pair in interaction_pair:
                                # Check if this interaction involves our second drug
                                interaction_concepts = pair.get('interactionConcept', [])
                                rxcuis_in_interaction = [ic.get('minConceptItem', {}).get('rxcui') 
                                                       for ic in interaction_concepts]
                                
                                if rxcui2 in rxcuis_in_interaction:
                                    severity = pair.get('severity', 'Unknown')
                                    description = pair.get('description', '')
                                    
                                    interactions.append({
                                        'drug1_rxcui': rxcui1,
                                        'drug2_rxcui': rxcui2,
                                        'severity': severity,
                                        'description': description,
                                        'source': 'DrugBank via RxNorm'
                                    })
            
            logger.info("Drug interactions retrieved", 
                       drugs_count=len(rxcui_list),
                       interactions_found=len(interactions))
            
            return interactions
            
        except Exception as e:
            logger.error("Failed to get drug interactions", error=str(e))
            return []
    
    async def get_drug_classes(self, rxcui: str) -> List[Dict[str, Any]]:
        """Get drug classification information."""
        try:
            endpoint = f"rxcui/{rxcui}/related.json"
            params = {'tty': 'IN+MIN'}  # Active ingredients
            
            response = await self._make_request(endpoint, params)
            
            related_group = response.get('relatedGroup', {})
            concept_group = related_group.get('conceptGroup', [])
            
            classes = []
            for group in concept_group:
                tty = group.get('tty')
                concept_properties = group.get('conceptProperties', [])
                
                for concept in concept_properties:
                    classes.append({
                        'rxcui': concept.get('rxcui'),
                        'name': concept.get('name'),
                        'tty': tty,
                        'classification_type': self._map_tty_to_classification(tty)
                    })
            
            return classes
            
        except Exception as e:
            logger.error("Failed to get drug classes", rxcui=rxcui, error=str(e))
            return []
    
    async def normalize_drug_list(self, drug_names: List[str]) -> List[Dict[str, Any]]:
        """Normalize a list of drug names to RxNorm concepts."""
        normalized_drugs = []
        
        for drug_name in drug_names:
            try:
                rxcui = await self.get_rxcui_by_name(drug_name)
                
                if rxcui:
                    properties = await self.get_drug_properties(rxcui)
                    classes = await self.get_drug_classes(rxcui)
                    
                    normalized_drug = {
                        'original_name': drug_name,
                        'rxcui': rxcui,
                        'normalized_name': properties.get('name', drug_name),
                        'properties': properties,
                        'classes': classes,
                        'normalization_success': True
                    }
                else:
                    normalized_drug = {
                        'original_name': drug_name,
                        'rxcui': None,
                        'normalized_name': drug_name,
                        'properties': {},
                        'classes': [],
                        'normalization_success': False
                    }
                
                normalized_drugs.append(normalized_drug)
                
            except Exception as e:
                logger.error("Failed to normalize drug", drug_name=drug_name, error=str(e))
                normalized_drugs.append({
                    'original_name': drug_name,
                    'rxcui': None,
                    'normalized_name': drug_name,
                    'properties': {},
                    'classes': [],
                    'normalization_success': False,
                    'error': str(e)
                })
        
        return normalized_drugs
    
    async def check_medication_safety(self, medication_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Comprehensive medication safety check."""
        try:
            # Extract drug names
            drug_names = [med.get('name', '') for med in medication_list if med.get('active', True)]
            
            if not drug_names:
                return {
                    'interactions': [],
                    'normalized_medications': [],
                    'safety_alerts': [],
                    'overall_risk': 'low'
                }
            
            # Normalize drugs
            normalized_drugs = await self.normalize_drug_list(drug_names)
            
            # Get RxCUIs for interaction checking
            rxcuis = [drug['rxcui'] for drug in normalized_drugs if drug['rxcui']]
            
            # Check interactions
            interactions = await self.get_drug_interactions(rxcuis) if len(rxcuis) > 1 else []
            
            # Generate safety alerts
            safety_alerts = self._generate_safety_alerts(normalized_drugs, interactions)
            
            # Determine overall risk
            overall_risk = self._calculate_overall_medication_risk(interactions, safety_alerts)
            
            return {
                'interactions': interactions,
                'normalized_medications': normalized_drugs,
                'safety_alerts': safety_alerts,
                'overall_risk': overall_risk,
                'medications_processed': len(drug_names),
                'medications_normalized': len([d for d in normalized_drugs if d['normalization_success']])
            }
            
        except Exception as e:
            logger.error("Medication safety check failed", error=str(e))
            return {
                'interactions': [],
                'normalized_medications': [],
                'safety_alerts': [f"Safety check failed: {str(e)}"],
                'overall_risk': 'unknown'
            }
    
    def _clean_drug_name(self, drug_name: str) -> str:
        """Clean drug name for RxNorm search."""
        # Remove dosage information and common suffixes
        clean_name = drug_name.strip().lower()
        
        # Remove dosage patterns
        import re
        clean_name = re.sub(r'\d+\s*(mg|mcg|g|ml|units?)', '', clean_name)
        clean_name = re.sub(r'\d+/\d+', '', clean_name)  # Remove ratios like 5/325
        
        # Remove common suffixes
        suffixes = ['er', 'xr', 'sr', 'cr', 'la', 'xl', 'hcl', 'hydrochloride', 'tablet', 'capsule']
        for suffix in suffixes:
            if clean_name.endswith(' ' + suffix):
                clean_name = clean_name[:-len(suffix)-1]
        
        return clean_name.strip()
    
    def _map_tty_to_classification(self, tty: str) -> str:
        """Map RxNorm term type to classification type."""
        mapping = {
            'IN': 'active_ingredient',
            'MIN': 'multiple_ingredients',
            'BN': 'brand_name',
            'SBD': 'semantic_branded_drug',
            'SCD': 'semantic_clinical_drug',
            'GPCK': 'generic_pack',
            'BPCK': 'branded_pack'
        }
        return mapping.get(tty, 'other')
    
    def _generate_safety_alerts(self, normalized_drugs: List[Dict], interactions: List[Dict]) -> List[str]:
        """Generate safety alerts based on drug analysis."""
        alerts = []
        
        # Check for normalization failures
        failed_normalizations = [d for d in normalized_drugs if not d['normalization_success']]
        if failed_normalizations:
            alerts.append(f"Could not normalize {len(failed_normalizations)} medications - manual review recommended")
        
        # Check for high-severity interactions
        high_severity_interactions = [i for i in interactions if i.get('severity', '').lower() in ['high', 'severe']]
        if high_severity_interactions:
            alerts.append(f"Found {len(high_severity_interactions)} high-severity drug interactions")
        
        # Check for moderate interactions
        moderate_interactions = [i for i in interactions if i.get('severity', '').lower() == 'moderate']
        if moderate_interactions:
            alerts.append(f"Found {len(moderate_interactions)} moderate drug interactions")
        
        # Check for duplicate drug classes (potential duplication)
        drug_classes = []
        for drug in normalized_drugs:
            for drug_class in drug.get('classes', []):
                drug_classes.append(drug_class.get('name', ''))
        
        class_counts = {}
        for drug_class in drug_classes:
            class_counts[drug_class] = class_counts.get(drug_class, 0) + 1
        
        duplicates = [cls for cls, count in class_counts.items() if count > 1]
        if duplicates:
            alerts.append(f"Potential medication duplication detected in classes: {', '.join(duplicates[:3])}")
        
        return alerts
    
    def _calculate_overall_medication_risk(self, interactions: List[Dict], alerts: List[str]) -> str:
        """Calculate overall medication risk level."""
        risk_score = 0
        
        # Score based on interactions
        for interaction in interactions:
            severity = interaction.get('severity', '').lower()
            if severity in ['high', 'severe']:
                risk_score += 3
            elif severity == 'moderate':
                risk_score += 2
            elif severity in ['low', 'mild']:
                risk_score += 1
        
        # Score based on alerts
        risk_score += len(alerts)
        
        # Determine risk level
        if risk_score >= 5:
            return 'high'
        elif risk_score >= 2:
            return 'moderate'
        else:
            return 'low'
