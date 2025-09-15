"""Claim extraction from care plan recommendations."""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class ClaimContext(BaseModel):
    """Patient context for claim extraction."""
    age: Optional[int] = None
    sex: Optional[str] = None
    conditions: List[str] = []
    medications: List[str] = []
    ckd: bool = False
    pregnancy: bool = False
    comorbidities: List[str] = []


class ExtractedClaim(BaseModel):
    """Extracted claim from recommendation."""
    id: str
    text: str
    context: ClaimContext
    policy: str  # benefit, safety, monitoring, education
    category: str  # dietary, exercise, medication, monitoring
    original_recommendation: str


class ClaimExtractor:
    """Rule-based claim extractor for care plan recommendations."""
    
    def __init__(self):
        # Exercise-related claim templates
        self.exercise_templates = [
            {
                "pattern": r"(\d+)\s*minutes?\s*of\s*(moderate|vigorous|low)\s*(intensity\s*)?(aerobic|cardio|exercise|activity)",
                "claim_template": "For adults with {conditions}, does {duration} min/week {intensity} exercise improve {outcome}?",
                "policy": "benefit",
                "category": "exercise"
            },
            {
                "pattern": r"(walking|running|cycling|swimming|dancing)\s*(\d+)\s*times?\s*per\s*week",
                "claim_template": "For adults with {conditions}, does {activity} {frequency} times/week improve {outcome}?",
                "policy": "benefit", 
                "category": "exercise"
            },
            {
                "pattern": r"(resistance|strength|weight)\s*training\s*(\d+)\s*times?\s*per\s*week",
                "claim_template": "For adults with {conditions}, does {activity} {frequency} times/week improve {outcome}?",
                "policy": "benefit",
                "category": "exercise"
            }
        ]
        
        # Dietary claim templates
        self.dietary_templates = [
            {
                "pattern": r"(reduce|limit|restrict)\s*(carbohydrate|carbs|sugar|sodium|salt)\s*(intake|consumption)",
                "claim_template": "For adults with {conditions}, does {action} {nutrient} intake improve {outcome}?",
                "policy": "benefit",
                "category": "dietary"
            },
            {
                "pattern": r"(increase|boost|add)\s*(fiber|protein|vegetables|fruits)\s*(intake|consumption)",
                "claim_template": "For adults with {conditions}, does {action} {nutrient} intake improve {outcome}?",
                "policy": "benefit",
                "category": "dietary"
            },
            {
                "pattern": r"(calorie|caloric)\s*(restriction|reduction|deficit)\s*of\s*(\d+)",
                "claim_template": "For adults with {conditions}, does {calorie} calorie {action} improve {outcome}?",
                "policy": "benefit",
                "category": "dietary"
            }
        ]
        
        # Medication safety templates
        self.medication_templates = [
            {
                "pattern": r"(start|initiate|begin)\s*(metformin|insulin|sulfonylurea|glp-1|sglt2)",
                "claim_template": "For adults with {conditions}, is {medication} safe and effective for {outcomes}?",
                "policy": "safety",
                "category": "medication"
            },
            {
                "pattern": r"(avoid|contraindicated|not recommended)\s*(sulfonylurea|metformin|insulin)",
                "claim_template": "For adults with {conditions}, should {medication} be avoided due to {reason}?",
                "policy": "safety",
                "category": "medication"
            },
            {
                "pattern": r"(monitor|check|measure)\s*(glucose|hba1c|blood pressure|weight)",
                "claim_template": "For adults with {conditions}, should {parameter} be monitored for {outcomes}?",
                "policy": "monitoring",
                "category": "monitoring"
            }
        ]
        
        # Outcome mappings
        self.outcome_mappings = {
            "diabetes": ["glycemic control", "HbA1c reduction", "glucose management"],
            "cardiovascular": ["cardiovascular health", "blood pressure control", "heart health"],
            "weight": ["weight management", "weight loss", "BMI reduction"],
            "general": ["health outcomes", "disease management", "quality of life"]
        }
    
    def extract_claims(self, care_plan: Dict[str, Any], patient_context: ClaimContext) -> List[ExtractedClaim]:
        """Extract verifiable claims from care plan recommendations."""
        claims = []
        claim_id = 1
        
        # Process each category in the care plan
        for category, recommendations in care_plan.items():
            if not recommendations or not isinstance(recommendations, dict):
                continue
                
            category_claims = self._extract_category_claims(
                category, recommendations, patient_context, claim_id
            )
            claims.extend(category_claims)
            claim_id += len(category_claims)
        
        logger.info("Extracted claims from care plan", 
                   total_claims=len(claims),
                   categories=list(care_plan.keys()))
        
        return claims
    
    def _extract_category_claims(
        self, 
        category: str, 
        recommendations: Dict[str, Any], 
        patient_context: ClaimContext,
        start_id: int
    ) -> List[ExtractedClaim]:
        """Extract claims from a specific category."""
        claims = []
        claim_id = start_id
        
        # Get relevant templates for this category
        templates = self._get_templates_for_category(category)
        
        # Extract text from recommendations
        recommendation_texts = self._extract_text_from_recommendations(recommendations)
        
        for text in recommendation_texts:
            for template in templates:
                matches = re.finditer(template["pattern"], text, re.IGNORECASE)
                
                for match in matches:
                    try:
                        claim = self._create_claim_from_match(
                            match, template, text, patient_context, category, f"c{claim_id}"
                        )
                        if claim:
                            claims.append(claim)
                            claim_id += 1
                    except Exception as e:
                        logger.warning("Failed to create claim from match",
                                     text=text[:50] + "...",
                                     error=str(e))
        
        return claims
    
    def _get_templates_for_category(self, category: str) -> List[Dict[str, Any]]:
        """Get claim templates for a specific category."""
        if category == "exercise":
            return self.exercise_templates
        elif category == "dietary":
            return self.dietary_templates
        elif category in ["medication_safety", "medication"]:
            return self.medication_templates
        else:
            return []
    
    def _extract_text_from_recommendations(self, recommendations: Dict[str, Any]) -> List[str]:
        """Extract text strings from recommendation dictionary."""
        texts = []
        
        for key, value in recommendations.items():
            if isinstance(value, str) and value.strip():
                texts.append(value.strip())
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        texts.append(item.strip())
            elif isinstance(value, dict):
                # Recursively extract from nested dictionaries
                nested_texts = self._extract_text_from_recommendations(value)
                texts.extend(nested_texts)
        
        return texts
    
    def _create_claim_from_match(
        self,
        match: re.Match,
        template: Dict[str, Any],
        original_text: str,
        patient_context: ClaimContext,
        category: str,
        claim_id: str
    ) -> Optional[ExtractedClaim]:
        """Create a claim from a regex match."""
        try:
            # Extract matched groups
            groups = match.groups()
            
            # Map groups to template variables
            variables = self._map_groups_to_variables(groups, template)
            
            # Determine conditions and outcomes
            conditions = self._determine_conditions(patient_context)
            outcomes = self._determine_outcomes(conditions)
            
            # Fill in the claim template
            claim_text = template["claim_template"].format(
                conditions=", ".join(conditions),
                outcomes=", ".join(outcomes),
                **variables
            )
            
            return ExtractedClaim(
                id=claim_id,
                text=claim_text,
                context=patient_context,
                policy=template["policy"],
                category=category,
                original_recommendation=original_text
            )
            
        except Exception as e:
            logger.warning("Failed to create claim from match", error=str(e))
            return None
    
    def _map_groups_to_variables(self, groups: Tuple[str, ...], template: Dict[str, Any]) -> Dict[str, str]:
        """Map regex groups to template variables."""
        variables = {}
        
        # Common mappings
        if len(groups) >= 1:
            variables["duration"] = groups[0] if groups[0] else "30"
            variables["activity"] = groups[0] if groups[0] else "exercise"
            variables["action"] = groups[0] if groups[0] else "increase"
            variables["nutrient"] = groups[0] if groups[0] else "fiber"
            variables["medication"] = groups[0] if groups[0] else "medication"
            variables["parameter"] = groups[0] if groups[0] else "glucose"
        
        if len(groups) >= 2:
            variables["intensity"] = groups[1] if groups[1] else "moderate"
            variables["frequency"] = groups[1] if groups[1] else "3"
            variables["calorie"] = groups[1] if groups[1] else "500"
        
        if len(groups) >= 3:
            variables["reason"] = groups[2] if groups[2] else "safety concerns"
        else:
            variables["reason"] = "safety concerns"
        
        return variables
    
    def _determine_conditions(self, patient_context: ClaimContext) -> List[str]:
        """Determine relevant conditions for the claim."""
        conditions = []
        
        # Add primary conditions
        if patient_context.conditions:
            conditions.extend(patient_context.conditions)
        
        # Add specific conditions based on context
        if patient_context.ckd:
            conditions.append("chronic kidney disease")
        if patient_context.pregnancy:
            conditions.append("pregnancy")
        
        # Default to diabetes if no conditions specified
        if not conditions:
            conditions.append("type 2 diabetes")
        
        return conditions
    
    def _determine_outcomes(self, conditions: List[str]) -> List[str]:
        """Determine relevant outcomes based on conditions."""
        outcomes = []
        
        for condition in conditions:
            condition_lower = condition.lower()
            
            if "diabetes" in condition_lower:
                outcomes.extend(self.outcome_mappings["diabetes"])
            elif "cardiovascular" in condition_lower or "heart" in condition_lower:
                outcomes.extend(self.outcome_mappings["cardiovascular"])
            elif "weight" in condition_lower or "obesity" in condition_lower:
                outcomes.extend(self.outcome_mappings["weight"])
            else:
                outcomes.extend(self.outcome_mappings["general"])
        
        # Remove duplicates and limit to top 3
        return list(set(outcomes))[:3]
    
    def extract_medication_safety_claims(
        self, 
        medications: List[str], 
        patient_context: ClaimContext
    ) -> List[ExtractedClaim]:
        """Extract safety claims for specific medications."""
        claims = []
        claim_id = 1
        
        for medication in medications:
            # Check for contraindications
            contraindication_claim = ExtractedClaim(
                id=f"med_safety_{claim_id}",
                text=f"For adults with {', '.join(patient_context.conditions or ['type 2 diabetes'])}, is {medication} safe and effective?",
                context=patient_context,
                policy="safety",
                category="medication",
                original_recommendation=f"Medication safety check for {medication}"
            )
            claims.append(contraindication_claim)
            claim_id += 1
            
            # Check for interactions
            interaction_claim = ExtractedClaim(
                id=f"med_interaction_{claim_id}",
                text=f"For adults taking {medication}, are there any significant drug interactions?",
                context=patient_context,
                policy="safety",
                category="medication",
                original_recommendation=f"Drug interaction check for {medication}"
            )
            claims.append(interaction_claim)
            claim_id += 1
        
        return claims


def create_claim_extractor() -> ClaimExtractor:
    """Factory function to create claim extractor."""
    return ClaimExtractor()
