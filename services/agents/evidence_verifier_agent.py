"""Evidence verification agent that interfaces with verification service."""

import httpx
from typing import Dict, Any
from datetime import datetime
import structlog

from base_agent import BaseAgent, AgentContext

logger = structlog.get_logger()


class EvidenceVerifierAgent(BaseAgent):
    """Agent responsible for evidence verification of care recommendations."""
    
    def __init__(self, verifier_service_url: str = "http://verifier:8000"):
        super().__init__("evidence_verifier", "Evidence-based verification of care recommendations")
        self.verifier_service_url = verifier_service_url
    
    async def process(self, context: AgentContext) -> AgentContext:
        """Verify care plan recommendations with evidence."""
        try:
            self.log_processing(context, "evidence_verification_started")
            
            # Get care plan from previous agent
            careplan_output = context.get_agent_output('careplan_generator')
            if not careplan_output:
                raise ValueError("Care plan required for evidence verification")
            
            care_plan = careplan_output.get('care_plan', {})
            
            # Prepare verification request
            verification_request = self._prepare_verification_request(context, care_plan)
            
            # Call evidence verification service
            verification_result = await self._call_verification_service(verification_request)
            
            # Process verification results
            processed_result = self._process_verification_result(verification_result)
            
            # Add to context
            context.add_agent_output('evidence_verifier', processed_result)
            
            self.log_processing(context, "evidence_verification_completed",
                              overall_score=processed_result.get('overall_score'),
                              status=processed_result.get('status'))
            
            return context
            
        except Exception as e:
            return self.handle_error(context, e)
    
    def _prepare_verification_request(self, context: AgentContext, 
                                    care_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare request for evidence verification service."""
        return {
            'patient_context': {
                'demographics': context.patient_profile,
                'conditions': context.conditions,
                'medications': context.medications,
                'vitals': context.vitals
            },
            'care_plan': care_plan,
            'focus_areas': self._identify_focus_areas(care_plan),
            'include_safety_check': True,
            'max_evidence_per_category': 5
        }
    
    def _identify_focus_areas(self, care_plan: Dict[str, Any]) -> list:
        """Identify key areas for focused evidence verification."""
        focus_areas = []
        
        # Check which care plan components have recommendations
        if care_plan.get('dietary', {}).get('recommendations'):
            focus_areas.append('dietary_interventions')
        
        if care_plan.get('exercise', {}).get('aerobic'):
            focus_areas.append('exercise_therapy')
        
        if care_plan.get('medication_safety', {}).get('current_regimen'):
            focus_areas.append('medication_safety')
        
        if care_plan.get('monitoring', {}).get('glucose'):
            focus_areas.append('glucose_monitoring')
        
        return focus_areas or ['general_diabetes_care']
    
    async def _call_verification_service(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Call evidence verification service."""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:  # Longer timeout for evidence gathering
                response = await client.post(
                    f"{self.verifier_service_url}/verify",
                    json=request
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.error("Verification service error", 
                                    status=response.status_code,
                                    response=response.text)
                    return self._get_fallback_verification()
                    
        except httpx.RequestError as e:
            self.logger.error("Failed to connect to verification service", error=str(e))
            return self._get_fallback_verification()
        except Exception as e:
            self.logger.error("Verification service call failed", error=str(e))
            return self._get_fallback_verification()
    
    def _process_verification_result(self, raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enhance verification results."""
        try:
            overall_score = raw_result.get('overall_score', 0.0)
            status = raw_result.get('status', 'flagged')
            evidence_links = raw_result.get('evidence_links', [])
            safety_alerts = raw_result.get('safety_alerts', [])
            warnings = raw_result.get('warnings', [])
            quality_breakdown = raw_result.get('quality_breakdown', {})
            
            # Process evidence links for clinical context
            processed_evidence = self._process_evidence_links(evidence_links)
            
            # Generate clinical interpretation
            clinical_interpretation = self._generate_clinical_interpretation(
                overall_score, status, quality_breakdown, safety_alerts
            )
            
            # Determine recommendation adjustments
            adjustments = self._determine_adjustments(status, safety_alerts, warnings)
            
            return {
                'overall_score': overall_score,
                'status': status,
                'evidence_strength': self._categorize_evidence_strength(overall_score),
                'evidence_links': processed_evidence,
                'safety_alerts': safety_alerts,
                'warnings': warnings,
                'quality_breakdown': quality_breakdown,
                'clinical_interpretation': clinical_interpretation,
                'recommended_adjustments': adjustments,
                'verification_metadata': {
                    'evidence_sources_checked': len(evidence_links),
                    'verification_timestamp': datetime.utcnow().isoformat(),
                    'guidelines_referenced': self._extract_guidelines_referenced(evidence_links)
                }
            }
            
        except Exception as e:
            self.logger.error("Failed to process verification result", error=str(e))
            return self._get_fallback_verification()
    
    def _process_evidence_links(self, evidence_links: list) -> list:
        """Process evidence links for clinical presentation."""
        processed = []
        
        for evidence in evidence_links:
            processed_item = {
                'title': evidence.get('title', ''),
                'source_type': evidence.get('source_type', ''),
                'quality_score': evidence.get('quality_score', 0.0),
                'url': evidence.get('url', ''),
                'clinical_relevance': self._assess_clinical_relevance(evidence),
                'summary': self._create_evidence_summary(evidence),
                'strength_rating': self._rate_evidence_strength(evidence)
            }
            processed.append(processed_item)
        
        # Sort by quality score and clinical relevance
        processed.sort(key=lambda x: (x['quality_score'], x['clinical_relevance']), reverse=True)
        
        return processed[:10]  # Top 10 most relevant
    
    def _assess_clinical_relevance(self, evidence: Dict[str, Any]) -> float:
        """Assess clinical relevance of evidence item."""
        relevance_score = 0.5  # Base score
        
        title = evidence.get('title', '').lower()
        source_type = evidence.get('source_type', '')
        
        # Higher relevance for guidelines
        if source_type == 'guideline':
            relevance_score += 0.3
        
        # Higher relevance for diabetes/cardiovascular content
        diabetes_terms = ['diabetes', 'glycemic', 'hba1c', 'glucose']
        cardio_terms = ['cardiovascular', 'blood pressure', 'hypertension', 'cholesterol']
        
        if any(term in title for term in diabetes_terms):
            relevance_score += 0.2
        if any(term in title for term in cardio_terms):
            relevance_score += 0.2
        
        # Higher relevance for lifestyle interventions
        lifestyle_terms = ['diet', 'exercise', 'lifestyle', 'physical activity']
        if any(term in title for term in lifestyle_terms):
            relevance_score += 0.1
        
        return min(1.0, relevance_score)
    
    def _create_evidence_summary(self, evidence: Dict[str, Any]) -> str:
        """Create clinical summary of evidence item."""
        title = evidence.get('title', '')
        source_type = evidence.get('source_type', '')
        quality_score = evidence.get('quality_score', 0.0)
        
        summary = f"{title} - "
        
        if source_type == 'guideline':
            summary += "Clinical practice guideline"
        elif source_type == 'rct':
            summary += "Randomized controlled trial"
        elif source_type == 'meta_analysis':
            summary += "Meta-analysis"
        else:
            summary += f"{source_type.replace('_', ' ').title()} study"
        
        # Add quality indicator
        if quality_score >= 0.8:
            summary += " (High quality evidence)"
        elif quality_score >= 0.6:
            summary += " (Moderate quality evidence)"
        else:
            summary += " (Lower quality evidence)"
        
        return summary
    
    def _rate_evidence_strength(self, evidence: Dict[str, Any]) -> str:
        """Rate the strength of evidence using standard grades."""
        quality_score = evidence.get('quality_score', 0.0)
        source_type = evidence.get('source_type', '')
        
        # Grade A: High-quality evidence
        if quality_score >= 0.85 and source_type in ['guideline', 'meta_analysis', 'rct']:
            return 'A'
        
        # Grade B: Moderate-quality evidence
        elif quality_score >= 0.7 and source_type in ['guideline', 'meta_analysis', 'rct', 'cohort']:
            return 'B'
        
        # Grade C: Low-quality evidence
        elif quality_score >= 0.5:
            return 'C'
        
        # Grade D: Very low-quality evidence
        else:
            return 'D'
    
    def _generate_clinical_interpretation(self, overall_score: float, status: str,
                                        quality_breakdown: Dict[str, int],
                                        safety_alerts: list) -> str:
        """Generate clinical interpretation of evidence verification."""
        interpretation = ""
        
        # Overall strength assessment
        if overall_score >= 0.8:
            interpretation += "Strong evidence supports the recommended care plan. "
        elif overall_score >= 0.6:
            interpretation += "Moderate evidence supports the care plan with some areas for consideration. "
        else:
            interpretation += "Limited evidence available; recommendations based on clinical guidelines and expert consensus. "
        
        # Quality breakdown
        high_quality = quality_breakdown.get('high', 0)
        moderate_quality = quality_breakdown.get('moderate', 0)
        
        if high_quality > 0:
            interpretation += f"{high_quality} high-quality evidence sources identified. "
        if moderate_quality > 0:
            interpretation += f"{moderate_quality} moderate-quality evidence sources support recommendations. "
        
        # Safety considerations
        if safety_alerts:
            interpretation += f"Important safety considerations identified ({len(safety_alerts)} alerts). "
        
        # Status-specific guidance
        if status == 'flagged':
            interpretation += "Recommendations require clinical review due to safety concerns or limited evidence."
        elif status == 'approved':
            interpretation += "Evidence quality meets clinical standards for implementation."
        
        return interpretation
    
    def _determine_adjustments(self, status: str, safety_alerts: list, 
                             warnings: list) -> list:
        """Determine recommended adjustments based on verification."""
        adjustments = []
        
        # Status-based adjustments
        if status == 'flagged':
            adjustments.append("Require clinician review before implementation")
            
            if safety_alerts:
                adjustments.append("Address safety concerns identified in evidence review")
            
            if any('limited_evidence' in warning for warning in warnings):
                adjustments.append("Consider additional clinical assessment given limited evidence")
        
        # Safety-specific adjustments
        black_box_warnings = [alert for alert in safety_alerts if 'black_box' in alert.lower()]
        if black_box_warnings:
            adjustments.append("Review medication black box warnings with patient")
        
        interaction_warnings = [alert for alert in safety_alerts if 'interaction' in alert.lower()]
        if interaction_warnings:
            adjustments.append("Monitor for drug interactions and adjust accordingly")
        
        # Warning-based adjustments
        if any('preliminary' in warning for warning in warnings):
            adjustments.append("Consider recommendations as preliminary pending additional evidence")
        
        if any('small_sample' in warning for warning in warnings):
            adjustments.append("Recognize evidence limitations from small sample studies")
        
        return adjustments
    
    def _categorize_evidence_strength(self, overall_score: float) -> str:
        """Categorize overall evidence strength."""
        if overall_score >= 0.8:
            return "Strong"
        elif overall_score >= 0.65:
            return "Moderate"
        elif overall_score >= 0.5:
            return "Limited"
        else:
            return "Very Limited"
    
    def _extract_guidelines_referenced(self, evidence_links: list) -> list:
        """Extract clinical guidelines referenced in evidence."""
        guidelines = []
        
        for evidence in evidence_links:
            if evidence.get('source_type') == 'guideline':
                title = evidence.get('title', '')
                
                if 'ada' in title.lower() or 'american diabetes' in title.lower():
                    guidelines.append('ADA Standards of Care')
                elif 'acc' in title.lower() or 'aha' in title.lower():
                    guidelines.append('ACC/AHA Guidelines')
                elif 'aace' in title.lower():
                    guidelines.append('AACE Guidelines')
                else:
                    guidelines.append('Clinical Practice Guideline')
        
        return list(set(guidelines))  # Remove duplicates
    
    def _get_fallback_verification(self) -> Dict[str, Any]:
        """Get fallback verification when service is unavailable."""
        return {
            'overall_score': 0.6,  # Moderate default
            'status': 'flagged',   # Conservative default
            'evidence_strength': 'Limited',
            'evidence_links': [
                {
                    'title': 'ADA Standards of Care in Diabetes 2024',
                    'source_type': 'guideline',
                    'quality_score': 0.95,
                    'url': 'https://diabetesjournals.org/care/issue/47/Supplement_1',
                    'clinical_relevance': 1.0,
                    'summary': 'American Diabetes Association clinical practice guidelines (High quality evidence)',
                    'strength_rating': 'A'
                }
            ],
            'safety_alerts': [],
            'warnings': ['Evidence verification service unavailable - using guideline defaults'],
            'quality_breakdown': {'high': 1, 'moderate': 0, 'low': 0},
            'clinical_interpretation': 'Evidence verification service unavailable. Recommendations based on established clinical guidelines (ADA Standards of Care). Clinical review recommended.',
            'recommended_adjustments': [
                'Manual evidence review recommended',
                'Verify against current clinical guidelines',
                'Consider individual patient factors'
            ],
            'verification_metadata': {
                'evidence_sources_checked': 1,
                'verification_timestamp': datetime.utcnow().isoformat(),
                'guidelines_referenced': ['ADA Standards of Care'],
                'fallback_used': True
            }
        }
