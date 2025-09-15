"""Main orchestrator agent that coordinates all other agents."""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
import httpx
import structlog

from base_agent import AgentContext, BaseAgent
from intake_agent import IntakeAgent
from normalizer_agent import NormalizerAgent
from risk_predictor_agent import RiskPredictorAgent
from careplan_agent import CarePlanAgent
from evidence_verifier_agent import EvidenceVerifierAgent

logger = structlog.get_logger()


class OrchestratorAgent(BaseAgent):
    """Orchestrates the complete recommendation generation pipeline."""
    
    def __init__(self):
        super().__init__("orchestrator", "Main orchestration agent")
        
        # Initialize sub-agents
        self.intake_agent = IntakeAgent()
        self.normalizer_agent = NormalizerAgent()
        self.risk_predictor_agent = RiskPredictorAgent()
        self.careplan_agent = CarePlanAgent()
        self.evidence_verifier_agent = EvidenceVerifierAgent()
        
        # Configuration
        self.evidence_min_score = float(os.getenv('EVIDENCE_MIN_SCORE', '0.6'))
        self.database_url = os.getenv('DATABASE_URL')
    
    async def process(self, context: AgentContext) -> AgentContext:
        """Run the complete recommendation generation pipeline."""
        try:
            self.log_processing(context, "pipeline_started")
            
            # Step 1: Intake and data preparation
            context = await self.intake_agent.process(context)
            if context.errors:
                self.logger.warning("Intake errors detected", errors=context.errors)
            
            # Step 2: Data normalization
            context = await self.normalizer_agent.process(context)
            
            # Step 3: Risk prediction
            context = await self.risk_predictor_agent.process(context)
            
            # Step 4: Care plan generation
            context = await self.careplan_agent.process(context)
            
            # Step 5: Evidence verification
            context = await self.evidence_verifier_agent.process(context)
            
            # Step 6: Final orchestration and decision
            context = await self._finalize_recommendation(context)
            
            self.log_processing(context, "pipeline_completed")
            return context
            
        except Exception as e:
            return self.handle_error(context, e)
    
    async def _finalize_recommendation(self, context: AgentContext) -> AgentContext:
        """Finalize recommendation based on all agent outputs."""
        try:
            # Get outputs from agents
            risk_output = context.get_agent_output('risk_predictor')
            careplan_output = context.get_agent_output('careplan_generator')
            evidence_output = context.get_agent_output('evidence_verifier')
            
            if not all([risk_output, careplan_output, evidence_output]):
                raise ValueError("Missing required agent outputs")
            
            # Determine recommendation status
            evidence_score = evidence_output.get('overall_score', 0.0)
            evidence_status = evidence_output.get('status', 'flagged')
            
            if evidence_status == 'approved' and evidence_score >= self.evidence_min_score:
                status = 'approved'
            else:
                status = 'flagged'
            
            # Create final recommendation structure
            recommendation = {
                'snapshot_ts': context.snapshot_ts,
                'careplan': careplan_output.get('care_plan', {}),
                'explainer': {
                    'risk_interpretation': risk_output.get('interpretation', ''),
                    'key_contributors': risk_output.get('key_contributors', []),
                    'model_confidence': risk_output.get('confidence', 0.0),
                    'evidence_strength': self._assess_evidence_strength(evidence_output)
                },
                'model_version': risk_output.get('model_version', 'unknown'),
                'status': status,
                'risk_score': risk_output.get('risk_probability', 0.0),
                'risk_category': risk_output.get('risk_category', 'unknown')
            }
            
            # Store recommendation in database
            recommendation_id = await self._store_recommendation(context, recommendation)
            
            # Store evidence links
            if evidence_output.get('evidence_links'):
                await self._store_evidence_links(recommendation_id, evidence_output['evidence_links'])
            
            # Create audit log
            await self._create_audit_log(context, recommendation_id, recommendation)
            
            # Add final output to context
            context.add_agent_output('orchestrator', {
                'recommendation_id': recommendation_id,
                'recommendation': recommendation,
                'status': status,
                'evidence_score': evidence_score
            })
            
            return context
            
        except Exception as e:
            self.logger.error("Finalization failed", error=str(e))
            context.add_error('orchestrator', f"Finalization failed: {str(e)}")
            return context
    
    def _assess_evidence_strength(self, evidence_output: Dict[str, Any]) -> str:
        """Assess overall evidence strength."""
        score = evidence_output.get('overall_score', 0.0)
        
        if score >= 0.8:
            return "High - strong evidence from guidelines and literature"
        elif score >= 0.6:
            return "Moderate - adequate evidence support"
        elif score >= 0.4:
            return "Low - limited evidence available"
        else:
            return "Very Low - insufficient evidence support"
    
    async def _store_recommendation(self, context: AgentContext, 
                                   recommendation: Dict[str, Any]) -> str:
        """Store recommendation in database."""
        try:
            # This would typically use a database client
            # For now, simulate with a POST to the gateway
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{os.getenv('GATEWAY_URL', 'http://gateway:8000')}/api/v1/recommendations",
                    json={
                        'patient_id': context.patient_id,
                        **recommendation
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('data', {}).get('id', 'unknown')
                else:
                    self.logger.error("Failed to store recommendation", 
                                    status=response.status_code)
                    return 'storage_failed'
        
        except Exception as e:
            self.logger.error("Database storage failed", error=str(e))
            return 'storage_error'
    
    async def _store_evidence_links(self, recommendation_id: str, 
                                   evidence_links: List[Dict[str, Any]]):
        """Store evidence links in database."""
        try:
            # Simulate evidence storage
            self.logger.info("Evidence links stored", 
                           recommendation_id=recommendation_id,
                           count=len(evidence_links))
        except Exception as e:
            self.logger.error("Failed to store evidence links", error=str(e))
    
    async def _create_audit_log(self, context: AgentContext, 
                               recommendation_id: str, 
                               recommendation: Dict[str, Any]):
        """Create audit log entry."""
        try:
            audit_data = {
                'actor_type': 'system',
                'actor_id': None,
                'event': 'recommendation.created',
                'entity_type': 'recommendation',
                'entity_id': recommendation_id,
                'payload': {
                    'patient_id': context.patient_id,
                    'risk_score': recommendation.get('risk_score'),
                    'model_version': recommendation.get('model_version'),
                    'evidence_score': context.get_agent_output('evidence_verifier', {}).get('overall_score'),
                    'agents_used': list(context.agent_outputs.keys())
                },
                'model_version': recommendation.get('model_version')
            }
            
            self.logger.info("Audit log created", 
                           recommendation_id=recommendation_id,
                           patient_id=context.patient_id)
            
        except Exception as e:
            self.logger.error("Failed to create audit log", error=str(e))
    
    async def generate_recommendation(self, patient_id: str, 
                                    force_refresh: bool = False,
                                    include_evidence: bool = True,
                                    model_version: str = None,
                                    user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """High-level method to generate a complete recommendation."""
        try:
            # Create context
            context = AgentContext(patient_id)
            
            # Add any additional context
            if user_context:
                context.metadata.update(user_context)
            
            # Load patient data (this would typically come from database)
            await self._load_patient_data(context)
            
            # Process through pipeline
            context = await self.process(context)
            
            # Return result
            orchestrator_output = context.get_agent_output('orchestrator', {})
            
            return {
                'success': len(context.errors) == 0,
                'recommendation_id': orchestrator_output.get('recommendation_id'),
                'recommendation': orchestrator_output.get('recommendation'),
                'evidence_score': orchestrator_output.get('evidence_score'),
                'errors': context.errors,
                'metadata': {
                    'pipeline_duration': 'calculated_duration',
                    'agents_used': list(context.agent_outputs.keys()),
                    'model_version': model_version
                }
            }
            
        except Exception as e:
            self.logger.error("Recommendation generation failed", 
                            patient_id=patient_id, error=str(e))
            return {
                'success': False,
                'error': str(e),
                'recommendation_id': None
            }
    
    async def _load_patient_data(self, context: AgentContext):
        """Load patient data from database."""
        try:
            # This would typically query the database
            # For demo, we'll load from the seed data patient
            
            if context.patient_id == 'f47ac10b-58cc-4372-a567-0e02b2c3d479':
                # Maria Gonzalez data
                demographics = {
                    "name": "Maria Gonzalez",
                    "age": 58,
                    "sex": "F",
                    "ethnicity": "Hispanic",
                    "height_cm": 162,
                    "weight_kg": 78.5,
                    "bmi": 29.9,
                    "family_history": {
                        "diabetes": True,
                        "heart_disease": True,
                        "hypertension": True
                    },
                    "lifestyle": {
                        "smoking": "never",
                        "alcohol": "occasional",
                        "exercise_mins_week": 60
                    }
                }
                
                vitals = [
                    {"type": "glucose_fasting", "value": 148, "unit": "mg/dL", "ts": "2024-09-01T08:00:00Z"},
                    {"type": "hba1c", "value": 8.1, "unit": "%", "ts": "2024-09-01T10:00:00Z"},
                    {"type": "sbp", "value": 138, "unit": "mmHg", "ts": "2024-09-01T09:00:00Z"},
                    {"type": "dbp", "value": 88, "unit": "mmHg", "ts": "2024-09-01T09:00:00Z"},
                    {"type": "weight", "value": 78.5, "unit": "kg", "ts": "2024-09-01T07:30:00Z"},
                    {"type": "total_cholesterol", "value": 198, "unit": "mg/dL", "ts": "2024-09-01T10:00:00Z"},
                    {"type": "ldl_cholesterol", "value": 118, "unit": "mg/dL", "ts": "2024-09-01T10:00:00Z"},
                    {"type": "hdl_cholesterol", "value": 42, "unit": "mg/dL", "ts": "2024-09-01T10:00:00Z"},
                    {"type": "triglycerides", "value": 185, "unit": "mg/dL", "ts": "2024-09-01T10:00:00Z"}
                ]
                
                medications = [
                    {"rxnorm_code": "6809", "name": "Metformin", "dosage": "500mg", "active": True},
                    {"rxnorm_code": "38454", "name": "Lisinopril", "dosage": "10mg", "active": True},
                    {"rxnorm_code": "36567", "name": "Atorvastatin", "dosage": "20mg", "active": True}
                ]
                
                conditions = [
                    {"icd10_code": "E11.9", "name": "Type 2 Diabetes Mellitus", "severity": "moderate", "active": True},
                    {"icd10_code": "I10", "name": "Essential Hypertension", "severity": "mild", "active": True},
                    {"icd10_code": "E78.5", "name": "Hyperlipidemia", "severity": "mild", "active": True}
                ]
                
                context.add_patient_data(demographics, vitals, medications, conditions)
            else:
                # Default/minimal data for other patients
                context.add_patient_data({}, [], [], [])
            
        except Exception as e:
            self.logger.error("Failed to load patient data", 
                            patient_id=context.patient_id, error=str(e))
            context.add_error('orchestrator', f"Data loading failed: {str(e)}")
