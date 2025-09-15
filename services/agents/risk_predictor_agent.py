"""Risk prediction agent that interfaces with ML service."""

import httpx
from typing import Dict, Any
from datetime import datetime
import structlog

from base_agent import BaseAgent, AgentContext

logger = structlog.get_logger()


class RiskPredictorAgent(BaseAgent):
    """Agent responsible for ML-based risk prediction."""
    
    def __init__(self, ml_service_url: str = "http://ml_risk:8000"):
        super().__init__("risk_predictor", "ML-based cardiometabolic risk prediction")
        self.ml_service_url = ml_service_url
    
    async def process(self, context: AgentContext) -> AgentContext:
        """Process patient data through ML risk prediction."""
        try:
            self.log_processing(context, "risk_prediction_started")
            
            # Prepare patient data for ML service
            patient_data = self._prepare_patient_data(context)
            
            # Call ML service for risk prediction
            risk_prediction = await self._call_ml_service(patient_data)
            
            # Process and validate the prediction
            processed_prediction = self._process_prediction(risk_prediction)
            
            # Add to context
            context.add_agent_output('risk_predictor', processed_prediction)
            
            self.log_processing(context, "risk_prediction_completed",
                              risk_score=processed_prediction.get('risk_probability'),
                              risk_category=processed_prediction.get('risk_category'))
            
            return context
            
        except Exception as e:
            return self.handle_error(context, e)
    
    def _prepare_patient_data(self, context: AgentContext) -> Dict[str, Any]:
        """Prepare patient data for ML service."""
        return {
            'demographics': context.patient_profile,
            'vitals': context.vitals,
            'conditions': context.conditions,
            'medications': context.medications
        }
    
    async def _call_ml_service(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """Call ML service for risk prediction."""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ml_service_url}/predict",
                    json={
                        "patient_data": patient_data,
                        "include_shap": True
                    }
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    self.logger.error("ML service error", 
                                    status=response.status_code,
                                    response=response.text)
                    return self._get_fallback_prediction()
                    
        except httpx.RequestError as e:
            self.logger.error("Failed to connect to ML service", error=str(e))
            return self._get_fallback_prediction()
        except Exception as e:
            self.logger.error("ML service call failed", error=str(e))
            return self._get_fallback_prediction()
    
    def _process_prediction(self, raw_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate ML prediction results."""
        try:
            # Extract core prediction data
            risk_probability = raw_prediction.get('risk_probability', 0.0)
            risk_category = raw_prediction.get('risk_category', 'unknown')
            model_version = raw_prediction.get('model_version', 'unknown')
            
            # Extract SHAP explanations
            shap_explanations = raw_prediction.get('shap_explanations', {})
            
            # Create interpretable explanation
            interpretation = self._create_risk_interpretation(
                risk_probability, risk_category, shap_explanations
            )
            
            # Extract key contributors
            key_contributors = self._extract_key_contributors(shap_explanations)
            
            return {
                'risk_probability': risk_probability,
                'risk_category': risk_category,
                'model_version': model_version,
                'interpretation': interpretation,
                'key_contributors': key_contributors,
                'confidence': self._calculate_confidence(raw_prediction),
                'shap_data': shap_explanations,
                'processed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error("Failed to process prediction", error=str(e))
            return self._get_fallback_prediction()
    
    def _create_risk_interpretation(self, risk_prob: float, risk_category: str, 
                                   shap_data: Dict[str, Any]) -> str:
        """Create human-readable risk interpretation."""
        try:
            base_interpretation = f"Based on the current health profile, the estimated 5-year cardiometabolic risk is {risk_prob:.1%}, categorized as {risk_category} risk."
            
            # Add context based on risk level
            if risk_category == 'high':
                context = " This indicates elevated risk requiring immediate attention and intervention."
            elif risk_category == 'moderate':
                context = " This suggests moderate risk that would benefit from lifestyle modifications and monitoring."
            else:
                context = " This indicates relatively low risk with continued focus on prevention."
            
            # Add top contributors if available
            contributors_text = ""
            if shap_data.get('patient_explanations'):
                top_explanations = shap_data['patient_explanations'][:3]
                if top_explanations:
                    contributors_text = f" Key contributing factors include: {', '.join(top_explanations)}."
            
            return base_interpretation + context + contributors_text
            
        except Exception as e:
            self.logger.error("Failed to create interpretation", error=str(e))
            return f"Estimated {risk_category} cardiometabolic risk based on current health profile."
    
    def _extract_key_contributors(self, shap_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key risk contributors from SHAP data."""
        try:
            contributors = []
            
            # Get top positive and negative contributors
            top_positive = shap_data.get('top_positive_contributors', [])
            top_negative = shap_data.get('top_negative_contributors', [])
            
            # Process positive contributors (increase risk)
            for contrib in top_positive[:3]:  # Top 3
                contributors.append({
                    'factor': self._humanize_feature_name(contrib.get('feature', '')),
                    'impact': 'increases_risk',
                    'strength': self._categorize_impact_strength(contrib.get('abs_shap_value', 0)),
                    'value': contrib.get('value', ''),
                    'explanation': self._get_factor_explanation(contrib.get('feature', ''), 'increase')
                })
            
            # Process negative contributors (decrease risk)
            for contrib in top_negative[:2]:  # Top 2
                contributors.append({
                    'factor': self._humanize_feature_name(contrib.get('feature', '')),
                    'impact': 'decreases_risk',
                    'strength': self._categorize_impact_strength(contrib.get('abs_shap_value', 0)),
                    'value': contrib.get('value', ''),
                    'explanation': self._get_factor_explanation(contrib.get('feature', ''), 'decrease')
                })
            
            return contributors
            
        except Exception as e:
            self.logger.error("Failed to extract contributors", error=str(e))
            return []
    
    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert ML feature names to human-readable terms."""
        name_mapping = {
            'age': 'Age',
            'hba1c': 'HbA1c level',
            'glucose_fasting': 'Fasting glucose',
            'sbp': 'Systolic blood pressure',
            'dbp': 'Diastolic blood pressure',
            'bmi': 'Body Mass Index',
            'total_cholesterol': 'Total cholesterol',
            'ldl_cholesterol': 'LDL cholesterol',
            'hdl_cholesterol': 'HDL cholesterol',
            'triglycerides': 'Triglycerides',
            'sex': 'Gender',
            'smoking_status': 'Smoking history',
            'hypertension_stage': 'Blood pressure control',
            'diabetes_controlled': 'Diabetes management',
            'metabolic_syndrome_score': 'Metabolic syndrome indicators'
        }
        
        return name_mapping.get(feature_name, feature_name.replace('_', ' ').title())
    
    def _categorize_impact_strength(self, shap_value: float) -> str:
        """Categorize the strength of a factor's impact."""
        if shap_value >= 0.1:
            return 'high'
        elif shap_value >= 0.05:
            return 'moderate'
        else:
            return 'low'
    
    def _get_factor_explanation(self, feature_name: str, direction: str) -> str:
        """Get explanation for why a factor increases/decreases risk."""
        explanations = {
            'age': {
                'increase': 'Advanced age is associated with increased cardiovascular risk',
                'decrease': 'Younger age is protective against cardiovascular events'
            },
            'hba1c': {
                'increase': 'Elevated HbA1c indicates poor glucose control, increasing complications risk',
                'decrease': 'Well-controlled HbA1c reduces diabetes-related complications'
            },
            'sbp': {
                'increase': 'High blood pressure increases cardiovascular disease risk',
                'decrease': 'Well-controlled blood pressure is protective'
            },
            'bmi': {
                'increase': 'Higher BMI is associated with increased metabolic risk',
                'decrease': 'Healthy weight range reduces metabolic complications'
            },
            'hdl_cholesterol': {
                'increase': 'Higher HDL cholesterol is protective against heart disease',
                'decrease': 'Low HDL cholesterol increases cardiovascular risk'
            }
        }
        
        return explanations.get(feature_name, {}).get(direction, 
            f"This factor {direction.replace('_', ' ')}s overall risk based on clinical evidence")
    
    def _calculate_confidence(self, prediction: Dict[str, Any]) -> float:
        """Calculate confidence score for the prediction."""
        try:
            # Base confidence from model
            model_confidence = prediction.get('model_confidence', 0.8)
            
            # Adjust based on data completeness
            shap_data = prediction.get('shap_explanations', {})
            feature_count = len(shap_data.get('feature_importance', []))
            
            # Higher confidence with more features
            if feature_count >= 10:
                completeness_bonus = 0.1
            elif feature_count >= 5:
                completeness_bonus = 0.05
            else:
                completeness_bonus = 0.0
            
            confidence = min(1.0, model_confidence + completeness_bonus)
            return round(confidence, 2)
            
        except Exception:
            return 0.8  # Default confidence
    
    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Get fallback prediction when ML service is unavailable."""
        return {
            'risk_probability': 0.15,  # Conservative default
            'risk_category': 'moderate',
            'model_version': 'fallback',
            'interpretation': 'Unable to calculate precise risk score. Using conservative estimate based on clinical guidelines.',
            'key_contributors': [
                {
                    'factor': 'Clinical Assessment',
                    'impact': 'requires_evaluation',
                    'strength': 'moderate',
                    'explanation': 'Full risk assessment requires complete patient evaluation'
                }
            ],
            'confidence': 0.3,  # Low confidence for fallback
            'shap_data': {},
            'processed_at': datetime.utcnow().isoformat(),
            'fallback_used': True
        }
