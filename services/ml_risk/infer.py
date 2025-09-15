"""Model inference and SHAP explanation service."""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
import structlog

from featurize import FeatureProcessor, extract_patient_features

logger = structlog.get_logger()


class RiskPredictor:
    """Cardiometabolic risk prediction with SHAP explanations."""
    
    def __init__(self, model_path: str = None):
        self.model = None
        self.feature_processor = None
        self.explainer = None
        self.model_metadata = {}
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model and create SHAP explainer."""
        try:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            # Load model artifact
            model_artifact = joblib.load(model_path)
            
            self.model = model_artifact['model']
            self.feature_processor = model_artifact['feature_processor']
            self.model_metadata = {
                'version': model_artifact.get('version', 'unknown'),
                'algorithm': model_artifact.get('algorithm', 'unknown'),
                'created_at': model_artifact.get('created_at', 'unknown'),
                'feature_names': model_artifact.get('feature_names', [])
            }
            
            # Create SHAP explainer
            self._create_shap_explainer()
            
            logger.info("Model loaded successfully", 
                       version=self.model_metadata['version'],
                       algorithm=self.model_metadata['algorithm'])
            
            return True
            
        except Exception as e:
            logger.error("Failed to load model", path=str(model_path), error=str(e))
            return False
    
    def _create_shap_explainer(self):
        """Create SHAP explainer for the loaded model."""
        try:
            # For tree-based models, use TreeExplainer
            if hasattr(self.model, 'predict_proba') and hasattr(self.model, 'feature_importances_'):
                self.explainer = shap.TreeExplainer(self.model)
            else:
                # For other models, use more general explainer
                # This requires background data
                logger.warning("Using general SHAP explainer - may be slower")
                self.explainer = None  # Will need background data
            
            logger.info("SHAP explainer created successfully")
            
        except Exception as e:
            logger.error("Failed to create SHAP explainer", error=str(e))
            self.explainer = None
    
    def predict_risk(self, patient_data: Dict[str, Any], 
                    include_shap: bool = True) -> Dict[str, Any]:
        """Predict cardiometabolic risk for a patient."""
        try:
            if not self.model or not self.feature_processor:
                raise ValueError("Model not loaded")
            
            # Extract features from patient data
            features = extract_patient_features(patient_data)
            
            # Convert to DataFrame for processing
            feature_df = pd.DataFrame([features])
            
            # Process features
            processed_features = self.feature_processor.transform(feature_df)
            
            # Make prediction
            risk_prob = self.model.predict_proba(processed_features)[0, 1]
            risk_category = self._categorize_risk(risk_prob)
            
            # Prepare result
            result = {
                'risk_probability': float(risk_prob),
                'risk_category': risk_category,
                'model_version': self.model_metadata['version'],
                'algorithm': self.model_metadata['algorithm'],
                'features_used': list(processed_features.columns),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Add SHAP explanations if requested
            if include_shap and self.explainer:
                shap_values = self._calculate_shap_values(processed_features)
                result['shap_explanations'] = shap_values
            
            return result
            
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            return {
                'error': str(e),
                'risk_probability': 0.0,
                'risk_category': 'unknown',
                'model_version': self.model_metadata.get('version', 'unknown')
            }
    
    def _calculate_shap_values(self, features: pd.DataFrame) -> Dict[str, Any]:
        """Calculate SHAP values and create explanations."""
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(features)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Binary classification returns list of arrays
                shap_values = shap_values[1]  # Use positive class
            
            # Get feature contributions
            feature_names = features.columns.tolist()
            feature_values = features.iloc[0].values
            contributions = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
            # Create feature importance ranking
            feature_importance = []
            for i, (name, value, contrib) in enumerate(zip(feature_names, feature_values, contributions)):
                feature_importance.append({
                    'feature': name,
                    'value': float(value),
                    'shap_value': float(contrib),
                    'abs_shap_value': abs(float(contrib)),
                    'impact': 'positive' if contrib > 0 else 'negative'
                })
            
            # Sort by absolute SHAP value
            feature_importance.sort(key=lambda x: x['abs_shap_value'], reverse=True)
            
            # Get top contributors
            top_positive = [f for f in feature_importance if f['impact'] == 'positive'][:5]
            top_negative = [f for f in feature_importance if f['impact'] == 'negative'][:5]
            
            # Create patient-friendly explanations
            explanations = self._create_patient_explanations(
                feature_importance[:10], feature_values, feature_names
            )
            
            return {
                'feature_importance': feature_importance[:15],  # Top 15 features
                'top_positive_contributors': top_positive,
                'top_negative_contributors': top_negative,
                'patient_explanations': explanations,
                'base_value': float(self.explainer.expected_value),
                'prediction': float(np.sum(contributions) + self.explainer.expected_value)
            }
            
        except Exception as e:
            logger.error("SHAP calculation failed", error=str(e))
            return {'error': str(e)}
    
    def _create_patient_explanations(self, top_features: List[Dict], 
                                   feature_values: np.ndarray, 
                                   feature_names: List[str]) -> List[str]:
        """Create patient-friendly explanations of risk factors."""
        explanations = []
        
        try:
            for feature_info in top_features[:5]:  # Top 5 features only
                feature_name = feature_info['feature']
                shap_value = feature_info['shap_value']
                feature_value = feature_info['value']
                
                explanation = self._get_feature_explanation(
                    feature_name, feature_value, shap_value
                )
                if explanation:
                    explanations.append(explanation)
            
            return explanations
            
        except Exception as e:
            logger.error("Failed to create patient explanations", error=str(e))
            return []
    
    def _get_feature_explanation(self, feature_name: str, value: float, 
                               shap_value: float) -> Optional[str]:
        """Get human-readable explanation for a feature."""
        impact = "increases" if shap_value > 0 else "decreases"
        
        explanations = {
            'age': f"Age ({value:.0f} years) {impact} cardiovascular risk",
            'hba1c': f"HbA1c level ({value:.1f}%) {impact} diabetes-related risk",
            'sbp': f"Systolic blood pressure ({value:.0f} mmHg) {impact} cardiovascular risk",
            'dbp': f"Diastolic blood pressure ({value:.0f} mmHg) {impact} cardiovascular risk",
            'bmi': f"BMI ({value:.1f}) {impact} metabolic risk",
            'glucose_fasting': f"Fasting glucose ({value:.0f} mg/dL) {impact} diabetes risk",
            'total_cholesterol': f"Total cholesterol ({value:.0f} mg/dL) {impact} cardiovascular risk",
            'ldl_cholesterol': f"LDL cholesterol ({value:.0f} mg/dL) {impact} cardiovascular risk",
            'hdl_cholesterol': f"HDL cholesterol ({value:.0f} mg/dL) {impact} cardiovascular protection",
            'triglycerides': f"Triglycerides ({value:.0f} mg/dL) {impact} metabolic risk",
            'sex': f"Gender {impact} baseline risk profile",
            'smoking_status': f"Smoking status {impact} cardiovascular risk",
            'hypertension_stage': f"Hypertension status {impact} cardiovascular risk",
            'diabetes_controlled': f"Diabetes control {impact} complication risk",
            'metabolic_syndrome_score': f"Metabolic syndrome indicators {impact} overall risk"
        }
        
        return explanations.get(feature_name)
    
    def _categorize_risk(self, probability: float) -> str:
        """Categorize risk probability into clinical categories."""
        if probability < 0.15:
            return 'low'
        elif probability < 0.30:
            return 'moderate'
        else:
            return 'high'
    
    def batch_predict(self, patients_data: List[Dict[str, Any]], 
                     include_shap: bool = False) -> List[Dict[str, Any]]:
        """Predict risk for multiple patients."""
        results = []
        
        for i, patient_data in enumerate(patients_data):
            try:
                result = self.predict_risk(patient_data, include_shap=include_shap)
                result['patient_index'] = i
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict for patient {i}", error=str(e))
                results.append({
                    'patient_index': i,
                    'error': str(e),
                    'risk_probability': 0.0,
                    'risk_category': 'unknown'
                })
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get global feature importance from the model."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_names = self.model_metadata.get('feature_names', [])
                
                if len(feature_names) == len(importance):
                    return dict(zip(feature_names, importance.tolist()))
                else:
                    return {f'feature_{i}': imp for i, imp in enumerate(importance)}
            else:
                return {}
                
        except Exception as e:
            logger.error("Failed to get feature importance", error=str(e))
            return {}
    
    def model_summary(self) -> Dict[str, Any]:
        """Get model summary information."""
        return {
            'metadata': self.model_metadata,
            'is_loaded': self.model is not None,
            'has_explainer': self.explainer is not None,
            'feature_importance': self.get_feature_importance()
        }


def load_default_model() -> RiskPredictor:
    """Load the default risk prediction model."""
    model_path = os.getenv('DEFAULT_MODEL_PATH', 'models/risk_lgbm_v0.1.bin')
    
    if not os.path.exists(model_path):
        logger.warning(f"Default model not found at {model_path}")
        return RiskPredictor()
    
    predictor = RiskPredictor(model_path)
    return predictor


# Global predictor instance (loaded on import)
default_predictor = None

def get_predictor() -> RiskPredictor:
    """Get the global predictor instance."""
    global default_predictor
    if default_predictor is None:
        default_predictor = load_default_model()
    return default_predictor


def predict_patient_risk(patient_data: Dict[str, Any], 
                        include_shap: bool = True) -> Dict[str, Any]:
    """Convenience function for risk prediction."""
    predictor = get_predictor()
    return predictor.predict_risk(patient_data, include_shap=include_shap)


if __name__ == "__main__":
    # CLI interface for testing
    import argparse
    
    parser = argparse.ArgumentParser(description='Risk prediction inference')
    parser.add_argument('--model', type=str, required=True, help='Model file path')
    parser.add_argument('--input', type=str, required=True, help='Patient data JSON file')
    parser.add_argument('--output', type=str, help='Output predictions file')
    parser.add_argument('--include-shap', action='store_true', help='Include SHAP explanations')
    
    args = parser.parse_args()
    
    # Load model
    predictor = RiskPredictor(args.model)
    
    # Load patient data
    with open(args.input, 'r') as f:
        patient_data = json.load(f)
    
    # Make prediction
    if isinstance(patient_data, list):
        results = predictor.batch_predict(patient_data, include_shap=args.include_shap)
    else:
        results = predictor.predict_risk(patient_data, include_shap=args.include_shap)
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
    else:
        print(json.dumps(results, indent=2))
