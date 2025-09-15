"""Simplified inference script for ML model."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import structlog

logger = structlog.get_logger()

class SimpleRiskPredictor:
    """Simple risk prediction model wrapper."""
    
    def __init__(self, model_path: str = "models/cardiometabolic_risk_model.pkl"):
        self.model_path = Path(model_path)
        self.feature_names_path = Path("models/feature_names.pkl")
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model and feature names."""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            if self.feature_names_path.exists():
                self.feature_names = joblib.load(self.feature_names_path)
                logger.info(f"Feature names loaded: {len(self.feature_names)} features")
            else:
                logger.error(f"Feature names file not found: {self.feature_names_path}")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def predict_risk(self, patient_data: dict) -> dict:
        """Predict risk for a single patient."""
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not loaded")
        
        try:
            # Convert patient data to DataFrame
            df = pd.DataFrame([patient_data])
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in df.columns:
                    df[feature] = 0  # Default value for missing features
            
            # Select and order features
            X = df[self.feature_names]
            
            # Get prediction and probability
            risk_proba = self.model.predict_proba(X)[0, 1]
            risk_binary = self.model.predict(X)[0]
            
            # Get feature importance for this prediction (simplified SHAP alternative)
            feature_importances = self.model.feature_importances_
            feature_contributions = {}
            
            for i, feature in enumerate(self.feature_names):
                contribution = X.iloc[0, i] * feature_importances[i]
                feature_contributions[feature] = float(contribution)
            
            # Sort by absolute contribution
            sorted_contributions = sorted(
                feature_contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            result = {
                "risk_probability": float(risk_proba),
                "risk_binary": int(risk_binary),
                "risk_level": "high" if risk_proba > 0.5 else "low",
                "confidence": float(max(risk_proba, 1 - risk_proba)),
                "feature_contributions": dict(sorted_contributions[:5]),  # Top 5 features
                "input_features": patient_data
            }
            
            logger.info(f"Risk prediction: {risk_proba:.3f} ({result['risk_level']})")
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

def test_inference():
    """Test the inference with sample data."""
    predictor = SimpleRiskPredictor()
    
    # Test cases
    test_patients = [
        {
            "age": 45,
            "bmi": 25,
            "systolic_bp": 120,
            "diastolic_bp": 80,
            "hba1c": 5.5,
            "glucose_fasting": 90,
            "total_cholesterol": 180,
            "hdl_cholesterol": 60,
            "ldl_cholesterol": 100,
            "triglycerides": 120,
            "smoking": 0,
            "family_history": 0
        },
        {
            "age": 65,
            "bmi": 32,
            "systolic_bp": 160,
            "diastolic_bp": 95,
            "hba1c": 8.0,
            "glucose_fasting": 180,
            "total_cholesterol": 250,
            "hdl_cholesterol": 35,
            "ldl_cholesterol": 160,
            "triglycerides": 200,
            "smoking": 1,
            "family_history": 1
        }
    ]
    
    for i, patient in enumerate(test_patients, 1):
        print(f"\n--- Patient {i} ---")
        result = predictor.predict_risk(patient)
        print(f"Risk Probability: {result['risk_probability']:.3f}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("Top Contributing Features:")
        for feature, contribution in list(result['feature_contributions'].items())[:3]:
            print(f"  {feature}: {contribution:.3f}")

if __name__ == "__main__":
    test_inference()
