"""
Keras Model Simulator with GPT API Integration
Simulates loading and prediction from three Keras models (arthritis, diabetes, hypertension)
using GPT API calls for actual processing.
"""

import json
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
MODEL_NAMES = ['arthritis', 'diabetes', 'hypertension']
MODEL_PATHS = {
    'arthritis': 'lstm_model_arthritis.keras',
    'diabetes': 'lstm_model_diabetes.keras', 
    'hypertension': 'lstm_model_hypertension.keras'
}

@dataclass
class ModelMetadata:
    """Model metadata structure."""
    name: str
    version: str
    input_shape: Tuple[int, ...]
    output_classes: List[str]
    feature_names: List[str]
    model_type: str
    created_at: str
    description: str

@dataclass
class PredictionResult:
    """Prediction result structure."""
    model_name: str
    prediction: float
    probability: float
    confidence: float
    risk_category: str
    explanation: str
    features_used: List[str]
    timestamp: str
    processing_time: float

class KerasModelSimulator:
    """Simulates Keras model loading and prediction using GPT API."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or OPENAI_API_KEY
        self.models = {}
        self.model_metadata = {}
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        })
        
    def load_model(self, model_name: str) -> bool:
        """Simulate loading a Keras model."""
        try:
            if model_name not in MODEL_NAMES:
                raise ValueError(f"Unknown model: {model_name}")
            
            model_path = MODEL_PATHS[model_name]
            
            # Simulate model loading (check if file exists)
            if not os.path.exists(model_path):
                print(f"Warning: Model file {model_path} not found, using simulation mode")
            
            # Create simulated model metadata
            metadata = self._create_model_metadata(model_name)
            self.model_metadata[model_name] = metadata
            self.models[model_name] = {
                'loaded': True,
                'metadata': metadata,
                'file_path': model_path
            }
            
            print(f"✅ Model '{model_name}' loaded successfully (simulated)")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model '{model_name}': {e}")
            return False
    
    def load_all_models(self) -> Dict[str, bool]:
        """Load all three models."""
        results = {}
        for model_name in MODEL_NAMES:
            results[model_name] = self.load_model(model_name)
        return results
    
    def _create_model_metadata(self, model_name: str) -> ModelMetadata:
        """Create simulated model metadata."""
        metadata_configs = {
            'arthritis': {
                'version': '1.0.0',
                'input_shape': (10, 1),  # 10 time steps, 1 feature
                'output_classes': ['no_arthritis', 'mild_arthritis', 'moderate_arthritis', 'severe_arthritis'],
                'feature_names': ['age', 'bmi', 'pain_score', 'stiffness', 'swelling', 'mobility', 'fatigue', 'sleep_quality', 'medication_adherence', 'activity_level'],
                'model_type': 'LSTM',
                'description': 'LSTM model for arthritis severity prediction based on patient symptoms and demographics'
            },
            'diabetes': {
                'version': '1.0.0', 
                'input_shape': (12, 1),
                'output_classes': ['no_diabetes', 'prediabetes', 'type2_diabetes', 'type1_diabetes'],
                'feature_names': ['age', 'bmi', 'glucose_fasting', 'hba1c', 'insulin_level', 'c_peptide', 'family_history', 'weight_change', 'exercise_frequency', 'diet_quality', 'stress_level', 'sleep_duration'],
                'model_type': 'LSTM',
                'description': 'LSTM model for diabetes type and severity prediction based on metabolic markers and lifestyle factors'
            },
            'hypertension': {
                'version': '1.0.0',
                'input_shape': (8, 1), 
                'output_classes': ['normal', 'elevated', 'stage1_hypertension', 'stage2_hypertension'],
                'feature_names': ['age', 'bmi', 'sbp', 'dbp', 'heart_rate', 'sodium_intake', 'stress_level', 'exercise_frequency'],
                'model_type': 'LSTM',
                'description': 'LSTM model for hypertension stage prediction based on blood pressure and cardiovascular risk factors'
            }
        }
        
        config = metadata_configs[model_name]
        return ModelMetadata(
            name=model_name,
            version=config['version'],
            input_shape=config['input_shape'],
            output_classes=config['output_classes'],
            feature_names=config['feature_names'],
            model_type=config['model_type'],
            created_at=datetime.now().isoformat(),
            description=config['description']
        )
    
    def predict(self, model_name: str, input_data: Dict[str, Any]) -> PredictionResult:
        """Simulate model prediction using GPT API."""
        start_time = time.time()
        
        try:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not loaded")
            
            metadata = self.model_metadata[model_name]
            
            # Prepare input for GPT API
            prompt = self._create_prediction_prompt(model_name, input_data, metadata)
            
            # Call GPT API
            gpt_response = self._call_gpt_api(prompt)
            
            # Parse GPT response
            prediction_data = self._parse_gpt_response(gpt_response, model_name)
            
            processing_time = time.time() - start_time
            
            return PredictionResult(
                model_name=model_name,
                prediction=prediction_data['prediction'],
                probability=prediction_data['probability'],
                confidence=prediction_data['confidence'],
                risk_category=prediction_data['risk_category'],
                explanation=prediction_data['explanation'],
                features_used=metadata.feature_names,
                timestamp=datetime.now().isoformat(),
                processing_time=processing_time
            )
            
        except Exception as e:
            print(f"❌ Prediction failed for model '{model_name}': {e}")
            return PredictionResult(
                model_name=model_name,
                prediction=0.0,
                probability=0.0,
                confidence=0.0,
                risk_category='unknown',
                explanation=f"Error: {str(e)}",
                features_used=[],
                timestamp=datetime.now().isoformat(),
                processing_time=time.time() - start_time
            )
    
    def _create_prediction_prompt(self, model_name: str, input_data: Dict[str, Any], metadata: ModelMetadata) -> str:
        """Create a detailed prompt for GPT API."""
        
        # Extract relevant features from input data
        features = self._extract_features(input_data, metadata.feature_names)
        
        prompt = f"""
You are a medical AI model simulating a {metadata.model_type} neural network for {model_name} prediction.

MODEL INFORMATION:
- Model Type: {metadata.model_type}
- Input Shape: {metadata.input_shape}
- Output Classes: {metadata.output_classes}
- Description: {metadata.description}

PATIENT INPUT DATA:
{json.dumps(features, indent=2)}

TASK:
Based on the input features, predict the {model_name} classification and provide:
1. A prediction score (0-1) representing the model's confidence
2. A probability distribution across the output classes
3. A risk category (low, moderate, high)
4. A clinical explanation of the prediction
5. Key contributing factors

OUTPUT FORMAT (JSON):
{{
    "prediction": 0.75,
    "probability": 0.75,
    "confidence": 0.85,
    "risk_category": "moderate",
    "explanation": "Based on the input features, the model predicts moderate risk due to...",
    "key_factors": ["factor1", "factor2", "factor3"]
}}

Please provide a realistic medical prediction based on the input data.
"""
        return prompt
    
    def _extract_features(self, input_data: Dict[str, Any], feature_names: List[str]) -> Dict[str, float]:
        """Extract and normalize features from input data."""
        features = {}
        
        # Default values for missing features
        default_values = {
            'age': 50.0, 'bmi': 25.0, 'pain_score': 3.0, 'stiffness': 2.0, 'swelling': 1.0,
            'mobility': 7.0, 'fatigue': 4.0, 'sleep_quality': 6.0, 'medication_adherence': 8.0,
            'activity_level': 5.0, 'glucose_fasting': 90.0, 'hba1c': 5.5, 'insulin_level': 10.0,
            'c_peptide': 2.0, 'family_history': 0.0, 'weight_change': 0.0, 'exercise_frequency': 3.0,
            'diet_quality': 6.0, 'stress_level': 4.0, 'sleep_duration': 7.0, 'sbp': 120.0,
            'dbp': 80.0, 'heart_rate': 70.0, 'sodium_intake': 5.0
        }
        
        # Extract from input data
        for feature in feature_names:
            if feature in input_data:
                features[feature] = float(input_data[feature])
            else:
                features[feature] = default_values.get(feature, 0.0)
        
        return features
    
    def _call_gpt_api(self, prompt: str) -> str:
        """Call OpenAI GPT API."""
        try:
            payload = {
                "model": "gpt-4",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant specializing in disease prediction and risk assessment. Provide accurate, clinical predictions based on patient data."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = self.session.post(OPENAI_API_URL, json=payload)
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"❌ GPT API call failed: {e}")
            # Return fallback response
            return json.dumps({
                "prediction": 0.5,
                "probability": 0.5,
                "confidence": 0.7,
                "risk_category": "moderate",
                "explanation": "Unable to process prediction due to API error",
                "key_factors": ["api_error"]
            })
    
    def _parse_gpt_response(self, response: str, model_name: str) -> Dict[str, Any]:
        """Parse GPT response and extract prediction data."""
        try:
            # Try to extract JSON from response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                # Fallback if no JSON found
                data = {
                    "prediction": 0.5,
                    "probability": 0.5,
                    "confidence": 0.7,
                    "risk_category": "moderate",
                    "explanation": response,
                    "key_factors": []
                }
            
            return {
                'prediction': float(data.get('prediction', 0.5)),
                'probability': float(data.get('probability', 0.5)),
                'confidence': float(data.get('confidence', 0.7)),
                'risk_category': data.get('risk_category', 'moderate'),
                'explanation': data.get('explanation', 'No explanation provided'),
                'key_factors': data.get('key_factors', [])
            }
            
        except Exception as e:
            print(f"❌ Failed to parse GPT response: {e}")
            return {
                'prediction': 0.5,
                'probability': 0.5,
                'confidence': 0.5,
                'risk_category': 'unknown',
                'explanation': f"Parse error: {str(e)}",
                'key_factors': []
            }
    
    def batch_predict(self, model_name: str, input_data_list: List[Dict[str, Any]]) -> List[PredictionResult]:
        """Simulate batch prediction."""
        results = []
        for i, input_data in enumerate(input_data_list):
            print(f"Processing prediction {i+1}/{len(input_data_list)} for model '{model_name}'")
            result = self.predict(model_name, input_data)
            result.patient_index = i
            results.append(result)
        return results
    
    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        if model_name not in self.models:
            return None
        
        metadata = self.model_metadata[model_name]
        return {
            'name': metadata.name,
            'version': metadata.version,
            'input_shape': metadata.input_shape,
            'output_classes': metadata.output_classes,
            'feature_names': metadata.feature_names,
            'model_type': metadata.model_type,
            'created_at': metadata.created_at,
            'description': metadata.description,
            'loaded': self.models[model_name]['loaded']
        }
    
    def list_models(self) -> List[str]:
        """List all loaded models."""
        return list(self.models.keys())


def create_sample_patient_data() -> Dict[str, Any]:
    """Create sample patient data for testing."""
    return {
        'demographics': {
            'age': 65,
            'sex': 'F',
            'bmi': 28.5,
            'weight_kg': 75.0,
            'height_cm': 165.0
        },
        'vitals': [
            {'type': 'glucose_fasting', 'value': 110, 'unit': 'mg/dL'},
            {'type': 'hba1c', 'value': 6.2, 'unit': '%'},
            {'type': 'sbp', 'value': 145, 'unit': 'mmHg'},
            {'type': 'dbp', 'value': 90, 'unit': 'mmHg'},
            {'type': 'heart_rate', 'value': 85, 'unit': 'bpm'}
        ],
        'conditions': [
            {'name': 'Type 2 Diabetes', 'active': True},
            {'name': 'Hypertension', 'active': True},
            {'name': 'Osteoarthritis', 'active': True}
        ],
        'medications': [
            {'name': 'Metformin', 'active': True},
            {'name': 'Lisinopril', 'active': True},
            {'name': 'Ibuprofen', 'active': True}
        ],
        'symptoms': {
            'pain_score': 6,
            'stiffness': 4,
            'swelling': 2,
            'mobility': 5,
            'fatigue': 7,
            'sleep_quality': 4
        }
    }


def main():
    """Main function demonstrating the simulator."""
    print("🏥 Keras Model Simulator with GPT API Integration")
    print("=" * 60)
    
    # Initialize simulator
    simulator = KerasModelSimulator()
    
    # Load all models
    print("\n📥 Loading models...")
    load_results = simulator.load_all_models()
    
    for model_name, success in load_results.items():
        status = "✅" if success else "❌"
        print(f"{status} {model_name}: {'Loaded' if success else 'Failed'}")
    
    # Create sample patient data
    patient_data = create_sample_patient_data()
    print(f"\n👤 Sample patient data created")
    
    # Test predictions for each model
    print("\n🔮 Running predictions...")
    
    for model_name in MODEL_NAMES:
        if model_name in simulator.models:
            print(f"\n--- {model_name.upper()} PREDICTION ---")
            
            # Make prediction
            result = simulator.predict(model_name, patient_data)
            
            # Display results
            print(f"Prediction Score: {result.prediction:.3f}")
            print(f"Probability: {result.probability:.3f}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"Risk Category: {result.risk_category}")
            print(f"Processing Time: {result.processing_time:.3f}s")
            print(f"Explanation: {result.explanation}")
            
            if hasattr(result, 'key_factors') and result.key_factors:
                print(f"Key Factors: {', '.join(result.key_factors)}")
    
    # Display model information
    print(f"\n📊 Model Information:")
    for model_name in MODEL_NAMES:
        if model_name in simulator.models:
            info = simulator.get_model_info(model_name)
            print(f"\n{model_name.upper()}:")
            print(f"  Type: {info['model_type']}")
            print(f"  Version: {info['version']}")
            print(f"  Input Shape: {info['input_shape']}")
            print(f"  Output Classes: {info['output_classes']}")
            print(f"  Features: {len(info['feature_names'])}")


if __name__ == "__main__":
    main()
