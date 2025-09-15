"""
Configuration file for Keras Model Simulator
"""

import os
from typing import Dict, List, Tuple

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-api-key-here')
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = "gpt-4"
OPENAI_TEMPERATURE = 0.3
OPENAI_MAX_TOKENS = 1000

# Model Configuration
MODEL_NAMES = ['arthritis', 'diabetes', 'hypertension']
MODEL_PATHS = {
    'arthritis': 'lstm_model_arthritis.keras',
    'diabetes': 'lstm_model_diabetes.keras', 
    'hypertension': 'lstm_model_hypertension.keras'
}

# Model Metadata
MODEL_CONFIGS = {
    'arthritis': {
        'version': '1.0.0',
        'input_shape': (10, 1),
        'output_classes': ['no_arthritis', 'mild_arthritis', 'moderate_arthritis', 'severe_arthritis'],
        'feature_names': [
            'age', 'bmi', 'pain_score', 'stiffness', 'swelling', 
            'mobility', 'fatigue', 'sleep_quality', 'medication_adherence', 'activity_level'
        ],
        'model_type': 'LSTM',
        'description': 'LSTM model for arthritis severity prediction based on patient symptoms and demographics'
    },
    'diabetes': {
        'version': '1.0.0', 
        'input_shape': (12, 1),
        'output_classes': ['no_diabetes', 'prediabetes', 'type2_diabetes', 'type1_diabetes'],
        'feature_names': [
            'age', 'bmi', 'glucose_fasting', 'hba1c', 'insulin_level', 'c_peptide',
            'family_history', 'weight_change', 'exercise_frequency', 'diet_quality', 
            'stress_level', 'sleep_duration'
        ],
        'model_type': 'LSTM',
        'description': 'LSTM model for diabetes type and severity prediction based on metabolic markers and lifestyle factors'
    },
    'hypertension': {
        'version': '1.0.0',
        'input_shape': (8, 1), 
        'output_classes': ['normal', 'elevated', 'stage1_hypertension', 'stage2_hypertension'],
        'feature_names': [
            'age', 'bmi', 'sbp', 'dbp', 'heart_rate', 
            'sodium_intake', 'stress_level', 'exercise_frequency'
        ],
        'model_type': 'LSTM',
        'description': 'LSTM model for hypertension stage prediction based on blood pressure and cardiovascular risk factors'
    }
}

# Default Feature Values
DEFAULT_FEATURE_VALUES = {
    'age': 50.0, 'bmi': 25.0, 'pain_score': 3.0, 'stiffness': 2.0, 'swelling': 1.0,
    'mobility': 7.0, 'fatigue': 4.0, 'sleep_quality': 6.0, 'medication_adherence': 8.0,
    'activity_level': 5.0, 'glucose_fasting': 90.0, 'hba1c': 5.5, 'insulin_level': 10.0,
    'c_peptide': 2.0, 'family_history': 0.0, 'weight_change': 0.0, 'exercise_frequency': 3.0,
    'diet_quality': 6.0, 'stress_level': 4.0, 'sleep_duration': 7.0, 'sbp': 120.0,
    'dbp': 80.0, 'heart_rate': 70.0, 'sodium_intake': 5.0
}

# Risk Categories
RISK_CATEGORIES = {
    'low': (0.0, 0.3),
    'moderate': (0.3, 0.7),
    'high': (0.7, 1.0)
}

# Logging Configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance Configuration
MAX_BATCH_SIZE = 100
REQUEST_TIMEOUT = 30
RETRY_ATTEMPTS = 3
RETRY_DELAY = 1.0
