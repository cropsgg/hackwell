"""
Example usage of the Keras Model Simulator
Demonstrates how to use the simulator for different healthcare scenarios.
"""

import os
import json
from keras_model_simulator import KerasModelSimulator, create_sample_patient_data

def example_arthritis_prediction():
    """Example: Arthritis severity prediction."""
    print("🦴 ARTHRITIS PREDICTION EXAMPLE")
    print("-" * 40)
    
    simulator = KerasModelSimulator()
    simulator.load_model('arthritis')
    
    # Patient with arthritis symptoms
    patient_data = {
        'demographics': {'age': 58, 'sex': 'F', 'bmi': 26.0},
        'symptoms': {
            'pain_score': 7,  # High pain
            'stiffness': 6,   # High stiffness
            'swelling': 4,    # Moderate swelling
            'mobility': 3,    # Low mobility
            'fatigue': 8,     # High fatigue
            'sleep_quality': 3,  # Poor sleep
            'medication_adherence': 6,  # Moderate adherence
            'activity_level': 2  # Low activity
        }
    }
    
    result = simulator.predict('arthritis', patient_data)
    
    print(f"Patient: 58-year-old female with high pain and stiffness")
    print(f"Prediction: {result.prediction:.3f}")
    print(f"Risk Category: {result.risk_category}")
    print(f"Explanation: {result.explanation}")
    print()

def example_diabetes_prediction():
    """Example: Diabetes type prediction."""
    print("🍯 DIABETES PREDICTION EXAMPLE")
    print("-" * 40)
    
    simulator = KerasModelSimulator()
    simulator.load_model('diabetes')
    
    # Patient with diabetes risk factors
    patient_data = {
        'demographics': {'age': 45, 'sex': 'M', 'bmi': 32.0},
        'vitals': [
            {'type': 'glucose_fasting', 'value': 125, 'unit': 'mg/dL'},
            {'type': 'hba1c', 'value': 7.8, 'unit': '%'},
            {'type': 'insulin_level', 'value': 15, 'unit': 'mU/L'}
        ],
        'lifestyle': {
            'exercise_frequency': 1,  # Low exercise
            'diet_quality': 3,        # Poor diet
            'stress_level': 8,        # High stress
            'sleep_duration': 5,      # Poor sleep
            'weight_change': 15       # Weight gain
        },
        'family_history': 1  # Family history of diabetes
    }
    
    result = simulator.predict('diabetes', patient_data)
    
    print(f"Patient: 45-year-old male with elevated glucose and HbA1c")
    print(f"Prediction: {result.prediction:.3f}")
    print(f"Risk Category: {result.risk_category}")
    print(f"Explanation: {result.explanation}")
    print()

def example_hypertension_prediction():
    """Example: Hypertension stage prediction."""
    print("🩺 HYPERTENSION PREDICTION EXAMPLE")
    print("-" * 40)
    
    simulator = KerasModelSimulator()
    simulator.load_model('hypertension')
    
    # Patient with hypertension
    patient_data = {
        'demographics': {'age': 62, 'sex': 'M', 'bmi': 29.0},
        'vitals': [
            {'type': 'sbp', 'value': 155, 'unit': 'mmHg'},
            {'type': 'dbp', 'value': 95, 'unit': 'mmHg'},
            {'type': 'heart_rate', 'value': 88, 'unit': 'bpm'}
        ],
        'lifestyle': {
            'sodium_intake': 8,       # High sodium
            'stress_level': 7,        # High stress
            'exercise_frequency': 2   # Low exercise
        }
    }
    
    result = simulator.predict('hypertension', patient_data)
    
    print(f"Patient: 62-year-old male with elevated blood pressure")
    print(f"Prediction: {result.prediction:.3f}")
    print(f"Risk Category: {result.risk_category}")
    print(f"Explanation: {result.explanation}")
    print()

def example_batch_prediction():
    """Example: Batch prediction for multiple patients."""
    print("👥 BATCH PREDICTION EXAMPLE")
    print("-" * 40)
    
    simulator = KerasModelSimulator()
    simulator.load_model('diabetes')
    
    # Multiple patients
    patients = [
        {
            'demographics': {'age': 35, 'sex': 'F', 'bmi': 22.0},
            'vitals': [{'type': 'glucose_fasting', 'value': 85, 'unit': 'mg/dL'}],
            'lifestyle': {'exercise_frequency': 5, 'diet_quality': 8}
        },
        {
            'demographics': {'age': 55, 'sex': 'M', 'bmi': 30.0},
            'vitals': [{'type': 'glucose_fasting', 'value': 140, 'unit': 'mg/dL'}],
            'lifestyle': {'exercise_frequency': 1, 'diet_quality': 3}
        },
        {
            'demographics': {'age': 28, 'sex': 'F', 'bmi': 24.0},
            'vitals': [{'type': 'glucose_fasting', 'value': 95, 'unit': 'mg/dL'}],
            'lifestyle': {'exercise_frequency': 4, 'diet_quality': 7}
        }
    ]
    
    results = simulator.batch_predict('diabetes', patients)
    
    print(f"Processed {len(results)} patients:")
    for i, result in enumerate(results):
        print(f"  Patient {i+1}: Prediction={result.prediction:.3f}, Risk={result.risk_category}")
    print()

def example_model_comparison():
    """Example: Compare predictions across all models."""
    print("🔄 MODEL COMPARISON EXAMPLE")
    print("-" * 40)
    
    simulator = KerasModelSimulator()
    simulator.load_all_models()
    
    # Single patient with multiple conditions
    patient_data = create_sample_patient_data()
    
    print(f"Patient: 65-year-old female with multiple conditions")
    print(f"Conditions: Diabetes, Hypertension, Arthritis")
    print()
    
    for model_name in ['arthritis', 'diabetes', 'hypertension']:
        if model_name in simulator.models:
            result = simulator.predict(model_name, patient_data)
            print(f"{model_name.upper()}:")
            print(f"  Prediction: {result.prediction:.3f}")
            print(f"  Risk: {result.risk_category}")
            print(f"  Confidence: {result.confidence:.3f}")
            print()

def example_custom_patient():
    """Example: Custom patient data input."""
    print("👤 CUSTOM PATIENT EXAMPLE")
    print("-" * 40)
    
    simulator = KerasModelSimulator()
    simulator.load_model('arthritis')
    
    # Custom patient input
    custom_patient = {
        'age': 42,
        'bmi': 24.5,
        'pain_score': 4,
        'stiffness': 3,
        'swelling': 1,
        'mobility': 8,
        'fatigue': 3,
        'sleep_quality': 7,
        'medication_adherence': 9,
        'activity_level': 6
    }
    
    result = simulator.predict('arthritis', custom_patient)
    
    print(f"Custom Patient: 42-year-old with moderate symptoms")
    print(f"Prediction: {result.prediction:.3f}")
    print(f"Risk Category: {result.risk_category}")
    print(f"Explanation: {result.explanation}")
    print()

def main():
    """Run all examples."""
    print("🏥 KERAS MODEL SIMULATOR EXAMPLES")
    print("=" * 50)
    print()
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️  Warning: OPENAI_API_KEY not set. Using fallback responses.")
        print("   Set your API key: export OPENAI_API_KEY='your-key-here'")
        print()
    
    try:
        example_arthritis_prediction()
        example_diabetes_prediction()
        example_hypertension_prediction()
        example_batch_prediction()
        example_model_comparison()
        example_custom_patient()
        
        print("✅ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()
