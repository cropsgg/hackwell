"""
Test script for Keras Model Simulator
Tests basic functionality without requiring API calls.
"""

import os
import sys
from unittest.mock import patch, MagicMock
import json

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from keras_model_simulator import KerasModelSimulator, create_sample_patient_data

def test_model_loading():
    """Test model loading functionality."""
    print("🧪 Testing model loading...")
    
    simulator = KerasModelSimulator()
    
    # Test loading individual models
    for model_name in ['arthritis', 'diabetes', 'hypertension']:
        success = simulator.load_model(model_name)
        assert success, f"Failed to load model: {model_name}"
        print(f"✅ {model_name} model loaded successfully")
    
    # Test loading all models
    results = simulator.load_all_models()
    assert all(results.values()), "Failed to load all models"
    print("✅ All models loaded successfully")
    
    return simulator

def test_model_metadata():
    """Test model metadata retrieval."""
    print("\n🧪 Testing model metadata...")
    
    simulator = KerasModelSimulator()
    simulator.load_all_models()
    
    for model_name in ['arthritis', 'diabetes', 'hypertension']:
        info = simulator.get_model_info(model_name)
        assert info is not None, f"No metadata for model: {model_name}"
        assert 'name' in info, f"Missing name in metadata for {model_name}"
        assert 'version' in info, f"Missing version in metadata for {model_name}"
        assert 'feature_names' in info, f"Missing features in metadata for {model_name}"
        print(f"✅ {model_name} metadata: {info['model_type']} v{info['version']}")

def test_feature_extraction():
    """Test feature extraction from patient data."""
    print("\n🧪 Testing feature extraction...")
    
    simulator = KerasModelSimulator()
    simulator.load_model('diabetes')
    
    patient_data = create_sample_patient_data()
    features = simulator._extract_features(patient_data, simulator.model_metadata['diabetes'].feature_names)
    
    assert len(features) > 0, "No features extracted"
    assert 'age' in features, "Age feature missing"
    assert 'bmi' in features, "BMI feature missing"
    assert 'glucose_fasting' in features, "Glucose feature missing"
    print(f"✅ Extracted {len(features)} features")

def test_prediction_without_api():
    """Test prediction functionality without API calls."""
    print("\n🧪 Testing prediction (mock API)...")
    
    simulator = KerasModelSimulator()
    simulator.load_model('arthritis')
    
    # Mock GPT API response
    mock_response = json.dumps({
        "prediction": 0.75,
        "probability": 0.75,
        "confidence": 0.85,
        "risk_category": "moderate",
        "explanation": "Based on the input features, the model predicts moderate arthritis risk due to high pain scores and reduced mobility.",
        "key_factors": ["pain_score", "mobility", "stiffness"]
    })
    
    with patch.object(simulator, '_call_gpt_api', return_value=mock_response):
        patient_data = {
            'age': 55,
            'bmi': 28.0,
            'pain_score': 6,
            'stiffness': 5,
            'mobility': 4
        }
        
        result = simulator.predict('arthritis', patient_data)
        
        assert result.model_name == 'arthritis', "Wrong model name"
        assert 0 <= result.prediction <= 1, "Invalid prediction score"
        assert result.risk_category in ['low', 'moderate', 'high'], "Invalid risk category"
        assert len(result.explanation) > 0, "No explanation provided"
        print(f"✅ Prediction: {result.prediction:.3f} ({result.risk_category})")

def test_batch_prediction():
    """Test batch prediction functionality."""
    print("\n🧪 Testing batch prediction...")
    
    simulator = KerasModelSimulator()
    simulator.load_model('diabetes')
    
    # Mock GPT API response
    mock_response = json.dumps({
        "prediction": 0.6,
        "probability": 0.6,
        "confidence": 0.8,
        "risk_category": "moderate",
        "explanation": "Moderate diabetes risk based on metabolic markers.",
        "key_factors": ["glucose_fasting", "hba1c"]
    })
    
    with patch.object(simulator, '_call_gpt_api', return_value=mock_response):
        patients = [
            {'age': 35, 'bmi': 22, 'glucose_fasting': 85},
            {'age': 55, 'bmi': 30, 'glucose_fasting': 140},
            {'age': 28, 'bmi': 24, 'glucose_fasting': 95}
        ]
        
        results = simulator.batch_predict('diabetes', patients)
        
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"
        for i, result in enumerate(results):
            assert result.model_name == 'diabetes', f"Wrong model name for patient {i}"
            assert 0 <= result.prediction <= 1, f"Invalid prediction for patient {i}"
        print(f"✅ Batch prediction: {len(results)} patients processed")

def test_error_handling():
    """Test error handling."""
    print("\n🧪 Testing error handling...")
    
    simulator = KerasModelSimulator()
    
    # Test prediction with unloaded model
    result = simulator.predict('nonexistent_model', {})
    assert result.model_name == 'nonexistent_model', "Wrong model name in error case"
    assert result.prediction == 0.0, "Non-zero prediction in error case"
    print("✅ Error handling for unloaded model works")
    
    # Test with invalid input data
    simulator.load_model('diabetes')
    result = simulator.predict('diabetes', None)
    assert result.prediction == 0.0, "Non-zero prediction with invalid input"
    print("✅ Error handling for invalid input works")

def test_sample_patient_data():
    """Test sample patient data creation."""
    print("\n🧪 Testing sample patient data...")
    
    patient_data = create_sample_patient_data()
    
    assert 'demographics' in patient_data, "Missing demographics"
    assert 'vitals' in patient_data, "Missing vitals"
    assert 'conditions' in patient_data, "Missing conditions"
    assert 'medications' in patient_data, "Missing medications"
    
    demographics = patient_data['demographics']
    assert 'age' in demographics, "Missing age"
    assert 'sex' in demographics, "Missing sex"
    assert 'bmi' in demographics, "Missing BMI"
    
    print(f"✅ Sample patient data: {demographics['age']}yo {demographics['sex']}, BMI {demographics['bmi']}")

def run_all_tests():
    """Run all tests."""
    print("🧪 KERAS MODEL SIMULATOR TESTS")
    print("=" * 40)
    
    try:
        # Test model loading
        simulator = test_model_loading()
        
        # Test metadata
        test_model_metadata()
        
        # Test feature extraction
        test_feature_extraction()
        
        # Test predictions (with mocked API)
        test_prediction_without_api()
        test_batch_prediction()
        
        # Test error handling
        test_error_handling()
        
        # Test sample data
        test_sample_patient_data()
        
        print("\n✅ ALL TESTS PASSED!")
        print("The Keras Model Simulator is working correctly.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
