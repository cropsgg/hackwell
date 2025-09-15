# Keras Model Simulator with GPT API Integration

A Python simulator that mimics loading and running predictions from three Keras LSTM models (arthritis, diabetes, hypertension) while using GPT API calls for actual processing. This provides a realistic simulation of ML model inference without requiring the actual model files.

## Features

- **Model Simulation**: Simulates loading three Keras LSTM models for healthcare predictions
- **GPT Integration**: Uses OpenAI GPT API for realistic medical predictions
- **Multiple Models**: Supports arthritis, diabetes, and hypertension prediction
- **Batch Processing**: Handles single and batch predictions
- **Rich Metadata**: Provides detailed model information and feature descriptions
- **Error Handling**: Robust error handling with fallback responses
- **Configurable**: Easy configuration through environment variables

## Models

### 1. Arthritis Model
- **Type**: LSTM
- **Input**: 10 features (age, BMI, pain score, stiffness, swelling, mobility, fatigue, sleep quality, medication adherence, activity level)
- **Output**: 4 classes (no arthritis, mild, moderate, severe arthritis)

### 2. Diabetes Model  
- **Type**: LSTM
- **Input**: 12 features (age, BMI, glucose, HbA1c, insulin, C-peptide, family history, weight change, exercise, diet, stress, sleep)
- **Output**: 4 classes (no diabetes, prediabetes, type 2 diabetes, type 1 diabetes)

### 3. Hypertension Model
- **Type**: LSTM
- **Input**: 8 features (age, BMI, systolic BP, diastolic BP, heart rate, sodium intake, stress, exercise)
- **Output**: 4 classes (normal, elevated, stage 1 hypertension, stage 2 hypertension)

## Installation

1. Install required packages:
```bash
pip install -r keras_simulator_requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Basic Usage

```python
from keras_model_simulator import KerasModelSimulator

# Initialize simulator
simulator = KerasModelSimulator()

# Load a model
simulator.load_model('diabetes')

# Make a prediction
patient_data = {
    'age': 45,
    'bmi': 28.5,
    'glucose_fasting': 120,
    'hba1c': 6.8
}

result = simulator.predict('diabetes', patient_data)
print(f"Prediction: {result.prediction}")
print(f"Risk Category: {result.risk_category}")
print(f"Explanation: {result.explanation}")
```

### Load All Models

```python
# Load all three models
simulator = KerasModelSimulator()
load_results = simulator.load_all_models()

for model_name, success in load_results.items():
    print(f"{model_name}: {'Loaded' if success else 'Failed'}")
```

### Batch Prediction

```python
# Multiple patients
patients = [
    {'age': 35, 'bmi': 22, 'glucose_fasting': 85},
    {'age': 55, 'bmi': 30, 'glucose_fasting': 140},
    {'age': 28, 'bmi': 24, 'glucose_fasting': 95}
]

results = simulator.batch_predict('diabetes', patients)
for i, result in enumerate(results):
    print(f"Patient {i+1}: {result.prediction:.3f} ({result.risk_category})")
```

### Model Information

```python
# Get model details
info = simulator.get_model_info('arthritis')
print(f"Model Type: {info['model_type']}")
print(f"Input Shape: {info['input_shape']}")
print(f"Features: {info['feature_names']}")
```

## Example Scenarios

### Arthritis Prediction
```python
patient_data = {
    'demographics': {'age': 58, 'sex': 'F', 'bmi': 26.0},
    'symptoms': {
        'pain_score': 7,      # High pain
        'stiffness': 6,       # High stiffness
        'mobility': 3,        # Low mobility
        'fatigue': 8,         # High fatigue
        'sleep_quality': 3    # Poor sleep
    }
}
```

### Diabetes Prediction
```python
patient_data = {
    'demographics': {'age': 45, 'sex': 'M', 'bmi': 32.0},
    'vitals': [
        {'type': 'glucose_fasting', 'value': 125},
        {'type': 'hba1c', 'value': 7.8}
    ],
    'lifestyle': {
        'exercise_frequency': 1,  # Low exercise
        'diet_quality': 3,        # Poor diet
        'stress_level': 8         # High stress
    }
}
```

### Hypertension Prediction
```python
patient_data = {
    'demographics': {'age': 62, 'sex': 'M', 'bmi': 29.0},
    'vitals': [
        {'type': 'sbp', 'value': 155},
        {'type': 'dbp', 'value': 95}
    ],
    'lifestyle': {
        'sodium_intake': 8,       # High sodium
        'exercise_frequency': 2   # Low exercise
    }
}
```

## Running Examples

Run the example script to see all features in action:

```bash
python example_keras_simulator.py
```

This will demonstrate:
- Individual model predictions
- Batch processing
- Model comparison
- Custom patient data
- Error handling

## Configuration

The simulator can be configured through environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `LOG_LEVEL`: Logging level (default: INFO)

## API Response Format

Each prediction returns a `PredictionResult` object with:

```python
@dataclass
class PredictionResult:
    model_name: str           # Name of the model used
    prediction: float         # Prediction score (0-1)
    probability: float        # Probability of prediction
    confidence: float         # Model confidence
    risk_category: str        # Risk category (low/moderate/high)
    explanation: str          # Clinical explanation
    features_used: List[str]  # Features used in prediction
    timestamp: str            # Prediction timestamp
    processing_time: float    # Time taken for prediction
```

## Error Handling

The simulator includes robust error handling:

- **API Errors**: Falls back to default responses if GPT API fails
- **Model Errors**: Graceful handling of missing or invalid models
- **Data Errors**: Default values for missing features
- **Parse Errors**: Safe JSON parsing with fallbacks

## File Structure

```
keras_model_simulator.py          # Main simulator class
example_keras_simulator.py        # Usage examples
keras_simulator_config.py         # Configuration settings
keras_simulator_requirements.txt  # Python dependencies
KERAS_SIMULATOR_README.md         # This documentation
```

## Dependencies

- `requests`: HTTP client for API calls
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `python-dotenv`: Environment variable management

## Notes

- The simulator works without actual Keras model files
- GPT API calls provide realistic medical predictions
- All processing is simulated but maintains realistic timing
- Suitable for demos, testing, and development environments
- Can be easily extended to support additional models

## License

This project is part of the Hackwell healthcare ML system.
