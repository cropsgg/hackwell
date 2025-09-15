"""Tests for ML model training pipeline."""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, Mock
import tempfile
import os

from train import RiskModelTrainer, generate_synthetic_data
from featurize import FeatureProcessor, extract_patient_features
from metrics import ModelMetrics


class TestRiskModelTrainer:
    """Test the risk model training pipeline."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        data = pd.DataFrame({
            'age': np.random.normal(55, 15, n_samples).clip(18, 90),
            'sex': np.random.choice([0, 1], n_samples),
            'bmi': np.random.normal(27, 5, n_samples).clip(15, 50),
            'hba1c': np.random.normal(6.5, 1.5, n_samples).clip(4, 15),
            'glucose_fasting': np.random.normal(110, 30, n_samples).clip(70, 400),
            'sbp': np.random.normal(130, 20, n_samples).clip(90, 200),
            'dbp': np.random.normal(80, 15, n_samples).clip(50, 120),
            'total_cholesterol': np.random.normal(200, 40, n_samples).clip(120, 350),
            'hdl_cholesterol': np.random.normal(50, 15, n_samples).clip(20, 100),
            'ldl_cholesterol': np.random.normal(120, 30, n_samples).clip(50, 250),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        })
        
        return data
    
    def test_trainer_initialization(self):
        """Test trainer initialization with different algorithms."""
        # Test LightGBM
        trainer_lgb = RiskModelTrainer(algorithm='lightgbm')
        assert trainer_lgb.algorithm == 'lightgbm'
        assert trainer_lgb.model is None
        
        # Test XGBoost
        trainer_xgb = RiskModelTrainer(algorithm='xgboost')
        assert trainer_xgb.algorithm == 'xgboost'
        
        # Test invalid algorithm
        with pytest.raises(ValueError):
            trainer = RiskModelTrainer(algorithm='invalid')
            trainer.create_model()
    
    def test_data_loading(self, sample_training_data):
        """Test data loading functionality."""
        trainer = RiskModelTrainer()
        
        # Save sample data to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_training_data.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test loading
            data, target_col = trainer.load_data(temp_file)
            
            assert isinstance(data, pd.DataFrame)
            assert target_col == 'target'
            assert len(data) == len(sample_training_data)
            
        finally:
            os.unlink(temp_file)
    
    def test_feature_preparation(self, sample_training_data):
        """Test feature preparation pipeline."""
        trainer = RiskModelTrainer()
        
        X, y = trainer.prepare_features(sample_training_data, 'target')
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(sample_training_data)
        assert trainer.feature_processor is not None
    
    def test_model_creation(self):
        """Test model creation for different algorithms."""
        # Test LightGBM
        trainer_lgb = RiskModelTrainer(algorithm='lightgbm')
        model_lgb = trainer_lgb.create_model()
        assert hasattr(model_lgb, 'fit')
        assert hasattr(model_lgb, 'predict')
        
        # Test XGBoost
        trainer_xgb = RiskModelTrainer(algorithm='xgboost')
        model_xgb = trainer_xgb.create_model()
        assert hasattr(model_xgb, 'fit')
        assert hasattr(model_xgb, 'predict')
    
    def test_model_training(self, sample_training_data):
        """Test complete model training pipeline."""
        trainer = RiskModelTrainer(algorithm='lightgbm')
        
        X, y = trainer.prepare_features(sample_training_data, 'target')
        metrics = trainer.train_model(X, y, validation_split=0.3)
        
        assert trainer.model is not None
        assert 'roc_auc' in metrics
        assert 'average_precision' in metrics
        assert 0 <= metrics['roc_auc'] <= 1
        assert 0 <= metrics['average_precision'] <= 1
    
    def test_cross_validation(self, sample_training_data):
        """Test cross-validation functionality."""
        trainer = RiskModelTrainer(algorithm='lightgbm')
        
        X, y = trainer.prepare_features(sample_training_data, 'target')
        cv_results = trainer.cross_validate(X, y, cv_folds=3)
        
        assert 'roc_auc_mean' in cv_results
        assert 'roc_auc_std' in cv_results
        assert 'average_precision_mean' in cv_results
        assert cv_results['roc_auc_mean'] >= 0
        assert cv_results['roc_auc_std'] >= 0
    
    def test_model_saving(self, sample_training_data):
        """Test model saving functionality."""
        trainer = RiskModelTrainer(algorithm='lightgbm')
        
        X, y = trainer.prepare_features(sample_training_data, 'target')
        trainer.train_model(X, y)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, 'test_model.bin')
            metadata = trainer.save_model(model_path, 'test_v1.0')
            
            assert os.path.exists(os.path.join(temp_dir, 'test_v1.0.bin'))
            assert metadata['model_version'] == 'test_v1.0'
            assert metadata['algorithm'] == 'lightgbm'


class TestFeatureProcessor:
    """Test feature processing and engineering."""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data."""
        return pd.DataFrame({
            'age': [45, 55, 65],
            'sex': [0, 1, 0],  # 0=Female, 1=Male
            'bmi': [25.0, 30.5, 22.1],
            'hba1c': [5.5, 8.2, 6.1],
            'sbp': [120, 145, 110],
            'dbp': [80, 95, 70],
            'total_cholesterol': [180, 220, 160],
            'hdl_cholesterol': [50, 35, 65],
            'ldl_cholesterol': [100, 150, 80]
        })
    
    def test_feature_processor_initialization(self):
        """Test feature processor initialization."""
        processor = FeatureProcessor()
        assert not processor.is_fitted
        assert len(processor.scalers) == 0
        assert len(processor.imputers) == 0
    
    def test_feature_processor_fitting(self, sample_raw_data):
        """Test feature processor fitting."""
        processor = FeatureProcessor()
        processor.fit(sample_raw_data)
        
        assert processor.is_fitted
        assert len(processor.scalers) > 0
        assert len(processor.feature_names) > 0
    
    def test_feature_transformation(self, sample_raw_data):
        """Test feature transformation."""
        processor = FeatureProcessor()
        processor.fit(sample_raw_data)
        
        transformed = processor.transform(sample_raw_data)
        
        assert isinstance(transformed, pd.DataFrame)
        assert len(transformed) == len(sample_raw_data)
        assert len(transformed.columns) >= len(sample_raw_data.columns)  # Engineered features added
    
    def test_feature_engineering(self, sample_raw_data):
        """Test engineered features creation."""
        processor = FeatureProcessor()
        processor.fit(sample_raw_data)
        transformed = processor.transform(sample_raw_data)
        
        # Check for engineered features
        assert 'bmi_category' in transformed.columns
        assert 'hypertension_stage' in transformed.columns
        assert 'chol_hdl_ratio' in transformed.columns
    
    def test_patient_feature_extraction(self):
        """Test patient feature extraction."""
        patient_data = {
            'demographics': {
                'age': 55,
                'sex': 'F',
                'bmi': 28.5,
                'ethnicity': 'Hispanic'
            },
            'vitals': [
                {'type': 'glucose_fasting', 'value': 120, 'unit': 'mg/dL'},
                {'type': 'hba1c', 'value': 7.2, 'unit': '%'},
                {'type': 'sbp', 'value': 140, 'unit': 'mmHg'}
            ],
            'conditions': [
                {'name': 'Type 2 Diabetes Mellitus', 'active': True}
            ],
            'medications': [
                {'name': 'Metformin', 'active': True}
            ]
        }
        
        features = extract_patient_features(patient_data)
        
        assert isinstance(features, dict)
        assert features['age'] == 55.0
        assert features['sex'] == 0.0  # Female
        assert features['glucose_fasting'] == 120.0
        assert features['hba1c'] == 7.2
        assert features['has_diabetes'] == 1.0
        assert features['on_metformin'] == 1.0


class TestModelMetrics:
    """Test model evaluation metrics."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n_samples = 100
        
        # Create realistic predictions
        y_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        y_pred_proba = np.random.beta(2, 5, n_samples)  # Skewed toward low probabilities
        y_pred = (y_pred_proba > 0.3).astype(int)
        
        return y_true, y_pred, y_pred_proba
    
    def test_metrics_calculation(self, sample_predictions):
        """Test comprehensive metrics calculation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        metrics_calc = ModelMetrics()
        metrics = metrics_calc.calculate_all_metrics(y_true, y_pred, y_pred_proba)
        
        # Check required metrics
        required_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'roc_auc', 'average_precision', 'brier_score',
            'sensitivity', 'specificity'
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_net_benefit_calculation(self, sample_predictions):
        """Test net benefit calculation for clinical decision making."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        metrics_calc = ModelMetrics()
        net_benefit = metrics_calc.calculate_net_benefit(y_true, y_pred_proba, threshold=0.2)
        
        assert isinstance(net_benefit, float)
    
    def test_classification_report(self, sample_predictions):
        """Test classification report generation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        metrics_calc = ModelMetrics()
        report = metrics_calc.generate_classification_report(y_true, y_pred)
        
        assert 'confusion_matrix' in report
        assert 'true_positive' in report['confusion_matrix']
    
    def test_fairness_metrics(self, sample_predictions):
        """Test fairness metrics calculation."""
        y_true, y_pred, y_pred_proba = sample_predictions
        
        # Create synthetic sensitive attributes
        age_groups = np.random.choice(['young', 'old'], len(y_true))
        sex_groups = np.random.choice(['M', 'F'], len(y_true))
        
        sensitive_attributes = {
            'age_group': age_groups,
            'sex': sex_groups
        }
        
        metrics_calc = ModelMetrics()
        fairness_metrics = metrics_calc.calculate_fairness_metrics(
            y_true, y_pred, y_pred_proba, sensitive_attributes
        )
        
        assert 'age_group' in fairness_metrics
        assert 'sex' in fairness_metrics


class TestSyntheticDataGeneration:
    """Test synthetic data generation for training."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation."""
        data = generate_synthetic_data(n_samples=100)
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 100
        assert 'target' in data.columns
        
        # Check realistic distributions
        assert data['age'].min() >= 18
        assert data['age'].max() <= 90
        assert data['bmi'].min() >= 15
        assert data['target'].isin([0, 1]).all()
    
    def test_synthetic_data_correlations(self):
        """Test that synthetic data has realistic correlations."""
        data = generate_synthetic_data(n_samples=1000)
        
        # Check correlations
        assert data['hba1c'].corr(data['glucose_fasting']) > 0.5  # HbA1c and glucose should correlate
        assert data['age'].corr(data['target']) > 0  # Age should increase risk
        assert data['bmi'].corr(data['target']) > 0  # BMI should increase risk


if __name__ == "__main__":
    pytest.main([__file__])
