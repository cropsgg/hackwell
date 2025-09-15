"""Training pipeline for cardiometabolic risk prediction models."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, calibration_curve
)
from sklearn.calibration import CalibratedClassifierCV
import structlog

from featurize import FeatureProcessor
from metrics import ModelMetrics

# Configure logging
logger = structlog.get_logger()


class RiskModelTrainer:
    """Train and evaluate cardiometabolic risk prediction models."""
    
    def __init__(self, algorithm: str = 'lightgbm', random_state: int = 42):
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.feature_processor = None
        self.metrics_calculator = ModelMetrics()
        
    def load_data(self, data_path: str) -> Tuple[pd.DataFrame, str]:
        """Load training data from file."""
        try:
            if data_path.endswith('.csv'):
                data = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                data = pd.read_parquet(data_path)
            else:
                raise ValueError(f"Unsupported file format: {data_path}")
            
            logger.info("Data loaded successfully", 
                       shape=data.shape, 
                       file=data_path)
            
            # Determine target column
            target_candidates = ['target', 'label', 'outcome', 'event', 'risk']
            target_col = None
            
            for col in target_candidates:
                if col in data.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError("No target column found. Expected one of: " + str(target_candidates))
            
            return data, target_col
            
        except Exception as e:
            logger.error("Failed to load data", path=data_path, error=str(e))
            raise
    
    def prepare_features(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training."""
        try:
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Initialize and fit feature processor
            self.feature_processor = FeatureProcessor()
            X_processed = self.feature_processor.fit_transform(X)
            
            logger.info("Features prepared", 
                       n_features=X_processed.shape[1],
                       n_samples=X_processed.shape[0],
                       target_distribution=y.value_counts().to_dict())
            
            return X_processed, y
            
        except Exception as e:
            logger.error("Failed to prepare features", error=str(e))
            raise
    
    def create_model(self, hyperparams: Dict[str, Any] = None) -> Any:
        """Create model with specified algorithm."""
        if hyperparams is None:
            hyperparams = {}
        
        if self.algorithm == 'lightgbm':
            default_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.1,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }
            default_params.update(hyperparams)
            return lgb.LGBMClassifier(**default_params)
        
        elif self.algorithm == 'xgboost':
            default_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': self.random_state
            }
            default_params.update(hyperparams)
            return xgb.XGBClassifier(**default_params)
        
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, 
                   validation_split: float = 0.2,
                   hyperparams: Dict[str, Any] = None) -> Dict[str, Any]:
        """Train the model with validation."""
        try:
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, 
                random_state=self.random_state, 
                stratify=y
            )
            
            # Create model
            self.model = self.create_model(hyperparams)
            
            # Train model
            if self.algorithm == 'lightgbm':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    eval_metric='binary_logloss',
                    early_stopping_rounds=50,
                    verbose=False
                )
            elif self.algorithm == 'xgboost':
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            
            # Calculate metrics
            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            y_pred = self.model.predict(X_val)
            
            metrics = self.metrics_calculator.calculate_all_metrics(
                y_val, y_pred, y_pred_proba
            )
            
            logger.info("Model training completed", 
                       algorithm=self.algorithm,
                       train_size=len(X_train),
                       val_size=len(X_val),
                       **metrics)
            
            return metrics
            
        except Exception as e:
            logger.error("Failed to train model", error=str(e))
            raise
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv_folds: int = 5) -> Dict[str, Any]:
        """Perform cross-validation."""
        try:
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                random_state=self.random_state)
            
            cv_scores = {
                'roc_auc': [],
                'average_precision': [],
                'brier_score': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train fold model
                fold_model = self.create_model()
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict
                y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
                
                # Calculate metrics
                cv_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
                cv_scores['average_precision'].append(
                    average_precision_score(y_fold_val, y_pred_proba)
                )
                cv_scores['brier_score'].append(
                    brier_score_loss(y_fold_val, y_pred_proba)
                )
                
                logger.info(f"Fold {fold + 1} completed", 
                           roc_auc=cv_scores['roc_auc'][-1],
                           avg_precision=cv_scores['average_precision'][-1])
            
            # Calculate mean and std
            cv_results = {}
            for metric, scores in cv_scores.items():
                cv_results[f'{metric}_mean'] = np.mean(scores)
                cv_results[f'{metric}_std'] = np.std(scores)
            
            logger.info("Cross-validation completed", **cv_results)
            return cv_results
            
        except Exception as e:
            logger.error("Cross-validation failed", error=str(e))
            raise
    
    def calibrate_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Calibrate model probabilities."""
        try:
            # Use Platt scaling for probability calibration
            calibrated_model = CalibratedClassifierCV(
                self.model, method='sigmoid', cv=3
            )
            calibrated_model.fit(X, y)
            self.model = calibrated_model
            
            logger.info("Model calibration completed")
            
        except Exception as e:
            logger.error("Model calibration failed", error=str(e))
            raise
    
    def save_model(self, output_path: str, model_version: str = None) -> Dict[str, Any]:
        """Save trained model and metadata."""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate version if not provided
            if model_version is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_version = f"risk_{self.algorithm}_v{timestamp}"
            
            # Save model
            model_file = output_path.parent / f"{model_version}.bin"
            joblib.dump({
                'model': self.model,
                'feature_processor': self.feature_processor,
                'algorithm': self.algorithm,
                'version': model_version,
                'created_at': datetime.now().isoformat(),
                'feature_names': self.feature_processor.feature_names if self.feature_processor else None
            }, model_file)
            
            # Create metadata
            metadata = {
                'model_version': model_version,
                'algorithm': self.algorithm,
                'model_file': str(model_file),
                'feature_count': len(self.feature_processor.feature_names) if self.feature_processor else 0,
                'created_at': datetime.now().isoformat(),
                'training_completed': True
            }
            
            # Save metadata
            metadata_file = output_path.parent / f"{model_version}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Model saved successfully", 
                       model_file=str(model_file),
                       version=model_version)
            
            return metadata
            
        except Exception as e:
            logger.error("Failed to save model", error=str(e))
            raise


def generate_synthetic_data(n_samples: int = 1000, output_path: str = None) -> pd.DataFrame:
    """Generate synthetic cardiometabolic data for training."""
    np.random.seed(42)
    
    # Demographics
    age = np.random.normal(55, 15, n_samples).clip(18, 90)
    sex = np.random.choice([0, 1], n_samples)  # 0=Female, 1=Male
    bmi = np.random.normal(27, 5, n_samples).clip(15, 50)
    
    # Vitals with realistic correlations
    hba1c = np.random.normal(6.5, 1.5, n_samples).clip(4, 15)
    glucose_fasting = 20 * hba1c + np.random.normal(0, 10, n_samples)
    glucose_fasting = glucose_fasting.clip(70, 400)
    
    sbp = 100 + 0.5 * age + 2 * bmi + np.random.normal(0, 15, n_samples)
    sbp = sbp.clip(90, 200)
    
    dbp = 0.6 * sbp + np.random.normal(0, 8, n_samples)
    dbp = dbp.clip(50, 120)
    
    # Cholesterol
    total_chol = np.random.normal(200, 40, n_samples).clip(120, 350)
    hdl_chol = np.where(sex == 1, 
                       np.random.normal(45, 10, n_samples),  # Male
                       np.random.normal(55, 12, n_samples))  # Female
    hdl_chol = hdl_chol.clip(20, 100)
    
    ldl_chol = total_chol - hdl_chol - np.random.normal(25, 10, n_samples)
    ldl_chol = ldl_chol.clip(50, 250)
    
    triglycerides = np.random.lognormal(4.8, 0.5, n_samples).clip(50, 500)
    
    # Create risk score based on realistic relationships
    risk_score = (
        0.02 * (age - 40) +  # Age effect
        0.3 * (hba1c > 7) +   # Diabetes
        0.2 * (sbp > 140) +   # Hypertension
        0.15 * (bmi > 30) +   # Obesity
        0.1 * (ldl_chol > 130) +  # High LDL
        -0.1 * (hdl_chol > 50) +  # Protective HDL
        0.05 * sex +          # Male risk
        np.random.normal(0, 0.1, n_samples)  # Random variation
    )
    
    # Convert to binary outcome
    risk_prob = 1 / (1 + np.exp(-risk_score))  # Sigmoid
    target = (risk_prob > 0.3).astype(int)  # 30% threshold for high risk
    
    # Create DataFrame
    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'hba1c': hba1c,
        'glucose_fasting': glucose_fasting,
        'sbp': sbp,
        'dbp': dbp,
        'total_cholesterol': total_chol,
        'hdl_cholesterol': hdl_chol,
        'ldl_cholesterol': ldl_chol,
        'triglycerides': triglycerides,
        'heart_rate': np.random.normal(72, 12, n_samples).clip(50, 120),
        'weight_kg': bmi * (1.7 ** 2),  # Approximate weight
        'height_cm': np.random.normal(170, 10, n_samples).clip(150, 200),
        'ethnicity': np.random.choice([0, 1, 2, 3, 4], n_samples),
        'smoking_status': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        'alcohol_use': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.3, 0.4, 0.1]),
        'target': target
    })
    
    # Add some missing values
    missing_cols = ['triglycerides', 'ldl_cholesterol', 'heart_rate']
    for col in missing_cols:
        missing_mask = np.random.random(n_samples) < 0.05  # 5% missing
        data.loc[missing_mask, col] = np.nan
    
    if output_path:
        data.to_csv(output_path, index=False)
        logger.info("Synthetic data generated", path=output_path, shape=data.shape)
    
    return data


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train cardiometabolic risk prediction model')
    parser.add_argument('--dataset', type=str, help='Path to training dataset')
    parser.add_argument('--algo', type=str, default='lightgbm', 
                       choices=['lightgbm', 'xgboost'], help='ML algorithm')
    parser.add_argument('--output', type=str, default='models/risk_model.bin', 
                       help='Output model path')
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--generate-data', action='store_true', 
                       help='Generate synthetic training data')
    parser.add_argument('--n-samples', type=int, default=5000, 
                       help='Number of synthetic samples')
    parser.add_argument('--cv-folds', type=int, default=5, 
                       help='Cross-validation folds')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Calibrate model probabilities')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = RiskModelTrainer(algorithm=args.algo)
        
        # Generate or load data
        if args.generate_data:
            logger.info("Generating synthetic training data")
            data = generate_synthetic_data(
                n_samples=args.n_samples,
                output_path=args.dataset
            )
            target_col = 'target'
        else:
            if not args.dataset:
                raise ValueError("Dataset path required when not generating data")
            data, target_col = trainer.load_data(args.dataset)
        
        # Prepare features
        X, y = trainer.prepare_features(data, target_col)
        
        # Cross-validation
        logger.info("Starting cross-validation")
        cv_results = trainer.cross_validate(X, y, cv_folds=args.cv_folds)
        
        # Train final model
        logger.info("Training final model")
        train_metrics = trainer.train_model(X, y)
        
        # Calibrate if requested
        if args.calibrate:
            logger.info("Calibrating model")
            trainer.calibrate_model(X, y)
        
        # Save model
        model_metadata = trainer.save_model(args.output, args.version)
        
        # Print summary
        logger.info("Training completed successfully",
                   model_version=model_metadata['model_version'],
                   cv_auc_mean=cv_results.get('roc_auc_mean', 0),
                   final_auc=train_metrics.get('roc_auc', 0))
        
        print(f"\nModel saved: {model_metadata['model_file']}")
        print(f"Version: {model_metadata['model_version']}")
        print(f"CV AUC: {cv_results.get('roc_auc_mean', 0):.3f} ± {cv_results.get('roc_auc_std', 0):.3f}")
        print(f"Final AUC: {train_metrics.get('roc_auc', 0):.3f}")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        raise


if __name__ == "__main__":
    main()
