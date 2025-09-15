"""Unified training script for heart disease and diabetes datasets."""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, accuracy_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
import structlog

from featurize import FeatureProcessor
from metrics import ModelMetrics
from data_loader import UnifiedDataLoader, create_sample_data

# Configure logging
logger = structlog.get_logger()


class UnifiedModelTrainer:
    """Train and evaluate models on heart disease and diabetes datasets."""
    
    def __init__(self, algorithm: str = 'lightgbm', random_state: int = 42):
        self.algorithm = algorithm
        self.random_state = random_state
        self.model = None
        self.feature_processor = None
        self.metrics_calculator = ModelMetrics()
        self.dataset_info = {}
        
    def load_data(self, 
                  heart_disease_path: str = None,
                  diabetes_path: str = None,
                  dataset_type: str = 'combined',
                  use_sample_data: bool = False) -> Tuple[pd.DataFrame, str]:
        """Load training data."""
        try:
            if use_sample_data:
                logger.info("Using sample data for training")
                data = create_sample_data()
                target_col = 'target'
                self.dataset_info = {
                    'type': 'sample',
                    'samples': len(data),
                    'features': len(data.columns) - 1
                }
                return data, target_col
            
            if dataset_type == 'combined' and heart_disease_path and diabetes_path:
                loader = UnifiedDataLoader(heart_disease_path, diabetes_path)
                data, target_col = loader.load_combined_data()
                self.dataset_info = {
                    'type': 'combined',
                    'samples': len(data),
                    'features': len(data.columns) - 1,
                    'datasets': ['heart_disease', 'diabetes']
                }
                
            elif dataset_type == 'heart' and heart_disease_path:
                loader = UnifiedDataLoader(heart_disease_path, "")
                data = loader.load_heart_disease_data()
                target_col = 'num'
                self.dataset_info = {
                    'type': 'heart_disease',
                    'samples': len(data),
                    'features': len(data.columns) - 1
                }
                
            elif dataset_type == 'diabetes' and diabetes_path:
                loader = UnifiedDataLoader("", diabetes_path)
                data = loader.load_diabetes_data()
                target_col = 'readmitted'
                self.dataset_info = {
                    'type': 'diabetes',
                    'samples': len(data),
                    'features': len(data.columns) - 1
                }
            else:
                raise ValueError("Invalid dataset configuration")
            
            logger.info("Data loaded successfully", 
                       shape=data.shape, 
                       target_col=target_col,
                       dataset_info=self.dataset_info)
            
            return data, target_col
            
        except Exception as e:
            logger.error("Failed to load data", error=str(e))
            raise
    
    def prepare_features(self, data: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for training."""
        try:
            # Separate features and target
            X = data.drop(columns=[target_col])
            y = data[target_col]
            
            # Remove non-feature columns
            non_feature_cols = ['source', 'dataset', 'encounter_id', 'patient_nbr']
            X = X.drop(columns=[col for col in non_feature_cols if col in X.columns])
            
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
        
        elif self.algorithm == 'random_forest':
            default_params = {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': self.random_state,
                'class_weight': 'balanced'
            }
            default_params.update(hyperparams)
            return RandomForestClassifier(**default_params)
        
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
            if self.algorithm in ['lightgbm', 'xgboost']:
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                self.model.fit(X_train, y_train)
            
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
                'brier_score': [],
                'accuracy': []
            }
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
                y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train fold model
                fold_model = self.create_model()
                fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict
                y_pred_proba = fold_model.predict_proba(X_fold_val)[:, 1]
                y_pred = fold_model.predict(X_fold_val)
                
                # Calculate metrics
                cv_scores['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
                cv_scores['average_precision'].append(
                    average_precision_score(y_fold_val, y_pred_proba)
                )
                cv_scores['brier_score'].append(
                    brier_score_loss(y_fold_val, y_pred_proba)
                )
                cv_scores['accuracy'].append(accuracy_score(y_fold_val, y_pred))
                
                logger.info(f"Fold {fold + 1} completed", 
                           roc_auc=cv_scores['roc_auc'][-1],
                           accuracy=cv_scores['accuracy'][-1])
            
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
                model_version = f"unified_{self.algorithm}_v{timestamp}"
            
            # Save model
            model_file = output_path.parent / f"{model_version}.bin"
            joblib.dump({
                'model': self.model,
                'feature_processor': self.feature_processor,
                'algorithm': self.algorithm,
                'version': model_version,
                'created_at': datetime.now().isoformat(),
                'feature_names': self.feature_processor.feature_names if self.feature_processor else None,
                'dataset_info': self.dataset_info
            }, model_file)
            
            # Create metadata
            metadata = {
                'model_version': model_version,
                'algorithm': self.algorithm,
                'model_file': str(model_file),
                'feature_count': len(self.feature_processor.feature_names) if self.feature_processor else 0,
                'created_at': datetime.now().isoformat(),
                'training_completed': True,
                'dataset_info': self.dataset_info
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
    
    def print_feature_importance(self, top_n: int = 20) -> None:
        """Print feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_processor.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\nTop {top_n} Feature Importances:")
            print(importance.head(top_n).to_string(index=False))
        else:
            print("Feature importance not available for this model type")


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description='Train unified model on heart disease and diabetes data')
    parser.add_argument('--heart-data', type=str, 
                       default='/Users/crops/Desktop/hackwell/heart+disease',
                       help='Path to heart disease data')
    parser.add_argument('--diabetes-data', type=str,
                       default='/Users/crops/Desktop/hackwell/diabetes+130-us+hospitals+for+years+1999-2008',
                       help='Path to diabetes data')
    parser.add_argument('--dataset-type', type=str, default='combined',
                       choices=['combined', 'heart', 'diabetes', 'sample'],
                       help='Type of dataset to use')
    parser.add_argument('--algo', type=str, default='lightgbm', 
                       choices=['lightgbm', 'xgboost', 'random_forest'], 
                       help='ML algorithm')
    parser.add_argument('--output', type=str, default='models/unified_model.bin', 
                       help='Output model path')
    parser.add_argument('--version', type=str, help='Model version')
    parser.add_argument('--cv-folds', type=int, default=5, 
                       help='Cross-validation folds')
    parser.add_argument('--calibrate', action='store_true', 
                       help='Calibrate model probabilities')
    parser.add_argument('--use-sample', action='store_true',
                       help='Use sample data instead of real datasets')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = UnifiedModelTrainer(algorithm=args.algo)
        
        # Load data
        if args.use_sample or args.dataset_type == 'sample':
            data, target_col = trainer.load_data(use_sample_data=True)
        else:
            data, target_col = trainer.load_data(
                heart_disease_path=args.heart_data,
                diabetes_path=args.diabetes_data,
                dataset_type=args.dataset_type
            )
        
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
        
        # Print feature importance
        trainer.print_feature_importance()
        
        # Save model
        model_metadata = trainer.save_model(args.output, args.version)
        
        # Print summary
        logger.info("Training completed successfully",
                   model_version=model_metadata['model_version'],
                   cv_auc_mean=cv_results.get('roc_auc_mean', 0),
                   final_auc=train_metrics.get('roc_auc', 0))
        
        print(f"\n{'='*60}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*60}")
        print(f"Model saved: {model_metadata['model_file']}")
        print(f"Version: {model_metadata['model_version']}")
        print(f"Algorithm: {args.algo}")
        print(f"Dataset: {model_metadata['dataset_info']['type']}")
        print(f"Samples: {model_metadata['dataset_info']['samples']}")
        print(f"Features: {model_metadata['feature_count']}")
        print(f"\nCross-Validation Results:")
        print(f"  AUC: {cv_results.get('roc_auc_mean', 0):.3f} ± {cv_results.get('roc_auc_std', 0):.3f}")
        print(f"  Accuracy: {cv_results.get('accuracy_mean', 0):.3f} ± {cv_results.get('accuracy_std', 0):.3f}")
        print(f"  Avg Precision: {cv_results.get('average_precision_mean', 0):.3f} ± {cv_results.get('average_precision_std', 0):.3f}")
        print(f"\nFinal Model Results:")
        print(f"  AUC: {train_metrics.get('roc_auc', 0):.3f}")
        print(f"  Accuracy: {train_metrics.get('accuracy', 0):.3f}")
        print(f"  Avg Precision: {train_metrics.get('average_precision', 0):.3f}")
        print(f"{'='*60}")
        
    except Exception as e:
        logger.error("Training failed", error=str(e))
        raise


if __name__ == "__main__":
    main()

