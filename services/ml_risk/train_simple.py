"""Simplified training script for ML model."""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import structlog

logger = structlog.get_logger()

def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic cardiometabolic data."""
    np.random.seed(42)
    
    data = {
        'age': np.random.normal(55, 15, n_samples).clip(18, 90),
        'bmi': np.random.normal(28, 5, n_samples).clip(15, 50),
        'systolic_bp': np.random.normal(130, 20, n_samples).clip(80, 200),
        'diastolic_bp': np.random.normal(80, 10, n_samples).clip(50, 120),
        'hba1c': np.random.normal(6.5, 1.2, n_samples).clip(4, 15),
        'glucose_fasting': np.random.normal(120, 30, n_samples).clip(70, 300),
        'total_cholesterol': np.random.normal(200, 40, n_samples).clip(100, 400),
        'hdl_cholesterol': np.random.normal(50, 15, n_samples).clip(20, 100),
        'ldl_cholesterol': np.random.normal(120, 30, n_samples).clip(50, 250),
        'triglycerides': np.random.normal(150, 50, n_samples).clip(50, 500),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    # Create target variable (high risk)
    # Higher risk with age, BMI, blood pressure, HbA1c, etc.
    risk_score = (
        (data['age'] - 40) / 50 * 0.3 +
        (data['bmi'] - 25) / 10 * 0.2 +
        (data['systolic_bp'] - 120) / 40 * 0.2 +
        (data['hba1c'] - 5.7) / 3 * 0.2 +
        data['smoking'] * 0.1 +
        data['family_history'] * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Convert to binary classification
    data['high_risk'] = (risk_score > 0.4).astype(int)
    
    return pd.DataFrame(data)

def train_model():
    """Train a simple risk prediction model."""
    logger.info("Generating synthetic data...")
    data = generate_synthetic_data(2000)
    
    # Prepare features and target
    feature_cols = [col for col in data.columns if col != 'high_risk']
    X = data[feature_cols]
    y = data['high_risk']
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Features: {list(X.columns)}")
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model
    logger.info("Training RandomForest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Model AUC: {auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Feature Importances:")
    print(importance.head(10))
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "cardiometabolic_risk_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names
    feature_names_path = models_dir / "feature_names.pkl"
    joblib.dump(list(X.columns), feature_names_path)
    logger.info(f"Feature names saved to {feature_names_path}")
    
    return model, X.columns

if __name__ == "__main__":
    train_model()
