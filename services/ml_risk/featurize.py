"""Feature engineering and preprocessing for ML models."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import structlog

logger = structlog.get_logger()


class FeatureProcessor:
    """Feature processing pipeline for cardiometabolic risk prediction."""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def fit(self, data: pd.DataFrame, target_col: Optional[str] = None) -> 'FeatureProcessor':
        """Fit preprocessing components on training data."""
        try:
            # Define feature categories
            numeric_features = [
                'age', 'bmi', 'hba1c', 'glucose_fasting', 'glucose_random',
                'sbp', 'dbp', 'heart_rate', 'weight_kg', 'height_cm',
                'total_cholesterol', 'ldl_cholesterol', 'hdl_cholesterol',
                'triglycerides', 'creatinine', 'egfr'
            ]
            
            categorical_features = [
                'sex', 'ethnicity', 'smoking_status', 'alcohol_use'
            ]
            
            # Fit imputers
            for feature in numeric_features:
                if feature in data.columns:
                    imputer = SimpleImputer(strategy='median')
                    imputer.fit(data[[feature]])
                    self.imputers[feature] = imputer
            
            for feature in categorical_features:
                if feature in data.columns:
                    imputer = SimpleImputer(strategy='most_frequent')
                    imputer.fit(data[[feature]])
                    self.imputers[feature] = imputer
            
            # Fit scalers for numeric features
            for feature in numeric_features:
                if feature in data.columns:
                    scaler = StandardScaler()
                    # Apply imputation first
                    if feature in self.imputers:
                        imputed_data = self.imputers[feature].transform(data[[feature]])
                        scaler.fit(imputed_data)
                    else:
                        scaler.fit(data[[feature]])
                    self.scalers[feature] = scaler
            
            # Fit encoders for categorical features
            for feature in categorical_features:
                if feature in data.columns:
                    encoder = LabelEncoder()
                    # Apply imputation first
                    if feature in self.imputers:
                        imputed_data = self.imputers[feature].transform(data[[feature]]).ravel()
                        encoder.fit(imputed_data)
                    else:
                        encoder.fit(data[feature].fillna('unknown'))
                    self.encoders[feature] = encoder
            
            # Store feature names for later use
            self.is_fitted = True  # Set fitted flag before transform
            processed_data = self.transform(data)
            self.feature_names = list(processed_data.columns)
            
            logger.info("Feature processor fitted successfully", 
                       n_features=len(self.feature_names),
                       n_numeric=len(self.scalers),
                       n_categorical=len(self.encoders))
            
            return self
            
        except Exception as e:
            logger.error("Failed to fit feature processor", error=str(e))
            raise
    
    def fit_transform(self, data: pd.DataFrame, target_col: Optional[str] = None) -> pd.DataFrame:
        """Fit the processor and transform data in one step."""
        return self.fit(data, target_col).transform(data)
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted preprocessors."""
        if not self.is_fitted:
            raise ValueError("FeatureProcessor must be fitted before transform")
        
        try:
            processed_data = data.copy()
            
            # Apply imputation and scaling to numeric features
            for feature, imputer in self.imputers.items():
                if feature in processed_data.columns:
                    # Impute missing values
                    imputed_values = imputer.transform(processed_data[[feature]])
                    processed_data[feature] = imputed_values.ravel()
                    
                    # Scale if numeric
                    if feature in self.scalers:
                        scaled_values = self.scalers[feature].transform(processed_data[[feature]])
                        processed_data[feature] = scaled_values.ravel()
            
            # Apply encoding to categorical features
            for feature, encoder in self.encoders.items():
                if feature in processed_data.columns:
                    # Fill missing with 'unknown' for encoding
                    processed_data[feature] = processed_data[feature].fillna('unknown')
                    # Encode
                    try:
                        processed_data[feature] = encoder.transform(processed_data[feature])
                    except ValueError:
                        # Handle unseen categories
                        processed_data[feature] = processed_data[feature].map(
                            lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
                        )
            
            # Create engineered features
            processed_data = self._engineer_features(processed_data)
            
            # Select only features that were present during fitting
            available_features = [f for f in self.feature_names if f in processed_data.columns]
            processed_data = processed_data[available_features]
            
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in processed_data.columns:
                    processed_data[feature] = 0.0
            
            # Ensure correct order
            processed_data = processed_data[self.feature_names]
            
            return processed_data
            
        except Exception as e:
            logger.error("Failed to transform features", error=str(e))
            raise
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features."""
        try:
            # BMI categories
            if 'bmi' in data.columns:
                data['bmi_category'] = pd.cut(
                    data['bmi'], 
                    bins=[0, 18.5, 25, 30, float('inf')], 
                    labels=[0, 1, 2, 3],
                    include_lowest=True
                ).astype(float)
            
            # Blood pressure categories
            if 'sbp' in data.columns and 'dbp' in data.columns:
                data['hypertension_stage'] = np.where(
                    (data['sbp'] >= 140) | (data['dbp'] >= 90), 1, 0
                )
                data['bp_ratio'] = data['sbp'] / (data['dbp'] + 1e-6)
            
            # Glucose control indicators
            if 'hba1c' in data.columns:
                data['diabetes_controlled'] = np.where(data['hba1c'] < 7.0, 1, 0)
                data['hba1c_severity'] = pd.cut(
                    data['hba1c'],
                    bins=[0, 5.7, 6.5, 7.0, 9.0, float('inf')],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                ).astype(float)
            
            # Cholesterol ratios
            if 'total_cholesterol' in data.columns and 'hdl_cholesterol' in data.columns:
                data['chol_hdl_ratio'] = data['total_cholesterol'] / (data['hdl_cholesterol'] + 1e-6)
            
            if 'ldl_cholesterol' in data.columns and 'hdl_cholesterol' in data.columns:
                data['ldl_hdl_ratio'] = data['ldl_cholesterol'] / (data['hdl_cholesterol'] + 1e-6)
            
            # Age groups
            if 'age' in data.columns:
                data['age_group'] = pd.cut(
                    data['age'],
                    bins=[0, 40, 50, 60, 70, float('inf')],
                    labels=[0, 1, 2, 3, 4],
                    include_lowest=True
                ).astype(float)
            
            # Metabolic syndrome components
            metabolic_components = []
            
            if 'bmi' in data.columns:
                metabolic_components.append((data['bmi'] >= 30).astype(int))
            
            if 'sbp' in data.columns and 'dbp' in data.columns:
                metabolic_components.append(
                    ((data['sbp'] >= 130) | (data['dbp'] >= 85)).astype(int)
                )
            
            if 'glucose_fasting' in data.columns:
                metabolic_components.append((data['glucose_fasting'] >= 100).astype(int))
            
            if 'triglycerides' in data.columns:
                metabolic_components.append((data['triglycerides'] >= 150).astype(int))
            
            if 'hdl_cholesterol' in data.columns and 'sex' in data.columns:
                # Sex-specific HDL thresholds
                hdl_low = np.where(
                    data['sex'] == 1,  # Assuming 1 = male
                    data['hdl_cholesterol'] < 40,
                    data['hdl_cholesterol'] < 50
                )
                metabolic_components.append(hdl_low.astype(int))
            
            if len(metabolic_components) > 0:
                data['metabolic_syndrome_score'] = sum(metabolic_components)
            
            # Cardiovascular risk factors count
            risk_factors = []
            
            if 'age' in data.columns and 'sex' in data.columns:
                # Age-based risk (sex-specific)
                age_risk = np.where(
                    data['sex'] == 1,  # Male
                    data['age'] >= 45,
                    data['age'] >= 55   # Female
                )
                risk_factors.append(age_risk.astype(int))
            
            if len(risk_factors) > 0:
                data['cv_risk_factors'] = sum(risk_factors)
            
            return data
            
        except Exception as e:
            logger.error("Failed to engineer features", error=str(e))
            return data


def extract_patient_features(patient_data: Dict[str, Any]) -> Dict[str, float]:
    """Extract features from patient data for inference."""
    try:
        features = {}
        
        # Demographics
        demographics = patient_data.get('demographics', {})
        features['age'] = float(demographics.get('age', 50))
        features['sex'] = 1.0 if demographics.get('sex') == 'M' else 0.0
        features['bmi'] = float(demographics.get('bmi', 25.0))
        features['weight_kg'] = float(demographics.get('weight_kg', 70.0))
        features['height_cm'] = float(demographics.get('height_cm', 170.0))
        
        # Ethnicity encoding (simplified)
        ethnicity_map = {
            'Caucasian': 0, 'African American': 1, 'Hispanic': 2, 
            'Asian': 3, 'Other': 4
        }
        features['ethnicity'] = float(ethnicity_map.get(
            demographics.get('ethnicity'), 4
        ))
        
        # Lifestyle factors
        lifestyle = demographics.get('lifestyle', {})
        smoking_map = {'never': 0, 'former': 1, 'current': 2}
        features['smoking_status'] = float(smoking_map.get(
            lifestyle.get('smoking'), 0
        ))
        
        alcohol_map = {'none': 0, 'rare': 1, 'occasional': 2, 'moderate': 3, 'heavy': 4}
        features['alcohol_use'] = float(alcohol_map.get(
            lifestyle.get('alcohol'), 1
        ))
        
        # Latest vitals
        vitals = patient_data.get('vitals', [])
        vital_values = {}
        
        # Get most recent value for each vital type
        for vital in vitals:
            vital_type = vital.get('type')
            value = vital.get('value')
            if vital_type and value is not None:
                vital_values[vital_type] = float(value)
        
        # Map vitals to features with defaults
        features['glucose_fasting'] = vital_values.get('glucose_fasting', 90.0)
        features['glucose_random'] = vital_values.get('glucose_random', 120.0)
        features['hba1c'] = vital_values.get('hba1c', 5.5)
        features['sbp'] = vital_values.get('sbp', 120.0)
        features['dbp'] = vital_values.get('dbp', 80.0)
        features['heart_rate'] = vital_values.get('heart_rate', 70.0)
        features['total_cholesterol'] = vital_values.get('total_cholesterol', 180.0)
        features['ldl_cholesterol'] = vital_values.get('ldl_cholesterol', 100.0)
        features['hdl_cholesterol'] = vital_values.get('hdl_cholesterol', 50.0)
        features['triglycerides'] = vital_values.get('triglycerides', 120.0)
        features['creatinine'] = vital_values.get('creatinine', 1.0)
        features['egfr'] = vital_values.get('egfr', 90.0)
        
        # Conditions (binary indicators)
        conditions = patient_data.get('conditions', [])
        condition_indicators = {
            'diabetes': False,
            'hypertension': False,
            'heart_disease': False,
            'kidney_disease': False,
            'hyperlipidemia': False
        }
        
        for condition in conditions:
            if condition.get('active', False):
                name = condition.get('name', '').lower()
                if 'diabetes' in name:
                    condition_indicators['diabetes'] = True
                elif 'hypertension' in name:
                    condition_indicators['hypertension'] = True
                elif 'heart' in name or 'cardiac' in name:
                    condition_indicators['heart_disease'] = True
                elif 'kidney' in name or 'renal' in name:
                    condition_indicators['kidney_disease'] = True
                elif 'lipid' in name or 'cholesterol' in name:
                    condition_indicators['hyperlipidemia'] = True
        
        for condition, value in condition_indicators.items():
            features[f'has_{condition}'] = float(value)
        
        # Medication indicators
        medications = patient_data.get('medications', [])
        med_indicators = {
            'metformin': False,
            'insulin': False,
            'ace_inhibitor': False,
            'beta_blocker': False,
            'statin': False,
            'diuretic': False
        }
        
        for med in medications:
            if med.get('active', False):
                name = med.get('name', '').lower()
                if 'metformin' in name:
                    med_indicators['metformin'] = True
                elif 'insulin' in name:
                    med_indicators['insulin'] = True
                elif any(x in name for x in ['lisinopril', 'enalapril', 'captopril']):
                    med_indicators['ace_inhibitor'] = True
                elif any(x in name for x in ['metoprolol', 'atenolol', 'propranolol']):
                    med_indicators['beta_blocker'] = True
                elif any(x in name for x in ['atorvastatin', 'simvastatin', 'rosuvastatin']):
                    med_indicators['statin'] = True
                elif any(x in name for x in ['hydrochlorothiazide', 'furosemide']):
                    med_indicators['diuretic'] = True
        
        for med, value in med_indicators.items():
            features[f'on_{med}'] = float(value)
        
        return features
        
    except Exception as e:
        logger.error("Failed to extract patient features", error=str(e))
        # Return default features
        return {
            'age': 50.0, 'sex': 0.0, 'bmi': 25.0, 'hba1c': 5.5,
            'glucose_fasting': 90.0, 'sbp': 120.0, 'dbp': 80.0,
            'total_cholesterol': 180.0, 'hdl_cholesterol': 50.0,
            'ldl_cholesterol': 100.0, 'triglycerides': 120.0
        }
