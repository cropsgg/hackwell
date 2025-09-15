"""Unified data loader for heart disease and diabetes datasets."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import structlog

logger = structlog.get_logger()


class HeartDiseaseLoader:
    """Loader for heart disease dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
        ]
        self.target_name = 'num'
    
    def load_cleveland_data(self) -> pd.DataFrame:
        """Load Cleveland heart disease data."""
        try:
            # Load processed Cleveland data
            data = pd.read_csv(
                self.data_path / "processed.cleveland.data",
                names=self.feature_names + [self.target_name],
                na_values=['?', -9.0]
            )
            
            # Convert target to binary (0 = no disease, 1+ = disease)
            data[self.target_name] = (data[self.target_name] > 0).astype(int)
            
            logger.info("Cleveland heart disease data loaded", 
                       shape=data.shape, 
                       target_distribution=data[self.target_name].value_counts().to_dict())
            
            return data
            
        except Exception as e:
            logger.error("Failed to load Cleveland data", error=str(e))
            raise
    
    def load_all_heart_data(self) -> pd.DataFrame:
        """Load all heart disease datasets combined."""
        datasets = []
        
        # Load each dataset
        datasets_to_load = [
            ("processed.cleveland.data", "Cleveland"),
            ("processed.hungarian.data", "Hungarian"),
            ("processed.switzerland.data", "Switzerland"),
            ("processed.va.data", "VA")
        ]
        
        for filename, source in datasets_to_load:
            try:
                data = pd.read_csv(
                    self.data_path / filename,
                    names=self.feature_names + [self.target_name],
                    na_values=['?', -9.0]
                )
                
                # Convert target to binary
                data[self.target_name] = (data[self.target_name] > 0).astype(int)
                data['source'] = source
                
                datasets.append(data)
                logger.info(f"{source} data loaded", shape=data.shape)
                
            except Exception as e:
                logger.warning(f"Failed to load {source} data", error=str(e))
                continue
        
        if not datasets:
            raise ValueError("No heart disease datasets could be loaded")
        
        # Combine all datasets
        combined_data = pd.concat(datasets, ignore_index=True)
        
        logger.info("All heart disease data combined", 
                   total_shape=combined_data.shape,
                   sources=combined_data['source'].value_counts().to_dict())
        
        return combined_data


class DiabetesLoader:
    """Loader for diabetes dataset."""
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
    
    def load_diabetes_data(self) -> pd.DataFrame:
        """Load diabetes readmission data."""
        try:
            # Load main diabetes data
            data = pd.read_csv(self.data_path / "diabetic_data.csv")
            
            # Load ID mappings
            admission_mapping = pd.read_csv(
                self.data_path / "IDS_mapping.csv",
                usecols=['admission_type_id', 'description']
            ).dropna()
            
            # Create mapping dictionaries
            admission_map = dict(zip(
                admission_mapping['admission_type_id'], 
                admission_mapping['description']
            ))
            
            # Preprocess the data
            processed_data = self._preprocess_diabetes_data(data, admission_map)
            
            logger.info("Diabetes data loaded and preprocessed", 
                       shape=processed_data.shape,
                       target_distribution=processed_data['readmitted'].value_counts().to_dict())
            
            return processed_data
            
        except Exception as e:
            logger.error("Failed to load diabetes data", error=str(e))
            raise
    
    def _preprocess_diabetes_data(self, data: pd.DataFrame, admission_map: Dict) -> pd.DataFrame:
        """Preprocess diabetes data for ML."""
        processed = data.copy()
        
        # Handle missing values
        processed = processed.replace('?', np.nan)
        
        # Convert age groups to numeric
        age_mapping = {
            '[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35,
            '[40-50)': 45, '[50-60)': 55, '[60-70)': 65, '[70-80)': 75,
            '[80-90)': 85, '[90-100)': 95
        }
        processed['age_numeric'] = processed['age'].map(age_mapping)
        
        # Convert gender to binary
        processed['sex'] = (processed['gender'] == 'Male').astype(int)
        
        # Convert race to numeric categories
        race_mapping = {
            'Caucasian': 0, 'AfricanAmerican': 1, 'Hispanic': 2,
            'Asian': 3, 'Other': 4
        }
        processed['race_numeric'] = processed['race'].map(race_mapping).fillna(4)
        
        # Convert admission type
        processed['admission_type'] = processed['admission_type_id'].map(admission_map)
        processed['is_emergency'] = (processed['admission_type'] == 'Emergency').astype(int)
        
        # Convert glucose and A1C results
        processed['max_glu_serum_numeric'] = processed['max_glu_serum'].map({
            'None': 0, 'Norm': 1, '>200': 2, '>300': 3
        }).fillna(0)
        
        processed['A1Cresult_numeric'] = processed['A1Cresult'].map({
            'None': 0, 'Norm': 1, '>7': 2, '>8': 3
        }).fillna(0)
        
        # Convert medication columns to binary
        medication_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
        for col in medication_cols:
            if col in processed.columns:
                processed[f'{col}_binary'] = (processed[col] == 'Steady').astype(int)
        
        # Convert change and diabetesMed
        processed['change_binary'] = (processed['change'] == 'Ch').astype(int)
        processed['diabetesMed_binary'] = (processed['diabetesMed'] == 'Yes').astype(int)
        
        # Create target variable (readmission within 30 days)
        processed['readmitted'] = (processed['readmitted'] == '<30').astype(int)
        
        # Select features for ML
        feature_cols = [
            'age_numeric', 'sex', 'race_numeric', 'admission_type_id',
            'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
            'num_lab_procedures', 'num_procedures', 'num_medications',
            'number_outpatient', 'number_emergency', 'number_inpatient',
            'number_diagnoses', 'max_glu_serum_numeric', 'A1Cresult_numeric',
            'is_emergency', 'change_binary', 'diabetesMed_binary'
        ]
        
        # Add medication features
        for col in medication_cols:
            if f'{col}_binary' in processed.columns:
                feature_cols.append(f'{col}_binary')
        
        # Create final dataset
        ml_data = processed[feature_cols + ['readmitted']].copy()
        
        # Handle remaining missing values
        ml_data = ml_data.fillna(ml_data.median())
        
        return ml_data


class UnifiedDataLoader:
    """Unified loader for both heart disease and diabetes datasets."""
    
    def __init__(self, heart_disease_path: str, diabetes_path: str):
        self.heart_loader = HeartDiseaseLoader(heart_disease_path)
        self.diabetes_loader = DiabetesLoader(diabetes_path)
    
    def load_heart_disease_data(self, use_all_datasets: bool = True) -> pd.DataFrame:
        """Load heart disease data."""
        if use_all_datasets:
            return self.heart_loader.load_all_heart_data()
        else:
            return self.heart_loader.load_cleveland_data()
    
    def load_diabetes_data(self) -> pd.DataFrame:
        """Load diabetes data."""
        return self.diabetes_loader.load_diabetes_data()
    
    def load_combined_data(self, 
                          include_heart: bool = True, 
                          include_diabetes: bool = True,
                          use_all_heart_datasets: bool = True) -> Tuple[pd.DataFrame, str]:
        """Load and combine both datasets."""
        datasets = []
        target_cols = []
        
        if include_heart:
            heart_data = self.load_heart_disease_data(use_all_heart_datasets)
            # Rename target for heart disease
            heart_data = heart_data.rename(columns={'num': 'heart_disease'})
            datasets.append(heart_data)
            target_cols.append('heart_disease')
        
        if include_diabetes:
            diabetes_data = self.load_diabetes_data()
            # Rename target for diabetes
            diabetes_data = diabetes_data.rename(columns={'readmitted': 'diabetes_readmission'})
            datasets.append(diabetes_data)
            target_cols.append('diabetes_readmission')
        
        if not datasets:
            raise ValueError("At least one dataset must be included")
        
        # Combine datasets
        if len(datasets) == 1:
            combined_data = datasets[0]
            target_col = target_cols[0]
        else:
            # For multiple datasets, we need to align features
            combined_data = self._align_datasets(datasets, target_cols)
            target_col = 'combined_target'  # We'll create this
        
        logger.info("Combined data loaded", 
                   shape=combined_data.shape,
                   target_columns=target_cols)
        
        return combined_data, target_col
    
    def _align_datasets(self, datasets: List[pd.DataFrame], target_cols: List[str]) -> pd.DataFrame:
        """Align features across datasets for combination."""
        # Find common features
        all_features = set()
        for dataset in datasets:
            all_features.update(dataset.columns)
        
        # Remove target columns from common features
        common_features = all_features - set(target_cols)
        
        # Create aligned datasets
        aligned_datasets = []
        for i, dataset in enumerate(datasets):
            aligned = pd.DataFrame(index=dataset.index)
            
            # Add common features
            for feature in common_features:
                if feature in dataset.columns:
                    aligned[feature] = dataset[feature]
                else:
                    aligned[feature] = 0  # Default value for missing features
            
            # Add target
            aligned[target_cols[i]] = dataset[target_cols[i]]
            
            # Add dataset identifier
            aligned['dataset'] = f'dataset_{i}'
            
            aligned_datasets.append(aligned)
        
        # Combine aligned datasets
        combined = pd.concat(aligned_datasets, ignore_index=True)
        
        # Create combined target (any positive outcome)
        combined['combined_target'] = 0
        for target_col in target_cols:
            if target_col in combined.columns:
                # Fill NaN values with 0 and ensure integer type
                target_series = combined[target_col].fillna(0).astype(int)
                combined['combined_target'] = combined['combined_target'].astype(int) | target_series
        
        return combined


def create_sample_data() -> pd.DataFrame:
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic features
    data = {
        'age': np.random.normal(55, 15, n_samples).clip(18, 90),
        'sex': np.random.choice([0, 1], n_samples),
        'bmi': np.random.normal(28, 5, n_samples).clip(15, 50),
        'sbp': np.random.normal(130, 20, n_samples).clip(80, 200),
        'dbp': np.random.normal(80, 10, n_samples).clip(50, 120),
        'hba1c': np.random.normal(6.5, 1.2, n_samples).clip(4, 15),
        'glucose_fasting': np.random.normal(120, 30, n_samples).clip(70, 300),
        'total_cholesterol': np.random.normal(200, 40, n_samples).clip(100, 400),
        'hdl_cholesterol': np.random.normal(50, 15, n_samples).clip(20, 100),
        'ldl_cholesterol': np.random.normal(120, 30, n_samples).clip(50, 250),
        'triglycerides': np.random.normal(150, 50, n_samples).clip(50, 500),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }
    
    # Create target based on realistic risk factors
    risk_score = (
        (data['age'] - 40) / 50 * 0.3 +
        (data['bmi'] - 25) / 10 * 0.2 +
        (data['sbp'] - 120) / 40 * 0.2 +
        (data['hba1c'] - 5.7) / 3 * 0.2 +
        data['smoking'] * 0.1 +
        data['family_history'] * 0.1 +
        np.random.normal(0, 0.1, n_samples)
    )
    
    data['target'] = (risk_score > 0.4).astype(int)
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Test the data loaders
    heart_path = "/Users/crops/Desktop/hackwell/heart+disease"
    diabetes_path = "/Users/crops/Desktop/hackwell/diabetes+130-us+hospitals+for+years+1999-2008"
    
    loader = UnifiedDataLoader(heart_path, diabetes_path)
    
    # Test heart disease loading
    print("Loading heart disease data...")
    heart_data = loader.load_heart_disease_data()
    print(f"Heart disease data shape: {heart_data.shape}")
    print(f"Target distribution: {heart_data['num'].value_counts().to_dict()}")
    
    # Test diabetes loading
    print("\nLoading diabetes data...")
    diabetes_data = loader.load_diabetes_data()
    print(f"Diabetes data shape: {diabetes_data.shape}")
    print(f"Target distribution: {diabetes_data['readmitted'].value_counts().to_dict()}")
    
    # Test combined loading
    print("\nLoading combined data...")
    combined_data, target_col = loader.load_combined_data()
    print(f"Combined data shape: {combined_data.shape}")
    print(f"Target column: {target_col}")
