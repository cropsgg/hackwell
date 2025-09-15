"""Data normalization agent for standardizing patient data."""

import re
from typing import Dict, Any, List
from datetime import datetime
import structlog

from base_agent import BaseAgent, AgentContext

logger = structlog.get_logger()


class NormalizerAgent(BaseAgent):
    """Agent responsible for data normalization and standardization."""
    
    def __init__(self):
        super().__init__("normalizer", "Data normalization and standardization")
        
        # Unit conversion mappings
        self.unit_conversions = {
            # Weight conversions to kg
            'weight': {
                'lb': lambda x: x * 0.453592,
                'lbs': lambda x: x * 0.453592,
                'pounds': lambda x: x * 0.453592,
                'kg': lambda x: x,
                'kilograms': lambda x: x
            },
            # Height conversions to cm
            'height': {
                'in': lambda x: x * 2.54,
                'inches': lambda x: x * 2.54,
                'ft': lambda x: x * 30.48,
                'feet': lambda x: x * 30.48,
                'cm': lambda x: x,
                'centimeters': lambda x: x,
                'm': lambda x: x * 100,
                'meters': lambda x: x * 100
            },
            # Glucose conversions to mg/dL
            'glucose': {
                'mg/dl': lambda x: x,
                'mg/dL': lambda x: x,
                'mmol/l': lambda x: x * 18.0182,
                'mmol/L': lambda x: x * 18.0182
            },
            # Cholesterol conversions to mg/dL
            'cholesterol': {
                'mg/dl': lambda x: x,
                'mg/dL': lambda x: x,
                'mmol/l': lambda x: x * 38.67,
                'mmol/L': lambda x: x * 38.67
            }
        }
        
        # Canonical vital types and their expected units
        self.canonical_vitals = {
            'glucose_fasting': 'mg/dL',
            'glucose_random': 'mg/dL',
            'hba1c': '%',
            'sbp': 'mmHg',
            'dbp': 'mmHg',
            'heart_rate': 'bpm',
            'weight': 'kg',
            'height': 'cm',
            'bmi': 'kg/m²',
            'total_cholesterol': 'mg/dL',
            'ldl_cholesterol': 'mg/dL',
            'hdl_cholesterol': 'mg/dL',
            'triglycerides': 'mg/dL',
            'creatinine': 'mg/dL',
            'temperature': '°C'
        }
    
    async def process(self, context: AgentContext) -> AgentContext:
        """Normalize and standardize patient data."""
        try:
            self.log_processing(context, "normalization_started")
            
            # Normalize demographics
            normalized_demographics = self._normalize_demographics(context.patient_profile)
            
            # Normalize vitals
            normalized_vitals = self._normalize_vitals(context.vitals)
            
            # Normalize medications
            normalized_medications = self._normalize_medications(context.medications)
            
            # Normalize conditions
            normalized_conditions = self._normalize_conditions(context.conditions)
            
            # Update context with normalized data
            context.patient_profile = normalized_demographics
            context.vitals = normalized_vitals
            context.medications = normalized_medications
            context.conditions = normalized_conditions
            
            # Prepare normalization summary
            normalization_summary = {
                'demographics_normalized': True,
                'vitals_normalized': len(normalized_vitals),
                'medications_normalized': len(normalized_medications),
                'conditions_normalized': len(normalized_conditions),
                'unit_conversions_applied': self._count_conversions(context.vitals),
                'data_quality_issues': self._identify_quality_issues(context),
                'missing_data_imputed': self._get_imputation_summary(context),
                'processed_at': datetime.utcnow().isoformat()
            }
            
            # Add to context
            context.add_agent_output('normalizer', normalization_summary)
            
            self.log_processing(context, "normalization_completed",
                              vitals_normalized=len(normalized_vitals),
                              conversions_applied=normalization_summary['unit_conversions_applied'])
            
            return context
            
        except Exception as e:
            return self.handle_error(context, e)
    
    def _normalize_demographics(self, demographics: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize demographic information."""
        normalized = demographics.copy()
        
        # Normalize sex values
        if 'sex' in normalized:
            sex_value = str(normalized['sex']).upper()
            if sex_value in ['MALE', 'M', '1']:
                normalized['sex'] = 'M'
            elif sex_value in ['FEMALE', 'F', '0']:
                normalized['sex'] = 'F'
            else:
                normalized['sex'] = 'O'  # Other
        
        # Normalize weight to kg
        if 'weight' in normalized or 'weight_kg' in normalized:
            weight_value = normalized.get('weight_kg') or normalized.get('weight')
            weight_unit = normalized.get('weight_unit', 'kg')
            
            if weight_value:
                normalized_weight = self._convert_unit(weight_value, weight_unit, 'weight')
                normalized['weight_kg'] = normalized_weight
                if 'weight' in normalized:
                    del normalized['weight']
        
        # Normalize height to cm
        if 'height' in normalized or 'height_cm' in normalized:
            height_value = normalized.get('height_cm') or normalized.get('height')
            height_unit = normalized.get('height_unit', 'cm')
            
            if height_value:
                normalized_height = self._convert_unit(height_value, height_unit, 'height')
                normalized['height_cm'] = normalized_height
                if 'height' in normalized:
                    del normalized['height']
        
        # Calculate/normalize BMI
        weight = normalized.get('weight_kg')
        height = normalized.get('height_cm')
        
        if weight and height and height > 0:
            height_m = height / 100
            calculated_bmi = weight / (height_m ** 2)
            
            # Use calculated BMI if not provided or if significantly different
            existing_bmi = normalized.get('bmi')
            if not existing_bmi or abs(calculated_bmi - existing_bmi) > 2:
                normalized['bmi'] = round(calculated_bmi, 1)
        
        # Normalize ethnicity
        if 'ethnicity' in normalized:
            ethnicity = str(normalized['ethnicity']).lower()
            ethnicity_mapping = {
                'white': 'Caucasian',
                'caucasian': 'Caucasian',
                'black': 'African American',
                'african american': 'African American',
                'hispanic': 'Hispanic',
                'latino': 'Hispanic',
                'asian': 'Asian',
                'pacific islander': 'Pacific Islander',
                'native american': 'Native American',
                'other': 'Other',
                'unknown': 'Unknown'
            }
            
            for key, value in ethnicity_mapping.items():
                if key in ethnicity:
                    normalized['ethnicity'] = value
                    break
        
        return normalized
    
    def _normalize_vitals(self, vitals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize vital signs data."""
        normalized_vitals = []
        
        for vital in vitals:
            normalized_vital = vital.copy()
            
            # Normalize vital type
            vital_type = self._normalize_vital_type(vital.get('type', ''))
            normalized_vital['type'] = vital_type
            
            # Get value and unit
            value = vital.get('value')
            unit = vital.get('unit', '')
            
            if value is not None and vital_type in self.canonical_vitals:
                # Convert to canonical unit
                canonical_unit = self.canonical_vitals[vital_type]
                
                if vital_type.startswith('glucose'):
                    normalized_value = self._convert_unit(value, unit, 'glucose')
                elif vital_type.endswith('cholesterol') or vital_type == 'triglycerides':
                    normalized_value = self._convert_unit(value, unit, 'cholesterol')
                else:
                    normalized_value = value  # Already in correct unit or no conversion needed
                
                normalized_vital['value'] = normalized_value
                normalized_vital['unit'] = canonical_unit
            
            # Normalize timestamp
            if 'ts' in normalized_vital:
                normalized_vital['ts'] = self._normalize_timestamp(normalized_vital['ts'])
            
            # Add source if missing
            if 'source' not in normalized_vital:
                normalized_vital['source'] = 'manual'
            
            normalized_vitals.append(normalized_vital)
        
        # Remove duplicates and sort by timestamp
        normalized_vitals = self._deduplicate_vitals(normalized_vitals)
        normalized_vitals.sort(key=lambda x: x.get('ts', ''), reverse=True)
        
        return normalized_vitals
    
    def _normalize_medications(self, medications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize medication data."""
        normalized_medications = []
        
        for medication in medications:
            normalized_med = medication.copy()
            
            # Normalize medication name
            if 'name' in normalized_med:
                normalized_med['name'] = self._normalize_medication_name(normalized_med['name'])
            
            # Normalize dosage format
            if 'dosage' in normalized_med:
                normalized_med['dosage'] = self._normalize_dosage(normalized_med['dosage'])
            
            # Ensure active status is boolean
            if 'active' in normalized_med:
                normalized_med['active'] = bool(normalized_med['active'])
            else:
                normalized_med['active'] = True  # Default to active
            
            # Normalize schedule if present
            if 'schedule' in normalized_med and isinstance(normalized_med['schedule'], dict):
                normalized_med['schedule'] = self._normalize_schedule(normalized_med['schedule'])
            
            normalized_medications.append(normalized_med)
        
        return normalized_medications
    
    def _normalize_conditions(self, conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize medical conditions."""
        normalized_conditions = []
        
        for condition in conditions:
            normalized_condition = condition.copy()
            
            # Normalize condition name
            if 'name' in normalized_condition:
                normalized_condition['name'] = self._normalize_condition_name(normalized_condition['name'])
            
            # Normalize severity
            if 'severity' in normalized_condition:
                severity = str(normalized_condition['severity']).lower()
                if severity in ['mild', 'low', '1']:
                    normalized_condition['severity'] = 'mild'
                elif severity in ['moderate', 'medium', '2']:
                    normalized_condition['severity'] = 'moderate'
                elif severity in ['severe', 'high', '3']:
                    normalized_condition['severity'] = 'severe'
            
            # Ensure active status is boolean
            if 'active' in normalized_condition:
                normalized_condition['active'] = bool(normalized_condition['active'])
            else:
                normalized_condition['active'] = True  # Default to active
            
            normalized_conditions.append(normalized_condition)
        
        return normalized_conditions
    
    def _normalize_vital_type(self, vital_type: str) -> str:
        """Normalize vital sign type names."""
        vital_type_lower = vital_type.lower().replace(' ', '_').replace('-', '_')
        
        # Mapping of common variations to canonical types
        type_mapping = {
            'glucose': 'glucose_fasting',
            'blood_glucose': 'glucose_fasting',
            'bg': 'glucose_fasting',
            'fbg': 'glucose_fasting',
            'fasting_glucose': 'glucose_fasting',
            'random_glucose': 'glucose_random',
            'rbg': 'glucose_random',
            'hemoglobin_a1c': 'hba1c',
            'hgba1c': 'hba1c',
            'a1c': 'hba1c',
            'systolic': 'sbp',
            'systolic_bp': 'sbp',
            'sys_bp': 'sbp',
            'diastolic': 'dbp',
            'diastolic_bp': 'dbp',
            'dias_bp': 'dbp',
            'heart_rate': 'heart_rate',
            'hr': 'heart_rate',
            'pulse': 'heart_rate',
            'weight': 'weight',
            'wt': 'weight',
            'height': 'height',
            'ht': 'height',
            'body_mass_index': 'bmi',
            'cholesterol': 'total_cholesterol',
            'total_chol': 'total_cholesterol',
            'ldl': 'ldl_cholesterol',
            'ldl_chol': 'ldl_cholesterol',
            'hdl': 'hdl_cholesterol',
            'hdl_chol': 'hdl_cholesterol',
            'triglycerides': 'triglycerides',
            'trig': 'triglycerides',
            'temperature': 'temperature',
            'temp': 'temperature'
        }
        
        # Check exact matches first
        if vital_type_lower in type_mapping:
            return type_mapping[vital_type_lower]
        
        # Check partial matches
        for pattern, canonical in type_mapping.items():
            if pattern in vital_type_lower:
                return canonical
        
        # Return original if no mapping found
        return vital_type
    
    def _convert_unit(self, value: float, from_unit: str, unit_type: str) -> float:
        """Convert value between units."""
        if unit_type not in self.unit_conversions:
            return value
        
        conversions = self.unit_conversions[unit_type]
        from_unit_clean = from_unit.lower().strip()
        
        if from_unit_clean in conversions:
            return conversions[from_unit_clean](value)
        
        return value
    
    def _normalize_timestamp(self, timestamp: str) -> str:
        """Normalize timestamp to ISO format."""
        try:
            # Try parsing different formats
            formats = [
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%dT%H:%M:%SZ',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%d',
                '%m/%d/%Y',
                '%m/%d/%y'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp, fmt)
                    return dt.isoformat()
                except ValueError:
                    continue
            
            # If all parsing fails, return original
            return timestamp
            
        except Exception:
            return timestamp
    
    def _normalize_medication_name(self, name: str) -> str:
        """Normalize medication name."""
        # Remove extra whitespace and standardize case
        normalized = ' '.join(name.split()).title()
        
        # Common medication name standardizations
        standardizations = {
            'Metformin Hcl': 'Metformin',
            'Metformin Hydrochloride': 'Metformin',
            'Lisinopril/Hctz': 'Lisinopril/HCTZ',
            'Atorvastatin Calcium': 'Atorvastatin'
        }
        
        for original, standard in standardizations.items():
            if original.lower() in normalized.lower():
                return standard
        
        return normalized
    
    def _normalize_dosage(self, dosage: str) -> str:
        """Normalize dosage format."""
        if not dosage:
            return dosage
        
        # Standardize units
        dosage = dosage.replace('milligrams', 'mg')
        dosage = dosage.replace('micrograms', 'mcg')
        dosage = dosage.replace('units', 'U')
        
        # Standardize format
        dosage = re.sub(r'\s+', ' ', dosage)  # Remove extra spaces
        
        return dosage.strip()
    
    def _normalize_schedule(self, schedule: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize medication schedule."""
        normalized = schedule.copy()
        
        # Standardize frequency terms
        if 'frequency' in normalized:
            freq = normalized['frequency'].lower()
            freq_mapping = {
                'once daily': 'once_daily',
                'twice daily': 'twice_daily',
                'three times daily': 'three_times_daily',
                'four times daily': 'four_times_daily',
                'bid': 'twice_daily',
                'tid': 'three_times_daily',
                'qid': 'four_times_daily',
                'qd': 'once_daily'
            }
            
            for original, standard in freq_mapping.items():
                if original in freq:
                    normalized['frequency'] = standard
                    break
        
        return normalized
    
    def _normalize_condition_name(self, name: str) -> str:
        """Normalize condition name."""
        # Remove extra whitespace and standardize case
        normalized = ' '.join(name.split()).title()
        
        # Common condition name standardizations
        standardizations = {
            'Dm': 'Diabetes Mellitus',
            'Dm Type 2': 'Type 2 Diabetes Mellitus',
            'T2dm': 'Type 2 Diabetes Mellitus',
            'Htn': 'Hypertension',
            'High Blood Pressure': 'Hypertension',
            'High Cholesterol': 'Hyperlipidemia',
            'Elevated Cholesterol': 'Hyperlipidemia'
        }
        
        for original, standard in standardizations.items():
            if original.lower() == normalized.lower():
                return standard
        
        return normalized
    
    def _deduplicate_vitals(self, vitals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate vital measurements."""
        seen = set()
        unique_vitals = []
        
        for vital in vitals:
            # Create key for deduplication
            key = (
                vital.get('type'),
                vital.get('value'),
                vital.get('ts', '')[:10]  # Use date only for deduplication
            )
            
            if key not in seen:
                seen.add(key)
                unique_vitals.append(vital)
        
        return unique_vitals
    
    def _count_conversions(self, vitals: List[Dict[str, Any]]) -> int:
        """Count the number of unit conversions applied."""
        # This is a simplified count - in practice, track conversions during processing
        return len([v for v in vitals if v.get('unit') in self.canonical_vitals.values()])
    
    def _identify_quality_issues(self, context: AgentContext) -> List[str]:
        """Identify data quality issues."""
        issues = []
        
        # Check for missing critical demographics
        demographics = context.patient_profile
        if not demographics.get('age'):
            issues.append('missing_age')
        if not demographics.get('sex'):
            issues.append('missing_sex')
        
        # Check for outlier vital values
        for vital in context.vitals:
            vital_type = vital.get('type')
            value = vital.get('value')
            
            if vital_type == 'glucose_fasting' and value and (value < 50 or value > 500):
                issues.append('outlier_glucose')
            elif vital_type == 'sbp' and value and (value < 70 or value > 250):
                issues.append('outlier_blood_pressure')
            elif vital_type == 'bmi' and value and (value < 15 or value > 60):
                issues.append('outlier_bmi')
        
        return issues
    
    def _get_imputation_summary(self, context: AgentContext) -> Dict[str, int]:
        """Get summary of data imputation performed."""
        # Placeholder for imputation tracking
        return {
            'missing_values_imputed': 0,
            'default_values_applied': 0,
            'calculated_values': 1 if context.patient_profile.get('bmi') else 0
        }
