"""Intake agent for processing and validating patient data."""

from typing import Dict, Any, List
from datetime import datetime
import structlog

from base_agent import BaseAgent, AgentContext

logger = structlog.get_logger()


class IntakeAgent(BaseAgent):
    """Agent responsible for patient data intake and initial processing."""
    
    def __init__(self):
        super().__init__("intake", "Patient data intake and validation")
        
        # Validation rules
        self.required_demographics = ['age', 'sex']
        self.vital_ranges = {
            'age': (0, 150),
            'glucose_fasting': (50, 500),
            'glucose_random': (50, 600),
            'hba1c': (3.0, 18.0),
            'sbp': (60, 250),
            'dbp': (30, 150),
            'heart_rate': (30, 200),
            'bmi': (10, 80),
            'weight_kg': (20, 500),
            'height_cm': (100, 250),
            'total_cholesterol': (50, 500),
            'ldl_cholesterol': (20, 400),
            'hdl_cholesterol': (10, 150),
            'triglycerides': (30, 1000)
        }
    
    async def process(self, context: AgentContext) -> AgentContext:
        """Process and validate patient data."""
        try:
            self.log_processing(context, "intake_started")
            
            # Validate patient profile
            profile_issues = self._validate_demographics(context.patient_profile)
            
            # Validate vitals
            vitals_issues = self._validate_vitals(context.vitals)
            
            # Validate medications
            medication_issues = self._validate_medications(context.medications)
            
            # Validate conditions
            condition_issues = self._validate_conditions(context.conditions)
            
            # Collect all validation issues
            all_issues = profile_issues + vitals_issues + medication_issues + condition_issues
            
            # Prepare intake summary
            intake_summary = {
                'validation_status': 'passed' if not all_issues else 'warnings',
                'issues': all_issues,
                'data_completeness': self._assess_data_completeness(context),
                'processed_at': datetime.utcnow().isoformat(),
                'demographics_valid': len(profile_issues) == 0,
                'vitals_count': len(context.vitals),
                'medications_count': len([m for m in context.medications if m.get('active', True)]),
                'conditions_count': len([c for c in context.conditions if c.get('active', True)])
            }
            
            # Add to context
            context.add_agent_output('intake', intake_summary)
            
            # Log any issues
            if all_issues:
                self.logger.warning("Data validation issues found", 
                                  patient_id=context.patient_id,
                                  issues=all_issues)
            
            self.log_processing(context, "intake_completed", 
                              status=intake_summary['validation_status'])
            
            return context
            
        except Exception as e:
            return self.handle_error(context, e)
    
    def _validate_demographics(self, demographics: Dict[str, Any]) -> List[str]:
        """Validate demographic information."""
        issues = []
        
        # Check required fields
        for field in self.required_demographics:
            if field not in demographics or demographics[field] is None:
                issues.append(f"Missing required demographic field: {field}")
        
        # Validate age
        age = demographics.get('age')
        if age is not None:
            if not isinstance(age, (int, float)) or age < 0 or age > 150:
                issues.append(f"Invalid age: {age}")
        
        # Validate sex
        sex = demographics.get('sex')
        if sex is not None and sex not in ['M', 'F', 'O']:
            issues.append(f"Invalid sex value: {sex}")
        
        # Validate BMI if present
        bmi = demographics.get('bmi')
        if bmi is not None:
            if not isinstance(bmi, (int, float)) or bmi < 10 or bmi > 80:
                issues.append(f"BMI out of expected range: {bmi}")
        
        # Validate weight/height consistency with BMI
        weight = demographics.get('weight_kg')
        height = demographics.get('height_cm')
        if all([weight, height, bmi]):
            calculated_bmi = weight / ((height / 100) ** 2)
            if abs(calculated_bmi - bmi) > 2.0:  # Allow some tolerance
                issues.append(f"BMI inconsistent with weight/height (calculated: {calculated_bmi:.1f})")
        
        return issues
    
    def _validate_vitals(self, vitals: List[Dict[str, Any]]) -> List[str]:
        """Validate vital signs data."""
        issues = []
        
        for i, vital in enumerate(vitals):
            # Check required fields
            if 'type' not in vital or 'value' not in vital:
                issues.append(f"Vital {i}: missing type or value")
                continue
            
            vital_type = vital['type']
            value = vital['value']
            
            # Validate value is numeric
            if not isinstance(value, (int, float)):
                issues.append(f"Vital {vital_type}: non-numeric value {value}")
                continue
            
            # Check ranges
            if vital_type in self.vital_ranges:
                min_val, max_val = self.vital_ranges[vital_type]
                if value < min_val or value > max_val:
                    issues.append(f"Vital {vital_type}: value {value} outside expected range ({min_val}-{max_val})")
            
            # Check timestamp format
            if 'ts' in vital:
                try:
                    datetime.fromisoformat(vital['ts'].replace('Z', '+00:00'))
                except ValueError:
                    issues.append(f"Vital {vital_type}: invalid timestamp format")
        
        # Check for critical missing vitals
        vital_types = {v.get('type') for v in vitals}
        critical_vitals = ['glucose_fasting', 'hba1c', 'sbp', 'dbp']
        missing_critical = [v for v in critical_vitals if v not in vital_types]
        
        if missing_critical:
            issues.append(f"Missing critical vitals: {', '.join(missing_critical)}")
        
        return issues
    
    def _validate_medications(self, medications: List[Dict[str, Any]]) -> List[str]:
        """Validate medication data."""
        issues = []
        
        for i, med in enumerate(medications):
            # Check required fields
            if 'name' not in med:
                issues.append(f"Medication {i}: missing name")
                continue
            
            # Validate RxNorm code format if present
            if 'rxnorm_code' in med and med['rxnorm_code']:
                rxnorm = med['rxnorm_code']
                if not isinstance(rxnorm, str) or not rxnorm.isdigit():
                    issues.append(f"Medication {med['name']}: invalid RxNorm code format")
            
            # Check dosage format
            if 'dosage' in med and med['dosage']:
                dosage = med['dosage']
                if not isinstance(dosage, str):
                    issues.append(f"Medication {med['name']}: invalid dosage format")
        
        return issues
    
    def _validate_conditions(self, conditions: List[Dict[str, Any]]) -> List[str]:
        """Validate medical conditions data."""
        issues = []
        
        for i, condition in enumerate(conditions):
            # Check required fields
            if 'name' not in condition:
                issues.append(f"Condition {i}: missing name")
                continue
            
            # Validate ICD-10 code format if present
            if 'icd10_code' in condition and condition['icd10_code']:
                icd10 = condition['icd10_code']
                if not isinstance(icd10, str) or len(icd10) < 3:
                    issues.append(f"Condition {condition['name']}: invalid ICD-10 code format")
            
            # Check severity values
            if 'severity' in condition and condition['severity']:
                severity = condition['severity']
                valid_severities = ['mild', 'moderate', 'severe']
                if severity not in valid_severities:
                    issues.append(f"Condition {condition['name']}: invalid severity '{severity}'")
        
        return issues
    
    def _assess_data_completeness(self, context: AgentContext) -> Dict[str, Any]:
        """Assess overall data completeness."""
        demographics = context.patient_profile
        vitals = context.vitals
        medications = context.medications
        conditions = context.conditions
        
        # Demographics completeness
        demo_fields = ['age', 'sex', 'bmi', 'weight_kg', 'height_cm', 'ethnicity']
        demo_present = sum(1 for field in demo_fields if demographics.get(field) is not None)
        demo_completeness = demo_present / len(demo_fields)
        
        # Vital signs completeness
        expected_vitals = ['glucose_fasting', 'hba1c', 'sbp', 'dbp', 'total_cholesterol', 
                          'ldl_cholesterol', 'hdl_cholesterol', 'triglycerides']
        vital_types = {v.get('type') for v in vitals}
        vitals_present = sum(1 for vtype in expected_vitals if vtype in vital_types)
        vitals_completeness = vitals_present / len(expected_vitals)
        
        # Overall completeness score
        overall_completeness = (demo_completeness * 0.3 + vitals_completeness * 0.7)
        
        return {
            'demographics_completeness': demo_completeness,
            'vitals_completeness': vitals_completeness,
            'overall_completeness': overall_completeness,
            'completeness_grade': self._grade_completeness(overall_completeness),
            'has_medications': len(medications) > 0,
            'has_conditions': len(conditions) > 0
        }
    
    def _grade_completeness(self, score: float) -> str:
        """Grade data completeness."""
        if score >= 0.9:
            return 'excellent'
        elif score >= 0.7:
            return 'good'
        elif score >= 0.5:
            return 'fair'
        else:
            return 'poor'
