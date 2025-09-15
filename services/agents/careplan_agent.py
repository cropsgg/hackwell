"""Care plan generation agent using clinical guidelines."""

from typing import Dict, Any, List
from datetime import datetime
import structlog

from base_agent import MockLLMAgent, AgentContext

logger = structlog.get_logger()


class CarePlanAgent(MockLLMAgent):
    """Agent responsible for generating evidence-based care plans."""
    
    def __init__(self):
        super().__init__("careplan_generator", "Evidence-based care plan generation")
        
        # Clinical guidelines and templates
        self.ada_guidelines = self._load_ada_guidelines()
        self.care_templates = self._load_care_templates()
    
    async def process(self, context: AgentContext) -> AgentContext:
        """Generate comprehensive care plan based on risk assessment."""
        try:
            self.log_processing(context, "careplan_generation_started")
            
            # Get risk prediction from previous agent
            risk_output = context.get_agent_output('risk_predictor')
            if not risk_output:
                raise ValueError("Risk prediction required for care plan generation")
            
            # Generate care plan components
            care_plan = await self._generate_care_plan(context, risk_output)
            
            # Validate and refine care plan
            validated_plan = self._validate_care_plan(care_plan, context)
            
            # Add to context
            context.add_agent_output('careplan_generator', {
                'care_plan': validated_plan,
                'generation_metadata': {
                    'risk_category': risk_output.get('risk_category'),
                    'guidelines_used': ['ADA 2024', 'ACC/AHA'],
                    'personalization_factors': self._get_personalization_factors(context),
                    'generated_at': datetime.utcnow().isoformat()
                }
            })
            
            self.log_processing(context, "careplan_generation_completed",
                              plan_components=len(validated_plan))
            
            return context
            
        except Exception as e:
            return self.handle_error(context, e)
    
    async def _generate_care_plan(self, context: AgentContext, 
                                 risk_output: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive care plan."""
        risk_category = risk_output.get('risk_category', 'moderate')
        risk_probability = risk_output.get('risk_probability', 0.2)
        
        # Generate each component of the care plan
        care_plan = {
            'dietary': await self._generate_dietary_plan(context, risk_category),
            'exercise': await self._generate_exercise_plan(context, risk_category),
            'medication_safety': await self._generate_medication_safety(context),
            'monitoring': await self._generate_monitoring_plan(context, risk_category),
            'education': await self._generate_education_plan(context, risk_category),
            'goals': self._generate_goals(context, risk_category, risk_probability)
        }
        
        return care_plan
    
    async def _generate_dietary_plan(self, context: AgentContext, 
                                   risk_category: str) -> Dict[str, Any]:
        """Generate dietary recommendations based on ADA guidelines."""
        patient_profile = context.patient_profile
        conditions = context.conditions
        
        # Base dietary recommendations
        dietary_plan = {
            'recommendations': [],
            'meal_planning': {},
            'targets': {}
        }
        
        # Check for diabetes
        has_diabetes = any('diabetes' in c.get('name', '').lower() 
                          for c in conditions if c.get('active'))
        
        if has_diabetes:
            dietary_plan['recommendations'].extend([
                "Follow a consistent carbohydrate intake pattern to help manage blood glucose",
                "Aim for 45-65% of calories from carbohydrates, emphasizing high-fiber, nutrient-dense sources",
                "Include 25-30g of fiber daily from vegetables, fruits, and whole grains",
                "Limit added sugars to <10% of total daily calories"
            ])
            
            dietary_plan['meal_planning'] = {
                'carb_counting': True,
                'portion_control': True,
                'timing': 'Consistent meal timing to support glucose control'
            }
            
            # Get current HbA1c for personalization
            hba1c = self._get_latest_vital(context.vitals, 'hba1c')
            if hba1c and hba1c > 7.0:
                dietary_plan['recommendations'].append(
                    "Consider more aggressive carbohydrate restriction given elevated HbA1c"
                )
        
        # Check for hypertension
        has_hypertension = any('hypertension' in c.get('name', '').lower() 
                              for c in conditions if c.get('active'))
        
        if has_hypertension:
            dietary_plan['recommendations'].extend([
                "Follow DASH dietary pattern emphasizing fruits, vegetables, and low-fat dairy",
                "Limit sodium intake to <2300mg daily (ideally <1500mg if tolerated)",
                "Increase potassium-rich foods (fruits, vegetables, low-fat dairy)"
            ])
        
        # BMI-based recommendations
        bmi = patient_profile.get('bmi')
        if bmi and bmi > 25:
            dietary_plan['recommendations'].extend([
                "Create a moderate caloric deficit (500-750 calories/day) for gradual weight loss",
                "Focus on nutrient-dense, lower-calorie foods",
                "Consider portion size reduction strategies"
            ])
            
            dietary_plan['targets']['weight_loss'] = "5-10% of current body weight over 6 months"
        
        # Risk-based intensification
        if risk_category == 'high':
            dietary_plan['recommendations'].append(
                "Consider consultation with registered dietitian for intensive nutrition therapy"
            )
        
        return dietary_plan
    
    async def _generate_exercise_plan(self, context: AgentContext, 
                                    risk_category: str) -> Dict[str, Any]:
        """Generate exercise recommendations based on guidelines."""
        patient_profile = context.patient_profile
        conditions = context.conditions
        age = patient_profile.get('age', 50)
        
        exercise_plan = {
            'aerobic': '',
            'resistance': '',
            'flexibility': '',
            'monitoring': '',
            'contraindications': [],
            'progression': {}
        }
        
        # Base ADA recommendations
        exercise_plan['aerobic'] = "150 minutes of moderate-intensity aerobic exercise per week, spread over at least 3 days"
        exercise_plan['resistance'] = "2-3 sessions per week targeting major muscle groups"
        exercise_plan['flexibility'] = "Stretching exercises 2-3 times per week"
        
        # Age-based modifications
        if age > 65:
            exercise_plan['aerobic'] = "150 minutes of moderate-intensity exercise or 75 minutes vigorous, with balance training"
            exercise_plan['contraindications'].append("Screen for cardiovascular disease before vigorous exercise")
        
        # Diabetes-specific recommendations
        has_diabetes = any('diabetes' in c.get('name', '').lower() 
                          for c in conditions if c.get('active'))
        
        if has_diabetes:
            exercise_plan['monitoring'] = "Check blood glucose before and after exercise, especially if on insulin or sulfonylureas"
            exercise_plan['contraindications'].extend([
                "Avoid exercise if blood glucose >300 mg/dL or if ketones present",
                "Use caution if proliferative retinopathy present"
            ])
        
        # Cardiovascular considerations
        has_heart_disease = any('heart' in c.get('name', '').lower() or 'cardiac' in c.get('name', '').lower()
                               for c in conditions if c.get('active'))
        
        if has_heart_disease or risk_category == 'high':
            exercise_plan['contraindications'].append("Consider stress testing before vigorous exercise program")
            exercise_plan['monitoring'] = "Monitor heart rate and symptoms during exercise"
        
        # Progression plan
        current_activity = patient_profile.get('lifestyle', {}).get('exercise_mins_week', 0)
        if current_activity < 150:
            exercise_plan['progression'] = {
                'week_1_2': "Start with 10-15 minutes daily, 3 days per week",
                'week_3_4': "Increase to 20-25 minutes daily, 4 days per week", 
                'week_5_8': "Build to 30 minutes daily, 5 days per week",
                'target': "150 minutes moderate intensity per week"
            }
        
        return exercise_plan
    
    async def _generate_medication_safety(self, context: AgentContext) -> Dict[str, Any]:
        """Generate medication safety recommendations."""
        medications = context.medications
        conditions = context.conditions
        
        safety_plan = {
            'current_regimen': '',
            'monitoring': '',
            'adherence': '',
            'interactions': [],
            'warnings': []
        }
        
        active_meds = [med for med in medications if med.get('active', True)]
        
        if active_meds:
            med_names = [med.get('name', '') for med in active_meds]
            safety_plan['current_regimen'] = f"Continue current medications: {', '.join(med_names)}"
            
            # Diabetes medication specific guidance
            diabetes_meds = ['metformin', 'insulin', 'glipizide', 'glyburide']
            has_diabetes_med = any(any(dm in med.get('name', '').lower() for dm in diabetes_meds) 
                                  for med in active_meds)
            
            if has_diabetes_med:
                safety_plan['monitoring'] = "Monitor blood glucose regularly, especially with medication or lifestyle changes"
                safety_plan['adherence'] = "Take diabetes medications as prescribed with appropriate timing relative to meals"
                safety_plan['warnings'].append("Be aware of hypoglycemia symptoms and treatment")
            
            # Blood pressure medication guidance
            bp_meds = ['lisinopril', 'losartan', 'amlodipine', 'metoprolol']
            has_bp_med = any(any(bm in med.get('name', '').lower() for bm in bp_meds) 
                            for med in active_meds)
            
            if has_bp_med:
                safety_plan['monitoring'] = "Monitor blood pressure regularly at home"
                safety_plan['warnings'].append("Rise slowly from sitting/lying to prevent dizziness")
            
            # Statin guidance
            statin_meds = ['atorvastatin', 'simvastatin', 'rosuvastatin']
            has_statin = any(any(sm in med.get('name', '').lower() for sm in statin_meds) 
                            for med in active_meds)
            
            if has_statin:
                safety_plan['warnings'].append("Report any unexplained muscle pain or weakness")
                safety_plan['monitoring'] = "Periodic liver function and muscle enzyme monitoring"
        
        else:
            safety_plan['current_regimen'] = "No current medications - continue non-pharmacological management"
        
        # General medication safety
        safety_plan['adherence'] = "Use pill organizers, set reminders, and maintain regular pharmacy communication"
        
        return safety_plan
    
    async def _generate_monitoring_plan(self, context: AgentContext, 
                                      risk_category: str) -> Dict[str, Any]:
        """Generate monitoring and follow-up plan."""
        conditions = context.conditions
        patient_profile = context.patient_profile
        
        monitoring_plan = {
            'glucose': '',
            'blood_pressure': '',
            'weight': '',
            'lab_followup': '',
            'appointment_frequency': '',
            'self_monitoring_tools': []
        }
        
        # Diabetes monitoring
        has_diabetes = any('diabetes' in c.get('name', '').lower() 
                          for c in conditions if c.get('active'))
        
        if has_diabetes:
            # Get current HbA1c for frequency determination
            hba1c = self._get_latest_vital(context.vitals, 'hba1c')
            if hba1c and hba1c > 8.0:
                monitoring_plan['glucose'] = "Daily fasting glucose monitoring, target <130 mg/dL"
                monitoring_plan['lab_followup'] = "Repeat HbA1c in 3 months, target <7.0%"
            else:
                monitoring_plan['glucose'] = "Periodic glucose monitoring as directed"
                monitoring_plan['lab_followup'] = "HbA1c every 6 months if stable, target <7.0%"
            
            monitoring_plan['self_monitoring_tools'].append("Blood glucose meter")
        
        # Blood pressure monitoring
        has_hypertension = any('hypertension' in c.get('name', '').lower() 
                              for c in conditions if c.get('active'))
        
        if has_hypertension or risk_category in ['moderate', 'high']:
            monitoring_plan['blood_pressure'] = "Weekly home blood pressure monitoring, target <130/80 mmHg"
            monitoring_plan['self_monitoring_tools'].append("Home blood pressure cuff")
        
        # Weight monitoring
        bmi = patient_profile.get('bmi')
        if bmi and bmi > 25:
            monitoring_plan['weight'] = "Daily weight monitoring, same time each day"
            monitoring_plan['self_monitoring_tools'].append("Digital scale")
        
        # Lipid monitoring
        monitoring_plan['lab_followup'] += " Annual lipid panel, LDL target <100 mg/dL (or <70 mg/dL if high risk)"
        
        # Appointment frequency based on risk
        if risk_category == 'high':
            monitoring_plan['appointment_frequency'] = "Follow-up in 3-6 months, more frequently if unstable"
        elif risk_category == 'moderate':
            monitoring_plan['appointment_frequency'] = "Follow-up every 6 months"
        else:
            monitoring_plan['appointment_frequency'] = "Annual follow-up unless problems arise"
        
        return monitoring_plan
    
    async def _generate_education_plan(self, context: AgentContext, 
                                     risk_category: str) -> Dict[str, Any]:
        """Generate patient education recommendations."""
        conditions = context.conditions
        
        education_plan = {
            'diabetes_self_management': '',
            'lifestyle_modification': '',
            'medication_education': '',
            'complication_prevention': '',
            'resources': []
        }
        
        # Diabetes education
        has_diabetes = any('diabetes' in c.get('name', '').lower() 
                          for c in conditions if c.get('active'))
        
        if has_diabetes:
            education_plan['diabetes_self_management'] = "Enroll in diabetes self-management education (DSMES) program"
            education_plan['complication_prevention'] = "Learn to recognize and manage hypoglycemia; annual eye and foot exams"
            education_plan['resources'].extend([
                "American Diabetes Association (diabetes.org)",
                "Diabetes self-management education classes",
                "Certified diabetes educator consultation"
            ])
        
        # General lifestyle education
        education_plan['lifestyle_modification'] = "Nutrition counseling and exercise guidance for cardiometabolic health"
        
        # Medication education
        education_plan['medication_education'] = "Understand purpose, timing, and side effects of all medications"
        
        # Risk-specific education
        if risk_category == 'high':
            education_plan['resources'].append("Intensive lifestyle intervention program")
            education_plan['complication_prevention'] = "Aggressive risk factor modification education"
        
        # General resources
        education_plan['resources'].extend([
            "Heart-healthy cooking classes",
            "Local exercise programs",
            "Blood pressure monitoring training"
        ])
        
        return education_plan
    
    def _generate_goals(self, context: AgentContext, risk_category: str, 
                       risk_probability: float) -> Dict[str, Any]:
        """Generate short-term and long-term goals."""
        conditions = context.conditions
        patient_profile = context.patient_profile
        
        goals = {
            'short_term': [],  # 1-3 months
            'long_term': [],   # 6-12 months
            'priorities': []
        }
        
        # Diabetes goals
        has_diabetes = any('diabetes' in c.get('name', '').lower() 
                          for c in conditions if c.get('active'))
        
        if has_diabetes:
            hba1c = self._get_latest_vital(context.vitals, 'hba1c')
            if hba1c and hba1c > 7.0:
                goals['short_term'].append(f"Improve HbA1c from {hba1c:.1f}% to <7.0%")
                goals['priorities'].append("Glucose control")
            
            goals['long_term'].append("Maintain HbA1c <7.0% consistently")
        
        # Blood pressure goals
        sbp = self._get_latest_vital(context.vitals, 'sbp')
        if sbp and sbp > 130:
            goals['short_term'].append(f"Lower blood pressure from {sbp:.0f} to <130/80 mmHg")
            goals['priorities'].append("Blood pressure control")
        
        # Weight goals
        bmi = patient_profile.get('bmi')
        if bmi and bmi > 25:
            target_loss = round(patient_profile.get('weight_kg', 70) * 0.07, 1)  # 7% loss
            goals['short_term'].append(f"Lose {target_loss} kg (5-10% body weight)")
            goals['long_term'].append("Maintain weight loss and healthy BMI")
            goals['priorities'].append("Weight management")
        
        # Exercise goals
        current_exercise = patient_profile.get('lifestyle', {}).get('exercise_mins_week', 0)
        if current_exercise < 150:
            goals['short_term'].append("Achieve 150 minutes moderate exercise per week")
            goals['long_term'].append("Maintain regular exercise routine")
            goals['priorities'].append("Physical activity")
        
        # Risk reduction goal
        target_risk_reduction = 0.05  # 5% absolute risk reduction
        target_risk = max(0.05, risk_probability - target_risk_reduction)
        goals['long_term'].append(f"Reduce cardiovascular risk to {target_risk:.1%} through comprehensive management")
        
        return goals
    
    def _validate_care_plan(self, care_plan: Dict[str, Any], 
                           context: AgentContext) -> Dict[str, Any]:
        """Validate and refine the generated care plan."""
        # Check for contraindications and conflicts
        validated_plan = care_plan.copy()
        
        # Add summary and priorities
        validated_plan['summary'] = self._generate_plan_summary(care_plan, context)
        validated_plan['priorities'] = care_plan.get('goals', {}).get('priorities', [])
        
        return validated_plan
    
    def _generate_plan_summary(self, care_plan: Dict[str, Any], 
                              context: AgentContext) -> str:
        """Generate executive summary of the care plan."""
        risk_output = context.get_agent_output('risk_predictor', {})
        risk_category = risk_output.get('risk_category', 'moderate')
        
        summary = f"Comprehensive care plan for {risk_category} cardiometabolic risk management. "
        
        priorities = care_plan.get('goals', {}).get('priorities', [])
        if priorities:
            summary += f"Priority areas include: {', '.join(priorities)}. "
        
        summary += "Plan emphasizes evidence-based lifestyle modifications, appropriate monitoring, and patient education."
        
        return summary
    
    def _get_latest_vital(self, vitals: List[Dict[str, Any]], vital_type: str) -> float:
        """Get the latest value for a specific vital type."""
        matching_vitals = [v for v in vitals if v.get('type') == vital_type]
        if matching_vitals:
            # Sort by timestamp and get most recent
            latest = sorted(matching_vitals, key=lambda x: x.get('ts', ''), reverse=True)[0]
            return latest.get('value')
        return None
    
    def _get_personalization_factors(self, context: AgentContext) -> List[str]:
        """Get factors used for care plan personalization."""
        factors = []
        
        # Demographics
        age = context.patient_profile.get('age')
        if age:
            if age > 65:
                factors.append('elderly_considerations')
            elif age < 40:
                factors.append('young_adult_focus')
        
        # Conditions
        condition_names = [c.get('name', '').lower() for c in context.conditions if c.get('active')]
        if any('diabetes' in name for name in condition_names):
            factors.append('diabetes_management')
        if any('hypertension' in name for name in condition_names):
            factors.append('blood_pressure_focus')
        
        # Risk level
        risk_output = context.get_agent_output('risk_predictor', {})
        risk_category = risk_output.get('risk_category')
        if risk_category:
            factors.append(f'{risk_category}_risk_intensity')
        
        return factors
    
    def _load_ada_guidelines(self) -> Dict[str, Any]:
        """Load ADA clinical guidelines (simplified for demo)."""
        return {
            'hba1c_target': 7.0,
            'bp_target': '130/80',
            'ldl_target': 100,
            'exercise_recommendation': '150_min_moderate',
            'carb_percentage': '45-65'
        }
    
    def _load_care_templates(self) -> Dict[str, Any]:
        """Load care plan templates (simplified for demo)."""
        return {
            'diabetes': 'comprehensive_diabetes_management',
            'hypertension': 'blood_pressure_control',
            'obesity': 'weight_management',
            'dyslipidemia': 'lipid_management'
        }
