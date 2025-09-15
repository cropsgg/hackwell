"""Patient-related Pydantic models."""

from datetime import datetime, date
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, ConfigDict
from uuid import UUID

from .common import (
    BaseResponse, TimestampMixin, VitalType, VitalUnit, 
    DataSource, RiskCategory, CommonValidators, Metadata
)


class Demographics(BaseModel):
    """Patient demographics."""
    name: Optional[str] = None
    age: int = Field(..., ge=0, le=150)
    sex: str = Field(..., regex="^(M|F|O)$")  # Male, Female, Other
    ethnicity: Optional[str] = None
    height_cm: Optional[float] = Field(None, ge=50, le=300)
    weight_kg: Optional[float] = Field(None, ge=1, le=500)
    bmi: Optional[float] = Field(None, ge=10, le=80)
    
    # Family history
    family_history: Optional[Dict[str, bool]] = Field(default_factory=dict)
    
    # Lifestyle factors
    lifestyle: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('age')
    def validate_age(cls, v):
        return CommonValidators.validate_age(v)
    
    @validator('bmi')
    def validate_bmi(cls, v):
        if v is not None:
            return CommonValidators.validate_bmi(v)
        return v


class ConsentInfo(BaseModel):
    """Patient consent information."""
    ehr_access: bool = False
    data_sharing: bool = False
    research_participation: bool = False
    marketing: bool = False
    
    # Granular permissions
    permissions: Optional[Dict[str, bool]] = Field(default_factory=dict)
    
    # Consent metadata
    consent_date: Optional[datetime] = None
    consent_version: Optional[str] = None


class PatientCreate(BaseModel):
    """Patient creation schema."""
    demographics: Demographics
    consent: Optional[ConsentInfo] = None


class PatientUpdate(BaseModel):
    """Patient update schema."""
    demographics: Optional[Demographics] = None
    consent: Optional[ConsentInfo] = None


class Patient(TimestampMixin):
    """Patient model."""
    id: UUID
    user_id: UUID
    demographics: Demographics
    consent: ConsentInfo = Field(default_factory=ConsentInfo)
    
    model_config = ConfigDict(from_attributes=True)


class PatientResponse(BaseResponse):
    """Patient response wrapper."""
    data: Patient


class PatientListResponse(BaseResponse):
    """Patient list response."""
    data: List[Patient]
    total: int
    offset: int
    limit: int


# Vital Signs Models
class VitalValue(BaseModel):
    """Individual vital sign measurement."""
    type: VitalType
    value: float
    unit: VitalUnit
    source: DataSource = DataSource.MANUAL
    ts: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('value')
    def validate_vital_value(cls, v, values):
        """Validate vital values based on type."""
        vital_type = values.get('type')
        
        if vital_type in [VitalType.GLUCOSE_FASTING, VitalType.GLUCOSE_RANDOM]:
            return CommonValidators.validate_glucose(v)
        elif vital_type in [VitalType.SBP, VitalType.DBP]:
            return CommonValidators.validate_blood_pressure(int(v))
        elif vital_type in [
            VitalType.TOTAL_CHOLESTEROL, 
            VitalType.LDL_CHOLESTEROL, 
            VitalType.HDL_CHOLESTEROL
        ]:
            return CommonValidators.validate_cholesterol(v)
        elif vital_type == VitalType.HBA1C:
            if not 3.0 <= v <= 18.0:
                raise ValueError("HbA1c must be between 3.0 and 18.0%")
        elif vital_type == VitalType.BMI:
            return CommonValidators.validate_bmi(v)
        
        return v


class VitalCreate(BaseModel):
    """Vital sign creation schema."""
    patient_id: UUID
    type: VitalType
    value: float
    unit: VitalUnit
    source: DataSource = DataSource.MANUAL
    ts: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class VitalUpdate(BaseModel):
    """Vital sign update schema."""
    value: Optional[float] = None
    unit: Optional[VitalUnit] = None
    source: Optional[DataSource] = None
    metadata: Optional[Dict[str, Any]] = None


class Vital(TimestampMixin):
    """Vital sign model."""
    id: UUID
    patient_id: UUID
    type: VitalType
    value: float
    unit: VitalUnit
    source: DataSource
    ts: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(from_attributes=True)


class VitalResponse(BaseResponse):
    """Vital sign response wrapper."""
    data: Vital


class VitalListResponse(BaseResponse):
    """Vital signs list response."""
    data: List[Vital]
    total: int


# Medical Conditions Models
class ConditionCreate(BaseModel):
    """Medical condition creation schema."""
    patient_id: UUID
    icd10_code: Optional[str] = None
    name: str
    severity: Optional[str] = None
    onset_date: Optional[date] = None
    active: bool = True


class ConditionUpdate(BaseModel):
    """Medical condition update schema."""
    icd10_code: Optional[str] = None
    name: Optional[str] = None
    severity: Optional[str] = None
    onset_date: Optional[date] = None
    active: Optional[bool] = None


class Condition(TimestampMixin):
    """Medical condition model."""
    id: UUID
    patient_id: UUID
    icd10_code: Optional[str] = None
    name: str
    severity: Optional[str] = None
    onset_date: Optional[date] = None
    active: bool = True
    
    model_config = ConfigDict(from_attributes=True)


class ConditionResponse(BaseResponse):
    """Condition response wrapper."""
    data: Condition


class ConditionListResponse(BaseResponse):
    """Conditions list response."""
    data: List[Condition]


# Medications Models
class MedicationSchedule(BaseModel):
    """Medication dosing schedule."""
    frequency: str  # once_daily, twice_daily, etc.
    times: List[str] = Field(default_factory=list)  # ["08:00", "20:00"]
    with_food: Optional[bool] = None
    special_instructions: Optional[str] = None


class MedicationCreate(BaseModel):
    """Medication creation schema."""
    patient_id: UUID
    rxnorm_code: Optional[str] = None
    name: str
    dosage: Optional[str] = None
    schedule: Optional[MedicationSchedule] = None
    active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class MedicationUpdate(BaseModel):
    """Medication update schema."""
    rxnorm_code: Optional[str] = None
    name: Optional[str] = None
    dosage: Optional[str] = None
    schedule: Optional[MedicationSchedule] = None
    active: Optional[bool] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None


class Medication(TimestampMixin):
    """Medication model."""
    id: UUID
    patient_id: UUID
    rxnorm_code: Optional[str] = None
    name: str
    dosage: Optional[str] = None
    schedule: Optional[MedicationSchedule] = None
    active: bool = True
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    
    model_config = ConfigDict(from_attributes=True)


class MedicationResponse(BaseResponse):
    """Medication response wrapper."""
    data: Medication


class MedicationListResponse(BaseResponse):
    """Medications list response."""
    data: List[Medication]


# Risk Assessment Models
class RiskFactors(BaseModel):
    """Key risk factors contributing to assessment."""
    factor: str
    value: Optional[str] = None
    impact: str  # low, moderate, high
    target: Optional[str] = None
    contribution: Optional[float] = None  # SHAP value


class RiskAssessment(BaseModel):
    """Patient risk assessment."""
    risk_score: float = Field(..., ge=0, le=1)
    risk_category: RiskCategory
    model_version: str
    confidence: Optional[float] = Field(None, ge=0, le=1)
    
    # Key contributing factors
    key_factors: List[RiskFactors] = Field(default_factory=list)
    
    # Model explanation
    interpretation: Optional[str] = None
    recommendations_summary: Optional[str] = None
    
    # Temporal context
    assessment_date: datetime = Field(default_factory=datetime.utcnow)
    next_assessment_due: Optional[datetime] = None


class RiskAssessmentResponse(BaseResponse):
    """Risk assessment response wrapper."""
    data: RiskAssessment


# Patient Summary Models
class VitalTrend(BaseModel):
    """Vital sign trend summary."""
    type: VitalType
    latest_value: float
    unit: VitalUnit
    trend: str  # improving, stable, worsening
    change_percent: Optional[float] = None
    last_measured: datetime


class PatientSummary(BaseModel):
    """Comprehensive patient summary."""
    patient: Patient
    risk_assessment: Optional[RiskAssessment] = None
    vital_trends: List[VitalTrend] = Field(default_factory=list)
    active_conditions: List[Condition] = Field(default_factory=list)
    active_medications: List[Medication] = Field(default_factory=list)
    
    # Statistics
    total_vitals: int = 0
    total_recommendations: int = 0
    last_activity: Optional[datetime] = None


class PatientSummaryResponse(BaseResponse):
    """Patient summary response wrapper."""
    data: PatientSummary


# Bulk operations
class BulkVitalCreate(BaseModel):
    """Bulk vital signs creation."""
    patient_id: UUID
    vitals: List[VitalValue]


class BulkVitalResponse(BaseResponse):
    """Bulk vital creation response."""
    data: List[Vital]
    created_count: int
    failed_count: int
    errors: List[str] = Field(default_factory=list)


# Search and filtering
class PatientFilter(BaseModel):
    """Patient filtering parameters."""
    risk_category: Optional[RiskCategory] = None
    age_min: Optional[int] = Field(None, ge=0)
    age_max: Optional[int] = Field(None, le=150)
    sex: Optional[str] = None
    has_condition: Optional[str] = None
    has_medication: Optional[str] = None
    last_activity_days: Optional[int] = None


class VitalFilter(BaseModel):
    """Vital signs filtering parameters."""
    types: Optional[List[VitalType]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    source: Optional[DataSource] = None
    limit: int = Field(default=100, le=1000)
