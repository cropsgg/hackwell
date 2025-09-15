"""Recommendation and care plan Pydantic models."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, validator, ConfigDict
from uuid import UUID

from .common import (
    BaseResponse, TimestampMixin, RecommendationStatus, 
    EvidenceSourceType, RiskCategory, validate_probability
)


# Care Plan Components
class DietaryRecommendation(BaseModel):
    """Dietary care plan component."""
    recommendations: List[str] = Field(default_factory=list)
    meal_planning: Optional[Dict[str, Any]] = Field(default_factory=dict)
    calorie_target: Optional[int] = None
    carb_target_grams: Optional[int] = None
    fiber_target_grams: Optional[int] = None
    sodium_limit_mg: Optional[int] = None


class ExerciseRecommendation(BaseModel):
    """Exercise care plan component."""
    aerobic: Optional[str] = None
    resistance: Optional[str] = None
    flexibility: Optional[str] = None
    monitoring: Optional[str] = None
    contraindications: List[str] = Field(default_factory=list)
    target_heart_rate: Optional[int] = None


class MedicationSafetyRecommendation(BaseModel):
    """Medication safety component."""
    current_regimen: Optional[str] = None
    monitoring: Optional[str] = None
    adherence: Optional[str] = None
    interactions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)


class MonitoringRecommendation(BaseModel):
    """Monitoring and follow-up component."""
    glucose: Optional[str] = None
    blood_pressure: Optional[str] = None
    weight: Optional[str] = None
    lab_followup: Optional[str] = None
    appointment_frequency: Optional[str] = None
    self_monitoring_tools: List[str] = Field(default_factory=list)


class EducationRecommendation(BaseModel):
    """Patient education component."""
    diabetes_self_management: Optional[str] = None
    hypoglycemia_recognition: Optional[str] = None
    lifestyle_modification: Optional[str] = None
    medication_education: Optional[str] = None
    resources: List[str] = Field(default_factory=list)


class CarePlan(BaseModel):
    """Comprehensive care plan."""
    dietary: Optional[DietaryRecommendation] = None
    exercise: Optional[ExerciseRecommendation] = None
    medication_safety: Optional[MedicationSafetyRecommendation] = None
    monitoring: Optional[MonitoringRecommendation] = None
    education: Optional[EducationRecommendation] = None
    
    # Summary and priorities
    summary: Optional[str] = None
    priorities: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    
    # Timeline
    short_term_goals: List[str] = Field(default_factory=list)  # 1-3 months
    long_term_goals: List[str] = Field(default_factory=list)   # 6-12 months


# Model Explanation Components
class SHAPExplanation(BaseModel):
    """SHAP feature contribution."""
    feature: str
    value: Optional[str] = None
    contribution: float  # SHAP value
    impact: str  # positive, negative, neutral
    interpretation: Optional[str] = None


class ModelExplanation(BaseModel):
    """Model prediction explanation."""
    risk_interpretation: Optional[str] = None
    key_contributors: List[SHAPExplanation] = Field(default_factory=list)
    model_confidence: Optional[float] = Field(None, ge=0, le=1)
    evidence_strength: Optional[str] = None
    
    # Patient-friendly explanation
    patient_summary: Optional[str] = None
    clinician_notes: Optional[str] = None
    
    @validator('model_confidence')
    def validate_confidence(cls, v):
        if v is not None:
            return validate_probability(v)
        return v


# Evidence and Verification
class EvidenceLink(BaseModel):
    """Evidence supporting recommendation."""
    source_type: EvidenceSourceType
    url: Optional[str] = None
    title: Optional[str] = None
    weight: float = Field(default=0.0, ge=0, le=1)
    snippet: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    # Quality indicators
    study_type: Optional[str] = None  # RCT, cohort, case-control, etc.
    evidence_level: Optional[str] = None  # A, B, C
    sample_size: Optional[int] = None
    publication_year: Optional[int] = None


class EvidenceVerification(BaseModel):
    """Evidence verification result."""
    status: str  # approved, flagged, rejected
    overall_score: float = Field(..., ge=0, le=1)
    evidence_links: List[EvidenceLink] = Field(default_factory=list)
    
    # Scoring breakdown
    guideline_score: Optional[float] = Field(None, ge=0, le=1)
    literature_score: Optional[float] = Field(None, ge=0, le=1)
    safety_score: Optional[float] = Field(None, ge=0, le=1)
    
    # Flags and warnings
    flags: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    # Verification metadata
    verified_at: datetime = Field(default_factory=datetime.utcnow)
    verification_version: Optional[str] = None


# Main Recommendation Models
class RecommendationCreate(BaseModel):
    """Recommendation creation schema."""
    patient_id: UUID
    careplan: CarePlan
    explainer: Optional[ModelExplanation] = None
    model_version: Optional[str] = None
    risk_score: Optional[float] = Field(None, ge=0, le=1)
    risk_category: Optional[RiskCategory] = None
    
    # Optional pre-computed evidence
    evidence_verification: Optional[EvidenceVerification] = None
    
    @validator('risk_score')
    def validate_risk_score(cls, v):
        if v is not None:
            return validate_probability(v)
        return v


class RecommendationUpdate(BaseModel):
    """Recommendation update schema."""
    careplan: Optional[CarePlan] = None
    explainer: Optional[ModelExplanation] = None
    status: Optional[RecommendationStatus] = None
    
    # Clinician action fields
    clinician_notes: Optional[str] = None
    justification: Optional[str] = None


class Recommendation(TimestampMixin):
    """Recommendation model."""
    id: UUID
    patient_id: UUID
    snapshot_ts: datetime
    careplan: CarePlan
    explainer: Optional[ModelExplanation] = None
    model_version: Optional[str] = None
    status: RecommendationStatus = RecommendationStatus.PENDING
    risk_score: Optional[float] = None
    risk_category: Optional[RiskCategory] = None
    
    # Approval workflow
    created_by_user_id: Optional[UUID] = None
    approved_by_user_id: Optional[UUID] = None
    approved_at: Optional[datetime] = None
    
    # Clinician interaction
    clinician_notes: Optional[str] = None
    justification: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class RecommendationWithEvidence(Recommendation):
    """Recommendation with evidence links."""
    evidence_verification: Optional[EvidenceVerification] = None
    evidence_links: List[EvidenceLink] = Field(default_factory=list)


class RecommendationResponse(BaseResponse):
    """Recommendation response wrapper."""
    data: RecommendationWithEvidence


class RecommendationListResponse(BaseResponse):
    """Recommendations list response."""
    data: List[RecommendationWithEvidence]
    total: int


# Clinician Actions
class ClinicianAction(BaseModel):
    """Clinician action on recommendation."""
    action: str  # approve, reject, override, flag
    justification: Optional[str] = None
    notes: Optional[str] = None
    
    # Override-specific fields
    override_careplan: Optional[CarePlan] = None
    override_reasoning: Optional[str] = None


class ClinicianActionResponse(BaseResponse):
    """Clinician action response."""
    data: Dict[str, Any]
    audit_log_id: UUID


# Recommendation Generation Request
class RecommendationRequest(BaseModel):
    """Request to generate new recommendation."""
    patient_id: UUID
    force_refresh: bool = False
    include_evidence: bool = True
    model_version: Optional[str] = None
    
    # Context parameters
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    priorities: Optional[List[str]] = None


class RecommendationGenerationResponse(BaseResponse):
    """Recommendation generation response."""
    data: RecommendationWithEvidence
    generation_metadata: Dict[str, Any]


# Batch Operations
class BulkRecommendationRequest(BaseModel):
    """Bulk recommendation generation."""
    patient_ids: List[UUID]
    model_version: Optional[str] = None
    include_evidence: bool = True


class BulkRecommendationResponse(BaseResponse):
    """Bulk recommendation response."""
    data: List[RecommendationWithEvidence]
    success_count: int
    failed_count: int
    errors: List[str] = Field(default_factory=list)


# Analytics and Reporting
class RecommendationStats(BaseModel):
    """Recommendation statistics."""
    total_recommendations: int
    by_status: Dict[str, int]
    by_risk_category: Dict[str, int]
    by_model_version: Dict[str, int]
    
    # Time-based stats
    created_last_30_days: int
    approved_last_30_days: int
    
    # Performance metrics
    avg_approval_time_hours: Optional[float] = None
    evidence_score_avg: Optional[float] = None


class PatientRecommendationHistory(BaseModel):
    """Patient recommendation history."""
    patient_id: UUID
    recommendations: List[RecommendationWithEvidence]
    stats: RecommendationStats
    
    # Trends
    risk_trend: List[Dict[str, Any]] = Field(default_factory=list)
    compliance_indicators: Optional[Dict[str, Any]] = None


# Search and Filtering
class RecommendationFilter(BaseModel):
    """Recommendation filtering parameters."""
    status: Optional[RecommendationStatus] = None
    risk_category: Optional[RiskCategory] = None
    model_version: Optional[str] = None
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    
    # Evidence filtering
    min_evidence_score: Optional[float] = Field(None, ge=0, le=1)
    has_flags: Optional[bool] = None
    
    # Clinician filtering
    created_by: Optional[UUID] = None
    approved_by: Optional[UUID] = None
    pending_approval: Optional[bool] = None


# WebSocket Updates
class RecommendationUpdate(BaseModel):
    """Real-time recommendation update."""
    recommendation_id: UUID
    patient_id: UUID
    event_type: str  # created, updated, approved, rejected
    status: RecommendationStatus
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Optional event data
    data: Optional[Dict[str, Any]] = None
    user_id: Optional[UUID] = None
