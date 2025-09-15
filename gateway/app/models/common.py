"""Common Pydantic models and schemas."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class BaseResponse(BaseModel):
    """Base response model."""
    success: bool = True
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = False
    error: str
    code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str
    environment: str
    services: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationParams(BaseModel):
    """Pagination parameters."""
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=100)


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    data: List[Any]
    total: int
    offset: int
    limit: int
    has_more: bool = False


class RiskCategory(str, Enum):
    """Risk category enumeration."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    UNKNOWN = "unknown"


class RecommendationStatus(str, Enum):
    """Recommendation status enumeration."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FLAGGED = "flagged"


class EvidenceSourceType(str, Enum):
    """Evidence source type enumeration."""
    GUIDELINE = "guideline"
    PUBMED = "pubmed"
    OPENFDA = "openfda"
    RXNORM = "rxnorm"
    ADA = "ada"


class UserRole(str, Enum):
    """User role enumeration."""
    PATIENT = "patient"
    CLINICIAN = "clinician"
    ADMIN = "admin"


class VitalType(str, Enum):
    """Vital signs type enumeration."""
    GLUCOSE_FASTING = "glucose_fasting"
    GLUCOSE_RANDOM = "glucose_random"
    HBA1C = "hba1c"
    SBP = "sbp"  # Systolic blood pressure
    DBP = "dbp"  # Diastolic blood pressure
    HEART_RATE = "heart_rate"
    WEIGHT = "weight"
    HEIGHT = "height"
    BMI = "bmi"
    TOTAL_CHOLESTEROL = "total_cholesterol"
    LDL_CHOLESTEROL = "ldl_cholesterol"
    HDL_CHOLESTEROL = "hdl_cholesterol"
    TRIGLYCERIDES = "triglycerides"
    TEMPERATURE = "temperature"
    OXYGEN_SATURATION = "oxygen_saturation"


class VitalUnit(str, Enum):
    """Vital signs units enumeration."""
    MG_DL = "mg/dL"
    MMOL_L = "mmol/L"
    PERCENT = "%"
    MMHG = "mmHg"
    BPM = "bpm"
    KG = "kg"
    LB = "lb"
    CM = "cm"
    IN = "in"
    CELSIUS = "°C"
    FAHRENHEIT = "°F"


class DataSource(str, Enum):
    """Data source enumeration."""
    MANUAL = "manual"
    DEVICE = "device"
    EHR = "ehr"
    LAB = "lab"
    IMPORT = "import"


class ModelAlgorithm(str, Enum):
    """ML model algorithm enumeration."""
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    RANDOM_FOREST = "random_forest"
    LOGISTIC_REGRESSION = "logistic_regression"


# Generic metadata model
class Metadata(BaseModel):
    """Generic metadata container."""
    model_config = ConfigDict(extra="allow")
    
    def get(self, key: str, default=None):
        """Get metadata value with default."""
        return getattr(self, key, default)


# Timestamp mixin
class TimestampMixin(BaseModel):
    """Mixin for timestamp fields."""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None


# Validation helpers
def validate_uuid(value: str) -> str:
    """Validate UUID format."""
    import uuid
    try:
        uuid.UUID(value)
        return value
    except ValueError:
        raise ValueError("Invalid UUID format")


def validate_percentage(value: float) -> float:
    """Validate percentage value (0-100)."""
    if not 0 <= value <= 100:
        raise ValueError("Percentage must be between 0 and 100")
    return value


def validate_probability(value: float) -> float:
    """Validate probability value (0-1)."""
    if not 0 <= value <= 1:
        raise ValueError("Probability must be between 0 and 1")
    return value


# Common field validators
class CommonValidators:
    """Common field validation functions."""
    
    @staticmethod
    def validate_age(age: int) -> int:
        """Validate age value."""
        if not 0 <= age <= 150:
            raise ValueError("Age must be between 0 and 150")
        return age
    
    @staticmethod
    def validate_bmi(bmi: float) -> float:
        """Validate BMI value."""
        if not 10 <= bmi <= 80:
            raise ValueError("BMI must be between 10 and 80")
        return bmi
    
    @staticmethod
    def validate_glucose(glucose: float) -> float:
        """Validate glucose value (mg/dL)."""
        if not 20 <= glucose <= 800:
            raise ValueError("Glucose must be between 20 and 800 mg/dL")
        return glucose
    
    @staticmethod
    def validate_blood_pressure(pressure: int) -> int:
        """Validate blood pressure value."""
        if not 40 <= pressure <= 300:
            raise ValueError("Blood pressure must be between 40 and 300 mmHg")
        return pressure
    
    @staticmethod
    def validate_cholesterol(cholesterol: float) -> float:
        """Validate cholesterol value (mg/dL)."""
        if not 50 <= cholesterol <= 500:
            raise ValueError("Cholesterol must be between 50 and 500 mg/dL")
        return cholesterol


# API versioning
class APIVersion(str, Enum):
    """API version enumeration."""
    V1 = "v1"
    V2 = "v2"


# Request/Response metadata
class RequestMetadata(BaseModel):
    """Request metadata for tracking."""
    request_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    session_id: Optional[str] = None


class ResponseMetadata(BaseModel):
    """Response metadata for tracking."""
    request_id: Optional[str] = None
    processing_time_ms: Optional[float] = None
    model_version: Optional[str] = None
    cache_hit: bool = False
