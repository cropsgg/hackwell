"""Authentication and authorization Pydantic models."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, EmailStr, validator, ConfigDict
from uuid import UUID

from .common import BaseResponse, UserRole, TimestampMixin


class TokenData(BaseModel):
    """JWT token data."""
    sub: str  # User ID
    email: Optional[str] = None
    roles: List[str] = Field(default_factory=list)
    exp: Optional[datetime] = None
    iat: Optional[datetime] = None
    iss: Optional[str] = None
    aud: Optional[str] = None


class Token(BaseModel):
    """Access token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # seconds
    refresh_token: Optional[str] = None


class TokenResponse(BaseResponse):
    """Token response wrapper."""
    data: Token


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    remember_me: bool = False


class UserRegistration(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    confirm_password: str
    role: UserRole = UserRole.PATIENT
    
    # Profile information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    
    # Role-specific metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator('password')
    def validate_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        
        # Check for at least one digit, one letter, one special character
        has_digit = any(c.isdigit() for c in v)
        has_letter = any(c.isalpha() for c in v)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v)
        
        if not (has_digit and has_letter and has_special):
            raise ValueError(
                'Password must contain at least one digit, one letter, and one special character'
            )
        
        return v


class PasswordReset(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordChange(BaseModel):
    """Password change request."""
    current_password: str
    new_password: str = Field(..., min_length=8)
    confirm_password: str
    
    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class UserProfile(BaseModel):
    """User profile information."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    language: str = "en"
    
    # Notification preferences
    email_notifications: bool = True
    sms_notifications: bool = False
    push_notifications: bool = True
    
    # Privacy settings
    profile_visibility: str = "private"  # public, private, contacts
    data_sharing_consent: bool = False


class UserProfileUpdate(BaseModel):
    """User profile update request."""
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    language: Optional[str] = None
    email_notifications: Optional[bool] = None
    sms_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    profile_visibility: Optional[str] = None
    data_sharing_consent: Optional[bool] = None


class User(TimestampMixin):
    """User model."""
    id: UUID
    email: str
    roles: List[UserRole]
    profile: Optional[UserProfile] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Account status
    is_active: bool = True
    is_verified: bool = False
    last_login: Optional[datetime] = None
    
    # Security
    password_changed_at: Optional[datetime] = None
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserResponse(BaseResponse):
    """User response wrapper."""
    data: User


class UserListResponse(BaseResponse):
    """User list response."""
    data: List[User]
    total: int


class RoleAssignment(BaseModel):
    """Role assignment request."""
    user_id: UUID
    role: UserRole
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RoleRevocation(BaseModel):
    """Role revocation request."""
    user_id: UUID
    role: UserRole


class PermissionCheck(BaseModel):
    """Permission check request."""
    user_id: UUID
    resource: str
    action: str
    resource_id: Optional[str] = None


class PermissionCheckResponse(BaseResponse):
    """Permission check response."""
    data: Dict[str, bool]


class AuditLogEntry(TimestampMixin):
    """Audit log entry model."""
    id: UUID
    actor_type: str
    actor_id: Optional[UUID] = None
    event: str
    entity_type: Optional[str] = None
    entity_id: Optional[UUID] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    payload_hash: Optional[str] = None
    model_version: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True)


class AuditLogResponse(BaseResponse):
    """Audit log response wrapper."""
    data: List[AuditLogEntry]
    total: int


class SessionInfo(BaseModel):
    """User session information."""
    session_id: str
    user_id: UUID
    created_at: datetime
    last_activity: datetime
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    is_active: bool = True


class SessionResponse(BaseResponse):
    """Session response wrapper."""
    data: SessionInfo


class MFASetup(BaseModel):
    """Multi-factor authentication setup."""
    method: str  # totp, sms, email
    phone: Optional[str] = None  # Required for SMS
    backup_codes: Optional[List[str]] = None


class MFAVerification(BaseModel):
    """MFA verification request."""
    code: str = Field(..., min_length=6, max_length=8)
    method: str


class MFAResponse(BaseResponse):
    """MFA response wrapper."""
    data: Dict[str, Any]


class ApiKey(TimestampMixin):
    """API key model."""
    id: UUID
    user_id: UUID
    name: str
    key_hash: str
    permissions: List[str] = Field(default_factory=list)
    is_active: bool = True
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class ApiKeyCreate(BaseModel):
    """API key creation request."""
    name: str = Field(..., max_length=100)
    permissions: List[str] = Field(default_factory=list)
    expires_in_days: Optional[int] = Field(None, gt=0, le=365)


class ApiKeyResponse(BaseResponse):
    """API key response wrapper."""
    data: Dict[str, Any]  # Includes the plain text key only on creation


# OAuth and Social Login
class OAuthProvider(BaseModel):
    """OAuth provider configuration."""
    provider: str  # google, github, etc.
    client_id: str
    redirect_uri: str
    scope: Optional[str] = None


class OAuthCallback(BaseModel):
    """OAuth callback data."""
    provider: str
    code: str
    state: Optional[str] = None


class SocialProfile(BaseModel):
    """Social login profile."""
    provider: str
    provider_id: str
    email: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None


# Account verification
class EmailVerification(BaseModel):
    """Email verification request."""
    token: str


class VerificationResponse(BaseResponse):
    """Verification response."""
    data: Dict[str, str]


# Rate limiting and security
class SecurityEvent(BaseModel):
    """Security event model."""
    event_type: str  # login_failed, password_reset, etc.
    user_id: Optional[UUID] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    severity: str = "info"  # info, warning, critical
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RateLimit(BaseModel):
    """Rate limit status."""
    limit: int
    remaining: int
    reset_time: datetime
    window_seconds: int
