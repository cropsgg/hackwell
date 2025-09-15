"""Security and authentication utilities."""

import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import structlog

from .config import settings
from .db import db_manager

logger = structlog.get_logger()

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token handler
security = HTTPBearer()


class SecurityManager:
    """Centralized security and authentication management."""
    
    def __init__(self):
        self.secret_key = settings.secret_key
        self.algorithm = settings.algorithm
        self.access_token_expire_minutes = settings.access_token_expire_minutes
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience
        })
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm],
                audience=settings.jwt_audience,
                issuer=settings.jwt_issuer
            )
            return payload
        except InvalidTokenError as e:
            logger.warning("Token verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)


# Global security manager
security_manager = SecurityManager()


class User:
    """User model for authentication."""
    
    def __init__(self, id: str, email: str, roles: List[str], metadata: Optional[Dict] = None):
        self.id = id
        self.email = email
        self.roles = roles
        self.metadata = metadata or {}
    
    def has_role(self, role: str) -> bool:
        """Check if user has specific role."""
        return role in self.roles
    
    def can_access_patient(self, patient_id: str) -> bool:
        """Check if user can access specific patient data."""
        # Patients can access their own data
        if self.has_role("patient") and self.metadata.get("patient_id") == patient_id:
            return True
        
        # Clinicians can access assigned patients
        if self.has_role("clinician"):
            assigned_patients = self.metadata.get("assigned_patients", [])
            return patient_id in assigned_patients
        
        # Admins can access all patients
        if self.has_role("admin"):
            return True
        
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "roles": self.roles,
            "metadata": self.metadata
        }


async def get_user_from_token(token_payload: Dict[str, Any]) -> User:
    """Get user details from token payload."""
    user_id = token_payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user ID"
        )
    
    # Get user roles and metadata from database
    query = """
    SELECT 
        ur.role,
        ur.metadata,
        CASE 
            WHEN ur.role = 'patient' THEN p.id
            ELSE NULL 
        END as patient_id,
        CASE 
            WHEN ur.role = 'clinician' THEN 
                COALESCE(
                    array_agg(cp.patient_id) FILTER (WHERE cp.patient_id IS NOT NULL),
                    ARRAY[]::uuid[]
                )
            ELSE ARRAY[]::uuid[]
        END as assigned_patients
    FROM user_roles ur
    LEFT JOIN patients p ON p.user_id = ur.user_id AND ur.role = 'patient'
    LEFT JOIN clinician_patients cp ON cp.clinician_user_id = ur.user_id AND ur.role = 'clinician'
    WHERE ur.user_id = $1
    GROUP BY ur.role, ur.metadata, p.id
    """
    
    try:
        rows = await db_manager.execute_query(query, user_id)
        
        if not rows:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or no roles assigned"
            )
        
        roles = []
        metadata = {}
        
        for row in rows:
            roles.append(row['role'])
            if row['metadata']:
                metadata.update(row['metadata'])
            
            if row['patient_id']:
                metadata['patient_id'] = str(row['patient_id'])
            
            if row['assigned_patients']:
                metadata['assigned_patients'] = [str(pid) for pid in row['assigned_patients']]
        
        # Get email from token or database
        email = token_payload.get("email", f"user-{user_id}@system")
        
        return User(
            id=user_id,
            email=email,
            roles=roles,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error("Failed to get user from token", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to verify user credentials"
        )


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    token_payload = security_manager.verify_token(token)
    return await get_user_from_token(token_payload)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user (additional checks can be added here)."""
    return current_user


# Role-based access control decorators
def require_role(required_role: str):
    """Decorator to require specific role."""
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not current_user.has_role(required_role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: {required_role} role required"
            )
        return current_user
    return role_checker


def require_patient_access(patient_id: str):
    """Check if user can access specific patient."""
    def access_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if not current_user.can_access_patient(patient_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: insufficient permissions for this patient"
            )
        return current_user
    return access_checker


# Common role dependencies
require_clinician = require_role("clinician")
require_admin = require_role("admin")
require_patient = require_role("patient")


async def require_patient_or_clinician_access(
    patient_id: str,
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require patient or clinician access to patient data."""
    if not current_user.can_access_patient(patient_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied: insufficient permissions for this patient"
        )
    return current_user


class AuditLogger:
    """Audit logging for security events."""
    
    @staticmethod
    async def log_access(
        user: User,
        action: str,
        resource: str,
        resource_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """Log user access events."""
        audit_data = {
            "actor_type": "user",
            "actor_id": user.id,
            "event": f"access.{action}",
            "entity_type": resource,
            "entity_id": resource_id,
            "payload": {
                "user_roles": user.roles,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }
        }
        
        try:
            await db_manager.call_function(
                "create_audit_log",
                audit_data["actor_type"],
                audit_data["actor_id"],
                audit_data["event"],
                audit_data["entity_type"],
                audit_data["entity_id"],
                json.dumps(audit_data["payload"])
            )
        except Exception as e:
            logger.error("Failed to write audit log", error=str(e), audit_data=audit_data)
    
    @staticmethod
    async def log_recommendation_action(
        user: User,
        action: str,
        recommendation_id: str,
        patient_id: str,
        model_version: Optional[str] = None,
        justification: Optional[str] = None
    ):
        """Log recommendation approval/override actions."""
        await AuditLogger.log_access(
            user=user,
            action=f"recommendation.{action}",
            resource="recommendation",
            resource_id=recommendation_id,
            metadata={
                "patient_id": patient_id,
                "model_version": model_version,
                "justification": justification
            }
        )


# Initialize audit logger
audit_logger = AuditLogger()


# Rate limiting (placeholder for Redis-based implementation)
class RateLimiter:
    """Rate limiting for API endpoints."""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def check_rate_limit(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> bool:
        """Check if request is within rate limit."""
        # TODO: Implement Redis-based rate limiting
        return True
    
    async def get_rate_limit_status(
        self,
        identifier: str,
        limit: int,
        window: int
    ) -> Dict[str, int]:
        """Get current rate limit status."""
        # TODO: Implement rate limit status
        return {
            "requests": 0,
            "limit": limit,
            "window": window,
            "reset_time": int(datetime.utcnow().timestamp()) + window
        }


# Security headers middleware configuration
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}
