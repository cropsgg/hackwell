"""Configuration management for AI Wellness Gateway."""

import os
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Application
    app_name: str = "AI Wellness Assistant"
    version: str = "0.1.0"
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database
    database_url: str = Field(..., env="DATABASE_URL")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Security
    secret_key: str = Field(..., env="SECRET_KEY")
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Supabase
    supabase_url: str = Field(..., env="SUPABASE_URL")
    supabase_anon_key: str = Field(..., env="SUPABASE_ANON_KEY")
    supabase_service_key: str = Field(..., env="SUPABASE_SERVICE_KEY")
    
    # JWT
    jwt_audience: str = Field(default="authenticated", env="JWT_AUDIENCE")
    jwt_issuer: str = Field(..., env="JWT_ISSUER")
    
    # External Services
    ml_risk_service_url: str = Field(default="http://ml_risk:8000", env="ML_RISK_URL")
    agents_service_url: str = Field(default="http://agents:8000", env="AGENTS_URL")
    verifier_service_url: str = Field(default="http://verifier:8000", env="VERIFIER_URL")
    ingest_service_url: str = Field(default="http://ingest:8000", env="INGEST_URL")
    ws_stream_service_url: str = Field(default="http://ws_stream:8000", env="WS_STREAM_URL")
    
    # External APIs
    pubmed_base_url: str = Field(
        default="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        env="PUBMED_BASE"
    )
    openfda_base_url: str = Field(default="https://api.fda.gov", env="OPENFDA_BASE")
    rxnorm_base_url: str = Field(
        default="https://rxnav.nlm.nih.gov/REST",
        env="RXNORM_BASE"
    )
    
    # Evidence scoring
    evidence_min_score: float = Field(default=0.6, env="EVIDENCE_MIN_SCORE")
    
    # CORS
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000"],
        env="ALLOWED_ORIGINS"
    )
    
    # ML Configuration
    ml_model_path: str = Field(default="services/ml_risk/models", env="ML_MODEL_PATH")
    default_model_version: str = Field(default="risk_lgbm_v0.1", env="DEFAULT_MODEL_VERSION")
    
    # Rate limiting
    rate_limit_requests: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    rate_limit_window: int = Field(default=60, env="RATE_LIMIT_WINDOW")
    
    # Monitoring
    enable_metrics: bool = Field(default=True, env="ENABLE_METRICS")
    metrics_path: str = Field(default="/metrics", env="METRICS_PATH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_cors_origins(self) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(self.allowed_origins, str):
            return [origin.strip() for origin in self.allowed_origins.split(",")]
        return self.allowed_origins


# Global settings instance
settings = Settings()


# Database configuration
def get_database_config():
    """Get database configuration for SQLAlchemy."""
    return {
        "url": settings.database_url,
        "echo": settings.debug,
        "pool_size": 10,
        "max_overflow": 20,
        "pool_pre_ping": True,
        "pool_recycle": 3600,
    }


# Redis configuration
def get_redis_config():
    """Get Redis configuration."""
    return {
        "url": settings.redis_url,
        "decode_responses": True,
        "socket_connect_timeout": 5,
        "socket_timeout": 5,
        "retry_on_timeout": True,
    }


# Logging configuration
def get_logging_config():
    """Get structured logging configuration."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": "structlog.stdlib.ProcessorFormatter",
                "processor": "structlog.dev.ConsoleRenderer",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "json",
            },
        },
        "loggers": {
            "": {
                "handlers": ["console"],
                "level": settings.log_level,
                "propagate": True,
            },
        },
    }


# Service URLs mapping
SERVICE_URLS = {
    "ml_risk": settings.ml_risk_service_url,
    "agents": settings.agents_service_url,
    "verifier": settings.verifier_service_url,
    "ingest": settings.ingest_service_url,
    "ws_stream": settings.ws_stream_service_url,
}


# Model configuration
MODEL_CONFIG = {
    "default_version": settings.default_model_version,
    "model_path": settings.ml_model_path,
    "supported_algorithms": ["lightgbm", "xgboost"],
    "feature_requirements": [
        "age", "sex", "bmi", "hba1c", "sbp", "dbp",
        "glucose_fasting", "total_cholesterol", "hdl_cholesterol"
    ]
}


# Security configuration
SECURITY_CONFIG = {
    "password_min_length": 8,
    "password_require_special": True,
    "session_timeout_minutes": settings.access_token_expire_minutes,
    "max_login_attempts": 5,
    "lockout_duration_minutes": 15,
}


# API rate limiting
RATE_LIMIT_CONFIG = {
    "default": f"{settings.rate_limit_requests}/{settings.rate_limit_window}",
    "ml_inference": "10/60",  # Stricter limit for ML endpoints
    "evidence_lookup": "20/60",  # Moderate limit for evidence queries
    "patient_data": "50/60",  # More generous for patient data access
}
