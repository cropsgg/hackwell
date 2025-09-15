"""Configuration for Evidence Verification Service."""

import os
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field


class EvidenceVerificationConfig(BaseSettings):
    """Configuration for evidence verification service."""
    
    # Service configuration
    service_name: str = Field(default="evidence-verification", env="SERVICE_NAME")
    version: str = Field(default="1.0.0", env="SERVICE_VERSION")
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Database configuration
    database_url: str = Field(..., env="DATABASE_URL")
    db_pool_min_size: int = Field(default=5, env="DB_POOL_MIN_SIZE")
    db_pool_max_size: int = Field(default=20, env="DB_POOL_MAX_SIZE")
    db_command_timeout: int = Field(default=60, env="DB_COMMAND_TIMEOUT")
    
    # Embeddings configuration
    embeddings_provider: str = Field(default="openai", env="EMBEDDINGS_PROVIDER")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    embeddings_model: str = Field(default="text-embedding-3-large", env="EMBEDDINGS_MODEL")
    embeddings_dimension: int = Field(default=3072, env="EMBEDDINGS_DIMENSION")
    embeddings_cache: bool = Field(default=True, env="EMBEDDINGS_CACHE")
    
    # Stance classification configuration
    stance_classifier: str = Field(default="deberta", env="STANCE_CLASSIFIER")
    stance_model: str = Field(default="microsoft/deberta-base-mnli", env="STANCE_MODEL")
    stance_confidence_threshold: float = Field(default=0.5, env="STANCE_CONFIDENCE_THRESHOLD")
    stance_batch_size: int = Field(default=8, env="STANCE_BATCH_SIZE")
    
    # Retrieval configuration
    k_semantic: int = Field(default=8, env="K_SEMANTIC")
    k_lexical: int = Field(default=8, env="K_LEXICAL")
    rrf_k: int = Field(default=60, env="RRF_K")
    max_evidence_per_claim: int = Field(default=16, env="MAX_EVIDENCE_PER_CLAIM")
    
    # External API configuration
    enable_pubmed: bool = Field(default=True, env="ENABLE_PUBMED")
    pubmed_api_key: Optional[str] = Field(default=None, env="PUBMED_API_KEY")
    pubmed_email: str = Field(default="your-email@example.com", env="PUBMED_EMAIL")
    pubmed_tool: str = Field(default="hackwell-evidence-verifier", env="PUBMED_TOOL")
    pubmed_max_retries: int = Field(default=3, env="PUBMED_MAX_RETRIES")
    pubmed_timeout: float = Field(default=30.0, env="PUBMED_TIMEOUT")
    
    enable_openfda: bool = Field(default=True, env="ENABLE_OPENFDA")
    openfda_api_key: Optional[str] = Field(default=None, env="OPENFDA_API_KEY")
    openfda_timeout: float = Field(default=30.0, env="OPENFDA_TIMEOUT")
    
    enable_rxnorm: bool = Field(default=True, env="ENABLE_RXNORM")
    rxnorm_timeout: float = Field(default=30.0, env="RXNORM_TIMEOUT")
    
    enable_ada: bool = Field(default=True, env="ENABLE_ADA")
    ada_guidelines_file: Optional[str] = Field(default=None, env="ADA_GUIDELINES_FILE")
    
    # Scoring configuration
    evidence_min_score: float = Field(default=0.3, env="EVIDENCE_MIN_SCORE")
    contradict_threshold: float = Field(default=0.55, env="CONTRADICT_THRESHOLD")
    support_threshold: float = Field(default=0.60, env="SUPPORT_THRESHOLD")
    contradict_safety_threshold: float = Field(default=0.40, env="CONTRADICT_SAFETY_THRESHOLD")
    
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
        extra = "ignore"
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return {
            "url": self.database_url,
            "min_size": self.db_pool_min_size,
            "max_size": self.db_pool_max_size,
            "command_timeout": self.db_command_timeout
        }
    
    def get_embeddings_config(self) -> Dict[str, Any]:
        """Get embeddings configuration."""
        return {
            "provider_type": self.embeddings_provider,
            "api_key": self.openai_api_key,
            "model": self.embeddings_model,
            "dimension": self.embeddings_dimension,
            "cache_embeddings": self.embeddings_cache
        }
    
    def get_stance_config(self) -> Dict[str, Any]:
        """Get stance classification configuration."""
        return {
            "classifier_type": self.stance_classifier,
            "model_name": self.stance_model,
            "confidence_threshold": self.stance_confidence_threshold,
            "batch_size": self.stance_batch_size
        }
    
    def get_retrieval_config(self) -> Dict[str, Any]:
        """Get retrieval configuration."""
        return {
            "k_semantic": self.k_semantic,
            "k_lexical": self.k_lexical,
            "rrf_k": self.rrf_k,
            "max_evidence_per_claim": self.max_evidence_per_claim
        }
    
    def get_external_apis_config(self) -> Dict[str, Any]:
        """Get external APIs configuration."""
        return {
            "pubmed": {
                "enabled": self.enable_pubmed,
                "api_key": self.pubmed_api_key,
                "email": self.pubmed_email,
                "tool": self.pubmed_tool,
                "max_retries": self.pubmed_max_retries,
                "timeout": self.pubmed_timeout
            },
            "openfda": {
                "enabled": self.enable_openfda,
                "api_key": self.openfda_api_key,
                "timeout": self.openfda_timeout
            },
            "rxnorm": {
                "enabled": self.enable_rxnorm,
                "timeout": self.rxnorm_timeout
            },
            "ada": {
                "enabled": self.enable_ada,
                "guidelines_file": self.ada_guidelines_file
            }
        }
    
    def get_scoring_config(self) -> Dict[str, Any]:
        """Get scoring configuration."""
        return {
            "evidence_min_score": self.evidence_min_score,
            "contradict_threshold": self.contradict_threshold,
            "support_threshold": self.support_threshold,
            "contradict_safety_threshold": self.contradict_safety_threshold
        }


# Global configuration instance
config = EvidenceVerificationConfig()


# Default configuration for development
DEFAULT_CONFIG = {
    "embeddings_provider": "openai",
    "embeddings_model": "text-embedding-3-large",
    "stance_classifier": "deberta",
    "stance_model": "microsoft/deberta-base-mnli",
    "k_semantic": 8,
    "k_lexical": 8,
    "enable_pubmed": True,
    "enable_openfda": True,
    "enable_rxnorm": True,
    "enable_ada": True,
    "max_evidence_per_claim": 16
}


def get_config() -> EvidenceVerificationConfig:
    """Get the global configuration instance."""
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration for development."""
    return DEFAULT_CONFIG.copy()
