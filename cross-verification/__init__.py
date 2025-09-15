"""Evidence Verifier (RAG) — Cross-Verification Service

A comprehensive evidence verification system that validates care plan recommendations
using RAG (Retrieval-Augmented Generation) with multiple evidence sources including
ADA Standards of Care, PubMed, openFDA, and RxNorm.
"""

from service import EvidenceVerificationService, create_verification_service
from models import (
    VerificationRequest,
    EvidenceVerificationResponse,
    EvidenceItem,
    ClaimResult,
    VerificationResult
)
from config import get_config, get_default_config

__version__ = "1.0.0"
__author__ = "Hackwell AI Wellness Assistant Team"

__all__ = [
    "EvidenceVerificationService",
    "create_verification_service",
    "VerificationRequest",
    "EvidenceVerificationResponse", 
    "EvidenceItem",
    "ClaimResult",
    "VerificationResult",
    "get_config",
    "get_default_config"
]
