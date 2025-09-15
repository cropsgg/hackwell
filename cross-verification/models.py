"""Pydantic models for evidence verification."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class EvidenceItem(BaseModel):
    """Individual evidence item."""
    source_type: str = Field(..., description="Type of evidence source")
    title: Optional[str] = Field(None, description="Evidence title")
    url: Optional[str] = Field(None, description="Evidence URL")
    pmid: Optional[str] = Field(None, description="PubMed ID")
    doi: Optional[str] = Field(None, description="DOI")
    stance: str = Field(..., description="Stance: support, contradict, neutral, warning")
    score: float = Field(..., ge=0, le=1, description="Evidence confidence score")
    snippet: str = Field(..., description="Evidence snippet")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ClaimResult(BaseModel):
    """Result for a single claim."""
    claim_id: str = Field(..., description="Unique claim identifier")
    claim_text: str = Field(..., description="Claim text")
    support_score: float = Field(..., ge=0, le=1, description="Support evidence score")
    contradict_score: float = Field(..., ge=0, le=1, description="Contradict evidence score")
    items: List[EvidenceItem] = Field(default_factory=list, description="Evidence items")
    verdict: str = Field(..., description="Claim verdict: approved, flagged")


class EvidenceVerificationResponse(BaseModel):
    """Evidence verification response."""
    recommendation_id: str = Field(..., description="Recommendation identifier")
    overall_status: str = Field(..., description="Overall status: approved, flagged")
    claims: List[ClaimResult] = Field(default_factory=list, description="Claim results")
    total_evidence: int = Field(0, description="Total evidence items")
    supporting_evidence: int = Field(0, description="Supporting evidence count")
    contradicting_evidence: int = Field(0, description="Contradicting evidence count")
    warning_evidence: int = Field(0, description="Warning evidence count")
    verification_timestamp: datetime = Field(default_factory=datetime.utcnow, description="Verification timestamp")


class VerificationRequest(BaseModel):
    """Evidence verification request."""
    recommendation_id: str = Field(..., description="Recommendation identifier")
    care_plan: Dict[str, Any] = Field(..., description="Care plan recommendations")
    patient_context: Dict[str, Any] = Field(..., description="Patient context")
    max_evidence_per_claim: int = Field(16, ge=1, le=50, description="Max evidence per claim")
    include_external_apis: bool = Field(True, description="Include external API evidence")


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    components: Dict[str, str] = Field(..., description="Component health status")


class VerificationStats(BaseModel):
    """Verification statistics."""
    recommendations: Dict[str, int] = Field(..., description="Recommendation statistics")
    evidence: Dict[str, int] = Field(..., description="Evidence statistics")
    period: str = Field(..., description="Statistics period")


class EvidenceSearchRequest(BaseModel):
    """Evidence search request."""
    query: str = Field(..., description="Search query")
    source_types: Optional[List[str]] = Field(None, description="Filter by source types")
    max_results: int = Field(10, ge=1, le=50, description="Maximum results")
    min_date: Optional[str] = Field(None, description="Minimum publication date")
    max_date: Optional[str] = Field(None, description="Maximum publication date")


class EvidenceSearchResponse(BaseModel):
    """Evidence search response."""
    query: str = Field(..., description="Search query")
    results: List[EvidenceItem] = Field(default_factory=list, description="Search results")
    total_results: int = Field(0, description="Total results found")
    search_time_ms: float = Field(0, description="Search time in milliseconds")


class ClaimExtractionRequest(BaseModel):
    """Claim extraction request."""
    care_plan: Dict[str, Any] = Field(..., description="Care plan recommendations")
    patient_context: Dict[str, Any] = Field(..., description="Patient context")


class ExtractedClaim(BaseModel):
    """Extracted claim."""
    id: str = Field(..., description="Claim identifier")
    text: str = Field(..., description="Claim text")
    context: Dict[str, Any] = Field(..., description="Claim context")
    policy: str = Field(..., description="Policy type: benefit, safety, monitoring")
    category: str = Field(..., description="Category: dietary, exercise, medication")
    original_recommendation: str = Field(..., description="Original recommendation text")


class ClaimExtractionResponse(BaseModel):
    """Claim extraction response."""
    claims: List[ExtractedClaim] = Field(default_factory=list, description="Extracted claims")
    total_claims: int = Field(0, description="Total claims extracted")


class StanceClassificationRequest(BaseModel):
    """Stance classification request."""
    claim: str = Field(..., description="Claim text")
    passage: str = Field(..., description="Passage text")


class StanceClassificationResponse(BaseModel):
    """Stance classification response."""
    stance: str = Field(..., description="Classified stance")
    confidence: float = Field(..., ge=0, le=1, description="Classification confidence")
    raw_scores: Dict[str, float] = Field(..., description="Raw classification scores")


class EmbeddingRequest(BaseModel):
    """Embedding generation request."""
    text: str = Field(..., description="Text to embed")
    model: Optional[str] = Field(None, description="Embedding model")


class EmbeddingResponse(BaseModel):
    """Embedding generation response."""
    embedding: List[float] = Field(..., description="Generated embedding")
    model: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")
    usage_tokens: int = Field(0, description="Token usage")


class BatchEmbeddingRequest(BaseModel):
    """Batch embedding generation request."""
    texts: List[str] = Field(..., description="Texts to embed")
    model: Optional[str] = Field(None, description="Embedding model")


class BatchEmbeddingResponse(BaseModel):
    """Batch embedding generation response."""
    embeddings: List[List[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used")
    dimension: int = Field(..., description="Embedding dimension")
    total_tokens: int = Field(0, description="Total token usage")


class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
