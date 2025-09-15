"""Confidence scoring and verdict aggregation for evidence verification."""

import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class EvidenceItem(BaseModel):
    """Individual evidence item with scoring."""
    source_type: str
    title: Optional[str] = None
    url: Optional[str] = None
    pmid: Optional[str] = None
    doi: Optional[str] = None
    stance: str  # support, contradict, neutral, warning
    score: float
    snippet: str
    metadata: Dict[str, Any] = {}


class ClaimResult(BaseModel):
    """Result for a single claim."""
    claim_id: str
    claim_text: str
    support_score: float
    contradict_score: float
    items: List[EvidenceItem]
    verdict: str  # approved, flagged


class VerificationResult(BaseModel):
    """Overall verification result."""
    recommendation_id: str
    overall_status: str  # approved, flagged
    claims: List[ClaimResult]
    total_evidence: int
    supporting_evidence: int
    contradicting_evidence: int
    warning_evidence: int
    verification_timestamp: datetime = datetime.utcnow()


class EvidenceScorer:
    """Evidence scoring and verdict aggregation system."""
    
    def __init__(self):
        # Source type weights based on PRD
        self.source_weights = {
            "guideline": 0.9,
            "rct": 0.8,
            "cohort": 0.6,
            "case": 0.3,
            "label_warning": 1.0,  # High weight but forces warning stance
            "meta_analysis": 0.85,
            "systematic_review": 0.85,
            "observational": 0.5,
            "case_report": 0.2,
            "expert_opinion": 0.25
        }
        
        # Scoring weights
        self.scoring_weights = {
            "source_weight": 0.4,
            "recency": 0.2,
            "retrieval": 0.2,
            "stance_confidence": 0.2
        }
        
        # Verdict thresholds
        self.verdict_thresholds = {
            "contradict_threshold": 0.55,
            "support_threshold": 0.60,
            "contradict_safety_threshold": 0.40
        }
    
    def score_evidence_collection(
        self, 
        evidence_items: List[Dict[str, Any]]
    ) -> VerificationResult:
        """Score a collection of evidence items and determine verdict."""
        try:
            # Group evidence by claim
            claims_evidence = self._group_evidence_by_claim(evidence_items)
            
            # Score each claim
            claim_results = []
            for claim_id, items in claims_evidence.items():
                claim_result = self._score_claim(claim_id, items)
                claim_results.append(claim_result)
            
            # Determine overall status
            overall_status = self._determine_overall_status(claim_results)
            
            # Calculate summary statistics
            total_evidence = sum(len(claim.items) for claim in claim_results)
            supporting_evidence = sum(
                len([item for item in claim.items if item.stance == "support"])
                for claim in claim_results
            )
            contradicting_evidence = sum(
                len([item for item in claim.items if item.stance in ["contradict", "warning"]])
                for claim in claim_results
            )
            warning_evidence = sum(
                len([item for item in claim.items if item.stance == "warning"])
                for claim in claim_results
            )
            
            return VerificationResult(
                recommendation_id=evidence_items[0].get("recommendation_id", "unknown"),
                overall_status=overall_status,
                claims=claim_results,
                total_evidence=total_evidence,
                supporting_evidence=supporting_evidence,
                contradicting_evidence=contradicting_evidence,
                warning_evidence=warning_evidence
            )
            
        except Exception as e:
            logger.error("Evidence scoring failed", error=str(e))
            return VerificationResult(
                recommendation_id="unknown",
                overall_status="flagged",
                claims=[],
                total_evidence=0,
                supporting_evidence=0,
                contradicting_evidence=0,
                warning_evidence=0
            )
    
    def _group_evidence_by_claim(self, evidence_items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group evidence items by claim ID."""
        claims_evidence = {}
        
        for item in evidence_items:
            claim_id = item.get("claim_id", "unknown")
            if claim_id not in claims_evidence:
                claims_evidence[claim_id] = []
            claims_evidence[claim_id].append(item)
        
        return claims_evidence
    
    def _score_claim(self, claim_id: str, items: List[Dict[str, Any]]) -> ClaimResult:
        """Score a single claim with its evidence items."""
        try:
            # Convert items to EvidenceItem objects
            evidence_items = []
            for item in items:
                evidence_item = EvidenceItem(
                    source_type=item.get("source_type", "unknown"),
                    title=item.get("title"),
                    url=item.get("url"),
                    pmid=item.get("pmid"),
                    doi=item.get("doi"),
                    stance=item.get("stance", "neutral"),
                    score=item.get("score", 0.0),
                    snippet=item.get("snippet", ""),
                    metadata=item.get("metadata", {})
                )
                evidence_items.append(evidence_item)
            
            # Calculate support and contradict scores
            support_score = self._calculate_aggregate_score(
                [item for item in evidence_items if item.stance == "support"]
            )
            contradict_score = self._calculate_aggregate_score(
                [item for item in evidence_items if item.stance in ["contradict", "warning"]]
            )
            
            # Determine verdict
            verdict = self._determine_claim_verdict(support_score, contradict_score, evidence_items)
            
            # Get claim text
            claim_text = items[0].get("claim_text", f"Claim {claim_id}") if items else f"Claim {claim_id}"
            
            return ClaimResult(
                claim_id=claim_id,
                claim_text=claim_text,
                support_score=support_score,
                contradict_score=contradict_score,
                items=evidence_items,
                verdict=verdict
            )
            
        except Exception as e:
            logger.error("Claim scoring failed", claim_id=claim_id, error=str(e))
            return ClaimResult(
                claim_id=claim_id,
                claim_text=f"Claim {claim_id}",
                support_score=0.0,
                contradict_score=0.0,
                items=[],
                verdict="flagged"
            )
    
    def _calculate_aggregate_score(self, items: List[EvidenceItem]) -> float:
        """Calculate aggregate score for items of the same stance."""
        if not items:
            return 0.0
        
        # Use maximum score for aggregation (as per PRD)
        return max(item.score for item in items)
    
    def _determine_claim_verdict(
        self, 
        support_score: float, 
        contradict_score: float, 
        items: List[EvidenceItem]
    ) -> str:
        """Determine verdict for a single claim."""
        # Check for warnings first
        has_warnings = any(item.stance == "warning" for item in items)
        if has_warnings and contradict_score >= self.verdict_thresholds["contradict_threshold"]:
            return "flagged"
        
        # Check for contradictions
        if contradict_score >= self.verdict_thresholds["contradict_threshold"]:
            return "flagged"
        
        # Check for sufficient support
        if (support_score >= self.verdict_thresholds["support_threshold"] and 
            contradict_score < self.verdict_thresholds["contradict_safety_threshold"]):
            return "approved"
        
        # Insufficient evidence
        return "flagged"
    
    def _determine_overall_status(self, claim_results: List[ClaimResult]) -> str:
        """Determine overall verification status."""
        # If any claim is flagged, overall status is flagged
        if any(claim.verdict == "flagged" for claim in claim_results):
            return "flagged"
        
        # If all claims are approved, overall status is approved
        if all(claim.verdict == "approved" for claim in claim_results):
            return "approved"
        
        # Default to flagged for safety
        return "flagged"
    
    def score_individual_evidence(
        self, 
        content: str,
        source_type: str,
        stance: str,
        stance_confidence: float,
        retrieval_score: float,
        pub_date: Optional[str] = None,
        quality: Optional[float] = None
    ) -> float:
        """Score an individual piece of evidence."""
        try:
            # Source weight
            source_weight = self.source_weights.get(source_type, 0.5)
            
            # Recency score
            recency_score = self._calculate_recency_score(pub_date)
            
            # Normalize retrieval score (assuming 0-1 range)
            normalized_retrieval = min(1.0, max(0.0, retrieval_score))
            
            # Normalize stance confidence
            normalized_stance_conf = min(1.0, max(0.0, stance_confidence))
            
            # Apply quality modifier if available
            if quality is not None:
                source_weight *= quality
            
            # Calculate weighted score
            score = (
                self.scoring_weights["source_weight"] * source_weight +
                self.scoring_weights["recency"] * recency_score +
                self.scoring_weights["retrieval"] * normalized_retrieval +
                self.scoring_weights["stance_confidence"] * normalized_stance_conf
            )
            
            # Cap at 1.0
            return min(1.0, score)
            
        except Exception as e:
            logger.error("Individual evidence scoring failed", error=str(e))
            return 0.0
    
    def _calculate_recency_score(self, pub_date: Optional[str]) -> float:
        """Calculate recency score based on publication date."""
        if not pub_date:
            return 0.5  # Neutral score for unknown dates
        
        try:
            # Parse date
            if isinstance(pub_date, str):
                if len(pub_date) == 4:  # Year only
                    pub_year = int(pub_date)
                    pub_date_obj = datetime(pub_year, 1, 1)
                else:
                    pub_date_obj = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            else:
                pub_date_obj = pub_date
            
            # Calculate years since publication
            years_since = (datetime.now() - pub_date_obj).days / 365.25
            
            # Apply decay function (1 - years/10, clamped to 0-1)
            recency_score = max(0.0, min(1.0, 1.0 - years_since / 10.0))
            
            return recency_score
            
        except Exception as e:
            logger.warning("Failed to calculate recency score", pub_date=pub_date, error=str(e))
            return 0.5
    
    def detect_safety_flags(self, content: str) -> List[str]:
        """Detect safety flags in content."""
        safety_keywords = [
            "warning", "caution", "contraindication", "adverse", "side effect",
            "risk", "danger", "avoid", "do not", "black box", "boxed warning",
            "severe", "serious", "fatal", "life-threatening", "emergency",
            "hypoglycemia", "hyperglycemia", "allergic reaction", "anaphylaxis"
        ]
        
        content_lower = content.lower()
        detected_flags = []
        
        for keyword in safety_keywords:
            if keyword in content_lower:
                detected_flags.append(keyword)
        
        return detected_flags
    
    def get_scoring_breakdown(
        self, 
        source_type: str,
        stance: str,
        stance_confidence: float,
        retrieval_score: float,
        pub_date: Optional[str] = None,
        quality: Optional[float] = None
    ) -> Dict[str, float]:
        """Get detailed scoring breakdown for debugging."""
        source_weight = self.source_weights.get(source_type, 0.5)
        recency_score = self._calculate_recency_score(pub_date)
        normalized_retrieval = min(1.0, max(0.0, retrieval_score))
        normalized_stance_conf = min(1.0, max(0.0, stance_confidence))
        
        if quality is not None:
            source_weight *= quality
        
        weighted_score = (
            self.scoring_weights["source_weight"] * source_weight +
            self.scoring_weights["recency"] * recency_score +
            self.scoring_weights["retrieval"] * normalized_retrieval +
            self.scoring_weights["stance_confidence"] * normalized_stance_conf
        )
        
        return {
            "source_weight": source_weight,
            "recency_score": recency_score,
            "retrieval_score": normalized_retrieval,
            "stance_confidence": normalized_stance_conf,
            "final_score": min(1.0, weighted_score),
            "weights": self.scoring_weights
        }


def create_evidence_scorer() -> EvidenceScorer:
    """Factory function to create evidence scorer."""
    return EvidenceScorer()
