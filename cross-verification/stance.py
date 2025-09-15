"""Stance classification using Natural Language Inference (NLI)."""

import asyncio
from typing import Tuple, List, Dict, Any, Optional
import structlog
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pydantic import BaseModel

logger = structlog.get_logger()


class StanceResult(BaseModel):
    """Result of stance classification."""
    stance: str  # support, contradict, neutral, warning
    confidence: float
    raw_scores: Dict[str, float]


class StanceClassifier:
    """NLI-based stance classifier for evidence verification."""
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-base-mnli",
        device: str = "auto",
        batch_size: int = 8,
        confidence_threshold: float = 0.5
    ):
        self.model_name = model_name
        self.device = self._get_device(device)
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Load model and tokenizer
        self.tokenizer = None
        self.model = None
        self._load_model()
        
        # Label mapping for MNLI
        self.label_map = {
            0: "contradict",  # contradiction
            1: "neutral",     # neutral
            2: "support"      # entailment
        }
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the NLI model and tokenizer."""
        try:
            logger.info("Loading NLI model", model=self.model_name, device=self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3  # contradiction, neutral, entailment
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("NLI model loaded successfully")
            
        except Exception as e:
            logger.error("Failed to load NLI model", error=str(e))
            raise
    
    async def classify(self, claim: str, passage: str) -> StanceResult:
        """Classify stance of a passage with respect to a claim."""
        try:
            # Prepare input
            inputs = self.tokenizer(
                claim,
                passage,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Extract scores
            scores = probabilities[0].cpu().numpy()
            raw_scores = {
                "contradict": float(scores[0]),
                "neutral": float(scores[1]),
                "support": float(scores[2])
            }
            
            # Determine stance and confidence
            predicted_idx = scores.argmax()
            predicted_stance = self.label_map[predicted_idx]
            confidence = float(scores[predicted_idx])
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                predicted_stance = "neutral"
                confidence = 1.0 - confidence  # Invert for neutral
            
            logger.debug("Stance classification completed",
                        claim=claim[:50] + "...",
                        passage=passage[:50] + "...",
                        stance=predicted_stance,
                        confidence=confidence)
            
            return StanceResult(
                stance=predicted_stance,
                confidence=confidence,
                raw_scores=raw_scores
            )
            
        except Exception as e:
            logger.error("Stance classification failed", 
                        claim=claim[:50] + "...",
                        error=str(e))
            # Return neutral stance on error
            return StanceResult(
                stance="neutral",
                confidence=0.0,
                raw_scores={"contradict": 0.0, "neutral": 1.0, "support": 0.0}
            )
    
    async def classify_batch(self, claim_passage_pairs: List[Tuple[str, str]]) -> List[StanceResult]:
        """Classify stance for multiple claim-passage pairs."""
        if not claim_passage_pairs:
            return []
        
        results = []
        
        # Process in batches
        for i in range(0, len(claim_passage_pairs), self.batch_size):
            batch = claim_passage_pairs[i:i + self.batch_size]
            batch_results = await self._classify_batch(batch)
            results.extend(batch_results)
        
        return results
    
    async def _classify_batch(self, claim_passage_pairs: List[Tuple[str, str]]) -> List[StanceResult]:
        """Classify a batch of claim-passage pairs."""
        try:
            # Prepare inputs
            claims, passages = zip(*claim_passage_pairs)
            
            inputs = self.tokenizer(
                list(claims),
                list(passages),
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
            
            # Process results
            results = []
            for i, probs in enumerate(probabilities):
                scores = probs.cpu().numpy()
                raw_scores = {
                    "contradict": float(scores[0]),
                    "neutral": float(scores[1]),
                    "support": float(scores[2])
                }
                
                predicted_idx = scores.argmax()
                predicted_stance = self.label_map[predicted_idx]
                confidence = float(scores[predicted_idx])
                
                # Apply confidence threshold
                if confidence < self.confidence_threshold:
                    predicted_stance = "neutral"
                    confidence = 1.0 - confidence
                
                results.append(StanceResult(
                    stance=predicted_stance,
                    confidence=confidence,
                    raw_scores=raw_scores
                ))
            
            logger.debug("Batch stance classification completed",
                        batch_size=len(claim_passage_pairs))
            
            return results
            
        except Exception as e:
            logger.error("Batch stance classification failed", error=str(e))
            # Return neutral stances for all pairs on error
            return [
                StanceResult(
                    stance="neutral",
                    confidence=0.0,
                    raw_scores={"contradict": 0.0, "neutral": 1.0, "support": 0.0}
                )
                for _ in claim_passage_pairs
            ]
    
    def detect_warning_indicators(self, text: str) -> bool:
        """Detect if text contains warning indicators that should be flagged."""
        warning_keywords = [
            "warning", "caution", "contraindication", "adverse", "side effect",
            "risk", "danger", "avoid", "do not", "black box", "boxed warning",
            "severe", "serious", "fatal", "life-threatening", "emergency"
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in warning_keywords)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "batch_size": self.batch_size,
            "confidence_threshold": self.confidence_threshold,
            "label_map": self.label_map
        }


class MockStanceClassifier(StanceClassifier):
    """Mock stance classifier for testing."""
    
    def __init__(self, **kwargs):
        # Don't load actual model
        self.model_name = "mock"
        self.device = "cpu"
        self.batch_size = kwargs.get("batch_size", 8)
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.label_map = {
            0: "contradict",
            1: "neutral", 
            2: "support"
        }
    
    def _load_model(self):
        """Skip model loading for mock."""
        pass
    
    async def classify(self, claim: str, passage: str) -> StanceResult:
        """Mock classification based on simple heuristics."""
        # Simple heuristic-based classification
        passage_lower = passage.lower()
        claim_lower = claim.lower()
        
        # Check for warning indicators
        if self.detect_warning_indicators(passage):
            return StanceResult(
                stance="warning",
                confidence=0.8,
                raw_scores={"contradict": 0.1, "neutral": 0.1, "support": 0.0}
            )
        
        # Simple keyword matching
        support_keywords = ["support", "evidence", "shows", "demonstrates", "proves", "confirms"]
        contradict_keywords = ["contradicts", "refutes", "disproves", "opposes", "against"]
        
        support_score = sum(1 for kw in support_keywords if kw in passage_lower)
        contradict_score = sum(1 for kw in contradict_keywords if kw in passage_lower)
        
        if support_score > contradict_score and support_score > 0:
            stance = "support"
            confidence = min(0.7, 0.5 + support_score * 0.1)
        elif contradict_score > support_score and contradict_score > 0:
            stance = "contradict"
            confidence = min(0.7, 0.5 + contradict_score * 0.1)
        else:
            stance = "neutral"
            confidence = 0.6
        
        raw_scores = {
            "contradict": 0.2 if stance == "contradict" else 0.1,
            "neutral": 0.3 if stance == "neutral" else 0.2,
            "support": 0.5 if stance == "support" else 0.1
        }
        
        return StanceResult(
            stance=stance,
            confidence=confidence,
            raw_scores=raw_scores
        )
    
    async def classify_batch(self, claim_passage_pairs: List[Tuple[str, str]]) -> List[StanceResult]:
        """Mock batch classification."""
        return [await self.classify(claim, passage) for claim, passage in claim_passage_pairs]


def create_stance_classifier(
    classifier_type: str = "deberta",
    model_name: Optional[str] = None,
    **kwargs
) -> StanceClassifier:
    """Factory function to create stance classifier."""
    
    if classifier_type == "deberta":
        model = model_name or "microsoft/deberta-base-mnli"
        return StanceClassifier(model_name=model, **kwargs)
    
    elif classifier_type == "mock":
        return MockStanceClassifier(**kwargs)
    
    else:
        raise ValueError(f"Unknown stance classifier: {classifier_type}")
