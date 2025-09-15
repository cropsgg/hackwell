"""Main evidence verification service orchestrator."""

import asyncio
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
from datetime import datetime
import structlog
import asyncpg
from pydantic import BaseModel

from embeddings import create_embeddings_provider, EmbeddingsProvider
from stance import create_stance_classifier, StanceClassifier
from claim_extractor import create_claim_extractor, ClaimExtractor, ClaimContext
from retriever import create_retriever, HybridRetriever
from scorer import create_evidence_scorer, EvidenceScorer
from adapters.pubmed_client import create_pubmed_client, PubMedClient
from adapters.openfda_client import create_openfda_client, OpenFDAClient
from adapters.rxnorm_client import create_rxnorm_client, RxNormClient
from adapters.ada_mapper import create_ada_mapper, ADAMapper

logger = structlog.get_logger()


class VerificationRequest(BaseModel):
    """Evidence verification request."""
    recommendation_id: str
    care_plan: Dict[str, Any]
    patient_context: Dict[str, Any]
    max_evidence_per_claim: int = 16
    include_external_apis: bool = True


class EvidenceVerificationService:
    """Main evidence verification service."""
    
    def __init__(
        self,
        db_pool: asyncpg.Pool,
        embeddings_provider: EmbeddingsProvider,
        stance_classifier: StanceClassifier,
        retriever: HybridRetriever,
        evidence_scorer: EvidenceScorer,
        pubmed_client: Optional[PubMedClient] = None,
        openfda_client: Optional[OpenFDAClient] = None,
        rxnorm_client: Optional[RxNormClient] = None,
        ada_mapper: Optional[ADAMapper] = None
    ):
        self.db_pool = db_pool
        self.embeddings_provider = embeddings_provider
        self.stance_classifier = stance_classifier
        self.retriever = retriever
        self.evidence_scorer = evidence_scorer
        self.pubmed_client = pubmed_client
        self.openfda_client = openfda_client
        self.rxnorm_client = rxnorm_client
        self.ada_mapper = ada_mapper
        
        # Initialize claim extractor
        self.claim_extractor = create_claim_extractor()
    
    async def verify_recommendation(self, request: VerificationRequest) -> Dict[str, Any]:
        """Verify a care plan recommendation with evidence."""
        try:
            logger.info("Starting evidence verification",
                       recommendation_id=request.recommendation_id)
            
            # Extract claims from care plan
            patient_context = ClaimContext(**request.patient_context)
            claims = self.claim_extractor.extract_claims(
                request.care_plan, 
                patient_context
            )
            
            if not claims:
                logger.warning("No claims extracted from care plan",
                              recommendation_id=request.recommendation_id)
                return self._create_empty_verification_result(request.recommendation_id)
            
            # Process each claim
            all_evidence = []
            for claim in claims:
                claim_evidence = await self._process_claim(claim, request)
                all_evidence.extend(claim_evidence)
            
            # Score evidence and determine verdict
            verification_result = self.evidence_scorer.score_evidence_collection(all_evidence)
            
            # Persist evidence links
            await self._persist_evidence_links(
                request.recommendation_id, 
                all_evidence
            )
            
            # Log audit event
            await self._log_verification_event(
                request.recommendation_id,
                verification_result,
                len(claims)
            )
            
            logger.info("Evidence verification completed",
                       recommendation_id=request.recommendation_id,
                       overall_status=verification_result.overall_status,
                       total_evidence=verification_result.total_evidence)
            
            return verification_result.dict()
            
        except Exception as e:
            logger.error("Evidence verification failed",
                        recommendation_id=request.recommendation_id,
                        error=str(e))
            return self._create_error_verification_result(request.recommendation_id, str(e))
    
    async def _process_claim(
        self, 
        claim, 
        request: VerificationRequest
    ) -> List[Dict[str, Any]]:
        """Process a single claim to gather evidence."""
        evidence_items = []
        
        try:
            # Build search query from claim
            search_query = self._build_search_query(claim)
            
            # Retrieve evidence from vector database
            retrieval_results = await self.retriever.retrieve_by_claim(
                search_query,
                claim.context.dict(),
                request.max_evidence_per_claim
            )
            
            # Process each retrieval result
            for result in retrieval_results:
                # Classify stance
                stance_result = await self.stance_classifier.classify(
                    claim.text, 
                    result.content
                )
                
                # Score the evidence
                evidence_score = self.evidence_scorer.score_individual_evidence(
                    content=result.content,
                    source_type=result.source_type,
                    stance=stance_result.stance,
                    stance_confidence=stance_result.confidence,
                    retrieval_score=result.combined_score,
                    pub_date=result.pub_date,
                    quality=result.metadata.get("quality")
                )
                
                # Create evidence item
                evidence_item = {
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "source_type": result.source_type,
                    "title": result.title,
                    "url": result.url,
                    "pmid": result.pmid,
                    "doi": result.doi,
                    "stance": stance_result.stance,
                    "score": evidence_score,
                    "snippet": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "metadata": {
                        "chunk_id": result.chunk_id,
                        "document_id": result.document_id,
                        "semantic_score": result.semantic_score,
                        "lexical_score": result.lexical_score,
                        "retrieval_score": result.combined_score,
                        "stance_confidence": stance_result.confidence,
                        "raw_scores": stance_result.raw_scores
                    }
                }
                evidence_items.append(evidence_item)
            
            # Add external API evidence if enabled
            if request.include_external_apis:
                external_evidence = await self._gather_external_evidence(claim, request)
                evidence_items.extend(external_evidence)
            
            logger.debug("Claim processed",
                        claim_id=claim.id,
                        evidence_count=len(evidence_items))
            
            return evidence_items
            
        except Exception as e:
            logger.error("Claim processing failed",
                        claim_id=claim.id,
                        error=str(e))
            return []
    
    def _build_search_query(self, claim) -> str:
        """Build search query from claim."""
        query_parts = [claim.text]
        
        # Add context information
        if claim.context.conditions:
            query_parts.extend(claim.context.conditions)
        
        if claim.context.age:
            if claim.context.age < 18:
                query_parts.append("pediatric")
            elif claim.context.age > 65:
                query_parts.append("elderly")
        
        if claim.context.ckd:
            query_parts.append("chronic kidney disease")
        
        if claim.context.pregnancy:
            query_parts.append("pregnancy")
        
        return " ".join(query_parts)
    
    async def _gather_external_evidence(
        self, 
        claim, 
        request: VerificationRequest
    ) -> List[Dict[str, Any]]:
        """Gather evidence from external APIs."""
        external_evidence = []
        
        try:
            # PubMed evidence
            if self.pubmed_client and claim.policy in ["benefit", "safety"]:
                pubmed_evidence = await self._get_pubmed_evidence(claim)
                external_evidence.extend(pubmed_evidence)
            
            # ADA guidelines
            if self.ada_mapper and claim.policy in ["benefit", "safety"]:
                ada_evidence = await self._get_ada_evidence(claim)
                external_evidence.extend(ada_evidence)
            
            # Drug safety evidence
            if (self.openfda_client and self.rxnorm_client and 
                claim.category == "medication" and claim.policy == "safety"):
                safety_evidence = await self._get_drug_safety_evidence(claim)
                external_evidence.extend(safety_evidence)
            
            logger.debug("External evidence gathered",
                        claim_id=claim.id,
                        external_count=len(external_evidence))
            
            return external_evidence
            
        except Exception as e:
            logger.error("External evidence gathering failed",
                        claim_id=claim.id,
                        error=str(e))
            return []
    
    async def _get_pubmed_evidence(self, claim) -> List[Dict[str, Any]]:
        """Get evidence from PubMed."""
        try:
            # Search PubMed
            articles = await self.pubmed_client.search_articles(
                query=claim.text,
                max_results=5
            )
            
            evidence_items = []
            for article in articles:
                # Classify stance
                content = f"{article.title}. {article.abstract or ''}"
                stance_result = await self.stance_classifier.classify(
                    claim.text, 
                    content
                )
                
                # Score evidence
                evidence_score = self.evidence_scorer.score_individual_evidence(
                    content=content,
                    source_type=article.study_type or "observational",
                    stance=stance_result.stance,
                    stance_confidence=stance_result.confidence,
                    retrieval_score=0.8,  # High score for direct API results
                    pub_date=article.pub_date,
                    quality=article.quality_score
                )
                
                evidence_items.append({
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "source_type": article.study_type or "observational",
                    "title": article.title,
                    "url": article.url,
                    "pmid": article.pmid,
                    "doi": article.doi,
                    "stance": stance_result.stance,
                    "score": evidence_score,
                    "snippet": content[:200] + "..." if len(content) > 200 else content,
                    "metadata": {
                        "authors": article.authors,
                        "journal": article.journal,
                        "pub_date": article.pub_date,
                        "quality_score": article.quality_score,
                        "stance_confidence": stance_result.confidence
                    }
                })
            
            return evidence_items
            
        except Exception as e:
            logger.error("PubMed evidence gathering failed", error=str(e))
            return []
    
    async def _get_ada_evidence(self, claim) -> List[Dict[str, Any]]:
        """Get evidence from ADA guidelines."""
        try:
            # Search ADA statements
            statements = self.ada_mapper.search_statements(
                query=claim.text,
                conditions=claim.context.conditions,
                max_results=3
            )
            
            evidence_items = []
            for statement in statements:
                # Classify stance
                stance_result = await self.stance_classifier.classify(
                    claim.text, 
                    statement.content
                )
                
                # Score evidence
                evidence_score = self.evidence_scorer.score_individual_evidence(
                    content=statement.content,
                    source_type="guideline",
                    stance=stance_result.stance,
                    stance_confidence=stance_result.confidence,
                    retrieval_score=0.9,  # High score for guidelines
                    pub_date=statement.last_updated,
                    quality=0.9  # High quality for ADA guidelines
                )
                
                evidence_items.append({
                    "claim_id": claim.id,
                    "claim_text": claim.text,
                    "source_type": "guideline",
                    "title": statement.title,
                    "url": statement.url,
                    "doi": None,
                    "pmid": None,
                    "stance": stance_result.stance,
                    "score": evidence_score,
                    "snippet": statement.content[:200] + "..." if len(statement.content) > 200 else statement.content,
                    "metadata": {
                        "section": statement.section,
                        "subsection": statement.subsection,
                        "evidence_level": statement.evidence_level,
                        "recommendation_class": statement.recommendation_class,
                        "citations": statement.citations,
                        "stance_confidence": stance_result.confidence
                    }
                })
            
            return evidence_items
            
        except Exception as e:
            logger.error("ADA evidence gathering failed", error=str(e))
            return []
    
    async def _get_drug_safety_evidence(self, claim) -> List[Dict[str, Any]]:
        """Get drug safety evidence from openFDA and RxNorm."""
        try:
            evidence_items = []
            
            # Extract drug names from claim
            drug_names = self._extract_drug_names_from_claim(claim)
            
            for drug_name in drug_names:
                # Normalize drug name
                normalized = await self.rxnorm_client.normalize_drug_names([drug_name])
                rxcui = normalized.get(drug_name)
                
                if rxcui:
                    # Get drug safety information
                    drug_labels = await self.openfda_client.search_drug(drug_name)
                    
                    for label in drug_labels:
                        # Check for warnings and contraindications
                        safety_texts = (
                            label.boxed_warnings + 
                            label.warnings + 
                            label.contraindications
                        )
                        
                        for safety_text in safety_texts:
                            # Classify stance
                            stance_result = await self.stance_classifier.classify(
                                claim.text, 
                                safety_text
                            )
                            
                            # Force warning stance for safety information
                            if any(keyword in safety_text.lower() for keyword in 
                                   ["warning", "caution", "contraindication", "adverse"]):
                                stance_result.stance = "warning"
                                stance_result.confidence = 0.9
                            
                            # Score evidence
                            evidence_score = self.evidence_scorer.score_individual_evidence(
                                content=safety_text,
                                source_type="label_warning",
                                stance=stance_result.stance,
                                stance_confidence=stance_result.confidence,
                                retrieval_score=0.95,  # Very high score for safety warnings
                                quality=1.0  # Maximum quality for regulatory warnings
                            )
                            
                            evidence_items.append({
                                "claim_id": claim.id,
                                "claim_text": claim.text,
                                "source_type": "label_warning",
                                "title": f"Safety Information for {drug_name}",
                                "url": label.url,
                                "doi": None,
                                "pmid": None,
                                "stance": stance_result.stance,
                                "score": evidence_score,
                                "snippet": safety_text[:200] + "..." if len(safety_text) > 200 else safety_text,
                                "metadata": {
                                    "drug_name": drug_name,
                                    "rxcui": rxcui,
                                    "manufacturer": label.manufacturer,
                                    "last_updated": label.last_updated,
                                    "stance_confidence": stance_result.confidence
                                }
                            })
            
            return evidence_items
            
        except Exception as e:
            logger.error("Drug safety evidence gathering failed", error=str(e))
            return []
    
    def _extract_drug_names_from_claim(self, claim) -> List[str]:
        """Extract drug names from claim text."""
        # Simple keyword extraction - could be enhanced with NER
        drug_keywords = [
            "metformin", "insulin", "sulfonylurea", "glp-1", "sglt2",
            "glipizide", "glyburide", "glimepiride", "pioglitazone",
            "rosiglitazone", "sitagliptin", "saxagliptin", "linagliptin",
            "exenatide", "liraglutide", "dulaglutide", "semaglutide",
            "empagliflozin", "canagliflozin", "dapagliflozin"
        ]
        
        claim_lower = claim.text.lower()
        found_drugs = []
        
        for drug in drug_keywords:
            if drug in claim_lower:
                found_drugs.append(drug)
        
        return found_drugs
    
    async def _persist_evidence_links(
        self, 
        recommendation_id: str, 
        evidence_items: List[Dict[str, Any]]
    ):
        """Persist evidence links to database."""
        try:
            async with self.db_pool.acquire() as conn:
                for item in evidence_items:
                    await conn.execute(
                        """
                        INSERT INTO evidence_links (
                            recommendation_id, source_type, url, title, weight,
                            stance, score, snippet, pmid, doi, metadata
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                        """,
                        recommendation_id,
                        item["source_type"],
                        item["url"],
                        item["title"],
                        item["score"],
                        item["stance"],
                        item["score"],
                        item["snippet"],
                        item["pmid"],
                        item["doi"],
                        item["metadata"]
                    )
            
            logger.info("Evidence links persisted",
                       recommendation_id=recommendation_id,
                       count=len(evidence_items))
            
        except Exception as e:
            logger.error("Failed to persist evidence links",
                        recommendation_id=recommendation_id,
                        error=str(e))
    
    async def _log_verification_event(
        self, 
        recommendation_id: str, 
        verification_result, 
        claim_count: int
    ):
        """Log verification event to audit trail."""
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO audit_logs (
                        actor_type, event, entity_type, entity_id, 
                        payload, model_version, ts
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    "system",
                    "evidence.verify.completed",
                    "recommendation",
                    recommendation_id,
                    {
                        "recommendation_id": recommendation_id,
                        "overall_status": verification_result.overall_status,
                        "claim_count": claim_count,
                        "total_evidence": verification_result.total_evidence,
                        "supporting_evidence": verification_result.supporting_evidence,
                        "contradicting_evidence": verification_result.contradicting_evidence,
                        "warning_evidence": verification_result.warning_evidence
                    },
                    "evidence_verifier_v1",
                    datetime.utcnow()
                )
            
            logger.info("Verification event logged",
                       recommendation_id=recommendation_id)
            
        except Exception as e:
            logger.error("Failed to log verification event",
                        recommendation_id=recommendation_id,
                        error=str(e))
    
    def _create_empty_verification_result(self, recommendation_id: str) -> Dict[str, Any]:
        """Create empty verification result for cases with no claims."""
        return {
            "recommendation_id": recommendation_id,
            "overall_status": "flagged",
            "claims": [],
            "total_evidence": 0,
            "supporting_evidence": 0,
            "contradicting_evidence": 0,
            "warning_evidence": 0,
            "verification_timestamp": datetime.utcnow().isoformat()
        }
    
    def _create_error_verification_result(self, recommendation_id: str, error: str) -> Dict[str, Any]:
        """Create error verification result."""
        return {
            "recommendation_id": recommendation_id,
            "overall_status": "flagged",
            "claims": [],
            "total_evidence": 0,
            "supporting_evidence": 0,
            "contradicting_evidence": 0,
            "warning_evidence": 0,
            "verification_timestamp": datetime.utcnow().isoformat(),
            "error": error
        }


def create_verification_service(
    db_pool: asyncpg.Pool,
    config: Dict[str, Any]
) -> EvidenceVerificationService:
    """Factory function to create verification service."""
    
    # Create embeddings provider
    embeddings_provider = create_embeddings_provider(
        provider_type=config.get("embeddings_provider", "openai"),
        api_key=config.get("openai_api_key"),
        model=config.get("embeddings_model", "text-embedding-3-large")
    )
    
    # Create stance classifier
    stance_classifier = create_stance_classifier(
        classifier_type=config.get("stance_classifier", "deberta"),
        model_name=config.get("stance_model", "microsoft/deberta-base-mnli")
    )
    
    # Create retriever
    retriever = create_retriever(
        retriever_type="hybrid",
        db_pool=db_pool,
        embeddings_provider=embeddings_provider,
        k_semantic=config.get("k_semantic", 8),
        k_lexical=config.get("k_lexical", 8)
    )
    
    # Create evidence scorer
    evidence_scorer = create_evidence_scorer()
    
    # Create external API clients
    pubmed_client = None
    if config.get("enable_pubmed", True):
        pubmed_client = create_pubmed_client(
            api_key=config.get("pubmed_api_key"),
            email=config.get("pubmed_email", "your-email@example.com")
        )
    
    openfda_client = None
    if config.get("enable_openfda", True):
        openfda_client = create_openfda_client(
            api_key=config.get("openfda_api_key")
        )
    
    rxnorm_client = None
    if config.get("enable_rxnorm", True):
        rxnorm_client = create_rxnorm_client()
    
    ada_mapper = None
    if config.get("enable_ada", True):
        ada_mapper = create_ada_mapper(
            guidelines_file=config.get("ada_guidelines_file")
        )
    
    return EvidenceVerificationService(
        db_pool=db_pool,
        embeddings_provider=embeddings_provider,
        stance_classifier=stance_classifier,
        retriever=retriever,
        evidence_scorer=evidence_scorer,
        pubmed_client=pubmed_client,
        openfda_client=openfda_client,
        rxnorm_client=rxnorm_client,
        ada_mapper=ada_mapper
    )
