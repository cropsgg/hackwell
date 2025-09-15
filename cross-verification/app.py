"""FastAPI application for Evidence Verification Service."""

import asyncio
from typing import Dict, Any, Optional
import structlog
import asyncpg
from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from service import EvidenceVerificationService, VerificationRequest, create_verification_service
from models import (
    EvidenceVerificationResponse, 
    EvidenceItem, 
    ClaimResult
)
from config import get_config

logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Evidence Verification Service",
    description="RAG-based evidence verification for care plan recommendations",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
verification_service: Optional[EvidenceVerificationService] = None


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    components: Dict[str, str]


@app.on_event("startup")
async def startup_event():
    """Initialize the verification service on startup."""
    global verification_service
    
    try:
        # Database configuration
        config = get_config()
        database_url = config.database_url
        
        # Create database pool
        db_pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        
        # Service configuration
        service_config = {
            "embeddings_provider": config.embeddings_provider,
            "openai_api_key": config.openai_api_key,
            "embeddings_model": config.embeddings_model,
            "stance_classifier": "deberta",
            "stance_model": config.stance_model,
            "k_semantic": config.k_semantic,
            "k_lexical": config.k_lexical,
            "enable_pubmed": True,
            "pubmed_email": config.pubmed_email,
            "enable_openfda": True,
            "enable_rxnorm": True,
            "enable_ada": True
        }
        
        # Create verification service
        verification_service = create_verification_service(db_pool, service_config)
        
        logger.info("Evidence verification service initialized")
        
    except Exception as e:
        logger.error("Failed to initialize verification service", error=str(e))
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global verification_service
    
    if verification_service and verification_service.db_pool:
        await verification_service.db_pool.close()
        logger.info("Evidence verification service shutdown")


def get_verification_service() -> EvidenceVerificationService:
    """Dependency to get verification service."""
    if verification_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Verification service not available"
        )
    return verification_service


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    components = {
        "database": "unknown",
        "embeddings": "unknown",
        "stance_classifier": "unknown",
        "retriever": "unknown",
        "external_apis": "unknown"
    }
    
    if verification_service:
        try:
            # Check database
            async with verification_service.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            components["database"] = "healthy"
            
            # Check other components
            components["embeddings"] = "healthy"
            components["stance_classifier"] = "healthy"
            components["retriever"] = "healthy"
            components["external_apis"] = "healthy"
            
        except Exception as e:
            logger.warning("Health check failed", error=str(e))
            components["database"] = "unhealthy"
    
    return HealthCheck(
        status="healthy" if all(v == "healthy" for v in components.values()) else "degraded",
        service="evidence-verification",
        version="1.0.0",
        components=components
    )


@app.post("/verify", response_model=EvidenceVerificationResponse)
async def verify_evidence(
    request: VerificationRequest,
    service: EvidenceVerificationService = Depends(get_verification_service)
):
    """Verify care plan recommendations with evidence."""
    try:
        logger.info("Evidence verification request received",
                   recommendation_id=request.recommendation_id)
        
        # Verify the recommendation
        result = await service.verify_recommendation(request)
        
        # Convert to response format
        response = EvidenceVerificationResponse(
            recommendation_id=result["recommendation_id"],
            overall_status=result["overall_status"],
            claims=[
                ClaimResult(
                    claim_id=claim["claim_id"],
                    claim_text=claim["claim_text"],
                    support_score=claim["support_score"],
                    contradict_score=claim["contradict_score"],
                    items=[
                        EvidenceItem(
                            source_type=item["source_type"],
                            title=item["title"],
                            url=item["url"],
                            pmid=item["pmid"],
                            doi=item["doi"],
                            stance=item["stance"],
                            score=item["score"],
                            snippet=item["snippet"],
                            metadata=item["metadata"]
                        )
                        for item in claim["items"]
                    ],
                    verdict=claim["verdict"]
                )
                for claim in result["claims"]
            ],
            total_evidence=result["total_evidence"],
            supporting_evidence=result["supporting_evidence"],
            contradicting_evidence=result["contradicting_evidence"],
            warning_evidence=result["warning_evidence"],
            verification_timestamp=result["verification_timestamp"]
        )
        
        logger.info("Evidence verification completed",
                   recommendation_id=request.recommendation_id,
                   overall_status=result["overall_status"])
        
        return response
        
    except Exception as e:
        logger.error("Evidence verification failed",
                    recommendation_id=request.recommendation_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evidence verification failed: {str(e)}"
        )


@app.post("/verify/batch")
async def verify_evidence_batch(
    requests: list[VerificationRequest],
    service: EvidenceVerificationService = Depends(get_verification_service)
):
    """Verify multiple care plan recommendations in batch."""
    try:
        logger.info("Batch evidence verification request received",
                   count=len(requests))
        
        # Process requests concurrently
        tasks = [service.verify_recommendation(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "index": i,
                    "recommendation_id": requests[i].recommendation_id,
                    "error": str(result)
                })
            else:
                successful_results.append(result)
        
        logger.info("Batch evidence verification completed",
                   successful=len(successful_results),
                   failed=len(failed_results))
        
        return {
            "successful": successful_results,
            "failed": failed_results,
            "total_processed": len(requests)
        }
        
    except Exception as e:
        logger.error("Batch evidence verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch evidence verification failed: {str(e)}"
        )


@app.get("/evidence/{recommendation_id}")
async def get_evidence(
    recommendation_id: str,
    service: EvidenceVerificationService = Depends(get_verification_service)
):
    """Get evidence links for a specific recommendation."""
    try:
        async with service.db_pool.acquire() as conn:
            # Get evidence links
            evidence_links = await conn.fetch(
                """
                SELECT 
                    id, source_type, url, title, weight, stance, score, 
                    snippet, pmid, doi, metadata, added_at
                FROM evidence_links
                WHERE recommendation_id = $1
                ORDER BY score DESC
                """,
                recommendation_id
            )
            
            # Get recommendation info
            recommendation = await conn.fetchrow(
                """
                SELECT id, patient_id, status, created_at
                FROM recommendations
                WHERE id = $1
                """,
                recommendation_id
            )
            
            if not recommendation:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Recommendation not found"
                )
            
            return {
                "recommendation_id": recommendation_id,
                "patient_id": str(recommendation["patient_id"]),
                "status": recommendation["status"],
                "created_at": recommendation["created_at"].isoformat(),
                "evidence_links": [
                    {
                        "id": str(link["id"]),
                        "source_type": link["source_type"],
                        "url": link["url"],
                        "title": link["title"],
                        "weight": float(link["weight"]),
                        "stance": link["stance"],
                        "score": float(link["score"]),
                        "snippet": link["snippet"],
                        "pmid": link["pmid"],
                        "doi": link["doi"],
                        "metadata": link["metadata"],
                        "added_at": link["added_at"].isoformat()
                    }
                    for link in evidence_links
                ],
                "total_evidence": len(evidence_links)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get evidence",
                    recommendation_id=recommendation_id,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evidence"
        )


@app.get("/stats")
async def get_verification_stats(
    service: EvidenceVerificationService = Depends(get_verification_service)
):
    """Get verification statistics."""
    try:
        async with service.db_pool.acquire() as conn:
            # Get overall stats
            stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_recommendations,
                    COUNT(CASE WHEN status = 'approved' THEN 1 END) as approved,
                    COUNT(CASE WHEN status = 'flagged' THEN 1 END) as flagged,
                    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending
                FROM recommendations
                WHERE created_at >= NOW() - INTERVAL '30 days'
                """
            )
            
            # Get evidence stats
            evidence_stats = await conn.fetchrow(
                """
                SELECT 
                    COUNT(*) as total_evidence,
                    COUNT(CASE WHEN stance = 'support' THEN 1 END) as supporting,
                    COUNT(CASE WHEN stance = 'contradict' THEN 1 END) as contradicting,
                    COUNT(CASE WHEN stance = 'warning' THEN 1 END) as warnings
                FROM evidence_links
                WHERE added_at >= NOW() - INTERVAL '30 days'
                """
            )
            
            return {
                "recommendations": {
                    "total": stats["total_recommendations"],
                    "approved": stats["approved"],
                    "flagged": stats["flagged"],
                    "pending": stats["pending"]
                },
                "evidence": {
                    "total": evidence_stats["total_evidence"],
                    "supporting": evidence_stats["supporting"],
                    "contradicting": evidence_stats["contradicting"],
                    "warnings": evidence_stats["warnings"]
                },
                "period": "30 days"
            }
            
    except Exception as e:
        logger.error("Failed to get verification stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
