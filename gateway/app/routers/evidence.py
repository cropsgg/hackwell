"""Evidence verification endpoints."""

from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Query
import httpx

from ..security import get_current_active_user, User, audit_logger
from ..models.recommendations import EvidenceLink, EvidenceVerification
from ..models.common import BaseResponse
from ..config import settings

logger = structlog.get_logger()
router = APIRouter()


@router.get("/recommendation/{recommendation_id}")
async def get_recommendation_evidence(
    recommendation_id: UUID,
    current_user: User = Depends(get_current_active_user)
):
    """Get evidence links for a specific recommendation."""
    try:
        # TODO: Verify user can access this recommendation
        
        # Get evidence from database
        from ..db import db_manager
        
        query = """
        SELECT 
            e.*,
            r.patient_id
        FROM evidence_links e
        JOIN recommendations r ON r.id = e.recommendation_id
        WHERE e.recommendation_id = $1
        ORDER BY e.weight DESC
        """
        
        result = await db_manager.execute_query(query, str(recommendation_id))
        
        if result:
            patient_id = str(result[0]['patient_id'])
            if not current_user.can_access_patient(patient_id):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied for this recommendation"
                )
        
        evidence_links = []
        for row in result:
            evidence = EvidenceLink(
                source_type=row['source_type'],
                url=row['url'],
                title=row['title'],
                weight=row['weight'],
                snippet=row['snippet'],
                metadata=row.get('metadata', {})
            )
            evidence_links.append(evidence)
        
        await audit_logger.log_access(
            user=current_user,
            action="view_evidence",
            resource="recommendation",
            resource_id=str(recommendation_id)
        )
        
        return BaseResponse(
            success=True,
            data={
                "recommendation_id": str(recommendation_id),
                "evidence_links": [e.dict() for e in evidence_links],
                "total_evidence": len(evidence_links)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get recommendation evidence",
                    recommendation_id=str(recommendation_id),
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evidence"
        )


@router.post("/verify")
async def verify_evidence(
    request_data: dict,
    current_user: User = Depends(get_current_active_user)
):
    """Manually trigger evidence verification for a care plan."""
    try:
        # Call evidence verifier service
        async with httpx.AsyncClient() as client:
            verifier_response = await client.post(
                f"{settings.verifier_service_url}/verify",
                json=request_data,
                timeout=60.0  # Evidence lookup can be slow
            )
            
            if verifier_response.status_code != 200:
                logger.error("Evidence verifier failed",
                           status=verifier_response.status_code,
                           response=verifier_response.text)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Evidence verification failed"
                )
            
            verification_data = verifier_response.json()
        
        await audit_logger.log_access(
            user=current_user,
            action="verify_evidence",
            resource="evidence_verification"
        )
        
        return BaseResponse(
            success=True,
            message="Evidence verification completed",
            data=verification_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to verify evidence", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to verify evidence"
        )


@router.get("/sources/pubmed")
async def search_pubmed(
    query: str = Query(..., description="PubMed search query"),
    limit: int = Query(10, ge=1, le=50),
    current_user: User = Depends(get_current_active_user)
):
    """Search PubMed for evidence."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.verifier_service_url}/pubmed/search",
                params={"query": query, "limit": limit},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="PubMed search failed"
                )
            
            search_results = response.json()
        
        await audit_logger.log_access(
            user=current_user,
            action="search_pubmed",
            resource="evidence_search",
            metadata={"query": query, "limit": limit}
        )
        
        return BaseResponse(
            success=True,
            data=search_results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("PubMed search failed", query=query, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search PubMed"
        )


@router.get("/sources/openfda")
async def search_openfda(
    drug_name: str = Query(..., description="Drug name to search"),
    current_user: User = Depends(get_current_active_user)
):
    """Search openFDA for drug safety information."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{settings.verifier_service_url}/openfda/drug",
                params={"drug_name": drug_name},
                timeout=30.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="openFDA search failed"
                )
            
            drug_info = response.json()
        
        await audit_logger.log_access(
            user=current_user,
            action="search_openfda",
            resource="evidence_search",
            metadata={"drug_name": drug_name}
        )
        
        return BaseResponse(
            success=True,
            data=drug_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("openFDA search failed", drug_name=drug_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search openFDA"
        )


@router.get("/guidelines/ada")
async def get_ada_guidelines(
    topic: str = Query(..., description="Guideline topic"),
    current_user: User = Depends(get_current_active_user)
):
    """Get ADA guidelines for a specific topic."""
    try:
        # TODO: Implement ADA guideline lookup
        # This would typically involve a curated database of guidelines
        # mapped to specific topics and recommendations
        
        guidelines = {
            "topic": topic,
            "guidelines": [
                {
                    "title": "ADA Standards of Care in Diabetes 2024",
                    "url": "https://diabetesjournals.org/care/issue/47/Supplement_1",
                    "relevance": "high",
                    "recommendations": []
                }
            ]
        }
        
        await audit_logger.log_access(
            user=current_user,
            action="search_ada_guidelines",
            resource="evidence_search",
            metadata={"topic": topic}
        )
        
        return BaseResponse(
            success=True,
            data=guidelines
        )
        
    except Exception as e:
        logger.error("ADA guidelines search failed", topic=topic, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to search ADA guidelines"
        )
