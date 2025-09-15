"""Recommendation management endpoints."""

from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Query
import httpx

from ..db import db_manager, QueryBuilder
from ..security import (
    get_current_active_user, require_patient_or_clinician_access, 
    require_clinician, User, audit_logger
)
from ..models.recommendations import (
    RecommendationRequest, RecommendationGenerationResponse,
    RecommendationWithEvidence, RecommendationResponse, RecommendationListResponse,
    ClinicianAction, ClinicianActionResponse
)
from ..models.common import BaseResponse
from ..config import settings

logger = structlog.get_logger()
router = APIRouter()


@router.post("/generate", response_model=RecommendationGenerationResponse)
async def generate_recommendation(
    request: RecommendationRequest,
    current_user: User = Depends(require_patient_or_clinician_access)
):
    """Generate new recommendation for a patient."""
    try:
        # Verify patient access
        if not current_user.can_access_patient(str(request.patient_id)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this patient"
            )
        
        # Call agent orchestrator service to generate recommendation
        async with httpx.AsyncClient() as client:
            agent_response = await client.post(
                f"{settings.agents_service_url}/orchestrator/generate",
                json={
                    "patient_id": str(request.patient_id),
                    "force_refresh": request.force_refresh,
                    "include_evidence": request.include_evidence,
                    "model_version": request.model_version,
                    "context": request.context
                },
                timeout=30.0
            )
            
            if agent_response.status_code != 200:
                logger.error("Agent service failed", 
                           status=agent_response.status_code,
                           response=agent_response.text)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to generate recommendation"
                )
            
            agent_data = agent_response.json()
        
        # Log recommendation generation
        await audit_logger.log_recommendation_action(
            user=current_user,
            action="generate",
            recommendation_id=agent_data.get("recommendation_id"),
            patient_id=str(request.patient_id),
            model_version=agent_data.get("model_version")
        )
        
        logger.info("Recommendation generated", 
                   patient_id=str(request.patient_id),
                   recommendation_id=agent_data.get("recommendation_id"),
                   user_id=current_user.id)
        
        # TODO: Convert agent response to RecommendationWithEvidence
        recommendation = RecommendationWithEvidence(
            id=agent_data.get("recommendation_id"),
            patient_id=request.patient_id,
            # ... populate from agent_data
        )
        
        return RecommendationGenerationResponse(
            success=True,
            message="Recommendation generated successfully",
            data=recommendation,
            generation_metadata=agent_data.get("metadata", {})
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to generate recommendation", 
                    patient_id=str(request.patient_id), 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate recommendation"
        )


@router.get("/{recommendation_id}", response_model=RecommendationResponse)
async def get_recommendation(
    recommendation_id: UUID,
    current_user: User = Depends(get_current_active_user)
):
    """Get recommendation details with evidence."""
    try:
        # Get recommendation with evidence
        query = """
        SELECT 
            r.*,
            p.id as patient_id,
            COALESCE(
                json_agg(
                    json_build_object(
                        'id', e.id,
                        'source_type', e.source_type,
                        'url', e.url,
                        'title', e.title,
                        'weight', e.weight,
                        'snippet', e.snippet,
                        'metadata', e.metadata
                    )
                ) FILTER (WHERE e.id IS NOT NULL),
                '[]'::json
            ) as evidence_links
        FROM recommendations r
        JOIN patients p ON p.id = r.patient_id
        LEFT JOIN evidence_links e ON e.recommendation_id = r.id
        WHERE r.id = $1
        GROUP BY r.id, p.id
        """
        
        result = await db_manager.execute_query(query, str(recommendation_id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Recommendation not found"
            )
        
        row = result[0]
        
        # Check patient access
        if not current_user.can_access_patient(str(row['patient_id'])):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this patient's recommendations"
            )
        
        # TODO: Convert row to RecommendationWithEvidence
        recommendation = RecommendationWithEvidence(
            id=row['id'],
            patient_id=row['patient_id'],
            # ... populate from row data
        )
        
        await audit_logger.log_access(
            user=current_user,
            action="view",
            resource="recommendation",
            resource_id=str(recommendation_id)
        )
        
        return RecommendationResponse(
            success=True,
            data=recommendation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get recommendation", 
                    recommendation_id=str(recommendation_id), 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve recommendation"
        )


@router.post("/{recommendation_id}/action", response_model=ClinicianActionResponse)
async def clinician_action(
    recommendation_id: UUID,
    action: ClinicianAction,
    current_user: User = Depends(require_clinician)
):
    """Perform clinician action on recommendation (approve/reject/override)."""
    try:
        # Verify clinician has access to this recommendation's patient
        patient_query = """
        SELECT r.patient_id 
        FROM recommendations r 
        JOIN clinician_patients cp ON cp.patient_id = r.patient_id
        WHERE r.id = $1 AND cp.clinician_user_id = $2 AND cp.active = true
        """
        
        access_result = await db_manager.execute_query(
            patient_query, str(recommendation_id), current_user.id
        )
        
        if not access_result:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this recommendation"
            )
        
        patient_id = str(access_result[0]['patient_id'])
        
        # Update recommendation status
        if action.action == "approve":
            new_status = "approved"
        elif action.action == "reject":
            new_status = "rejected"
        elif action.action == "override":
            new_status = "approved"  # Approved with override
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid action: {action.action}"
            )
        
        update_query = """
        UPDATE recommendations 
        SET 
            status = $1,
            approved_by_user_id = $2,
            approved_at = NOW(),
            clinician_notes = $3,
            justification = $4,
            updated_at = NOW()
        WHERE id = $5
        RETURNING *
        """
        
        update_result = await db_manager.execute_query(
            update_query,
            new_status,
            current_user.id,
            action.notes,
            action.justification,
            str(recommendation_id)
        )
        
        if not update_result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update recommendation"
            )
        
        # Create audit log
        audit_log_id = await db_manager.call_function(
            "create_audit_log",
            "clinician",
            current_user.id,
            f"recommendation.{action.action}",
            "recommendation",
            str(recommendation_id),
            {
                "patient_id": patient_id,
                "action": action.action,
                "justification": action.justification,
                "notes": action.notes,
                "model_version": update_result[0].get("model_version")
            }
        )
        
        logger.info("Clinician action completed",
                   recommendation_id=str(recommendation_id),
                   action=action.action,
                   clinician_id=current_user.id,
                   patient_id=patient_id)
        
        return ClinicianActionResponse(
            success=True,
            message=f"Recommendation {action.action} completed successfully",
            data={
                "recommendation_id": str(recommendation_id),
                "action": action.action,
                "status": new_status,
                "timestamp": update_result[0]["updated_at"].isoformat()
            },
            audit_log_id=audit_log_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process clinician action",
                    recommendation_id=str(recommendation_id),
                    action=action.action,
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process clinician action"
        )


@router.get("/patient/{patient_id}", response_model=RecommendationListResponse)
async def get_patient_recommendations(
    patient_id: UUID,
    current_user: User = Depends(require_patient_or_clinician_access),
    status_filter: Optional[str] = Query(None, description="Filter by status"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """Get recommendations for a specific patient."""
    try:
        # Verify patient access
        if not current_user.can_access_patient(str(patient_id)):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied for this patient"
            )
        
        query = QueryBuilder.get_recommendations_with_evidence(str(patient_id))
        result = await db_manager.execute_query(query, str(patient_id))
        
        # TODO: Apply status filter and pagination
        # TODO: Convert to RecommendationWithEvidence objects
        recommendations = []
        
        await audit_logger.log_access(
            user=current_user,
            action="list_recommendations",
            resource="patient",
            resource_id=str(patient_id)
        )
        
        return RecommendationListResponse(
            success=True,
            data=recommendations,
            total=len(result)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get patient recommendations",
                    patient_id=str(patient_id),
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve patient recommendations"
        )
