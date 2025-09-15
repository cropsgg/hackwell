"""Clinician-specific endpoints."""

from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Query

from ..db import db_manager, QueryBuilder
from ..security import require_clinician, User, audit_logger
from ..models.patients import PatientListResponse, Patient
from ..models.recommendations import RecommendationListResponse
from ..models.common import BaseResponse

logger = structlog.get_logger()
router = APIRouter()


@router.get("/patients", response_model=PatientListResponse)
async def get_assigned_patients(
    current_user: User = Depends(require_clinician),
    risk_filter: Optional[str] = Query(None, description="Filter by risk: low, moderate, high"),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """Get patients assigned to the current clinician."""
    try:
        query = QueryBuilder.get_clinician_patients(current_user.id)
        result = await db_manager.execute_query(query, current_user.id)
        
        patients = []
        for row in result[offset:offset + limit]:
            patient = Patient(
                id=row['id'],
                demographics=row['demographics'],
                # Add risk summary to metadata for dashboard
                consent={"latest_risk_score": row.get('latest_risk_score', 0)}
            )
            patients.append(patient)
        
        await audit_logger.log_access(
            user=current_user,
            action="list_assigned_patients",
            resource="patient_list"
        )
        
        return PatientListResponse(
            success=True,
            data=patients,
            total=len(result),
            offset=offset,
            limit=limit
        )
        
    except Exception as e:
        logger.error("Failed to get assigned patients", clinician_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve assigned patients"
        )


@router.get("/dashboard")
async def get_clinician_dashboard(
    current_user: User = Depends(require_clinician)
):
    """Get clinician dashboard data."""
    try:
        # TODO: Implement dashboard with:
        # - High-risk patients needing attention
        # - Pending recommendations for approval
        # - Recent patient activities
        # - Performance metrics
        
        dashboard_data = {
            "high_risk_patients": 0,
            "pending_approvals": 0,
            "recent_activities": [],
            "performance_metrics": {
                "avg_approval_time_hours": 0,
                "recommendations_approved": 0,
                "patients_managed": 0
            }
        }
        
        await audit_logger.log_access(
            user=current_user,
            action="view_dashboard",
            resource="dashboard"
        )
        
        return BaseResponse(
            success=True,
            data=dashboard_data
        )
        
    except Exception as e:
        logger.error("Failed to get dashboard", clinician_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve dashboard data"
        )


@router.get("/recommendations/pending", response_model=RecommendationListResponse)
async def get_pending_recommendations(
    current_user: User = Depends(require_clinician),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """Get recommendations pending clinician approval."""
    try:
        query = """
        SELECT 
            r.id,
            r.patient_id,
            r.snapshot_ts,
            r.careplan,
            r.explainer,
            r.model_version,
            r.status,
            r.risk_score,
            r.risk_category,
            r.created_at,
            p.demographics
        FROM recommendations r
        JOIN patients p ON p.id = r.patient_id
        JOIN clinician_patients cp ON cp.patient_id = p.id
        WHERE cp.clinician_user_id = $1
        AND r.status = 'pending'
        ORDER BY r.created_at DESC
        OFFSET $2 LIMIT $3
        """
        
        result = await db_manager.execute_query(query, current_user.id, offset, limit)
        
        # TODO: Convert to RecommendationWithEvidence objects
        recommendations = []
        
        await audit_logger.log_access(
            user=current_user,
            action="view_pending_recommendations",
            resource="recommendations"
        )
        
        return RecommendationListResponse(
            success=True,
            data=recommendations,
            total=len(result)
        )
        
    except Exception as e:
        logger.error("Failed to get pending recommendations", clinician_id=current_user.id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve pending recommendations"
        )


# TODO: Implement additional clinician endpoints
# - Patient assignment management
# - Recommendation approval/override
# - Clinical notes and documentation
# - Performance analytics
