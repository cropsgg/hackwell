"""Patient management endpoints."""

from datetime import datetime
from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse

from ..db import db_manager, QueryBuilder
from ..security import get_current_active_user, require_patient_or_clinician_access, User, audit_logger
from ..models.patients import (
    Patient, PatientCreate, PatientUpdate, PatientResponse, PatientListResponse,
    PatientSummary, PatientSummaryResponse, Vital, VitalCreate, VitalListResponse,
    Medication, MedicationCreate, MedicationListResponse, Condition, ConditionCreate,
    ConditionListResponse, VitalFilter, PatientFilter
)
from ..models.common import ErrorResponse

logger = structlog.get_logger()
router = APIRouter()


@router.post("", response_model=PatientResponse)
async def create_patient(
    patient_data: PatientCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new patient profile."""
    try:
        # Only patients can create their own profile, or admins/clinicians can create profiles
        if not (current_user.has_role("patient") or 
                current_user.has_role("clinician") or 
                current_user.has_role("admin")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to create patient profile"
            )
        
        # Insert patient record
        query = """
        INSERT INTO patients (user_id, demographics, consent)
        VALUES ($1, $2, $3)
        RETURNING id, user_id, demographics, consent, created_at, updated_at
        """
        
        result = await db_manager.execute_query(
            query,
            current_user.id,
            patient_data.demographics.dict(),
            patient_data.consent.dict() if patient_data.consent else {}
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create patient profile"
            )
        
        patient_row = result[0]
        patient = Patient(
            id=patient_row['id'],
            user_id=patient_row['user_id'],
            demographics=patient_row['demographics'],
            consent=patient_row['consent'],
            created_at=patient_row['created_at'],
            updated_at=patient_row['updated_at']
        )
        
        # Log patient creation
        await audit_logger.log_access(
            user=current_user,
            action="create",
            resource="patient",
            resource_id=str(patient.id)
        )
        
        logger.info("Patient profile created", patient_id=str(patient.id), user_id=current_user.id)
        
        return PatientResponse(
            success=True,
            message="Patient profile created successfully",
            data=patient
        )
        
    except Exception as e:
        logger.error("Failed to create patient", error=str(e), user_id=current_user.id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create patient profile"
        )


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(
    patient_id: UUID,
    current_user: User = Depends(require_patient_or_clinician_access)
):
    """Get patient details."""
    try:
        query = QueryBuilder.get_patient_summary(str(patient_id))
        result = await db_manager.execute_query(query, str(patient_id))
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        patient_row = result[0]
        patient = Patient(
            id=patient_row['id'],
            user_id=patient_row['user_id'] if 'user_id' in patient_row else None,
            demographics=patient_row['demographics'],
            consent=patient_row.get('consent', {}),
            created_at=patient_row['created_at'],
            updated_at=patient_row.get('updated_at')
        )
        
        await audit_logger.log_access(
            user=current_user,
            action="view",
            resource="patient",
            resource_id=str(patient_id)
        )
        
        return PatientResponse(
            success=True,
            data=patient
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get patient", patient_id=str(patient_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve patient"
        )


@router.get("/{patient_id}/summary", response_model=PatientSummaryResponse)
async def get_patient_summary(
    patient_id: UUID,
    current_user: User = Depends(require_patient_or_clinician_access)
):
    """Get comprehensive patient summary including risk assessment."""
    try:
        # Get patient basic info
        patient_query = QueryBuilder.get_patient_summary(str(patient_id))
        patient_result = await db_manager.execute_query(patient_query, str(patient_id))
        
        if not patient_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Patient not found"
            )
        
        patient_row = patient_result[0]
        patient = Patient(
            id=patient_row['id'],
            user_id=patient_row.get('user_id'),
            demographics=patient_row['demographics'],
            consent=patient_row.get('consent', {}),
            created_at=patient_row['created_at']
        )
        
        # TODO: Get risk assessment from ML service
        # TODO: Get vital trends
        # TODO: Get active conditions and medications
        
        summary = PatientSummary(
            patient=patient,
            risk_assessment=None,  # TODO: Implement
            vital_trends=[],       # TODO: Implement
            active_conditions=[],  # TODO: Implement
            active_medications=[], # TODO: Implement
            total_vitals=0,
            total_recommendations=0
        )
        
        await audit_logger.log_access(
            user=current_user,
            action="view_summary",
            resource="patient",
            resource_id=str(patient_id)
        )
        
        return PatientSummaryResponse(
            success=True,
            data=summary
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get patient summary", patient_id=str(patient_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve patient summary"
        )


@router.get("/{patient_id}/vitals", response_model=VitalListResponse)
async def get_patient_vitals(
    patient_id: UUID,
    vital_filter: VitalFilter = Depends(),
    current_user: User = Depends(require_patient_or_clinician_access)
):
    """Get patient vital signs with filtering."""
    try:
        query = QueryBuilder.get_patient_vitals(str(patient_id), vital_filter.limit)
        vitals_result = await db_manager.execute_query(query, str(patient_id), vital_filter.limit)
        
        # Convert to Vital objects
        vitals = []
        for row in vitals_result:
            vital = Vital(
                id=row.get('id'),  # May need to generate if not in query
                patient_id=patient_id,
                type=row['type'],
                value=row['value'],
                unit=row['unit'],
                ts=row['ts'],
                source=row.get('source', 'unknown')
            )
            vitals.append(vital)
        
        await audit_logger.log_access(
            user=current_user,
            action="view_vitals",
            resource="patient",
            resource_id=str(patient_id)
        )
        
        return VitalListResponse(
            success=True,
            data=vitals,
            total=len(vitals)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get patient vitals", patient_id=str(patient_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve patient vitals"
        )


@router.post("/{patient_id}/vitals", response_model=VitalListResponse)
async def create_vital(
    patient_id: UUID,
    vital_data: VitalCreate,
    current_user: User = Depends(require_patient_or_clinician_access)
):
    """Create a new vital sign measurement."""
    try:
        # Ensure patient_id matches
        vital_data.patient_id = patient_id
        
        query = """
        INSERT INTO vitals (patient_id, type, value, unit, source, ts)
        VALUES ($1, $2, $3, $4, $5, $6)
        RETURNING id, patient_id, type, value, unit, source, ts, created_at
        """
        
        ts = vital_data.ts or datetime.utcnow()
        
        result = await db_manager.execute_query(
            query,
            str(patient_id),
            vital_data.type.value,
            vital_data.value,
            vital_data.unit.value,
            vital_data.source.value,
            ts
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create vital measurement"
            )
        
        vital_row = result[0]
        vital = Vital(
            id=vital_row['id'],
            patient_id=vital_row['patient_id'],
            type=vital_row['type'],
            value=vital_row['value'],
            unit=vital_row['unit'],
            source=vital_row['source'],
            ts=vital_row['ts'],
            created_at=vital_row['created_at']
        )
        
        await audit_logger.log_access(
            user=current_user,
            action="create_vital",
            resource="patient",
            resource_id=str(patient_id),
            metadata={"vital_type": vital_data.type.value, "value": vital_data.value}
        )
        
        logger.info("Vital measurement created", 
                   patient_id=str(patient_id), 
                   vital_type=vital_data.type.value,
                   user_id=current_user.id)
        
        return VitalListResponse(
            success=True,
            message="Vital measurement recorded successfully",
            data=[vital],
            total=1
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create vital", patient_id=str(patient_id), error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record vital measurement"
        )


# TODO: Implement remaining endpoints
# - Update patient
# - Get/create medications
# - Get/create conditions
# - Delete operations (with proper authorization)
# - Bulk operations

@router.get("")
async def list_patients(
    current_user: User = Depends(get_current_active_user),
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100)
):
    """List patients (for clinicians and admins)."""
    if not (current_user.has_role("clinician") or current_user.has_role("admin")):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to list patients"
        )
    
    # TODO: Implement patient listing with proper RLS
    return PatientListResponse(
        success=True,
        data=[],
        total=0,
        offset=offset,
        limit=limit
    )
