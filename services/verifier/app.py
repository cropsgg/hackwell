"""Evidence Verification Service - FastAPI Application."""

import asyncio
import os
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from pubmed_client import PubMedClient
from openfda_client import OpenFDAClient
from rxnorm_client import RxNormClient
from scorer import EvidenceScorer

# Configure logging
logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Evidence Verification Service",
    description="Verify care plan recommendations using PubMed, openFDA, and RxNorm",
    version="1.0.0"
)

# Pydantic models
class CarePlanComponent(BaseModel):
    """Individual care plan component."""
    category: str = Field(..., description="Category (dietary, exercise, medication, etc.)")
    recommendations: List[str] = Field(default_factory=list)
    specific_interventions: List[str] = Field(default_factory=list)


class EvidenceVerificationRequest(BaseModel):
    """Evidence verification request."""
    patient_context: Dict[str, Any] = Field(default_factory=dict)
    care_plan: Dict[str, Any] = Field(..., description="Care plan to verify")
    focus_areas: List[str] = Field(default_factory=list, description="Specific areas to focus verification")
    include_safety_check: bool = True
    max_evidence_per_category: int = 5


class EvidenceItem(BaseModel):
    """Individual evidence item."""
    source_type: str
    title: str
    url: Optional[str] = None
    snippet: Optional[str] = None
    quality_score: float
    weight: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvidenceVerificationResponse(BaseModel):
    """Evidence verification response."""
    overall_score: float
    status: str  # approved, flagged, rejected
    evidence_links: List[EvidenceItem]
    safety_alerts: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    quality_breakdown: Dict[str, int] = Field(default_factory=dict)
    verification_metadata: Dict[str, Any] = Field(default_factory=dict)


class DrugSafetyRequest(BaseModel):
    """Drug safety verification request."""
    medications: List[Dict[str, Any]]
    patient_context: Dict[str, Any] = Field(default_factory=dict)


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = "healthy"
    services_available: Dict[str, bool] = Field(default_factory=dict)


# Global clients
pubmed_client = None
openfda_client = None
rxnorm_client = None
evidence_scorer = None


@app.on_event("startup")
async def startup_event():
    """Initialize external API clients."""
    global pubmed_client, openfda_client, rxnorm_client, evidence_scorer
    
    try:
        pubmed_client = PubMedClient()
        openfda_client = OpenFDAClient()
        rxnorm_client = RxNormClient()
        evidence_scorer = EvidenceScorer()
        
        logger.info("Evidence Verification service started")
        
    except Exception as e:
        logger.error("Failed to initialize verification service", error=str(e))


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    services = {
        "pubmed": pubmed_client is not None,
        "openfda": openfda_client is not None,
        "rxnorm": rxnorm_client is not None,
        "scorer": evidence_scorer is not None
    }
    
    overall_status = "healthy" if all(services.values()) else "degraded"
    
    return HealthCheck(
        status=overall_status,
        services_available=services
    )


@app.post("/verify", response_model=EvidenceVerificationResponse)
async def verify_evidence(request: EvidenceVerificationRequest):
    """Verify care plan recommendations with evidence."""
    if not all([pubmed_client, openfda_client, rxnorm_client, evidence_scorer]):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Evidence verification services not available"
        )
    
    try:
        logger.info("Starting evidence verification", 
                   care_plan_categories=list(request.care_plan.keys()))
        
        # Collect evidence for each care plan component
        all_evidence = []
        safety_alerts = []
        
        # Process care plan components
        for category, recommendations in request.care_plan.items():
            if isinstance(recommendations, dict):
                category_evidence = await _verify_category(
                    category, recommendations, request.patient_context,
                    request.max_evidence_per_category
                )
                all_evidence.extend(category_evidence)
        
        # Medication safety check if requested
        if request.include_safety_check:
            medications = request.patient_context.get('medications', [])
            if medications:
                med_safety = await _check_medication_safety(medications)
                safety_alerts.extend(med_safety.get('safety_alerts', []))
                
                # Add drug interaction evidence
                if med_safety.get('interactions'):
                    interaction_evidence = _format_interaction_evidence(med_safety['interactions'])
                    all_evidence.extend(interaction_evidence)
        
        # Score the evidence collection
        scoring_result = evidence_scorer.score_evidence_collection(all_evidence)
        
        # Format response
        evidence_items = []
        for evidence in scoring_result.get('scored_evidence', []):
            item = EvidenceItem(
                source_type=evidence['source_type'],
                title=evidence.get('title', ''),
                url=evidence.get('url'),
                snippet=evidence.get('content') or evidence.get('snippet'),
                quality_score=evidence['quality_score'],
                weight=evidence['weight'],
                metadata=evidence.get('metadata', {})
            )
            evidence_items.append(item)
        
        # Sort by quality score
        evidence_items.sort(key=lambda x: x.quality_score, reverse=True)
        
        response = EvidenceVerificationResponse(
            overall_score=scoring_result['overall_score'],
            status=scoring_result['status'],
            evidence_links=evidence_items[:20],  # Limit to top 20
            safety_alerts=safety_alerts + scoring_result.get('flags', []),
            warnings=scoring_result.get('warnings', []),
            quality_breakdown=scoring_result.get('quality_breakdown', {}),
            verification_metadata={
                'evidence_sources_checked': len(all_evidence),
                'verification_timestamp': scoring_result.get('timestamp'),
                'focus_areas': request.focus_areas
            }
        )
        
        logger.info("Evidence verification completed",
                   overall_score=response.overall_score,
                   status=response.status,
                   evidence_count=len(evidence_items))
        
        return response
        
    except Exception as e:
        logger.error("Evidence verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Evidence verification failed"
        )


@app.post("/drug-safety")
async def check_drug_safety(request: DrugSafetyRequest):
    """Check drug safety and interactions."""
    if not rxnorm_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RxNorm service not available"
        )
    
    try:
        safety_result = await rxnorm_client.check_medication_safety(request.medications)
        
        # Get detailed drug information from OpenFDA
        drug_details = []
        if openfda_client:
            for medication in request.medications:
                drug_name = medication.get('name', '')
                if drug_name:
                    drug_info = await openfda_client.get_comprehensive_drug_info(drug_name)
                    drug_details.append(drug_info)
        
        return {
            "medication_safety": safety_result,
            "drug_details": drug_details,
            "overall_safety_score": _calculate_overall_safety_score(safety_result, drug_details)
        }
        
    except Exception as e:
        logger.error("Drug safety check failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Drug safety check failed"
        )


@app.get("/pubmed/search")
async def search_pubmed(query: str, limit: int = 10):
    """Search PubMed for evidence."""
    if not pubmed_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PubMed service not available"
        )
    
    try:
        results = await pubmed_client.search_and_summarize(query, limit)
        return {"query": query, "results": results, "count": len(results)}
        
    except Exception as e:
        logger.error("PubMed search failed", query=query, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="PubMed search failed"
        )


@app.get("/openfda/drug")
async def get_drug_info(drug_name: str):
    """Get FDA drug information."""
    if not openfda_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OpenFDA service not available"
        )
    
    try:
        drug_info = await openfda_client.get_comprehensive_drug_info(drug_name)
        return drug_info
        
    except Exception as e:
        logger.error("OpenFDA lookup failed", drug_name=drug_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OpenFDA lookup failed"
        )


@app.get("/rxnorm/normalize")
async def normalize_drug(drug_name: str):
    """Normalize drug name using RxNorm."""
    if not rxnorm_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RxNorm service not available"
        )
    
    try:
        normalized = await rxnorm_client.normalize_drug_list([drug_name])
        return normalized[0] if normalized else {}
        
    except Exception as e:
        logger.error("RxNorm normalization failed", drug_name=drug_name, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="RxNorm normalization failed"
        )


# Helper functions
async def _verify_category(category: str, recommendations: Dict[str, Any], 
                          patient_context: Dict[str, Any], max_evidence: int) -> List[Dict[str, Any]]:
    """Verify evidence for a specific care plan category."""
    evidence = []
    
    try:
        # Build search queries based on category and recommendations
        queries = _build_evidence_queries(category, recommendations, patient_context)
        
        # Search PubMed for each query
        for query in queries[:3]:  # Limit to top 3 queries per category
            pubmed_results = await pubmed_client.search_and_summarize(query, max_evidence)
            
            for result in pubmed_results:
                evidence_item = {
                    'source_type': result.get('study_type', 'observational'),
                    'title': result.get('title', ''),
                    'url': result.get('url', ''),
                    'content': f"{result.get('title', '')} - {category} evidence",
                    'snippet': result.get('title', '')[:200],
                    'metadata': {
                        'pmid': result.get('pmid'),
                        'journal': result.get('journal'),
                        'pub_date': result.get('pub_date'),
                        'quality_score': result.get('quality_score', 0.5),
                        'category': category,
                        'search_query': query
                    }
                }
                evidence.append(evidence_item)
        
        # Add guideline evidence for diabetes/cardiovascular categories
        if category in ['dietary', 'exercise', 'medication_safety', 'monitoring']:
            guideline_evidence = _get_guideline_evidence(category, recommendations)
            evidence.extend(guideline_evidence)
        
    except Exception as e:
        logger.error("Category verification failed", category=category, error=str(e))
    
    return evidence


def _build_evidence_queries(category: str, recommendations: Dict[str, Any], 
                           patient_context: Dict[str, Any]) -> List[str]:
    """Build PubMed search queries for evidence verification."""
    queries = []
    
    # Get patient conditions for context
    conditions = patient_context.get('conditions', [])
    condition_terms = []
    for condition in conditions:
        if condition.get('active') and 'diabetes' in condition.get('name', '').lower():
            condition_terms.append('diabetes')
        elif condition.get('active') and 'hypertension' in condition.get('name', '').lower():
            condition_terms.append('hypertension')
    
    if not condition_terms:
        condition_terms = ['diabetes', 'cardiovascular disease']  # Default
    
    # Category-specific query building
    if category == 'dietary':
        for condition in condition_terms:
            queries.append(f'diet therapy AND {condition} AND glycemic control')
            queries.append(f'nutritional intervention AND {condition} AND outcomes')
    
    elif category == 'exercise':
        for condition in condition_terms:
            queries.append(f'exercise therapy AND {condition} AND clinical outcomes')
            queries.append(f'physical activity AND {condition} AND randomized controlled trial')
    
    elif category == 'medication_safety':
        medications = patient_context.get('medications', [])
        for med in medications[:2]:  # Limit to first 2 medications
            med_name = med.get('name', '')
            if med_name:
                queries.append(f'{med_name} AND safety AND diabetes')
                queries.append(f'{med_name} AND adverse effects AND clinical trial')
    
    elif category == 'monitoring':
        for condition in condition_terms:
            queries.append(f'monitoring AND {condition} AND guidelines')
            queries.append(f'self-monitoring AND {condition} AND effectiveness')
    
    # Fallback general queries
    if not queries:
        for condition in condition_terms:
            queries.append(f'{category} AND {condition} AND systematic review')
    
    return queries


def _get_guideline_evidence(category: str, recommendations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Get evidence from clinical guidelines."""
    guideline_evidence = []
    
    # ADA Guidelines evidence
    ada_evidence = {
        'source_type': 'guideline',
        'title': 'ADA Standards of Care in Diabetes 2024',
        'url': 'https://diabetesjournals.org/care/issue/47/Supplement_1',
        'content': f'American Diabetes Association clinical practice recommendations for {category}',
        'snippet': f'Evidence-based {category} recommendations from ADA Standards of Care',
        'metadata': {
            'organization': 'American Diabetes Association',
            'year': '2024',
            'guideline_type': 'clinical_practice',
            'evidence_level': 'A',
            'category': category
        }
    }
    guideline_evidence.append(ada_evidence)
    
    # Add other relevant guidelines based on category
    if category in ['dietary', 'exercise']:
        lifestyle_evidence = {
            'source_type': 'guideline',
            'title': 'Lifestyle Management Standards of Care in Diabetes',
            'url': 'https://diabetesjournals.org/care/article/47/Supplement_1/S20/153278',
            'content': f'ADA guidelines for {category} in diabetes management',
            'snippet': f'Comprehensive {category} recommendations for diabetes care',
            'metadata': {
                'organization': 'American Diabetes Association',
                'section': 'Lifestyle Management',
                'evidence_level': 'A',
                'category': category
            }
        }
        guideline_evidence.append(lifestyle_evidence)
    
    return guideline_evidence


async def _check_medication_safety(medications: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check medication safety using RxNorm and OpenFDA."""
    try:
        # Use RxNorm for interaction checking
        safety_result = await rxnorm_client.check_medication_safety(medications)
        
        # Enhance with OpenFDA safety information
        enhanced_alerts = []
        for medication in medications:
            if medication.get('active', True):
                drug_name = medication.get('name', '')
                if drug_name:
                    drug_safety = await openfda_client.get_comprehensive_drug_info(drug_name)
                    
                    # Add black box warnings
                    if drug_safety.get('black_box_warning'):
                        enhanced_alerts.append(f"Black box warning for {drug_name}")
                    
                    # Add significant adverse events
                    adverse_events = drug_safety.get('recent_adverse_events', [])
                    severe_events = [e for e in adverse_events if e.get('severity') == 'severe']
                    if severe_events:
                        enhanced_alerts.append(
                            f"Severe adverse events reported for {drug_name}: {severe_events[0].get('reaction')}"
                        )
        
        # Combine safety information
        all_alerts = safety_result.get('safety_alerts', []) + enhanced_alerts
        
        return {
            'safety_alerts': all_alerts,
            'interactions': safety_result.get('interactions', []),
            'overall_risk': safety_result.get('overall_risk', 'unknown'),
            'normalized_medications': safety_result.get('normalized_medications', [])
        }
        
    except Exception as e:
        logger.error("Medication safety check failed", error=str(e))
        return {
            'safety_alerts': [f"Safety check failed: {str(e)}"],
            'interactions': [],
            'overall_risk': 'unknown'
        }


def _format_interaction_evidence(interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Format drug interactions as evidence items."""
    evidence = []
    
    for interaction in interactions:
        evidence_item = {
            'source_type': 'rxnorm',
            'title': f"Drug Interaction: {interaction.get('drug1_rxcui')} and {interaction.get('drug2_rxcui')}",
            'url': 'https://rxnav.nlm.nih.gov/',
            'content': interaction.get('description', ''),
            'snippet': f"Severity: {interaction.get('severity', 'Unknown')}",
            'metadata': {
                'interaction_severity': interaction.get('severity'),
                'source': interaction.get('source'),
                'drug1': interaction.get('drug1_rxcui'),
                'drug2': interaction.get('drug2_rxcui')
            }
        }
        evidence.append(evidence_item)
    
    return evidence


def _calculate_overall_safety_score(safety_result: Dict, drug_details: List[Dict]) -> float:
    """Calculate overall safety score for medications."""
    score = 1.0
    
    # Reduce score for interactions
    interactions = safety_result.get('interactions', [])
    high_severity = len([i for i in interactions if i.get('severity', '').lower() == 'high'])
    moderate_severity = len([i for i in interactions if i.get('severity', '').lower() == 'moderate'])
    
    score -= high_severity * 0.3
    score -= moderate_severity * 0.1
    
    # Reduce score for safety alerts
    alerts = safety_result.get('safety_alerts', [])
    score -= len(alerts) * 0.05
    
    # Reduce score for black box warnings
    black_box_count = sum(1 for drug in drug_details if drug.get('black_box_warning'))
    score -= black_box_count * 0.2
    
    return max(0.0, min(1.0, score))


# Exception handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000)),
        log_level="info"
    )
