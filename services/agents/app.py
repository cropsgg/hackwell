"""GenAI Agents Service - FastAPI Application."""

import os
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from base_agent import AgentContext
from orchestrator_agent import OrchestratorAgent

# Configure logging
logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="GenAI Agents Service",
    description="MCP-style agent orchestration for AI wellness recommendations",
    version="1.0.0"
)

# Pydantic models
class RecommendationGenerationRequest(BaseModel):
    """Request to generate recommendation."""
    patient_id: str
    force_refresh: bool = False
    include_evidence: bool = True
    model_version: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = "healthy"
    agents_loaded: bool = False
    orchestrator_ready: bool = False


# Global orchestrator
orchestrator = None


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator agent."""
    global orchestrator
    
    try:
        orchestrator = OrchestratorAgent()
        logger.info("GenAI Agents service started")
        
    except Exception as e:
        logger.error("Failed to initialize agents service", error=str(e))


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="healthy" if orchestrator is not None else "degraded",
        agents_loaded=orchestrator is not None,
        orchestrator_ready=orchestrator is not None
    )


@app.post("/orchestrator/generate")
async def generate_recommendation(request: RecommendationGenerationRequest):
    """Generate recommendation using agent orchestration."""
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not available"
        )
    
    try:
        logger.info("Starting recommendation generation", 
                   patient_id=request.patient_id)
        
        result = await orchestrator.generate_recommendation(
            patient_id=request.patient_id,
            force_refresh=request.force_refresh,
            include_evidence=request.include_evidence,
            model_version=request.model_version,
            user_context=request.context
        )
        
        return result
        
    except Exception as e:
        logger.error("Recommendation generation failed", 
                    patient_id=request.patient_id, 
                    error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Recommendation generation failed"
        )


@app.post("/agents/intake")
async def run_intake_agent(patient_data: Dict[str, Any]):
    """Run intake agent standalone."""
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agents not available"
        )
    
    try:
        context = AgentContext(patient_data.get('patient_id', 'unknown'))
        context.add_patient_data(
            patient_data.get('demographics', {}),
            patient_data.get('vitals', []),
            patient_data.get('medications', []),
            patient_data.get('conditions', [])
        )
        
        result_context = await orchestrator.intake_agent.process(context)
        
        return {
            "success": len(result_context.errors) == 0,
            "output": result_context.get_agent_output('intake'),
            "errors": result_context.errors
        }
        
    except Exception as e:
        logger.error("Intake agent failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Intake agent failed"
        )


@app.post("/agents/normalize")
async def run_normalizer_agent(patient_data: Dict[str, Any]):
    """Run normalizer agent standalone."""
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agents not available"
        )
    
    try:
        context = AgentContext(patient_data.get('patient_id', 'unknown'))
        context.add_patient_data(
            patient_data.get('demographics', {}),
            patient_data.get('vitals', []),
            patient_data.get('medications', []),
            patient_data.get('conditions', [])
        )
        
        result_context = await orchestrator.normalizer_agent.process(context)
        
        return {
            "success": len(result_context.errors) == 0,
            "normalized_data": {
                "demographics": result_context.patient_profile,
                "vitals": result_context.vitals,
                "medications": result_context.medications,
                "conditions": result_context.conditions
            },
            "normalization_summary": result_context.get_agent_output('normalizer'),
            "errors": result_context.errors
        }
        
    except Exception as e:
        logger.error("Normalizer agent failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Normalizer agent failed"
        )


@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents."""
    if orchestrator is None:
        return {"error": "Orchestrator not initialized"}
    
    return {
        "orchestrator": "ready",
        "agents": {
            "intake": "ready",
            "normalizer": "ready", 
            "risk_predictor": "ready",
            "careplan": "ready",
            "evidence_verifier": "ready"
        },
        "external_services": {
            "ml_risk_service": "unknown",
            "evidence_verifier_service": "unknown"
        }
    }


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
