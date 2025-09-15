"""FastAPI application for ML Risk Prediction Service."""

import os
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import structlog

from infer import RiskPredictor, predict_patient_risk

# Configure logging
logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="ML Risk Prediction Service",
    description="Cardiometabolic risk prediction with SHAP explanations",
    version="1.0.0"
)

# Pydantic models
class PatientData(BaseModel):
    """Patient data for risk prediction."""
    demographics: Dict[str, Any] = Field(default_factory=dict)
    vitals: List[Dict[str, Any]] = Field(default_factory=list)
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    medications: List[Dict[str, Any]] = Field(default_factory=list)


class RiskPredictionRequest(BaseModel):
    """Risk prediction request."""
    patient_data: PatientData
    include_shap: bool = True
    model_version: str = None


class RiskPredictionResponse(BaseModel):
    """Risk prediction response."""
    risk_probability: float
    risk_category: str
    model_version: str
    algorithm: str
    features_used: List[str]
    timestamp: str
    shap_explanations: Dict[str, Any] = None


class BatchRiskPredictionRequest(BaseModel):
    """Batch risk prediction request."""
    patients_data: List[PatientData]
    include_shap: bool = False


class HealthCheck(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool = False
    model_version: str = None


# Global predictor
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the ML model on startup."""
    global predictor
    
    try:
        model_path = os.getenv('MODEL_PATH', 'models/risk_lgbm_v0.1.bin')
        predictor = RiskPredictor(model_path)
        
        if predictor.model is None:
            logger.warning("Model not loaded, using default configuration")
        else:
            logger.info("ML Risk service started", 
                       model_version=predictor.model_metadata.get('version'))
    
    except Exception as e:
        logger.error("Failed to initialize ML service", error=str(e))
        predictor = RiskPredictor()  # Empty predictor


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint."""
    global predictor
    
    model_loaded = predictor is not None and predictor.model is not None
    model_version = None
    
    if model_loaded:
        model_version = predictor.model_metadata.get('version', 'unknown')
    
    return HealthCheck(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_version=model_version
    )


@app.post("/predict", response_model=RiskPredictionResponse)
async def predict_risk(request: RiskPredictionRequest):
    """Predict cardiometabolic risk for a patient."""
    global predictor
    
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not available"
        )
    
    try:
        # Convert Pydantic model to dict
        patient_data = request.patient_data.dict()
        
        # Make prediction
        result = predictor.predict_risk(
            patient_data, 
            include_shap=request.include_shap
        )
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {result['error']}"
            )
        
        return RiskPredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Risk prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal prediction error"
        )


@app.post("/predict/batch")
async def predict_risk_batch(request: BatchRiskPredictionRequest):
    """Predict risk for multiple patients."""
    global predictor
    
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not available"
        )
    
    try:
        # Convert to list of dicts
        patients_data = [patient.dict() for patient in request.patients_data]
        
        # Make batch predictions
        results = predictor.batch_predict(
            patients_data,
            include_shap=request.include_shap
        )
        
        return {"predictions": results, "count": len(results)}
        
    except Exception as e:
        logger.error("Batch prediction failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction error"
        )


@app.get("/model/info")
async def get_model_info():
    """Get model information and metadata."""
    global predictor
    
    if predictor is None:
        return {"error": "Model not initialized"}
    
    return predictor.model_summary()


@app.get("/model/features")
async def get_model_features():
    """Get model feature names and importance."""
    global predictor
    
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not available"
        )
    
    return {
        "feature_names": predictor.model_metadata.get('feature_names', []),
        "feature_importance": predictor.get_feature_importance()
    }


@app.post("/explain")
async def explain_prediction(request: RiskPredictionRequest):
    """Get detailed SHAP explanations for a prediction."""
    global predictor
    
    if predictor is None or predictor.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="ML model not available"
        )
    
    try:
        patient_data = request.patient_data.dict()
        
        # Force SHAP explanations
        result = predictor.predict_risk(patient_data, include_shap=True)
        
        if 'error' in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Explanation failed: {result['error']}"
            )
        
        # Return only the explanation part
        return {
            "risk_probability": result.get("risk_probability"),
            "risk_category": result.get("risk_category"),
            "shap_explanations": result.get("shap_explanations", {}),
            "model_version": result.get("model_version")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Explanation failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Explanation error"
        )


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
