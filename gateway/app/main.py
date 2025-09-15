"""AI Wellness Assistant - FastAPI Gateway Application."""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any

import structlog
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from .config import settings, get_logging_config
from .db import init_database, close_database, health_check
from .security import SECURITY_HEADERS
from .routers import patients, clinicians, recommendations, evidence, stream
from .models.common import ErrorResponse, HealthCheck

# Configure structured logging
logging.config.dictConfig(get_logging_config())
logger = structlog.get_logger()

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration')
ml_inference_count = Counter('ml_inference_total', 'Total ML inference requests', ['model_version', 'status'])
recommendation_count = Counter('recommendations_total', 'Total recommendations created', ['status'])


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting AI Wellness Assistant Gateway", version=settings.version)
    
    try:
        await init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Failed to initialize database", error=str(e))
        raise
    
    # Additional startup tasks
    await startup_health_checks()
    
    logger.info("Gateway startup complete")
    
    yield  # Application runs here
    
    # Shutdown
    logger.info("Shutting down AI Wellness Assistant Gateway")
    await close_database()
    logger.info("Gateway shutdown complete")


async def startup_health_checks():
    """Perform startup health checks."""
    try:
        # Database health check
        db_health = await health_check()
        if db_health.get("status") != "healthy":
            logger.warning("Database health check degraded", health=db_health)
        
        # TODO: Add checks for external services
        # - ML Risk Service
        # - Evidence Verifier
        # - Agent Orchestrator
        
        logger.info("Startup health checks completed")
        
    except Exception as e:
        logger.error("Startup health check failed", error=str(e))
        # Don't fail startup for health check issues
        pass


# FastAPI application
app = FastAPI(
    title=settings.app_name,
    description="AI-powered cardiometabolic risk assessment and personalized care planning",
    version=settings.version,
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

# Trusted host middleware
if settings.environment == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.wellnessai.com", "localhost"]  # Configure for production
    )


# Security headers middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    for header, value in SECURITY_HEADERS.items():
        response.headers[header] = value
    
    return response


# Request logging and metrics middleware
@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    """Log requests and collect metrics."""
    start_time = datetime.utcnow()
    
    # Generate request ID
    request_id = f"req_{int(start_time.timestamp() * 1000000)}"
    
    # Add request context
    request.state.request_id = request_id
    request.state.start_time = start_time
    
    # Log request
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None
    )
    
    # Process request
    try:
        response = await call_next(request)
        
        # Calculate duration
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        # Record metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code
        ).inc()
        
        request_duration.observe(duration)
        
        # Log response
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2)
        )
        
        # Add request ID to response headers
        response.headers["X-Request-ID"] = request_id
        
        return response
        
    except Exception as e:
        duration = (datetime.utcnow() - start_time).total_seconds()
        
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(e),
            duration_ms=round(duration * 1000, 2)
        )
        
        # Record error metrics
        request_count.labels(
            method=request.method,
            endpoint=request.url.path,
            status=500
        ).inc()
        
        raise


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        "Validation error",
        request_id=getattr(request.state, "request_id", None),
        errors=exc.errors()
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            code="VALIDATION_ERROR",
            details={"errors": exc.errors()}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    request_id = getattr(request.state, "request_id", None)
    
    logger.error(
        "Unhandled exception",
        request_id=request_id,
        error=str(exc),
        exc_info=True
    )
    
    # Don't expose internal errors in production
    if settings.environment == "production":
        error_message = "Internal server error"
        details = None
    else:
        error_message = str(exc)
        details = {"type": type(exc).__name__}
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=error_message,
            code="INTERNAL_ERROR",
            details=details
        ).dict()
    )


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_endpoint():
    """Application health check."""
    try:
        # Check database
        db_health = await health_check()
        
        # TODO: Check external services
        services = {
            "database": db_health.get("status", "unknown"),
            "ml_risk": "unknown",  # TODO: Implement service checks
            "agents": "unknown",
            "verifier": "unknown",
        }
        
        # Determine overall status
        unhealthy_services = [k for k, v in services.items() if v == "unhealthy"]
        if unhealthy_services:
            overall_status = "unhealthy"
        elif any(v == "degraded" for v in services.values()):
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return HealthCheck(
            status=overall_status,
            version=settings.version,
            environment=settings.environment,
            services=services
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=HealthCheck(
                status="unhealthy",
                version=settings.version,
                environment=settings.environment,
                services={"error": str(e)}
            ).dict()
        )


# Metrics endpoint (Prometheus)
@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    if not settings.enable_metrics:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": "Metrics disabled"}
        )
    
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# API versioning prefix
API_V1_PREFIX = "/api/v1"

# Include routers
app.include_router(
    patients.router,
    prefix=f"{API_V1_PREFIX}/patients",
    tags=["patients"]
)

app.include_router(
    clinicians.router,
    prefix=f"{API_V1_PREFIX}/clinician",
    tags=["clinicians"]
)

app.include_router(
    recommendations.router,
    prefix=f"{API_V1_PREFIX}/recommendations",
    tags=["recommendations"]
)

app.include_router(
    evidence.router,
    prefix=f"{API_V1_PREFIX}/evidence",
    tags=["evidence"]
)

app.include_router(
    stream.router,
    prefix=f"{API_V1_PREFIX}/stream",
    tags=["streaming"]
)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": settings.version,
        "environment": settings.environment,
        "documentation": "/docs" if settings.debug else None,
        "health": "/health",
        "metrics": "/metrics" if settings.enable_metrics else None,
        "api_version": "v1",
        "api_base": f"{API_V1_PREFIX}",
        "status": "operational"
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )
