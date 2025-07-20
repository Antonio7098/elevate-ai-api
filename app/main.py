from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, Any
from app.core.config import settings
from app.api import endpoints
from app.api.blueprint_lifecycle_endpoints import lifecycle_router
from app.core.indexing import evaluate_answer
from app.core.services import initialize_services, shutdown_services

# Initialize FastAPI app
app = FastAPI(
    title="Elevate AI API",
    description="AI-powered learning co-pilot API",
    version="0.1.0",
    debug=settings.debug
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://127.0.0.1:3000", 
        "http://127.0.0.1:3001",
        "http://localhost:3002",
        "http://127.0.0.1:3002",
        "http://localhost:3003",
        "http://127.0.0.1:3003",
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security scheme
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header."""
    # Skip authentication if no API key is configured (for development)
    if not settings.api_key:
        return None
    
    if not credentials or credentials.credentials != settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Include API routes
app.include_router(
    endpoints.router,
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)]
)

# Include Blueprint Lifecycle routes
app.include_router(
    lifecycle_router,
    prefix="/api/v1",
    dependencies=[Depends(verify_api_key)],
    tags=["Blueprint Lifecycle"]
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup."""
    try:
        await initialize_services()
    except Exception as e:
        print(f"Failed to initialize services: {e}")
        # Don't raise here to allow the app to start for development

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown services on application shutdown."""
    await shutdown_services()

# Add health endpoints without authentication
@app.get("/api/health", include_in_schema=False)
async def api_health_check():
    """API health check endpoint for frontend."""
    return {"status": "healthy", "api": "running"}

@app.options("/api/health", include_in_schema=False)
async def api_health_check_options():
    """OPTIONS handler for API health check endpoint."""
    return {"status": "healthy", "api": "running"}

@app.get("/api/", include_in_schema=False)
async def api_root():
    """API root endpoint for frontend health checks."""
    return {"status": "healthy", "api": "running", "version": "0.1.0"}

@app.options("/api/", include_in_schema=False)
async def api_root_options():
    """OPTIONS handler for API root endpoint."""
    return {"status": "healthy", "api": "running", "version": "0.1.0"}

# Compatibility endpoint for frontend
@app.post("/evaluate-answer", include_in_schema=False)
async def evaluate_answer_compatibility(request: Dict[str, Any]):
    """Compatibility endpoint for frontend answer evaluation."""
    try:
        # Extract the required fields from the frontend payload
        question_id = request.get("questionId")
        user_answer = request.get("userAnswer")
        
        if not question_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="questionId is required"
            )
        
        if not user_answer:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="userAnswer is required"
            )
        
        # Call the actual answer evaluation logic
        result = await evaluate_answer(
            question_id=question_id,
            user_answer=user_answer
        )
        
        # Return the response in the format expected by the frontend
        return {
            "correctedAnswer": result.get("corrected_answer", ""),
            "marksAvailable": result.get("marks_available", 0),
            "marksAchieved": result.get("marks_achieved", 0)
        }
        
    except HTTPException:
        # Re-raise HTTPExceptions as-is
        raise
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Answer evaluation failed: {str(e)}"
        )

@app.options("/evaluate-answer", include_in_schema=False)
async def evaluate_answer_compatibility_options():
    """OPTIONS handler for compatibility endpoint."""
    return {"status": "ready"}

@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Elevate AI API is running", "version": "0.1.0"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    ) 