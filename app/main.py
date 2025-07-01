from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.config import settings
from app.api import endpoints

# Initialize FastAPI app
app = FastAPI(
    title="Elevate AI API",
    description="AI-powered learning co-pilot API",
    version="0.1.0",
    debug=settings.debug
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