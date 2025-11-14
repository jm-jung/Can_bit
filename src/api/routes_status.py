"""
Status and health check API routes
"""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

from src.core.config import settings

router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: str


class VersionResponse(BaseModel):
    """Version information response model"""
    version: str
    project_name: str
    environment: str


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        HealthResponse with status and timestamp
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/version", response_model=VersionResponse)
async def get_version():
    """
    Get API version information
    
    Returns:
        VersionResponse with version details
    """
    return VersionResponse(
        version=settings.VERSION,
        project_name=settings.PROJECT_NAME,
        environment=settings.ENVIRONMENT
    )

