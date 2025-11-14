"""
Dependencies for API routes
"""
from typing import Generator, Any
from functools import lru_cache
from fastapi import Request

from src.core.config import settings


def get_db_session() -> Generator[Any, None, None]:
    """
    Database session dependency
    
    Yields:
        Database session object
        
    Example:
        # TODO: Implement actual database session
        # from sqlalchemy.orm import Session
        # db = SessionLocal()
        # try:
        #     yield db
        # finally:
        #     db.close()
    """
    # Placeholder for database session
    session = None
    try:
        yield session
    finally:
        if session:
            # Close session if needed
            pass


@lru_cache()
def get_settings():
    """
    Get cached settings instance
    
    Returns:
        Settings object
    """
    return settings


def get_model(request: Request) -> Any:
    """
    Get ML model from app state
    
    Args:
        request: FastAPI request object
        
    Returns:
        Model object stored in app.state.model
        
    Raises:
        HTTPException: If model is not loaded
    """
    model = request.app.state.model
    
    if model is None:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please check server logs."
        )
    
    return model

