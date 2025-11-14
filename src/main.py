"""
FastAPI ML Prediction Web Application
Main entry point for the application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import settings
from src.core.logging import setup_logging
from src.ml.predictor import load_model
from src.api.routes_prediction import router as prediction_router
from src.api.routes_status import router as status_router
import logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app
    Handles startup and shutdown events
    """
    # Startup: Load model and store in app.state
    logger.info("Loading ML model...")
    
    try:
        model = load_model()
        app.state.model = model
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load model: {str(e)}. App will continue without model.")
        app.state.model = None
    
    yield
    
    # Shutdown: Cleanup resources if needed
    logger.info("Shutting down application...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ML Prediction Web API",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(status_router, tags=["status"])
app.include_router(prediction_router, prefix="/api/v1", tags=["prediction"])

