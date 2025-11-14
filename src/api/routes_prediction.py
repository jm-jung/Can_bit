"""
Prediction API routes
"""
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np

from src.api.deps import get_model

router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    features: List[float]
    model_type: str = "xgb"  # "xgb" or "lstm"


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    prediction: float
    model_type: str
    confidence: float = 0.0


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model: Any = Depends(get_model)
):
    """
    Make prediction using ML model
    
    Args:
        request: Prediction request with features and model type
        model: ML model object injected via dependency
        
    Returns:
        PredictionResponse with prediction result
    """
    try:
        # Convert features to numpy array
        features_array = np.array(request.features).reshape(1, -1)
        
        # TODO: Implement actual prediction logic
        # - Preprocess features if needed
        # - Make prediction using model
        # - Calculate confidence if possible
        
        # Dummy prediction for template
        # Replace with actual model prediction:
        # prediction = model.predict(features_array)[0]
        
        prediction = 0.0
        confidence = 0.0
        
        return PredictionResponse(
            prediction=prediction,
            model_type=request.model_type,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

