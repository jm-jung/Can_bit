"""
Prediction utilities
"""
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
import joblib

from src.ml.model_xgb import XGBModel
from src.ml.model_lstm import LSTMModel
from src.ml.feature_engineering import extract_features_for_prediction
from src.core.config import settings


class Predictor:
    """Unified predictor for different model types"""
    
    def __init__(self):
        """Initialize predictor with models"""
        self.xgb_model: Optional[XGBModel] = None
        self.lstm_model: Optional[LSTMModel] = None
        self._load_models()
    
    def _load_models(self):
        """Load trained models"""
        # TODO: Load models from configured paths
        # if Path(settings.XGB_MODEL_PATH).exists():
        #     self.xgb_model = XGBModel(settings.XGB_MODEL_PATH)
        # if Path(settings.LSTM_MODEL_PATH).exists():
        #     self.lstm_model = LSTMModel(settings.LSTM_MODEL_PATH)
        pass
    
    def predict(
        self,
        features: np.ndarray,
        model_type: str = "xgb"
    ) -> Dict[str, Any]:
        """
        Make prediction using specified model
        
        Args:
            features: Input features array
            model_type: Type of model to use ("xgb" or "lstm")
            
        Returns:
            Dictionary with prediction and metadata
        """
        # TODO: Implement prediction logic
        # - Select model based on model_type
        # - Preprocess features if needed
        # - Make prediction
        # - Calculate confidence if possible
        # - Return result dictionary
        
        prediction = 0.0
        confidence = 0.0
        
        return {
            "prediction": prediction,
            "model_type": model_type,
            "confidence": confidence
        }
    
    def predict_from_raw(
        self,
        raw_data: Dict[str, Any],
        model_type: str = "xgb"
    ) -> Dict[str, Any]:
        """
        Make prediction from raw input data
        
        Args:
            raw_data: Raw input data dictionary
            model_type: Type of model to use
            
        Returns:
            Dictionary with prediction and metadata
        """
        # TODO: Extract features and predict
        # features = extract_features_for_prediction(raw_data)
        # return self.predict(features, model_type)
        pass


# Global predictor instance
predictor = Predictor()


def load_model(model_path: Optional[str] = None) -> Any:
    """
    Load model from disk using joblib
    
    Args:
        model_path: Path to model file. If None, uses default from settings.
        
    Returns:
        Loaded model object (e.g., XGBoost model)
        
    Example:
        model = load_model("./models/xgb_model.pkl")
        prediction = model.predict(features)
    """
    if model_path is None:
        model_path = settings.XGB_MODEL_PATH
    
    model_file = Path(model_path)
    
    if not model_file.exists():
        # Return None or raise exception if model file doesn't exist
        # For development, return None to allow app to start without model
        return None
    
    try:
        # Load model using joblib
        model = joblib.load(model_path)
        return model
    except Exception as e:
        # Log error and return None or re-raise
        # For now, re-raise to make errors visible
        raise FileNotFoundError(f"Failed to load model from {model_path}: {str(e)}")

