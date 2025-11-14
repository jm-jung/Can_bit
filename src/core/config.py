"""
Configuration management
"""
import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings
import yaml


class Settings(BaseSettings):
    """Application settings"""
    
    # Project settings
    PROJECT_NAME: str = "ML Prediction API"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    
    # API settings
    API_PREFIX: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./app.db"
    
    # ML Model settings
    MODEL_DIR: str = "./models"
    XGB_MODEL_PATH: str = "./models/xgb_model.pkl"
    LSTM_MODEL_PATH: str = "./models/lstm_model.h5"
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


def load_config_yaml(config_path: str = "config.yaml") -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Dictionary with configuration values
    """
    config_file = Path(config_path)
    if config_file.exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            return config or {}
    return {}


# Load YAML config and merge with environment variables
_yaml_config = load_config_yaml()

# Create settings instance
settings = Settings(
    **{k: v for k, v in _yaml_config.items() if hasattr(Settings, k)}
)

