"""
Configuration management
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
import yaml

# 프로젝트 루트 디렉토리 계산 (src/core/config.py 기준으로 상위 2단계)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Application settings"""

    # Pydantic v2 스타일 설정
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 환경변수/추가키 들어와도 그냥 무시 (안전)
    )

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
    XGB_MODEL_PATH: str = "../models/xgb_model.pkl"
    LSTM_MODEL_PATH: str = "../models/lstm_model.h5"
    # LSTM + Attention 모델 경로 (프로젝트 루트 기준)
    LSTM_ATTN_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "lstm_attn_v1.pt")

    # DL LSTM + Attention strategy thresholds
    DL_LSTM_ATTN_THRESHOLD_UP: float = 0.55
    DL_LSTM_ATTN_THRESHOLD_DOWN: float = 0.45

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"

    # ✅ Binance 관련 설정 (이번에 추가한 것들)
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    BINANCE_SYMBOL: str = "BTC/USDT"
    BINANCE_SANDBOX_MODE: bool = True
    BINANCE_LIVE_TRADING: bool = False


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


# YAML config 로드
_yaml_config = load_config_yaml()

# ✅ YAML에서 온 값 중 Settings에 존재하는 필드만 골라서 override
_filtered_yaml = {k: v for k, v in _yaml_config.items() if k in Settings.model_fields}

# ✅ 환경변수 + YAML override를 합쳐서 Settings 생성
settings = Settings(**_filtered_yaml)
