"""
Configuration management
"""
import os
from pathlib import Path
from typing import List, Optional

from pydantic import Field
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
    XGB_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "xgb_model.pkl")
    LSTM_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "lstm_model.h5")
    # LSTM + Attention 모델 경로 (프로젝트 루트 기준)
    LSTM_ATTN_MODEL_PATH: str = str(PROJECT_ROOT / "models" / "lstm_attn_v1.pt")

    # DL LSTM + Attention strategy thresholds
    # LSTM threshold 통일: strategy와 backtest에서 동일한 값 사용
    # NOTE: threshold_up/down can be overridden via API query params or .env file
    LSTM_ATTN_THRESHOLD_UP: float = Field(0.55, env="LSTM_ATTN_THRESHOLD_UP")
    LSTM_ATTN_THRESHOLD_DOWN: float = Field(0.45, env="LSTM_ATTN_THRESHOLD_DOWN")
    # Legacy alias (하위 호환성)
    DL_LSTM_ATTN_THRESHOLD_UP: float = Field(0.55, env="DL_LSTM_ATTN_THRESHOLD_UP")
    DL_LSTM_ATTN_THRESHOLD_DOWN: float = Field(0.45, env="DL_LSTM_ATTN_THRESHOLD_DOWN")

    # Event feature pipeline
    EVENTS_ENABLED: bool = True
    EVENTS_LOOKBACK_MINUTES: int = 60
    EVENTS_TIMEFRAME: str = "1h"
    EVENTS_DATA_PATH: str = "data/events/events.parquet"
    EVENTS_RAW_PATH: str = "data/events/raw"
    NEWS_API_KEY: Optional[str] = None
    X_BEARER_TOKEN: Optional[str] = None

    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "./logs/app.log"
    
    # LSTM Training Debug settings
    DEBUG_LSTM_INFERENCE_SAMPLES: bool = Field(
        default=True,
        description="Validation 셋 샘플 출력 여부"
    )
    
    DEBUG_SMALL_OVERFIT: bool = Field(
        default=False,
        description="True이면 소규모 데이터셋(64개 샘플)에 대해 overfit 테스트 실행"
    )
    
    DEBUG_GRADIENT_LOGGING: bool = Field(
        default=True,
        description="True이면 gradient norm과 weight 변화를 로깅"
    )
    
    # LSTM Label Generation settings
    LSTM_RETURN_HORIZON: int = Field(
        default=5,  # was 10, now 5 to get more local signal
        description="LSTM 라벨 생성 시 미래 수익률을 계산할 horizon (몇 개의 캔들 후를 볼지)"
    )
    
    LSTM_LABEL_POS_THRESHOLD: float = Field(
        default=0.0015,  # was 0.003, now 0.0015 to get more samples
        description="미래 수익률이 이 값 이상이면 label=1 (강한 상승으로 간주)"
    )
    
    LSTM_LABEL_NEG_THRESHOLD: float = Field(
        default=-0.0015,  # was -0.003, now -0.0015 to get more samples
        description="미래 수익률이 이 값 이하이면 label=0 (강한 하락으로 간주)"
    )
    
    LSTM_LABEL_IGNORE_MARGIN: float = Field(
        default=0.0,  # was 0.0005, now 0.0 to remove ignore zone and get more samples
        description="미래 수익률이 (-margin, +margin) 안에 있으면 ambiguous zone으로 보고 학습에서 제외"
    )
    
    # LSTM Loss function settings
    LSTM_USE_FOCAL_LOSS: bool = Field(
        default=False,  # was True, now False to use BCEWithLogitsLoss with pos_weight
        description="True이면 BCE 대신 Focal Loss를 사용하여 상수 예측 collapse를 줄이려 시도"
    )
    
    LSTM_FOCAL_ALPHA: float = Field(
        default=0.25,
        description="Focal Loss alpha 파라미터 (양성 클래스 중요도)"
    )
    
    LSTM_FOCAL_GAMMA: float = Field(
        default=2.0,
        description="Focal Loss gamma 파라미터 (easy sample down-weighting 강도)"
    )

    # ✅ Binance 관련 설정 (이번에 추가한 것들)
    BINANCE_API_KEY: Optional[str] = None
    BINANCE_API_SECRET: Optional[str] = None
    BINANCE_SYMBOL: str = "BTC/USDT"
    BINANCE_SANDBOX_MODE: bool = True
    BINANCE_LIVE_TRADING: bool = False
    
    # ML Threshold Optimization settings
    USE_OPTIMIZED_THRESHOLDS: bool = Field(
        default=False,
        description="If True, automatically load optimized thresholds from JSON files"
    )
    THRESHOLD_OPTIMIZATION_METRIC: str = Field(
        default="sharpe",
        description="Metric to optimize: 'total_return' or 'sharpe'"
    )
    THRESHOLD_SYMBOL: str = Field(
        default="BTCUSDT",
        description="Symbol key used when loading optimized thresholds (e.g., BTCUSDT)"
    )
    THRESHOLD_TIMEFRAME: str = Field(
        default="1m",
        description="Timeframe key used when loading optimized thresholds (e.g., 1m)"
    )
    LONG_THRESHOLD_MIN: float = Field(
        default=0.40,
        description="Minimum long threshold for grid search"
    )
    LONG_THRESHOLD_MAX: float = Field(
        default=0.70,
        description="Maximum long threshold for grid search"
    )
    LONG_THRESHOLD_STEP: float = Field(
        default=0.05,
        description="Step size for long threshold grid search"
    )
    SHORT_THRESHOLD_MIN: float = Field(
        default=0.30,
        description="Minimum short threshold for grid search"
    )
    SHORT_THRESHOLD_MAX: float = Field(
        default=0.50,
        description="Maximum short threshold for grid search"
    )
    SHORT_THRESHOLD_STEP: float = Field(
        default=0.05,
        description="Step size for short threshold grid search"
    )
    
    # Backtest settings (commission and slippage)
    COMMISSION_RATE: float = Field(
        default=0.0004,
        description="Commission rate per trade side (0.0004 = 0.04% = 4 bps)"
    )
    SLIPPAGE_RATE: float = Field(
        default=0.0005,
        description="Slippage rate per trade side (0.0005 = 0.05% = 5 bps)"
    )
    
    # Phase E: ML Pipeline Structural Improvements
    # Feature Scaling
    ML_SCALER_TYPE: str = Field(
        default="none",
        description="Feature scaler type: 'standard', 'robust', 'minmax', or 'none'"
    )
    
    # Labeling (Phase E defaults)
    ML_LABEL_MODE: str = Field(
        default="classification",
        description="Label mode: 'classification' or 'regression'"
    )
    ML_LABEL_LONG_THRESHOLD: float = Field(
        default=0.0030,
        description="Long label threshold (0.0030 = 0.3%)"
    )
    ML_LABEL_SHORT_THRESHOLD: float = Field(
        default=0.0030,
        description="Short label threshold (0.0030 = 0.3%)"
    )
    ML_LABEL_HOLD_THRESHOLD: float = Field(
        default=0.0010,
        description="Hold label threshold (0.0010 = 0.1%)"
    )
    ML_LABEL_HORIZON: int = Field(
        default=20,
        description="Label horizon (number of periods ahead)"
    )
    ML_ENABLE_HOLD_LABELS: bool = Field(
        default=False,
        description="Enable HOLD labels and exclude them from training"
    )
    
    # Volatility Regime
    ML_REGIME_ATR_PERCENTILE: float = Field(
        default=0.70,
        description="ATR percentile threshold for HIGH_VOL classification"
    )
    ML_REGIME_HIGH_VOL_MULTIPLIER: float = Field(
        default=1.3,
        description="Threshold multiplier for HIGH_VOL regime"
    )
    ML_REGIME_LOW_VOL_MULTIPLIER: float = Field(
        default=0.8,
        description="Threshold multiplier for LOW_VOL regime"
    )
    
    # Backtest Trading Logic (Phase E)
    BACKTEST_MAX_HOLDING_BARS: int | None = Field(
        default=None,
        description="Maximum holding bars before auto-close (None = no limit)"
    )
    BACKTEST_TP_PERCENT: float | None = Field(
        default=None,
        description="Take profit percentage (e.g., 0.003 = 0.3%, None = disabled)"
    )
    BACKTEST_SL_PERCENT: float | None = Field(
        default=None,
        description="Stop loss percentage (e.g., 0.002 = 0.2%, None = disabled)"
    )
    BACKTEST_USE_CONFIDENCE_FILTER: bool = Field(
        default=False,
        description="Enable confidence-based entry filter"
    )
    BACKTEST_CONFIDENCE_QUANTILE: float = Field(
        default=0.85,
        description="Confidence quantile threshold (0.85 = top 15%)"
    )
    BACKTEST_CONFIDENCE_LOOKBACK: int = Field(
        default=100,
        description="Lookback window for confidence filter"
    )
    BACKTEST_DAILY_LOSS_LIMIT: float | None = Field(
        default=None,
        description="Daily loss limit (kill switch, None = disabled)"
    )


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
