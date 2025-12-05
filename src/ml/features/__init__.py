"""
Modular feature blocks for ML pipeline.

Phase E: Structural improvements - Feature pipeline refactoring.
"""
from .feature_blocks import (
    BaseFeatureBlock,
    EventFeatureBlock,
    TrendFeatureBlock,
    VolatilityFeatureBlock,
    RegimeFeatureBlock,
    FeatureBlockManager,
)

# Backward-compatible export for train_xgb (Phase E refactor 이후 호환용)
# build_ml_dataset는 상위 디렉토리의 features.py 모듈에 정의되어 있음
# 순환 참조를 피하기 위해 importlib를 사용하여 모듈을 직접 로드
import importlib.util
from pathlib import Path

# 상위 디렉토리의 features.py 모듈 파일을 직접 로드
_features_module_path = Path(__file__).parent.parent / "features.py"
_spec = importlib.util.spec_from_file_location("ml_features_module", _features_module_path)
_ml_features_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ml_features_module)

# Re-export functions
build_ml_dataset = _ml_features_module.build_ml_dataset
build_feature_frame = _ml_features_module.build_feature_frame

__all__ = [
    "BaseFeatureBlock",
    "EventFeatureBlock",
    "TrendFeatureBlock",
    "VolatilityFeatureBlock",
    "RegimeFeatureBlock",
    "FeatureBlockManager",
    # Backward compatibility exports
    "build_ml_dataset",
    "build_feature_frame",
]

