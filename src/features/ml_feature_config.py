"""
ML Feature Configuration for XGBoost strategy.

This module provides configuration classes to control which features are used
in ML model training and inference, enabling feature preset management and
overfitting control.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MLFeatureConfig:
    """
    Configuration for ML feature generation.
    
    Controls which feature categories are included in the feature set.
    Default preset "base" matches the existing production feature set exactly.
    """
    
    # Base features (always used in production)
    use_base_price_features: bool = True
    """Basic OHLCV-derived features (close, high, low, volume, etc.)"""
    
    use_indicator_features: bool = True
    """Existing technical indicators (EMA, SMA, RSI, etc.)"""
    
    use_event_features: bool = True
    """Event-based features (from event aggregator)"""
    
    # Extended features (Stage 2 additions, disabled by default)
    use_extended_trend_features: bool = False
    """Extended trend features (log returns, EMA distances, trend slopes)"""
    
    use_volatility_features: bool = False
    """Volatility features (ATR, True Range, return std dev)"""
    
    use_volume_features: bool = False
    """Volume-based features (volume MA, z-score, ratios)"""
    
    use_structure_features: bool = False
    """Candle structure features (body, shadows, direction counts)"""
    
    preset_name: str = "base"
    """Preset identifier for this configuration"""
    
    @classmethod
    def from_preset(cls, preset_name: str) -> "MLFeatureConfig":
        """
        Create configuration from a preset.
        
        Presets:
        - "base": Current production feature set (100% compatible with existing models)
        - "extended_safe": Base + extended trend + volatility features
        - "extended_full": All features enabled
        
        Args:
            preset_name: Name of the preset ("base", "extended_safe", "extended_full")
        
        Returns:
            MLFeatureConfig instance
        """
        if preset_name == "base":
            return cls(
                use_base_price_features=True,
                use_indicator_features=True,
                use_event_features=True,
                use_extended_trend_features=False,
                use_volatility_features=False,
                use_volume_features=False,
                use_structure_features=False,
                preset_name="base",
            )
        elif preset_name == "extended_safe":
            return cls(
                use_base_price_features=True,
                use_indicator_features=True,
                use_event_features=True,
                use_extended_trend_features=True,
                use_volatility_features=True,
                use_volume_features=False,
                use_structure_features=False,
                preset_name="extended_safe",
            )
        elif preset_name == "extended_full":
            return cls(
                use_base_price_features=True,
                use_indicator_features=True,
                use_event_features=True,
                use_extended_trend_features=True,
                use_volatility_features=True,
                use_volume_features=True,
                use_structure_features=True,
                preset_name="extended_full",
            )
        else:
            raise ValueError(
                f"Unknown preset: {preset_name}. "
                f"Available presets: 'base', 'extended_safe', 'extended_full'"
            )
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"MLFeatureConfig(preset={self.preset_name}, "
            f"base={self.use_base_price_features}, "
            f"indicator={self.use_indicator_features}, "
            f"event={self.use_event_features}, "
            f"trend={self.use_extended_trend_features}, "
            f"vol={self.use_volatility_features}, "
            f"volume={self.use_volume_features}, "
            f"struct={self.use_structure_features})"
        )

