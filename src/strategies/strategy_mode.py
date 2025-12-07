"""
Strategy mode definitions for ML trading strategies.

This module defines the different modes a strategy can operate in:
- BOTH: Trade both long and short positions
- LONG_ONLY: Only trade long positions
- SHORT_ONLY: Only trade short positions
"""
from __future__ import annotations

from enum import Enum


class StrategyMode(Enum):
    """Strategy trading mode."""
    
    BOTH = "both"
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    
    @classmethod
    def from_string(cls, value: str | None) -> StrategyMode:
        """
        Convert string to StrategyMode enum.
        
        Args:
            value: String value ("both", "long_only", "short_only") or None
        
        Returns:
            StrategyMode enum value (default: BOTH if None or invalid)
        """
        if value is None:
            return cls.BOTH
        
        value_lower = value.lower().strip()
        for mode in cls:
            if mode.value == value_lower:
                return mode
        
        # Default to BOTH if invalid
        return cls.BOTH
    
    def to_string(self) -> str:
        """Convert enum to string value."""
        return self.value
    
    def allows_long(self) -> bool:
        """Check if this mode allows long positions."""
        return self in (StrategyMode.BOTH, StrategyMode.LONG_ONLY)
    
    def allows_short(self) -> bool:
        """Check if this mode allows short positions."""
        return self in (StrategyMode.BOTH, StrategyMode.SHORT_ONLY)

