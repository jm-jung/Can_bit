"""
FR2 preset: microstructure_v1.

Base + FR1 extended_safe_v1 + order flow, funding/position pressure,
liquidation cluster, aggressive volume. Use with MLFeatureConfig.from_preset("microstructure_v1").
"""
from __future__ import annotations

PRESET_NAME = "microstructure_v1"
