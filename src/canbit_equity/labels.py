"""Forward labels with next-open entry / horizon-close exit."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import QQQConfig


def add_labels(features: pd.DataFrame, cfg: QQQConfig = QQQConfig(), cost_bps_per_side: float = 5.0) -> pd.DataFrame:
    d = features.copy().reset_index(drop=True)
    cost = 2.0 * (cost_bps_per_side / 10000.0)  # round-trip
    open_ = d["open_adj"].astype(float)
    close = d["close_adj"].astype(float)
    for h in cfg.target_horizons:
        # entry at t+1 open, exit at t+h close (horizon sessions after signal day)
        entry = open_.shift(-1)
        exit_ = close.shift(-h)
        gross = exit_ / entry - 1.0
        net = gross - cost
        d[f"forward_return_{h}d_gross"] = gross
        d[f"forward_return_{h}d_net"] = net
        d[f"label_long_{h}d"] = (net > 0).astype(int)
    # drop rows without full forward horizon for primary
    d = d.dropna(subset=[f"forward_return_{cfg.primary_horizon}d_net"]).reset_index(drop=True)
    return d
