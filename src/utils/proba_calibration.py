"""
Probability calibration for 3-class (long / short / flat) TCN outputs.

Temperature scaling: pseudo-logits from proba, then softmax(logits/T) to get
calibrated pl, ps (and implicit pf). T=1.0 preserves original proba.
"""
from __future__ import annotations

import numpy as np


def temperature_scale_3class(
    pl: np.ndarray,
    ps: np.ndarray,
    T: float,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply temperature scaling to 3-class proba (long, short, flat).

    - pl, ps: 1d arrays of proba_long, proba_short (same length).
    - pf = 1 - pl - ps (clamped to >= eps).
    - Pseudo-logits: log(p+eps); scaled by T; softmax gives calibrated distribution.
    - T=1.0 => output equals input (no change).

    Returns:
        (pl_cal, ps_cal) as float32, same shape as pl/ps.
    """
    pl = np.asarray(pl, dtype=np.float64)
    ps = np.asarray(ps, dtype=np.float64)
    pl = np.clip(pl, eps, 1.0)
    ps = np.clip(ps, eps, 1.0)
    # pf so that pl + ps + pf = 1; then clip pf to >= eps and renormalize would be ideal,
    # but spec says pf = max(1 - pl - ps, eps). Then we need pl+ps+pf to sum to 1:
    # use normalized 3-class: pf = 1 - pl - ps, then clip and renormalize.
    pf = 1.0 - pl - ps
    pf = np.clip(pf, eps, 1.0)
    # Renormalize so pl+ps+pf = 1 (in case pl+ps > 1)
    s = pl + ps + pf
    pl, ps, pf = pl / s, ps / s, pf / s
    # Pseudo logits (log(p))
    log_pl = np.log(pl + eps)
    log_ps = np.log(ps + eps)
    log_pf = np.log(pf + eps)
    # Scaled by T: logits_i / T (equivalent to log(p^(1/T)))
    logits = np.stack([log_pl / T, log_ps / T, log_pf / T], axis=-1)
    # Softmax with overflow protection: subtract max
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp_ = np.exp(logits)
    sm = exp_ / np.sum(exp_, axis=-1, keepdims=True)
    pl_cal = sm[..., 0].astype(np.float32)
    ps_cal = sm[..., 1].astype(np.float32)
    return pl_cal, ps_cal
