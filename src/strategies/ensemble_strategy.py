from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np

EnsembleMode = Literal[
    "baseline_only",
    "fr2_c4_only",
    "override",
    "confirmation",
    "blend_0.7_0.3",
    "blend_0.5_0.5",
    "blend_0.3_0.7",
]


@dataclass
class EnsembleInputs:
    """Aligned per-bar probabilities for baseline and FR2."""

    pl_base: np.ndarray
    ps_base: np.ndarray
    pl_fr2: np.ndarray
    ps_fr2: np.ndarray
    c4_active: np.ndarray  # bool mask for FR2 C4 gate (uptrend & high_vol & persistence>=6)


def _p_flat(pl: np.ndarray, ps: np.ndarray) -> np.ndarray:
    p_flat = 1.0 - pl - ps
    return np.clip(p_flat, 0.0, 1.0)


def _direction_from_proba(pl: float, ps: float, p_flat: float) -> Literal["LONG", "SHORT", "FLAT"]:
    """3-class argmax direction."""
    if pl >= ps and pl >= p_flat:
        return "LONG"
    if ps >= pl and ps >= p_flat:
        return "SHORT"
    return "FLAT"


def build_fr2_c4_mask(
    regime_uptrend_and_high_vol: np.ndarray,
    persistence_bars: int = 6,
) -> np.ndarray:
    """
    Build FR2 C4 gate mask.

    Definition:
    - trend = uptrend
    - volatility = high_vol
    - min_persistence_bars = 6
    """
    regime_mask = np.asarray(regime_uptrend_and_high_vol, dtype=bool)
    n = len(regime_mask)
    if n == 0:
        return np.zeros(0, dtype=bool)

    if persistence_bars <= 1:
        return regime_mask

    out = np.zeros(n, dtype=bool)
    P = int(persistence_bars)
    for i in range(P - 1, n):
        if np.all(regime_mask[i - P + 1 : i + 1]):
            out[i] = True
    return out


def build_ensemble_proba(
    inputs: EnsembleInputs,
    mode: EnsembleMode,
    *,
    threshold_signal: float = 0.60,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-bar ensemble probabilities (p_long, p_short) for a given mode.

    The flat probability is always reconstructed as p_flat = 1 - p_long - p_short.
    """
    pl_b = np.asarray(inputs.pl_base, dtype=float)
    ps_b = np.asarray(inputs.ps_base, dtype=float)
    pl_f = np.asarray(inputs.pl_fr2, dtype=float)
    ps_f = np.asarray(inputs.ps_fr2, dtype=float)
    c4 = np.asarray(inputs.c4_active, dtype=bool)

    if not (len(pl_b) == len(ps_b) == len(pl_f) == len(ps_f) == len(c4)):
        raise ValueError("EnsembleInputs arrays must all have the same length")

    n = len(pl_b)
    pl_out = np.zeros(n, dtype=float)
    ps_out = np.zeros(n, dtype=float)

    if mode == "baseline_only":
        pl_out[:] = pl_b
        ps_out[:] = ps_b
        return pl_out, ps_out

    if mode == "fr2_c4_only":
        # Use FR2 probabilities only inside C4; outside C4 force FLAT.
        pl_out[c4] = pl_f[c4]
        ps_out[c4] = ps_f[c4]
        # Outside C4 remains (0,0) → p_flat=1
        return pl_out, ps_out

    if mode.startswith("blend_"):
        if mode == "blend_0.7_0.3":
            w_base, w_fr2 = 0.7, 0.3
        elif mode == "blend_0.5_0.5":
            w_base, w_fr2 = 0.5, 0.5
        elif mode == "blend_0.3_0.7":
            w_base, w_fr2 = 0.3, 0.7
        else:
            raise ValueError(f"Unknown blend mode: {mode}")

        # Default: baseline
        pl_out[:] = pl_b
        ps_out[:] = ps_b

        # When C4 active: blend baseline and FR2 scores
        pl_out[c4] = w_base * pl_b[c4] + w_fr2 * pl_f[c4]
        ps_out[c4] = w_base * ps_b[c4] + w_fr2 * ps_f[c4]
        return pl_out, ps_out

    # Modes that work at signal level: override / confirmation
    p_flat_b = _p_flat(pl_b, ps_b)
    p_flat_f = _p_flat(pl_f, ps_f)

    if mode == "override":
        # If C4 active: use FR2 signal (argmax); else baseline signal.
        for i in range(n):
            if c4[i]:
                d = _direction_from_proba(pl_f[i], ps_f[i], p_flat_f[i])
                if d == "LONG":
                    pl_out[i], ps_out[i] = max(pl_f[i], threshold_signal), 0.0
                elif d == "SHORT":
                    pl_out[i], ps_out[i] = 0.0, max(ps_f[i], threshold_signal)
                else:
                    pl_out[i], ps_out[i] = 0.0, 0.0
            else:
                d = _direction_from_proba(pl_b[i], ps_b[i], p_flat_b[i])
                if d == "LONG":
                    pl_out[i], ps_out[i] = max(pl_b[i], threshold_signal), 0.0
                elif d == "SHORT":
                    pl_out[i], ps_out[i] = 0.0, max(ps_b[i], threshold_signal)
                else:
                    pl_out[i], ps_out[i] = 0.0, 0.0
        return pl_out, ps_out

    if mode == "confirmation":
        # Trade only when both models agree on direction; otherwise FLAT.
        for i in range(n):
            d_b = _direction_from_proba(pl_b[i], ps_b[i], p_flat_b[i])
            d_f = _direction_from_proba(pl_f[i], ps_f[i], p_flat_f[i])
            if d_b == d_f and d_b in ("LONG", "SHORT"):
                if d_b == "LONG":
                    # Conservative: use min of the two confidences and enforce threshold.
                    conf = max(min(pl_b[i], pl_f[i]), threshold_signal)
                    pl_out[i], ps_out[i] = conf, 0.0
                else:
                    conf = max(min(ps_b[i], ps_f[i]), threshold_signal)
                    pl_out[i], ps_out[i] = 0.0, conf
            else:
                pl_out[i], ps_out[i] = 0.0, 0.0
        return pl_out, ps_out

    raise ValueError(f"Unsupported ensemble mode: {mode}")

