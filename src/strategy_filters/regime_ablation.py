"""
진단용 레짐 라벨 및 ablation 진입 차단 마스크.

전략 로직과 분리: 백테스트 엔진에서 진입 직전 스킵 여부만 결정한다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sideways_bar_mask(df: pd.DataFrame) -> np.ndarray:
    """
    진입 바 기준 sideways 라벨 (엔진 no_sideways와 동일 정의).

    trend_24 = close.pct_change(24); sideways iff abs(trend_24) < 0.001.
    """
    if "close" not in df.columns:
        raise ValueError("df must contain 'close'")
    close = df["close"].astype(float)
    trend_24 = close.pct_change(24)
    t = trend_24.to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return ((np.abs(t) < 0.001) & np.isfinite(t))


_VALID_MODES = frozenset(
    {
        "none",
        "no_sideways",
        "no_high_vol",
        "only_ema_below",
        "no_sideways_high_vol",
        "no_sideways_ema_above",
        "no_high_vol_ema_above",
        "strict_all",
        # 레짐 조합 (진입 차단): diagnostics 전용
        "combo_f1_sideways_high_vol",
        "combo_f2_sideways_ema_above",
        "combo_f3_high_vol_ema_above",
        "combo_f4_sideways_high_vol_ema_above",
        "combo_f5_worst_keys",
    }
)


def compute_regime_components(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    바별 레짐 부품 (엔진 차단·분석 공통).

    trend 상태는 엔진 no_sideways와 동일: abs(trend_24)<0.001 이고 유한할 때만 sideways.

    Returns:
        ema_above, is_sideways, is_high_vol, is_mid_vol, is_low_vol,
        trend_label (sideways|up|down),
        regime_key (예: above_sideways_high)
    """
    if "close" not in df.columns:
        raise ValueError("df must contain 'close'")
    close = df["close"].astype(float)
    n = len(close)
    ema200 = close.rolling(200, min_periods=200).mean()
    ema_above_arr = ((close >= ema200) & ema200.notna()).to_numpy(dtype=bool)

    trend_24 = close.pct_change(24)
    t = trend_24.to_numpy(dtype=float)
    is_sideways = compute_sideways_bar_mask(df)

    trend_label = np.empty(n, dtype=object)
    for i in range(n):
        ti = t[i]
        if is_sideways[i]:
            trend_label[i] = "sideways"
        elif not np.isfinite(ti):
            trend_label[i] = "sideways"
        elif ti >= 0.001:
            trend_label[i] = "up"
        else:
            trend_label[i] = "down"

    ret1 = close.pct_change()
    rv30 = ret1.rolling(30, min_periods=30).std()
    valid_rv = rv30.dropna()
    if len(valid_rv) == 0:
        q25 = np.nan
        q75 = np.nan
    else:
        q25 = float(valid_rv.quantile(0.25))
        q75 = float(valid_rv.quantile(0.75))
    rv = rv30.to_numpy(dtype=float)
    is_high_vol = (rv > q75) & np.isfinite(rv) & np.isfinite(q75)
    is_low_vol = (rv < q25) & np.isfinite(rv) & np.isfinite(q25)
    is_mid_vol = np.isfinite(rv) & (~is_high_vol) & (~is_low_vol)

    vol_bucket = np.empty(n, dtype=object)
    for i in range(n):
        if is_high_vol[i]:
            vol_bucket[i] = "high"
        elif is_low_vol[i]:
            vol_bucket[i] = "low"
        elif is_mid_vol[i]:
            vol_bucket[i] = "mid"
        else:
            vol_bucket[i] = "mid"

    ema_side = np.where(ema_above_arr, "above", "below")
    regime_key = np.empty(n, dtype=object)
    for i in range(n):
        regime_key[i] = f"{ema_side[i]}_{trend_label[i]}_{vol_bucket[i]}"

    return ema_above_arr, is_sideways, is_high_vol, is_mid_vol, is_low_vol, trend_label, regime_key


POSITIVE_REGIME_FILTER_MODES = frozenset(
    {
        "none",
        "allow_up_only",
        "allow_up_mid_high_vol",
        "allow_up_mid_vol_only",
        "allow_up_above_mid_high",
        "allow_non_sideways_mid_high",
        "allow_down_or_up_mid_high",
        "allow_ema_below_or_strong_up",
        "allow_strict_quality",
    }
)


def compute_positive_regime_allow_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    """
    Positive filter: True = 진입 허용 가능 레짐.

    기존 compute_regime_components() 재사용 (ema / trend / vol).
    """
    mode_norm = (mode or "none").strip().lower()
    if mode_norm in ("none", "", "p0_none"):
        return np.ones(len(df), dtype=bool)
    if mode_norm not in POSITIVE_REGIME_FILTER_MODES:
        raise ValueError(
            f"Unknown positive_regime_filter_mode={mode!r}; "
            f"expected one of {sorted(POSITIVE_REGIME_FILTER_MODES)}"
        )
    if "close" not in df.columns:
        raise ValueError("df must contain 'close'")

    close = df["close"].astype(float)
    ema200 = close.rolling(200, min_periods=200).mean()
    ema_valid = ema200.notna().to_numpy(dtype=bool)

    (
        ema_above,
        _sw,
        is_high_vol,
        is_mid_vol,
        _is_low,
        trend_label,
        _rk,
    ) = compute_regime_components(df)

    n = len(df)
    tl = pd.Series(trend_label)
    trend_up = (tl == "up").to_numpy(dtype=bool)
    trend_down = (tl == "down").to_numpy(dtype=bool)
    trend_sw = (tl == "sideways").to_numpy(dtype=bool)

    vol_mid_high = is_mid_vol | is_high_vol
    ema_below_ok = (~ema_above) & ema_valid

    allow = np.zeros(n, dtype=bool)

    if mode_norm == "allow_up_only":
        allow = trend_up
    elif mode_norm == "allow_up_mid_high_vol":
        allow = trend_up & vol_mid_high
    elif mode_norm == "allow_up_mid_vol_only":
        allow = trend_up & is_mid_vol
    elif mode_norm == "allow_up_above_mid_high":
        allow = trend_up & ema_above & vol_mid_high
    elif mode_norm == "allow_non_sideways_mid_high":
        allow = (~trend_sw) & vol_mid_high
    elif mode_norm == "allow_down_or_up_mid_high":
        allow = (trend_up | trend_down) & vol_mid_high
    elif mode_norm == "allow_ema_below_or_strong_up":
        allow = ema_below_ok | (trend_up & vol_mid_high)
    elif mode_norm == "allow_strict_quality":
        allow = trend_up & ema_above & is_mid_vol

    return allow


def compute_positive_regime_block_mask(df: pd.DataFrame, mode: str) -> np.ndarray:
    """True = 진입 차단 (허용 레짐의 반대)."""
    allow = compute_positive_regime_allow_mask(df, mode)
    return ~allow


ADAPTIVE_POSITIVE_MODES = frozenset(
    {
        "none",
        "recent_return_gate_001",
        "recent_return_gate_0015",
        "recent_return_gate_002",
        "volatility_mid_high",
        "trend_strength_001",
        "trend_strength_0015",
        "trend_strength_002",
        "hybrid_regime",
        "hybrid_defensive",
        "combined_best",
    }
)


def compute_adaptive_positive_block_mask(
    df: pd.DataFrame,
    adaptive_mode: str,
    positive_mode: str = "allow_ema_below_or_strong_up",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Adaptive positive regime filter: gate ON인 바에서만 positive filter 적용.

    Returns:
        adaptive_gate_on  – True = 이 바에서 gate 활성 (positive filter 적용 대상)
        positive_block    – True = positive filter가 차단 (gate 무관)
        final_block       – True = 실제 차단 (gate ON & positive block)
    """
    mode = (adaptive_mode or "none").strip().lower()
    if mode not in ADAPTIVE_POSITIVE_MODES:
        raise ValueError(f"Unknown adaptive_positive_regime_mode={adaptive_mode!r}")

    n = len(df)
    if mode == "none":
        return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

    close = df["close"].astype(float)

    (
        ema_above,
        is_sideways,
        is_high_vol,
        is_mid_vol,
        is_low_vol,
        trend_label,
        _rk,
    ) = compute_regime_components(df)

    trend_24 = close.pct_change(24).to_numpy(dtype=float)
    vol_mid_high = is_mid_vol | is_high_vol
    tl = pd.Series(trend_label)
    trend_sw = (tl == "sideways").to_numpy(dtype=bool)
    trend_up = (tl == "up").to_numpy(dtype=bool)

    gate = np.zeros(n, dtype=bool)

    if mode == "recent_return_gate_001":
        with np.errstate(invalid="ignore"):
            gate = (np.abs(trend_24) >= 0.001) & np.isfinite(trend_24)
    elif mode == "recent_return_gate_0015":
        with np.errstate(invalid="ignore"):
            gate = (np.abs(trend_24) >= 0.0015) & np.isfinite(trend_24)
    elif mode == "recent_return_gate_002":
        with np.errstate(invalid="ignore"):
            gate = (np.abs(trend_24) >= 0.002) & np.isfinite(trend_24)
    elif mode == "volatility_mid_high":
        gate = vol_mid_high
    elif mode == "trend_strength_001":
        with np.errstate(invalid="ignore"):
            gate = (np.abs(trend_24) >= 0.001) & np.isfinite(trend_24)
    elif mode == "trend_strength_0015":
        with np.errstate(invalid="ignore"):
            gate = (np.abs(trend_24) >= 0.0015) & np.isfinite(trend_24)
    elif mode == "trend_strength_002":
        with np.errstate(invalid="ignore"):
            gate = (np.abs(trend_24) >= 0.002) & np.isfinite(trend_24)
    elif mode == "hybrid_regime":
        gate = (~trend_sw) & vol_mid_high
    elif mode == "hybrid_defensive":
        gate = ema_above & (trend_sw | trend_up) & (~is_high_vol)
    elif mode == "combined_best":
        gate = vol_mid_high | ((np.abs(trend_24) >= 0.001) & np.isfinite(trend_24))
        gate = gate & ~(is_low_vol & trend_sw)

    positive_block = compute_positive_regime_block_mask(df, positive_mode)
    final_block = gate & positive_block
    return gate, positive_block, final_block


def compute_regime_ablation_block_mask(
    df: pd.DataFrame,
    mode: str,
    *,
    regime_combo_block_keys: tuple[str, ...] | None = None,
) -> np.ndarray:
    """
    바 인덱스 i에서 진입을 차단하면 True.

    - ema200: close rolling mean window=200
    - trend_24: 24바 단순 수익률 (pct_change(24))
    - realized_vol_30: 1바 수익률의 30바 rolling std
    - vol_bucket: 전역 rv30 분포의 25/75 분위 기준 low/mid/high
    """
    mode_norm = (mode or "none").strip().lower()
    if mode_norm not in _VALID_MODES:
        raise ValueError(
            f"Unknown regime_filter_mode={mode!r}; expected one of {sorted(_VALID_MODES)}"
        )
    n = len(df)
    if mode_norm == "none":
        return np.zeros(n, dtype=bool)
    if "close" not in df.columns:
        raise ValueError("df must contain 'close'")

    (
        ema_above,
        _is_sideways,
        is_high_vol,
        _is_mid,
        _is_low,
        _trend_label,
        regime_key_arr,
    ) = compute_regime_components(df)

    sw = compute_sideways_bar_mask(df)

    if mode_norm == "no_sideways":
        return sw
    if mode_norm == "no_high_vol":
        return is_high_vol
    if mode_norm == "only_ema_below":
        return ema_above
    if mode_norm == "no_sideways_high_vol":
        return sw | is_high_vol
    if mode_norm == "no_sideways_ema_above":
        return sw | ema_above
    if mode_norm == "no_high_vol_ema_above":
        return is_high_vol | ema_above
    if mode_norm == "strict_all":
        return sw | is_high_vol | ema_above
    if mode_norm == "combo_f1_sideways_high_vol":
        return sw & is_high_vol
    if mode_norm == "combo_f2_sideways_ema_above":
        return sw & ema_above
    if mode_norm == "combo_f3_high_vol_ema_above":
        return is_high_vol & ema_above
    if mode_norm == "combo_f4_sideways_high_vol_ema_above":
        return sw & is_high_vol & ema_above
    if mode_norm == "combo_f5_worst_keys":
        keys = regime_combo_block_keys or ()
        if not keys:
            return np.zeros(n, dtype=bool)
        block = frozenset(keys)
        out = np.zeros(n, dtype=bool)
        for i in range(n):
            if regime_key_arr[i] in block:
                out[i] = True
        return out

    return np.zeros(n, dtype=bool)
