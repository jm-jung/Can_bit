from __future__ import annotations

"""
FR2 B strategy (override_ensemble, threshold=0.6, regime=off) realtime signal.

CRITICAL:
- model/feature/probability logic is reused as-is from existing components
- if dependencies/models are missing, returns safe HOLD
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from src.services.ohlcv_service import load_ohlcv_df
from src.strategies.ensemble_strategy import EnsembleInputs, build_ensemble_proba

logger = logging.getLogger(__name__)

Signal = Literal["LONG", "SHORT", "HOLD"]


class FR2BOutput(TypedDict):
    timestamp: str
    close: float
    proba_long: float | None
    proba_short: float | None
    signal: Signal


@dataclass(frozen=True)
class FR2BConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "5m"
    threshold: float = 0.6
    models_dir: Path = Path("data/diagnostics/models")
    base_model: str = "tcn_h15_t0p004.pt"
    fr2_model: str = "tcn_h15_micro_v1.pt"
    # Safety/perf: avoid heavy recompute every loop
    max_rows: int = 6000  # enough for TCN windows; keeps runtime bounded
    min_recompute_seconds: int = 60  # throttle even if caller loops faster


_cache_last_ts: str | None = None
_cache_last_out: FR2BOutput | None = None
_cache_last_compute_at: float | None = None


def _align_by_timestamp(df_b: pd.DataFrame, pl_b: np.ndarray, ps_b: np.ndarray, df_f: pd.DataFrame, pl_f: np.ndarray, ps_f: np.ndarray):
    b = df_b.copy()
    f = df_f.copy()
    for d in (b, f):
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    b = b.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    f = f.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    b["pl_base"] = pl_b[: len(b)]
    b["ps_base"] = ps_b[: len(b)]
    f["pl_fr2"] = pl_f[: len(f)]
    f["ps_fr2"] = ps_f[: len(f)]
    joined = b.merge(f[["timestamp", "pl_fr2", "ps_fr2"]], on="timestamp", how="inner")
    joined = joined.sort_values("timestamp").reset_index(drop=True)
    df_bt = joined[["timestamp", "close", "high", "low"]].copy()
    pl_b2 = joined["pl_base"].to_numpy(dtype=np.float32)
    ps_b2 = joined["ps_base"].to_numpy(dtype=np.float32)
    pl_f2 = joined["pl_fr2"].to_numpy(dtype=np.float32)
    ps_f2 = joined["ps_fr2"].to_numpy(dtype=np.float32)
    return df_bt, pl_b2, ps_b2, pl_f2, ps_f2


def _infer_proba_for_model(model_path: Path, df: pd.DataFrame, feature_preset: str, symbol: str, timeframe: str, include_microstructure: bool):
    from src.features.ml_feature_config import MLFeatureConfig
    from src.ml.features import build_feature_frame
    from src.dl.tcn_model import TCNSignalModel
    from src.indicators.basic import add_basic_indicators

    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    df2 = add_basic_indicators(df)
    cfg = MLFeatureConfig.from_preset(feature_preset)
    features = build_feature_frame(df2, symbol=symbol, timeframe=timeframe, feature_config=cfg).dropna()
    model = TCNSignalModel(model_path=model_path, use_events=True, feature_config=cfg)
    if not model.is_loaded():
        raise RuntimeError("TCN model load failed")
    pl, ps = model.predict_proba_batch(features=features, symbol=symbol, timeframe=timeframe, batch_size=512, temperature=1.0)
    return np.asarray(pl, dtype=np.float32), np.asarray(ps, dtype=np.float32), features


def fr2_b_strategy(cfg: FR2BConfig = FR2BConfig()) -> FR2BOutput:
    """
    Compute FR2 B strategy signal using latest available 5m candles.
    """
    try:
        global _cache_last_ts, _cache_last_out, _cache_last_compute_at
        import time

        # Load OHLCV (5m) for base and FR2 (microstructure)
        df_base = load_ohlcv_df(timeframe=cfg.timeframe, symbol=cfg.symbol, include_microstructure=False)
        df_fr2 = load_ohlcv_df(timeframe=cfg.timeframe, symbol=cfg.symbol, include_microstructure=True)
        if df_base.empty or df_fr2.empty:
            raise RuntimeError("OHLCV empty")

        # Fast path: if last candle timestamp unchanged, reuse cached decision
        last_ts = str(df_base.iloc[-1]["timestamp"])
        now = time.time()
        if _cache_last_out is not None and _cache_last_ts == last_ts:
            return _cache_last_out
        # Throttle recompute if caller loops too fast
        if _cache_last_compute_at is not None and (now - _cache_last_compute_at) < float(cfg.min_recompute_seconds):
            if _cache_last_out is not None:
                return _cache_last_out

        # Limit to recent window for speed
        df_base = df_base.tail(cfg.max_rows).reset_index(drop=True)
        df_fr2 = df_fr2.tail(cfg.max_rows).reset_index(drop=True)

        base_path = cfg.models_dir / cfg.base_model
        fr2_path = cfg.models_dir / cfg.fr2_model

        pl_b, ps_b, _ = _infer_proba_for_model(base_path, df_base, "base", cfg.symbol, cfg.timeframe, False)
        pl_f, ps_f, _ = _infer_proba_for_model(fr2_path, df_fr2, "microstructure_v1", cfg.symbol, cfg.timeframe, True)

        df_bt, pl_b2, ps_b2, pl_f2, ps_f2 = _align_by_timestamp(df_base, pl_b, ps_b, df_fr2, pl_f, ps_f)
        inputs = EnsembleInputs(pl_base=pl_b2, ps_base=ps_b2, pl_fr2=pl_f2, ps_fr2=ps_f2, c4_active=None)
        pl_e, ps_e = build_ensemble_proba(inputs, mode="override")

        last = df_bt.iloc[-1]
        pL = float(pl_e[-1])
        pS = float(ps_e[-1])
        close = float(last["close"])
        ts = str(last["timestamp"])

        # Decision: argmax with min_max_proba gate (same as D12 legacy entry gate)
        pF = max(0.0, 1.0 - pL - pS)
        max_p = max(pL, pS, pF)
        if max_p < cfg.threshold:
            sig: Signal = "HOLD"
        else:
            if pL >= pS and pL >= pF:
                sig = "LONG"
            elif pS >= pL and pS >= pF:
                sig = "SHORT"
            else:
                sig = "HOLD"

        return FR2BOutput(timestamp=ts, close=close, proba_long=pL, proba_short=pS, signal=sig)
    except Exception as e:
        logger.warning(f"[FR2-B] strategy failed: {e}")
        try:
            df = load_ohlcv_df(timeframe=cfg.timeframe, symbol=cfg.symbol)
            last = df.iloc[-1]
            return FR2BOutput(
                timestamp=str(last.get("timestamp", "")),
                close=float(last.get("close", 0.0) or 0.0),
                proba_long=None,
                proba_short=None,
                signal="HOLD",
            )
        except Exception:
            return FR2BOutput(timestamp="", close=0.0, proba_long=None, proba_short=None, signal="HOLD")
    finally:
        try:
            # update cache on success path where last_ts exists
            if "ts" in locals() and "close" in locals() and "sig" in locals() and "pL" in locals() and "pS" in locals():
                _cache_last_ts = locals()["ts"]
                _cache_last_out = FR2BOutput(
                    timestamp=locals()["ts"],
                    close=float(locals()["close"]),
                    proba_long=float(locals()["pL"]),
                    proba_short=float(locals()["pS"]),
                    signal=locals()["sig"],
                )
                import time as _t
                _cache_last_compute_at = _t.time()
        except Exception:
            pass

