"""
Meta Layer V2 — append-only dataset factory (diagnostics only).

Each row = one production candidate trade with V2 features and multi-label targets.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import _margin
from scripts.diagnostics.validate_h8_soft_risk_gate import (
    Variant,
    _entropy,
    _h8_pass,
    _is_loss_cluster_trade,
)
from scripts.diagnostics.validate_hybrid_gate_replay import HYBRID_CANDIDATES
from scripts.diagnostics.validate_q2_penalty_tournament import apply_penalties
from scripts.diagnostics.validate_quality_score_replay import (
    FEE_RATE,
    POSITION_SIZE,
    SLIPPAGE_RATE,
    score_q2,
)
from scripts.run_daily_paper import PAPER_MAX_HOLDING_BARS, load_ohlcv, simulate_signals_paper

OUT_DIR = Path("data/diagnostics/meta_layer")
HISTORY_DIR = OUT_DIR / "history"
MASTER_PATH = OUT_DIR / "meta_dataset_v2.parquet"
MANIFEST_PATH = OUT_DIR / "meta_dataset_v2_manifest.json"
LATEST_POINTER = OUT_DIR / "meta_label_dataset_latest.json"

META_V2_VERSION = "2.0"
MODEL_ID = "tcn_production_replay"
SOURCE_REPLAY = "production_candidate_pipeline"

BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]
PENALTY_D = frozenset({"D"})
PENALTY_BDI = frozenset({"B", "D", "I"})

MAX_TRADE_LOSS = -0.01
MAX_DAILY_LOSS = -0.02
MAX_DRAWDOWN = -0.05
MAX_CONSEC_LOSSES = 5

ENTRY_FEATURE_COLS_V2 = [
    "p_long", "p_short", "p_flat", "max_proba", "margin", "entropy",
    "trend_strength", "realized_vol", "recent_return_24", "ema_distance", "price_vs_ema",
    "h8_pass", "h8_fail", "hybrid_a_danger", "direction_long",
    "vol_bucket_high", "vol_bucket_mid", "trend_up", "trend_down", "trend_sideways",
    "q2_score", "q2_pd_score", "q2_bdi_score",
    "q2_pd_penalty_score", "q2_bdi_penalty_score", "hybrid_a_danger_score",
    "danger_cluster_count", "false_high_signature_flag",
    "long_highvol_flag", "trend_up_highvol_flag", "confidence_overextension",
    "volatility_expansion_rate", "entropy_delta", "margin_delta",
    "recent_calibration_error", "regime_transition_flag", "realized_vol_change",
    "rolling_false_positive_density",
]

# backward compat alias
ENTRY_FEATURE_COLS = ENTRY_FEATURE_COLS_V2

LABEL_COLS = [
    "calibration_label", "drawdown_risk_label", "trade_quality_label", "scale_target_label",
    "binary_bad_trade", "binary_good_trade", "quality_class", "suggested_scale_target",
]

LEAKAGE_COLS = {
    "exit_price", "exit_reason", "raw_return", "net_return", "mae", "mfe",
    "rfe_flag", "ks_cluster_flag", "loss_cluster_flag", "hold_bars", "exit_ts",
    *LABEL_COLS,
}


def _max_proba(t: Dict[str, Any]) -> float:
    return max(float(t["p_long"]), float(t["p_short"]), float(t["p_flat"]))


def _ohlcv_features(df: pd.DataFrame) -> Tuple[Dict[int, Dict[str, Any]], pd.DataFrame]:
    px = pd.to_numeric(df["close"], errors="coerce")
    ema = px.ewm(span=20, adjust=False).mean()
    rr = px.pct_change()
    rr24 = px.pct_change(24)
    rv = rr.rolling(24).std()
    rv12 = rr.rolling(12).std()
    ema_dist = (px - ema) / ema.replace(0, np.nan)
    ts_strength = rr.rolling(12).mean().abs()
    feat_df = pd.DataFrame({
        "df_idx": np.arange(len(df)),
        "price_vs_ema": (px > ema).astype(int),
        "ema_distance": ema_dist,
        "recent_return_24": rr24,
        "realized_vol": rv,
        "realized_vol_12": rv12,
        "trend_strength": ts_strength,
        "volatility_expansion_rate": (rv / rv.shift(12).replace(0, np.nan)) - 1.0,
        "realized_vol_change": rv - rv.shift(1),
    })
    feat_map = feat_df.set_index("df_idx").to_dict(orient="index")
    return feat_map, feat_df


def _score_bucket(score: float) -> str:
    if score >= 0.80:
        return "0.80+"
    if score >= 0.65:
        return "0.65-0.80"
    if score >= 0.50:
        return "0.50-0.65"
    return "<0.50"


def _trade_id(entry_ts: Any, df_idx: int, direction: str, entry_price: float) -> str:
    raw = f"{entry_ts}|{df_idx}|{direction}|{entry_price:.8f}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _path_metrics(entry_price: float, side: str, i0: int, hb: int, ticks: List[Dict[str, Any]]) -> Dict[str, float]:
    mae = mfe = 0.0
    for j in range(i0, min(i0 + hb + 1, len(ticks))):
        px = float(ticks[j]["price"])
        raw = (px - entry_price) / entry_price if side == "BUY" else (entry_price - px) / entry_price
        mae = min(mae, raw)
        mfe = max(mfe, raw)
    return {"mae": mae, "mfe": mfe}


def _scale_target_label(net_return: float, rfe_flag: bool, loss_cluster_flag: bool, mae: float) -> float:
    if rfe_flag or loss_cluster_flag or mae <= -0.008:
        return 0.15
    if net_return < -0.003:
        return 0.15
    if net_return <= 0.001:
        return 0.40
    if net_return <= 0.005:
        return 0.70
    return 1.00


def _trade_quality_label(
    net_return: float, rfe_flag: bool, loss_cluster_flag: bool,
    mae: float, mfe: float, hold_bars: int,
) -> int:
    chop = hold_bars >= 8 and abs(mfe) < 0.002 and abs(net_return) < 0.002
    vol_shock = mae <= -0.006 and mfe < abs(mae) * 0.3
    if net_return < -0.003 or rfe_flag or loss_cluster_flag or vol_shock:
        return 0
    if net_return > 0.003 and not rfe_flag and not loss_cluster_flag and not chop:
        return 2
    return 1


def _calibration_label(q2_score: float, max_proba: float, net_return: float, scaled_return: float) -> int:
    high_conf = q2_score >= 0.80 or max_proba >= 0.55
    if high_conf and (net_return < 0 or scaled_return < 0):
        return 1
    if q2_score >= 0.65 and net_return < -0.001:
        return 1
    return 0


def _drawdown_risk_label(mae: float, rfe_flag: bool, net_return: float) -> int:
    if rfe_flag or mae <= -0.005 or net_return <= -0.005:
        return 1
    if mae <= -0.003:
        return 1
    return 0


def _rolling_context(prior: List[Dict[str, Any]]) -> Dict[str, float]:
    if not prior:
        return {"danger_cluster_count": 0.0, "recent_calibration_error": 0.0, "rolling_false_positive_density": 0.0}
    window = prior[-20:]
    fh = sum(1 for r in window if r.get("false_high_signature_flag"))
    cal_err = np.mean([
        float(r.get("q2_score", 0)) - (1.0 if float(r.get("net_return", 0)) > 0 else 0.0)
        for r in window
    ])
    danger = sum(1 for r in window if r.get("false_high_signature_flag") or r.get("hybrid_a_danger"))
    return {
        "danger_cluster_count": float(danger),
        "recent_calibration_error": float(cal_err),
        "rolling_false_positive_density": float(fh / max(len(window), 1)),
    }


def _entry_features_v2(
    t: Dict[str, Any],
    ticks: List[Dict[str, Any]],
    tick_i: int,
    feat_map: Dict[int, Dict[str, Any]],
    direction: str,
    rolling_ctx: Dict[str, float],
) -> Dict[str, Any]:
    idx = int(t.get("df_idx", -1))
    extra = feat_map.get(idx, {})
    ent = _entropy(t)
    h8 = _h8_pass(t)
    q2 = score_q2(t, feat_map)
    q2_pd = apply_penalties(t, feat_map, set(PENALTY_D))
    q2_bdi = apply_penalties(t, feat_map, set(PENALTY_BDI))
    trend = str(t.get("trend_label") or "")
    vol = str(t.get("vol_bucket") or "")
    ts = float(extra.get("trend_strength", 0) or 0) if pd.notna(extra.get("trend_strength")) else 0.0
    rv = float(extra.get("realized_vol", 0) or 0) if pd.notna(extra.get("realized_vol")) else 0.0
    vol_exp = float(extra.get("volatility_expansion_rate", 0) or 0) if pd.notna(extra.get("volatility_expansion_rate")) else 0.0

    prev_t = ticks[tick_i - 1] if tick_i > 0 else t
    prev_trend = str(prev_t.get("trend_label") or "")
    ent_prev = _entropy(prev_t)
    margin_prev = _margin(prev_t)

    is_long = direction == "LONG"
    false_high_sig = int(is_long and trend == "up" and vol == "high" and ent <= 0.90)
    hybrid_danger = bool(HYBRID_A.block_fn(t, feat_map))

    conf_overext = 0.0
    if float(t["p_long"]) > 0.55 and ent < 0.92:
        weak_trend = ts < 0.0003
        if weak_trend or vol_exp > 0.15:
            conf_overext = min(1.0, (float(t["p_long"]) - 0.5) * (1.0 - ent) * (1.0 + max(vol_exp, 0)))

    base = {
        "p_long": float(t["p_long"]), "p_short": float(t["p_short"]), "p_flat": float(t["p_flat"]),
        "max_proba": _max_proba(t), "margin": _margin(t), "entropy": ent,
        "trend_state": trend,
        "trend_strength": ts,
        "vol_bucket": vol,
        "realized_vol": rv,
        "recent_return_24": float(extra.get("recent_return_24", np.nan)) if pd.notna(extra.get("recent_return_24")) else np.nan,
        "ema_distance": float(extra.get("ema_distance", np.nan)) if pd.notna(extra.get("ema_distance")) else np.nan,
        "price_vs_ema": int(extra.get("price_vs_ema", 0)) if pd.notna(extra.get("price_vs_ema")) else 0,
        "h8_pass": h8, "h8_fail": not h8,
        "hybrid_a_danger": hybrid_danger,
        "q2_score": q2, "q2_pd_score": q2_pd, "q2_bdi_score": q2_bdi,
        "q2_bucket": _score_bucket(q2), "q2_pd_bucket": _score_bucket(q2_pd), "q2_bdi_bucket": _score_bucket(q2_bdi),
        "direction_long": int(is_long),
        "vol_bucket_high": int(vol == "high"), "vol_bucket_mid": int(vol == "mid"),
        "trend_up": int(trend == "up"), "trend_down": int(trend == "down"), "trend_sideways": int(trend == "sideways"),
        "q2_pd_penalty_score": max(0.0, q2 - q2_pd),
        "q2_bdi_penalty_score": max(0.0, q2 - q2_bdi),
        "hybrid_a_danger_score": float(hybrid_danger),
        "false_high_signature_flag": false_high_sig,
        "long_highvol_flag": int(is_long and vol == "high"),
        "trend_up_highvol_flag": int(trend == "up" and vol == "high"),
        "confidence_overextension": float(conf_overext),
        "volatility_expansion_rate": vol_exp,
        "entropy_delta": ent - ent_prev,
        "margin_delta": _margin(t) - margin_prev,
        "regime_transition_flag": int(trend != prev_trend),
        "realized_vol_change": float(extra.get("realized_vol_change", 0) or 0) if pd.notna(extra.get("realized_vol_change")) else 0.0,
        **rolling_ctx,
    }
    return base


def _entry_features(t: Dict[str, Any], feat_map: Dict[int, Dict[str, Any]], direction: str) -> Dict[str, Any]:
    """Replay-safe wrapper (no tick index context)."""
    idx = int(t.get("df_idx", -1))
    extra = feat_map.get(idx, {})
    dummy_ticks = [t]
    ctx = _rolling_context([])
    return _entry_features_v2(t, dummy_ticks, 0, feat_map, direction, ctx)


def _collect_production_trades_v2(
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    replay_ts: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False

    def _close(pos: Dict[str, Any], px: float, reason: str) -> None:
        nonlocal eq, peak, daily_pnl, consec_losses, open_pos
        raw = (
            (px - pos["entry_price"]) / pos["entry_price"]
            if pos["side"] == "BUY"
            else (pos["entry_price"] - px) / pos["entry_price"]
        )
        net = raw - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
        scale = float(pos["scale"])
        scaled = net * scale
        eq *= 1.0 + scaled * POSITION_SIZE
        peak = max(peak, eq)
        daily_pnl = eq - 1.0
        consec_losses = consec_losses + 1 if scaled < 0 else 0
        i0 = int(pos["entry_idx"])
        t_entry = ticks[i0]
        path = _path_metrics(float(pos["entry_price"]), pos["side"], i0, int(pos["hold_bars"]), ticks)
        tr_stub = {"entry_idx": i0, "direction": pos["direction"], "hold_bars": int(pos["hold_bars"]), "exit_reason": reason, "net_return": net}
        rfe_flag = reason == "risk_force_exit"
        lc_flag = _is_loss_cluster_trade(tr_stub, t_entry)
        ks_flag = reason == "kill_switch_close"
        entry_ts = pos.get("timestamp")
        exit_ts = ticks[min(i0 + int(pos["hold_bars"]), len(ticks) - 1)].get("timestamp")
        feats = pos["entry_feats"]
        row = {
            "trade_id": pos["trade_id"],
            "timestamp": entry_ts,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "df_idx": int(t_entry.get("df_idx", i0)),
            "direction": pos["direction"],
            "entry_price": float(pos["entry_price"]),
            "exit_price": px,
            "exit_reason": reason,
            "hold_bars": int(pos["hold_bars"]),
            "source_replay": SOURCE_REPLAY,
            "model_id": MODEL_ID,
            "replay_timestamp": replay_ts,
            "meta_v2_version": META_V2_VERSION,
            **feats,
            "raw_return": raw,
            "net_return": net,
            "scaled_return": scaled,
            "mae": path["mae"],
            "mfe": path["mfe"],
            "rfe_flag": rfe_flag,
            "ks_cluster_flag": ks_flag,
            "loss_cluster_flag": lc_flag,
            "calibration_label": _calibration_label(feats["q2_score"], feats["max_proba"], net, scaled),
            "drawdown_risk_label": _drawdown_risk_label(path["mae"], rfe_flag, net),
            "trade_quality_label": _trade_quality_label(net, rfe_flag, lc_flag, path["mae"], path["mfe"], int(pos["hold_bars"])),
            "scale_target_label": _scale_target_label(net, rfe_flag, lc_flag, path["mae"]),
            "binary_bad_trade": int(net < 0 or rfe_flag or lc_flag),
            "binary_good_trade": int(net > 0 and not rfe_flag and not lc_flag),
            "quality_class": _trade_quality_label(net, rfe_flag, lc_flag, path["mae"], path["mfe"], int(pos["hold_bars"])),
            "suggested_scale_target": _scale_target_label(net, rfe_flag, lc_flag, path["mae"]),
        }
        rows.append(row)
        closed.append({**feats, "net_return": net, "false_high_signature_flag": feats.get("false_high_signature_flag", 0)})
        open_pos = None

    for i, t in enumerate(ticks):
        px = float(t["price"])
        sig = t.get("signal")
        vol = str(t.get("vol_bucket") or "")
        trend = str(t.get("trend_label") or "")

        exit_signal = False
        reason = ""
        if ks_triggered and open_pos is not None:
            exit_signal, reason = True, "kill_switch_close"
        if open_pos is not None:
            open_pos["hold_bars"] += 1
            if sig is not None and (
                (open_pos["side"] == "BUY" and sig == "SHORT") or (open_pos["side"] == "SELL" and sig == "LONG")
            ):
                exit_signal, reason = True, reason or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal, reason = True, reason or "max_holding_bars"

        if exit_signal and open_pos is not None:
            _close(open_pos, px, reason or "exit_signal")
            continue

        if open_pos is not None:
            unreal = (
                (px - open_pos["entry_price"]) / open_pos["entry_price"]
                if open_pos["side"] == "BUY"
                else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            )
            if unreal <= MAX_TRADE_LOSS:
                _close(open_pos, px, "risk_force_exit")
            continue

        if vol not in ("mid", "high"):
            continue
        if (trend != "sideways" and _entropy(t) > 1.0) or sig is None:
            continue
        dd = (eq - peak) / peak if peak > 0 else 0.0
        if ks_triggered or daily_pnl <= MAX_DAILY_LOSS or dd <= MAX_DRAWDOWN or consec_losses >= MAX_CONSEC_LOSSES:
            ks_triggered = True
            continue

        direction = "LONG" if sig == "LONG" else "SHORT"
        side = "BUY" if sig == "LONG" else "SELL"
        ctx = _rolling_context(closed)
        feats = _entry_features_v2(t, ticks, i, feat_map, direction, ctx)
        open_pos = {
            "entry_idx": i,
            "entry_price": px,
            "direction": direction,
            "side": side,
            "hold_bars": 0,
            "scale": 1.0,
            "timestamp": t.get("timestamp"),
            "trade_id": _trade_id(t.get("timestamp"), int(t.get("df_idx", i)), direction, px),
            "entry_feats": feats,
        }

    return pd.DataFrame(rows)


def _regime_summary(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {}
    return {
        "vol_distribution": df["vol_bucket"].value_counts().to_dict() if "vol_bucket" in df.columns else {},
        "trend_distribution": df["trend_state"].value_counts().to_dict() if "trend_state" in df.columns else {},
        "long_ratio": float((df["direction"] == "LONG").mean()) if "direction" in df.columns else 0.0,
        "false_high_rate": float(df["false_high_signature_flag"].mean()) if "false_high_signature_flag" in df.columns else 0.0,
    }


def _dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": int(len(df)),
        "label_distribution": {
            "calibration_label": df["calibration_label"].value_counts().to_dict() if "calibration_label" in df.columns else {},
            "drawdown_risk_label": df["drawdown_risk_label"].value_counts().to_dict() if "drawdown_risk_label" in df.columns else {},
            "trade_quality_label": df["trade_quality_label"].value_counts().to_dict() if "trade_quality_label" in df.columns else {},
            "scale_target_label": df["scale_target_label"].value_counts().to_dict() if "scale_target_label" in df.columns else {},
            "binary_bad_trade": df["binary_bad_trade"].value_counts().to_dict() if "binary_bad_trade" in df.columns else {},
        },
        "vol_distribution": df["vol_bucket"].value_counts().to_dict() if "vol_bucket" in df.columns else {},
        "regime_distribution": df["trend_state"].value_counts().to_dict() if "trend_state" in df.columns else {},
        "long_short_ratio": {
            "LONG": int((df["direction"] == "LONG").sum()) if "direction" in df.columns else 0,
            "SHORT": int((df["direction"] == "SHORT").sum()) if "direction" in df.columns else 0,
        },
        **(_regime_summary(df)),
    }


def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [c for c in ("entry_ts", "df_idx", "replay_timestamp") if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last")
    return df.drop_duplicates(subset=["trade_id"], keep="last").reset_index(drop=True)


def _append_master(batch: pd.DataFrame, replay_ts: str) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    if MASTER_PATH.exists():
        master = pd.read_parquet(MASTER_PATH)
        combined = pd.concat([master, batch], ignore_index=True)
    else:
        combined = batch.copy()

    combined = _dedupe(combined)
    combined = combined.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)

    combined.to_parquet(MASTER_PATH, index=False)
    snap = HISTORY_DIR / f"meta_dataset_v2_{replay_ts}.parquet"
    batch.to_parquet(snap, index=False)

    stats = _dataset_statistics(combined)
    manifest = {
        "meta_v2_version": META_V2_VERSION,
        "last_replay_timestamp": replay_ts,
        "source_replay": SOURCE_REPLAY,
        "model_id": MODEL_ID,
        "master_path": str(MASTER_PATH),
        "history_snapshot": str(snap),
        "total_rows": stats["rows"],
        "batch_rows": int(len(batch)),
        "regime_summary": _regime_summary(combined),
        "statistics": stats,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    pointer = {"path": str(MASTER_PATH), "version": META_V2_VERSION, **stats}
    LATEST_POINTER.write_text(json.dumps(pointer, indent=2, default=str), encoding="utf-8")
    return combined


def build_dataset(append: bool = True) -> Tuple[pd.DataFrame, Path]:
    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")
    feat_map, _ = _ohlcv_features(df)
    ticks, _ = simulate_signals_paper(df)
    if not ticks:
        raise SystemExit("No ticks")

    replay_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = _collect_production_trades_v2(ticks, feat_map, replay_ts)
    if batch.empty:
        raise SystemExit("No production trades collected")

    if append:
        dataset = _append_master(batch, replay_ts)
        out_path = MASTER_PATH
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        out_path = OUT_DIR / f"meta_label_dataset_{replay_ts}.parquet"
        batch.to_parquet(out_path, index=False)
        dataset = batch

    summary_path = OUT_DIR / f"meta_dataset_v2_summary_{replay_ts}.md"
    _write_summary_report(summary_path, dataset, replay_ts)
    return dataset, out_path


def _write_summary_report(path: Path, df: pd.DataFrame, replay_ts: str) -> None:
    stats = _dataset_statistics(df)
    lines = [
        "# Meta Dataset V2 Summary",
        "",
        f"- replay_timestamp: {replay_ts}",
        f"- version: {META_V2_VERSION}",
        f"- rows: {stats['rows']}",
        "",
        "## Label Distribution",
        json.dumps(stats["label_distribution"], indent=2),
        "",
        "## Vol / Regime",
        f"- vol: {stats.get('vol_distribution', {})}",
        f"- regime: {stats.get('regime_distribution', {})}",
        f"- LONG/SHORT: {stats.get('long_short_ratio', {})}",
        f"- false_high_rate: {df['false_high_signature_flag'].mean():.3f}" if "false_high_signature_flag" in df.columns else "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_v2_dataset() -> Tuple[pd.DataFrame, Path]:
    if MASTER_PATH.exists():
        return pd.read_parquet(MASTER_PATH), MASTER_PATH
    pointer = LATEST_POINTER
    if pointer.exists():
        path = Path(json.loads(pointer.read_text(encoding="utf-8"))["path"])
        return pd.read_parquet(path), path
    candidates = sorted(OUT_DIR.glob("meta_label_dataset_*.parquet"))
    if not candidates:
        raise SystemExit("No meta dataset found")
    return pd.read_parquet(candidates[-1]), candidates[-1]


def main() -> None:
    dataset, out_path = build_dataset(append=True)
    stats = _dataset_statistics(dataset)
    print(f"meta_v2_version: {META_V2_VERSION}")
    print(f"dataset_rows: {stats['rows']}")
    print(f"calibration_label: {stats['label_distribution'].get('calibration_label', {})}")
    print(f"trade_quality_label: {stats['label_distribution'].get('trade_quality_label', {})}")
    print(f"scale_target_label: {stats['label_distribution'].get('scale_target_label', {})}")
    print(f"long_short: {stats['long_short_ratio']}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
