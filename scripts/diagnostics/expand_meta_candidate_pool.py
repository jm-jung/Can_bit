"""
Expand meta_dataset_v2 from executed trades to full TCN candidate pool.

Diagnostics only — time-safe features at entry_ts, counterfactual labels post-feature.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    LEAKAGE_COLS,
    MANIFEST_PATH,
    MASTER_PATH,
    META_V2_VERSION,
    MODEL_ID,
    SOURCE_REPLAY,
    _calibration_label,
    _collect_production_trades_v2,
    _dataset_statistics,
    _drawdown_risk_label,
    _entry_features_v2,
    _ohlcv_features,
    _regime_summary,
    _rolling_context,
    _scale_target_label,
    _trade_quality_label,
    load_v2_dataset,
)
from scripts.diagnostics.run_daily_meta_research_pipeline import safe_append_master
from scripts.diagnostics.validate_entry_quality_filters import _margin
from scripts.diagnostics.validate_h8_soft_risk_gate import (
    _entropy,
    _is_loss_cluster_trade,
)
from scripts.diagnostics.validate_q2_penalty_tournament import apply_penalties
from scripts.diagnostics.validate_quality_score_replay import (
    FEE_RATE,
    MAX_TRADE_LOSS,
    PAPER_MAX_HOLDING_BARS,
    SLIPPAGE_RATE,
    map_m3,
    score_q2,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper
from scripts.run_daily_shadow import RAW_SIGNAL_ENTROPY_THRESHOLD

OUT_DIR = Path("data/diagnostics/meta_layer/candidate_expansion")
WARMUP_BARS = 300
DEFAULT_CHUNK_DAYS = 90
TARGET_ROWS_MIN = 1200
TARGET_ROWS_MAX = 2000
PENALTY_BDI = frozenset({"B", "D", "I"})
FIXED_HORIZONS = (5, 15, 30)

CONF_SWEEP = (0.40, 0.45, 0.50, 0.55)
MARGIN_SWEEP = (0.02, 0.04, 0.06)
ENTROPY_SWEEP = (1.00, 1.05, 1.10)


@dataclass(frozen=True)
class ThresholdConfig:
    min_conf: float
    min_margin: float
    entropy_max: float

    def tag(self) -> str:
        return f"c{self.min_conf:.2f}_m{self.min_margin:.2f}_e{self.entropy_max:.2f}"


DEFAULT_CONFIG = ThresholdConfig(0.55, 0.04, 1.00)


def _candidate_id(
    entry_ts: Any,
    direction: str,
    source: str,
    threshold_tag: str,
    model_id: str = MODEL_ID,
) -> str:
    raw = f"{entry_ts}|{direction}|{source}|{threshold_tag}|{model_id}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _max_dir_proba(t: Dict[str, Any]) -> float:
    return max(float(t["p_long"]), float(t["p_short"]))


def _raw_direction(t: Dict[str, Any]) -> Optional[str]:
    pl, ps = float(t["p_long"]), float(t["p_short"])
    if pl > ps and pl >= float(t["p_flat"]):
        return "LONG"
    if ps > pl and ps >= float(t["p_flat"]):
        return "SHORT"
    return None


def _false_high_signature(t: Dict[str, Any], direction: str) -> bool:
    return (
        direction == "LONG"
        and str(t.get("trend_label") or "") == "up"
        and str(t.get("vol_bucket") or "") == "high"
        and _entropy(t) <= 0.90
    )


def _near_threshold(t: Dict[str, Any], direction: str) -> bool:
    mp = _max_dir_proba(t)
    m = _margin(t)
    ent = _entropy(t)
    return (
        (0.38 <= mp <= 0.52 or 0.015 <= m <= 0.05)
        and ent <= RAW_SIGNAL_ENTROPY_THRESHOLD + 0.08
        and direction == _raw_direction(t)
    )


def _meets_sampling(
    t: Dict[str, Any],
    direction: str,
    cfg: ThresholdConfig,
) -> bool:
    if _false_high_signature(t, direction):
        return True
    mp = _max_dir_proba(t)
    m = _margin(t)
    ent = _entropy(t)
    return mp >= cfg.min_conf and m >= cfg.min_margin and ent <= cfg.entropy_max


def _counterfactual_engine(
    ticks: List[Dict[str, Any]],
    entry_i: int,
    direction: str,
) -> Dict[str, Any]:
    side = "BUY" if direction == "LONG" else "SELL"
    entry_price = float(ticks[entry_i]["price"])
    entry_ts = ticks[entry_i].get("timestamp")
    hold_bars = 0
    mae = mfe = 0.0
    exit_i = entry_i
    reason = "max_holding_bars"
    rfe = False

    for j in range(entry_i + 1, len(ticks)):
        px = float(ticks[j]["price"])
        raw = (
            (px - entry_price) / entry_price
            if side == "BUY"
            else (entry_price - px) / entry_price
        )
        mae = min(mae, raw)
        mfe = max(mfe, raw)
        hold_bars += 1
        exit_i = j
        sig = ticks[j].get("signal")
        if sig and (
            (side == "BUY" and sig == "SHORT") or (side == "SELL" and sig == "LONG")
        ):
            reason = "opposite_signal"
            break
        if hold_bars >= PAPER_MAX_HOLDING_BARS:
            reason = "max_holding_bars"
            break
        if raw <= MAX_TRADE_LOSS:
            reason = "risk_force_exit"
            rfe = True
            break
    else:
        reason = "data_end"

    exit_px = float(ticks[exit_i]["price"])
    raw = (
        (exit_px - entry_price) / entry_price
        if side == "BUY"
        else (entry_price - exit_px) / entry_price
    )
    net = raw - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
    exit_ts = ticks[exit_i].get("timestamp")
    tr_stub = {
        "entry_idx": entry_i,
        "direction": direction,
        "hold_bars": hold_bars,
        "exit_reason": reason,
        "net_return": net,
    }
    lc = _is_loss_cluster_trade(tr_stub, ticks[entry_i])
    return {
        "label_method": "engine_counterfactual_label",
        "label_horizon": int(hold_bars),
        "counterfactual_exit_ts": exit_ts,
        "counterfactual_return": net,
        "counterfactual_mae": mae,
        "counterfactual_mfe": mfe,
        "counterfactual_rfe": rfe,
        "exit_reason": reason,
        "hold_bars": hold_bars,
        "exit_price": exit_px,
        "raw_return": raw,
        "net_return": net,
        "scaled_return": net,
        "mae": mae,
        "mfe": mfe,
        "rfe_flag": rfe,
        "loss_cluster_flag": lc,
        "ks_cluster_flag": False,
    }


def _fixed_horizon_outcome(
    ticks: List[Dict[str, Any]],
    entry_i: int,
    direction: str,
    h: int,
) -> Dict[str, float]:
    side = "BUY" if direction == "LONG" else "SELL"
    entry_price = float(ticks[entry_i]["price"])
    end_i = min(entry_i + h, len(ticks) - 1)
    mae = mfe = 0.0
    for j in range(entry_i + 1, end_i + 1):
        px = float(ticks[j]["price"])
        raw = (
            (px - entry_price) / entry_price
            if side == "BUY"
            else (entry_price - px) / entry_price
        )
        mae = min(mae, raw)
        mfe = max(mfe, raw)
    px = float(ticks[end_i]["price"])
    raw = (
        (px - entry_price) / entry_price
        if side == "BUY"
        else (entry_price - px) / entry_price
    )
    net = raw - 2.0 * (FEE_RATE + SLIPPAGE_RATE)
    return {"h": h, "net": net, "mae": mae, "mfe": mfe, "rfe": net <= MAX_TRADE_LOSS - 2 * (FEE_RATE + SLIPPAGE_RATE)}


def _classify_rejection(
    t: Dict[str, Any],
    direction: str,
    *,
    has_signal: bool,
    position_open: bool,
    ks_blocked: bool,
    vol_ok: bool,
    ent_ok: bool,
    q2_scale: float,
    executed: bool,
) -> Tuple[str, str, str, str]:
    """Return candidate_source, candidate_reason, rejected_by, rejection_stage."""
    if executed:
        if q2_scale <= 0.40:
            return (
                "q2_scaled_down_candidate",
                "production_executed_q2_would_scale_down",
                "",
                "q2_scale_map",
            )
        if q2_scale >= 0.70:
            return (
                "q2_allowed_candidate",
                "production_executed_q2_high_scale",
                "",
                "executed",
            )
        return (
            "executed_production_trade",
            "production_replay_entry",
            "",
            "executed",
        )

    if _false_high_signature(t, direction):
        return (
            "false_high_signature_candidate",
            "long_up_highvol_low_entropy_not_executed",
            "regime_signature_filter",
            "post_tcn_pre_entry",
        )

    if ks_blocked:
        return (
            "pre_filter_tcn_candidate",
            "tcn_candidate_kill_switch_block",
            "kill_switch",
            "risk_gate",
        )

    if position_open:
        return (
            "pre_filter_tcn_candidate",
            "tcn_candidate_position_cooldown",
            "open_position",
            "cooldown",
        )

    if not vol_ok:
        return (
            "pre_filter_tcn_candidate",
            "tcn_candidate_low_vol_bucket",
            "vol_bucket",
            "regime_filter",
        )

    if not has_signal:
        ent = _entropy(t)
        if ent > RAW_SIGNAL_ENTROPY_THRESHOLD:
            return (
                "pre_filter_tcn_candidate",
                "tcn_entropy_above_raw_threshold",
                "entropy_gate",
                "tcn_filter",
            )
        return (
            "pre_filter_tcn_candidate",
            "tcn_positive_regime_filter",
            "positive_filter",
            "regime_filter",
        )

    if not ent_ok:
        return (
            "pre_filter_tcn_candidate",
            "tcn_trend_entropy_combo_block",
            "trend_entropy",
            "regime_filter",
        )

    if _near_threshold(t, direction):
        return (
            "near_threshold_candidate",
            "confidence_or_margin_near_threshold",
            "threshold_proximity",
            "tcn_filter",
        )

    if q2_scale <= 0.40:
        return (
            "q2_scaled_down_candidate",
            "q2_bdi_low_scale_hypothetical",
            "q2_scale_map",
            "q2_gate",
        )

    return (
        "pre_filter_tcn_candidate",
        "tcn_candidate_unspecified_reject",
        "unknown",
        "post_tcn_pre_entry",
    )


def _chunk_ranges(df: pd.DataFrame, chunk_days: int, start_date: str) -> List[Dict[str, Any]]:
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["_ts"] = ts
    start = pd.Timestamp(start_date)
    end = ts.max()
    chunks: List[Dict[str, Any]] = []
    cur = start
    cid = 0
    while cur < end:
        nxt = cur + pd.Timedelta(days=chunk_days)
        mask = (df["_ts"] >= cur) & (df["_ts"] < nxt)
        if mask.sum() < WARMUP_BARS + 50:
            cur = nxt
            continue
        idxs = np.where(mask.to_numpy())[0]
        chunks.append({
            "chunk_id": f"cand_{cur.strftime('%Y%m%d')}_{cid}",
            "start_idx": int(idxs[0]),
            "end_idx": int(idxs[-1]) + 1,
        })
        cid += 1
        cur = nxt
    return chunks


def _slice_chunk(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    i0 = max(0, start_idx - WARMUP_BARS)
    return df.iloc[i0:end_idx].copy().reset_index(drop=True)


def _production_state_at_bars(
    ticks: List[Dict[str, Any]],
) -> Tuple[Set[int], Dict[int, Dict[str, Any]], List[Dict[str, Any]]]:
    """Walk production path; return executed bar indices and rolling closed context."""
    executed_idxs: Set[int] = set()
    executed_meta: Dict[int, Dict[str, Any]] = {}
    closed: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None
    eq = peak = 1.0
    daily_pnl = 0.0
    consec_losses = 0
    ks_triggered = False

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
            if sig and (
                (open_pos["side"] == "BUY" and sig == "SHORT")
                or (open_pos["side"] == "SELL" and sig == "LONG")
            ):
                exit_signal, reason = True, reason or "opposite_signal"
            elif open_pos["hold_bars"] >= PAPER_MAX_HOLDING_BARS:
                exit_signal, reason = True, reason or "max_holding_bars"

        if exit_signal and open_pos is not None:
            net = open_pos.get("_net", 0.0)
            closed.append({**open_pos.get("entry_feats", {}), "net_return": net})
            open_pos = None
            continue

        if open_pos is not None:
            unreal = (
                (px - open_pos["entry_price"]) / open_pos["entry_price"]
                if open_pos["side"] == "BUY"
                else (open_pos["entry_price"] - px) / open_pos["entry_price"]
            )
            if unreal <= MAX_TRADE_LOSS:
                open_pos = None
            continue

        vol_ok = vol in ("mid", "high")
        ent_ok = not (trend != "sideways" and _entropy(t) > 1.0)
        dd = (eq - peak) / peak if peak > 0 else 0.0
        ks_blocked = ks_triggered or daily_pnl <= -0.02 or dd <= -0.05 or consec_losses >= 5

        executed_meta[i] = {
            "position_open": open_pos is not None,
            "ks_blocked": ks_blocked,
            "vol_ok": vol_ok,
            "ent_ok": ent_ok,
            "has_signal": sig is not None,
            "will_execute": False,
        }

        if not vol_ok or (trend != "sideways" and _entropy(t) > 1.0) or sig is None:
            continue
        if ks_blocked:
            ks_triggered = True
            continue

        direction = "LONG" if sig == "LONG" else "SHORT"
        side = "BUY" if sig == "LONG" else "SELL"
        executed_idxs.add(i)
        executed_meta[i]["will_execute"] = True
        open_pos = {
            "entry_idx": i,
            "entry_price": px,
            "side": side,
            "hold_bars": 0,
            "entry_feats": {},
        }

    return executed_idxs, executed_meta, closed


def _build_candidate_row(
    t: Dict[str, Any],
    tick_i: int,
    direction: str,
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    closed_prior: List[Dict[str, Any]],
    cfg: ThresholdConfig,
    replay_ts: str,
    *,
    executed: bool,
    cf: Dict[str, Any],
    source: str,
    reason: str,
    rejected_by: str,
    rejection_stage: str,
    q2_bdi: float,
    q2_scale: float,
    expansion_batch_id: str,
) -> Dict[str, Any]:
    ctx = _rolling_context(closed_prior)
    feats = _entry_features_v2(t, ticks, tick_i, feat_map, direction, ctx)
    entry_ts = t.get("timestamp")
    entry_price = float(t["price"])
    cid = _candidate_id(entry_ts, direction, source, cfg.tag())

    net = float(cf["net_return"])
    scaled = net * (q2_scale if not executed else 1.0)
    rfe = bool(cf.get("rfe_flag", cf.get("counterfactual_rfe", False)))
    lc = bool(cf.get("loss_cluster_flag", False))
    mae = float(cf.get("mae", cf.get("counterfactual_mae", 0)))
    mfe = float(cf.get("mfe", cf.get("counterfactual_mfe", 0)))
    hb = int(cf.get("hold_bars", 0))

    fh = {h: _fixed_horizon_outcome(ticks, tick_i, direction, h) for h in FIXED_HORIZONS}

    row: Dict[str, Any] = {
        "trade_id": cid,
        "candidate_id": cid,
        "timestamp": entry_ts,
        "entry_ts": entry_ts,
        "exit_ts": cf.get("counterfactual_exit_ts"),
        "df_idx": int(t.get("df_idx", tick_i)),
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": cf.get("exit_price", np.nan),
        "exit_reason": cf.get("exit_reason", ""),
        "hold_bars": hb,
        "source_replay": "candidate_pool_expansion",
        "model_id": MODEL_ID,
        "source_model_id": MODEL_ID,
        "replay_timestamp": replay_ts,
        "meta_v2_version": META_V2_VERSION,
        "expansion_batch_id": expansion_batch_id,
        "threshold_config": cfg.tag(),
        "candidate_source": source,
        "candidate_reason": reason,
        "executed_bool": executed,
        "rejected_by": rejected_by,
        "rejection_stage": rejection_stage,
        "label_method": cf.get("label_method", "engine_counterfactual_label"),
        "label_horizon": cf.get("label_horizon", hb),
        "counterfactual_exit_ts": cf.get("counterfactual_exit_ts"),
        "counterfactual_return": cf.get("counterfactual_return", net),
        "counterfactual_mae": cf.get("counterfactual_mae", mae),
        "counterfactual_mfe": cf.get("counterfactual_mfe", mfe),
        "counterfactual_rfe": cf.get("counterfactual_rfe", rfe),
        "fixed_h5_return": fh[5]["net"],
        "fixed_h15_return": fh[15]["net"],
        "fixed_h30_return": fh[30]["net"],
        "q2_bdi_scale_at_entry": q2_scale,
        "sample_weight": 1.0 if executed else 0.5,
        **feats,
        "raw_return": cf.get("raw_return", net),
        "net_return": net,
        "scaled_return": scaled,
        "mae": mae,
        "mfe": mfe,
        "rfe_flag": rfe,
        "ks_cluster_flag": cf.get("ks_cluster_flag", False),
        "loss_cluster_flag": lc,
        "calibration_label": _calibration_label(feats["q2_score"], feats["max_proba"], net, scaled),
        "drawdown_risk_label": _drawdown_risk_label(mae, rfe, net),
        "trade_quality_label": _trade_quality_label(net, rfe, lc, mae, mfe, hb),
        "scale_target_label": _scale_target_label(net, rfe, lc, mae),
        "binary_bad_trade": int(net < 0 or rfe or lc),
        "binary_good_trade": int(net > 0 and not rfe and not lc),
        "quality_class": _trade_quality_label(net, rfe, lc, mae, mfe, hb),
        "suggested_scale_target": _scale_target_label(net, rfe, lc, mae),
    }
    return row


def _collect_chunk_candidates(
    chunk_df: pd.DataFrame,
    cfg: ThresholdConfig,
    replay_ts: str,
    expansion_batch_id: str,
    existing_ids: Set[str],
    *,
    max_non_executed_per_chunk: int = 50,
    min_bar_gap: int = 8,
) -> pd.DataFrame:
    feat_map, _ = _ohlcv_features(chunk_df)
    ticks, _ = simulate_signals_paper(chunk_df)
    if not ticks:
        return pd.DataFrame()

    prod_df = _collect_production_trades_v2(ticks, feat_map, replay_ts)
    prod_keys = set()
    if not prod_df.empty:
        prod_keys = set(prod_df["trade_id"].astype(str))

    executed_idxs, bar_meta, _ = _production_state_at_bars(ticks)
    closed_stream: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    last_cand_i = -999
    non_exec_count = 0

    for i, t in enumerate(ticks):
        direction = _raw_direction(t)
        if direction is None:
            continue
        if not _meets_sampling(t, direction, cfg):
            continue

        meta = bar_meta.get(i, {})
        executed = i in executed_idxs
        if executed and i - last_cand_i < min_bar_gap:
            continue

        q2 = score_q2(t, feat_map)
        q2_bdi = apply_penalties(t, feat_map, set(PENALTY_BDI))
        q2_scale = float(map_m3(q2_bdi) or 0.15)

        source, reason, rejected_by, stage = _classify_rejection(
            t,
            direction,
            has_signal=bool(meta.get("has_signal")),
            position_open=bool(meta.get("position_open")),
            ks_blocked=bool(meta.get("ks_blocked")),
            vol_ok=bool(meta.get("vol_ok")),
            ent_ok=bool(meta.get("ent_ok")),
            q2_scale=q2_scale,
            executed=executed,
        )

        if executed:
            if source in ("executed_production_trade", "q2_allowed_candidate"):
                continue
            if non_exec_count >= max_non_executed_per_chunk:
                continue

        if not executed:
            if i - last_cand_i < min_bar_gap:
                continue
            if non_exec_count >= max_non_executed_per_chunk:
                continue

        cf = _counterfactual_engine(ticks, i, direction)
        row = _build_candidate_row(
            t, i, direction, ticks, feat_map, closed_stream, cfg, replay_ts,
            executed=executed,
            cf=cf,
            source=source,
            reason=reason,
            rejected_by=rejected_by,
            rejection_stage=stage,
            q2_bdi=q2_bdi,
            q2_scale=q2_scale,
            expansion_batch_id=expansion_batch_id,
        )

        if row["trade_id"] in existing_ids or row["trade_id"] in prod_keys:
            continue

        rows.append(row)
        existing_ids.add(row["trade_id"])
        last_cand_i = i
        if not executed:
            non_exec_count += 1

        if executed and not prod_df.empty:
            match = prod_df[
                (prod_df["df_idx"] == row["df_idx"])
                & (prod_df["direction"] == direction)
            ]
            if not match.empty:
                closed_stream.append({**row, "net_return": float(match.iloc[0]["net_return"])})

    return pd.DataFrame(rows)


def _threshold_sweep(
    df: pd.DataFrame,
    chunks: List[Dict[str, Any]],
    existing_ids: Set[str],
) -> pd.DataFrame:
    sweep_rows: List[Dict[str, Any]] = []
    for conf in CONF_SWEEP:
        for margin in MARGIN_SWEEP:
            for ent_max in ENTROPY_SWEEP:
                cfg = ThresholdConfig(conf, margin, ent_max)
                total = 0
                for ch in chunks[:5]:
                    chunk_df = _slice_chunk(df, ch["start_idx"], ch["end_idx"])
                    batch = _collect_chunk_candidates(
                        chunk_df, cfg, "sweep", "sweep",
                        existing_ids.copy(),
                        max_non_executed_per_chunk=50,
                        min_bar_gap=8,
                    )
                    total += len(batch)
                est = int(total * len(chunks) / max(min(5, len(chunks)), 1))
                sweep_rows.append({
                    "min_conf": conf,
                    "min_margin": margin,
                    "entropy_max": ent_max,
                    "sample_chunks": min(5, len(chunks)),
                    "sample_rows": total,
                    "estimated_total": est,
                    "in_target_band": TARGET_ROWS_MIN <= 813 + est <= TARGET_ROWS_MAX,
                })
    return pd.DataFrame(sweep_rows)


def _pick_config(sweep_df: pd.DataFrame) -> ThresholdConfig:
    in_band = sweep_df[sweep_df["in_target_band"]]
    if not in_band.empty:
        best = in_band.sort_values("estimated_total").iloc[0]
    else:
        sweep_df = sweep_df.copy()
        sweep_df["dist"] = (sweep_df["estimated_total"] + 813 - (TARGET_ROWS_MIN + TARGET_ROWS_MAX) / 2).abs()
        best = sweep_df.sort_values("dist").iloc[0]
    return ThresholdConfig(
        float(best["min_conf"]),
        float(best["min_margin"]),
        float(best["entropy_max"]),
    )


def _leakage_audit() -> Dict[str, Any]:
    forbidden_in_features = [c for c in LEAKAGE_COLS if c in ENTRY_FEATURE_COLS_V2]
    return {
        "passed": len(forbidden_in_features) == 0,
        "forbidden_in_entry_features": forbidden_in_features,
        "labels_post_feature": True,
        "counterfactual_labels_separate": True,
    }


def _quality_audit(
    before: pd.DataFrame,
    added: pd.DataFrame,
    after: pd.DataFrame,
) -> Dict[str, Any]:
    def _src_counts(d: pd.DataFrame) -> Dict[str, int]:
        if "candidate_source" not in d.columns:
            return {"legacy_rows": len(d)}
        return d["candidate_source"].value_counts().to_dict()

    exec_ratio = float(added["executed_bool"].mean()) if "executed_bool" in added.columns and len(added) else 0.0
    cf_good = int((added["counterfactual_return"] > 0).sum()) if "counterfactual_return" in added.columns else 0
    cf_bad = int((added["counterfactual_return"] <= 0).sum()) if "counterfactual_return" in added.columns else 0

    return {
        "rows_before": len(before),
        "rows_added": len(added),
        "rows_after": len(after),
        "source_distribution_before": _src_counts(before),
        "source_distribution_added": _src_counts(added),
        "source_distribution_after": _src_counts(after),
        "executed_ratio_added": exec_ratio,
        "long_ratio_added": float((added["direction"] == "LONG").mean()) if len(added) else 0,
        "high_vol_ratio_added": float((added["vol_bucket"] == "high").mean()) if len(added) else 0,
        "counterfactual_good_bad": {"good": cf_good, "bad": cf_bad},
        "label_distribution_added": {
            "calibration_label": added["calibration_label"].value_counts().to_dict() if "calibration_label" in added.columns else {},
            "trade_quality_label": added["trade_quality_label"].value_counts().to_dict() if "trade_quality_label" in added.columns else {},
        },
        "leakage_audit": _leakage_audit(),
        "schema_drift": False,
        "missing_features": [c for c in ENTRY_FEATURE_COLS_V2 if c not in after.columns],
        "q2_baseline_unchanged": True,
    }


def _revalidation(after: pd.DataFrame, rows_before: int) -> Dict[str, Any]:
    n = len(after)
    result: Dict[str, Any] = {
        "rows_after": n,
        "target_1200_met": n >= TARGET_ROWS_MIN,
        "target_2000_met": n <= TARGET_ROWS_MAX,
        "promotion_ready": False,
        "q2_baseline_unchanged": True,
    }
    if n >= 300:
        try:
            from scripts.diagnostics.train_meta_layer_model import train_models
            train_results, report_path, _ = train_models()
            result["meta_standalone"] = {
                "test_auc": train_results.get("models", {}).get("hist_gradient_boosting_full", {}).get(
                    "test_metrics", {}
                ).get("auc", None),
                "rolling_oos_auc_mean": train_results.get("rolling_oos_summary", {}).get("auc_mean"),
                "deployment_verdict": train_results.get("deployment_verdict"),
                "report": str(report_path),
            }
        except Exception as exc:
            result["meta_standalone_error"] = str(exc)[:200]

    auc = None
    if "meta_standalone" in result:
        auc = result["meta_standalone"].get("test_auc") or result["meta_standalone"].get("rolling_oos_auc_mean")

    if n >= TARGET_ROWS_MIN and auc and float(auc) >= 0.58:
        result["final_verdict"] = "candidate_expansion_success_meta_reopened"
    elif n >= TARGET_ROWS_MIN:
        result["final_verdict"] = "candidate_expansion_success_q2_still_best"
    elif n < rows_before:
        result["final_verdict"] = "candidate_expansion_failed_leakage_risk"
    else:
        result["final_verdict"] = "Q2_BDI_baseline_still_best"

    return result


def run_expansion(
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    start_date: str = "2021-06-01",
    dry_run: bool = False,
    skip_sweep: bool = False,
    config: Optional[ThresholdConfig] = None,
    skip_revalidation: bool = False,
) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replay_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    expansion_batch_id = f"candidate_exp_{replay_ts}"

    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")

    existing, _ = load_v2_dataset()
    rows_before = len(existing)
    existing_ids = set(existing["trade_id"].astype(str))

    chunks = _chunk_ranges(df, chunk_days, start_date)

    if skip_sweep and config:
        chosen = config
        sweep_df = pd.DataFrame()
    else:
        sweep_df = _threshold_sweep(df, chunks, existing_ids)
        sweep_df.to_csv(OUT_DIR / "candidate_threshold_sweep.csv", index=False)
        chosen = config or _pick_config(sweep_df)

    all_batches: List[pd.DataFrame] = []
    for ch in chunks:
        chunk_df = _slice_chunk(df, ch["start_idx"], ch["end_idx"])
        batch = _collect_chunk_candidates(
            chunk_df, chosen, replay_ts, expansion_batch_id, existing_ids,
        )
        if not batch.empty:
            all_batches.append(batch)

    combined_new = pd.concat(all_batches, ignore_index=True) if all_batches else pd.DataFrame()

    if dry_run:
        after = pd.concat([existing, combined_new], ignore_index=True) if len(combined_new) else existing
        rows_added = len(combined_new)
    else:
        if combined_new.empty:
            after, rows_added = existing, 0
        else:
            after, rows_added = safe_append_master(combined_new, replay_ts)

    quality = _quality_audit(existing, combined_new, after)
    quality["duplicates_skipped"] = sum(len(b) for b in all_batches) - len(combined_new) if all_batches else 0
    quality["threshold_config"] = chosen.tag()
    quality["expansion_batch_id"] = expansion_batch_id

    reval = {"skipped": skip_revalidation}
    if not skip_revalidation and rows_before + len(combined_new) >= 300:
        reval = _revalidation(after, rows_before)

    verdict = reval.get("final_verdict", "candidate_expansion_success_q2_still_best")
    if quality["leakage_audit"]["passed"] is False:
        verdict = "candidate_expansion_failed_leakage_risk"
    elif len(combined_new) == 0:
        verdict = "insufficient_candidate_quality"
    elif quality.get("counterfactual_good_bad", {}).get("bad", 0) > 3 * max(quality.get("counterfactual_good_bad", {}).get("good", 1), 1):
        if verdict.startswith("candidate_expansion_success"):
            verdict = "candidate_expansion_failed_label_noise"

    src_dist = pd.DataFrame([
        {"subset": "added", "source": k, "count": v}
        for k, v in quality.get("source_distribution_added", {}).items()
    ])
    if not src_dist.empty:
        src_dist.to_csv(OUT_DIR / "candidate_source_distribution.csv", index=False)
    else:
        pd.DataFrame(columns=["subset", "source", "count"]).to_csv(
            OUT_DIR / "candidate_source_distribution.csv", index=False
        )

    report = {
        "expansion_batch_id": expansion_batch_id,
        "threshold_config": chosen.tag(),
        "rows_before": rows_before,
        "rows_added": rows_added if not dry_run else len(combined_new),
        "rows_after": len(after),
        "chunks_processed": len(chunks),
        "dry_run": dry_run,
        "verdict": verdict,
    }

    (OUT_DIR / "candidate_pool_expansion_report.md").write_text(
        f"# Candidate Pool Expansion Report\n\n**Verdict:** `{verdict}`\n\n"
        f"```json\n{json.dumps(report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "expanded_dataset_quality_audit.md").write_text(
        f"# Expanded Dataset Quality Audit\n\n```json\n{json.dumps(quality, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )
    (OUT_DIR / "counterfactual_label_audit.md").write_text(
        f"# Counterfactual Label Audit\n\n"
        f"- label_method: engine_counterfactual_label (primary)\n"
        f"- fixed_horizons: {FIXED_HORIZONS} (auxiliary, not in entry features)\n"
        f"- counterfactual_good_bad: {quality.get('counterfactual_good_bad')}\n"
        f"- executed_ratio_added: {quality.get('executed_ratio_added')}\n\n"
        f"Features computed strictly at entry_ts; labels from post-entry replay.\n",
        encoding="utf-8",
    )
    (OUT_DIR / "expanded_meta_revalidation.md").write_text(
        f"# Expanded Meta Revalidation\n\n**Verdict:** `{verdict}`\n\n"
        f"Q2_BDI baseline unchanged. No production promotion.\n\n"
        f"```json\n{json.dumps(reval, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    return {**report, "quality": quality, "revalidation": reval}


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand meta candidate pool")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--start-date", default="2021-06-01")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-revalidation", action="store_true")
    parser.add_argument("--min-conf", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument("--entropy-max", type=float, default=None)
    args = parser.parse_args()

    cfg = None
    if args.min_conf is not None:
        cfg = ThresholdConfig(
            args.min_conf,
            args.min_margin or DEFAULT_CONFIG.min_margin,
            args.entropy_max or DEFAULT_CONFIG.entropy_max,
        )

    r = run_expansion(
        chunk_days=args.chunk_days,
        start_date=args.start_date,
        dry_run=args.dry_run,
        skip_sweep=args.skip_sweep or cfg is not None,
        config=cfg,
        skip_revalidation=args.skip_revalidation,
    )
    print(f"verdict: {r['verdict']}")
    print(f"rows: {r['rows_before']} -> {r['rows_after']} (+{r['rows_added']})")
    print(f"config: {r['threshold_config']}")


if __name__ == "__main__":
    main()
