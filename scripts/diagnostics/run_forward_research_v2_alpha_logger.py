"""
Diagnostics-only Forward Research V2 alpha logger.

This script never calls private/order/account/position APIs and never writes to
production/live/order/state paths. It creates paper-only Research V2 candidates
and resolves prior pending paper rows under:
data/diagnostics/forward_research_v2_alpha_logger/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("data/diagnostics/forward_research_v2_alpha_logger")
RUNS = ROOT / "runs"
STATE = ROOT / "state"
METRICS = ROOT / "metrics"
AUDIT = ROOT / "audit"
SYMBOL = "BTCUSDT"
COST = 0.0006
SLIPPAGE = 0.0002
MAX_BARS = 12000
EXIT_POLICIES = {
    "X1_fixed_24": 24,
    "X2_fixed_48": 48,
    "X3_fixed_96": 96,
    "X4_fixed_144": 144,
    "X7_vol_adjusted_MAE_stop": 48,
    "X10_first_3x_cost_plus_move": 72,
    "X11_trailing_vol_adjusted_proxy": 96,
    "X14_trailing_plus_MAE_stop": 96,
    "X15_setup_specific_exit": 48,
    "X16_timeframe_matched_exit_15m": 36,
    "X17_timeframe_matched_exit_30m": 72,
    "X18_timeframe_matched_exit_1h": 144,
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in sections.items():
        lines += [f"## {key}", ""]
        if isinstance(value, pd.DataFrame):
            lines += ["```csv", value.head(50).to_csv(index=False), "```"]
        elif isinstance(value, (dict, list)):
            lines += ["```json", _json(value), "```"]
        else:
            lines.append(str(value))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_dirs() -> None:
    for p in [RUNS, STATE, METRICS, AUDIT]:
        p.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> Dict[str, Any]:
    return {"path": str(path.relative_to(REPO_ROOT) if path.exists() and path.is_absolute() else path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() and path.is_file() else "", "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _safety_paths() -> List[Path]:
    rels = ["models/tcn_v1.pt", "data/diagnostics/tcn_no_events.pt", "scripts/diagnostics/run_false_high_r7_monitor.py", "scripts/diagnostics/run_false_high_r7_daily_monitor.py", "ops/run_false_high_r7_daily_monitor.sh", "ops/launchd", "risk", "risk_manager", "state", "live", "orders"]
    out: List[Path] = []
    for rel in rels:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:300])
    return out


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _load_ohlcv() -> pd.DataFrame:
    p = REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv"
    meta = REPO_ROOT / "data/diagnostics/data_sync/canonical_data_paths.json"
    if meta.exists():
        try:
            q = REPO_ROOT / json.loads(meta.read_text()).get("canonical_5m_path", "")
            if q.exists():
                p = q
        except Exception:
            pass
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.rename(columns={"open_time": "timestamp", "time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").tail(MAX_BARS).reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))
    df["close_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df


def _tf_minutes(tf: str) -> int:
    return {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}[tf]


def _features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["close"].pct_change()
    out["range"] = out["high"] / out["low"] - 1
    out["body"] = (out["close"] - out["open"]).abs() / out["open"]
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]
    win = max(6, int(24 * 5 / max(_tf_minutes(tf), 5)))
    slow = max(win * 3, win + 5)
    out["atr_proxy"] = out["range"].rolling(win, min_periods=max(3, win // 2)).mean()
    out["realized_vol"] = out["return"].rolling(win, min_periods=max(3, win // 2)).std()
    out["rolling_high"] = out["high"].rolling(win, min_periods=max(3, win // 2)).max()
    out["rolling_low"] = out["low"].rolling(win, min_periods=max(3, win // 2)).min()
    out["range_position"] = (out["close"] - out["rolling_low"]) / (out["rolling_high"] - out["rolling_low"]).replace(0, np.nan)
    out["ema_fast"] = out["close"].ewm(span=max(3, win // 2), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    out["trend_direction"] = np.select([out["ema_fast"] > out["ema_slow"], out["ema_fast"] < out["ema_slow"]], ["up", "down"], default="flat")
    out["trend_strength"] = (out["ema_fast"] / out["ema_slow"] - 1).abs()
    out["swing_high"] = out["high"].rolling(win, min_periods=max(3, win // 2)).max().shift(1)
    out["swing_low"] = out["low"].rolling(win, min_periods=max(3, win // 2)).min().shift(1)
    out["breakout_proxy"] = (out["close"] > out["swing_high"]) | (out["close"] < out["swing_low"])
    out["failed_breakout_proxy"] = ((out["high"] > out["swing_high"]) & (out["close"] < out["swing_high"])) | ((out["low"] < out["swing_low"]) & (out["close"] > out["swing_low"]))
    out["compression_score"] = out["atr_proxy"].rolling(slow, min_periods=win).rank(pct=True).rsub(1)
    out["expansion_score"] = out["atr_proxy"].rolling(slow, min_periods=win).rank(pct=True)
    return out


def _closed(ohlcv: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        return _features(ohlcv.copy(), tf)
    rule = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}[tf]
    res = ohlcv.set_index("timestamp")[["open", "high", "low", "close", "volume"]].resample(rule, label="right", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    res = res.rename(columns={"timestamp": "close_ts"})
    res["timestamp"] = res["close_ts"] - pd.Timedelta(minutes=_tf_minutes(tf))
    return _features(res[["timestamp", "close_ts", "open", "high", "low", "close", "volume"]], tf)


def _regime(mtf: pd.DataFrame, tf: str) -> np.ndarray:
    trend = mtf[f"tf{tf}_trend_direction"].astype(str)
    comp = mtf[f"tf{tf}_compression_score"].fillna(0)
    exp = mtf[f"tf{tf}_expansion_score"].fillna(0)
    pos = mtf[f"tf{tf}_range_position"]
    strength = mtf[f"tf{tf}_trend_strength"].fillna(0)
    rv = mtf[f"tf{tf}_realized_vol"].fillna(0)
    return np.select([pos.isna(), comp > 0.75, (exp > 0.85) & (rv > 0.003), (strength > 0.01) & (pos > 0.85) & trend.eq("up"), (strength > 0.01) & (pos > 0.85) & ~trend.eq("up"), trend.eq("up") & (strength > 0.001), trend.eq("down") & (strength > 0.001), pos.between(0.2, 0.8)], ["REG_NO_TRADE", "REG_COMPRESSION", "REG_HIGH_VOL_TRAP", "REG_LATE_TREND", "REG_EXHAUSTION", "REG_TREND_UP", "REG_TREND_DOWN", "REG_RANGE"], default="REG_CHOP")


def _build_mtf(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = {tf: _closed(ohlcv, tf) for tf in ["5m", "15m", "30m", "1h", "4h", "1d"]}
    joined = frames["5m"][["timestamp", "close_ts", "bar_index", "open", "high", "low", "close", "volume", "return", "body", "upper_wick", "lower_wick"]].copy()
    audits = []
    for tf in ["15m", "30m", "1h", "4h", "1d"]:
        f = frames[tf].sort_values("close_ts")
        small = f.rename(columns={c: f"tf{tf}_{c}" for c in f.columns if c != "close_ts"})
        joined = pd.merge_asof(joined.sort_values("close_ts"), small.sort_values("close_ts"), on="close_ts", direction="backward")
        joined[f"tf{tf}_last_closed_ts"] = pd.merge_asof(joined[["close_ts"]].sort_values("close_ts"), f[["close_ts"]].rename(columns={"close_ts": "src_close_ts"}).sort_values("src_close_ts"), left_on="close_ts", right_on="src_close_ts", direction="backward")["src_close_ts"]
        joined[f"tf{tf}_age_bars"] = (joined["close_ts"] - joined[f"tf{tf}_last_closed_ts"]).dt.total_seconds() / 60 / _tf_minutes(tf)
        audits.append({"timeframe": tf, "rows": len(f), "leakage_rows": int((joined[f"tf{tf}_last_closed_ts"] > joined["close_ts"]).sum()), "missing_rate": float(joined[f"tf{tf}_close"].isna().mean())})
    regime = joined[["timestamp", "close_ts", "bar_index", "close"]].copy()
    for tf in ["1d", "4h", "1h", "30m", "15m"]:
        regime[f"regime_{tf}"] = _regime(joined, tf)
    failed_15m = joined["tf15m_failed_breakout_proxy"].fillna(False).astype(bool)
    regime["trigger_5m_context"] = np.select([(joined["close"] > joined["open"]) & (joined["close"] > joined["tf15m_close"]), (joined["close"] < joined["open"]) & (joined["close"] < joined["tf15m_close"]), failed_15m], ["TRG_5M_BULL_RECLAIM", "TRG_5M_BEAR_REJECT", "TRG_5M_SWEEP"], default="TRG_5M_NEUTRAL")
    reg = regime[["regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m"]].astype(str)
    bull = reg.apply(lambda c: c.str.contains("TREND_UP|COMPRESSION", regex=True), axis=0).sum(axis=1)
    bear = reg.apply(lambda c: c.str.contains("TREND_DOWN|EXHAUSTION", regex=True), axis=0).sum(axis=1)
    regime["regime_alignment_score"] = (bull.sub(bear).abs() / 5).clip(0, 1)
    regime["regime_conflict_score"] = (np.minimum(bull, bear) / 5).clip(0, 1)
    low_move = joined["tf1h_atr_proxy"].fillna(0) / COST < 3
    high_vol = regime["regime_1h"].str.contains("HIGH_VOL_TRAP") | regime["regime_4h"].str.contains("HIGH_VOL_TRAP")
    chop = regime["regime_1h"].str.contains("CHOP|NO_TRADE", regex=True) | regime["regime_30m"].str.contains("CHOP|NO_TRADE", regex=True)
    late = regime["regime_1h"].str.contains("LATE_TREND|EXHAUSTION", regex=True) | regime["regime_4h"].str.contains("LATE_TREND|EXHAUSTION", regex=True)
    regime["bad_regime_score"] = (low_move.astype(float) * 0.30 + high_vol.astype(float) * 0.25 + chop.astype(float) * 0.20 + late.astype(float) * 0.15 + regime["regime_conflict_score"] * 0.20).clip(0, 1)
    regime["bad_regime_reasons"] = np.select([low_move, high_vol, chop, late], ["low_expected_move_to_cost", "high_vol_trap", "chop_no_edge", "late_trend_exhaustion"], default="none")
    regime["alpha_search_allowed"] = regime["bad_regime_score"] < 0.75
    regime["production_block"] = False
    joined["expected_move_to_cost_ratio"] = joined["tf1h_atr_proxy"].fillna(joined["tf30m_atr_proxy"]) / COST
    joined = joined.merge(regime.drop(columns=["timestamp", "close"]), on=["close_ts", "bar_index"], how="left")
    return joined, pd.DataFrame(audits)


def _candidate_hash(row: pd.Series) -> str:
    fields = ["close_ts", "generator_id", "direction", "regime_1h", "regime_15m", "bad_regime_score"]
    return hashlib.sha256("|".join(str(row.get(f, "")) for f in fields).encode()).hexdigest()[:16]


def _generate_candidates(mtf: pd.DataFrame, run_ts: str) -> pd.DataFrame:
    rows = []
    recent = mtf.tail(288)
    for _, r in recent.iterrows():
        checks = [
            ("R2G6_4H_1H_exhaustion_failed_breakout_reversal", "V2S6_4H_or_1H_exhaustion_15m_failed_breakout_reversal", "TF6_4h_macro_1h_regime_15m_setup_5m_trigger", ("LATE_TREND" in str(r["regime_1h"]) or "EXHAUSTION" in str(r["regime_4h"])) and bool(r["tf15m_failed_breakout_proxy"]) and r["tf15m_range_position"] < 0.35, ("LATE_TREND" in str(r["regime_1h"]) or "EXHAUSTION" in str(r["regime_4h"])) and bool(r["tf15m_failed_breakout_proxy"]) and r["tf15m_range_position"] > 0.65),
            ("R2G1_1H_trend_15m_pullback_5m_reclaim", "V2S1_1H_trend_15m_pullback_5m_reclaim", "TF4_1h_regime_15m_setup_5m_trigger", r["regime_1h"] == "REG_TREND_UP" and r["tf15m_range_position"] < 0.45 and r["trigger_5m_context"] == "TRG_5M_BULL_RECLAIM", r["regime_1h"] == "REG_TREND_DOWN" and r["tf15m_range_position"] > 0.55 and r["trigger_5m_context"] == "TRG_5M_BEAR_REJECT"),
            ("R2G2_1H_range_15m_edge_5m_reversal", "V2S2_1H_range_15m_edge_5m_reversal", "TF4_1h_regime_15m_setup_5m_trigger", r["regime_1h"] == "REG_RANGE" and r["tf15m_range_position"] < 0.20 and r["lower_wick"] > r["body"], r["regime_1h"] == "REG_RANGE" and r["tf15m_range_position"] > 0.80 and r["upper_wick"] > r["body"]),
            ("R2G3_30m_breakout_15m_retest_5m_trigger", "V2S3_30m_breakout_15m_retest_5m_trigger", "TF3_30m_primary_15m_setup_5m_trigger", bool(r["tf30m_breakout_proxy"]) and r["close"] > r["tf15m_close"], bool(r["tf30m_breakout_proxy"]) and r["close"] < r["tf15m_close"]),
            ("R2G4_1H_compression_15m_expansion_5m_followthrough", "V2S4_1H_compression_15m_expansion_5m_followthrough", "TF4_1h_regime_15m_setup_5m_trigger", r["tf1h_compression_score"] > 0.70 and r["tf15m_expansion_score"] > 0.70 and r["close"] > r["open"], r["tf1h_compression_score"] > 0.70 and r["tf15m_expansion_score"] > 0.70 and r["close"] < r["open"]),
            ("R2G7_liquidity_sweep_mtf_reclaim", "V2S7_liquidity_sweep_mtf_reclaim", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", bool(r["tf30m_failed_breakout_proxy"]) and r["tf15m_range_position"] < 0.35, bool(r["tf30m_failed_breakout_proxy"]) and r["tf15m_range_position"] > 0.65),
        ]
        votes = 0
        for gid, setup_id, stack, long_c, short_c in checks:
            direction = "LONG" if long_c and not short_c else "SHORT" if short_c and not long_c else ""
            if not direction or r["expected_move_to_cost_ratio"] < 2:
                continue
            row = {
                "run_ts": run_ts,
                "candidate_ts": r["close_ts"],
                "entry_ts": r["close_ts"] + pd.Timedelta(minutes=5),
                "symbol": SYMBOL,
                "direction": direction,
                "timeframe_stack": stack,
                "setup_name": setup_id,
                "generator_id": gid,
                "regime_1d": r["regime_1d"],
                "regime_4h": r["regime_4h"],
                "regime_1h": r["regime_1h"],
                "regime_30m": r["regime_30m"],
                "regime_15m": r["regime_15m"],
                "trigger_5m_context": r["trigger_5m_context"],
                "bad_regime_score": r["bad_regime_score"],
                "bad_regime_reasons": r["bad_regime_reasons"],
                "expected_move_proxy": r["tf1h_atr_proxy"],
                "expected_move_to_cost_ratio": r["expected_move_to_cost_ratio"],
                "expected_edge_score": r["regime_alignment_score"] + r["expected_move_to_cost_ratio"] / 10 - r["bad_regime_score"],
                "q2_decision": "snapshot_unavailable",
                "q2_score": np.nan,
                "q2_scale": np.nan,
                "r7_score": np.nan,
                "r7_high_hazard": False,
                "tcn_p_long": np.nan,
                "tcn_p_short": np.nan,
                "tcn_p_flat": np.nan,
                "tcn_entropy": np.nan,
                "tcn_margin": np.nan,
                "tcn_alignment": True,
                "entry_fill_policy": "next_open_default",
                "allowed_usage": "diagnostics_paper_only",
                "production_action_none": True,
            }
            row["feature_snapshot_hash"] = _candidate_hash(pd.Series({**row, **r.to_dict()}))
            row["research_v2_paper_trade_id"] = f"{SYMBOL}_{gid}_{pd.Timestamp(row['candidate_ts']).strftime('%Y%m%d%H%M')}_{row['feature_snapshot_hash']}"
            rows.append(row)
            votes += 1
        if votes >= 3 and rows:
            ens = rows[-1].copy()
            ens["generator_id"] = "R2G14_ensemble_strict"
            ens["setup_name"] = "V2S14_ensemble_strict"
            ens["research_v2_paper_trade_id"] = f"{SYMBOL}_R2G14_{pd.Timestamp(ens['candidate_ts']).strftime('%Y%m%d%H%M')}_{ens['feature_snapshot_hash']}"
            rows.append(ens)
        if votes >= 2 and rows:
            ens = rows[-1].copy()
            ens["generator_id"] = "R2G15_ensemble_balanced"
            ens["setup_name"] = "V2S15_ensemble_balanced"
            ens["research_v2_paper_trade_id"] = f"{SYMBOL}_R2G15_{pd.Timestamp(ens['candidate_ts']).strftime('%Y%m%d%H%M')}_{ens['feature_snapshot_hash']}"
            rows.append(ens)
    return pd.DataFrame(rows).drop_duplicates(["research_v2_paper_trade_id", "generator_id"]) if rows else pd.DataFrame()


def _resolve_rows(pending: pd.DataFrame, ohlcv: pd.DataFrame, now_ts: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pending.empty:
        return pending, pd.DataFrame()
    o = ohlcv.set_index("close_ts")
    resolved = []
    keep = []
    for _, row in pending.iterrows():
        entry_ts = pd.to_datetime(row["entry_ts"])
        horizon = int(EXIT_POLICIES.get(row["exit_policy_id"], 24))
        sub = o[(o.index >= entry_ts) & (o.index <= entry_ts + pd.Timedelta(minutes=5 * horizon))].copy()
        if len(sub) < horizon:
            keep.append(row.to_dict())
            continue
        entry = float(sub.iloc[0]["open"]) * (1 + (SLIPPAGE if row["direction"] == "LONG" else -SLIPPAGE))
        if row["direction"] == "LONG":
            cr, fav, adv = sub["close"] / entry - 1, sub["high"] / entry - 1, sub["low"] / entry - 1
        else:
            cr, fav, adv = entry / sub["close"] - 1, entry / sub["low"] - 1, entry / sub["high"] - 1
        exit_pos = horizon - 1
        net = float(cr.iloc[exit_pos]) - COST
        out = row.to_dict()
        out.update({"pending_or_resolved": "resolved", "resolution_ts": sub.index[exit_pos], "paper_entry_price": entry, "paper_exit_price": float(sub.iloc[exit_pos]["close"]), "net_after_cost": net, "MFE": float(fav.max()), "MAE": float(adv.min()), "RFE": bool(adv.min() <= -0.006), "MFE_to_cost_ratio": float(fav.max() / COST), "MAE_to_cost_ratio": float(abs(adv.min()) / COST), "time_to_MFE": int(fav.values.argmax()) + 1, "time_to_MAE": int(adv.values.argmin()) + 1, "entry_quality_label": "FORWARD_PENDING_RESOLVED", "exit_quality_label": "FORWARD_POLICY", "censored_flag": False, "max_holding_flag": False})
        resolved.append(out)
    return pd.DataFrame(keep), pd.DataFrame(resolved)


def _state_read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _safety_audit(before: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    after = {"selected_hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status()}
    writes = [{"path": str(p.relative_to(REPO_ROOT)), "under_forward_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(AUDIT / "write_path_audit.csv", index=False)
    ok_hash = before["selected_hashes"] == after["selected_hashes"]
    audit = {"production_hash_unchanged": ok_hash, "all_writes_diagnostics_only": all(w["under_forward_root"] for w in writes), "no_private_api_calls": True, "no_order_endpoint_calls": True, "production_ready": False, "promotion_ready": False}
    _write_md(AUDIT / "forward_logger_safety_audit.md", "Forward Logger Safety Audit", audit)
    _write_md(run_dir / "forward_logger_report.md", "Forward Logger Report", audit)
    return audit


def run_once(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "symbol": SYMBOL, "production_action": "none", "private_api_calls": False, "order_endpoint_calls": False}
    _ensure_dirs()
    run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_iso = datetime.now(timezone.utc).isoformat()
    run_dir = RUNS / run_ts
    run_dir.mkdir(parents=True, exist_ok=True)
    before = {"selected_hashes": [_hash_path(p) for p in _safety_paths()], "python": sys.version, "platform": platform.platform()}
    ohlcv = _load_ohlcv()
    mtf, align = _build_mtf(ohlcv)
    candidates = _generate_candidates(mtf, run_iso)
    regime_cols = [c for c in mtf.columns if c in {"close_ts", "bar_index", "regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m", "bad_regime_score", "bad_regime_reasons", "alpha_search_allowed"}]
    mtf[regime_cols].tail(288).to_parquet(run_dir / "forward_regime_map.parquet", index=False)
    candidates.to_parquet(run_dir / "forward_candidates.parquet", index=False)
    all_candidates_path = STATE / "forward_research_v2_all_candidates.parquet"
    prev_all = _state_read(all_candidates_path)
    all_candidates = pd.concat([prev_all, candidates], ignore_index=True).drop_duplicates(["research_v2_paper_trade_id", "generator_id"]) if len(candidates) or len(prev_all) else pd.DataFrame()
    all_candidates.to_parquet(all_candidates_path, index=False)
    pending_path = STATE / "forward_research_v2_pending_trades.parquet"
    resolved_path = STATE / "forward_research_v2_resolved_trades.parquet"
    pending = _state_read(pending_path)
    new_pending = []
    for _, c in candidates.iterrows():
        for policy in EXIT_POLICIES:
            row = c.to_dict()
            row.update({"exit_policy_id": policy, "pending_or_resolved": "pending", "paper_entry_price": np.nan, "resolution_ts": pd.NaT, "paper_exit_price": np.nan, "net_after_cost": np.nan, "MFE": np.nan, "MAE": np.nan, "RFE": False, "MFE_to_cost_ratio": np.nan, "MAE_to_cost_ratio": np.nan, "time_to_MFE": np.nan, "time_to_MAE": np.nan, "entry_quality_label": "FORWARD_PENDING", "exit_quality_label": "FORWARD_PENDING", "censored_flag": False, "max_holding_flag": False})
            new_pending.append(row)
    pending = pd.concat([pending, pd.DataFrame(new_pending)], ignore_index=True).drop_duplicates(["research_v2_paper_trade_id", "exit_policy_id"]) if len(new_pending) or len(pending) else pd.DataFrame()
    pending, newly_resolved = _resolve_rows(pending, ohlcv, ohlcv["close_ts"].max())
    resolved_prev = _state_read(resolved_path)
    resolved = pd.concat([resolved_prev, newly_resolved], ignore_index=True).drop_duplicates(["research_v2_paper_trade_id", "exit_policy_id"]) if len(newly_resolved) or len(resolved_prev) else pd.DataFrame()
    pending.to_parquet(pending_path, index=False)
    resolved.to_parquet(resolved_path, index=False)
    milestones = pd.DataFrame([{"milestone": n, "resolved": len(resolved), "reached": len(resolved) >= n} for n in [20, 50, 100, 200, 500]])
    milestones.to_csv(METRICS / "forward_milestone_summary.csv", index=False)
    quality = resolved.groupby("generator_id").agg(rows=("research_v2_paper_trade_id", "size"), expectancy=("net_after_cost", "mean"), rfe_rate=("RFE", "mean")) .reset_index() if len(resolved) else pd.DataFrame(columns=["generator_id", "rows", "expectancy", "rfe_rate"])
    quality.to_csv(METRICS / "forward_setup_quality_summary.csv", index=False)
    if len(resolved) and resolved["net_after_cost"].notna().sum() >= 10:
        tmp = resolved.copy()
        tmp["edge_decile"] = pd.qcut(tmp["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")
        dec = tmp.groupby("edge_decile").agg(rows=("research_v2_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), rfe_rate=("RFE", "mean")).reset_index()
    else:
        dec = pd.DataFrame(columns=["edge_decile", "rows", "net_mean", "rfe_rate"])
    dec.to_csv(METRICS / "forward_expected_edge_deciles.csv", index=False)
    summary = {"run_ts": run_iso, "symbol": SYMBOL, "candidates": len(candidates), "pending": len(pending), "newly_resolved": len(newly_resolved), "resolved_total": len(resolved), "discord_attempted": False, "discord_status": "webhook_not_configured_or_not_used", "production_action": "none", "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}
    (run_dir / "forward_run_summary.json").write_text(_json(summary), encoding="utf-8")
    align.to_csv(run_dir / "mtf_alignment_audit.csv", index=False)
    audit = _safety_audit(before, run_dir)
    summary.update({"safety": audit})
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_once(dry_run=args.dry_run)
    print(_json(result) if args.json else f"forward_research_v2_logger candidates={result.get('candidates', 0)} production_action=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
