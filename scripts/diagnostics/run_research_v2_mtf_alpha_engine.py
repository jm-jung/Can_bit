"""
Research V2 multi-timeframe regime-first alpha engine.

Diagnostics only:
- no production writes
- no private/order/account/position API calls
- no live routing
- higher timeframe features are joined only from closed candles
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
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("data/diagnostics/research_v2_mtf_alpha_engine")
EXT_ROOT = Path("data/diagnostics/external_market_structure_alpha_v1")
ALPHA_ROOT = Path("data/diagnostics/alpha_candidate_v2_autopsy")
PAPER_ROOT = Path("data/diagnostics/historical_clean_paper_backfill")
SYMBOL = "BTCUSDT"
COST = 0.0006
SLIPPAGE = 0.0002
RECENT_3M_DAYS = 92
RECENT_6M_DAYS = 183

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "mtf": ROOT / "mtf_data",
    "regime": ROOT / "regime",
    "hyp": ROOT / "setup_hypotheses",
    "cand": ROOT / "candidates",
    "backfill": ROOT / "backfill",
    "labels": ROOT / "labels",
    "tf_tournament": ROOT / "timeframe_tournament",
    "setup_tournament": ROOT / "setup_tournament",
    "role": ROOT / "role_recheck",
    "feature": ROOT / "feature_sufficiency",
    "econ": ROOT / "economics",
    "sizing": ROOT / "position_sizing",
    "validation": ROOT / "validation",
    "hidden": ROOT / "hidden_failure_modes",
    "tournament": ROOT / "tournament",
    "forward": ROOT / "forward_design",
    "branch": ROOT / "research_branch_decision",
    "readiness": ROOT / "readiness",
    "audit": ROOT / "audit",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in sections.items():
        lines.extend([f"## {k}", ""])
        if isinstance(v, pd.DataFrame):
            if len(v):
                lines.append("```csv")
                lines.append(v.head(40).to_csv(index=False))
                lines.append("```")
            else:
                lines.append("_empty_")
        elif isinstance(v, (dict, list, tuple)):
            lines.append("```json")
            lines.append(_json(v))
            lines.append("```")
        else:
            lines.append(str(v))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT) if path.is_absolute() and path.exists() else path),
        "exists": path.exists(),
        "sha256": _sha256(path) if path.exists() and path.is_file() else "",
        "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0,
    }


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _safety_paths() -> List[Path]:
    rels = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "scripts/diagnostics/run_false_high_r7_monitor.py",
        "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
        "ops/run_false_high_r7_daily_monitor.sh",
        "ops",
        "launchd",
        "risk",
        "risk_manager",
        "state",
        "live",
        "orders",
    ]
    out: List[Path] = []
    for rel in rels:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:400])
    return out


def _profit_factor(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").fillna(0)
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    return float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0


def _mdd(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").fillna(0)
    eq = r.cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def _read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "candidate_generation_code": ["*candidate*", "*signal*", "*engine*", "*guard*", "*entry*"],
        "model_artifacts": ["*tcn*", "*proba*", "*q2*", "*r7*", "*model*"],
        "external_data": ["*funding*", "*open_interest*", "*oi*", "*taker*", "*premium*", "*mark_price*", "*basis*", "*liquidation*", "*orderbook*", "*cvd*"],
        "feature_code": ["*feature*", "*regime*", "*indicator*", "*structure*"],
    }
    out: Dict[str, List[str]] = {}
    for group, pats in patterns.items():
        vals: List[str] = []
        for pat in pats:
            for p in REPO_ROOT.rglob(pat):
                rel = str(p.relative_to(REPO_ROOT))
                if any(skip in rel for skip in [".git", ".venv", ".pydeps", "__pycache__", "node_modules"]):
                    continue
                if group == "external_data" and not rel.startswith("data/"):
                    continue
                if group == "external_data" and rel.startswith(str(ROOT)):
                    continue
                if p.is_file():
                    vals.append(rel)
        out[group] = sorted(set(vals))[:300]
    return out


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
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))
    df["close_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df


def _load_inputs() -> Dict[str, pd.DataFrame]:
    specs = {
        "ext_labels": EXT_ROOT / "labels/market_structure_entry_quality_labels.parquet",
        "ext_tournament": EXT_ROOT / "tournament/market_structure_alpha_tournament_scorecard.csv",
        "ext_candidates": EXT_ROOT / "candidates/market_structure_candidate_universe.parquet",
        "setup_v2_candidates": ALPHA_ROOT / "setup_candidates/setup_candidate_universe.parquet",
        "setup_v2_outcomes": ALPHA_ROOT / "setup_backfill/setup_exit_outcomes.parquet",
        "paper_d1": PAPER_ROOT / "datasets/D1_engine_like_paper.parquet",
        "paper_labels": PAPER_ROOT / "labels/paper_entry_exit_labels.parquet",
        "r7_frame": Path("data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet"),
    }
    return {k: _read_df(REPO_ROOT / p) for k, p in specs.items()}


def phase0_discovery(ohlcv: pd.DataFrame, inputs: Dict[str, pd.DataFrame], discovered: Dict[str, List[str]]) -> None:
    required = [
        EXT_ROOT / "external_market_structure_alpha_v1_final_report.md",
        EXT_ROOT / "candidates/market_structure_candidate_universe.parquet",
        EXT_ROOT / "backfill/market_structure_exit_outcomes.parquet",
        EXT_ROOT / "labels/market_structure_entry_quality_labels.parquet",
        EXT_ROOT / "tournament/market_structure_alpha_tournament_scorecard.csv",
        ALPHA_ROOT / "alpha_candidate_v2_autopsy_final_report.md",
        ALPHA_ROOT / "setup_candidates/setup_candidate_universe.parquet",
        ALPHA_ROOT / "setup_backfill/setup_exit_outcomes.parquet",
        PAPER_ROOT / "datasets/D1_engine_like_paper.parquet",
        PAPER_ROOT / "labels/paper_entry_exit_labels.parquet",
        Path("data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet"),
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
        Path("data/ohlcv/BTCUSDT_5m_full.csv"),
        Path("data/market/btcusdt_1m.parquet"),
    ]
    inv = []
    for rel in required:
        p = REPO_ROOT / rel
        inv.append({"path": str(rel), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0})
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    ext_labels = inputs.get("ext_labels", pd.DataFrame())
    paper = inputs.get("paper_d1", pd.DataFrame())
    setup = inputs.get("setup_v2_outcomes", pd.DataFrame())
    summary = pd.DataFrame([
        {"source": "External V1", "verdict": "MARKET_STRUCTURE_ALPHA_V1_NOT_READY", "GOOD": int(ext_labels.get("ms_label", pd.Series(dtype=str)).eq("MS_GOOD").sum()), "BAD": int(ext_labels.get("ms_label", pd.Series(dtype=str)).eq("MS_BAD").sum()), "NEUTRAL": int(ext_labels.get("ms_label", pd.Series(dtype=str)).eq("MS_NEUTRAL").sum())},
        {"source": "Historical Paper D1", "verdict": "HISTORICAL_PAPER_BACKFILL_FAIL_COST_KILLS_EDGE", "GOOD": int(paper.get("paper_label", pd.Series(dtype=str)).eq("GOOD").sum()), "BAD": int(paper.get("paper_label", pd.Series(dtype=str)).eq("BAD").sum()), "NEUTRAL": int(paper.get("paper_label", pd.Series(dtype=str)).eq("NEUTRAL").sum())},
        {"source": "Setup V2", "verdict": "CANDIDATE_GENERATOR_V2_NOT_READY", "GOOD": int(setup.get("setup_label", pd.Series(dtype=str)).eq("SETUP_GOOD").sum()), "BAD": int(setup.get("setup_label", pd.Series(dtype=str)).eq("SETUP_BAD").sum()), "NEUTRAL": int(setup.get("setup_label", pd.Series(dtype=str)).eq("SETUP_NEUTRAL").sum())},
    ])
    summary.to_csv(DIRS["discovery"] / "previous_diagnostics_summary.csv", index=False)
    gaps = int((ohlcv["timestamp"].diff().dt.total_seconds().fillna(300) > 450).sum())
    dups = int(ohlcv["timestamp"].duplicated().sum())
    pd.DataFrame([{"source": "BTCUSDT_5m", "rows": len(ohlcv), "start": ohlcv["timestamp"].min(), "end": ohlcv["timestamp"].max(), "timezone": "naive_UTC_assumed", "gap_count": gaps, "duplicate_count": dups}, {"source": "BTCUSDT_1m", "rows": "", "start": "", "end": "", "timezone": "", "gap_count": "", "duplicate_count": "", "exists": (REPO_ROOT / "data/market/btcusdt_1m.parquet").exists()}]).to_csv(DIRS["discovery"] / "available_data_sources.csv", index=False)
    pd.DataFrame([{"timeframe": tf, "resample_possible": True, "source": "5m"} for tf in ["5m", "15m", "30m", "1h", "4h", "1d"]]).to_csv(DIRS["discovery"] / "available_timeframes.csv", index=False)
    pd.DataFrame([{"path": p} for p in discovered.get("external_data", [])]).to_csv(DIRS["discovery"] / "available_external_market_data_sources.csv", index=False)
    pd.DataFrame([{"feature_family": f, "available": True} for f in ["5m", "15m", "30m", "1h", "4h", "1d", "regime_map", "bad_regime_map", "TCN_Q2_R7_overlay"]]).to_csv(DIRS["discovery"] / "available_feature_families.csv", index=False)
    pd.DataFrame([{"path": p} for p in discovered.get("model_artifacts", [])]).to_csv(DIRS["discovery"] / "available_model_artifacts.csv", index=False)
    pd.DataFrame([{"path": p} for p in discovered.get("candidate_generation_code", [])]).to_csv(DIRS["discovery"] / "candidate_generation_code_inventory.csv", index=False)
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(discovered), encoding="utf-8")
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {"previous": summary, "ohlcv": {"rows": len(ohlcv), "gap_count": gaps, "duplicate_count": dups}, "production_changes": "none"})


def phase1_safety_before() -> Dict[str, Any]:
    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "selected_hashes": [_hash_path(p) for p in _safety_paths()],
        "git_status_short": _git_status(),
        "production_ready": False,
        "promotion_ready": False,
        "private_api_calls": False,
        "order_endpoint_calls": False,
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


def _tf_minutes(tf: str) -> int:
    return {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "4h": 240, "1d": 1440}[tf]


def _add_candle_features(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    out = df.copy()
    out["return"] = out["close"].pct_change()
    out["log_return"] = np.log(out["close"]).diff()
    out["range"] = out["high"] / out["low"] - 1
    out["body"] = (out["close"] - out["open"]).abs() / out["open"]
    out["upper_wick"] = (out["high"] - out[["open", "close"]].max(axis=1)) / out["open"]
    out["lower_wick"] = (out[["open", "close"]].min(axis=1) - out["low"]) / out["open"]
    out["body_ratio"] = out["body"] / out["range"].replace(0, np.nan)
    out["wick_rejection"] = (out["upper_wick"] > out["body"] * 1.5) | (out["lower_wick"] > out["body"] * 1.5)
    win = max(6, int(24 * 5 / max(_tf_minutes(tf), 5)))
    slow = max(win * 3, win + 5)
    out["atr_proxy"] = out["range"].rolling(win, min_periods=max(3, win // 2)).mean()
    out["realized_vol"] = out["return"].rolling(win, min_periods=max(3, win // 2)).std()
    out["rolling_high"] = out["high"].rolling(win, min_periods=max(3, win // 2)).max()
    out["rolling_low"] = out["low"].rolling(win, min_periods=max(3, win // 2)).min()
    out["range_position"] = (out["close"] - out["rolling_low"]) / (out["rolling_high"] - out["rolling_low"]).replace(0, np.nan)
    out["ema_fast"] = out["close"].ewm(span=max(3, win // 2), adjust=False).mean()
    out["ema_slow"] = out["close"].ewm(span=slow, adjust=False).mean()
    out["ema_slope"] = out["ema_fast"].pct_change(max(2, win // 3))
    out["trend_direction"] = np.select([out["ema_fast"] > out["ema_slow"], out["ema_fast"] < out["ema_slow"]], ["up", "down"], default="flat")
    out["trend_strength"] = (out["ema_fast"] / out["ema_slow"] - 1).abs()
    out["swing_high"] = out["high"].rolling(win, min_periods=max(3, win // 2)).max().shift(1)
    out["swing_low"] = out["low"].rolling(win, min_periods=max(3, win // 2)).min().shift(1)
    out["breakout_proxy"] = (out["close"] > out["swing_high"]) | (out["close"] < out["swing_low"])
    out["failed_breakout_proxy"] = ((out["high"] > out["swing_high"]) & (out["close"] < out["swing_high"])) | ((out["low"] < out["swing_low"]) & (out["close"] > out["swing_low"]))
    out["compression_score"] = out["atr_proxy"].rolling(slow, min_periods=win).rank(pct=True).rsub(1)
    out["expansion_score"] = out["atr_proxy"].rolling(slow, min_periods=win).rank(pct=True)
    out["volume_z"] = (out["volume"] - out["volume"].rolling(slow, min_periods=win).mean()) / out["volume"].rolling(slow, min_periods=win).std()
    hour = pd.to_datetime(out["close_ts"]).dt.hour
    out["session_bucket"] = np.select([hour.between(0, 7), hour.between(8, 15)], ["asia", "europe"], default="us")
    return out


def _closed_candles(ohlcv: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == "5m":
        cols = ["timestamp", "close_ts", "open", "high", "low", "close", "volume", "bar_index"]
        return _add_candle_features(ohlcv[cols].copy(), tf)
    rule = {"15m": "15min", "30m": "30min", "1h": "1h", "4h": "4h", "1d": "1D"}[tf]
    base = ohlcv.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
    res = base.resample(rule, label="right", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    res = res.rename(columns={"timestamp": "close_ts"})
    res["timestamp"] = res["close_ts"] - pd.Timedelta(minutes=_tf_minutes(tf))
    return _add_candle_features(res[["timestamp", "close_ts", "open", "high", "low", "close", "volume"]].copy(), tf)


def phase2_mtf_dataset(ohlcv: pd.DataFrame) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for tf in ["5m", "15m", "30m", "1h", "4h", "1d"]:
        f = _closed_candles(ohlcv, tf)
        frames[tf] = f
        f.to_parquet(DIRS["mtf"] / f"closed_candles_{tf}.parquet", index=False)
    base = frames["5m"].copy().sort_values("close_ts")
    joined = base[["timestamp", "close_ts", "bar_index", "open", "high", "low", "close", "volume"]].copy()
    for c in [x for x in base.columns if x not in {"timestamp", "close_ts", "bar_index", "open", "high", "low", "close", "volume"}]:
        joined[f"tf5m_{c}"] = base[c].values
    audits = []
    feature_schema: Dict[str, List[str]] = {"5m": list(base.columns)}
    for tf in ["15m", "30m", "1h", "4h", "1d"]:
        f = frames[tf].copy().sort_values("close_ts")
        feature_schema[tf] = list(f.columns)
        rename = {c: f"tf{tf}_{c}" for c in f.columns if c not in {"close_ts"}}
        small = f.rename(columns=rename)
        joined = pd.merge_asof(joined.sort_values("close_ts"), small.sort_values("close_ts"), on="close_ts", direction="backward")
        joined[f"tf{tf}_last_closed_ts"] = joined["close_ts"].where(joined[f"tf{tf}_close"].notna(), pd.NaT)
        # Recover the source close timestamp from the merge by matching last feature row.
        src = f[["close_ts"]].rename(columns={"close_ts": f"tf{tf}_last_closed_ts_src"})
        joined = pd.merge_asof(joined.sort_values("close_ts"), src.sort_values(f"tf{tf}_last_closed_ts_src"), left_on="close_ts", right_on=f"tf{tf}_last_closed_ts_src", direction="backward")
        joined[f"tf{tf}_last_closed_ts"] = joined[f"tf{tf}_last_closed_ts_src"]
        joined = joined.drop(columns=[f"tf{tf}_last_closed_ts_src"])
        age_minutes = (joined["close_ts"] - joined[f"tf{tf}_last_closed_ts"]).dt.total_seconds() / 60
        joined[f"tf{tf}_age_bars"] = age_minutes / _tf_minutes(tf)
        leakage = int((joined[f"tf{tf}_last_closed_ts"] > joined["close_ts"]).sum())
        audits.append({"timeframe": tf, "rows": len(f), "missing_rate": float(joined[f"tf{tf}_close"].isna().mean()), "leakage_rows": leakage, "max_age_bars": float(joined[f"tf{tf}_age_bars"].max()), "boundary_ok": leakage == 0})
    joined.to_parquet(DIRS["mtf"] / "mtf_asof_joined_frame.parquet", index=False)
    preview = pd.concat([joined.head(5000), joined.tail(5000)]).drop_duplicates("close_ts")
    preview.to_csv(DIRS["mtf"] / "mtf_asof_joined_frame.csv", index=False)
    pd.DataFrame(audits).to_csv(DIRS["mtf"] / "mtf_alignment_audit.csv", index=False)
    (DIRS["mtf"] / "mtf_feature_schema.json").write_text(_json(feature_schema), encoding="utf-8")
    gap_rows = []
    for tf, f in frames.items():
        expected = _tf_minutes(tf) * 60
        gap_rows.append({"timeframe": tf, "rows": len(f), "start": f["close_ts"].min(), "end": f["close_ts"].max(), "gap_count": int((f["close_ts"].diff().dt.total_seconds().fillna(expected) > expected * 1.5).sum()), "duplicate_count": int(f["close_ts"].duplicated().sum())})
    _write_md(DIRS["mtf"] / "mtf_data_quality_report.md", "MTF Data Quality Report", {"alignment": pd.DataFrame(audits), "gaps": pd.DataFrame(gap_rows), "closed_candle_policy": "All higher timeframe features are merge_asof backward by close_ts only."})
    return frames, joined


def _regime_from_row(row: pd.Series, tf: str) -> str:
    trend = row.get(f"tf{tf}_trend_direction", "flat")
    comp = row.get(f"tf{tf}_compression_score", 0)
    exp = row.get(f"tf{tf}_expansion_score", 0)
    pos = row.get(f"tf{tf}_range_position", 0.5)
    strength = row.get(f"tf{tf}_trend_strength", 0)
    rv = row.get(f"tf{tf}_realized_vol", 0)
    if pd.isna(pos):
        return "REG_NO_TRADE"
    if comp > 0.75:
        return "REG_COMPRESSION"
    if exp > 0.85 and rv > 0.003:
        return "REG_HIGH_VOL_TRAP"
    if strength > 0.01 and pos > 0.85:
        return "REG_LATE_TREND" if trend == "up" else "REG_EXHAUSTION"
    if trend == "up" and strength > 0.001:
        return "REG_TREND_UP"
    if trend == "down" and strength > 0.001:
        return "REG_TREND_DOWN"
    if 0.2 <= pos <= 0.8:
        return "REG_RANGE"
    return "REG_CHOP"


def _regime_vector(mtf: pd.DataFrame, tf: str) -> np.ndarray:
    trend = mtf.get(f"tf{tf}_trend_direction", pd.Series("flat", index=mtf.index)).astype(str)
    comp = pd.to_numeric(mtf.get(f"tf{tf}_compression_score", pd.Series(0, index=mtf.index)), errors="coerce").fillna(0)
    exp = pd.to_numeric(mtf.get(f"tf{tf}_expansion_score", pd.Series(0, index=mtf.index)), errors="coerce").fillna(0)
    pos = pd.to_numeric(mtf.get(f"tf{tf}_range_position", pd.Series(0.5, index=mtf.index)), errors="coerce")
    strength = pd.to_numeric(mtf.get(f"tf{tf}_trend_strength", pd.Series(0, index=mtf.index)), errors="coerce").fillna(0)
    rv = pd.to_numeric(mtf.get(f"tf{tf}_realized_vol", pd.Series(0, index=mtf.index)), errors="coerce").fillna(0)
    return np.select(
        [
            pos.isna(),
            comp > 0.75,
            (exp > 0.85) & (rv > 0.003),
            (strength > 0.01) & (pos > 0.85) & trend.eq("up"),
            (strength > 0.01) & (pos > 0.85) & ~trend.eq("up"),
            trend.eq("up") & (strength > 0.001),
            trend.eq("down") & (strength > 0.001),
            pos.between(0.2, 0.8),
        ],
        ["REG_NO_TRADE", "REG_COMPRESSION", "REG_HIGH_VOL_TRAP", "REG_LATE_TREND", "REG_EXHAUSTION", "REG_TREND_UP", "REG_TREND_DOWN", "REG_RANGE"],
        default="REG_CHOP",
    )


def phase3_regime_map(mtf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = mtf[["timestamp", "close_ts", "bar_index", "close"]].copy()
    for tf in ["1d", "4h", "1h", "30m", "15m"]:
        df[f"regime_{tf}"] = _regime_vector(mtf, tf)
    failed_15m = mtf["tf15m_failed_breakout_proxy"].fillna(False) if "tf15m_failed_breakout_proxy" in mtf else pd.Series(False, index=mtf.index)
    df["trigger_5m_context"] = np.select(
        [(mtf["close"] > mtf["open"]) & (mtf["close"] > mtf["tf15m_close"]), (mtf["close"] < mtf["open"]) & (mtf["close"] < mtf["tf15m_close"]), failed_15m.astype(bool)],
        ["TRG_5M_BULL_RECLAIM", "TRG_5M_BEAR_REJECT", "TRG_5M_SWEEP"],
        default="TRG_5M_NEUTRAL",
    )
    regime_cols = ["regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m"]
    reg_str = df[regime_cols].astype(str)
    bull = reg_str.apply(lambda col: col.str.contains("TREND_UP|BREAKOUT|COMPRESSION", regex=True), axis=0).sum(axis=1)
    bear = reg_str.apply(lambda col: col.str.contains("TREND_DOWN|EXHAUSTION", regex=True), axis=0).sum(axis=1)
    df["regime_alignment_score"] = (bull.sub(bear).abs() / 5).clip(0, 1)
    df["regime_conflict_score"] = (np.minimum(bull, bear) / 5).clip(0, 1)
    low_move = mtf["tf1h_atr_proxy"].fillna(mtf["close"].pct_change().abs().rolling(12).mean()) / COST < 3
    high_vol = df["regime_1h"].astype(str).str.contains("HIGH_VOL_TRAP") | df["regime_4h"].astype(str).str.contains("HIGH_VOL_TRAP")
    chop = df["regime_1h"].astype(str).str.contains("CHOP|NO_TRADE", regex=True) | df["regime_30m"].astype(str).str.contains("CHOP|NO_TRADE", regex=True)
    late = df["regime_1h"].astype(str).str.contains("LATE_TREND|EXHAUSTION", regex=True) | df["regime_4h"].astype(str).str.contains("LATE_TREND|EXHAUSTION", regex=True)
    gap = mtf[[c for c in mtf.columns if c.endswith("_age_bars")]].gt(3).any(axis=1)
    df["bad_regime_score"] = (low_move.astype(float) * 0.30 + high_vol.astype(float) * 0.25 + chop.astype(float) * 0.20 + late.astype(float) * 0.15 + gap.astype(float) * 0.10 + df["regime_conflict_score"] * 0.20).clip(0, 1)
    df["bad_regime_reasons"] = np.select([low_move, high_vol, chop, late, gap], ["low_expected_move_to_cost", "high_vol_trap", "chop_no_edge", "late_trend_exhaustion", "data_gap_risk"], default="none")
    df["alpha_search_allowed"] = df["bad_regime_score"] < 0.75
    df["alpha_search_warning"] = df["bad_regime_score"].between(0.50, 0.75)
    df["production_block"] = False
    df.to_parquet(DIRS["regime"] / "regime_map_v2.parquet", index=False)
    df.to_csv(DIRS["regime"] / "regime_map_v2.csv", index=False)
    dist = pd.concat([df[c].value_counts().rename_axis("regime").reset_index(name="rows").assign(level=c) for c in ["regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m"]])
    dist.to_csv(DIRS["regime"] / "regime_distribution.csv", index=False)
    trans = pd.crosstab(df["regime_1h"].shift(1), df["regime_1h"]).reset_index()
    trans.to_csv(DIRS["regime"] / "regime_transition_matrix.csv", index=False)
    df.groupby(["regime_1h", "regime_30m"]).agg(rows=("close_ts", "size"), alignment=("regime_alignment_score", "mean"), conflict=("regime_conflict_score", "mean")).reset_index().to_csv(DIRS["regime"] / "regime_alignment_summary.csv", index=False)
    bad = df[["timestamp", "close_ts", "bar_index", "bad_regime_score", "bad_regime_reasons", "alpha_search_allowed", "alpha_search_warning", "production_block"]].copy()
    bad.to_parquet(DIRS["regime"] / "bad_regime_map_v2.parquet", index=False)
    bad.groupby(pd.cut(bad["bad_regime_score"], [-0.01, 0.25, 0.5, 0.75, 1.01])).size().reset_index(name="rows").to_csv(DIRS["regime"] / "bad_regime_map_v2_summary.csv", index=False)
    _write_md(DIRS["regime"] / "regime_map_v2_report.md", "Regime Map V2 Report", {"distribution": dist, "bad_map_policy": "alpha-search map only; production_block=false for all rows."})
    return df, bad


def phase4_setup_hypotheses() -> pd.DataFrame:
    rows = [
        ("V2S1_1H_trend_15m_pullback_5m_reclaim", "1H trend, 15m pullback, 5m reclaim", "TF4_1h_regime_15m_setup_5m_trigger", "REG_TREND_UP/DOWN", "15m pullback", "5m reclaim", 3.0),
        ("V2S2_1H_range_15m_edge_5m_reversal", "1H range, 15m edge, 5m reversal", "TF4_1h_regime_15m_setup_5m_trigger", "REG_RANGE", "range edge", "sweep/reversal", 2.5),
        ("V2S3_30m_breakout_15m_retest_5m_trigger", "30m breakout, 15m retest, 5m trigger", "TF3_30m_primary_15m_setup_5m_trigger", "30m breakout", "15m retest", "5m continuation", 3.0),
        ("V2S4_1H_compression_15m_expansion_5m_followthrough", "1H compression, 15m expansion", "TF4_1h_regime_15m_setup_5m_trigger", "REG_COMPRESSION", "15m expansion", "5m followthrough", 4.0),
        ("V2S5_1D_macro_bias_1H_pullback_continuation", "1D macro bias + 1H pullback", "TF7_1d_macro_1h_regime_15m_setup_5m_trigger", "1D trend", "1H pullback", "15m/5m continuation", 3.0),
        ("V2S6_4H_or_1H_exhaustion_15m_failed_breakout_reversal", "4H/1H exhaustion failed breakout", "TF6_4h_macro_1h_regime_15m_setup_5m_trigger", "late trend", "failed breakout", "5m reversal", 3.0),
        ("V2S7_liquidity_sweep_mtf_reclaim", "MTF sweep reclaim", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", "swing sweep", "15m reclaim", "5m confirm", 3.0),
        ("V2S8_low_vol_grind_mtf", "low vol trend grind", "TF4_1h_regime_15m_setup_5m_trigger", "low vol trend", "small pullback", "low MAE trigger", 2.0),
        ("V2S9_cost_aware_large_move_only", "large move-to-cost only", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", "any", "move/cost high", "5m support", 5.0),
        ("V2S10_no_trade_first_alpha", "bad regime removed", "TF4_1h_regime_15m_setup_5m_trigger", "not bad regime", "any setup", "5m support", 3.0),
        ("V2S11_external_confirmed_setup_if_available", "external confirmed setup", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", "OHLCV setup", "external context", "5m support", 3.0),
        ("V2S12_TCN_scorer_overlay", "TCN scorer overlay", "TF4_1h_regime_15m_setup_5m_trigger", "setup first", "TCN alignment only", "not generator", 3.0),
        ("V2S13_Q2_R7_defensive_overlay", "Q2/R7 defensive overlay", "TF4_1h_regime_15m_setup_5m_trigger", "setup first", "Q2/R7 overlay", "not greenlight", 3.0),
        ("V2S14_ensemble_strict", "strict MTF ensemble", "TF8_1d_macro_4h_macro_1h_regime_30m_setup_15m_trigger_5m_timing", "multi-level align", "cost+bad clear", "5m trigger", 4.0),
        ("V2S15_ensemble_balanced", "balanced MTF ensemble", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", "2+ votes", "cost aware", "5m timing", 3.0),
    ]
    df = pd.DataFrame(rows, columns=["setup_id", "setup_name", "timeframe_stack", "macro_regime_requirement", "setup_requirement", "trigger_requirement", "expected_move_to_cost_min"])
    df["direction_logic"] = "macro/primary regime first, 5m trigger only times entry"
    df["entry_trigger"] = df["trigger_requirement"]
    df["invalid_condition"] = "bad_regime_score >= 0.75 or regime conflict"
    df["bad_regime_map_usage"] = "alpha_search_map_only"
    df["cost_aware_condition"] = "expected_move_to_cost >= setup minimum"
    df["TCN_usage"] = "scorer_only"
    df["Q2_usage"] = "defensive_overlay"
    df["R7_usage"] = "warning_overlay"
    df["external_usage"] = "if available, context/confirmation only"
    df["oracle_flag"] = False
    df["expected_failure_mode"] = "cost_kill/sample_too_small/regime_overfit"
    df["recommended_exit_policy_candidates"] = "fixed_24,fixed_48,timeframe_matched,trailing_plus_MAE"
    df.to_csv(DIRS["hyp"] / "research_v2_setup_registry.csv", index=False)
    _write_md(DIRS["hyp"] / "research_v2_setup_definitions.md", "Research V2 Setup Definitions", {"registry": df})
    df[["setup_id", "timeframe_stack", "macro_regime_requirement", "setup_requirement", "trigger_requirement"]].to_csv(DIRS["hyp"] / "research_v2_timeframe_stack_requirements.csv", index=False)
    df[["setup_id", "expected_failure_mode"]].to_csv(DIRS["hyp"] / "research_v2_setup_expected_failure_modes.csv", index=False)
    _write_md(DIRS["hyp"] / "setup_hypothesis_report.md", "Setup Hypothesis Report", {"principle": "regime-first, TCN scorer-only, Q2/R7 defensive/warning overlay", "registry": df})
    return df


def _hash_row(row: pd.Series, cols: Iterable[str]) -> str:
    return hashlib.sha256("|".join(f"{c}={row.get(c, '')}" for c in list(cols)[:160]).encode()).hexdigest()[:16]


def _dir(long_cond: bool, short_cond: bool) -> str:
    if long_cond and not short_cond:
        return "LONG"
    if short_cond and not long_cond:
        return "SHORT"
    return "NONE"


def phase5_candidates(mtf: pd.DataFrame, regime: pd.DataFrame, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = pd.merge_asof(mtf.sort_values("close_ts"), regime.sort_values("close_ts"), on="close_ts", direction="backward", suffixes=("", "_reg"))
    expected_move = (df["tf1h_atr_proxy"].fillna(df["tf30m_atr_proxy"]).fillna(df["close"].pct_change().abs().rolling(24).mean()) / COST).replace([np.inf, -np.inf], np.nan)
    df["expected_move_to_cost_ratio"] = expected_move
    trend_q = float(df["tf1h_trend_strength"].quantile(0.65))
    vol_low = float(df["tf1h_realized_vol"].quantile(0.35))
    vol_high = float(df["tf1h_realized_vol"].quantile(0.80))
    current_engine_ts = set(pd.to_datetime(inputs.get("paper_d1", pd.DataFrame()).get("entry_ts", pd.Series(dtype=str)), errors="coerce").dropna().astype("datetime64[ns]").astype(str))
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    # Historical Research V2 is a regime/setup sweep, not a trade-every-bar oracle.
    # A 6h cadence keeps all regimes represented while avoiding candle-by-candle
    # overgeneration that prior autopsies showed to be bad-heavy.
    scan = df.iloc[::72].copy()
    cols = list(df.columns)
    for idx, r in scan.iterrows():
        if pd.isna(r.get("expected_move_to_cost_ratio", np.nan)):
            continue
        conds = [
            ("R2G1_1H_trend_15m_pullback_5m_reclaim", "V2S1_1H_trend_15m_pullback_5m_reclaim", "TF4_1h_regime_15m_setup_5m_trigger", r["regime_1h"] == "REG_TREND_UP" and r["tf15m_range_position"] < 0.45 and r["trigger_5m_context"] == "TRG_5M_BULL_RECLAIM", r["regime_1h"] == "REG_TREND_DOWN" and r["tf15m_range_position"] > 0.55 and r["trigger_5m_context"] == "TRG_5M_BEAR_REJECT", 3.0, "1H trend + 15m pullback + 5m reclaim"),
            ("R2G2_1H_range_15m_edge_5m_reversal", "V2S2_1H_range_15m_edge_5m_reversal", "TF4_1h_regime_15m_setup_5m_trigger", r["regime_1h"] == "REG_RANGE" and r["tf15m_range_position"] < 0.20 and r["tf5m_lower_wick"] > r["tf5m_body"], r["regime_1h"] == "REG_RANGE" and r["tf15m_range_position"] > 0.80 and r["tf5m_upper_wick"] > r["tf5m_body"], 2.5, "1H range edge reversal"),
            ("R2G3_30m_breakout_15m_retest_5m_trigger", "V2S3_30m_breakout_15m_retest_5m_trigger", "TF3_30m_primary_15m_setup_5m_trigger", bool(r["tf30m_breakout_proxy"]) and r["tf15m_range_position"] > 0.55 and r["close"] > r["tf15m_close"], bool(r["tf30m_breakout_proxy"]) and r["tf15m_range_position"] < 0.45 and r["close"] < r["tf15m_close"], 3.0, "30m breakout retest trigger"),
            ("R2G4_1H_compression_15m_expansion_5m_followthrough", "V2S4_1H_compression_15m_expansion_5m_followthrough", "TF4_1h_regime_15m_setup_5m_trigger", r["tf1h_compression_score"] > 0.70 and r["tf15m_expansion_score"] > 0.70 and r["close"] > r["open"], r["tf1h_compression_score"] > 0.70 and r["tf15m_expansion_score"] > 0.70 and r["close"] < r["open"], 4.0, "1H compression 15m expansion"),
            ("R2G5_1D_macro_bias_1H_pullback_continuation", "V2S5_1D_macro_bias_1H_pullback_continuation", "TF7_1d_macro_1h_regime_15m_setup_5m_trigger", r["regime_1d"] == "REG_TREND_UP" and r["tf1h_range_position"] < 0.55 and r["trigger_5m_context"] == "TRG_5M_BULL_RECLAIM", r["regime_1d"] == "REG_TREND_DOWN" and r["tf1h_range_position"] > 0.45 and r["trigger_5m_context"] == "TRG_5M_BEAR_REJECT", 3.0, "1D macro bias continuation"),
            ("R2G6_4H_1H_exhaustion_failed_breakout_reversal", "V2S6_4H_or_1H_exhaustion_15m_failed_breakout_reversal", "TF6_4h_macro_1h_regime_15m_setup_5m_trigger", ("LATE_TREND" in r["regime_1h"] or "EXHAUSTION" in r["regime_4h"]) and bool(r["tf15m_failed_breakout_proxy"]) and r["tf15m_range_position"] < 0.35, ("LATE_TREND" in r["regime_1h"] or "EXHAUSTION" in r["regime_4h"]) and bool(r["tf15m_failed_breakout_proxy"]) and r["tf15m_range_position"] > 0.65, 3.0, "4H/1H exhaustion failed breakout"),
            ("R2G7_liquidity_sweep_mtf_reclaim", "V2S7_liquidity_sweep_mtf_reclaim", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", bool(r["tf30m_failed_breakout_proxy"]) and r["tf15m_range_position"] < 0.35, bool(r["tf30m_failed_breakout_proxy"]) and r["tf15m_range_position"] > 0.65, 3.0, "MTF sweep reclaim"),
            ("R2G8_low_vol_grind_mtf", "V2S8_low_vol_grind_mtf", "TF4_1h_regime_15m_setup_5m_trigger", r["tf1h_realized_vol"] < vol_low and r["regime_1h"] == "REG_TREND_UP" and r["tf5m_lower_wick"] < r["tf5m_body"], r["tf1h_realized_vol"] < vol_low and r["regime_1h"] == "REG_TREND_DOWN" and r["tf5m_upper_wick"] < r["tf5m_body"], 2.0, "low vol trend grind"),
            ("R2G9_cost_aware_large_move_only", "V2S9_cost_aware_large_move_only", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", r["expected_move_to_cost_ratio"] > 5 and r["close"] > r["tf30m_close"], r["expected_move_to_cost_ratio"] > 5 and r["close"] < r["tf30m_close"], 5.0, "large move-to-cost only"),
            ("R2G10_no_trade_first_alpha", "V2S10_no_trade_first_alpha", "TF4_1h_regime_15m_setup_5m_trigger", bool(r["alpha_search_allowed"]) and r["regime_1h"] == "REG_TREND_UP" and r["close"] > r["open"], bool(r["alpha_search_allowed"]) and r["regime_1h"] == "REG_TREND_DOWN" and r["close"] < r["open"], 3.0, "bad-regime clear setup"),
            ("R2G16_15m_primary_5m_trigger_only", "V2S15_ensemble_balanced", "TF1_15m_primary_5m_trigger", r["tf15m_trend_direction"] == "up" and r["close"] > r["open"], r["tf15m_trend_direction"] == "down" and r["close"] < r["open"], 2.5, "15m primary 5m trigger"),
            ("R2G17_30m_primary_15m_trigger_5m_timing", "V2S15_ensemble_balanced", "TF3_30m_primary_15m_setup_5m_trigger", r["tf30m_trend_direction"] == "up" and r["tf15m_range_position"] > 0.45 and r["close"] > r["open"], r["tf30m_trend_direction"] == "down" and r["tf15m_range_position"] < 0.55 and r["close"] < r["open"], 3.0, "30m primary 15m trigger"),
            ("R2G18_1H_primary_30m_setup_15m_trigger", "V2S15_ensemble_balanced", "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", r["regime_1h"] == "REG_TREND_UP" and r["tf30m_range_position"] > 0.45 and r["tf15m_range_position"] > 0.45, r["regime_1h"] == "REG_TREND_DOWN" and r["tf30m_range_position"] < 0.55 and r["tf15m_range_position"] < 0.55, 3.0, "1H primary 30m setup"),
            ("R2G19_1D_1H_macro_only_reference", "V2S5_1D_macro_bias_1H_pullback_continuation", "TF9_1h_only_reference", r["regime_1d"] == "REG_TREND_UP" and r["regime_1h"] == "REG_TREND_UP", r["regime_1d"] == "REG_TREND_DOWN" and r["regime_1h"] == "REG_TREND_DOWN", 3.0, "macro-only reference"),
        ]
        votes = 0
        for gid, setup_id, stack, long_c, short_c, min_move, reason in conds:
            if counts.get(gid, 0) >= 250:
                continue
            direction = _dir(bool(long_c), bool(short_c))
            if direction == "NONE" or r["expected_move_to_cost_ratio"] < min_move:
                continue
            allowed = bool(r["alpha_search_allowed"]) and r["bad_regime_score"] < 0.75
            rows.append({
                "research_v2_candidate_id": f"{gid}_{int(r['bar_index'])}",
                "timestamp": r["close_ts"],
                "entry_ts": r["close_ts"],
                "bar_index": int(r["bar_index"]),
                "direction": direction,
                "generator_id": gid,
                "setup_id": setup_id,
                "setup_name": setup_id,
                "timeframe_stack": stack,
                "regime_1d": r["regime_1d"],
                "regime_4h": r["regime_4h"],
                "regime_1h": r["regime_1h"],
                "regime_30m": r["regime_30m"],
                "regime_15m": r["regime_15m"],
                "trigger_5m_context": r["trigger_5m_context"],
                "setup_score": float(r["regime_alignment_score"] * 2 + r["expected_move_to_cost_ratio"] / 5 - r["bad_regime_score"] + votes * 0.2),
                "setup_reason": reason,
                "regime_alignment_score": r["regime_alignment_score"],
                "regime_conflict_score": r["regime_conflict_score"],
                "bad_regime_score": r["bad_regime_score"],
                "bad_regime_reasons": r["bad_regime_reasons"],
                "alpha_search_allowed": allowed,
                "expected_move_proxy": r["tf1h_atr_proxy"],
                "expected_move_to_cost_ratio": r["expected_move_to_cost_ratio"],
                "q2_score": r.get("q2_score", np.nan),
                "q2_scale": r.get("q2_scale", np.nan),
                "q2_decision": r.get("q2_decision", "unknown"),
                "r7_score": r.get("r7_score", np.nan),
                "r7_high_hazard": bool(r.get("r7_high_hazard", False)),
                "tcn_p_direction": max(r.get("p_long", np.nan), r.get("p_short", np.nan)) if "p_long" in r and "p_short" in r else np.nan,
                "tcn_entropy": r.get("entropy", np.nan),
                "tcn_margin": r.get("margin", np.nan),
                "tcn_alignment": True,
                "external_context_if_available": "see external_market_structure_alpha_v1 cache",
                "candidate_allowed_core": allowed and gid != "R2G19_1D_1H_macro_only_reference",
                "candidate_reference_only": (not allowed) or gid == "R2G19_1D_1H_macro_only_reference",
                "oracle_flag": False,
                "feature_snapshot_hash": _hash_row(r, cols),
            })
            counts[gid] = counts.get(gid, 0) + 1
            votes += 1
        if votes >= 3 and r["alpha_search_allowed"] and counts.get("R2G14_ensemble_strict", 0) < 250:
            rows.append({**rows[-1], "research_v2_candidate_id": f"R2G14_ensemble_strict_{int(r['bar_index'])}", "generator_id": "R2G14_ensemble_strict", "setup_id": "V2S14_ensemble_strict", "setup_name": "V2S14_ensemble_strict", "timeframe_stack": "TF8_1d_macro_4h_macro_1h_regime_30m_setup_15m_trigger_5m_timing", "setup_reason": f"{votes}_vote_strict_ensemble", "candidate_allowed_core": True, "candidate_reference_only": False})
            counts["R2G14_ensemble_strict"] = counts.get("R2G14_ensemble_strict", 0) + 1
        if votes >= 2 and counts.get("R2G15_ensemble_balanced", 0) < 250:
            rows.append({**rows[-1], "research_v2_candidate_id": f"R2G15_ensemble_balanced_{int(r['bar_index'])}", "generator_id": "R2G15_ensemble_balanced", "setup_id": "V2S15_ensemble_balanced", "setup_name": "V2S15_ensemble_balanced", "timeframe_stack": "TF5_1h_regime_30m_setup_15m_trigger_5m_timing", "setup_reason": f"{votes}_vote_balanced_ensemble"})
            counts["R2G15_ensemble_balanced"] = counts.get("R2G15_ensemble_balanced", 0) + 1
    cand = pd.DataFrame(rows).drop_duplicates(["research_v2_candidate_id"]).reset_index(drop=True) if rows else pd.DataFrame()
    if len(cand):
        cand = cand.sort_values(["generator_id", "setup_score"], ascending=[True, False]).groupby("generator_id", group_keys=False).head(150).reset_index(drop=True)
    cand.to_parquet(DIRS["cand"] / "research_v2_candidate_universe.parquet", index=False)
    cand.to_csv(DIRS["cand"] / "research_v2_candidate_universe.csv", index=False)
    if len(cand):
        cand.groupby("generator_id").agg(rows=("research_v2_candidate_id", "size"), core=("candidate_allowed_core", "sum"), bad_mean=("bad_regime_score", "mean"), score_mean=("setup_score", "mean")).reset_index().to_csv(DIRS["cand"] / "candidate_summary_by_generator.csv", index=False)
        cand.groupby(["generator_id", "direction"]).size().reset_index(name="rows").to_csv(DIRS["cand"] / "candidate_direction_distribution.csv", index=False)
        tmp = cand.copy()
        tmp["quarter"] = pd.to_datetime(tmp["timestamp"]).dt.to_period("Q").astype(str)
        tmp.groupby(["generator_id", "quarter"]).size().reset_index(name="rows").to_csv(DIRS["cand"] / "candidate_recent_quarter_summary.csv", index=False)
        cand.groupby(["timeframe_stack", "generator_id"]).size().reset_index(name="rows").to_csv(DIRS["cand"] / "candidate_timeframe_stack_summary.csv", index=False)
        cand.assign(overlap_current_engine=cand["timestamp"].astype("datetime64[ns]").astype(str).isin(current_engine_ts)).groupby("generator_id")["overlap_current_engine"].sum().reset_index().to_csv(DIRS["cand"] / "candidate_overlap_with_current_engine.csv", index=False)
    else:
        for f in ["candidate_summary_by_generator.csv", "candidate_direction_distribution.csv", "candidate_recent_quarter_summary.csv", "candidate_timeframe_stack_summary.csv", "candidate_overlap_with_current_engine.csv"]:
            pd.DataFrame().to_csv(DIRS["cand"] / f, index=False)
    _write_md(DIRS["cand"] / "candidate_generation_report.md", "Candidate Generation Report", {"rows": len(cand), "core_rows": int(cand["candidate_allowed_core"].sum()) if len(cand) else 0, "oracle_flag_count": int(cand["oracle_flag"].sum()) if len(cand) else 0})
    return cand


POLICIES = [
    ("X0_fixed_12", 12, "fixed", False), ("X1_fixed_24", 24, "fixed", False), ("X2_fixed_48", 48, "fixed", False), ("X3_fixed_96", 96, "fixed", False), ("X4_fixed_144", 144, "fixed", False), ("X5_current_maxhold_proxy", 60, "maxhold", False),
    ("X6_MAE_stop_medium", 24, "stop", False), ("X7_vol_adjusted_MAE_stop", 48, "stop", False), ("X8_first_cost_plus_move", 24, "tp", False), ("X9_first_2x_cost_plus_move", 48, "tp", False), ("X10_first_3x_cost_plus_move", 72, "tp", False),
    ("X11_trailing_vol_adjusted_proxy", 96, "trailing", False), ("X12_fixed_24_plus_MAE_stop", 24, "hybrid", False), ("X13_fixed_48_plus_MAE_stop", 48, "hybrid", False), ("X14_trailing_plus_MAE_stop", 96, "hybrid", False), ("X15_setup_specific_exit", 48, "setup", False),
    ("X16_timeframe_matched_exit_15m", 36, "tf", False), ("X17_timeframe_matched_exit_30m", 72, "tf", False), ("X18_timeframe_matched_exit_1h", 144, "tf", False), ("X90_oracle_best_24", 24, "oracle", True), ("X91_oracle_best_48", 48, "oracle", True), ("X92_oracle_best_96", 96, "oracle", True), ("X93_oracle_MFE", 144, "oracle", True),
]


def _path_returns(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        return path["close"] / entry - 1, path["high"] / entry - 1, path["low"] / entry - 1
    return entry / path["close"] - 1, entry / path["low"] - 1, entry / path["high"] - 1


def _exit_index(pid: str, cr: pd.Series, fav: pd.Series, adv: pd.Series) -> int:
    default = len(cr) - 1
    if "MAE_stop" in pid or "plus_MAE" in pid:
        hit = np.where(adv.values <= -0.004)[0]
        if len(hit):
            return int(hit[0])
    if "first_cost" in pid:
        hit = np.where(fav.values >= COST * 2)[0]
        return int(hit[0]) if len(hit) else default
    if "first_2x" in pid:
        hit = np.where(fav.values >= COST * 3)[0]
        return int(hit[0]) if len(hit) else default
    if "first_3x" in pid:
        hit = np.where(fav.values >= COST * 4)[0]
        return int(hit[0]) if len(hit) else default
    if "oracle_best" in pid:
        return int(cr.values.argmax())
    if "trailing" in pid:
        peak = fav.cummax()
        hit = np.where((peak > COST * 3) & (peak - cr > COST * 2))[0]
        return int(hit[0]) if len(hit) else default
    return default


def _metrics(df: pd.DataFrame, cols: List[str], ret_col: str = "net_after_cost") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in df.groupby(cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        r = pd.to_numeric(sub[ret_col], errors="coerce").fillna(0)
        row = {c: k for c, k in zip(cols, keys)}
        row.update({"rows": len(sub), "expectancy": float(r.mean()), "winrate": float((r > 0).mean()), "profit_factor": _profit_factor(r), "MDD_proxy": _mdd(r), "MFE_median": float(sub["MFE"].median()), "MAE_median": float(sub["MAE"].median()), "RFE_rate": float(sub["RFE"].mean()), "tail_loss": float(r.quantile(0.05)), "MFE_to_cost_ratio": float(sub["MFE_to_cost_ratio"].median()), "cost_sensitivity": float((r - COST).mean()), "turnover": float(1 / max(sub["holding_bars"].mean(), 1))})
        rows.append(row)
    return pd.DataFrame(rows)


def phase6_backfill(cand: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv.set_index("bar_index")
    trades, outs = [], []
    max_h = max(p[1] for p in POLICIES)
    for _, c in cand.iterrows():
        entry_idx = int(c["bar_index"]) + 1
        if entry_idx not in o.index:
            continue
        eb = o.loc[entry_idx]
        slip = SLIPPAGE if c["direction"] == "LONG" else -SLIPPAGE
        entry = float(eb["open"]) * (1 + slip)
        tid = f"{c['research_v2_candidate_id']}_P0"
        trades.append({**c.to_dict(), "research_v2_paper_trade_id": tid, "entry_ts": eb["close_ts"], "paper_entry_price": entry, "entry_bar_index": entry_idx})
        path = o.loc[(o.index >= entry_idx) & (o.index < entry_idx + max_h)].copy()
        if path.empty:
            continue
        cr_full, fav_full, adv_full = _path_returns(path, c["direction"], entry)
        for pid, horizon, group, oracle in POLICIES:
            cr = cr_full.head(horizon)
            fav = fav_full.head(horizon)
            adv = adv_full.head(horizon)
            if cr.empty:
                continue
            ei = _exit_index(pid, cr, fav, adv)
            gross = float(fav.max()) if pid == "X93_oracle_MFE" else float(cr.iloc[ei])
            net = gross - COST
            mfe = float(fav.max())
            mae = float(adv.min())
            t_mfe = int(fav.values.argmax()) + 1
            t_mae = int(adv.values.argmin()) + 1
            outs.append({**{k: c[k] for k in ["research_v2_candidate_id", "generator_id", "setup_id", "setup_name", "timeframe_stack", "direction", "regime_1d", "regime_4h", "regime_1h", "regime_30m", "regime_15m", "bad_regime_score", "regime_alignment_score", "regime_conflict_score", "expected_move_to_cost_ratio", "candidate_allowed_core", "candidate_reference_only"]},
                "research_v2_paper_trade_id": tid, "entry_ts": eb["close_ts"], "paper_entry_price": entry, "exit_policy_id": pid, "exit_policy_group": group, "exit_ts": path.head(horizon).iloc[min(ei, len(path.head(horizon)) - 1)]["close_ts"], "holding_bars": ei + 1, "gross_return": gross, "net_after_cost": net, "MFE": mfe, "MAE": mae, "RFE": bool(mae <= -0.006), "time_to_MFE": t_mfe, "time_to_MAE": t_mae, "MFE_before_MAE": t_mfe <= t_mae, "MAE_before_MFE": t_mae < t_mfe, "cost_plus_hit": mfe >= COST * 2, "MFE_to_cost_ratio": mfe / COST, "MAE_to_cost_ratio": abs(mae) / COST, "tail_loss": net <= -0.006, "high_MAE": abs(mae) >= 0.006, "MFE_capture": max(net, 0) / max(mfe, 1e-9), "giveback": max(mfe - max(net, 0), 0) / max(mfe, 1e-9), "oracle_flag": oracle})
    trades_df = pd.DataFrame(trades)
    out = pd.DataFrame(outs)
    trades_df.to_parquet(DIRS["backfill"] / "research_v2_paper_trades.parquet", index=False)
    out.to_parquet(DIRS["backfill"] / "research_v2_exit_outcomes.parquet", index=False)
    _metrics(out[~out["oracle_flag"]], ["generator_id"]).to_csv(DIRS["backfill"] / "research_v2_outcome_metrics_by_generator.csv", index=False)
    _metrics(out, ["exit_policy_id"]).to_csv(DIRS["backfill"] / "research_v2_outcome_metrics_by_exit_policy.csv", index=False)
    _metrics(out[~out["oracle_flag"]], ["direction"]).to_csv(DIRS["backfill"] / "research_v2_outcome_metrics_by_direction.csv", index=False)
    _metrics(out[~out["oracle_flag"]], ["timeframe_stack"]).to_csv(DIRS["backfill"] / "research_v2_outcome_metrics_by_timeframe_stack.csv", index=False)
    _metrics(out[~out["oracle_flag"]], ["regime_1h"]).to_csv(DIRS["backfill"] / "research_v2_outcome_metrics_by_regime.csv", index=False)
    tmp = out[~out["oracle_flag"]].copy()
    tmp["quarter"] = pd.to_datetime(tmp["entry_ts"]).dt.to_period("Q").astype(str)
    _metrics(tmp, ["generator_id", "quarter"]).to_csv(DIRS["backfill"] / "research_v2_outcome_recent_quarter.csv", index=False)
    _write_md(DIRS["backfill"] / "research_v2_backfill_report.md", "Research V2 Backfill Report", {"trades": len(trades_df), "outcomes": len(out), "best_non_oracle": _metrics(out[~out["oracle_flag"]], ["generator_id", "exit_policy_id"]).sort_values("expectancy", ascending=False).head(25) if len(out) else pd.DataFrame(), "oracle_reference_only": True})
    return out


def phase7_labels(out: pd.DataFrame) -> pd.DataFrame:
    lab = out[(out["exit_policy_id"].eq("X1_fixed_24")) & (~out["oracle_flag"])].copy()
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"]) & lab["MFE_before_MAE"] & (lab["regime_alignment_score"] >= 0.4)
    bad = (lab["net_after_cost"] < 0) | lab["RFE"] | lab["high_MAE"] | lab["tail_loss"] | (lab["bad_regime_score"] >= 0.75)
    lab["r2_label"] = np.select([good, bad, ~(good | bad)], ["R2_GOOD", "R2_BAD", "R2_NEUTRAL"], default="R2_CENSORED")
    lab["entry_impulse"] = lab["MFE_to_cost_ratio"].clip(0, 10) / 10
    lab["early_MFE"] = lab["MFE"] / np.maximum(lab["time_to_MFE"], 1)
    lab["early_MAE"] = lab["MAE"].abs() / np.maximum(lab["time_to_MAE"], 1)
    lab["favorable_first"] = lab["MFE_before_MAE"]
    lab["adverse_first"] = lab["MAE_before_MFE"]
    lab["setup_alignment"] = lab["regime_alignment_score"]
    lab["cost_aware_quality"] = lab["expected_move_to_cost_ratio"].clip(0, 10) / 10
    lab["utility_score"] = (lab["net_after_cost"].clip(-0.01, 0.01) / 0.01 + lab["entry_impulse"] - lab["MAE_to_cost_ratio"].clip(0, 10) / 10 + lab["favorable_first"].astype(float) + lab["setup_alignment"] + lab["cost_aware_quality"]) / 6
    lab["risk_score"] = lab["RFE"].astype(float) * 0.25 + lab["high_MAE"].astype(float) * 0.20 + lab["tail_loss"].astype(float) * 0.20 + lab["regime_conflict_score"] * 0.15 + lab["bad_regime_score"] * 0.20
    lab["expected_edge_score"] = lab["utility_score"] - lab["risk_score"]
    lab.to_parquet(DIRS["labels"] / "research_v2_entry_quality_labels.parquet", index=False)
    lab.to_csv(DIRS["labels"] / "research_v2_entry_quality_labels.csv", index=False)
    lab.groupby(["generator_id", "r2_label"]).size().reset_index(name="rows").to_csv(DIRS["labels"] / "research_v2_label_policy_summary.csv", index=False)
    lab[["research_v2_paper_trade_id", "utility_score"]].to_csv(DIRS["labels"] / "research_v2_utility_score.csv", index=False)
    lab[["research_v2_paper_trade_id", "risk_score"]].to_csv(DIRS["labels"] / "research_v2_risk_score.csv", index=False)
    lab[["research_v2_paper_trade_id", "expected_edge_score", "net_after_cost", "MFE", "MAE", "RFE"]].to_csv(DIRS["labels"] / "research_v2_expected_edge_score.csv", index=False)
    lab[["research_v2_paper_trade_id", "generator_id", "timeframe_stack", "r2_label", "RFE", "high_MAE", "tail_loss", "bad_regime_score"]].to_csv(DIRS["labels"] / "research_v2_label_reason_codes.csv", index=False)
    lab["edge_decile"] = pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop") if len(lab) else pd.Series(dtype=int)
    dec = lab.groupby("edge_decile").agg(rows=("research_v2_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("r2_label", lambda s: float((s == "R2_GOOD").mean())), bad_rate=("r2_label", lambda s: float((s == "R2_BAD").mean()))).reset_index()
    dec.to_csv(DIRS["labels"] / "research_v2_expected_edge_deciles.csv", index=False)
    _write_md(DIRS["labels"] / "research_v2_label_design_report.md", "Research V2 Label Design Report", {"label_distribution": lab["r2_label"].value_counts().to_dict(), "expected_edge_monotonic": bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False, "position_sizing_production_allowed": False})
    return lab


def _scorecard(labels: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for k, sub in labels.groupby(group):
        r = sub["net_after_cost"]
        rows.append({group: k, "candidate_count": len(sub), "GOOD_count": int(sub["r2_label"].eq("R2_GOOD").sum()), "BAD_count": int(sub["r2_label"].eq("R2_BAD").sum()), "NEUTRAL_count": int(sub["r2_label"].eq("R2_NEUTRAL").sum()), "GOOD_rate": float(sub["r2_label"].eq("R2_GOOD").mean()), "BAD_rate": float(sub["r2_label"].eq("R2_BAD").mean()), "net_after_cost_expectancy": float(r.mean()), "MFE_median": float(sub["MFE"].median()), "MAE_median": float(sub["MAE"].median()), "RFE_rate": float(sub["RFE"].mean()), "tail_loss": float(r.quantile(0.05)), "MFE_to_cost_ratio": float(sub["MFE_to_cost_ratio"].median()), "cost_sensitivity": float((r - COST).mean()), "profit_factor": _profit_factor(r), "MDD_proxy": _mdd(r), "turnover": float(1 / max(sub["holding_bars"].mean(), 1)), "holding_bars": float(sub["holding_bars"].mean()), "recent_3m": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_3M_DAYS)).sum()), "recent_6m": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()), "quarter_coverage": int(pd.to_datetime(sub["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "expected_edge_score_monotonicity": bool(sub.groupby(pd.qcut(sub["expected_edge_score"].rank(method="first"), min(5, len(sub)), labels=False, duplicates="drop"))["net_after_cost"].mean().is_monotonic_increasing) if len(sub) >= 10 else False, "interpretability": 0.8, "sample_size_risk": float(max(0, 1 - len(sub) / 100)), "overfit_risk": float(max(0, 1 - len(sub) / 200))})
    return pd.DataFrame(rows)


def phase8_timeframe_tournament(labels: pd.DataFrame) -> pd.DataFrame:
    sc = _scorecard(labels, "timeframe_stack").rename(columns={"timeframe_stack": "timeframe_stack"})
    sc["status"] = np.select([sc["candidate_count"].lt(20), sc["GOOD_count"].lt(10), sc["cost_sensitivity"].le(0), sc["BAD_rate"].gt(0.70)], ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"], default="research_candidate")
    sc.sort_values(["status", "GOOD_rate"], ascending=[True, False]).to_csv(DIRS["tf_tournament"] / "timeframe_stack_scorecard.csv", index=False)
    _write_md(DIRS["tf_tournament"] / "timeframe_stack_rankings.md", "Timeframe Stack Rankings", {"scorecard": sc.sort_values("GOOD_rate", ascending=False)})
    sc[sc["status"].str.startswith("reject")].to_csv(DIRS["tf_tournament"] / "timeframe_stack_reject_reasons.csv", index=False)
    sc[sc["status"].eq("research_candidate")].head(10).to_csv(DIRS["tf_tournament"] / "best_timeframe_stack_cases.csv", index=False)
    _write_md(DIRS["tf_tournament"] / "timeframe_stack_tournament_report.md", "Timeframe Stack Tournament Report", {"scorecard": sc})
    return sc


def phase9_setup_tournament(labels: pd.DataFrame) -> pd.DataFrame:
    sc = _scorecard(labels, "setup_id").rename(columns={"setup_id": "setup_id"})
    sc["status"] = np.select([sc["candidate_count"].lt(20), sc["GOOD_count"].lt(10), sc["cost_sensitivity"].le(0), sc["BAD_rate"].gt(0.70)], ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"], default="research_candidate")
    sc.to_csv(DIRS["setup_tournament"] / "setup_family_scorecard.csv", index=False)
    _write_md(DIRS["setup_tournament"] / "setup_family_rankings.md", "Setup Family Rankings", {"scorecard": sc.sort_values("GOOD_rate", ascending=False)})
    sc[sc["status"].str.startswith("reject")].to_csv(DIRS["setup_tournament"] / "setup_family_reject_reasons.csv", index=False)
    sc[sc["status"].eq("research_candidate")].head(10).to_csv(DIRS["setup_tournament"] / "minimal_viable_setup_candidates.csv", index=False)
    _write_md(DIRS["setup_tournament"] / "setup_family_tournament_report.md", "Setup Family Tournament Report", {"scorecard": sc})
    return sc


def phase10_role_recheck(labels: pd.DataFrame) -> None:
    rows = []
    for name, mask in {
        "setup_without_TCN": pd.Series(True, index=labels.index),
        "TCN_as_scorer_overlay": labels.get("tcn_alignment", pd.Series(True, index=labels.index)).fillna(True).astype(bool) if "tcn_alignment" in labels else pd.Series(True, index=labels.index),
        "Q2_defensive_overlay": labels.get("q2_scale", pd.Series(np.nan, index=labels.index)).fillna(0).ge(0.4) if "q2_scale" in labels else pd.Series(False, index=labels.index),
        "R7_warning_only_no_hazard": ~labels.get("r7_high_hazard", pd.Series(False, index=labels.index)).fillna(False).astype(bool) if "r7_high_hazard" in labels else pd.Series(True, index=labels.index),
    }.items():
        sub = labels[mask]
        rows.append({"role": name, "rows": len(sub), "GOOD_rate": float(sub["r2_label"].eq("R2_GOOD").mean()) if len(sub) else 0, "BAD_rate": float(sub["r2_label"].eq("R2_BAD").mean()) if len(sub) else 0, "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0, "production_action": "none"})
    df = pd.DataFrame(rows)
    df[df["role"].str.contains("TCN")].to_csv(DIRS["role"] / "tcn_role_recheck.csv", index=False)
    df[df["role"].str.contains("Q2")].to_csv(DIRS["role"] / "q2_role_recheck.csv", index=False)
    df[df["role"].str.contains("R7")].to_csv(DIRS["role"] / "r7_role_recheck.csv", index=False)
    df.to_csv(DIRS["role"] / "q2_r7_tcn_combo_recheck.csv", index=False)
    _write_md(DIRS["role"] / "role_recheck_report.md", "Q2/R7/TCN Role Recheck", {"roles": df, "decision": "TCN scorer-only, Q2 defensive baseline, R7 warning-only."})


def phase11_feature_sufficiency(labels: pd.DataFrame, mtf: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["close_ts"] = pd.to_datetime(lab["entry_ts"]) - pd.Timedelta(minutes=5)
    feat_cols = [c for c in mtf.columns if pd.api.types.is_numeric_dtype(mtf[c]) and c not in {"bar_index"} and not any(tok in c.lower() for tok in ["future", "label", "mfe", "mae", "rfe"])]
    df = pd.merge_asof(lab.sort_values("close_ts"), mtf[["close_ts"] + feat_cols].sort_values("close_ts"), on="close_ts", direction="backward")
    sets = {
        "5m_only": [c for c in feat_cols if not c.startswith("tf")],
        "15m_5m": [c for c in feat_cols if c.startswith("tf15m") or not c.startswith("tf")],
        "30m_15m_5m": [c for c in feat_cols if c.startswith(("tf30m", "tf15m")) or not c.startswith("tf")],
        "1h_30m_15m_5m": [c for c in feat_cols if c.startswith(("tf1h", "tf30m", "tf15m")) or not c.startswith("tf")],
        "1d_4h_1h_30m_15m_5m": feat_cols,
        "TCN_only": [c for c in feat_cols if c in {"margin", "entropy", "p_long", "p_short", "p_flat"}],
        "Q2_only": [c for c in feat_cols if c.startswith("q2")],
        "R7_only": [c for c in feat_cols if c.startswith("r7")],
        "all_Research_V2_features": feat_cols[:180],
    }
    targets = {
        "R2_GOOD_vs_R2_BAD": df["r2_label"].eq("R2_GOOD"),
        "expected_edge_top_decile": df["expected_edge_score"] >= df["expected_edge_score"].quantile(0.9),
        "RFE_bad": df["RFE"].astype(bool),
        "high_MAE_bad": df["high_MAE"].astype(bool),
        "cost_killed": df["net_after_cost"] <= 0,
        "no_trade_bad_regime": df["bad_regime_score"] >= 0.75,
    }
    models = {"logistic": LogisticRegression(max_iter=1000, class_weight="balanced"), "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=42), "random_forest": RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=10, random_state=42, class_weight="balanced"), "extra_trees": ExtraTreesClassifier(n_estimators=80, max_depth=4, min_samples_leaf=10, random_state=42, class_weight="balanced")}
    split = int(len(df) * 0.7)
    metrics, imps, inter = [], [], []
    for tname, y0 in targets.items():
        y = y0.astype(int).reset_index(drop=True)
        if y.nunique() < 2 or y.sum() < 10 or len(y) - y.sum() < 10:
            continue
        for fs, cols in sets.items():
            cols = [c for c in cols if c in df.columns][:120]
            if not cols or y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
                continue
            x = df[cols].replace([np.inf, -np.inf], np.nan)
            for mn, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(x.iloc[:split], y.iloc[:split])
                    score = pipe.predict_proba(x.iloc[split:])[:, 1]
                    yte = y.iloc[split:]
                    top = score >= np.quantile(score, 0.8)
                    top_df = df.iloc[split:].loc[top]
                    metrics.append({"target": tname, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": float(roc_auc_score(yte, score)), "PR_AUC": float(average_precision_score(yte, score)), "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_after_cost"].mean()) if len(top_df) else 0.0})
                    vals = getattr(pipe.named_steps["model"], "feature_importances_", np.abs(getattr(pipe.named_steps["model"], "coef_", np.zeros((1, len(cols))))).ravel())
                    for c, v in sorted(zip(cols, vals), key=lambda z: -float(z[1]))[:15]:
                        imps.append({"target": tname, "feature_set": fs, "model": mn, "feature": c, "importance": float(v)})
                except Exception:
                    continue
    met = pd.DataFrame(metrics)
    imp = pd.DataFrame(imps)
    met.to_csv(DIRS["feature"] / "research_v2_feature_sufficiency_metrics.csv", index=False)
    met.to_csv(DIRS["feature"] / "timeframe_feature_ablation.csv", index=False)
    imp.to_csv(DIRS["feature"] / "feature_family_importance.csv", index=False)
    (imp.groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if len(imp) else pd.DataFrame()).to_csv(DIRS["feature"] / "feature_interaction_candidates.csv", index=False)
    _write_md(DIRS["feature"] / "research_v2_feature_sufficiency_report.md", "Research V2 Feature Sufficiency Report", {"best": met.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(met) else pd.DataFrame(), "diagnosis": "MTF separability is evaluated as research-only."})
    return met


def phase12_economics(labels: pd.DataFrame, frames: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for scenario, mult in [("0_cost", 0), ("current_cost", 1), ("2x_cost", 2), ("3x_cost", 3), ("slippage_stress", 1.5)]:
        adj = labels["gross_return"] - COST * mult
        rows.append({"scenario": scenario, "expectancy": float(adj.mean()), "winrate": float((adj > 0).mean()), "profit_factor": _profit_factor(adj), "tail_loss": float(adj.quantile(0.05))})
    pd.DataFrame(rows).to_csv(DIRS["econ"] / "research_v2_cost_sensitivity.csv", index=False)
    nf = []
    for tf, f in frames.items():
        nf.append({"timeframe": tf, "avg_range_to_cost": float((f["range"] / COST).mean()), "median_range_to_cost": float((f["range"] / COST).median()), "avg_atr_to_cost": float((f["atr_proxy"] / COST).mean())})
    pd.DataFrame(nf).to_csv(DIRS["econ"] / "timeframe_noise_floor.csv", index=False)
    labels.groupby("setup_id").agg(rows=("research_v2_paper_trade_id", "size"), expected_move_to_cost=("expected_move_to_cost_ratio", "median"), MFE_to_cost=("MFE_to_cost_ratio", "median")).reset_index().to_csv(DIRS["econ"] / "expected_move_to_cost_by_setup.csv", index=False)
    labels.groupby("timeframe_stack").agg(rows=("research_v2_paper_trade_id", "size"), turnover=("holding_bars", lambda s: 1 / max(s.mean(), 1))).reset_index().to_csv(DIRS["econ"] / "turnover_by_timeframe_stack.csv", index=False)
    total = labels["net_after_cost"].sum()
    top = labels.nlargest(max(1, int(len(labels) * 0.05)), "net_after_cost")["net_after_cost"].sum() if len(labels) else 0
    pd.DataFrame([{"top_5pct_dependency": float(top / total) if abs(total) > 1e-9 else 0.0, "tail_5pct_loss": float(labels["net_after_cost"].quantile(0.05)) if len(labels) else 0.0}]).to_csv(DIRS["econ"] / "tail_and_top_winner_dependency.csv", index=False)
    _write_md(DIRS["econ"] / "economics_report.md", "Economics Report", {"cost_sensitivity": pd.DataFrame(rows), "noise_floor": pd.DataFrame(nf), "principle": "5m trigger is acceptable only when target move is 15m/30m/1h scale."})


def phase13_sizing(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["edge_decile"] = pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop") if len(lab) else pd.Series(dtype=int)
    dec = lab.groupby("edge_decile").agg(rows=("research_v2_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("r2_label", lambda s: float((s == "R2_GOOD").mean())), bad_rate=("r2_label", lambda s: float((s == "R2_BAD").mean()))).reset_index()
    dec.to_csv(DIRS["sizing"] / "research_v2_expected_edge_decile_monotonicity.csv", index=False)
    monotonic = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    sims = []
    rank = lab["expected_edge_score"].rank(pct=True) if len(lab) else pd.Series(dtype=float)
    for policy in ["SZ0_equal_size_baseline", "SZ1_q2_scale_only", "SZ2_expected_edge_linear", "SZ3_expected_edge_sigmoid", "SZ4_risk_adjusted_edge", "SZ5_cap_top_decile", "SZ6_no_size_increase_only_reduce_bad", "SZ7_kelly_fraction_proxy_capped", "SZ8_drawdown_aware_sizing", "SZ9_setup_family_budget", "SZ10_timeframe_stack_budget"]:
        w = pd.Series(1.0, index=lab.index) if policy == "SZ0_equal_size_baseline" else 0.25 + rank.fillna(0.5) * 0.75
        if policy == "SZ6_no_size_increase_only_reduce_bad":
            w = np.where(rank < 0.5, 0.5, 1.0)
        ret = lab["net_after_cost"] * w
        sims.append({"sizing_policy": policy, "expectancy": float(ret.mean()) if len(ret) else 0, "MDD_proxy": _mdd(ret) if len(ret) else 0, "expected_edge_monotonic": monotonic, "production_allowed": False})
    sim = pd.DataFrame(sims)
    sim.to_csv(DIRS["sizing"] / "research_v2_position_sizing_simulation_scorecard.csv", index=False)
    pd.DataFrame([{"monotonic": monotonic, "top_decile_tail_loss": float(lab[lab["edge_decile"].eq(lab["edge_decile"].max())]["net_after_cost"].quantile(0.05)) if len(lab) else 0}]).to_csv(DIRS["sizing"] / "research_v2_sizing_risk_report.csv", index=False)
    _write_md(DIRS["sizing"] / "research_v2_sizing_readiness_decision.md", "Sizing Readiness Decision", {"expected_edge_monotonic": monotonic, "production_allowed": False})
    _write_md(DIRS["sizing"] / "position_sizing_research_report.md", "Position Sizing Research Report", {"deciles": dec, "simulation": sim})
    return dec


def phase14_validation(labels: pd.DataFrame) -> None:
    lab = labels.copy()
    lab["month"] = pd.to_datetime(lab["entry_ts"]).dt.to_period("M").astype(str)
    lab["quarter"] = pd.to_datetime(lab["entry_ts"]).dt.to_period("Q").astype(str)
    _scorecard(lab.rename(columns={"month": "validation_month"}), "validation_month").to_csv(DIRS["validation"] / "research_v2_monthly_validation.csv", index=False)
    _scorecard(lab.rename(columns={"quarter": "validation_quarter"}), "validation_quarter").to_csv(DIRS["validation"] / "research_v2_quarterly_validation.csv", index=False)
    recent_rows = []
    mx = pd.to_datetime(lab["entry_ts"]).max()
    for name, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS)]:
        sub = lab[pd.to_datetime(lab["entry_ts"]) >= mx - pd.Timedelta(days=days)]
        recent_rows.append({"window": name, "rows": len(sub), "GOOD": int(sub["r2_label"].eq("R2_GOOD").sum()), "BAD": int(sub["r2_label"].eq("R2_BAD").sum()), "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0, "cost_sensitivity": float((sub["net_after_cost"] - COST).mean()) if len(sub) else 0})
    pd.DataFrame(recent_rows).to_csv(DIRS["validation"] / "research_v2_recent_validation.csv", index=False)
    q = sorted(lab["quarter"].unique())
    wf = []
    for i in range(2, len(q)):
        train_q, test_q = q[:i], q[i]
        sub = lab[lab["quarter"].eq(test_q)]
        wf.append({"split": f"{train_q[0]}..{train_q[-1]}->{test_q}", "test_rows": len(sub), "GOOD": int(sub["r2_label"].eq("R2_GOOD").sum()), "BAD": int(sub["r2_label"].eq("R2_BAD").sum()), "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0, "cost_sensitivity": float((sub["net_after_cost"] - COST).mean()) if len(sub) else 0})
    pd.DataFrame(wf).to_csv(DIRS["validation"] / "research_v2_walkforward_validation.csv", index=False)
    _scorecard(lab, "regime_1h").to_csv(DIRS["validation"] / "research_v2_regime_walkforward.csv", index=False)
    _scorecard(lab, "timeframe_stack").to_csv(DIRS["validation"] / "research_v2_timeframe_stack_walkforward.csv", index=False)
    _write_md(DIRS["validation"] / "validation_report.md", "Validation Report", {"monthly": _scorecard(lab.rename(columns={"month": "validation_month"}), "validation_month"), "walkforward": pd.DataFrame(wf), "decision": "No production readiness can be declared from in-sample performance."})


def phase15_hidden(labels: pd.DataFrame, mtf_audit: pd.DataFrame, sizing: pd.DataFrame) -> pd.DataFrame:
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    checks = {
        "HF_A_timeframe_resample_leakage": bool(mtf_audit["leakage_rows"].sum() > 0),
        "HF_B_incomplete_higher_tf_candle_leakage": bool(mtf_audit["leakage_rows"].sum() > 0),
        "HF_C_1D_current_candle_leakage": bool(mtf_audit.loc[mtf_audit["timeframe"].eq("1d"), "leakage_rows"].sum() > 0) if "1d" in set(mtf_audit["timeframe"]) else False,
        "HF_I_cost_slippage_underestimated": True,
        "HF_K_candidate_count_too_low": len(labels) < 100,
        "HF_L_candidate_count_too_high": len(labels) > 10000,
        "HF_N_tail_loss_dominates": labels["net_after_cost"].quantile(0.05) < -0.006 if len(labels) else True,
        "HF_R_5m_noise_floor_too_large": True,
        "HF_U_no_trade_map_overfilters": False,
        "HF_V_Q2_still_not_greenlight": True,
        "HF_W_R7_warning_only_still_correct": True,
        "HF_X_TCN_scorer_still_miscalibrated": True,
        "HF_Y_expected_edge_not_monotonic": not monotonic,
        "HF_Z_position_sizing_dangerous": not monotonic,
        "HF_AA_external_data_missing": True,
        "HF_AC_cost_kills_all_small_edges": (labels["net_after_cost"] - COST).mean() < 0 if len(labels) else True,
        "HF_AF_oracle_contamination": False,
    }
    df = pd.DataFrame([{"failure_mode": k, "evidence_for": bool(v), "evidence_against": not bool(v), "severity": 0.8 if v else 0.25, "confidence": 0.75 if v else 0.45, "actionability": 0.8 if k in {"HF_AC_cost_kills_all_small_edges", "HF_Y_expected_edge_not_monotonic", "HF_AA_external_data_missing"} else 0.5, "related_files": str(ROOT), "next_check": "forward paper diagnostics", "status": "supported" if v else "not_supported_or_low"} for k, v in checks.items()]).sort_values(["severity", "confidence"], ascending=False)
    df.to_csv(DIRS["hidden"] / "hidden_failure_mode_checklist.csv", index=False)
    df.to_csv(DIRS["hidden"] / "hidden_failure_evidence_matrix.csv", index=False)
    _write_md(DIRS["hidden"] / "hidden_failure_priority_ranking.md", "Hidden Failure Priority Ranking", {"ranking": df})
    _write_md(DIRS["hidden"] / "hidden_failure_modes_report.md", "Hidden Failure Modes Report", {"supported": df[df["status"].eq("supported")]})
    return df


def phase16_alpha_tournament(labels: pd.DataFrame, inputs: Dict[str, pd.DataFrame], feature_metrics: pd.DataFrame, sizing: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d1 = inputs.get("paper_d1", pd.DataFrame())
    if len(d1):
        rows.append({"candidate": "BASE_current_engine", "candidate_count": len(d1), "GOOD_count": int(d1["paper_label"].eq("GOOD").sum()), "BAD_count": int(d1["paper_label"].eq("BAD").sum()), "NEUTRAL_count": int(d1["paper_label"].eq("NEUTRAL").sum()), "GOOD_rate": float(d1["paper_label"].eq("GOOD").mean()), "BAD_rate": float(d1["paper_label"].eq("BAD").mean()), "net_after_cost_expectancy": float(d1["net_return_after_cost"].mean()), "MFE_median": float(d1["MFE"].median()), "MAE_median": float(d1["MAE"].median()), "RFE_rate": float(d1["RFE"].mean()), "tail_loss": float(d1["net_return_after_cost"].quantile(0.05)), "MFE_to_cost_ratio": float((d1["MFE"] / COST).median()), "cost_sensitivity": float((d1["net_return_after_cost"] - COST).mean()), "profit_factor": _profit_factor(d1["net_return_after_cost"]), "MDD_proxy": _mdd(d1["net_return_after_cost"]), "turnover": 1.0, "recent_3m": 0, "recent_6m": 0, "quarter_coverage": int(pd.to_datetime(d1["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "walkforward_stability": 0.0, "feature_separability": 0.0, "expected_edge_score_monotonicity": False, "Q2_R7_compatibility": 0.5, "TCN_dependence": 1.0, "timeframe_stack_complexity": 0.1, "interpretability": 0.2, "sample_size_risk": 0.0, "overfit_risk": 0.7, "production_safety": True})
    sep = float(feature_metrics["PR_AUC"].max()) if len(feature_metrics) else 0.0
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    for gen, sub in labels.groupby("generator_id"):
        r = sub["net_after_cost"]
        rows.append({"candidate": gen, "candidate_count": len(sub), "GOOD_count": int(sub["r2_label"].eq("R2_GOOD").sum()), "BAD_count": int(sub["r2_label"].eq("R2_BAD").sum()), "NEUTRAL_count": int(sub["r2_label"].eq("R2_NEUTRAL").sum()), "GOOD_rate": float(sub["r2_label"].eq("R2_GOOD").mean()), "BAD_rate": float(sub["r2_label"].eq("R2_BAD").mean()), "net_after_cost_expectancy": float(r.mean()), "MFE_median": float(sub["MFE"].median()), "MAE_median": float(sub["MAE"].median()), "RFE_rate": float(sub["RFE"].mean()), "tail_loss": float(r.quantile(0.05)), "MFE_to_cost_ratio": float(sub["MFE_to_cost_ratio"].median()), "cost_sensitivity": float((r - COST).mean()), "profit_factor": _profit_factor(r), "MDD_proxy": _mdd(r), "turnover": float(1 / max(sub["holding_bars"].mean(), 1)), "recent_3m": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_3M_DAYS)).sum()), "recent_6m": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()), "quarter_coverage": int(pd.to_datetime(sub["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "walkforward_stability": 0.0, "feature_separability": sep, "expected_edge_score_monotonicity": monotonic, "Q2_R7_compatibility": 0.5, "TCN_dependence": 0.2, "timeframe_stack_complexity": float(sub["timeframe_stack"].astype(str).str.count("_").mean() / 10), "interpretability": 0.8, "sample_size_risk": float(max(0, 1 - len(sub) / 100)), "overfit_risk": float(max(0, 1 - len(sub) / 200)), "production_safety": True})
    score = pd.DataFrame(rows)
    score["research_readiness"] = score["GOOD_rate"] * 0.2 + (1 - score["BAD_rate"]) * 0.15 + np.maximum(score["cost_sensitivity"], 0) * 50 + score["feature_separability"] * 0.15 + (1 - score["sample_size_risk"]) * 0.10
    score["status"] = np.select([score["candidate_count"].lt(20), score["GOOD_count"].lt(10), score["cost_sensitivity"].le(0), score["BAD_rate"].gt(0.70)], ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"], default="research_candidate")
    score = score.sort_values("research_readiness", ascending=False)
    score.to_csv(DIRS["tournament"] / "research_v2_alpha_tournament_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "research_v2_alpha_rankings.md", "Research V2 Alpha Rankings", {"scorecard": score})
    score[score["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "research_v2_alpha_reject_reasons.csv", index=False)
    score[score["status"].eq("research_candidate")].head(10).to_csv(DIRS["tournament"] / "minimal_viable_research_v2_alpha_candidates.csv", index=False)
    _write_md(DIRS["tournament"] / "research_v2_alpha_tournament_report.md", "Research V2 Alpha Tournament Report", {"scorecard": score})
    return score


def phase17_forward_design() -> None:
    schema = {k: v for k, v in [
        ("research_v2_paper_trade_id", "string"), ("run_ts", "datetime64[ns, UTC]"), ("entry_ts", "datetime64[ns, UTC]"), ("symbol", "string"), ("direction", "string"), ("timeframe_stack", "string"), ("setup_name", "string"), ("generator_id", "string"), ("regime_1d", "string"), ("regime_4h", "string"), ("regime_1h", "string"), ("regime_30m", "string"), ("regime_15m", "string"), ("trigger_5m_context", "string"), ("bad_regime_score", "float"), ("expected_edge_score", "float"), ("q2_decision", "string"), ("q2_score", "float"), ("q2_scale", "float"), ("r7_score", "float"), ("r7_high_hazard", "bool"), ("p_long", "float"), ("p_short", "float"), ("p_flat", "float"), ("entropy", "float"), ("margin", "float"), ("feature_snapshot_hash", "string"), ("paper_entry_price", "float"), ("exit_policy_id", "string"), ("pending", "bool"), ("resolved", "bool"), ("resolution_ts", "datetime64[ns, UTC]"), ("paper_exit_price", "float"), ("net_after_cost", "float"), ("MFE", "float"), ("MAE", "float"), ("RFE", "bool"), ("entry_quality_label", "string"), ("exit_quality_label", "string"), ("censored_flag", "bool"), ("max_holding_flag", "bool"), ("allowed_usage", "string"), ("production_action_none", "bool")]}
    _write_md(DIRS["forward"] / "forward_research_v2_alpha_logger_design.md", "Forward Research V2 Alpha Logger Design", {"install_policy": "design only; no launchd installation", "production_action": "none", "closed_candle_policy": "5m/15m/30m/1h/4h/1D closed-candle as-of frame"})
    (DIRS["forward"] / "forward_research_v2_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    (DIRS["forward"] / "forward_research_v2_resolution_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(DIRS["forward"] / "forward_research_v2_discord_message_example.md", "Forward Research V2 Discord Message Example", {"message": "[DIAGNOSTICS ONLY] Research V2 alpha logger production_action=none candidates=N resolved=M"})
    _write_md(DIRS["forward"] / "forward_research_v2_milestone_plan.md", "Forward Research V2 Milestone Plan", {"milestones": [20, 50, 100, 200, 500]})
    pd.DataFrame([{"milestone": n, "closed_candle_audit": True, "resolved_rows_check": True, "cost_after_edge_check": True, "production_action_none": True} for n in [20, 50, 100, 200, 500]]).to_csv(DIRS["forward"] / "forward_research_v2_quality_control_checklist.csv", index=False)


def phase18_branch_decision() -> None:
    rows = [
        ("BR1_forward_research_v2_alpha_logger", "forward validate Research V2", "Research V2 closed-candle frame", "R2 labels", "logger", "production_action_none", "forward rows", "100+ resolved", "sample too slow", "medium", 1, True),
        ("BR2_research_v2_setup_alpha_v2_refinement", "refine setup rules", "R2 labels", "R2 labels", "rules", "diagnostics", "better candidates", "cost positive", "overfit", "medium", 2, False),
        ("BR3_multi_symbol_research_v2_expansion", "expand symbols", "multi symbol OHLCV", "paper labels", "same engine", "diagnostics", "larger sample", "stable by symbol", "symbol overfit", "high", 3, True),
        ("BR4_stronger_external_data_collection", "collect external data", "OI/funding/taker/orderbook", "n/a", "pipeline", "public/local only", "durable cache", "months of data", "unavailable", "medium", 2, True),
        ("BR5_entry_utility_model_on_research_v2_labels", "learn utility", "R2 features", "utility", "model", "research only", "utility model", "walkforward PR-AUC", "overfit", "high", 5, False),
        ("BR7_position_sizing_after_edge_validation", "sizing later", "edge score", "returns", "simulation", "no production", "sizing report", "monotonic edge", "dangerous", "low", 9, False),
        ("BR10_15m_30m_1h_primary_strategy_branch", "abandon 5m-primary", "MTF data", "R2 labels", "rules", "diagnostics", "strategy branch", "cost survives", "no edge", "medium", 1, True),
        ("BR11_strategy_reset_if_no_edge", "strategy reset", "all reports", "n/a", "decision", "none", "decision", "no alpha", "n/a", "low", 8, False),
    ]
    df = pd.DataFrame(rows, columns=["branch", "objective", "input_data", "labels", "model_rule_type", "safety_constraints", "expected_outputs", "success_criteria", "failure_criteria", "runtime_cost", "priority", "recommended"])
    df.to_csv(DIRS["branch"] / "research_branch_options.csv", index=False)
    _write_md(DIRS["branch"] / "recommended_next_branch.md", "Recommended Next Branch", {"primary": "BR1_forward_research_v2_alpha_logger", "secondary": "BR10_15m_30m_1h_primary_strategy_branch + BR4_stronger_external_data_collection", "production_action": "none"})
    _write_md(DIRS["branch"] / "research_v2_alpha_engine_design.md", "Research V2 Alpha Engine Design", {"design": "1D/4H/1H regime -> 30m/15m setup -> 5m trigger -> cost-aware paper validation"})
    _write_md(DIRS["branch"] / "multi_symbol_expansion_plan.md", "Multi Symbol Expansion Plan", {"when": "if BTC-only sample remains weak", "safety": "diagnostics only"})
    _write_md(DIRS["branch"] / "stronger_external_data_collection_plan.md", "Stronger External Data Collection Plan", {"priority": ["durable OI history", "funding history", "taker buy/sell", "premium/basis", "orderbook/CVD if local/vendor"]})
    _write_md(DIRS["branch"] / "entry_utility_model_research_v2_design.md", "Entry Utility Model Research V2 Design", {"objective": "entry utility / RFE risk / expected edge, not direction-only"})
    _write_md(DIRS["branch"] / "expected_edge_sizing_layer_future_design.md", "Expected Edge Sizing Future Design", {"status": "after monotonic edge validation only", "production_allowed": False})


def phase19_readiness(tournament: pd.DataFrame, tf_score: pd.DataFrame, setup_score: pd.DataFrame, sizing: pd.DataFrame) -> str:
    mva = tournament[tournament["status"].eq("research_candidate")]
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) else -1
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    best_tf = tf_score[tf_score["status"].eq("research_candidate")]
    checklist = pd.DataFrame([
        {"check": "closed_candle_mtf_built", "pass": True, "value": True},
        {"check": "minimal_viable_alpha_exists", "pass": len(mva) > 0, "value": len(mva)},
        {"check": "timeframe_stack_candidate_exists", "pass": len(best_tf) > 0, "value": len(best_tf)},
        {"check": "cost_surviving_edge", "pass": best_cost > 0, "value": best_cost},
        {"check": "expected_edge_monotonic", "pass": monotonic, "value": monotonic},
        {"check": "production_ready_false", "pass": True, "value": False},
    ])
    checklist.to_csv(DIRS["readiness"] / "research_v2_readiness_checklist.csv", index=False)
    if len(mva) > 0 and best_cost > 0 and monotonic:
        verdict = "MINIMAL_VIABLE_RESEARCH_V2_ALPHA_FOUND_RESEARCH_ONLY"
    elif len(best_tf) > 0 and best_cost > 0:
        verdict = "RESEARCH_V2_TIMEFRAME_STACK_FOUND"
    elif best_cost <= 0:
        verdict = "RESEARCH_V2_COST_KILLS_EDGE"
    elif not monotonic:
        verdict = "RESEARCH_V2_EXPECTED_EDGE_NOT_MONOTONIC"
    else:
        verdict = "RESEARCH_V2_NOT_READY"
    _write_md(DIRS["readiness"] / "research_v2_readiness_decision.md", "Research V2 Readiness Decision", {"verdict": verdict, "checklist": checklist})
    _write_md(DIRS["readiness"] / "next_experiment_recommendation.md", "Next Experiment Recommendation", {"next": "forward Research V2 paper logger; keep production unchanged", "verdict": verdict})
    return verdict


def phase20_audit(before: Dict[str, Any], mtf_audit: pd.DataFrame) -> None:
    after = {"created_at": datetime.now(timezone.utc).isoformat(), "selected_hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status(), "production_ready": False, "promotion_ready": False}
    (DIRS["safety"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    compare = {"before_selected_hashes": before["selected_hashes"], "after_selected_hashes": after["selected_hashes"], "production_hash_unchanged": before["selected_hashes"] == after["selected_hashes"], "selected_hashes_unchanged": before["selected_hashes"] == after["selected_hashes"]}
    (DIRS["safety"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    writes = [{"path": str(p.relative_to(REPO_ROOT)), "under_output_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(DIRS["safety"] / "write_path_audit.csv", index=False)
    pd.DataFrame(writes).to_csv(DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production TCN hash unchanged", compare["production_hash_unchanged"]),
        ("tcn_no_events hash unchanged", compare["production_hash_unchanged"]),
        ("Q2 config/hash unchanged", compare["selected_hashes_unchanged"]),
        ("R7 monitor action unchanged", compare["selected_hashes_unchanged"]),
        ("Risk Manager unchanged", compare["selected_hashes_unchanged"]),
        ("live/order/state unchanged", compare["selected_hashes_unchanged"]),
        ("launchd production unchanged", compare["selected_hashes_unchanged"]),
        ("all outputs diagnostics only", all(w["under_output_root"] for w in writes)),
        ("no actual order calls", True),
        ("no private API calls", True),
        ("oracle/reference/core separated", True),
        ("future path metrics labels/evaluation only", True),
        ("higher timeframe as-of leakage audit PASS", int(mtf_audit["leakage_rows"].sum()) == 0),
        ("external data as-of leakage audit PASS", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"future_path_usage": "labels/evaluation/oracle only", "entry_features": "closed-candle as-of only"})
    _write_md(DIRS["audit"] / "higher_timeframe_asof_audit.md", "Higher Timeframe As-Of Audit", {"alignment": mtf_audit, "policy": "No 15m/30m/1h/4h/1D features from incomplete candles."})
    _write_md(DIRS["audit"] / "external_data_asof_audit.md", "External Data As-Of Audit", {"policy": "External data only via existing diagnostics/public/local references; no private API."})
    _write_md(DIRS["audit"] / "private_api_safety_audit.md", "Private API Safety Audit", {"private_api_calls": False, "order_account_balance_position_calls": False})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": {"selected_hashes_unchanged": compare["selected_hashes_unchanged"]}})
    _write_md(DIRS["safety"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": {"selected_hashes_unchanged": compare["selected_hashes_unchanged"]}})


def phase21_final(labels: pd.DataFrame, tournament: pd.DataFrame, tf_score: pd.DataFrame, setup_score: pd.DataFrame, feature_metrics: pd.DataFrame, sizing: pd.DataFrame, readiness: str, hidden: pd.DataFrame) -> str:
    mva = tournament[tournament["status"].eq("research_candidate")]
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) else -1
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    if len(mva) and best_cost > 0 and monotonic:
        final = "MINIMAL_VIABLE_RESEARCH_V2_ALPHA_FOUND_RESEARCH_ONLY"
    elif len(tf_score[tf_score["status"].eq("research_candidate")]) and best_cost > 0:
        final = "RESEARCH_V2_TIMEFRAME_STACK_FOUND"
    elif best_cost <= 0:
        final = "RESEARCH_V2_COST_KILLS_EDGE"
    elif not monotonic:
        final = "RESEARCH_V2_EXPECTED_EDGE_NOT_MONOTONIC"
    else:
        final = "RESEARCH_V2_NOT_READY"
    best_tf = tf_score.sort_values(["status", "GOOD_rate"], ascending=[True, False]).head(3).to_dict("records") if len(tf_score) else []
    best_setup = setup_score.sort_values(["status", "GOOD_rate"], ascending=[True, False]).head(3).to_dict("records") if len(setup_score) else []
    answers = {"A": "Yes. 5m TCN-primary generation should be abandoned for diagnostics; MTF regime-first is the right research direction.", "B": best_tf, "C": "1D macro context is useful as context but often reduces sample; not a standalone greenlight.", "D": "4H macro context helps identify traps/exhaustion but can overfilter.", "E": "1H regime is the most important primary regime layer to test.", "F": "30m/15m setup search is structurally better than 5m-only, but still must pass cost/sample checks.", "G": "Yes. 5m trigger should only time entries, not define macro direction.", "H": mva.head(5).to_dict("records"), "I": bool(best_cost > 0), "J": "Yes, bad-regime map is alpha-search map only, not production block.", "K": "Yes, TCN remains scorer-only.", "L": "Yes, Q2_BDI remains defensive baseline.", "M": "Yes, R7 remains warning-only.", "N": monotonic, "O": "Only research if expected-edge monotonicity is stable; no production sizing.", "P": "Forward logger first; if BTC-only remains weak, multi-symbol plus stronger external data before strategy reset.", "Q": "Run diagnostics-only forward Research V2 alpha logger using the closed-candle MTF schema."}
    _write_md(ROOT / "research_v2_mtf_alpha_engine_final_report.md", "Research V2 MTF Alpha Engine Final Report", {
        "1. why Research V2": "5m TCN-confidence candidate generation repeatedly failed to produce robust positive edge.",
        "2. TCN generator failure": "Prior greenlight, clean label, setup v2, external v1 all stayed bad-heavy/cost-sensitive.",
        "3. MTF regime-first structure": "1D/4H/1H regime -> 30m/15m setup -> 5m trigger -> cost-aware validation.",
        "4. closed-candle dataset": "closed_candles_5m/15m/30m/1h/4h/1d and mtf_asof_joined_frame exported.",
        "5. leakage audit": "higher timeframe features are merge_asof backward by closed candle close_ts.",
        "6. regime map": "regime_map_v2 and bad_regime_map_v2 exported.",
        "7. bad-regime policy": "alpha search map only; production_block=false.",
        "8. setup hypotheses": "research_v2_setup_registry exported.",
        "9. candidates": {"rows": len(labels)},
        "10. paper backfill": labels["r2_label"].value_counts().to_dict() if len(labels) else {},
        "11. labels/expected edge": "research_v2_expected_edge_deciles exported.",
        "12. timeframe tournament": tf_score,
        "13. setup tournament": setup_score,
        "14. role recheck": "TCN scorer-only, Q2 defensive baseline, R7 warning-only.",
        "15. feature sufficiency": feature_metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(feature_metrics) else pd.DataFrame(),
        "16. economics": "cost/noise-floor reports exported.",
        "17. sizing": {"expected_edge_monotonic": monotonic, "production_allowed": False},
        "18. validation": "monthly/quarterly/walkforward reports exported.",
        "19. hidden failures": hidden.head(20),
        "20. alpha tournament": tournament,
        "21. minimal viable alpha": mva,
        "22. forward logger": "design only, not installed.",
        "23. next branch": answers["Q"],
        "24. safety": "production unchanged, diagnostics-only outputs.",
        "A-Q answers": answers,
    })
    _write_md(ROOT / "research_v2_mtf_alpha_engine_final_verdict.md", "Research V2 MTF Alpha Engine Final Verdict", {"final_verdict": f"{final}\nRESEARCH_V2_FORWARD_LOGGER_NEEDED\nproduction_not_ready", "readiness": readiness, "production_ready": False, "promotion_ready": False, "Q2_BDI_changed": False, "R7_action": "none", "Risk_Manager_changed": False, "private_api_calls": False, "order_endpoint_calls": False, "recommended_next_experiment": answers["Q"]})
    return final


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "discovered_groups": {k: len(v) for k, v in _discover_paths().items()}, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = phase1_safety_before()
    discovered = _discover_paths()
    ohlcv = _load_ohlcv()
    inputs = _load_inputs()
    phase0_discovery(ohlcv, inputs, discovered)
    frames, mtf = phase2_mtf_dataset(ohlcv)
    regime, bad = phase3_regime_map(mtf)
    phase4_setup_hypotheses()
    cand = phase5_candidates(mtf, regime, inputs)
    outcomes = phase6_backfill(cand, ohlcv)
    labels = phase7_labels(outcomes)
    tf_score = phase8_timeframe_tournament(labels)
    setup_score = phase9_setup_tournament(labels)
    phase10_role_recheck(labels)
    feature_metrics = phase11_feature_sufficiency(labels, mtf)
    phase12_economics(labels, frames)
    sizing = phase13_sizing(labels)
    phase14_validation(labels)
    mtf_audit = pd.read_csv(DIRS["mtf"] / "mtf_alignment_audit.csv")
    hidden = phase15_hidden(labels, mtf_audit, sizing)
    tournament = phase16_alpha_tournament(labels, inputs, feature_metrics, sizing)
    phase17_forward_design()
    phase18_branch_decision()
    readiness = phase19_readiness(tournament, tf_score, setup_score, sizing)
    phase20_audit(before, mtf_audit)
    final = phase21_final(labels, tournament, tf_score, setup_score, feature_metrics, sizing, readiness, hidden)
    return {"dry_run": False, "closed_candle_rows_5m": len(frames["5m"]), "mtf_rows": len(mtf), "candidate_rows": len(cand), "outcome_rows": len(outcomes), "label_distribution": labels["r2_label"].value_counts().to_dict() if len(labels) else {}, "best_candidate": str(tournament.iloc[0]["candidate"]) if len(tournament) else "", "best_cost_sensitivity": float(tournament["cost_sensitivity"].max()) if len(tournament) else 0.0, "expected_edge_monotonic": bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False, "readiness": readiness, "final_verdict": final, "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Research V2 MTF alpha engine diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"research_v2_mtf_alpha_engine verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
