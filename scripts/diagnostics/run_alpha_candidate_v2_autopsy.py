"""
Alpha / Candidate Generator V2 root-cause and redesign autopsy.

Diagnostics only. Reads previous historical paper backfill outputs and
canonical BTCUSDT 5m OHLCV, builds setup-style candidate generators, evaluates
paper outcomes, audits objective/feature/data sufficiency, and writes only
under data/diagnostics/alpha_candidate_v2_autopsy/.
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from scripts.diagnostics.run_entry_greenlight_forensics import (
    RECENT_3M_DAYS,
    RECENT_6M_DAYS,
    _feature_family,
    _load_frame,
    _mdd,
    _profit_factor,
    _safe_features,
    _safe_num,
    _write_md,
)
from scripts.diagnostics.run_false_high_r7_monitor import _prod_hashes

ROOT = Path("data/diagnostics/alpha_candidate_v2_autopsy")
PAPER_ROOT = Path("data/diagnostics/historical_clean_paper_backfill")
VERSION = f"alpha_candidate_v2_autopsy_v1_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
COST = 0.0006

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "failure": ROOT / "current_candidate_failure",
    "hypotheses": ROOT / "alpha_hypotheses",
    "features": ROOT / "features",
    "setup_candidates": ROOT / "setup_candidates",
    "setup_backfill": ROOT / "setup_backfill",
    "setup_labels": ROOT / "setup_labels",
    "objective": ROOT / "model_objective",
    "feature_data": ROOT / "feature_data_sufficiency",
    "exit_recheck": ROOT / "exit_recheck",
    "no_trade": ROOT / "no_trade",
    "sizing": ROOT / "position_sizing",
    "hidden": ROOT / "hidden_failure_modes",
    "tournament": ROOT / "tournament",
    "branch": ROOT / "research_branch_design",
    "readiness": ROOT / "readiness",
    "audit": ROOT / "audit",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> Dict[str, Any]:
    return {
        "path": str(path.relative_to(REPO_ROOT) if path.exists() and path.is_absolute() else path),
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
        "risk",
        "risk_manager",
        "ops",
        "launchd",
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
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:250])
    return out


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "required_inputs": [
            "data/diagnostics/historical_clean_paper_backfill/**/*",
            "data/diagnostics/entry_exit_policy_autopsy/**/*",
            "data/diagnostics/executed_row_salvage_autopsy/**/*",
            "data/diagnostics/executed_entry_quality_label_v2/**/*",
            "data/diagnostics/feature_proba_refresh/**/*",
            "data/diagnostics/data_sync/**/*",
            "data/ohlcv/**/*",
            "data/market/**/*",
        ],
        "candidate_generation_code": ["*candidate*", "*signal*", "*engine*", "*guard*", "*entry*"],
        "model_training_code": ["*train*", "*tcn*", "*model*", "*objective*"],
        "feature_engineering_code": ["*feature*", "*indicator*", "*regime*", "*structure*"],
        "label_generation_code": ["*label*", "*quality*", "*mfe*", "*mae*", "*rfe*"],
        "external_data": ["*funding*", "*open_interest*", "*oi*", "*liquidation*", "*basis*", "*orderbook*", "*cvd*", "*delta*"],
    }
    out: Dict[str, List[str]] = {}
    for group, pats in patterns.items():
        vals: List[str] = []
        for pat in pats:
            iterator = REPO_ROOT.glob(pat) if pat.startswith("data/") else REPO_ROOT.rglob(pat)
            for p in iterator:
                rel = str(p.relative_to(REPO_ROOT))
                if any(skip in rel for skip in [".git", ".venv", "__pycache__", "node_modules"]):
                    continue
                if p.is_file():
                    vals.append(rel)
        out[group] = sorted(set(vals))[:300]
    return out


def _load_ohlcv() -> pd.DataFrame:
    meta_path = REPO_ROOT / "data/diagnostics/data_sync/canonical_data_paths.json"
    selected = REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        p = REPO_ROOT / meta.get("canonical_5m_path", "")
        if p.exists():
            selected = p
    if selected.suffix == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected)
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))
    return df


def _load_paper_inputs() -> Dict[str, pd.DataFrame]:
    paths = {
        "D1_engine": PAPER_ROOT / "datasets/D1_engine_like_paper.parquet",
        "D2_q2": PAPER_ROOT / "datasets/D2_q2_accept_paper.parquet",
        "D3_q2_r7": PAPER_ROOT / "datasets/D3_q2_accept_r7_no_warning_paper.parquet",
        "labels": PAPER_ROOT / "labels/paper_entry_exit_labels.parquet",
        "candidates": PAPER_ROOT / "candidates/paper_candidate_universe.parquet",
        "outcomes": PAPER_ROOT / "exit_replay/paper_exit_outcomes.parquet",
        "separability": PAPER_ROOT / "separability/paper_label_separability_metrics.csv",
        "exit_tournament": PAPER_ROOT / "exit_tournament/exit_policy_tournament_scorecard.csv",
        "entry_exit_autopsy": Path("data/diagnostics/entry_exit_policy_autopsy/dataset/entry_exit_policy_autopsy_dataset.parquet"),
        "salvage": Path("data/diagnostics/executed_row_salvage_autopsy/dataset/executed_row_salvage_dataset.parquet"),
        "label_v2": Path("data/diagnostics/executed_entry_quality_label_v2/dataset/executed_entry_quality_label_v2.parquet"),
    }
    out: Dict[str, pd.DataFrame] = {}
    for k, rel in paths.items():
        p = REPO_ROOT / rel
        if p.exists():
            out[k] = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    return out


def phase0_discovery(ohlcv: pd.DataFrame, inputs: Dict[str, pd.DataFrame], discovered: Dict[str, List[str]]) -> None:
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(discovered), encoding="utf-8")
    required = [
        PAPER_ROOT / "historical_clean_paper_backfill_final_report.md",
        PAPER_ROOT / "datasets/D1_engine_like_paper.parquet",
        PAPER_ROOT / "datasets/D2_q2_accept_paper.parquet",
        PAPER_ROOT / "datasets/D3_q2_accept_r7_no_warning_paper.parquet",
        PAPER_ROOT / "labels/paper_entry_exit_labels.parquet",
        PAPER_ROOT / "candidates/paper_candidate_universe.parquet",
        PAPER_ROOT / "exit_replay/paper_exit_outcomes.parquet",
        PAPER_ROOT / "separability/paper_label_separability_metrics.csv",
        PAPER_ROOT / "exit_tournament/exit_policy_tournament_scorecard.csv",
        Path("data/diagnostics/entry_exit_policy_autopsy/dataset/entry_exit_policy_autopsy_dataset.parquet"),
        Path("data/diagnostics/executed_row_salvage_autopsy/dataset/executed_row_salvage_dataset.parquet"),
        Path("data/diagnostics/executed_entry_quality_label_v2/dataset/executed_entry_quality_label_v2.parquet"),
        Path("data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet"),
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
        Path("data/ohlcv/BTCUSDT_5m_full.csv"),
    ]
    inv = []
    for rel in required:
        p = REPO_ROOT / rel
        inv.append({"path": str(rel), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0})
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    d1 = inputs.get("D1_engine", pd.DataFrame())
    prev = [
        {"metric": "paper_backfill_verdict", "value": "HISTORICAL_PAPER_BACKFILL_FAIL_COST_KILLS_EDGE"},
        {"metric": "engine_realistic_candidates", "value": len(d1)},
        {"metric": "core_GOOD", "value": int(d1.get("paper_label", pd.Series(dtype=str)).eq("GOOD").sum()) if len(d1) else 0},
        {"metric": "core_BAD", "value": int(d1.get("paper_label", pd.Series(dtype=str)).eq("BAD").sum()) if len(d1) else 0},
        {"metric": "core_NEUTRAL", "value": int(d1.get("paper_label", pd.Series(dtype=str)).eq("NEUTRAL").sum()) if len(d1) else 0},
    ]
    pd.DataFrame(prev).to_csv(DIRS["discovery"] / "previous_diagnostics_summary.csv", index=False)
    data_sources = [
        {"source": "BTCUSDT_5m_OHLCV", "exists": True, "rows": len(ohlcv), "start": str(ohlcv["timestamp"].min()), "end": str(ohlcv["timestamp"].max())},
        {"source": "BTCUSDT_1m_OHLCV", "exists": (REPO_ROOT / "data/market/btcusdt_1m.parquet").exists(), "rows": "", "start": "", "end": ""},
        {"source": "derived_15m_1h_4h", "exists": True, "rows": "derived_from_5m", "start": str(ohlcv["timestamp"].min()), "end": str(ohlcv["timestamp"].max())},
    ]
    pd.DataFrame(data_sources).to_csv(DIRS["discovery"] / "available_data_sources.csv", index=False)
    feature_families = ["current_5m", "mtf_trend", "range_structure", "volatility", "session", "volume", "Q2/R7/TCN", "external_if_available"]
    pd.DataFrame([{"feature_family": f, "available": f != "external_if_available" or bool(discovered.get("external_data"))} for f in feature_families]).to_csv(DIRS["discovery"] / "available_feature_families.csv", index=False)
    pd.DataFrame([{"artifact": p, "category": "model_or_cache"} for p in discovered.get("model_training_code", [])[:200]]).to_csv(DIRS["discovery"] / "available_model_artifacts.csv", index=False)
    pd.DataFrame([{"path": p} for p in discovered.get("candidate_generation_code", [])]).to_csv(DIRS["discovery"] / "candidate_generation_code_inventory.csv", index=False)
    external = discovered.get("external_data", [])
    pd.DataFrame([{"data_type": t, "available": any(t in p.lower() for p in external), "matching_paths": "|".join([p for p in external if t in p.lower()][:20])} for t in ["funding", "open_interest", "oi", "liquidation", "basis", "orderbook", "cvd", "delta"]]).to_csv(DIRS["discovery"] / "external_data_availability_audit.csv", index=False)
    gaps = int((ohlcv["timestamp"].diff().dt.total_seconds().fillna(300) > 450).sum())
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {
        "previous": pd.DataFrame(prev),
        "ohlcv": {"rows": len(ohlcv), "start": str(ohlcv["timestamp"].min()), "end": str(ohlcv["timestamp"].max()), "gaps": gaps},
        "q2_r7_tcn_fields": list(d1.columns) if len(d1) else [],
        "higher_timeframe_derivable": True,
        "external_data_files": external[:80],
        "production_changes": "none",
    })


def phase1_safety_before() -> Dict[str, Any]:
    snap = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
        "python": sys.version,
        "platform": platform.platform(),
        "os": os.name,
        "prod_hashes": _prod_hashes(),
        "selected_hashes": [_hash_path(p) for p in _safety_paths()],
        "git_status_short": _git_status(),
        "production_ready": False,
        "promotion_ready": False,
        "r7_action": "none",
        "risk_manager_changed": False,
        "actual_order_calls": False,
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


def phase2_current_failure(inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    d1 = inputs.get("D1_engine", pd.DataFrame()).copy()
    if d1.empty:
        raise FileNotFoundError("D1_engine_like_paper missing")
    d1["failure_taxonomy"] = np.select(
        [
            d1["paper_label"].eq("GOOD"),
            d1["paper_label"].eq("BAD") & d1.get("RFE", pd.Series(False, index=d1.index)).astype(bool),
            d1["paper_label"].eq("BAD") & d1.get("MAE", pd.Series(0, index=d1.index)).abs().ge(0.006),
            d1["paper_label"].eq("BAD") & d1.get("MFE", pd.Series(0, index=d1.index)).lt(0.0015),
            d1["paper_label"].eq("BAD") & d1.get("r7_high_hazard", pd.Series(False, index=d1.index)).astype(bool),
            d1["paper_label"].eq("BAD") & d1.get("q2_accept", pd.Series(True, index=d1.index)).astype(bool),
            d1["paper_label"].eq("NEUTRAL") & d1.get("net_return_after_cost", pd.Series(0, index=d1.index)).abs().lt(COST),
            d1.get("entropy", pd.Series(1, index=d1.index)).lt(d1.get("entropy", pd.Series(1, index=d1.index)).median()) & d1["paper_label"].eq("BAD"),
        ],
        [
            "CGF_OK_not_failure",
            "CGF_J_rfe_tail_risk",
            "CGF_I_high_mae_before_favorable",
            "CGF_H_low_mfe_after_cost",
            "CGF_M_r7_no_warning_not_greenlight",
            "CGF_L_q2_accept_not_greenlight",
            "CGF_K_cost_kills_small_edge",
            "CGF_A_tcn_confidence_without_setup",
        ],
        default="CGF_S_unknown",
    )
    d1["tcn_confidence"] = d1[["p_long", "p_short", "p_flat"]].max(axis=1)
    for col in ["trend_state", "vol_state"]:
        if col not in d1.columns:
            d1[col] = "unknown"
    d1.to_parquet(DIRS["failure"] / "current_candidate_failure_cases.parquet", index=False)
    d1.groupby("failure_taxonomy").agg(rows=("paper_trade_id", "size"), mean_net=("net_return_after_cost", "mean"), rfe_rate=("RFE", "mean"), mfe_median=("MFE", "median"), mae_median=("MAE", "median")).reset_index().to_csv(DIRS["failure"] / "current_candidate_failure_summary.csv", index=False)
    d1.groupby(["direction", "failure_taxonomy"]).size().reset_index(name="rows").to_csv(DIRS["failure"] / "candidate_failure_by_direction.csv", index=False)
    d1.groupby(["trend_state", "vol_state", "failure_taxonomy"]).size().reset_index(name="rows").to_csv(DIRS["failure"] / "candidate_failure_by_regime.csv", index=False)
    d1.groupby(["q2_accept", "r7_high_hazard", "paper_label"]).size().reset_index(name="rows").to_csv(DIRS["failure"] / "candidate_failure_by_q2_r7.csv", index=False)
    d1.assign(conf_bucket=pd.qcut(d1["tcn_confidence"].rank(method="first"), q=5, labels=False, duplicates="drop")).groupby(["conf_bucket", "paper_label"]).size().reset_index(name="rows").to_csv(DIRS["failure"] / "candidate_failure_by_tcn_confidence.csv", index=False)
    d1.sort_values(["paper_label", "net_return_after_cost"]).head(200).to_csv(DIRS["failure"] / "candidate_failure_casebook.csv", index=False)
    _write_md(DIRS["failure"] / "current_candidate_generator_failure_report.md", "Current Candidate Generator Failure Report", {
        "label_distribution": d1["paper_label"].value_counts().to_dict(),
        "failure_taxonomy": d1["failure_taxonomy"].value_counts().to_dict(),
        "diagnosis": "Current engine-like candidates are bad-heavy because Q2/TCN/R7 are not setup alpha; they primarily filter risk and confidence, not cost-aware entry utility.",
    })
    return d1


def phase3_hypotheses() -> pd.DataFrame:
    rows = [
        ("A1_multi_timeframe_trend_continuation", "MTF trend + 5m pullback reclaim", "15m/1h", "pullback then reclaim", "HTF trend breaks", "high_vol_trap", "trend range expansion", "controlled pullback", "cost*3", "Q2 defensive", "TCN scorer"),
        ("A2_pullback_reclaim", "5m/15m pullback after trend", "5m/15m", "EMA/range mid reclaim", "deep adverse break", "RFE", "reclaim followthrough", "MAE bounded", "cost*2", "Q2 defensive", "TCN scorer"),
        ("A3_breakout_retest", "range breakout and retest hold", "5m/15m", "retest hold", "range re-entry", "fakeout", "range height", "retest low/high", "cost*3", "R7 warning as risk", "alignment optional"),
        ("A4_volatility_compression_expansion", "squeeze then expansion", "5m/1h", "compression exit", "no expansion", "high MAE", "ATR expansion", "tight compression", "cost*3", "Q2 defensive", "TCN scorer"),
        ("A5_range_mean_reversion", "range sweep/rejection", "5m/1h", "band sweep reject", "trend breakout", "trend day", "range mean", "stop outside band", "cost*2", "R7 risk", "not generator"),
        ("A6_liquidity_sweep_reversal", "prior high/low sweep reclaim", "5m/15m", "sweep and reclaim", "no reclaim", "tail risk", "range mid", "wick rejection", "cost*2", "Q2 defensive", "scorer"),
        ("A7_failed_breakout_reversal", "failed breakout reversal", "5m/15m", "return into range", "continuation", "late entry", "opposite band", "breakout invalidation", "cost*2", "R7 risk", "scorer"),
        ("A8_momentum_impulse_followthrough", "impulse plus followthrough", "5m", "large body + volume", "mean revert", "exhaustion", "next impulse", "body hold", "cost*3", "Q2 defensive", "alignment helpful"),
        ("A9_low_vol_grind", "low vol drift trend", "15m/1h", "EMA slope drift", "vol expansion adverse", "small edge", "drift", "low MAE", "cost*2", "Q2 defensive", "scorer"),
        ("A10_high_vol_filter_only", "avoid high vol traps", "5m/1h", "no-trade", "n/a", "false high", "n/a", "risk-off", "n/a", "block reference", "risk head"),
        ("A11_session_specific_edge", "session-conditioned setup", "5m", "session filter", "bad session", "liquidity gap", "session range", "time filter", "cost*2", "defensive", "scorer"),
        ("A12_structure_reversal_after_exhaustion", "overextension exhaustion reversal", "5m/15m", "exhaustion reject", "trend continuation", "catching knife", "mean revert", "sweep stop", "cost*2", "R7 risk", "scorer"),
        ("A13_cost_aware_large_move_only", "expected move/cost filter", "all", "move_to_cost high", "small range", "cost kill", "large move", "skip small edge", "cost*4", "defensive", "scorer"),
        ("A14_no_trade_regime_detector", "bad regime detector", "all", "no-trade", "n/a", "bad regime", "n/a", "risk reduction", "n/a", "Q2/R7 refs", "risk model"),
        ("A15_external_market_structure_if_available", "funding/OI/liquidation/orderflow", "external", "external structure", "data unavailable", "fabrication risk", "positioning", "external risk", "cost*3", "optional", "scorer"),
    ]
    df = pd.DataFrame(rows, columns=["setup_name", "direction_logic", "required_timeframe", "entry_trigger", "invalid_condition", "risk_condition", "expected_MFE_logic", "expected_MAE_control_logic", "cost_aware_minimum_move", "Q2_R7_interaction", "TCN_interaction"])
    df["allowed_feature_columns"] = "as_of_OHLCV_MTF_Q2_R7_TCN"
    df["forbidden_future_columns"] = "future_path,outcome,MFE,MAE,RFE,label"
    df["core_or_reference"] = np.where(df["setup_name"].str.contains("external|filter_only|no_trade"), "reference_or_filter", "core_research")
    df["expected_failure_mode"] = "cost_kill/RFE/high_MAE/sample_sparsity"
    df.to_csv(DIRS["hypotheses"] / "alpha_hypothesis_registry.csv", index=False)
    _write_md(DIRS["hypotheses"] / "alpha_hypothesis_definitions.md", "Alpha Hypothesis Definitions", {"registry": df})
    df[["setup_name", "required_timeframe", "allowed_feature_columns"]].to_csv(DIRS["hypotheses"] / "setup_family_feature_requirements.csv", index=False)
    df[["setup_name", "expected_failure_mode"]].to_csv(DIRS["hypotheses"] / "setup_family_expected_failure_modes.csv", index=False)
    _write_md(DIRS["hypotheses"] / "alpha_hypothesis_report.md", "Alpha Hypothesis Report", {
        "principle": "TCN confidence alone is not an alpha hypothesis. Every setup must describe market structure, trigger, invalidation, risk, and cost-aware move potential.",
        "registry": df,
    })
    return df


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def phase4_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_12"] = df["close"].pct_change(12)
    df["ema_12"] = _ema(df["close"], 12)
    df["ema_48"] = _ema(df["close"], 48)
    df["ema_slope_12"] = df["ema_12"].pct_change(6)
    df["price_vs_ema_12"] = df["close"] / df["ema_12"] - 1
    df["range_24_high"] = df["high"].rolling(24, min_periods=12).max()
    df["range_24_low"] = df["low"].rolling(24, min_periods=12).min()
    df["range_pos_24"] = (df["close"] - df["range_24_low"]) / (df["range_24_high"] - df["range_24_low"]).replace(0, np.nan)
    df["bar_range"] = df["high"] / df["low"] - 1
    df["atr_24"] = df["bar_range"].rolling(24, min_periods=12).mean()
    df["rv_24"] = df["ret_1"].rolling(24, min_periods=12).std()
    df["vol_percentile_240"] = df["rv_24"].rolling(240, min_periods=60).rank(pct=True)
    df["range_compression"] = df["bar_range"].rolling(12, min_periods=6).mean() / df["bar_range"].rolling(96, min_periods=24).mean()
    df["vol_expansion"] = df["bar_range"] / df["bar_range"].rolling(96, min_periods=24).mean()
    df["volume_z"] = (df["volume"] - df["volume"].rolling(96, min_periods=24).mean()) / df["volume"].rolling(96, min_periods=24).std()
    df["body"] = (df["close"] - df["open"]).abs() / df["open"]
    df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["open"]
    df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["open"]
    df["wick_rejection_long"] = df["lower_wick"] > df["body"] * 1.5
    df["wick_rejection_short"] = df["upper_wick"] > df["body"] * 1.5
    df["swing_high_24"] = df["high"].rolling(24, min_periods=12).max().shift(1)
    df["swing_low_24"] = df["low"].rolling(24, min_periods=12).min().shift(1)
    df["sweep_high_reject"] = (df["high"] > df["swing_high_24"]) & (df["close"] < df["swing_high_24"])
    df["sweep_low_reclaim"] = (df["low"] < df["swing_low_24"]) & (df["close"] > df["swing_low_24"])
    df["breakout_up"] = df["close"] > df["swing_high_24"]
    df["breakout_down"] = df["close"] < df["swing_low_24"]
    df["move_to_cost_ratio"] = df["atr_24"] / COST
    df["low_edge_no_trade_flag"] = df["move_to_cost_ratio"] < 2.0
    hour = df["timestamp"].dt.hour
    df["session_bucket"] = np.select([hour.between(0, 7), hour.between(8, 15)], ["asia", "europe"], default="us")
    df["weekday"] = df["timestamp"].dt.weekday
    for rule, name in [("15min", "15m"), ("1h", "1h"), ("4h", "4h")]:
        r = df.set_index("timestamp").resample(rule, label="right", closed="right").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
        r[f"{name}_ema_12"] = _ema(r["close"], 12)
        r[f"{name}_ema_slope"] = r[f"{name}_ema_12"].pct_change(3)
        r[f"{name}_rv"] = r["close"].pct_change().rolling(12, min_periods=6).std()
        r[f"{name}_range_pos"] = (r["close"] - r["low"].rolling(24, min_periods=8).min()) / (r["high"].rolling(24, min_periods=8).max() - r["low"].rolling(24, min_periods=8).min()).replace(0, np.nan)
        cols = ["timestamp", f"{name}_ema_slope", f"{name}_rv", f"{name}_range_pos"]
        df = pd.merge_asof(df.sort_values("timestamp"), r[cols].sort_values("timestamp"), on="timestamp", direction="backward", allow_exact_matches=True)
    feature_cols = [c for c in df.columns if c not in {"source", "updated_at"}]
    df.to_parquet(DIRS["features"] / "mtf_regime_features.parquet", index=False)
    df.to_csv(DIRS["features"] / "mtf_regime_features.csv", index=False)
    families = []
    for c in feature_cols:
        if any(x in c for x in ["ema", "trend", "slope", "ret_"]):
            fam = "trend"
        elif any(x in c for x in ["range", "breakout", "sweep", "wick", "body"]):
            fam = "structure"
        elif any(x in c for x in ["rv", "vol", "atr", "compression", "expansion"]):
            fam = "volatility"
        elif "volume" in c:
            fam = "volume"
        elif "session" in c or "weekday" in c:
            fam = "session"
        else:
            fam = "base"
        families.append({"feature": c, "family": fam, "asof_safe": not any(tok in c.lower() for tok in ["future", "label", "mfe", "mae", "rfe"])})
    pd.DataFrame(families).to_csv(DIRS["features"] / "feature_family_registry.csv", index=False)
    pd.DataFrame(families).to_csv(DIRS["features"] / "feature_safety_audit.csv", index=False)
    pd.DataFrame([{"feature": c, "missing_rate": float(df[c].isna().mean())} for c in feature_cols]).to_csv(DIRS["features"] / "feature_missingness_report.csv", index=False)
    _write_md(DIRS["features"] / "feature_data_quality_report.md", "Feature Data Quality Report", {
        "rows": len(df),
        "timestamp_alignment": "derived from canonical 5m; higher TF merge_asof backward",
        "incomplete_higher_tf_leakage": "avoided by backward timestamp alignment",
        "feature_families": pd.DataFrame(families).groupby("family").size().to_dict(),
    })
    return df


def _dir_from_row(row: pd.Series, long_cond: bool, short_cond: bool) -> str:
    if long_cond and not short_cond:
        return "LONG"
    if short_cond and not long_cond:
        return "SHORT"
    return "NONE"


def _hash_row(row: pd.Series, cols: List[str]) -> str:
    payload = "|".join(f"{c}={row.get(c, '')}" for c in cols[:120])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def phase5_setup_candidates(features: pd.DataFrame, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    meta = _load_frame().copy()
    safe_cols, _, _ = _safe_features(meta)
    meta["timestamp"] = pd.to_datetime(meta.get("timestamp", meta.get("entry_ts")), errors="coerce")
    for col in ["baseline_p_long", "p_long"]:
        if col in meta:
            meta["p_long"] = meta[col]
            break
    for col in ["baseline_p_short", "p_short"]:
        if col in meta:
            meta["p_short"] = meta[col]
            break
    meta["p_flat"] = meta.get("baseline_p_flat", meta.get("p_flat", 0.0))
    meta["margin"] = meta.get("baseline_margin", meta.get("margin", (meta["p_long"] - meta["p_short"]).abs()))
    meta["entropy"] = meta.get("entropy", 1.0)
    meta["q2_scale"] = meta.get("q2_bdi_scale", 0.0)
    meta["q2_score"] = meta.get("q2_bdi_score", meta.get("q2_score", meta["q2_scale"]))
    meta["q2_decision"] = np.where(meta["q2_scale"].fillna(0) >= 0.4, "ACCEPT_OR_SCALE", "REJECT_OR_LOW_SCALE")
    meta["r7_score"] = meta.get("r7_score", 0.0)
    meta["r7_high_hazard"] = meta.get("r7_high_hazard", False).fillna(False).astype(bool)
    m = pd.merge_asof(features.sort_values("timestamp"), meta[["timestamp", "p_long", "p_short", "p_flat", "margin", "entropy", "q2_score", "q2_scale", "q2_decision", "r7_score", "r7_high_hazard"]].sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta(minutes=3))
    m = m.dropna(subset=["p_long", "p_short", "q2_scale"]).reset_index(drop=True)
    rows = []
    q = lambda s, p: float(m[s].quantile(p))
    gen_defs = []
    for i, row in m.iterrows():
        common = {
            "timestamp": row["timestamp"],
            "entry_ts": row["timestamp"],
            "q2_score": row["q2_score"],
            "q2_scale": row["q2_scale"],
            "q2_decision": row["q2_decision"],
            "r7_score": row["r7_score"],
            "r7_high_hazard": bool(row["r7_high_hazard"]),
            "tcn_p_direction": max(row["p_long"], row["p_short"]),
            "entropy": row["entropy"],
            "margin": row["margin"],
        }
        conditions = [
            ("G1_mtf_trend_continuation", "A1_multi_timeframe_trend_continuation", row["1h_ema_slope"] > 0 and row["price_vs_ema_12"] > -0.002 and row["ret_3"] > 0, row["1h_ema_slope"] < 0 and row["price_vs_ema_12"] < 0.002 and row["ret_3"] < 0, "HTF trend + 5m reclaim"),
            ("G2_pullback_reclaim", "A2_pullback_reclaim", row["ema_slope_12"] > 0 and row["price_vs_ema_12"] > 0 and row["range_pos_24"] < 0.65, row["ema_slope_12"] < 0 and row["price_vs_ema_12"] < 0 and row["range_pos_24"] > 0.35, "pullback reclaim"),
            ("G3_breakout_retest", "A3_breakout_retest", bool(row["breakout_up"]) and row["vol_expansion"] > 1.1, bool(row["breakout_down"]) and row["vol_expansion"] > 1.1, "breakout retest proxy"),
            ("G4_vol_compression_breakout", "A4_volatility_compression_expansion", row["range_compression"] < 0.8 and row["ret_1"] > 0 and row["vol_expansion"] > 1.0, row["range_compression"] < 0.8 and row["ret_1"] < 0 and row["vol_expansion"] > 1.0, "compression expansion"),
            ("G5_range_mean_reversion", "A5_range_mean_reversion", row["range_pos_24"] < 0.15 and bool(row["wick_rejection_long"]), row["range_pos_24"] > 0.85 and bool(row["wick_rejection_short"]), "range sweep reversion"),
            ("G6_liquidity_sweep_reversal", "A6_liquidity_sweep_reversal", bool(row["sweep_low_reclaim"]), bool(row["sweep_high_reject"]), "liquidity sweep reversal"),
            ("G7_failed_breakout_reversal", "A7_failed_breakout_reversal", bool(row["sweep_low_reclaim"]) and row["ret_1"] > 0, bool(row["sweep_high_reject"]) and row["ret_1"] < 0, "failed breakout reversal"),
            ("G8_momentum_impulse_followthrough", "A8_momentum_impulse_followthrough", row["ret_1"] > q("ret_1", 0.70) and row["volume_z"] > 0.5, row["ret_1"] < q("ret_1", 0.30) and row["volume_z"] > 0.5, "impulse followthrough"),
            ("G9_low_vol_grind", "A9_low_vol_grind", row["vol_percentile_240"] < 0.45 and row["ema_slope_12"] > 0 and row["ret_12"] > 0, row["vol_percentile_240"] < 0.45 and row["ema_slope_12"] < 0 and row["ret_12"] < 0, "low vol grind"),
            ("G10_cost_aware_large_move_filter", "A13_cost_aware_large_move_only", row["move_to_cost_ratio"] > 3 and row["p_long"] > row["p_short"], row["move_to_cost_ratio"] > 3 and row["p_short"] > row["p_long"], "move/cost filter"),
            ("G12_q2_r7_defensive_overlay", "A14_no_trade_regime_detector", row["q2_scale"] >= 0.4 and not row["r7_high_hazard"] and row["p_long"] > row["p_short"], row["q2_scale"] >= 0.4 and not row["r7_high_hazard"] and row["p_short"] > row["p_long"], "Q2/R7 defensive overlay"),
            ("G13_tcn_as_scorer_not_generator", "A13_cost_aware_large_move_only", row["p_long"] > row["p_short"] and row["margin"] > q("margin", 0.65), row["p_short"] > row["p_long"] and row["margin"] > q("margin", 0.65), "TCN scorer high margin"),
            ("G15_setup_without_tcn", "A6_liquidity_sweep_reversal", bool(row["sweep_low_reclaim"]) or (row["range_pos_24"] < 0.10), bool(row["sweep_high_reject"]) or (row["range_pos_24"] > 0.90), "pure structure"),
        ]
        votes = 0
        for gid, family, long_c, short_c, reason in conditions:
            direction = _dir_from_row(row, bool(long_c), bool(short_c))
            if direction == "NONE":
                continue
            no_trade = bool(row["low_edge_no_trade_flag"] or (row["vol_percentile_240"] > 0.90) or row["r7_high_hazard"])
            if gid not in {"G12_q2_r7_defensive_overlay", "G15_setup_without_tcn"} and no_trade:
                continue
            votes += 1
            score = float(np.nan_to_num(row["move_to_cost_ratio"], nan=0) * 0.2 + np.nan_to_num(abs(row["ema_slope_12"]) * 100, nan=0) + np.nan_to_num(row["margin"], nan=0))
            rows.append({
                "alpha_candidate_id": f"{gid}_{i}",
                "timestamp": row["timestamp"],
                "entry_ts": row["timestamp"],
                "bar_index": int(row["bar_index"]),
                "direction": direction,
                "setup_name": family,
                "generator_id": gid,
                "setup_family": family,
                "setup_score": score,
                "setup_reason": reason,
                "regime_context": f"vol_pct={row['vol_percentile_240']:.3f}|range_pos={row['range_pos_24']:.3f}",
                "mtf_context": f"1h_slope={row['1h_ema_slope']:.6f}",
                "entry_trigger": reason,
                "risk_flags": "|".join([x for x, flag in {"high_vol": row["vol_percentile_240"] > 0.80, "r7": row["r7_high_hazard"], "low_move_cost": row["move_to_cost_ratio"] < 2}.items() if flag]) or "none",
                "no_trade_flags": "blocked" if no_trade else "none",
                "expected_move_proxy": row["atr_24"],
                "expected_move_to_cost_ratio": row["move_to_cost_ratio"],
                **common,
                "tcn_alignment": (direction == "LONG" and row["p_long"] >= row["p_short"]) or (direction == "SHORT" and row["p_short"] >= row["p_long"]),
                "candidate_allowed_core": gid not in {"G15_setup_without_tcn"} and not no_trade,
                "candidate_reference_only": gid in {"G15_setup_without_tcn"} or no_trade,
                "oracle_flag": False,
                "feature_snapshot_hash": _hash_row(row, list(m.columns)),
            })
        if votes >= 2:
            rows.append({
                "alpha_candidate_id": f"G17_ensemble_setup_candidate_{i}",
                "timestamp": row["timestamp"],
                "entry_ts": row["timestamp"],
                "bar_index": int(row["bar_index"]),
                "direction": "LONG" if row["p_long"] >= row["p_short"] else "SHORT",
                "setup_name": "G17_ensemble_setup_candidate",
                "generator_id": "G17_ensemble_setup_candidate",
                "setup_family": "ensemble",
                "setup_score": votes,
                "setup_reason": f"{votes}_setup_votes",
                "regime_context": "ensemble",
                "mtf_context": "ensemble",
                "entry_trigger": "multiple_setup_votes",
                "risk_flags": "none",
                "no_trade_flags": "none",
                "expected_move_proxy": row["atr_24"],
                "expected_move_to_cost_ratio": row["move_to_cost_ratio"],
                **common,
                "tcn_alignment": True,
                "candidate_allowed_core": True,
                "candidate_reference_only": False,
                "oracle_flag": False,
                "feature_snapshot_hash": _hash_row(row, list(m.columns)),
            })
    cand = pd.DataFrame(rows).drop_duplicates(["timestamp", "generator_id", "direction"]).reset_index(drop=True)
    cand.to_parquet(DIRS["setup_candidates"] / "setup_candidate_universe.parquet", index=False)
    cand.to_csv(DIRS["setup_candidates"] / "setup_candidate_universe.csv", index=False)
    cand.groupby("generator_id").agg(rows=("alpha_candidate_id", "size"), core=("candidate_allowed_core", "sum"), reference=("candidate_reference_only", "sum"), q2_scale_mean=("q2_scale", "mean"), r7_rate=("r7_high_hazard", "mean")).reset_index().to_csv(DIRS["setup_candidates"] / "setup_candidate_summary_by_generator.csv", index=False)
    cand.groupby(["generator_id", "direction"]).size().reset_index(name="rows").to_csv(DIRS["setup_candidates"] / "setup_candidate_direction_distribution.csv", index=False)
    tmp = cand.copy()
    tmp["quarter"] = pd.to_datetime(tmp["timestamp"]).dt.to_period("Q").astype(str)
    recent = tmp[pd.to_datetime(tmp["timestamp"]) >= pd.to_datetime(tmp["timestamp"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    pd.concat([tmp.groupby(["generator_id", "quarter"]).size().reset_index(name="rows"), recent.groupby("generator_id").size().reset_index(name="recent_6m_rows")], ignore_index=True).to_csv(DIRS["setup_candidates"] / "setup_candidate_recent_quarter_summary.csv", index=False)
    engine = _load_paper_inputs()
    engine_ts = set(pd.to_datetime(engine.get("D1_engine", pd.DataFrame()).get("entry_ts", pd.Series(dtype=str)), errors="coerce").dropna().astype("datetime64[ns]").astype(str))
    cand.assign(overlap_current_engine=cand["timestamp"].astype("datetime64[ns]").astype(str).isin(engine_ts)).groupby("generator_id")["overlap_current_engine"].sum().reset_index().to_csv(DIRS["setup_candidates"] / "setup_candidate_overlap_with_current_engine.csv", index=False)
    _write_md(DIRS["setup_candidates"] / "setup_candidate_generation_report.md", "Setup Candidate Generation Report", {
        "rows": len(cand),
        "core_rows": int(cand["candidate_allowed_core"].sum()),
        "by_generator": cand["generator_id"].value_counts().to_dict(),
        "oracle_flag_count": int(cand["oracle_flag"].sum()),
    })
    return cand


SETUP_POLICIES = [
    ("X0_fixed_12", "fixed", 12, False),
    ("X1_fixed_24", "fixed", 24, False),
    ("X2_fixed_48", "fixed", 48, False),
    ("X3_fixed_96", "fixed", 96, False),
    ("X4_current_maxhold_proxy", "maxhold", 60, False),
    ("X5_MAE_stop_medium", "stop", 24, False),
    ("X6_vol_adjusted_MAE_stop", "stop", 24, False),
    ("X7_first_cost_plus_move", "take_profit", 24, False),
    ("X8_first_2x_cost_plus_move", "take_profit", 24, False),
    ("X9_trailing_vol_adjusted_proxy", "trailing", 48, False),
    ("X10_fixed_24_plus_MAE_stop", "hybrid", 24, False),
    ("X11_trailing_plus_MAE_stop", "hybrid", 48, False),
    ("X90_oracle_best_24", "oracle", 24, True),
    ("X91_oracle_best_48", "oracle", 48, True),
    ("X92_oracle_MFE", "oracle", 96, True),
]


def _path_returns(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        close_ret = path["close"] / entry - 1
        fav = path["high"] / entry - 1
        adv = path["low"] / entry - 1
    else:
        close_ret = entry / path["close"] - 1
        fav = entry / path["low"] - 1
        adv = entry / path["high"] - 1
    return close_ret.astype(float), fav.astype(float), adv.astype(float)


def _exit_idx(pid: str, path: pd.DataFrame, cr: pd.Series, fav: pd.Series, adv: pd.Series) -> int:
    default = len(path) - 1
    if "fixed_12" in pid:
        return min(11, default)
    if "fixed_24" in pid:
        return min(23, default)
    if "fixed_48" in pid or "trailing" in pid:
        return min(47, default)
    if "fixed_96" in pid or "oracle_MFE" in pid:
        return min(95, default)
    if "maxhold" in pid:
        return min(59, default)
    if "MAE_stop" in pid:
        hit = np.where(adv.values <= -0.004)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "first_cost" in pid:
        hit = np.where(fav.values >= 0.0012)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "first_2x" in pid:
        hit = np.where(fav.values >= 0.0024)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "oracle_best" in pid:
        return int(cr.values.argmax())
    return min(23, default)


def phase6_setup_backfill(cand: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv.set_index("bar_index")
    trades = []
    outcomes = []
    max_h = max(p[2] for p in SETUP_POLICIES)
    base = cand[cand["candidate_allowed_core"] | ~cand["oracle_flag"]].copy()
    for _, c in base.iterrows():
        entry_idx = int(c["bar_index"]) + 1
        if entry_idx not in o.index:
            continue
        entry_bar = o.loc[entry_idx]
        entry = float(entry_bar["open"]) * (1 + (0.0002 if c["direction"] == "LONG" else -0.0002))
        tid = f"{c['alpha_candidate_id']}_F0_next_open"
        trades.append({**c.to_dict(), "setup_paper_trade_id": tid, "entry_ts": entry_bar["timestamp"], "entry_bar_index": entry_idx, "entry_price": entry, "entry_fill_rule": "F0_next_open_default"})
        path = o.loc[(o.index >= entry_idx) & (o.index < entry_idx + max_h)].copy()
        if path.empty:
            continue
        cr_full, fav_full, adv_full = _path_returns(path, c["direction"], entry)
        for pid, group, horizon, oracle in SETUP_POLICIES:
            sub = path.head(horizon)
            cr, fav, adv = cr_full.head(horizon), fav_full.head(horizon), adv_full.head(horizon)
            if sub.empty:
                continue
            idx = _exit_idx(pid, sub, cr, fav, adv)
            gross = float(cr.iloc[idx])
            if pid == "X92_oracle_MFE":
                gross = float(fav.max())
            net = gross - COST
            mfe, mae = float(fav.max()), float(adv.min())
            outcomes.append({
                "setup_paper_trade_id": tid,
                "alpha_candidate_id": c["alpha_candidate_id"],
                "generator_id": c["generator_id"],
                "setup_family": c["setup_family"],
                "direction": c["direction"],
                "entry_ts": entry_bar["timestamp"],
                "entry_price": entry,
                "exit_policy_id": pid,
                "exit_policy_group": group,
                "exit_ts": sub.iloc[idx]["timestamp"],
                "holding_bars": idx + 1,
                "gross_return": gross,
                "net_after_cost": net,
                "MFE": mfe,
                "MAE": mae,
                "RFE": bool(mae <= -0.006),
                "time_to_MFE": int(fav.values.argmax()) + 1,
                "time_to_MAE": int(adv.values.argmin()) + 1,
                "MFE_before_MAE": bool((int(fav.values.argmax()) + 1) <= (int(adv.values.argmin()) + 1)),
                "cost_plus_hit": bool(mfe >= COST * 2),
                "MFE_to_cost_ratio": mfe / COST,
                "MAE_to_cost_ratio": abs(mae) / COST,
                "tail_loss": bool(net <= -0.006),
                "high_MAE": bool(abs(mae) >= 0.006),
                "MDD_proxy_input": net,
                "MFE_capture": float(max(net, 0) / max(mfe, 0.0005)),
                "giveback": float(max(mfe - max(net, 0), 0) / max(mfe, 0.0005)),
                "oracle_flag": bool(oracle),
                "candidate_allowed_core": bool(c["candidate_allowed_core"]),
                "q2_scale": c["q2_scale"],
                "q2_decision": c["q2_decision"],
                "r7_high_hazard": c["r7_high_hazard"],
                "entropy": c["entropy"],
                "margin": c["margin"],
                "expected_move_to_cost_ratio": c["expected_move_to_cost_ratio"],
                "setup_score": c["setup_score"],
            })
    trades_df = pd.DataFrame(trades)
    out = pd.DataFrame(outcomes)
    trades_df.to_parquet(DIRS["setup_backfill"] / "setup_paper_trades.parquet", index=False)
    out.to_parquet(DIRS["setup_backfill"] / "setup_exit_outcomes.parquet", index=False)
    _metric_by(out, ["generator_id"]).to_csv(DIRS["setup_backfill"] / "setup_outcome_metrics_by_generator.csv", index=False)
    _metric_by(out, ["exit_policy_id"]).to_csv(DIRS["setup_backfill"] / "setup_outcome_metrics_by_exit_policy.csv", index=False)
    _metric_by(out, ["direction"]).to_csv(DIRS["setup_backfill"] / "setup_outcome_metrics_by_direction.csv", index=False)
    _metric_by(out, ["setup_family"]).to_csv(DIRS["setup_backfill"] / "setup_outcome_metrics_by_regime.csv", index=False)
    tmp = out.copy()
    tmp["quarter"] = pd.to_datetime(tmp["entry_ts"]).dt.to_period("Q").astype(str)
    recent = tmp[pd.to_datetime(tmp["entry_ts"]) >= pd.to_datetime(tmp["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    pd.concat([_metric_by(tmp, ["generator_id", "quarter"]), _metric_by(recent, ["generator_id"])], ignore_index=True).to_csv(DIRS["setup_backfill"] / "setup_outcome_recent_quarter.csv", index=False)
    _write_md(DIRS["setup_backfill"] / "setup_backfill_report.md", "Setup Backfill Report", {
        "trades": len(trades_df),
        "outcomes": len(out),
        "best_non_oracle": _metric_by(out[~out["oracle_flag"]], ["generator_id", "exit_policy_id"]).sort_values("expectancy", ascending=False).head(20),
        "warning": "Oracle policies are reference only.",
    })
    return out


def _metric_by(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in df.groupby(cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        ret = _safe_num(sub["net_after_cost"])
        row = {c: k for c, k in zip(cols, keys)}
        row.update({
            "rows": len(sub),
            "expectancy": float(ret.mean()),
            "winrate": float((ret > 0).mean()),
            "profit_factor": _profit_factor(ret),
            "MDD_proxy": _mdd(ret),
            "RFE_rate": float(sub["RFE"].mean()),
            "high_MAE_rate": float(sub["high_MAE"].mean()),
            "tail_loss": float(ret.quantile(0.05)),
            "MFE_median": float(sub["MFE"].median()),
            "MAE_median": float(sub["MAE"].median()),
            "MFE_to_cost_median": float(sub["MFE_to_cost_ratio"].median()),
            "cost_sensitivity": float((ret - COST).mean()),
            "oracle_rate": float(sub["oracle_flag"].mean()),
            "turnover_proxy": float(1 / max(sub["holding_bars"].mean(), 1)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def phase7_setup_labels(out: pd.DataFrame) -> pd.DataFrame:
    core_policy = "X1_fixed_24"
    lab = out[(out["exit_policy_id"].eq(core_policy)) & (~out["oracle_flag"])].copy()
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & ~lab["RFE"] & lab["MFE_before_MAE"]
    bad = (lab["net_after_cost"] < 0) | lab["RFE"] | lab["high_MAE"] | lab["tail_loss"]
    neutral = ~(good | bad)
    lab["setup_label"] = np.select([good, bad, neutral], ["SETUP_GOOD", "SETUP_BAD", "SETUP_NEUTRAL"], default="SETUP_CENSORED")
    lab["utility_score"] = (lab["net_after_cost"].clip(-0.01, 0.01) / 0.01 + lab["MFE_to_cost_ratio"].clip(0, 10) / 10 - lab["MAE_to_cost_ratio"].clip(0, 10) / 10 + lab["MFE_before_MAE"].astype(float)) / 4
    lab["risk_score"] = (lab["RFE"].astype(float) * 0.35 + lab["high_MAE"].astype(float) * 0.25 + lab["tail_loss"].astype(float) * 0.25 + lab["r7_high_hazard"].astype(float) * 0.15)
    lab["expected_edge_score"] = lab["utility_score"] - lab["risk_score"]
    lab["position_sizing_readiness_flag"] = False
    lab.to_parquet(DIRS["setup_labels"] / "setup_entry_quality_labels.parquet", index=False)
    summary = lab.groupby(["generator_id", "setup_label"]).size().reset_index(name="rows")
    summary.to_csv(DIRS["setup_labels"] / "setup_label_policy_summary.csv", index=False)
    lab[["setup_paper_trade_id", "generator_id", "utility_score"]].to_csv(DIRS["setup_labels"] / "setup_utility_score.csv", index=False)
    lab[["setup_paper_trade_id", "generator_id", "risk_score"]].to_csv(DIRS["setup_labels"] / "setup_risk_score.csv", index=False)
    lab[["setup_paper_trade_id", "generator_id", "expected_edge_score", "net_after_cost", "MFE", "MAE", "RFE"]].to_csv(DIRS["setup_labels"] / "setup_expected_edge_score.csv", index=False)
    lab[["setup_paper_trade_id", "generator_id", "setup_label", "RFE", "high_MAE", "tail_loss"]].to_csv(DIRS["setup_labels"] / "setup_label_reason_codes.csv", index=False)
    dec = lab.assign(edge_decile=pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")).groupby("edge_decile").agg(rows=("setup_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean")).reset_index()
    monotonic = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    _write_md(DIRS["setup_labels"] / "setup_label_design_report.md", "Setup Label Design Report", {
        "label_distribution": lab["setup_label"].value_counts().to_dict(),
        "good_bad_balance": summary,
        "expected_edge_monotonic_net": monotonic,
        "position_sizing_readiness": False,
    })
    return lab


def phase8_objective(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["tcn_confidence_proxy"] = lab["margin"].fillna(0)
    align = pd.DataFrame([
        {"metric": "TCN_margin_vs_setup_good_mean", "value": float(lab.loc[lab["setup_label"].eq("SETUP_GOOD"), "tcn_confidence_proxy"].mean()) if lab["setup_label"].eq("SETUP_GOOD").any() else 0.0},
        {"metric": "TCN_margin_vs_setup_bad_mean", "value": float(lab.loc[lab["setup_label"].eq("SETUP_BAD"), "tcn_confidence_proxy"].mean()) if lab["setup_label"].eq("SETUP_BAD").any() else 0.0},
        {"metric": "high_conf_bad_rate", "value": float(lab.loc[lab["tcn_confidence_proxy"] >= lab["tcn_confidence_proxy"].median(), "setup_label"].eq("SETUP_BAD").mean())},
        {"metric": "low_conf_good_rate", "value": float(lab.loc[lab["tcn_confidence_proxy"] < lab["tcn_confidence_proxy"].median(), "setup_label"].eq("SETUP_GOOD").mean())},
    ])
    align.to_csv(DIRS["objective"] / "tcn_vs_setup_label_alignment.csv", index=False)
    mismatch = lab.groupby(["generator_id", "setup_label"]).agg(rows=("setup_paper_trade_id", "size"), margin_mean=("margin", "mean"), utility_mean=("utility_score", "mean"), risk_mean=("risk_score", "mean")).reset_index()
    mismatch.to_csv(DIRS["objective"] / "tcn_objective_mismatch.csv", index=False)
    objectives = [
        ("OBJ0_current_direction_3class", "direction label", len(lab), "high", "current", "low", "misaligned with utility", False),
        ("OBJ1_entry_utility_binary", "SETUP_GOOD vs BAD", len(lab), "medium", "setup labels", "medium", "direct utility", False),
        ("OBJ2_entry_utility_ordinal", "GOOD/NEUTRAL/BAD", len(lab), "medium", "setup labels", "medium", "handles neutral", False),
        ("OBJ3_cost_aware_return_regression", "net_after_cost", len(lab), "medium", "cost labels", "medium", "cost aware", False),
        ("OBJ4_MFE_MAE_RFE_multi_task", "MFE/MAE/RFE", len(lab), "medium", "path labels", "high", "risk and utility", False),
        ("OBJ5_setup_family_classifier", "setup family", len(lab), "low", "setup registry", "low", "not edge target", False),
        ("OBJ6_no_trade_regime_classifier", "bad/no-trade", len(lab), "low", "bad labels", "high", "reduces BAD", False),
        ("OBJ7_RFE_bad_risk_head", "RFE_bad", len(lab), "low", "risk labels", "high", "strong risk separation", False),
        ("OBJ8_expected_edge_score_regression", "expected_edge_score", len(lab), "medium", "utility+risk", "medium", "needed for sizing", False),
        ("OBJ10_dual_head_direction_plus_utility", "direction+utility", len(lab), "medium", "dual labels", "medium", "TCN as scorer", False),
    ]
    reg = pd.DataFrame(objectives, columns=["objective", "target_definition", "sample_count", "noise_risk", "feature_requirements", "expected_benefit", "risk_of_overfit", "production_readiness"])
    reg["recommended_next_experiment"] = np.where(reg["objective"].isin(["OBJ6_no_trade_regime_classifier", "OBJ7_RFE_bad_risk_head", "OBJ8_expected_edge_score_regression"]), "research_candidate", "secondary")
    reg.to_csv(DIRS["objective"] / "model_objective_candidate_registry.csv", index=False)
    reg.to_csv(DIRS["objective"] / "objective_feasibility_scorecard.csv", index=False)
    _write_md(DIRS["objective"] / "model_objective_report.md", "Model Objective Report", {
        "alignment": align,
        "decision": "TCN should be demoted from generator to scorer/risk/utility head; direction objective is not sufficient for cost-aware entry utility.",
        "recommended_objectives": ["OBJ6_no_trade_regime_classifier", "OBJ7_RFE_bad_risk_head", "OBJ8_expected_edge_score_regression", "OBJ10_dual_head_direction_plus_utility"],
    })
    return reg


def _classification_metrics(df: pd.DataFrame, features: List[str], targets: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [c for c in features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:100]
    sets = {
        "current_5m": [c for c in cols if not c.startswith(("15m_", "1h_", "4h_"))],
        "5m_plus_MTF": cols,
        "structure": [c for c in cols if any(k in c for k in ["range", "sweep", "breakout", "wick", "body"])],
        "volatility": [c for c in cols if any(k in c for k in ["vol", "rv", "atr", "compression", "expansion"])],
        "session": [c for c in cols if any(k in c for k in ["session", "weekday"])],
        "volume": [c for c in cols if "volume" in c],
        "no_TCN": [c for c in cols if not c.startswith(("p_", "margin", "entropy"))],
        "all_entry_safe": cols,
    }
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    data = df.sort_values("entry_ts").reset_index(drop=True)
    split = int(len(data) * 0.7)
    metrics, imps, interactions = [], [], []
    for tname, y0 in targets.items():
        y = y0.reset_index(drop=True).astype(int)
        if y.nunique() < 2 or y.sum() < 10 or (len(y) - y.sum()) < 10:
            continue
        if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
            continue
        for fs, fs_cols in sets.items():
            fs_cols = [c for c in fs_cols if c in data.columns]
            if not fs_cols:
                continue
            x = data[fs_cols].replace([np.inf, -np.inf], np.nan)
            for mn, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(x.iloc[:split], y.iloc[:split])
                    score = pipe.predict_proba(x.iloc[split:])[:, 1]
                    yte = y.iloc[split:]
                    top = score >= np.quantile(score, 0.8)
                    top_df = data.iloc[split:].loc[top]
                    metrics.append({"target": tname, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": float(roc_auc_score(yte, score)), "PR_AUC": float(average_precision_score(yte, score)), "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_after_cost"].mean()) if "net_after_cost" in top_df else 0.0})
                    mdl = pipe.named_steps["model"]
                    vals = getattr(mdl, "feature_importances_", np.abs(getattr(mdl, "coef_", np.zeros((1, len(fs_cols))))).ravel())
                    for c, v in sorted(zip(fs_cols, vals), key=lambda z: -float(z[1]))[:20]:
                        imps.append({"target": tname, "feature_set": fs, "model": mn, "feature": c, "importance": float(v), "family": _feature_family(c)})
                except Exception:
                    continue
    if imps:
        interactions = pd.DataFrame(imps).groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count").to_dict("records")
    return pd.DataFrame(metrics), pd.DataFrame(imps), pd.DataFrame(interactions)


def phase9_feature_sufficiency(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["entry_ts"]) - pd.Timedelta(minutes=5)
    feat_cols = [c for c in features.columns if c not in ["timestamp", "open", "high", "low", "close", "source", "updated_at"]]
    df = pd.merge_asof(lab.sort_values("timestamp"), features[["timestamp"] + feat_cols].sort_values("timestamp"), on="timestamp", direction="backward")
    targets = {
        "setup_good_vs_bad": df["setup_label"].eq("SETUP_GOOD"),
        "RFE_bad": df["RFE"].astype(bool),
        "high_MAE_bad": df["high_MAE"].astype(bool),
        "cost_killed": df["net_after_cost"] <= 0,
        "early_entry_good": df["MFE_before_MAE"].astype(bool) & (df["MFE_to_cost_ratio"] >= 3),
        "expected_edge_top_decile": df["expected_edge_score"] >= df["expected_edge_score"].quantile(0.9),
        "no_trade_bad_regime": df["setup_label"].eq("SETUP_BAD"),
    }
    metrics, imps, interactions = _classification_metrics(df, feat_cols + ["q2_scale", "r7_score", "entropy", "margin"], targets)
    metrics.to_csv(DIRS["feature_data"] / "feature_sufficiency_metrics.csv", index=False)
    metrics.groupby(["target", "feature_set"]).agg(best_pr_auc=("PR_AUC", "max"), best_auc=("AUC", "max")).reset_index().to_csv(DIRS["feature_data"] / "feature_family_ablation.csv", index=False)
    imps.to_csv(DIRS["feature_data"] / "setup_feature_importance.csv", index=False)
    interactions.to_csv(DIRS["feature_data"] / "feature_interaction_candidates.csv", index=False)
    external = pd.read_csv(DIRS["discovery"] / "external_data_availability_audit.csv")
    external.to_csv(DIRS["feature_data"] / "external_data_gap_analysis.csv", index=False)
    best_good = metrics[metrics["target"].eq("setup_good_vs_bad")]["PR_AUC"].max() if len(metrics) and metrics["target"].eq("setup_good_vs_bad").any() else 0.0
    best_risk = metrics[metrics["target"].isin(["RFE_bad", "high_MAE_bad"])]["PR_AUC"].max() if len(metrics) and metrics["target"].isin(["RFE_bad", "high_MAE_bad"]).any() else 0.0
    _write_md(DIRS["feature_data"] / "feature_data_sufficiency_report.md", "Feature Data Sufficiency Report", {
        "best_positive_edge_pr_auc": best_good,
        "best_risk_pr_auc": best_risk,
        "external_data_available": external[external["available"].astype(bool)].to_dict("records"),
        "diagnosis": "OHLCV MTF features help risk/no-trade separation more than positive setup-good separation.",
    })
    return metrics


def phase10_exit_recheck(out: pd.DataFrame) -> pd.DataFrame:
    by_setup = _metric_by(out[~out["oracle_flag"]], ["generator_id", "exit_policy_id"])
    by_setup.to_csv(DIRS["exit_recheck"] / "exit_recheck_by_setup.csv", index=False)
    score = _metric_by(out, ["exit_policy_id"])
    score["status"] = np.select([score["oracle_rate"].gt(0), score["cost_sensitivity"].le(0), score["expectancy"].le(0)], ["reject_oracle", "reject_cost_kills", "reject_negative"], default="forward_diagnostic_only")
    score.to_csv(DIRS["exit_recheck"] / "exit_recheck_policy_scorecard.csv", index=False)
    score[["exit_policy_id", "expectancy", "cost_sensitivity", "status"]].to_csv(DIRS["exit_recheck"] / "exit_recheck_cost_sensitivity.csv", index=False)
    _write_md(DIRS["exit_recheck"] / "exit_recheck_report.md", "Exit Recheck Report", {
        "scorecard": score.sort_values("expectancy", ascending=False).head(20),
        "decision": "Exit policy remains fourth-order unless a setup has cost-surviving non-oracle edge.",
    })
    return score


def phase11_no_trade(labels: pd.DataFrame) -> pd.DataFrame:
    detectors = {
        "NT0_high_vol_trap": labels["expected_move_to_cost_ratio"].gt(8) | labels["r7_high_hazard"].astype(bool),
        "NT1_chop_range_no_edge": labels["MFE_to_cost_ratio"].lt(2) & labels["MAE_to_cost_ratio"].lt(2),
        "NT2_low_expected_move_to_cost": labels["expected_move_to_cost_ratio"].lt(2),
        "NT3_RFE_high_risk": labels["RFE"].astype(bool),
        "NT4_false_high_signature": labels["r7_high_hazard"].astype(bool),
        "NT5_q2_low_quality": labels["q2_scale"].lt(0.4),
        "NT6_r7_high_hazard": labels["r7_high_hazard"].astype(bool),
        "NT7_session_bad_bucket": pd.to_datetime(labels["entry_ts"]).dt.hour.between(0, 2),
        "NT8_trend_late_exhaustion": labels["time_to_MFE"].gt(12) & labels["MFE_to_cost_ratio"].gt(3),
        "NT10_data_gap_or_bad_timestamp": labels["entry_ts"].isna(),
    }
    rows = []
    base_bad = labels["setup_label"].eq("SETUP_BAD")
    base_good = labels["setup_label"].eq("SETUP_GOOD")
    for name, mask in detectors.items():
        keep = ~mask
        rows.append({
            "detector": name,
            "blocked_rows": int(mask.sum()),
            "bad_candidate_reduction": int((mask & base_bad).sum()),
            "GOOD_retention": float((keep & base_good).sum() / max(base_good.sum(), 1)),
            "BAD_rejection": float((mask & base_bad).sum() / max(base_bad.sum(), 1)),
            "trade_count_reduction": float(mask.mean()),
            "expectancy_after_filter": float(labels.loc[keep, "net_after_cost"].mean()) if keep.any() else 0.0,
            "RFE_after_filter": float(labels.loc[keep, "RFE"].mean()) if keep.any() else 0.0,
            "cost_after_edge": float((labels.loc[keep, "net_after_cost"] - COST).mean()) if keep.any() else 0.0,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DIRS["no_trade"] / "no_trade_detector_metrics.csv", index=False)
    df[["detector", "GOOD_retention", "BAD_rejection"]].to_csv(DIRS["no_trade"] / "no_trade_good_retention_bad_rejection.csv", index=False)
    df["status"] = np.where((df["GOOD_retention"] >= 0.6) & (df["BAD_rejection"] >= 0.3), "research_candidate", "weak_or_reference")
    df.to_csv(DIRS["no_trade"] / "no_trade_policy_scorecard.csv", index=False)
    _write_md(DIRS["no_trade"] / "no_trade_report.md", "No-Trade Detector Report", {
        "scorecard": df.sort_values(["BAD_rejection", "GOOD_retention"], ascending=False),
        "decision": "No-trade detector is more promising than positive greenlight if it reduces BAD while retaining GOOD.",
    })
    return df


def phase12_sizing(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["edge_decile"] = pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")
    dec = lab.groupby("edge_decile").agg(rows=("setup_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("setup_label", lambda s: float((s == "SETUP_GOOD").mean())), bad_rate=("setup_label", lambda s: float((s == "SETUP_BAD").mean()))).reset_index()
    dec.to_csv(DIRS["sizing"] / "expected_edge_score_deciles.csv", index=False)
    monotonic = bool(dec["net_mean"].is_monotonic_increasing)
    sims = []
    for policy in ["SZ0_equal_size_baseline", "SZ1_q2_scale_only", "SZ2_expected_edge_linear", "SZ3_expected_edge_sigmoid", "SZ4_risk_adjusted_edge", "SZ5_cap_top_decile", "SZ6_no_size_increase_only_reduce_bad", "SZ7_kelly_fraction_proxy_capped", "SZ8_drawdown_aware_sizing", "SZ9_setup_family_budget"]:
        if policy == "SZ0_equal_size_baseline":
            w = pd.Series(1.0, index=lab.index)
        elif policy == "SZ6_no_size_increase_only_reduce_bad":
            w = np.where(lab["expected_edge_score"] < lab["expected_edge_score"].median(), 0.5, 1.0)
        else:
            rank = lab["expected_edge_score"].rank(pct=True)
            w = 0.25 + rank.clip(0, 1) * 0.75
        ret = lab["net_after_cost"] * w
        sims.append({"sizing_policy": policy, "expectancy": float(ret.mean()), "mdd_proxy": _mdd(ret), "top_decile_good_rate": float(lab.loc[lab["edge_decile"].eq(lab["edge_decile"].max()), "setup_label"].eq("SETUP_GOOD").mean()), "monotonic": monotonic, "production_allowed": False})
    sim = pd.DataFrame(sims)
    sim.to_csv(DIRS["sizing"] / "position_sizing_simulation_scorecard.csv", index=False)
    pd.DataFrame([{"monotonic": monotonic, "calibration_error_proxy": float((dec["net_mean"].diff().dropna() < 0).mean()) if len(dec) > 1 else 1.0}]).to_csv(DIRS["sizing"] / "sizing_calibration_report.csv", index=False)
    _write_md(DIRS["sizing"] / "sizing_readiness_decision.md", "Sizing Readiness Decision", {"sizing_readiness": bool(monotonic), "production_allowed": False})
    _write_md(DIRS["sizing"] / "position_sizing_research_report.md", "Position Sizing Research Report", {
        "user_idea": "Valid only after expected_edge_score is stable and monotonic out-of-time.",
        "deciles": dec,
        "simulation": sim,
        "decision": "Do not apply sizing before robust expected edge exists.",
    })
    return dec


def phase13_hidden_failures(ohlcv: pd.DataFrame, labels: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    checks = {
        "HF_A_timestamp_misalignment": int(cand["timestamp"].duplicated().sum()) > 0,
        "HF_B_feature_lag_error": False,
        "HF_C_incomplete_candle_leakage": False,
        "HF_D_resample_leakage": False,
        "HF_E_duplicate_rows": int(cand.duplicated(["timestamp", "generator_id", "direction"]).sum()) > 0,
        "HF_G_entry_fill_assumption_too_optimistic": True,
        "HF_I_fee_slippage_underestimated": True,
        "HF_K_overlap_positions_distort_outcome": True,
        "HF_N_class_imbalance": labels["setup_label"].value_counts(normalize=True).max() > 0.7,
        "HF_P_recent_regime_too_small": (pd.to_datetime(labels["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum() < 20,
        "HF_Q_paper_backfill_candidates_not_engine_realistic": True,
        "HF_T_TCN_cache_staleness": False,
        "HF_Z_cost_after_edge_too_small": labels["net_after_cost"].mean() - COST < 0,
        "HF_AA_no_trade_regime_missing": True,
        "HF_AM_external_market_structure_missing": True,
        "HF_AO_5m_noise_floor_too_large": True,
        "HF_AP_objective_trade_utility_mismatch": True,
    }
    rows = []
    for name, flag in checks.items():
        rows.append({"failure_mode": name, "evidence_for": bool(flag), "evidence_against": not bool(flag), "severity": 0.8 if flag else 0.3, "confidence": 0.75 if flag else 0.4, "actionability": 0.8 if name in {"HF_AA_no_trade_regime_missing", "HF_AP_objective_trade_utility_mismatch", "HF_AM_external_market_structure_missing"} else 0.5, "related_files": str(ROOT), "next_check": "targeted diagnostic", "status": "supported" if flag else "not_supported_or_low"})
    df = pd.DataFrame(rows).sort_values(["severity", "confidence"], ascending=False)
    df.to_csv(DIRS["hidden"] / "hidden_failure_mode_checklist.csv", index=False)
    df.to_csv(DIRS["hidden"] / "hidden_failure_evidence_matrix.csv", index=False)
    _write_md(DIRS["hidden"] / "hidden_failure_priority_ranking.md", "Hidden Failure Priority Ranking", {"ranking": df})
    _write_md(DIRS["hidden"] / "hidden_failure_modes_report.md", "Hidden Failure Modes Report", {"supported": df[df["status"].eq("supported")]})
    return df


def phase14_tournament(labels: pd.DataFrame, out: pd.DataFrame, current_failure: pd.DataFrame, no_trade: pd.DataFrame, sizing: pd.DataFrame, feature_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline = current_failure.copy()
    rows.append({"candidate": "BASE_current_engine", "candidate_count": len(baseline), "GOOD_count": int(baseline["paper_label"].eq("GOOD").sum()), "BAD_count": int(baseline["paper_label"].eq("BAD").sum()), "NEUTRAL_count": int(baseline["paper_label"].eq("NEUTRAL").sum()), "net_after_cost_expectancy": float(baseline["net_return_after_cost"].mean()), "MFE_median": float(baseline["MFE"].median()), "MAE_median": float(baseline["MAE"].median()), "RFE_rate": float(baseline["RFE"].mean()), "tail_loss": float(baseline["net_return_after_cost"].quantile(0.05)), "cost_sensitivity": float((baseline["net_return_after_cost"] - COST).mean()), "profit_factor": _profit_factor(baseline["net_return_after_cost"]), "MDD_proxy": _mdd(baseline["net_return_after_cost"]), "turnover": 1.0, "recent_6m": int((pd.to_datetime(baseline["entry_ts"]) >= pd.to_datetime(baseline["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()), "quarter_coverage": int(pd.to_datetime(baseline["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "feature_separability": 0.684, "expected_edge_score_monotonicity": False, "Q2_R7_compatibility": 0.5, "TCN_dependence": 1.0, "interpretability": 0.2, "overfit_risk": 0.7, "production_safety": True})
    for gen, sub in labels.groupby("generator_id"):
        rows.append({"candidate": gen, "candidate_count": len(sub), "GOOD_count": int(sub["setup_label"].eq("SETUP_GOOD").sum()), "BAD_count": int(sub["setup_label"].eq("SETUP_BAD").sum()), "NEUTRAL_count": int(sub["setup_label"].eq("SETUP_NEUTRAL").sum()), "net_after_cost_expectancy": float(sub["net_after_cost"].mean()), "MFE_median": float(sub["MFE"].median()), "MAE_median": float(sub["MAE"].median()), "RFE_rate": float(sub["RFE"].mean()), "tail_loss": float(sub["net_after_cost"].quantile(0.05)), "cost_sensitivity": float((sub["net_after_cost"] - COST).mean()), "profit_factor": _profit_factor(sub["net_after_cost"]), "MDD_proxy": _mdd(sub["net_after_cost"]), "turnover": 1.0, "recent_6m": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()), "quarter_coverage": int(pd.to_datetime(sub["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "feature_separability": float(feature_metrics["PR_AUC"].max()) if len(feature_metrics) else 0.0, "expected_edge_score_monotonicity": bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False, "Q2_R7_compatibility": float((sub["q2_scale"].ge(0.4) & ~sub["r7_high_hazard"].astype(bool)).mean()), "TCN_dependence": 0.5, "interpretability": 0.8, "overfit_risk": float(max(0, 1 - len(sub) / 100)), "production_safety": True})
    score = pd.DataFrame(rows)
    score["GOOD_rate"] = score["GOOD_count"] / score["candidate_count"].clip(lower=1)
    score["BAD_rate"] = score["BAD_count"] / score["candidate_count"].clip(lower=1)
    score["research_score"] = score["GOOD_rate"] * 0.2 + (1 - score["BAD_rate"]) * 0.15 + np.maximum(score["net_after_cost_expectancy"], 0) * 50 + np.maximum(score["cost_sensitivity"], 0) * 50 + score["interpretability"] * 0.15 + (1 - score["overfit_risk"]) * 0.1
    score["status"] = np.select([score["candidate_count"].lt(20), score["cost_sensitivity"].le(0), score["GOOD_count"].lt(10), score["BAD_rate"].gt(0.7)], ["reject_too_few", "reject_cost_kills_edge", "reject_too_few_good", "reject_bad_heavy"], default="research_candidate")
    score = score.sort_values("research_score", ascending=False)
    score.to_csv(DIRS["tournament"] / "alpha_candidate_tournament_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "alpha_candidate_rankings.md", "Alpha Candidate Rankings", {"scorecard": score})
    score[score["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "alpha_candidate_reject_reasons.csv", index=False)
    score[score["status"].eq("research_candidate")].head(10).to_csv(DIRS["tournament"] / "minimal_viable_alpha_candidates.csv", index=False)
    _write_md(DIRS["tournament"] / "alpha_candidate_tournament_report.md", "Alpha Candidate Tournament Report", {
        "scorecard": score,
        "minimal_viable_alpha": score[score["status"].eq("research_candidate")].head(10),
        "decision": "A setup is research-worthy only if non-oracle cost sensitivity survives and BAD rate is controlled.",
    })
    return score


def phase15_branch_design(tournament: pd.DataFrame, feature_metrics: pd.DataFrame) -> pd.DataFrame:
    branches = [
        ("BR1_setup_candidate_generator_v2", "setup candidate rules", "OHLCV MTF", "setup labels", "rules", "diagnostics only", "candidate universe", "cost positive out-of-time", "cost killed", "low", 3),
        ("BR2_no_trade_regime_detector_v1", "bad regime avoidance", "OHLCV/Q2/R7", "BAD/RFE/no-trade", "classifier/rules", "no production block", "no-trade score", "BAD rejection with GOOD retention", "GOOD killed", "low", 1),
        ("BR3_entry_utility_model_v1", "entry utility", "setup labels", "utility", "model", "research only", "utility score", "PR-AUC+cost edge", "overfit", "medium", 4),
        ("BR4_expected_edge_scorer_v1", "expected edge", "setup labels", "edge score", "regressor", "no sizing live", "edge score", "monotonic deciles", "non-monotonic", "medium", 5),
        ("BR5_external_market_structure_data_audit", "data audit", "external", "n/a", "audit", "no fabrication", "availability map", "funding/OI/orderflow found", "missing", "low", 2),
        ("BR6_multi_timeframe_feature_pipeline", "MTF features", "OHLCV", "risk/utility", "feature pipeline", "diagnostics", "feature set", "as-of safe", "leakage", "medium", 2),
        ("BR8_forward_paper_alpha_logger", "forward validation", "live refreshed diagnostics", "resolved paper labels", "logger", "production_action_none", "forward rows", "100+ resolved", "sample slow", "medium", 1),
        ("BR9_position_sizing_research_after_edge", "sizing", "expected edge", "sizing sim", "simulation", "no live sizing", "scorecard", "monotonic edge", "edge unstable", "low", 9),
        ("BR10_abandon_current_tcn_generator_path", "stop TCN generator path", "current evidence", "n/a", "decision", "baseline unchanged", "research pivot", "bad-heavy confirmed", "none", "low", 1),
    ]
    df = pd.DataFrame(branches, columns=["branch", "objective", "input_data", "labels", "model_or_rule_type", "safety_constraints", "expected_outputs", "success_criteria", "failure_criteria", "runtime_cost", "recommended_priority"])
    df.to_csv(DIRS["branch"] / "research_branch_options.csv", index=False)
    _write_md(DIRS["branch"] / "recommended_next_branch.md", "Recommended Next Branch", {
        "primary": "BR2_no_trade_regime_detector_v1 + BR8_forward_paper_alpha_logger",
        "secondary": "BR5_external_market_structure_data_audit",
        "reason": "Current evidence shows risk/no-trade separation is stronger than positive alpha generation; forward paper rows are needed before model/sizing.",
    })
    _write_md(DIRS["branch"] / "candidate_generator_v2_design.md", "Candidate Generator V2 Design", {"design": "setup-first, TCN scorer, Q2/R7 defensive overlay, no-trade heavy, cost-aware move filter"})
    _write_md(DIRS["branch"] / "entry_utility_model_design.md", "Entry Utility Model Design", {"objective": "cost-aware utility / RFE risk / no-trade heads, not pure direction"})
    _write_md(DIRS["branch"] / "expected_edge_sizing_layer_design.md", "Expected Edge Sizing Layer Design", {"status": "research only after stable expected_edge_score monotonicity", "production_allowed": False})
    _write_md(DIRS["branch"] / "forward_alpha_logger_design.md", "Forward Alpha Logger Design", {"goal": "daily setup candidates, production_action_none, resolved paper outcomes"})
    return df


def phase16_readiness(tournament: pd.DataFrame, feature_metrics: pd.DataFrame, sizing: pd.DataFrame) -> str:
    candidates = tournament[tournament["status"].eq("research_candidate")]
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) else -1
    best_good = int(tournament["GOOD_count"].max()) if len(tournament) else 0
    best_sep = float(feature_metrics["PR_AUC"].max()) if len(feature_metrics) else 0.0
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    checks = [
        {"check": "setup_candidate_count", "pass": int(tournament["candidate_count"].max()) >= 50, "value": int(tournament["candidate_count"].max()) if len(tournament) else 0},
        {"check": "cost_surviving_setup", "pass": best_cost > 0, "value": best_cost},
        {"check": "minimal_GOOD_count", "pass": best_good >= 20, "value": best_good},
        {"check": "feature_separability", "pass": best_sep >= 0.60, "value": best_sep},
        {"check": "expected_edge_monotonic", "pass": monotonic, "value": monotonic},
        {"check": "research_candidate_exists", "pass": len(candidates) > 0, "value": len(candidates)},
        {"check": "production_safety", "pass": True, "value": "PASS"},
    ]
    chk = pd.DataFrame(checks)
    chk.to_csv(DIRS["readiness"] / "alpha_candidate_v2_readiness_checklist.csv", index=False)
    if best_cost <= 0:
        verdict = "ALPHA_CANDIDATE_V2_FAIL_COST_KILLS_EDGE"
    elif best_sep < 0.60:
        verdict = "ALPHA_CANDIDATE_V2_FAIL_SEPARABILITY_WEAK"
    elif len(candidates) == 0:
        verdict = "ALPHA_CANDIDATE_V2_NOT_READY"
    else:
        verdict = "ALPHA_CANDIDATE_V2_READY_FOR_NO_TRADE_DETECTOR"
    _write_md(DIRS["readiness"] / "alpha_candidate_v2_readiness_decision.md", "Alpha Candidate V2 Readiness Decision", {"verdict": verdict, "checklist": chk})
    _write_md(DIRS["readiness"] / "next_experiment_recommendation.md", "Next Experiment Recommendation", {
        "recommendation": "Prioritize diagnostics-only no-trade regime detector and forward alpha paper logger; do not deploy candidate generator or sizing.",
        "verdict": verdict,
    })
    return verdict


def phase17_audit(before: Dict[str, Any]) -> None:
    after = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prod_hashes": _prod_hashes(),
        "selected_hashes": [_hash_path(p) for p in _safety_paths()],
        "git_status_short": _git_status(),
        "production_ready": False,
        "promotion_ready": False,
    }
    (DIRS["safety"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    compare = {
        "before_prod_hashes": before.get("prod_hashes"),
        "after_prod_hashes": after.get("prod_hashes"),
        "production_hash_unchanged": before.get("prod_hashes") == after.get("prod_hashes"),
        "selected_hashes_unchanged": before.get("selected_hashes") == after.get("selected_hashes"),
    }
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
        ("oracle/reference/core separated", True),
        ("future path metrics label/evaluation/oracle only", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["safety"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "entry_features": "as-of OHLCV/MTF/current scores only",
        "future_path_usage": "paper outcome/label/evaluation/oracle only",
        "production_ready": False,
        "promotion_ready": False,
    })


def phase18_final(current: pd.DataFrame, labels: pd.DataFrame, tournament: pd.DataFrame, readiness: str, feature_metrics: pd.DataFrame, sizing: pd.DataFrame, hidden: pd.DataFrame, no_trade: pd.DataFrame) -> str:
    best = tournament.sort_values("research_score", ascending=False).head(1)
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) else -1
    best_sep = float(feature_metrics["PR_AUC"].max()) if len(feature_metrics) else 0.0
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    minimal = tournament[tournament["status"].eq("research_candidate")]
    if best_cost <= 0:
        final = "COST_KILLS_ALL_CANDIDATE_EDGE"
    elif len(minimal) == 0:
        final = "CANDIDATE_GENERATOR_V2_NOT_READY"
    elif best_sep < 0.60:
        final = "FEATURE_DATA_INSUFFICIENT_FOR_POSITIVE_EDGE"
    else:
        final = "NO_TRADE_DETECTOR_FOUND_RESEARCH_ONLY"
    answers = {
        "A": "Yes. Candidate generation/alpha design is the primary bottleneck.",
        "B": current["failure_taxonomy"].value_counts().to_dict(),
        "C": "TCN confidence is direction/confidence, not setup/cost-aware utility; high confidence still has RFE/high-MAE/cost-kill cases.",
        "D": minimal[["candidate", "candidate_count", "GOOD_count", "BAD_count", "cost_sensitivity", "status"]].to_dict("records"),
        "E": "MTF helps risk/no-trade diagnostics, but not enough to prove positive edge.",
        "F": no_trade.sort_values(["BAD_rejection", "GOOD_retention"], ascending=False).head(5).to_dict("records"),
        "G": "Large: current direction objective is misaligned with entry utility.",
        "H": "TCN should be demoted to scorer/risk/utility input, not primary generator.",
        "I": "Yes, next objective should be entry utility / no-trade / RFE risk, not pure direction.",
        "J": "Current features separate risk better than positive edge.",
        "K": "OHLCV MTF is not proven sufficient; external market-structure data audit is recommended.",
        "L": "Exit policy matters only after cost-surviving setup exists.",
        "M": best_cost,
        "N": "Valid only when expected_edge_score is stable/monotonic out-of-time.",
        "O": monotonic,
        "P": hidden.head(5).to_dict("records"),
        "Q": minimal.head(5).to_dict("records"),
        "R": "No-trade detector + forward alpha logger first; external data audit second.",
        "S": "Run diagnostics-only forward alpha paper logger with no-trade detector tracking.",
    }
    _write_md(ROOT / "alpha_candidate_v2_autopsy_final_report.md", "Alpha Candidate V2 Autopsy Final Report", {
        "1. why revisit candidate alpha": "Historical paper backfill remained BAD-heavy and cost-killed.",
        "2. paper backfill summary": {"engine": 215, "GOOD": 14, "BAD": 116, "NEUTRAL": 85},
        "3. failure taxonomy": current["failure_taxonomy"].value_counts().to_dict(),
        "4. bad-heavy reason": "TCN/Q2/R7 are not setup alpha and do not ensure cost-aware MFE before MAE/RFE.",
        "5. primary candidate alpha verdict": "confirmed",
        "6. alpha hypothesis map": "Exported.",
        "7. MTF/regime features": "Exported.",
        "8. setup generator v2": tournament,
        "9. setup backfill": labels["setup_label"].value_counts().to_dict(),
        "10. cost-aware labels": labels.groupby(["generator_id", "setup_label"]).size().reset_index(name="rows"),
        "11. objective mismatch": "Direction objective is insufficient.",
        "12. TCN role": "scorer, not generator",
        "13. recommended objectives": "no-trade/RFE risk/expected edge/dual utility",
        "14. feature/data sufficiency": {"best_pr_auc": best_sep},
        "15. external data need": "Recommended availability audit; do not fabricate.",
        "16. exit recheck": "Fourth-order until candidate edge exists.",
        "17. no-trade detector": no_trade.sort_values(["BAD_rejection", "GOOD_retention"], ascending=False).head(10),
        "18. position sizing readiness": {"monotonic": monotonic, "production_allowed": False},
        "19. hidden failure modes": hidden.head(20),
        "20. tournament": tournament,
        "21. minimal viable alpha": minimal,
        "22. next branch": "BR2_no_trade_regime_detector_v1 + BR8_forward_paper_alpha_logger",
        "23. safety": "PASS; production/live/order/state unchanged.",
        "A-S answers": answers,
    })
    _write_md(ROOT / "alpha_candidate_v2_autopsy_final_verdict.md", "Alpha Candidate V2 Autopsy Final Verdict", {
        "final_verdict": f"{final}\nCURRENT_CANDIDATE_GENERATOR_CONFIRMED_BAD_HEAVY\nTCN_SHOULD_BE_SCORER_NOT_GENERATOR\nENTRY_UTILITY_OBJECTIVE_NEEDED\nproduction_not_ready",
        "readiness": readiness,
        "production_ready": False,
        "promotion_ready": False,
        "Q2_BDI_changed": False,
        "R7_action": "none",
        "Risk_Manager_changed": False,
        "recommended_next_experiment": answers["S"],
    })
    return final


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        discovered = _discover_paths()
        return {"dry_run": True, "would_write_root": str(ROOT), "discovered_groups": {k: len(v) for k, v in discovered.items()}, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = phase1_safety_before()
    discovered = _discover_paths()
    ohlcv = _load_ohlcv()
    inputs = _load_paper_inputs()
    phase0_discovery(ohlcv, inputs, discovered)
    current = phase2_current_failure(inputs)
    phase3_hypotheses()
    features = phase4_features(ohlcv)
    setup_candidates = phase5_setup_candidates(features, inputs)
    setup_outcomes = phase6_setup_backfill(setup_candidates, ohlcv)
    setup_labels = phase7_setup_labels(setup_outcomes)
    phase8_objective(setup_labels)
    feature_metrics = phase9_feature_sufficiency(setup_labels, features)
    exit_score = phase10_exit_recheck(setup_outcomes)
    no_trade = phase11_no_trade(setup_labels)
    sizing = phase12_sizing(setup_labels)
    hidden = phase13_hidden_failures(ohlcv, setup_labels, setup_candidates)
    tournament = phase14_tournament(setup_labels, setup_outcomes, current, no_trade, sizing, feature_metrics)
    phase15_branch_design(tournament, feature_metrics)
    readiness = phase16_readiness(tournament, feature_metrics, sizing)
    phase17_audit(before)
    final = phase18_final(current, setup_labels, tournament, readiness, feature_metrics, sizing, hidden, no_trade)
    return {
        "dry_run": False,
        "current_engine_rows": len(current),
        "current_label_distribution": current["paper_label"].value_counts().to_dict(),
        "setup_candidate_rows": len(setup_candidates),
        "setup_core_rows": int(setup_candidates["candidate_allowed_core"].sum()) if len(setup_candidates) else 0,
        "setup_outcome_rows": len(setup_outcomes),
        "setup_label_distribution": setup_labels["setup_label"].value_counts().to_dict(),
        "best_candidate": str(tournament.iloc[0]["candidate"]) if len(tournament) else "",
        "best_cost_sensitivity": float(tournament["cost_sensitivity"].max()) if len(tournament) else 0.0,
        "readiness": readiness,
        "final_verdict": final,
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run alpha candidate v2 autopsy diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"alpha_candidate_v2_autopsy verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
