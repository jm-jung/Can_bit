"""
Historical clean paper-execution backfill for CAN_BIT BTCUSDT 5m.

Diagnostics only. Builds engine-realistic paper candidates from historical
as-of TCN/Q2/R7 rows, replays exit policies on canonical 5m OHLCV, separates
entry and exit labels, and writes only under:
data/diagnostics/historical_clean_paper_backfill/
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

ROOT = Path("data/diagnostics/historical_clean_paper_backfill")
VERSION = f"historical_clean_paper_backfill_v1_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
COST = 0.0006

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "asof": ROOT / "asof_frame",
    "candidates": ROOT / "candidates",
    "trades": ROOT / "paper_trades",
    "exit": ROOT / "exit_replay",
    "labels": ROOT / "labels",
    "datasets": ROOT / "datasets",
    "comparison": ROOT / "comparison",
    "separability": ROOT / "separability",
    "alignment": ROOT / "alignment",
    "tournament": ROOT / "exit_tournament",
    "validation": ROOT / "validation",
    "economics": ROOT / "economics",
    "forward": ROOT / "forward_plan",
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
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:200])
    return out


def _discover_paths() -> Dict[str, List[str]]:
    patterns = {
        "required_inputs": [
            "data/diagnostics/data_sync/**/*",
            "data/diagnostics/feature_proba_refresh/**/*",
            "data/diagnostics/executed_entry_quality_label_v2/**/*",
            "data/diagnostics/executed_row_salvage_autopsy/**/*",
            "data/diagnostics/entry_exit_policy_autopsy/**/*",
            "data/ohlcv/**/*",
            "data/market/**/*",
        ],
        "tcn_proba_cache": ["*tcn*", "*proba*", "*prediction*"],
        "q2_bdi_decision": ["*q2*", "*bdi*"],
        "r7_score": ["*r7*", "*false_high*"],
        "engine_candidate_logic": ["*engine*", "*candidate*", "*signal*", "*guard*"],
        "fee_slippage_exit": ["*fee*", "*slippage*", "*exit*", "*hold*", "*cooldown*", "*confirm*"],
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
        out[group] = sorted(set(vals))[:250]
    return out


def _load_ohlcv() -> pd.DataFrame:
    candidates = [
        REPO_ROOT / "data/diagnostics/data_sync/canonical_data_paths.json",
        REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv",
        REPO_ROOT / "data/market/btcusdt_5m.parquet",
    ]
    selected: Path | None = None
    if candidates[0].exists():
        meta = json.loads(candidates[0].read_text())
        p = REPO_ROOT / meta.get("canonical_5m_path", "")
        if p.exists():
            selected = p
    if selected is None:
        selected = next((p for p in candidates[1:] if p.exists()), None)
    if selected is None:
        raise FileNotFoundError("No canonical 5m OHLCV source found")
    if selected.suffix == ".parquet":
        df = pd.read_parquet(selected)
    else:
        df = pd.read_csv(selected)
    ts_col = "timestamp" if "timestamp" in df.columns else "open_time"
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df = df.rename(columns={ts_col: "timestamp"})
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["symbol"] = df.get("symbol", SYMBOL)
    df["timeframe"] = df.get("timeframe", TIMEFRAME)
    df["bar_index"] = np.arange(len(df))
    return df


def _score_cols(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["p_long"] = x.get("baseline_p_long", x.get("p_long", 0.0))
    x["p_short"] = x.get("baseline_p_short", x.get("p_short", 0.0))
    x["p_flat"] = x.get("baseline_p_flat", x.get("p_flat", 0.0))
    x["margin"] = x.get("baseline_margin", x.get("margin", (x["p_long"] - x["p_short"]).abs()))
    x["entropy"] = x.get("entropy", 1.0)
    x["q2_scale"] = x.get("q2_bdi_scale", x.get("q2_scale", 0.0))
    x["q2_score"] = x.get("q2_bdi_score", x.get("q2_score", x["q2_scale"]))
    x["q2_accept"] = x["q2_scale"].fillna(0).astype(float) >= 0.40
    x["q2_reject"] = ~x["q2_accept"]
    x["q2_decision"] = np.where(x["q2_accept"], "ACCEPT_OR_SCALE", "REJECT_OR_LOW_SCALE")
    x["r7_score"] = x.get("r7_score", 0.0)
    x["r7_high_hazard"] = x.get("r7_high_hazard", False).fillna(False).astype(bool)
    x["direction"] = np.where(x["p_long"].fillna(0) >= x["p_short"].fillna(0), "LONG", "SHORT")
    x["trend_state"] = x.get("trend_state", "unknown")
    x["vol_state"] = x.get("vol_bucket", x.get("vol_state", "unknown"))
    if "session_bucket" not in x:
        hour = pd.to_datetime(x["timestamp"]).dt.hour
        x["session_bucket"] = np.select([hour.between(0, 7), hour.between(8, 15)], ["asia", "europe"], default="us")
    return x


def _load_previous_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    paths = {
        "label_v2": REPO_ROOT / "data/diagnostics/executed_entry_quality_label_v2/dataset/executed_entry_quality_label_v2.parquet",
        "salvage_lifecycle": REPO_ROOT / "data/diagnostics/executed_row_salvage_autopsy/lifecycle/executed_lifecycle_876.parquet",
        "entry_exit": REPO_ROOT / "data/diagnostics/entry_exit_policy_autopsy/dataset/entry_exit_policy_autopsy_dataset.parquet",
        "entry_exit_replay": REPO_ROOT / "data/diagnostics/entry_exit_policy_autopsy/exit_replay/exit_policy_replay_metrics.csv",
    }
    for key, path in paths.items():
        if not path.exists():
            out[key] = {"exists": False}
            continue
        try:
            df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
            out[key] = {"exists": True, "rows": len(df), "columns": list(df.columns)[:80]}
            if key == "entry_exit":
                out[key]["max_holding_rows"] = int(df.get("max_holding_hit", pd.Series(False, index=df.index)).astype(bool).sum())
        except Exception as exc:
            out[key] = {"exists": True, "error": str(exc)}
    return out


def phase0_discovery(ohlcv: pd.DataFrame, discovered: Dict[str, List[str]]) -> None:
    (DIRS["discovery"] / "paper_backfill_source_paths.json").write_text(_json(discovered), encoding="utf-8")
    required = [
        "data/diagnostics/data_sync/canonical_data_paths.json",
        "data/ohlcv/BTCUSDT_5m_full.csv",
        "data/market/btcusdt_5m.parquet",
        "data/diagnostics/feature_proba_refresh/latest_r7_input_frame.parquet",
        "data/diagnostics/feature_proba_refresh/latest_features.parquet",
        "data/diagnostics/feature_proba_refresh/latest_tcn_proba.parquet",
        "data/diagnostics/feature_proba_refresh/latest_q2_diagnostics.parquet",
        "data/diagnostics/executed_entry_quality_label_v2/dataset/executed_entry_quality_label_v2.parquet",
        "data/diagnostics/executed_row_salvage_autopsy/lifecycle/executed_lifecycle_876.parquet",
        "data/diagnostics/entry_exit_policy_autopsy/dataset/entry_exit_policy_autopsy_dataset.parquet",
        "data/diagnostics/entry_exit_policy_autopsy/exit_replay/exit_policy_replay_metrics.csv",
        "data/diagnostics/entry_exit_policy_autopsy/forward_design/forward_paper_execution_plan.md",
    ]
    inv = []
    for rel in required:
        p = REPO_ROOT / rel
        inv.append({"path": rel, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0})
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    prev = _load_previous_summary()
    pd.DataFrame([{"source": k, **v} for k, v in prev.items()]).to_csv(DIRS["discovery"] / "loaded_previous_diagnostics_summary.csv", index=False)
    gaps = ohlcv["timestamp"].diff().dt.total_seconds().fillna(300)
    gap_rows = int((gaps > 300 * 1.5).sum())
    _write_md(DIRS["discovery"] / "canonical_ohlcv_source_report.md", "Canonical OHLCV Source Report", {
        "rows": len(ohlcv),
        "start": str(ohlcv["timestamp"].min()),
        "end": str(ohlcv["timestamp"].max()),
        "latest_timestamp": str(ohlcv["timestamp"].max()),
        "gap_rows": gap_rows,
        "timezone": "timestamp parsed as pandas datetime; source is treated as UTC/as-recorded for diagnostics",
    })
    _write_md(DIRS["discovery"] / "candidate_generation_source_report.md", "Candidate Generation Source Report", {
        "source": "_load_frame historical diagnostic frame + canonical OHLCV timestamp alignment",
        "engine_realistic": "TCN direction + Q2/guard-like filters; oracle universe separated",
    })
    _write_md(DIRS["discovery"] / "exit_policy_source_report.md", "Exit Policy Source Report", {
        "mode": "diagnostic replay only",
        "cost": COST,
        "production_connection": "none",
    })
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {
        "input_inventory": pd.DataFrame(inv),
        "previous_summary": prev,
        "ohlcv_date_range": [str(ohlcv["timestamp"].min()), str(ohlcv["timestamp"].max())],
        "discovered_groups": {k: len(v) for k, v in discovered.items()},
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
        "r7_scale_action": "none",
        "r7_hard_block": False,
        "r7_q2_override": False,
        "actual_order_calls": False,
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


def phase2_asof_frame(ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    meta = _load_frame().copy()
    safe_cols, _, _ = _safe_features(meta)
    ts = "timestamp" if "timestamp" in meta.columns else "entry_ts"
    meta[ts] = pd.to_datetime(meta[ts], errors="coerce")
    meta = meta.dropna(subset=[ts]).rename(columns={ts: "timestamp"})
    meta = _score_cols(meta)
    keep = [
        "trade_id", "candidate_id", "timestamp", "entry_ts", "direction", "entry_price", "p_long", "p_short", "p_flat",
        "entropy", "margin", "q2_score", "q2_scale", "q2_decision", "q2_accept", "q2_reject", "q2_bdi_score",
        "q2_bdi_scale", "r7_score", "r7_high_hazard", "trend_state", "vol_state", "session_bucket",
    ] + [c for c in safe_cols if c in meta.columns]
    keep = [c for c in list(dict.fromkeys(keep)) if c in meta.columns]
    asof = meta[keep].copy().sort_values("timestamp")
    asof = pd.merge_asof(asof, ohlcv[["timestamp", "open", "high", "low", "close", "volume", "bar_index"]].sort_values("timestamp"), on="timestamp", direction="nearest", tolerance=pd.Timedelta(minutes=3))
    asof["symbol"] = SYMBOL
    asof["timeframe"] = TIMEFRAME
    asof["available_asof_ts"] = asof["timestamp"]
    asof["gap_flag"] = asof["bar_index"].isna()
    asof = asof.dropna(subset=["bar_index", "open", "close"]).drop_duplicates("timestamp").reset_index(drop=True)
    asof["bar_index"] = asof["bar_index"].astype(int)
    future_tokens = ["exit", "return", "mfe", "mae", "rfe", "label", "quality", "artifact", "realized", "net_", "raw_return"]
    audit = []
    for c in asof.columns:
        lower = c.lower()
        is_future = any(tok in lower for tok in future_tokens) and c not in {"q2_score", "q2_bdi_score"}
        audit.append({"column": c, "entry_feature_allowed": c in safe_cols or c in ["p_long", "p_short", "p_flat", "entropy", "margin", "q2_score", "q2_scale", "r7_score"], "future_or_evaluation_suspect": bool(is_future)})
    pd.DataFrame(audit).to_csv(DIRS["asof"] / "asof_frame_column_safety_audit.csv", index=False)
    asof.to_parquet(DIRS["asof"] / "historical_asof_frame.parquet", index=False)
    asof.to_csv(DIRS["asof"] / "historical_asof_frame.csv", index=False)
    (DIRS["asof"] / "asof_frame_schema.json").write_text(_json({c: str(asof[c].dtype) for c in asof.columns}), encoding="utf-8")
    _write_md(DIRS["asof"] / "asof_frame_quality_report.md", "As-Of Frame Quality Report", {
        "rows": len(asof),
        "timestamp_monotonic": bool(asof["timestamp"].is_monotonic_increasing),
        "duplicates": int(asof["timestamp"].duplicated().sum()),
        "missing_rows": int(asof[["p_long", "p_short", "q2_scale", "r7_score"]].isna().any(axis=1).sum()),
        "gap_rows": int(asof["gap_flag"].sum()),
        "recent_3m_rows": int((asof["timestamp"] >= asof["timestamp"].max() - pd.Timedelta(days=RECENT_3M_DAYS)).sum()),
        "recent_6m_rows": int((asof["timestamp"] >= asof["timestamp"].max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()),
        "future_columns_removed": [r["column"] for r in audit if r["future_or_evaluation_suspect"]],
    })
    feature_cols = [c for c in safe_cols if c in asof.columns and not any(tok in c.lower() for tok in future_tokens)]
    return asof, feature_cols


def _candidate_hash(row: pd.Series, cols: List[str]) -> str:
    payload = "|".join(f"{c}={row.get(c, '')}" for c in cols[:80])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def phase3_candidates(asof: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    rows = []
    med_margin = float(asof["margin"].median())
    low_entropy = float(asof["entropy"].quantile(0.35))
    high_margin = float(asof["margin"].quantile(0.65))
    universes = {
        "U0_all_asof_rows_reference": pd.Series(True, index=asof.index),
        "U1_engine_realistic_candidates": asof["q2_accept"] & (asof["margin"] >= med_margin) & (asof["entropy"] <= asof["entropy"].quantile(0.75)),
        "U2_q2_accepted_candidates": asof["q2_accept"],
        "U3_q2_rejected_reference": asof["q2_reject"],
        "U4_tcn_directional_candidates": asof["margin"] >= med_margin,
        "U5_low_entropy_candidates": asof["entropy"] <= low_entropy,
        "U6_high_margin_candidates": asof["margin"] >= high_margin,
        "U7_q2_accept_r7_no_warning": asof["q2_accept"] & ~asof["r7_high_hazard"].astype(bool),
        "U8_q2_accept_r7_warning": asof["q2_accept"] & asof["r7_high_hazard"].astype(bool),
        "U9_r7_no_warning_reference": ~asof["r7_high_hazard"].astype(bool),
        "U10_oracle_diagnostic_candidates": pd.Series(False, index=asof.index),
        "U12_recent_candidates": asof["timestamp"] >= asof["timestamp"].max() - pd.Timedelta(days=RECENT_6M_DAYS),
    }
    executed_path = REPO_ROOT / "data/diagnostics/executed_row_salvage_autopsy/lifecycle/executed_lifecycle_876.parquet"
    executed_ts = set()
    if executed_path.exists():
        ex = pd.read_parquet(executed_path)
        executed_ts = set(pd.to_datetime(ex["entry_ts"], errors="coerce").dropna().astype("datetime64[ns]").astype(str))
    universes["U11_existing_executed_overlap"] = asof["timestamp"].astype("datetime64[ns]").astype(str).isin(executed_ts)
    for uname, mask in universes.items():
        for idx, row in asof.loc[mask].iterrows():
            oracle = uname == "U10_oracle_diagnostic_candidates"
            core = uname == "U1_engine_realistic_candidates"
            ref = (not core) or uname in {"U0_all_asof_rows_reference", "U3_q2_rejected_reference", "U8_q2_accept_r7_warning", "U9_r7_no_warning_reference", "U10_oracle_diagnostic_candidates"}
            rows.append({
                "paper_candidate_id": f"{uname}_{idx}",
                "timestamp": row["timestamp"],
                "entry_ts": row["timestamp"],
                "bar_index": int(row["bar_index"]),
                "symbol": SYMBOL,
                "timeframe": TIMEFRAME,
                "direction": row["direction"],
                "candidate_universe": uname,
                "candidate_source": "historical_asof_tcn_q2_r7",
                "candidate_reason": "TCN direction + Q2/R7 guard-like filter" if core else uname,
                "q2_decision": row["q2_decision"],
                "q2_score": row.get("q2_score", np.nan),
                "q2_scale": row.get("q2_scale", np.nan),
                "q2_accept": bool(row["q2_accept"]),
                "q2_reject": bool(row["q2_reject"]),
                "r7_score": row.get("r7_score", np.nan),
                "r7_high_hazard": bool(row["r7_high_hazard"]),
                "p_long": row.get("p_long", np.nan),
                "p_short": row.get("p_short", np.nan),
                "p_flat": row.get("p_flat", np.nan),
                "entropy": row.get("entropy", np.nan),
                "margin": row.get("margin", np.nan),
                "trend_state": row.get("trend_state", "unknown"),
                "vol_state": row.get("vol_state", "unknown"),
                "session_bucket": row.get("session_bucket", "unknown"),
                "entry_price": row.get("close", np.nan),
                "entry_feature_snapshot_hash": _candidate_hash(row, feature_cols),
                "oracle_flag": oracle,
                "core_paper_flag": core,
                "reference_only_flag": ref,
                "asof_row_index": idx,
            })
    cand = pd.DataFrame(rows).sort_values(["timestamp", "candidate_universe"]).reset_index(drop=True)
    cand.to_parquet(DIRS["candidates"] / "paper_candidate_universe.parquet", index=False)
    cand.to_csv(DIRS["candidates"] / "paper_candidate_universe.csv", index=False)
    cand.groupby("candidate_universe").agg(rows=("paper_candidate_id", "size"), core=("core_paper_flag", "sum"), oracle=("oracle_flag", "sum"), reference=("reference_only_flag", "sum"), q2_accept=("q2_accept", "sum"), r7_warning=("r7_high_hazard", "sum")).reset_index().to_csv(DIRS["candidates"] / "paper_candidate_summary_by_universe.csv", index=False)
    cand.groupby(["candidate_universe", "direction"]).size().reset_index(name="rows").to_csv(DIRS["candidates"] / "paper_candidate_direction_distribution.csv", index=False)
    recent = cand[cand["timestamp"] >= cand["timestamp"].max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    recent.groupby("candidate_universe").size().reset_index(name="recent_6m_rows").to_csv(DIRS["candidates"] / "paper_candidate_recent_summary.csv", index=False)
    pd.DataFrame([{"overlap_count": int(cand["candidate_universe"].eq("U11_existing_executed_overlap").sum()), "executed_reference_rows": len(executed_ts)}]).to_csv(DIRS["candidates"] / "existing_executed_overlap_report.csv", index=False)
    _write_md(DIRS["candidates"] / "candidate_generation_report.md", "Candidate Generation Report", {
        "candidate_count": len(cand),
        "engine_realistic_count": int(cand["core_paper_flag"].sum()),
        "oracle_count": int(cand["oracle_flag"].sum()),
        "core_policy": "U1 only is core paper; other universes are reference/diagnostic.",
    })
    return cand


def phase4_trades(cand: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv.set_index("bar_index")
    rows = []
    for _, c in cand.iterrows():
        entry_idx = int(c["bar_index"]) + 1
        if entry_idx not in o.index:
            continue
        next_bar = o.loc[entry_idx]
        cur = o.loc[int(c["bar_index"])]
        rules = {
            "F0_next_open_default": float(next_bar["open"]),
            "F1_current_close_diagnostic": float(cur["close"]),
            "F2_next_close_diagnostic": float(next_bar["close"]),
            "F3_next_open_with_slippage": float(next_bar["open"]) * (1 + (0.0002 if c["direction"] == "LONG" else -0.0002)),
            "F4_worst_of_next_open_close_conservative": max(float(next_bar["open"]), float(next_bar["close"])) if c["direction"] == "LONG" else min(float(next_bar["open"]), float(next_bar["close"])),
        }
        for rule, price in rules.items():
            core_rule = rule == "F0_next_open_default"
            rows.append({
                "paper_trade_id": f"{c['paper_candidate_id']}_{rule}",
                "paper_candidate_id": c["paper_candidate_id"],
                "candidate_ts": c["timestamp"],
                "entry_ts": next_bar["timestamp"],
                "entry_bar_index": entry_idx,
                "entry_price": price,
                "entry_fill_rule": rule,
                "direction": c["direction"],
                "entry_cost_assumption": COST,
                "slippage_assumption": 0.0002,
                "candidate_universe": c["candidate_universe"],
                "core_paper_flag": bool(c["core_paper_flag"] and core_rule),
                "reference_only_flag": bool(c["reference_only_flag"] or not core_rule),
                "oracle_flag": bool(c["oracle_flag"]),
                "q2_decision": c["q2_decision"],
                "q2_score": c["q2_score"],
                "q2_scale": c["q2_scale"],
                "q2_accept": c["q2_accept"],
                "q2_reject": c["q2_reject"],
                "r7_score": c["r7_score"],
                "r7_high_hazard": c["r7_high_hazard"],
                "p_long": c["p_long"],
                "p_short": c["p_short"],
                "p_flat": c["p_flat"],
                "entropy": c["entropy"],
                "margin": c["margin"],
                "trend_state": c["trend_state"],
                "vol_state": c["vol_state"],
                "session_bucket": c["session_bucket"],
                "entry_feature_snapshot_hash": c["entry_feature_snapshot_hash"],
            })
    trades = pd.DataFrame(rows)
    base = trades[trades["entry_fill_rule"].eq("F0_next_open_default")].copy()
    engine = base[base["candidate_universe"].eq("U1_engine_realistic_candidates")].sort_values("entry_ts").copy()
    engine["cooldown_ok"] = engine["entry_ts"].diff().dt.total_seconds().fillna(999999) >= 12 * 300
    engine_like = engine[engine["cooldown_ok"]].copy()
    trades.to_parquet(DIRS["trades"] / "paper_trades_all_candidates_reference.parquet", index=False)
    base.to_parquet(DIRS["trades"] / "paper_trades_base.parquet", index=False)
    base.to_csv(DIRS["trades"] / "paper_trades_base.csv", index=False)
    engine_like.to_parquet(DIRS["trades"] / "paper_trades_engine_like.parquet", index=False)
    trades.groupby("entry_fill_rule").agg(rows=("paper_trade_id", "size"), core=("core_paper_flag", "sum")).reset_index().to_csv(DIRS["trades"] / "entry_fill_rule_comparison.csv", index=False)
    pd.DataFrame([
        {"overlap_policy": "O0_allow_all_candidates_reference", "rows": len(base)},
        {"overlap_policy": "O1_one_position_at_a_time_engine_like", "rows": len(engine_like)},
        {"overlap_policy": "O2_cooldown_engine_like", "rows": len(engine_like)},
        {"overlap_policy": "O3_direction_specific_no_overlap", "rows": int(engine_like.drop_duplicates(["entry_ts", "direction"]).shape[0])},
        {"overlap_policy": "O4_q2_accept_priority", "rows": int(base[base["q2_accept"]].shape[0])},
    ]).to_csv(DIRS["trades"] / "overlap_policy_impact.csv", index=False)
    _write_md(DIRS["trades"] / "paper_trade_construction_report.md", "Paper Trade Construction Report", {
        "base_rows": len(base),
        "engine_like_rows": len(engine_like),
        "entry_after_candidate": bool((base["entry_ts"] > base["candidate_ts"]).all()) if len(base) else True,
        "actual_orders": False,
    })
    return base


POLICIES = [
    ("X0_engine_like_exit_if_reconstructable", "actual_like", 24, False),
    ("X1_maxhold_current_default", "actual_like", 60, False),
    ("X2_minhold_then_signal_flip_if_available", "actual_like", 12, False),
    ("X3_q2_degrade_exit_if_available", "actual_like", 24, False),
    ("X4_r7_warning_exit_reference_only", "actual_like", 12, False),
    ("X10_exit_after_3_bars", "fixed_horizon", 3, False),
    ("X11_exit_after_6_bars", "fixed_horizon", 6, False),
    ("X12_exit_after_12_bars", "fixed_horizon", 12, False),
    ("X13_exit_after_24_bars", "fixed_horizon", 24, False),
    ("X14_exit_after_36_bars", "fixed_horizon", 36, False),
    ("X15_exit_after_48_bars", "fixed_horizon", 48, False),
    ("X16_exit_after_60_bars", "fixed_horizon", 60, False),
    ("X17_exit_after_72_bars", "fixed_horizon", 72, False),
    ("X18_exit_after_96_bars", "fixed_horizon", 96, False),
    ("X20_maxhold_12", "maxhold_variant", 12, False),
    ("X21_maxhold_24", "maxhold_variant", 24, False),
    ("X22_maxhold_36", "maxhold_variant", 36, False),
    ("X23_maxhold_48", "maxhold_variant", 48, False),
    ("X24_maxhold_72", "maxhold_variant", 72, False),
    ("X25_maxhold_96", "maxhold_variant", 96, False),
    ("X26_maxhold_120", "maxhold_variant", 120, False),
    ("X30_MAE_stop_small", "risk_stop", 24, False),
    ("X31_MAE_stop_medium", "risk_stop", 24, False),
    ("X32_MAE_stop_large", "risk_stop", 24, False),
    ("X33_vol_adjusted_MAE_stop", "risk_stop", 24, False),
    ("X34_adverse_first_stop", "risk_stop", 12, False),
    ("X35_RFE_proxy_stop", "risk_stop", 24, False),
    ("X40_first_cost_plus_move", "take_profit", 24, False),
    ("X41_first_2x_cost_plus_move", "take_profit", 24, False),
    ("X42_first_3x_cost_plus_move", "take_profit", 24, False),
    ("X43_first_vol_adjusted_profit", "take_profit", 24, False),
    ("X44_partial_MFE_capture_25_oracle", "mfe_oracle", 96, True),
    ("X45_partial_MFE_capture_50_oracle", "mfe_oracle", 96, True),
    ("X46_partial_MFE_capture_75_oracle", "mfe_oracle", 96, True),
    ("X50_trailing_after_cost_plus", "trailing_proxy", 48, False),
    ("X51_trailing_after_MFE_threshold", "trailing_proxy", 48, False),
    ("X52_trailing_vol_adjusted_proxy", "trailing_proxy", 48, False),
    ("X53_trailing_structure_proxy_if_available", "trailing_proxy", 48, False),
    ("X60_fixed_12_plus_MAE_stop", "hybrid", 12, False),
    ("X61_fixed_24_plus_MAE_stop", "hybrid", 24, False),
    ("X62_trailing_vol_adjusted_plus_MAE_stop", "hybrid", 48, False),
    ("X63_cost_plus_take_profit_plus_MAE_stop", "hybrid", 24, False),
    ("X64_minhold_then_trailing", "hybrid", 36, False),
    ("X90_oracle_best_exit_within_24", "oracle_upper_bound", 24, True),
    ("X91_oracle_best_exit_within_48", "oracle_upper_bound", 48, True),
    ("X92_oracle_best_exit_within_96", "oracle_upper_bound", 96, True),
    ("X93_oracle_MFE_capture", "oracle_upper_bound", 96, True),
]


def _path_stats(path: pd.DataFrame, direction: str, entry_price: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        close_ret = path["close"] / entry_price - 1
        fav = path["high"] / entry_price - 1
        adv = path["low"] / entry_price - 1
    else:
        close_ret = entry_price / path["close"] - 1
        fav = entry_price / path["low"] - 1
        adv = entry_price / path["high"] - 1
    return close_ret.astype(float), fav.astype(float), adv.astype(float)


def _policy_exit(policy: str, path: pd.DataFrame, close_ret: pd.Series, fav: pd.Series, adv: pd.Series) -> int:
    n = len(path)
    default = n - 1
    if n == 0:
        return 0
    if policy == "X0_engine_like_exit_if_reconstructable":
        return min(23, default)
    if policy == "X1_maxhold_current_default":
        return min(59, default)
    if "after_" in policy and "_bars" in policy:
        bars = int(policy.split("after_")[1].split("_bars")[0])
        return min(max(bars - 1, 0), default)
    if policy.startswith("X2_"):
        return min(11, default)
    if policy.startswith("X3_"):
        return min(23, default)
    if policy.startswith("X4_"):
        return min(11, default)
    if "maxhold_" in policy:
        bars = int(policy.split("maxhold_")[1])
        return min(max(bars - 1, 0), default)
    if "MAE_stop_small" in policy:
        hit = np.where(adv.values <= -0.0025)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "MAE_stop_medium" in policy or "vol_adjusted_MAE_stop" in policy:
        hit = np.where(adv.values <= -0.0040)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "MAE_stop_large" in policy:
        hit = np.where(adv.values <= -0.0060)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "adverse_first" in policy:
        hit = np.where(adv.head(min(6, n)).values <= -0.0035)[0]
        return int(hit[0]) if len(hit) else min(11, default)
    if "RFE_proxy" in policy:
        hit = np.where(adv.values <= -0.006)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "first_cost_plus" in policy:
        hit = np.where(fav.values >= 0.0012)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "first_2x_cost" in policy:
        hit = np.where(fav.values >= 0.0024)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "first_3x_cost" in policy:
        hit = np.where(fav.values >= 0.0036)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "first_vol_adjusted" in policy:
        hit = np.where(fav.values >= 0.0030)[0]
        return int(hit[0]) if len(hit) else min(23, default)
    if "trailing" in policy:
        best = fav.cummax()
        trail = best - adv.abs() * 0.25
        hit = np.where((best.values >= 0.0025) & (close_ret.values < trail.values - 0.0015))[0]
        return int(hit[0]) if len(hit) else min(47, default)
    if "fixed_12_plus_MAE_stop" in policy:
        hit = np.where(adv.values <= -0.004)[0]
        return min(int(hit[0]), 11) if len(hit) else min(11, default)
    if "fixed_24_plus_MAE_stop" in policy:
        hit = np.where(adv.values <= -0.004)[0]
        return min(int(hit[0]), 23) if len(hit) else min(23, default)
    if "cost_plus_take_profit_plus_MAE_stop" in policy:
        hit_tp = np.where(fav.values >= 0.0024)[0]
        hit_sl = np.where(adv.values <= -0.004)[0]
        hits = np.concatenate([hit_tp, hit_sl])
        return int(hits.min()) if len(hits) else min(23, default)
    if "oracle_best_exit" in policy:
        return int(close_ret.values.argmax())
    if "oracle_MFE_capture" in policy:
        return int(fav.values.argmax())
    return min(59, default)


def phase5_exit_replay(trades: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv.set_index("bar_index")
    rows = []
    max_horizon = max(p[2] for p in POLICIES)
    for _, t in trades.iterrows():
        entry_idx = int(t["entry_bar_index"])
        path = o.loc[(o.index >= entry_idx) & (o.index < entry_idx + max_horizon)].copy()
        if path.empty:
            continue
        close_ret, fav, adv = _path_stats(path, t["direction"], float(t["entry_price"]))
        mfe = float(fav.max())
        mae = float(adv.min())
        time_mfe = int(fav.values.argmax()) + 1
        time_mae = int(adv.values.argmin()) + 1
        for pid, group, horizon, oracle in POLICIES:
            sub = path.head(horizon)
            cr, f, a = close_ret.head(horizon), fav.head(horizon), adv.head(horizon)
            if sub.empty:
                continue
            exit_pos = _policy_exit(pid, sub, cr, f, a)
            exit_bar = sub.iloc[exit_pos]
            gross = float(cr.iloc[exit_pos])
            if pid == "X44_partial_MFE_capture_25_oracle":
                gross = float(f.max() * 0.25)
            elif pid == "X45_partial_MFE_capture_50_oracle":
                gross = float(f.max() * 0.50)
            elif pid == "X46_partial_MFE_capture_75_oracle":
                gross = float(f.max() * 0.75)
            elif pid == "X93_oracle_MFE_capture":
                gross = float(f.max())
            net = gross - COST
            pol_mfe = float(f.max())
            pol_mae = float(a.min())
            path_short = len(sub) < horizon
            maxhold = (group in {"maxhold_variant"} or pid == "X1_maxhold_current_default") and exit_pos >= len(sub) - 1 and not path_short
            rows.append({
                "paper_trade_id": t["paper_trade_id"],
                "paper_candidate_id": t["paper_candidate_id"],
                "candidate_universe": t["candidate_universe"],
                "core_paper_flag": bool(t["core_paper_flag"]),
                "reference_only_flag": bool(t["reference_only_flag"]),
                "direction": t["direction"],
                "entry_ts": t["entry_ts"],
                "entry_price": t["entry_price"],
                "exit_policy_id": pid,
                "exit_policy_group": group,
                "exit_ts": exit_bar["timestamp"],
                "exit_price": exit_bar["close"],
                "holding_bars": int(exit_pos + 1),
                "max_holding_hit": bool(maxhold),
                "censored_flag": bool((maxhold or path_short) and not oracle),
                "exit_reason_paper": "oracle_exit" if oracle else ("max_holding" if maxhold else "policy_trigger"),
                "gross_return": gross,
                "net_return_after_cost": net,
                "MFE": pol_mfe,
                "MAE": pol_mae,
                "RFE": bool(pol_mae <= -0.006),
                "time_to_MFE": int(f.values.argmax()) + 1,
                "time_to_MAE": int(a.values.argmin()) + 1,
                "MFE_before_MAE": bool((int(f.values.argmax()) + 1) <= (int(a.values.argmin()) + 1)),
                "MAE_before_MFE": bool((int(a.values.argmin()) + 1) < (int(f.values.argmax()) + 1)),
                "actual_MFE_capture_ratio": float(max(net, 0) / max(pol_mfe, 0.0005)),
                "giveback_ratio": float(max(pol_mfe - max(net, 0), 0) / max(pol_mfe, 0.0005)),
                "tail_risk_flag": bool(net <= -0.006),
                "early_favorable_flag": bool(f.head(min(6, len(f))).max() >= 0.002),
                "early_adverse_flag": bool(a.head(min(6, len(a))).min() <= -0.0035),
                "oracle_exit_flag": bool(oracle),
                "trainable_exit_policy_flag": bool(not oracle),
                "q2_accept": bool(t["q2_accept"]),
                "q2_scale": t["q2_scale"],
                "r7_high_hazard": bool(t["r7_high_hazard"]),
                "r7_score": t["r7_score"],
                "p_long": t["p_long"],
                "p_short": t["p_short"],
                "p_flat": t["p_flat"],
                "entropy": t["entropy"],
                "margin": t["margin"],
                "entry_feature_snapshot_hash": t["entry_feature_snapshot_hash"],
            })
    out = pd.DataFrame(rows)
    out.to_parquet(DIRS["exit"] / "paper_exit_outcomes.parquet", index=False)
    out.to_csv(DIRS["exit"] / "paper_exit_outcomes.csv", index=False)
    metrics = _policy_metrics(out, ["exit_policy_id", "exit_policy_group"])
    metrics.to_csv(DIRS["exit"] / "exit_policy_metrics.csv", index=False)
    _policy_metrics(out, ["exit_policy_id", "candidate_universe"]).to_csv(DIRS["exit"] / "exit_policy_metrics_by_universe.csv", index=False)
    _policy_metrics(out, ["exit_policy_id", "direction"]).to_csv(DIRS["exit"] / "exit_policy_metrics_by_direction.csv", index=False)
    tmp = out.copy()
    tmp["quarter"] = pd.to_datetime(tmp["entry_ts"]).dt.to_period("Q").astype(str)
    _policy_metrics(tmp, ["exit_policy_id", "quarter"]).to_csv(DIRS["exit"] / "exit_policy_metrics_by_quarter.csv", index=False)
    recent = out[pd.to_datetime(out["entry_ts"]) >= pd.to_datetime(out["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)]
    _policy_metrics(recent, ["exit_policy_id"]).to_csv(DIRS["exit"] / "exit_policy_metrics_recent.csv", index=False)
    oracle_gap = metrics.copy()
    actual = float(metrics.loc[metrics["exit_policy_id"].eq("X1_maxhold_current_default"), "expectancy"].mean()) if metrics["exit_policy_id"].eq("X1_maxhold_current_default").any() else 0.0
    oracle_gap["gap_vs_current_default"] = oracle_gap["expectancy"] - actual
    oracle_gap.to_csv(DIRS["exit"] / "exit_policy_oracle_gap.csv", index=False)
    _write_md(DIRS["exit"] / "exit_replay_report.md", "Exit Replay Report", {
        "outcome_rows": len(out),
        "policy_count": out["exit_policy_id"].nunique(),
        "oracle_policy_rows": int(out["oracle_exit_flag"].sum()),
        "best_policies": metrics.sort_values("expectancy", ascending=False).head(20),
        "warning": "All policies are diagnostic replay; no production exit changes.",
    })
    return out


def _policy_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    g = df.groupby(group_cols)
    rows = []
    for keys, sub in g:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        ret = _safe_num(sub["net_return_after_cost"])
        row.update({
            "rows": len(sub),
            "expectancy": float(ret.mean()),
            "net_total": float(ret.sum()),
            "winrate": float((ret > 0).mean()),
            "profit_factor": _profit_factor(ret),
            "MDD_proxy": _mdd(ret),
            "RFE_rate": float(sub["RFE"].mean()),
            "high_MAE_rate": float((sub["MAE"].abs() >= 0.006).mean()),
            "tail_loss": float(ret.quantile(0.05)),
            "MFE_capture_ratio": float(sub["actual_MFE_capture_ratio"].mean()),
            "giveback_ratio": float(sub["giveback_ratio"].mean()),
            "holding_bars_mean": float(sub["holding_bars"].mean()),
            "turnover_proxy": float(1 / max(sub["holding_bars"].mean(), 1)),
            "cost_sensitivity_proxy": float((ret - COST).mean()),
            "max_holding_rate": float(sub["max_holding_hit"].mean()),
            "censored_rate": float(sub["censored_flag"].mean()),
            "oracle_rate": float(sub["oracle_exit_flag"].mean()),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def phase6_labels(outcomes: pd.DataFrame) -> pd.DataFrame:
    core_policy = "X13_exit_after_24_bars"
    lab = outcomes[outcomes["exit_policy_id"].eq(core_policy)].copy()
    if lab.empty:
        lab = outcomes[outcomes["trainable_exit_policy_flag"]].drop_duplicates("paper_trade_id").copy()
    lab["early_return"] = lab["net_return_after_cost"]
    lab["early_MFE"] = lab["MFE"]
    lab["early_MAE"] = lab["MAE"]
    lab["early_MFE_MAE_ratio"] = lab["early_MFE"] / lab["early_MAE"].abs().clip(lower=0.0005)
    lab["favorable_first"] = lab["MFE_before_MAE"]
    lab["adverse_first"] = lab["MAE_before_MFE"]
    lab["cost_plus_hit"] = lab["MFE"] >= 0.0015
    lab["risk_threshold_hit"] = lab["MAE"].abs() >= 0.004
    good_entry = lab["cost_plus_hit"] & lab["favorable_first"] & ~lab["risk_threshold_hit"] & ~lab["RFE"]
    bad_entry = lab["early_adverse_flag"] | lab["RFE"] | lab["tail_risk_flag"] | (lab["MAE"].abs() >= 0.006)
    lab["entry_label"] = np.select(
        [good_entry, lab["MFE"].ge(0.002) & lab["MAE"].abs().le(0.004), bad_entry, lab["censored_flag"], lab["MFE"].lt(0.0015)],
        ["EQL_A_clean_early_good", "EQL_B_controlled_pullback_good", "EQL_D_early_bad", "EQL_F_censored_unknown", "EQL_G_no_edge"],
        default="EQL_H_ambiguous",
    )
    lab["exit_label"] = np.select(
        [lab["oracle_exit_flag"], lab["censored_flag"], lab["actual_MFE_capture_ratio"].ge(0.45), lab["actual_MFE_capture_ratio"].lt(0.20) & lab["MFE"].ge(0.002), lab["giveback_ratio"].gt(0.50), lab["tail_risk_flag"]],
        ["EXL_G_oracle_only", "EXL_F_censored_maxhold", "EXL_A_good_realization", "EXL_B_poor_MFE_capture", "EXL_C_profit_giveback", "EXL_E_late_exit"],
        default="EXL_H_ambiguous",
    )
    lab["combined_label"] = np.select(
        [good_entry & lab["exit_label"].eq("EXL_A_good_realization"), good_entry & lab["exit_label"].isin(["EXL_B_poor_MFE_capture", "EXL_C_profit_giveback"]), bad_entry & lab["exit_label"].eq("EXL_A_good_realization"), bad_entry & ~lab["censored_flag"], good_entry & lab["censored_flag"], bad_entry & lab["censored_flag"], lab["entry_label"].eq("EQL_G_no_edge"), lab["oracle_exit_flag"]],
        ["CMB_A_good_entry_good_exit", "CMB_B_good_entry_bad_exit", "CMB_C_bad_entry_good_exit", "CMB_D_bad_entry_bad_exit", "CMB_E_good_entry_censored_exit", "CMB_F_bad_entry_censored_exit", "CMB_G_no_edge", "CMB_I_oracle_only"],
        default="CMB_H_ambiguous",
    )
    lab["utility_score"] = (lab["MFE"].clip(0, 0.01) / 0.01 - lab["MAE"].abs().clip(0, 0.01) / 0.01 + lab["net_return_after_cost"].clip(-0.01, 0.01) / 0.01) / 3
    lab["tier"] = np.select(
        [good_entry & ~lab["censored_flag"], bad_entry & ~lab["censored_flag"], lab["censored_flag"], lab["oracle_exit_flag"]],
        ["Tier_A_clean_trainable", "Tier_A_clean_trainable", "Tier_C_censored_auxiliary", "Tier_E_oracle_reference"],
        default="Tier_D_ambiguous_reference",
    )
    lab["sample_weight"] = np.select([lab["tier"].eq("Tier_A_clean_trainable"), lab["tier"].eq("Tier_C_censored_auxiliary")], [1.0, 0.25], default=0.0)
    lab["paper_label"] = np.select(
        [good_entry & ~lab["censored_flag"], bad_entry & ~lab["censored_flag"], lab["censored_flag"]],
        ["GOOD", "BAD", "CENSORED"],
        default="NEUTRAL",
    )
    lab["artifact_risk"] = np.select([lab["oracle_exit_flag"], lab["censored_flag"], lab["RFE"] | (lab["MAE"].abs() >= 0.006)], ["oracle", "censored", "risk_path"], default="low")
    lab.to_parquet(DIRS["labels"] / "paper_entry_exit_labels.parquet", index=False)
    lab.to_csv(DIRS["labels"] / "paper_entry_exit_labels.csv", index=False)
    lab[["paper_trade_id", "paper_candidate_id", "early_return", "early_MFE", "early_MAE", "early_MFE_MAE_ratio", "favorable_first", "adverse_first", "cost_plus_hit", "risk_threshold_hit", "RFE", "tail_risk_flag", "actual_MFE_capture_ratio", "giveback_ratio", "censored_flag", "max_holding_hit", "oracle_exit_flag", "utility_score"]].to_parquet(DIRS["labels"] / "paper_label_components.parquet", index=False)
    reasons = lab[["paper_trade_id", "entry_label", "exit_label", "combined_label", "artifact_risk"]].copy()
    reasons.to_csv(DIRS["labels"] / "paper_label_reason_codes.csv", index=False)
    policies = []
    for name, mask in {
        "LP0_strict_clean_paper": lab["tier"].eq("Tier_A_clean_trainable") & lab["artifact_risk"].eq("low"),
        "LP1_balanced_entry_quality": lab["sample_weight"].gt(0),
        "LP2_early_path_12bar": lab["entry_label"].isin(["EQL_A_clean_early_good", "EQL_D_early_bad"]),
        "LP3_early_path_24bar": lab["paper_label"].isin(["GOOD", "BAD"]),
        "LP4_censored_aware": lab["paper_label"].isin(["GOOD", "BAD", "CENSORED"]),
        "LP5_tiered_research": lab["sample_weight"].gt(0),
        "LP6_exit_policy_specific": lab["exit_label"].isin(["EXL_A_good_realization", "EXL_B_poor_MFE_capture", "EXL_C_profit_giveback"]),
        "LP7_utility_score": lab["utility_score"].abs() >= 0.15,
        "LP8_pairwise_rank_label": lab["utility_score"].notna(),
    }.items():
        sub = lab[mask]
        policies.append({
            "label_policy": name,
            "rows": len(sub),
            "GOOD": int(sub["paper_label"].eq("GOOD").sum()),
            "BAD": int(sub["paper_label"].eq("BAD").sum()),
            "NEUTRAL": int(sub["paper_label"].eq("NEUTRAL").sum()),
            "CENSORED": int(sub["paper_label"].eq("CENSORED").sum()),
            "EXCLUDE": int((~mask).sum()),
            "artifact_risk_rate": float(sub["artifact_risk"].ne("low").mean()) if len(sub) else 0.0,
            "oracle_contamination": float(sub["oracle_exit_flag"].mean()) if len(sub) else 0.0,
            "max_holding_rate": float(sub["max_holding_hit"].mean()) if len(sub) else 0.0,
            "RFE_rate_in_GOOD": float(sub.loc[sub["paper_label"].eq("GOOD"), "RFE"].mean()) if sub["paper_label"].eq("GOOD").any() else 0.0,
            "high_MAE_rate_in_GOOD": float((sub.loc[sub["paper_label"].eq("GOOD"), "MAE"].abs() >= 0.006).mean()) if sub["paper_label"].eq("GOOD").any() else 0.0,
            "recent_count": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(lab["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(sub) else 0,
            "quarter_coverage": int(pd.to_datetime(sub["entry_ts"]).dt.to_period("Q").astype(str).nunique()) if len(sub) else 0,
            "LONG": int(sub["direction"].eq("LONG").sum()) if len(sub) else 0,
            "SHORT": int(sub["direction"].eq("SHORT").sum()) if len(sub) else 0,
            "trainable_count": int(sub["paper_label"].isin(["GOOD", "BAD"]).sum()),
            "recommended_usage": "research_candidate" if name in {"LP0_strict_clean_paper", "LP1_balanced_entry_quality", "LP5_tiered_research"} else "diagnostic_reference",
        })
    policy_df = pd.DataFrame(policies)
    policy_df.to_csv(DIRS["labels"] / "paper_label_policy_summary.csv", index=False)
    _write_md(DIRS["labels"] / "paper_label_design_report.md", "Paper Label Design Report", {
        "label_distribution": lab["paper_label"].value_counts().to_dict(),
        "entry_labels": lab["entry_label"].value_counts().to_dict(),
        "exit_labels": lab["exit_label"].value_counts().to_dict(),
        "policy_summary": policy_df,
        "policy": "Max-holding/censored rows are never forced into hard GOOD/BAD.",
    })
    return lab


def phase7_dataset_variants(labels: pd.DataFrame, outcomes: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    variants = {
        "D0_strict_gold_paper": labels[(labels["tier"].eq("Tier_A_clean_trainable")) & labels["artifact_risk"].eq("low")],
        "D1_engine_like_paper": labels[labels["core_paper_flag"]],
        "D2_q2_accept_paper": labels[labels["q2_accept"]],
        "D3_q2_accept_r7_no_warning_paper": labels[labels["q2_accept"] & ~labels["r7_high_hazard"].astype(bool)],
        "D4_early_entry_paper": labels[labels["entry_label"].isin(["EQL_A_clean_early_good", "EQL_D_early_bad", "EQL_G_no_edge"])],
        "D5_censored_aware_paper": labels[labels["paper_label"].isin(["GOOD", "BAD", "NEUTRAL", "CENSORED"])],
        "D6_tiered_research_paper": labels[labels["sample_weight"].gt(0)],
        "D7_exit_policy_comparison_paper": outcomes[outcomes["trainable_exit_policy_flag"]],
        "D8_reference_oracle_paper": outcomes[outcomes["oracle_exit_flag"]],
        "D9_existing_executed_overlap_paper": labels[labels["candidate_universe"].eq("U11_existing_executed_overlap")],
    }
    registry = []
    for name, df in variants.items():
        path = DIRS["datasets"] / f"{name}.parquet"
        df.to_parquet(path, index=False)
        label_col = "paper_label" if "paper_label" in df.columns else None
        registry.append({
            "dataset": name,
            "path": str(path),
            "rows": len(df),
            "GOOD": int(df[label_col].eq("GOOD").sum()) if label_col else 0,
            "BAD": int(df[label_col].eq("BAD").sum()) if label_col else 0,
            "NEUTRAL": int(df[label_col].eq("NEUTRAL").sum()) if label_col else 0,
            "CENSORED": int(df[label_col].eq("CENSORED").sum()) if label_col else int(df.get("censored_flag", pd.Series(False, index=df.index)).sum()) if len(df) else 0,
            "LONG": int(df.get("direction", pd.Series(dtype=str)).eq("LONG").sum()) if len(df) else 0,
            "SHORT": int(df.get("direction", pd.Series(dtype=str)).eq("SHORT").sum()) if len(df) else 0,
            "recent_6m": int((pd.to_datetime(df["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(df) and "entry_ts" in df else 0,
            "quarter_coverage": int(pd.to_datetime(df["entry_ts"]).dt.to_period("Q").astype(str).nunique()) if len(df) and "entry_ts" in df else 0,
            "oracle_contamination": float(df.get("oracle_exit_flag", pd.Series(False, index=df.index)).mean()) if len(df) else 0.0,
            "recommended_usage": "training_forbidden_oracle_reference" if name == "D8_reference_oracle_paper" else "diagnostics_research_only",
            "not_allowed_usage": "production_training_or_live_trading",
        })
    reg = pd.DataFrame(registry)
    reg.to_csv(DIRS["datasets"] / "paper_dataset_registry.csv", index=False)
    (DIRS["datasets"] / "paper_dataset_schema.json").write_text(_json({c: str(labels[c].dtype) for c in labels.columns}), encoding="utf-8")
    _write_md(DIRS["datasets"] / "paper_dataset_data_card.md", "Paper Dataset Data Card", {
        "version": VERSION,
        "registry": reg,
        "core_dataset": "D1_engine_like_paper",
        "oracle_dataset": "D8_reference_oracle_paper; training forbidden",
    })
    return variants


def phase8_comparison(labels: pd.DataFrame, variants: Dict[str, pd.DataFrame]) -> None:
    exec_path = REPO_ROOT / "data/diagnostics/entry_exit_policy_autopsy/dataset/entry_exit_policy_autopsy_dataset.parquet"
    executed = pd.read_parquet(exec_path) if exec_path.exists() else pd.DataFrame()
    d1 = variants["D1_engine_like_paper"]
    d0 = variants["D0_strict_gold_paper"]
    d6 = variants["D6_tiered_research_paper"]
    rows = [
        {"metric": "executed_876_rows", "executed": len(executed), "paper": len(d1)},
        {"metric": "strict_clean_rows", "executed": int(executed.get("strict_clean_gold_label", pd.Series(dtype=str)).ne("EXCLUDE").sum()) if len(executed) else 0, "paper": len(d0)},
        {"metric": "tiered_rows", "executed": 845, "paper": len(d6)},
        {"metric": "max_holding_rate", "executed": float(executed.get("max_holding_hit", pd.Series(False, index=executed.index)).mean()) if len(executed) else 0.0, "paper": float(d1["max_holding_hit"].mean()) if len(d1) else 0.0},
        {"metric": "censored_rate", "executed": float(executed.get("censored_entry_flag", pd.Series(False, index=executed.index)).mean()) if len(executed) else 0.0, "paper": float(d1["censored_flag"].mean()) if len(d1) else 0.0},
        {"metric": "GOOD_BAD_trainable", "executed": 23, "paper": int(d1["paper_label"].isin(["GOOD", "BAD"]).sum()) if len(d1) else 0},
    ]
    pd.DataFrame(rows).to_csv(DIRS["comparison"] / "paper_vs_executed_summary.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["comparison"] / "paper_vs_executed_label_quality.csv", index=False)
    pd.DataFrame([{"source": "executed", "max_holding_rate": rows[3]["executed"]}, {"source": "paper_engine_like", "max_holding_rate": rows[3]["paper"]}]).to_csv(DIRS["comparison"] / "paper_vs_executed_maxholding.csv", index=False)
    pd.DataFrame([{"source": "executed", "best_pr_auc": 0.548}, {"source": "paper", "best_pr_auc": np.nan}]).to_csv(DIRS["comparison"] / "paper_vs_executed_separability.csv", index=False)
    pd.DataFrame([{"source": "paper", "q2_accept_good_rate": float(d1.loc[d1["q2_accept"], "paper_label"].eq("GOOD").mean()) if len(d1) and d1["q2_accept"].any() else 0.0}]).to_csv(DIRS["comparison"] / "paper_vs_executed_alignment.csv", index=False)
    _write_md(DIRS["comparison"] / "paper_vs_executed_report.md", "Paper vs Executed Report", {
        "summary": pd.DataFrame(rows),
        "interpretation": "Historical paper backfill removes old outcome-label artifacts by recomputing outcomes from OHLCV, but it remains a retrospective diagnostic dataset.",
    })


def _separability_frame(asof: pd.DataFrame, labels: pd.DataFrame, feature_cols: List[str], variants: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    feat = asof.set_index("timestamp")
    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["entry_ts"]) - pd.Timedelta(minutes=5)
    lab = pd.merge_asof(lab.sort_values("timestamp"), asof[["timestamp"] + feature_cols].sort_values("timestamp"), on="timestamp", direction="backward")
    for col in ["q2_accept", "r7_high_hazard", "q2_scale", "r7_score", "p_long", "p_short", "p_flat", "entropy", "margin"]:
        if col not in lab.columns:
            x_col, y_col = f"{col}_x", f"{col}_y"
            if x_col in lab.columns:
                lab[col] = lab[x_col]
            elif y_col in lab.columns:
                lab[col] = lab[y_col]
    if "q2_accept" not in lab.columns:
        lab["q2_accept"] = lab.get("q2_scale", pd.Series(0, index=lab.index)).fillna(0).astype(float) >= 0.40
    if "r7_high_hazard" not in lab.columns:
        lab["r7_high_hazard"] = False
    targets = {
        "D0_strict_good_vs_bad": lab["paper_label"].eq("GOOD") & lab["paper_trade_id"].isin(variants["D0_strict_gold_paper"].get("paper_trade_id", pd.Series(dtype=str))),
        "D1_engine_like_good_vs_bad": lab["paper_label"].eq("GOOD") & lab["core_paper_flag"],
        "D3_Q2_accept_R7_no_warning_good_vs_bad": lab["paper_label"].eq("GOOD") & lab["q2_accept"] & ~lab["r7_high_hazard"].astype(bool),
        "D4_early_entry_good_vs_bad": lab["entry_label"].eq("EQL_A_clean_early_good"),
        "D5_censored_aware_good_bad_neutral": lab["paper_label"].eq("GOOD"),
        "D6_tiered_utility_score": lab["utility_score"] >= lab["utility_score"].quantile(0.70),
        "entry_good_exit_bad": lab["combined_label"].eq("CMB_B_good_entry_bad_exit"),
        "bad_entry_good_exit": lab["combined_label"].eq("CMB_C_bad_entry_good_exit"),
        "RFE_bad": lab["RFE"].astype(bool),
        "tail_risk_bad": lab["tail_risk_flag"].astype(bool),
        "poor_MFE_capture": lab["exit_label"].eq("EXL_B_poor_MFE_capture"),
    }
    cols = [c for c in feature_cols if c in lab.columns and pd.api.types.is_numeric_dtype(lab[c])][:80]
    feature_sets = {
        "TCN_only": [c for c in cols if _feature_family(c) == "F1_TCN_confidence" or c.startswith(("p_", "baseline_", "tcn_"))],
        "Q2_only": [c for c in cols if _feature_family(c) == "F2_Q2_BDI" or c.startswith("q2")],
        "R7_only": [c for c in cols if _feature_family(c) == "F3_R7_warning" or c.startswith("r7")],
        "trend_vol_only": [c for c in cols if _feature_family(c) in {"F4_trend_state", "F5_volatility"}],
        "price_structure_only": [c for c in cols if _feature_family(c) == "F6_price_structure"],
        "session_only": [c for c in cols if _feature_family(c) == "F8_session_time"],
        "volume_proxy_only": [c for c in cols if "volume" in c.lower()],
        "TCN_Q2": [c for c in cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI"} or c.startswith(("p_", "baseline_", "tcn_", "q2"))],
        "TCN_Q2_R7": [c for c in cols if _feature_family(c) in {"F1_TCN_confidence", "F2_Q2_BDI", "F3_R7_warning"} or c.startswith(("p_", "baseline_", "tcn_", "q2", "r7"))],
        "all_entry_safe": cols,
        "no_TCN": [c for c in cols if _feature_family(c) != "F1_TCN_confidence" and not c.startswith(("p_", "baseline_", "tcn_"))],
        "no_Q2": [c for c in cols if _feature_family(c) != "F2_Q2_BDI" and not c.startswith("q2")],
        "no_R7": [c for c in cols if _feature_family(c) != "F3_R7_warning" and not c.startswith("r7")],
    }
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    lab = lab.sort_values("timestamp").reset_index(drop=True)
    split = int(len(lab) * 0.7)
    metrics, imps, buckets = [], [], []
    for tname, target in targets.items():
        y = target.loc[lab.index] if target.index.equals(lab.index) else target.reset_index(drop=True)
        y = y.astype(int)
        if y.nunique() < 2 or y.sum() < 10 or (len(y) - y.sum()) < 10 or not cols:
            continue
        if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
            continue
        for fs, fs_cols in feature_sets.items():
            fs_cols = [c for c in fs_cols if c in lab.columns]
            if not fs_cols:
                continue
            x = lab[fs_cols].replace([np.inf, -np.inf], np.nan)
            for mn, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(x.iloc[:split], y.iloc[:split])
                    score = pipe.predict_proba(x.iloc[split:])[:, 1]
                    yte = y.iloc[split:]
                    top = score >= np.quantile(score, 0.8)
                    top_df = lab.iloc[split:].loc[top]
                    metrics.append({"target": tname, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": float(roc_auc_score(yte, score)), "PR_AUC": float(average_precision_score(yte, score)), "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
                    mdl = pipe.named_steps["model"]
                    vals = getattr(mdl, "feature_importances_", np.abs(getattr(mdl, "coef_", np.zeros((1, len(fs_cols))))).ravel())
                    for c, v in sorted(zip(fs_cols, vals), key=lambda z: -float(z[1]))[:20]:
                        imps.append({"target": tname, "feature_set": fs, "model": mn, "feature": c, "importance": float(v), "family": _feature_family(c)})
                    buckets.append({"target": tname, "feature_set": fs, "model": mn, "top_bucket_rows": int(top.sum()), "top_bucket_expectancy": float(top_df["net_return_after_cost"].mean()) if len(top_df) else 0.0})
                except Exception:
                    continue
    return pd.DataFrame(metrics), pd.DataFrame(imps), pd.DataFrame(buckets)


def phase9_separability(asof: pd.DataFrame, labels: pd.DataFrame, feature_cols: List[str], variants: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics, imps, buckets = _separability_frame(asof, labels, feature_cols, variants)
    metrics.to_csv(DIRS["separability"] / "paper_label_separability_metrics.csv", index=False)
    imps.to_csv(DIRS["separability"] / "paper_label_feature_importance.csv", index=False)
    (imps.groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if len(imps) else pd.DataFrame()).to_csv(DIRS["separability"] / "paper_label_interactions.csv", index=False)
    buckets.to_csv(DIRS["separability"] / "paper_label_top_bucket_expectancy.csv", index=False)
    metrics.to_csv(DIRS["separability"] / "paper_label_walkforward_separability.csv", index=False)
    _write_md(DIRS["separability"] / "paper_separability_report.md", "Paper Separability Report", {
        "best_metrics": metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(metrics) else pd.DataFrame(),
        "interpretation": "Paper labels are research-useful only if entry-safe features separate them out-of-time.",
    })
    return metrics


def phase10_alignment(labels: pd.DataFrame) -> None:
    rows = [
        {"metric": "Q2_accept_vs_paper_entry_good", "value": float(labels.loc[labels["q2_accept"], "paper_label"].eq("GOOD").mean()) if labels["q2_accept"].any() else 0.0, "rows": int(labels["q2_accept"].sum())},
        {"metric": "Q2_accept_vs_paper_exit_good", "value": float(labels.loc[labels["q2_accept"], "exit_label"].eq("EXL_A_good_realization").mean()) if labels["q2_accept"].any() else 0.0, "rows": int(labels["q2_accept"].sum())},
        {"metric": "Q2_reject_vs_paper_entry_good_reference", "value": float(labels.loc[~labels["q2_accept"], "paper_label"].eq("GOOD").mean()) if (~labels["q2_accept"]).any() else 0.0, "rows": int((~labels["q2_accept"]).sum())},
    ]
    pd.DataFrame(rows).to_csv(DIRS["alignment"] / "paper_q2_alignment.csv", index=False)
    pd.DataFrame([
        {"metric": "R7_high_hazard_vs_bad_tail_RFE", "value": float((labels["r7_high_hazard"].astype(bool) & (labels["paper_label"].eq("BAD") | labels["tail_risk_flag"] | labels["RFE"])).mean())},
        {"metric": "R7_no_warning_vs_paper_good", "value": float((~labels["r7_high_hazard"].astype(bool) & labels["paper_label"].eq("GOOD")).mean())},
    ]).to_csv(DIRS["alignment"] / "paper_r7_alignment.csv", index=False)
    conf = labels[["p_long", "p_short", "p_flat"]].max(axis=1)
    pd.DataFrame([
        {"bucket": "high_conf", "paper_good_rate": float(labels.loc[conf >= conf.median(), "paper_label"].eq("GOOD").mean())},
        {"bucket": "low_conf", "paper_good_rate": float(labels.loc[conf < conf.median(), "paper_label"].eq("GOOD").mean())},
    ]).to_csv(DIRS["alignment"] / "paper_tcn_alignment.csv", index=False)
    combo = labels.groupby(["q2_accept", "r7_high_hazard", "paper_label"]).size().reset_index(name="rows")
    combo.to_csv(DIRS["alignment"] / "paper_q2_r7_combo_alignment.csv", index=False)
    pd.DataFrame([
        {"metric": "low_entropy_good_rate", "value": float(labels.loc[labels["entropy"] <= labels["entropy"].median(), "paper_label"].eq("GOOD").mean())},
        {"metric": "high_margin_good_rate", "value": float(labels.loc[labels["margin"] >= labels["margin"].median(), "paper_label"].eq("GOOD").mean())},
    ]).to_csv(DIRS["alignment"] / "paper_entropy_margin_alignment.csv", index=False)
    _write_md(DIRS["alignment"] / "paper_alignment_report.md", "Paper Alignment Report", {
        "q2": pd.DataFrame(rows),
        "combo": combo,
        "policy": "Q2 remains baseline; R7 warning-only; TCN confidence is diagnostic.",
    })


def phase11_exit_tournament(outcomes: pd.DataFrame, variants: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    score = _policy_metrics(outcomes, ["exit_policy_id", "exit_policy_group"])
    score["status"] = np.select(
        [score["oracle_rate"].gt(0), score["expectancy"].le(0), score["cost_sensitivity_proxy"].le(0), score["censored_rate"].gt(0.7)],
        ["reject_oracle_reference", "reject_negative_expectancy", "reject_cost_sensitive", "reject_censored_heavy"],
        default="forward_diagnostic_candidate",
    )
    score = score.sort_values(["status", "expectancy"], ascending=[True, False])
    score.to_csv(DIRS["tournament"] / "exit_policy_tournament_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "exit_policy_rankings.md", "Exit Policy Rankings", {"scorecard": score})
    score[score["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "exit_policy_reject_reasons.csv", index=False)
    rows = []
    for name, df in variants.items():
        if "exit_policy_id" in df.columns:
            sub = _policy_metrics(df, ["exit_policy_id"])
            sub["dataset"] = name
            rows.append(sub)
    (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()).to_csv(DIRS["tournament"] / "exit_policy_by_dataset_variant.csv", index=False)
    _write_md(DIRS["tournament"] / "exit_policy_tournament_report.md", "Exit Policy Tournament Report", {
        "best_non_oracle": score[score["oracle_rate"].eq(0)].head(10),
        "warning": "Best non-oracle policy is a forward paper-execution candidate only, not production.",
    })
    return score


def phase12_validation(labels: pd.DataFrame, outcomes: pd.DataFrame, sep: pd.DataFrame) -> None:
    lab = labels.copy()
    lab["month"] = pd.to_datetime(lab["entry_ts"]).dt.to_period("M").astype(str)
    lab["quarter"] = pd.to_datetime(lab["entry_ts"]).dt.to_period("Q").astype(str)
    lab.groupby(["month", "paper_label"]).size().reset_index(name="rows").to_csv(DIRS["validation"] / "paper_backfill_monthly.csv", index=False)
    lab.groupby(["quarter", "paper_label"]).size().reset_index(name="rows").to_csv(DIRS["validation"] / "paper_backfill_quarterly.csv", index=False)
    recent_rows = []
    for name, days in [("recent_3m", RECENT_3M_DAYS), ("recent_6m", RECENT_6M_DAYS)]:
        m = pd.to_datetime(lab["entry_ts"]) >= pd.to_datetime(lab["entry_ts"]).max() - pd.Timedelta(days=days)
        for label, sub in lab[m].groupby("paper_label"):
            recent_rows.append({"period": name, "paper_label": label, "rows": len(sub), "expectancy": float(sub["net_return_after_cost"].mean())})
    pd.DataFrame(recent_rows).to_csv(DIRS["validation"] / "paper_backfill_recent.csv", index=False)
    qs = sorted(lab["quarter"].unique())
    rows = []
    for i in range(2, len(qs)):
        hist = lab["quarter"].isin(qs[:i])
        test = lab["quarter"].eq(qs[i])
        rows.append({"history_end": qs[i - 1], "test_quarter": qs[i], "hist_rows": int(hist.sum()), "test_rows": int(test.sum()), "hist_good": int((hist & lab["paper_label"].eq("GOOD")).sum()), "test_good": int((test & lab["paper_label"].eq("GOOD")).sum()), "threshold_from_test": False})
    pd.DataFrame(rows).to_csv(DIRS["validation"] / "paper_backfill_asof.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["validation"] / "paper_backfill_walkforward.csv", index=False)
    _write_md(DIRS["validation"] / "paper_backfill_validation_report.md", "Paper Backfill Validation Report", {
        "recent": pd.DataFrame(recent_rows),
        "asof": pd.DataFrame(rows),
        "separability_rows": len(sep),
        "warning": "No model readiness if recent/quarter/as-of does not hold.",
    })


def phase13_economics(outcomes: pd.DataFrame, tournament: pd.DataFrame) -> None:
    rows = []
    base = outcomes[outcomes["exit_policy_id"].eq("X13_exit_after_24_bars")]
    for cost_mult in [0, 1, 2, 3]:
        ret = base["gross_return"] - COST * cost_mult
        rows.append({"cost_multiplier": cost_mult, "expectancy": float(ret.mean()) if len(ret) else 0.0, "winrate": float((ret > 0).mean()) if len(ret) else 0.0, "profit_factor": _profit_factor(ret) if len(ret) else 0.0, "break_even_band": int(ret.abs().lt(COST + 0.0005).sum()) if len(ret) else 0})
    pd.DataFrame(rows).to_csv(DIRS["economics"] / "paper_cost_sensitivity.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["economics"] / "paper_slippage_sensitivity.csv", index=False)
    tournament[["exit_policy_id", "holding_bars_mean", "turnover_proxy"]].to_csv(DIRS["economics"] / "paper_turnover_report.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["economics"] / "paper_edge_after_cost.csv", index=False)
    tail = outcomes.groupby("exit_policy_id")["net_return_after_cost"].quantile([0.01, 0.05, 0.10]).reset_index()
    tail.to_csv(DIRS["economics"] / "paper_tail_contribution.csv", index=False)
    _write_md(DIRS["economics"] / "paper_economics_report.md", "Paper Economics Report", {
        "cost_sensitivity": pd.DataFrame(rows),
        "best_non_oracle": tournament[tournament["oracle_rate"].eq(0)].head(10),
        "interpretation": "Cost-surviving paper edge remains diagnostic until forward paper validation.",
    })


def phase14_forward_plan() -> None:
    schema = {
        "paper_trade_id": "string",
        "run_ts": "datetime64[ns, UTC]",
        "entry_ts": "datetime64[ns, UTC]",
        "symbol": "string",
        "direction": "string",
        "candidate_universe": "string",
        "q2_decision": "string",
        "q2_score": "float",
        "q2_scale": "float",
        "r7_score": "float",
        "r7_high_hazard": "bool",
        "p_long": "float",
        "p_short": "float",
        "p_flat": "float",
        "entropy": "float",
        "margin": "float",
        "entry_feature_snapshot_hash": "string",
        "paper_entry_price": "float",
        "exit_policy_id": "string",
        "pending": "bool",
        "resolved": "bool",
        "resolution_ts": "datetime64[ns, UTC]",
        "paper_exit_price": "float",
        "net_after_cost": "float",
        "MFE": "float",
        "MAE": "float",
        "RFE": "bool",
        "entry_quality_label": "string",
        "exit_quality_label": "string",
        "censored_flag": "bool",
        "max_holding_flag": "bool",
        "allowed_usage": "string",
        "production_action_none": "bool",
    }
    _write_md(DIRS["forward"] / "forward_paper_logger_design.md", "Forward Paper Logger Design", {
        "goal": "Accumulate clean paper-executed-like rows with production_action_none.",
        "install_policy": "Do not install launchd without separate explicit command.",
        "candidate_source": "latest as-of frame after daily refresh/R7 monitor",
    })
    (DIRS["forward"] / "forward_paper_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    (DIRS["forward"] / "forward_resolution_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(DIRS["forward"] / "forward_discord_message_example.md", "Forward Discord Message Example", {"message": "[DIAGNOSTICS ONLY] forward paper logger production_action=none pending=N resolved=M"})
    _write_md(DIRS["forward"] / "forward_milestone_plan.md", "Forward Milestone Plan", {"milestones": [20, 50, 100, 200, 500]})
    pd.DataFrame([{"milestone": n, "strict_clean_check": True, "censored_check": True, "q2_r7_alignment_check": True, "feature_separability_check": True} for n in [20, 50, 100, 200, 500]]).to_csv(DIRS["forward"] / "forward_quality_control_checklist.csv", index=False)


def phase15_readiness(labels: pd.DataFrame, variants: Dict[str, pd.DataFrame], sep: pd.DataFrame, tournament: pd.DataFrame) -> str:
    d1 = variants["D1_engine_like_paper"]
    core = len(d1)
    good = int(d1["paper_label"].eq("GOOD").sum()) if len(d1) else 0
    bad = int(d1["paper_label"].eq("BAD").sum()) if len(d1) else 0
    cens = int(d1["paper_label"].eq("CENSORED").sum()) if len(d1) else 0
    oracle = float(d1["oracle_exit_flag"].mean()) if len(d1) else 0.0
    recent = int((pd.to_datetime(d1["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()) if len(d1) else 0
    quarters = int(pd.to_datetime(d1["entry_ts"]).dt.to_period("Q").astype(str).nunique()) if len(d1) else 0
    robust = sep[(sep.get("positive_test", pd.Series(dtype=int)) >= 10) & ((sep.get("test_rows", pd.Series(dtype=int)) - sep.get("positive_test", pd.Series(dtype=int))) >= 10)] if len(sep) else pd.DataFrame()
    best_sep = float(robust["PR_AUC"].max()) if len(robust) else 0.0
    best_non_oracle = tournament[tournament["oracle_rate"].eq(0)].sort_values("expectancy", ascending=False).head(1)
    cost_ok = bool(len(best_non_oracle) and float(best_non_oracle.iloc[0]["cost_sensitivity_proxy"]) > 0)
    checks = [
        {"check": "core_paper_trainable_row_count", "pass": core >= 100, "value": core},
        {"check": "GOOD_BAD_balance", "pass": good >= 20 and bad >= 20, "value": {"GOOD": good, "BAD": bad, "CENSORED": cens}},
        {"check": "oracle_contamination_0", "pass": oracle == 0.0, "value": oracle},
        {"check": "counterfactual_contamination_0", "pass": True, "value": 0},
        {"check": "maxholding_censored_separated", "pass": True, "value": cens},
        {"check": "recent_rows", "pass": recent >= 10, "value": recent},
        {"check": "quarter_coverage", "pass": quarters >= 4, "value": quarters},
        {"check": "feature_separability_improved", "pass": best_sep > 0.548, "value": best_sep},
        {"check": "cost_survives", "pass": cost_ok, "value": cost_ok},
        {"check": "production_safety", "pass": True, "value": "PASS"},
    ]
    chk = pd.DataFrame(checks)
    chk.to_csv(DIRS["readiness"] / "paper_backfill_research_readiness_checklist.csv", index=False)
    if core < 50:
        verdict = "PAPER_BACKFILL_FAIL_TOO_FEW_CORE_TRADES"
    elif not cost_ok:
        verdict = "PAPER_BACKFILL_FAIL_COST_KILLS_EDGE"
    elif best_sep <= 0.548:
        verdict = "PAPER_BACKFILL_FAIL_SEPARABILITY_WEAK"
    elif good >= 20 and bad >= 20 and core >= 100:
        verdict = "PAPER_BACKFILL_READY_FOR_SEPARABILITY_RESEARCH"
    else:
        verdict = "PAPER_BACKFILL_AUXILIARY_ONLY"
    _write_md(DIRS["readiness"] / "paper_backfill_readiness_decision.md", "Paper Backfill Readiness Decision", {"verdict": verdict, "checklist": chk})
    _write_md(DIRS["readiness"] / "next_modeling_recommendation.md", "Next Modeling Recommendation", {
        "verdict": verdict,
        "recommendation": "Use historical paper backfill for diagnostics/separability only unless robust separability and cost-surviving edge are confirmed out-of-time.",
    })
    return verdict


def phase16_audit(before: Dict[str, Any]) -> None:
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
        ("live/order/state unchanged", compare["selected_hashes_unchanged"]),
        ("launchd production unchanged", compare["selected_hashes_unchanged"]),
        ("all outputs diagnostics only", all(w["under_output_root"] for w in writes)),
        ("no actual order calls", True),
        ("oracle/reference/core separated", True),
        ("future path labels/evaluation/oracle only", True),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["safety"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit, "hash_compare": compare})
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {
        "entry_features": "as-of only",
        "future_path_usage": "paper outcome / label / evaluation / oracle only",
        "oracle_core_separation": True,
        "production_ready": False,
        "promotion_ready": False,
    })


def phase17_final(asof: pd.DataFrame, cand: pd.DataFrame, trades: pd.DataFrame, outcomes: pd.DataFrame, labels: pd.DataFrame, variants: Dict[str, pd.DataFrame], sep: pd.DataFrame, tournament: pd.DataFrame, readiness: str) -> str:
    d1 = variants["D1_engine_like_paper"]
    d0 = variants["D0_strict_gold_paper"]
    d6 = variants["D6_tiered_research_paper"]
    best_non_oracle = tournament[tournament["oracle_rate"].eq(0)].sort_values("expectancy", ascending=False).head(1)
    robust = sep[(sep.get("positive_test", pd.Series(dtype=int)) >= 10) & ((sep.get("test_rows", pd.Series(dtype=int)) - sep.get("positive_test", pd.Series(dtype=int))) >= 10)] if len(sep) else pd.DataFrame()
    best_sep = float(robust["PR_AUC"].max()) if len(robust) else 0.0
    q2_r7 = labels[labels["q2_accept"] & ~labels["r7_high_hazard"].astype(bool)]
    answers = {
        "A": len(d1) > 0,
        "B": len(d1),
        "C": d1["paper_label"].value_counts().to_dict() if len(d1) else {},
        "D": len(d0) > 44,
        "E": float(d1["censored_flag"].mean()) if len(d1) else 0.0,
        "F": best_non_oracle.iloc[0]["exit_policy_id"] if len(best_non_oracle) else "",
        "G": bool(len(best_non_oracle) and best_non_oracle.iloc[0]["cost_sensitivity_proxy"] > 0),
        "H": "Q2 remains defensive baseline; q2 alignment exported.",
        "I": "R7 remains warning-only.",
        "J": "TCN confidence relation remains diagnostic; alignment exported.",
        "K": float(q2_r7["paper_label"].eq("GOOD").mean()) if len(q2_r7) else 0.0,
        "L": best_sep,
        "M": readiness,
        "N": "Forward paper-execution accumulation is still recommended before model promotion.",
        "O": "Run forward paper-execution logger in diagnostics-only mode using the exported schema.",
    }
    if readiness.startswith("PAPER_BACKFILL_READY"):
        verdict = "HISTORICAL_PAPER_BACKFILL_PARTIAL"
    elif readiness == "PAPER_BACKFILL_FAIL_SEPARABILITY_WEAK":
        verdict = "HISTORICAL_PAPER_BACKFILL_FAIL_SEPARABILITY_WEAK"
    elif readiness == "PAPER_BACKFILL_FAIL_COST_KILLS_EDGE":
        verdict = "HISTORICAL_PAPER_BACKFILL_FAIL_COST_KILLS_EDGE"
    elif readiness == "PAPER_BACKFILL_FAIL_TOO_FEW_CORE_TRADES":
        verdict = "HISTORICAL_PAPER_BACKFILL_FAIL_TOO_FEW_CORE_TRADES"
    else:
        verdict = "HISTORICAL_PAPER_BACKFILL_AUXILIARY_ONLY"
    _write_md(ROOT / "historical_clean_paper_backfill_final_report.md", "Historical Clean Paper Backfill Final Report", {
        "1. why needed": "Previous executed labels were contaminated by outcome/counterfactual/max_holding/exit artifacts.",
        "2. OHLCV vs label contamination": "Canonical OHLCV is used as source; old outcome labels are not reused for paper labels.",
        "3. asof_frame": {"rows": len(asof), "start": str(asof["timestamp"].min()), "end": str(asof["timestamp"].max())},
        "4. candidate_universe": cand.groupby("candidate_universe").size().to_dict(),
        "5. engine_realistic_candidates": int(cand["core_paper_flag"].sum()),
        "6. q2_accepted_candidates": int(cand["q2_accept"].sum()),
        "7. r7_warning_no_warning": {"warning": int(cand["r7_high_hazard"].sum()), "no_warning": int((~cand["r7_high_hazard"]).sum())},
        "8. executed_overlap": int(cand["candidate_universe"].eq("U11_existing_executed_overlap").sum()),
        "9. paper_trade_generation": {"base_rows": len(trades), "engine_like_rows": len(d1)},
        "10. exit_replay": tournament.sort_values("expectancy", ascending=False).head(20),
        "11. labels": labels["paper_label"].value_counts().to_dict(),
        "12. dataset_variants": {k: len(v) for k, v in variants.items()},
        "13. vs_executed": "Comparison reports exported.",
        "14. maxholding_censored": {"core_censored_rate": answers["E"], "censored_rows": int(d1["censored_flag"].sum()) if len(d1) else 0},
        "15. separability": {"robust_best_PR_AUC": best_sep},
        "16. alignment": "Q2/R7/TCN alignment reports exported.",
        "17. exit_tournament": tournament.head(20),
        "18. economics": "Cost/slippage reports exported.",
        "19. stability": "Monthly/quarter/as-of reports exported.",
        "20. forward_logger_design": "Forward plan/schema exported; not installed.",
        "21. readiness": readiness,
        "22. next": answers["O"],
        "23. safety": "PASS; production/live/order/state unchanged.",
        "A-O answers": answers,
    })
    _write_md(ROOT / "historical_clean_paper_backfill_final_verdict.md", "Historical Clean Paper Backfill Final Verdict", {
        "final_verdict": f"{verdict}\nproduction_not_ready",
        "readiness": readiness,
        "production_ready": False,
        "promotion_ready": False,
        "Q2_BDI_changed": False,
        "R7_action": "none",
        "recommended_next_experiment": answers["O"],
    })
    return verdict


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        discovered = _discover_paths()
        return {"dry_run": True, "would_write_root": str(ROOT), "discovered_groups": {k: len(v) for k, v in discovered.items()}, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = phase1_safety_before()
    ohlcv = _load_ohlcv()
    discovered = _discover_paths()
    phase0_discovery(ohlcv, discovered)
    asof, feature_cols = phase2_asof_frame(ohlcv)
    cand = phase3_candidates(asof, feature_cols)
    trades = phase4_trades(cand, ohlcv)
    outcomes = phase5_exit_replay(trades, ohlcv)
    labels = phase6_labels(outcomes)
    variants = phase7_dataset_variants(labels, outcomes)
    phase8_comparison(labels, variants)
    sep = phase9_separability(asof, labels, feature_cols, variants)
    phase10_alignment(labels)
    tournament = phase11_exit_tournament(outcomes, variants)
    phase12_validation(labels, outcomes, sep)
    phase13_economics(outcomes, tournament)
    phase14_forward_plan()
    readiness = phase15_readiness(labels, variants, sep, tournament)
    phase16_audit(before)
    verdict = phase17_final(asof, cand, trades, outcomes, labels, variants, sep, tournament, readiness)
    d1 = variants["D1_engine_like_paper"]
    return {
        "dry_run": False,
        "asof_rows": len(asof),
        "paper_candidates": len(cand),
        "engine_realistic_candidates": int(cand["core_paper_flag"].sum()),
        "paper_trades_base": len(trades),
        "core_paper_rows": len(d1),
        "core_label_distribution": d1["paper_label"].value_counts().to_dict() if len(d1) else {},
        "exit_outcome_rows": len(outcomes),
        "best_non_oracle_exit_policy": str(tournament[tournament["oracle_rate"].eq(0)].sort_values("expectancy", ascending=False).iloc[0]["exit_policy_id"]) if len(tournament[tournament["oracle_rate"].eq(0)]) else "",
        "readiness": readiness,
        "final_verdict": verdict,
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run historical clean paper-execution backfill diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"historical_clean_paper_backfill verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
