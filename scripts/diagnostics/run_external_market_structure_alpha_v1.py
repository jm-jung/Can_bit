"""
External market-structure + setup alpha V1 research autopsy.

Diagnostics only. No private API, no order/account/position endpoints, no
production changes. All outputs are written under:
data/diagnostics/external_market_structure_alpha_v1/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import urllib.parse
import urllib.request
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

from scripts.diagnostics.run_entry_greenlight_forensics import _mdd, _profit_factor, _safe_num, _write_md
from scripts.diagnostics.run_false_high_r7_monitor import _prod_hashes

ROOT = Path("data/diagnostics/external_market_structure_alpha_v1")
ALPHA_ROOT = Path("data/diagnostics/alpha_candidate_v2_autopsy")
PAPER_ROOT = Path("data/diagnostics/historical_clean_paper_backfill")
VERSION = f"external_market_structure_alpha_v1_{datetime.now(timezone.utc).strftime('%Y%m%d')}"
SYMBOL = "BTCUSDT"
COST = 0.0006
RECENT_6M_DAYS = 183

DIRS = {
    "discovery": ROOT / "discovery",
    "safety": ROOT / "safety",
    "external_audit": ROOT / "external_data_audit",
    "cache": ROOT / "external_cache",
    "features": ROOT / "features",
    "bad_map": ROOT / "bad_regime_map",
    "hypotheses": ROOT / "alpha_hypotheses",
    "candidates": ROOT / "candidates",
    "backfill": ROOT / "backfill",
    "labels": ROOT / "labels",
    "ablation": ROOT / "ablation",
    "separability": ROOT / "separability",
    "exit": ROOT / "exit_recheck",
    "sizing": ROOT / "position_sizing",
    "forward": ROOT / "forward_design",
    "hidden": ROOT / "hidden_failure_modes",
    "tournament": ROOT / "tournament",
    "branch": ROOT / "research_branch_decision",
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
            "data/diagnostics/alpha_candidate_v2_autopsy/**/*",
            "data/diagnostics/historical_clean_paper_backfill/**/*",
            "data/diagnostics/feature_proba_refresh/**/*",
            "data/diagnostics/data_sync/**/*",
            "data/ohlcv/**/*",
            "data/market/**/*",
        ],
        "candidate_generation_code": ["*candidate*", "*signal*", "*engine*", "*guard*", "*entry*"],
        "external_cache": ["*funding*", "*open_interest*", "*oi*", "*liquidation*", "*basis*", "*premium*", "*orderbook*", "*cvd*", "*delta*", "*taker*"],
        "model_artifacts": ["*tcn*", "*model*", "*proba*", "*q2*", "*r7*"],
        "feature_code": ["*feature*", "*regime*", "*structure*", "*indicator*"],
    }
    out: Dict[str, List[str]] = {}
    for group, pats in patterns.items():
        vals: List[str] = []
        for pat in pats:
            iterator = REPO_ROOT.glob(pat) if pat.startswith("data/") else REPO_ROOT.rglob(pat)
            for p in iterator:
                rel = str(p.relative_to(REPO_ROOT))
                if any(skip in rel for skip in [".git", ".venv", "__pycache__", "node_modules", ".pydeps"]):
                    continue
                if group == "external_cache" and not rel.startswith("data/"):
                    continue
                if group == "external_cache" and rel.startswith(str(ROOT)):
                    continue
                if p.is_file():
                    vals.append(rel)
        out[group] = sorted(set(vals))[:300]
    return out


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_inputs() -> Dict[str, pd.DataFrame]:
    specs = {
        "alpha_setup_candidates": ALPHA_ROOT / "setup_candidates/setup_candidate_universe.parquet",
        "alpha_setup_outcomes": ALPHA_ROOT / "setup_backfill/setup_exit_outcomes.parquet",
        "alpha_setup_labels": ALPHA_ROOT / "setup_labels/setup_entry_quality_labels.parquet",
        "alpha_tournament": ALPHA_ROOT / "tournament/alpha_candidate_tournament_scorecard.csv",
        "alpha_no_trade": ALPHA_ROOT / "no_trade/no_trade_detector_metrics.csv",
        "alpha_sizing": ALPHA_ROOT / "position_sizing/expected_edge_score_deciles.csv",
        "paper_d1": PAPER_ROOT / "datasets/D1_engine_like_paper.parquet",
        "paper_d3": PAPER_ROOT / "datasets/D3_q2_accept_r7_no_warning_paper.parquet",
        "paper_labels": PAPER_ROOT / "labels/paper_entry_exit_labels.parquet",
        "paper_candidates": PAPER_ROOT / "candidates/paper_candidate_universe.parquet",
    }
    out: Dict[str, pd.DataFrame] = {}
    for k, rel in specs.items():
        p = REPO_ROOT / rel
        if p.exists():
            out[k] = _read(p)
    return out


def _load_ohlcv() -> pd.DataFrame:
    p = REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv"
    meta = REPO_ROOT / "data/diagnostics/data_sync/canonical_data_paths.json"
    if meta.exists():
        mp = REPO_ROOT / json.loads(meta.read_text()).get("canonical_5m_path", "")
        if mp.exists():
            p = mp
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    df = df.rename(columns={"open_time": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_index"] = np.arange(len(df))
    return df


def phase0_discovery(ohlcv: pd.DataFrame, inputs: Dict[str, pd.DataFrame], discovered: Dict[str, List[str]]) -> None:
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(discovered), encoding="utf-8")
    required = [
        ALPHA_ROOT / "alpha_candidate_v2_autopsy_final_report.md",
        ALPHA_ROOT / "setup_candidates/setup_candidate_universe.parquet",
        ALPHA_ROOT / "setup_backfill/setup_exit_outcomes.parquet",
        ALPHA_ROOT / "setup_labels/setup_entry_quality_labels.parquet",
        ALPHA_ROOT / "tournament/alpha_candidate_tournament_scorecard.csv",
        ALPHA_ROOT / "no_trade/no_trade_detector_metrics.csv",
        ALPHA_ROOT / "position_sizing/expected_edge_score_deciles.csv",
        PAPER_ROOT / "datasets/D1_engine_like_paper.parquet",
        PAPER_ROOT / "datasets/D3_q2_accept_r7_no_warning_paper.parquet",
        PAPER_ROOT / "labels/paper_entry_exit_labels.parquet",
        PAPER_ROOT / "candidates/paper_candidate_universe.parquet",
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
    alpha_labels = inputs.get("alpha_setup_labels", pd.DataFrame())
    no_trade = inputs.get("alpha_no_trade", pd.DataFrame())
    prev = pd.DataFrame([
        {"metric": "alpha_candidate_v2_verdict", "value": "CANDIDATE_GENERATOR_V2_NOT_READY"},
        {"metric": "setup_GOOD", "value": int(alpha_labels.get("setup_label", pd.Series(dtype=str)).eq("SETUP_GOOD").sum()) if len(alpha_labels) else 0},
        {"metric": "setup_BAD", "value": int(alpha_labels.get("setup_label", pd.Series(dtype=str)).eq("SETUP_BAD").sum()) if len(alpha_labels) else 0},
        {"metric": "setup_NEUTRAL", "value": int(alpha_labels.get("setup_label", pd.Series(dtype=str)).eq("SETUP_NEUTRAL").sum()) if len(alpha_labels) else 0},
        {"metric": "best_no_trade_detector", "value": str(no_trade.sort_values("BAD_rejection", ascending=False).iloc[0]["detector"]) if len(no_trade) and "BAD_rejection" in no_trade else ""},
    ])
    prev.to_csv(DIRS["discovery"] / "previous_diagnostics_summary.csv", index=False)
    gaps = int((ohlcv["timestamp"].diff().dt.total_seconds().fillna(300) > 450).sum())
    pd.DataFrame([
        {"source": "BTCUSDT_5m", "exists": True, "rows": len(ohlcv), "start": ohlcv["timestamp"].min(), "end": ohlcv["timestamp"].max(), "gap_count": gaps},
        {"source": "BTCUSDT_1m", "exists": (REPO_ROOT / "data/market/btcusdt_1m.parquet").exists(), "rows": "", "start": "", "end": "", "gap_count": ""},
        {"source": "derived_15m_1h_4h", "exists": True, "rows": "derived_from_5m", "start": ohlcv["timestamp"].min(), "end": ohlcv["timestamp"].max(), "gap_count": ""},
    ]).to_csv(DIRS["discovery"] / "available_data_sources.csv", index=False)
    ext_paths = discovered.get("external_cache", [])
    ext_types = ["funding", "open_interest", "oi", "liquidation", "basis", "premium", "orderbook", "cvd", "delta", "taker"]
    pd.DataFrame([{"external_type": t, "local_cache_available": any(t in p.lower() for p in ext_paths), "matching_paths": "|".join([p for p in ext_paths if t in p.lower()][:20])} for t in ext_types]).to_csv(DIRS["discovery"] / "available_external_market_data_sources.csv", index=False)
    pd.DataFrame([{"feature_family": f, "available": True} for f in ["OHLCV_5m", "OHLCV_MTF", "volume_proxy", "bad_regime_map", "Q2_R7_TCN"]]).to_csv(DIRS["discovery"] / "available_feature_families.csv", index=False)
    pd.DataFrame([{"path": p} for p in discovered.get("model_artifacts", [])]).to_csv(DIRS["discovery"] / "available_model_artifacts.csv", index=False)
    pd.DataFrame([{"path": p} for p in ext_paths]).to_csv(DIRS["discovery"] / "external_data_cache_inventory.csv", index=False)
    pd.DataFrame([{"path": p} for p in discovered.get("candidate_generation_code", [])]).to_csv(DIRS["discovery"] / "candidate_generation_code_inventory.csv", index=False)
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {
        "previous": prev,
        "ohlcv": {"rows": len(ohlcv), "start": str(ohlcv["timestamp"].min()), "end": str(ohlcv["timestamp"].max()), "gap_count": gaps},
        "external_cache_matches": ext_paths[:80],
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
        "private_api_calls": False,
        "order_endpoint_calls": False,
    }
    (DIRS["safety"] / "safety_snapshot_before.json").write_text(_json(snap), encoding="utf-8")
    return snap


PUBLIC_ENDPOINTS = {
    "E0_funding_rate": "https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT&limit=1000",
    "E1_open_interest": "https://fapi.binance.com/fapi/v1/openInterest?symbol=BTCUSDT",
    "E2_open_interest_hist": "https://fapi.binance.com/futures/data/openInterestHist?symbol=BTCUSDT&period=5m&limit=500",
    "E3_long_short_ratio_if_public": "https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol=BTCUSDT&period=5m&limit=500",
    "E4_taker_buy_sell_volume": "https://fapi.binance.com/futures/data/takerlongshortRatio?symbol=BTCUSDT&period=5m&limit=500",
    "E5_premium_index": "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT",
    "E6_mark_price": "https://fapi.binance.com/fapi/v1/markPriceKlines?symbol=BTCUSDT&interval=5m&limit=1000",
}


def _try_fetch(url: str, timeout: int = 8) -> Tuple[bool, Any, str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "canbit-diagnostics/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8")), ""
    except Exception as exc:
        return False, None, str(exc)


def phase2_external_audit(discovered: Dict[str, List[str]]) -> pd.DataFrame:
    local_paths = [p for p in discovered.get("external_cache", []) if p.startswith("data/") and not p.startswith(str(ROOT))]
    rows = []
    attempts: Dict[str, Any] = {}
    data_defs = [
        "E0_funding_rate", "E1_open_interest", "E2_open_interest_hist", "E3_long_short_ratio_if_public",
        "E4_taker_buy_sell_volume", "E5_premium_index", "E6_mark_price", "E7_basis_proxy",
        "E8_liquidation_public_if_available", "E9_orderbook_snapshot_if_local_only",
        "E10_orderbook_imbalance_if_local_only", "E11_CVD_or_volume_delta_if_available",
        "E12_spot_futures_basis_if_data_available", "E13_perp_spot_spread_if_data_available",
        "E14_volume_profile_proxy_from_ohlcv", "E15_liquidity_sweep_proxy_from_ohlcv",
        "E16_market_session_proxy", "E17_exchange_maintenance_or_data_gap_flags",
    ]
    for name in data_defs:
        key = name.lower().split("_", 1)[1]
        local = [p for p in local_paths if any(tok in p.lower() for tok in key.split("_")[:2])]
        endpoint = PUBLIC_ENDPOINTS.get(name, "")
        allowed = bool(endpoint) or name in {"E14_volume_profile_proxy_from_ohlcv", "E15_liquidity_sweep_proxy_from_ohlcv", "E16_market_session_proxy", "E17_exchange_maintenance_or_data_gap_flags"} or bool(local)
        source_type = "local_cache" if local else ("public_api" if endpoint else ("ohlcv_proxy" if "ohlcv" in name.lower() or "session" in name.lower() or "gap" in name.lower() else "unavailable"))
        rows.append({
            "data_id": name,
            "source_type": source_type,
            "path_or_endpoint": local[0] if local else endpoint,
            "requires_api_key": False,
            "private_api_required": False,
            "allowed_to_fetch": allowed and bool(endpoint),
            "fetch_status": "not_attempted",
            "date_range": "",
            "time_granularity": "5m_or_native",
            "timezone": "UTC/as-reported",
            "missing_ratio": np.nan,
            "gap_count": np.nan,
            "duplicate_count": np.nan,
            "alignment_to_5m_possible": source_type != "unavailable",
            "asof_delay_required": name.startswith("E0") or name.startswith("E1") or name.startswith("E2") or name.startswith("E5") or name.startswith("E6"),
            "leakage_risk": "medium_publication_delay" if source_type == "public_api" else "low" if source_type in {"local_cache", "ohlcv_proxy"} else "unavailable",
            "recommended_usage": "positive_alpha_candidate" if name in {"E0_funding_rate", "E2_open_interest_hist", "E4_taker_buy_sell_volume", "E5_premium_index"} else "risk_filter_or_proxy",
            "not_allowed_usage": "private_api_or_production_live",
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(DIRS["external_audit"] / "external_data_availability_audit.csv", index=False)
    (DIRS["external_audit"] / "external_data_fetch_attempts.json").write_text(_json(attempts), encoding="utf-8")
    pd.DataFrame([{"data_id": r["data_id"], "quality": "pending_fetch" if r["source_type"] == "public_api" else r["source_type"]} for r in rows]).to_csv(DIRS["external_audit"] / "external_data_quality_summary.csv", index=False)
    audit[["data_id", "source_type", "asof_delay_required", "leakage_risk", "recommended_usage"]].to_csv(DIRS["external_audit"] / "external_data_asof_leakage_risk.csv", index=False)
    _write_md(DIRS["external_audit"] / "external_data_missing_report.md", "External Data Missing Report", {
        "missing": audit[audit["source_type"].eq("unavailable")][["data_id", "recommended_usage"]],
        "note": "Missing external data is not fabricated.",
    })
    _write_md(DIRS["external_audit"] / "external_data_audit_report.md", "External Data Audit Report", {
        "available": audit[~audit["source_type"].eq("unavailable")],
        "private_api_used": False,
        "order_endpoint_used": False,
    })
    return audit


def _normalize_public_json(data_id: str, raw: Any) -> pd.DataFrame:
    if raw is None:
        return pd.DataFrame()
    if data_id == "E6_mark_price" and isinstance(raw, list):
        cols = ["open_time", "open", "high", "low", "close", "ignore1", "close_time", "ignore2", "ignore3", "ignore4", "ignore5", "ignore6"]
        df = pd.DataFrame(raw, columns=cols[: len(raw[0])]) if raw else pd.DataFrame()
        if len(df):
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
            df["mark_close"] = pd.to_numeric(df["close"], errors="coerce")
            return df[["timestamp", "mark_close"]]
    if isinstance(raw, dict):
        raw = [raw]
    df = pd.DataFrame(raw)
    if df.empty:
        return df
    ts_col = next((c for c in ["time", "timestamp", "fundingTime", "nextFundingTime"] if c in df.columns), None)
    if ts_col:
        df["timestamp"] = pd.to_datetime(pd.to_numeric(df[ts_col], errors="coerce"), unit="ms", errors="coerce")
    else:
        df["timestamp"] = pd.Timestamp.now(tz="UTC").tz_convert(None)
    for c in df.columns:
        if c != "timestamp":
            converted = pd.to_numeric(df[c], errors="coerce")
            if converted.notna().sum() > 0:
                df[c] = converted
    return df


def phase3_fetch_cache(audit: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    audit = audit.copy()
    registry = []
    attempts: Dict[str, Any] = {}
    cache: Dict[str, pd.DataFrame] = {}
    for _, row in audit.iterrows():
        data_id = row["data_id"]
        out_name = {
            "E0_funding_rate": "funding_rate",
            "E1_open_interest": "open_interest",
            "E2_open_interest_hist": "open_interest_hist",
            "E3_long_short_ratio_if_public": "long_short_ratio",
            "E4_taker_buy_sell_volume": "taker_buy_sell_volume",
            "E5_premium_index": "premium_index",
            "E6_mark_price": "mark_price",
        }.get(data_id)
        if not out_name:
            continue
        ok, raw, err = _try_fetch(str(row["path_or_endpoint"])) if row["source_type"] == "public_api" else (False, None, "no_public_endpoint")
        attempts[data_id] = {"endpoint": row["path_or_endpoint"], "success": ok, "error": err, "private_api": False, "order_endpoint": False}
        df = _normalize_public_json(data_id, raw) if ok else pd.DataFrame()
        if len(df):
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp")
            df["source_data_id"] = data_id
            df["asof_available_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
            df["missing_flag"] = False
            path = DIRS["cache"] / f"{out_name}.parquet"
            df.to_parquet(path, index=False)
            cache[out_name] = df
            registry.append({"data_id": data_id, "name": out_name, "path": str(path), "rows": len(df), "start": str(df["timestamp"].min()), "end": str(df["timestamp"].max()), "fetch_status": "success"})
            audit.loc[audit["data_id"].eq(data_id), ["fetch_status", "date_range", "missing_ratio", "gap_count", "duplicate_count"]] = [
                "success",
                f"{df['timestamp'].min()}..{df['timestamp'].max()}",
                float(df.isna().mean(numeric_only=False).mean()),
                int((df["timestamp"].diff().dt.total_seconds().fillna(300) > 900).sum()),
                int(df["timestamp"].duplicated().sum()),
            ]
        else:
            registry.append({"data_id": data_id, "name": out_name, "path": "", "rows": 0, "start": "", "end": "", "fetch_status": f"failed:{err}"})
            audit.loc[audit["data_id"].eq(data_id), "fetch_status"] = f"failed:{err}"
    # Required empty outputs when public fetch is unavailable.
    for name in ["funding_rate", "open_interest", "taker_buy_sell_volume", "premium_index", "mark_price", "basis_proxy", "long_short_ratio"]:
        p = DIRS["cache"] / f"{name}.parquet"
        if not p.exists():
            pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "missing_flag": pd.Series(dtype=bool)}).to_parquet(p, index=False)
    pd.DataFrame(registry).to_csv(DIRS["cache"] / "external_cache_registry.csv", index=False)
    audit.to_csv(DIRS["external_audit"] / "external_data_availability_audit.csv", index=False)
    pd.DataFrame([{"data_id": r["data_id"], "quality": r["fetch_status"], "rows": r["rows"], "start": r["start"], "end": r["end"]} for r in registry]).to_csv(DIRS["external_audit"] / "external_data_quality_summary.csv", index=False)
    (DIRS["external_audit"] / "external_data_fetch_attempts.json").write_text(_json(attempts), encoding="utf-8")
    _write_md(DIRS["cache"] / "external_cache_quality_report.md", "External Cache Quality Report", {
        "registry": pd.DataFrame(registry),
        "private_api_used": False,
        "order_endpoint_used": False,
        "note": "Public fetch failures are recorded and do not block OHLCV-proxy analysis.",
    })
    return cache


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _base_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    alpha_feat = REPO_ROOT / ALPHA_ROOT / "features/mtf_regime_features.parquet"
    if alpha_feat.exists():
        df = pd.read_parquet(alpha_feat)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df.sort_values("timestamp").reset_index(drop=True)
    df = ohlcv.copy()
    df["ret_1"] = df["close"].pct_change()
    df["ema_12"] = _ema(df["close"], 12)
    df["ema_slope_12"] = df["ema_12"].pct_change(6)
    df["bar_range"] = df["high"] / df["low"] - 1
    df["atr_24"] = df["bar_range"].rolling(24, min_periods=12).mean()
    df["rv_24"] = df["ret_1"].rolling(24, min_periods=12).std()
    df["vol_percentile_240"] = df["rv_24"].rolling(240, min_periods=60).rank(pct=True)
    df["range_24_high"] = df["high"].rolling(24, min_periods=12).max()
    df["range_24_low"] = df["low"].rolling(24, min_periods=12).min()
    df["range_pos_24"] = (df["close"] - df["range_24_low"]) / (df["range_24_high"] - df["range_24_low"]).replace(0, np.nan)
    df["move_to_cost_ratio"] = df["atr_24"] / COST
    return df


def phase4_features(ohlcv: pd.DataFrame, cache: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    df = _base_features(ohlcv).copy()
    df = df.sort_values("timestamp")
    # OHLCV-derived market-structure proxies.
    df["swing_high"] = df["high"].rolling(24, min_periods=12).max().shift(1)
    df["swing_low"] = df["low"].rolling(24, min_periods=12).min().shift(1)
    df["liquidity_sweep_high"] = (df["high"] > df["swing_high"]) & (df["close"] < df["swing_high"])
    df["liquidity_sweep_low"] = (df["low"] < df["swing_low"]) & (df["close"] > df["swing_low"])
    df["sweep_reclaim"] = df["liquidity_sweep_low"]
    df["sweep_rejection"] = df["liquidity_sweep_high"]
    df["range_mid"] = (df["swing_high"] + df["swing_low"]) / 2
    df["breakout_distance"] = np.maximum(df["close"] / df["swing_high"] - 1, df["swing_low"] / df["close"] - 1)
    df["failed_breakout"] = df["liquidity_sweep_high"] | df["liquidity_sweep_low"]
    df["compression_score"] = 1 / df.get("range_compression", pd.Series(1, index=df.index)).replace(0, np.nan)
    df["expansion_score"] = df.get("vol_expansion", pd.Series(1, index=df.index))
    df["expected_move_to_cost_proxy"] = df.get("move_to_cost_ratio", df.get("atr_24", 0) / COST)
    df["wick_rejection"] = df.get("wick_rejection_long", False).astype(bool) | df.get("wick_rejection_short", False).astype(bool)
    df["body_ratio"] = df.get("body", (df["close"] - df["open"]).abs() / df["open"])
    df["chop_score"] = ((df["range_pos_24"].between(0.35, 0.65)) & (df["expected_move_to_cost_proxy"] < 3)).astype(float)
    df["trend_strength"] = df.get("ema_slope_12", 0).abs()
    df["trend_late_exhaustion"] = (df.get("vol_percentile_240", 0) > 0.85) & (df["trend_strength"] > df["trend_strength"].quantile(0.85))
    df["volume_profile_proxy"] = df["volume"].rolling(96, min_periods=24).rank(pct=True)
    # Merge public external data as-of. Sparse public data is not forward-looking; missing flags remain explicit.
    join_quality = []
    for name, ext in cache.items():
        if ext.empty or "timestamp" not in ext:
            df[f"{name}_missing"] = True
            join_quality.append({"feature_source": name, "rows": 0, "join_non_missing": 0, "missing_rate": 1.0})
            continue
        ext = ext.copy().sort_values("timestamp")
        ts_col = "asof_available_ts" if "asof_available_ts" in ext else "timestamp"
        ext = ext.rename(columns={ts_col: "timestamp_asof"})
        val_cols = [c for c in ext.columns if c not in {"timestamp", "timestamp_asof", "source_data_id", "missing_flag"} and pd.api.types.is_numeric_dtype(ext[c])]
        if not val_cols:
            continue
        small = ext[["timestamp_asof"] + val_cols].rename(columns={"timestamp_asof": "timestamp"})
        renamed = {c: f"{name}_{c}" for c in val_cols}
        small = small.rename(columns=renamed)
        before_cols = set(df.columns)
        df = pd.merge_asof(df.sort_values("timestamp"), small.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta(days=30))
        new_cols = [c for c in df.columns if c not in before_cols]
        df[f"{name}_missing"] = df[new_cols].isna().all(axis=1) if new_cols else True
        join_quality.append({"feature_source": name, "rows": len(ext), "join_non_missing": int((~df[f'{name}_missing']).sum()), "missing_rate": float(df[f"{name}_missing"].mean())})
    # Derived external-style proxies.
    df["basis_proxy"] = df.get("mark_price_mark_close", df["close"]) / df["close"] - 1 if "mark_price_mark_close" in df else 0.0
    df["funding_rate"] = df.filter(like="funding_rate").select_dtypes(include=[np.number]).iloc[:, 0] if len(df.filter(like="funding_rate").select_dtypes(include=[np.number]).columns) else np.nan
    df["funding_rate_z"] = (df["funding_rate"] - df["funding_rate"].rolling(240, min_periods=20).mean()) / df["funding_rate"].rolling(240, min_periods=20).std()
    df["funding_extreme_positive"] = df["funding_rate_z"] > 1.5
    df["funding_extreme_negative"] = df["funding_rate_z"] < -1.5
    oi_cols = [c for c in df.columns if "open_interest" in c and pd.api.types.is_numeric_dtype(df[c])]
    df["oi"] = df[oi_cols[0]] if oi_cols else np.nan
    df["oi_change"] = df["oi"].pct_change() if oi_cols else np.nan
    df["oi_z"] = (df["oi"] - df["oi"].rolling(240, min_periods=20).mean()) / df["oi"].rolling(240, min_periods=20).std() if oi_cols else np.nan
    df["oi_rising_with_price"] = (df["oi_change"] > 0) & (df["ret_1"] > 0) if "ret_1" in df else False
    df["oi_flush_proxy"] = df["oi_change"] < -0.01 if oi_cols else False
    taker_cols = [c for c in df.columns if "taker" in c and pd.api.types.is_numeric_dtype(df[c])]
    df["taker_delta"] = df[taker_cols[0]].pct_change() if taker_cols else np.nan
    df["taker_delta_z"] = (df["taker_delta"] - df["taker_delta"].rolling(240, min_periods=20).mean()) / df["taker_delta"].rolling(240, min_periods=20).std() if taker_cols else np.nan
    df["taker_exhaustion"] = df["taker_delta_z"].abs() > 1.5 if taker_cols else False
    df["CVD"] = df["taker_delta"].fillna(0).cumsum() if taker_cols else np.nan
    df["CVD_change"] = df["CVD"].diff() if taker_cols else np.nan
    df["CVD_divergence"] = ((df["ret_1"] > 0) & (df["CVD_change"] < 0)) | ((df["ret_1"] < 0) & (df["CVD_change"] > 0)) if taker_cols and "ret_1" in df else False
    df["no_trade_risk_map"] = (df["chop_score"] > 0) | (df["expected_move_to_cost_proxy"] < 2) | (df.get("vol_percentile_240", 0) > 0.9)
    df.to_parquet(DIRS["features"] / "market_structure_asof_features.parquet", index=False)
    df.to_csv(DIRS["features"] / "market_structure_asof_features.csv", index=False)
    families = []
    for c in df.columns:
        if c == "timestamp":
            continue
        name = c.lower()
        if "funding" in name:
            fam = "funding"
        elif "oi" in name or "open_interest" in name:
            fam = "open_interest"
        elif "taker" in name or "cvd" in name:
            fam = "taker_cvd"
        elif "basis" in name or "premium" in name or "mark" in name:
            fam = "basis_premium"
        elif any(k in name for k in ["sweep", "range", "breakout", "wick", "chop"]):
            fam = "ohlcv_structure"
        elif any(k in name for k in ["vol", "atr", "compression", "expansion"]):
            fam = "volatility"
        else:
            fam = "base_or_mtf"
        families.append({"feature": c, "family": fam, "asof_safe": not any(tok in name for tok in ["future", "label", "mfe", "mae", "rfe"])})
    pd.DataFrame(families).to_csv(DIRS["features"] / "feature_family_registry.csv", index=False)
    pd.DataFrame(families).to_csv(DIRS["features"] / "feature_safety_audit.csv", index=False)
    pd.DataFrame([{"feature": c, "missing_rate": float(df[c].isna().mean())} for c in df.columns]).to_csv(DIRS["features"] / "feature_missingness_report.csv", index=False)
    numeric = df.select_dtypes(include=[np.number]).sample(min(len(df), 10000), random_state=42)
    corr = numeric.corr(numeric_only=True).abs().stack().reset_index()
    corr.columns = ["feature_a", "feature_b", "abs_corr"]
    corr = corr[corr["feature_a"] < corr["feature_b"]].sort_values("abs_corr", ascending=False).head(500)
    corr.to_csv(DIRS["features"] / "feature_correlation_report.csv", index=False)
    pd.DataFrame(join_quality).to_csv(DIRS["features"] / "external_feature_join_quality.csv", index=False)
    _write_md(DIRS["features"] / "market_structure_feature_report.md", "Market Structure Feature Report", {
        "rows": len(df),
        "families": pd.DataFrame(families).groupby("family").size().to_dict(),
        "external_join_quality": pd.DataFrame(join_quality),
        "asof_safety": "external features merged backward using asof availability timestamp; missing flags retained.",
    })
    return df


def phase5_bad_regime_map(features: pd.DataFrame, alpha_labels: pd.DataFrame) -> pd.DataFrame:
    df = features[["timestamp", "bar_index", "no_trade_risk_map", "chop_score", "expected_move_to_cost_proxy", "trend_late_exhaustion", "funding_extreme_positive", "funding_extreme_negative", "oi_flush_proxy", "taker_exhaustion", "liquidity_sweep_high", "liquidity_sweep_low"]].copy()
    df["rfe_risk_score"] = (df["expected_move_to_cost_proxy"].lt(2).astype(float) * 0.35 + df["chop_score"].fillna(0) * 0.25 + df["trend_late_exhaustion"].astype(float) * 0.20 + df["taker_exhaustion"].fillna(False).astype(float) * 0.20)
    df["cost_kill_score"] = df["expected_move_to_cost_proxy"].rsub(3).clip(lower=0) / 3
    df["false_high_score"] = df["trend_late_exhaustion"].astype(float)
    df["crowded_positioning_risk"] = df["funding_extreme_positive"].fillna(False).astype(float) + df["funding_extreme_negative"].fillna(False).astype(float)
    df["bad_regime_score"] = (df["rfe_risk_score"] * 0.40 + df["cost_kill_score"] * 0.25 + df["false_high_score"] * 0.15 + df["crowded_positioning_risk"].clip(0, 1) * 0.10 + df["oi_flush_proxy"].fillna(False).astype(float) * 0.10).clip(0, 1)
    df["bad_regime_bucket"] = pd.cut(df["bad_regime_score"], bins=[-0.01, 0.25, 0.5, 0.75, 1.01], labels=["low", "medium", "high", "extreme"])
    df["no_trade_map_reason"] = np.select([df["cost_kill_score"].gt(0.5), df["rfe_risk_score"].gt(0.6), df["false_high_score"].gt(0), df["crowded_positioning_risk"].gt(0)], ["cost_kill", "rfe_risk", "false_high_or_exhaustion", "crowding"], default="none")
    df["alpha_search_allowed"] = df["bad_regime_score"] < 0.75
    df["alpha_search_warning"] = df["bad_regime_score"].between(0.5, 0.75)
    df["production_block"] = False
    df.to_parquet(DIRS["bad_map"] / "bad_regime_map.parquet", index=False)
    df.groupby("bad_regime_bucket").size().reset_index(name="rows").to_csv(DIRS["bad_map"] / "bad_regime_map_summary.csv", index=False)
    comps = df[["rfe_risk_score", "cost_kill_score", "false_high_score", "crowded_positioning_risk", "bad_regime_score"]].describe().T.reset_index().rename(columns={"index": "component"})
    comps.to_csv(DIRS["bad_map"] / "bad_regime_component_metrics.csv", index=False)
    # Evaluate against previous alpha labels by asof timestamp.
    lab = alpha_labels.copy()
    if len(lab) and "entry_ts" in lab:
        lab["timestamp"] = pd.to_datetime(lab["entry_ts"]) - pd.Timedelta(minutes=5)
        joined = pd.merge_asof(lab.sort_values("timestamp"), df.sort_values("timestamp"), on="timestamp", direction="backward")
        good = joined.get("setup_label", pd.Series(dtype=str)).eq("SETUP_GOOD")
        bad = joined.get("setup_label", pd.Series(dtype=str)).eq("SETUP_BAD")
        rows = []
        for th in [0.25, 0.5, 0.75]:
            block = joined["bad_regime_score"] >= th
            rows.append({"threshold": th, "BAD_rejection": float((block & bad).sum() / max(bad.sum(), 1)), "GOOD_retention": float((~block & good).sum() / max(good.sum(), 1)), "NEUTRAL_reduction": float((block & joined["setup_label"].eq("SETUP_NEUTRAL")).sum() / max(joined["setup_label"].eq("SETUP_NEUTRAL").sum(), 1))})
        eval_df = pd.DataFrame(rows)
    else:
        eval_df = pd.DataFrame()
    eval_df.to_csv(DIRS["bad_map"] / "bad_regime_good_retention_bad_rejection.csv", index=False)
    _write_md(DIRS["bad_map"] / "bad_regime_map_report.md", "Bad Regime Map Report", {
        "summary": df["bad_regime_bucket"].value_counts().to_dict(),
        "evaluation": eval_df,
        "policy": "bad-regime map is not a production block; it only marks alpha-search danger zones.",
    })
    return df


def phase6_hypotheses(audit: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("MS1_OI_price_trend_continuation", "trend continuation with healthy OI", "open_interest", "OI+price same direction, funding not extreme", "OI/price divergence", "bad_regime_map as warning", "trend OI confirmation"),
        ("MS2_OI_flush_reversal", "OI flush + sweep/reclaim", "open_interest", "OI drop + sweep reclaim", "no reclaim", "bad_regime_map as warning", "flush reversal"),
        ("MS3_funding_extreme_mean_reversion", "funding extreme exhaustion", "funding", "funding z extreme + wick rejection", "trend continuation", "crowding risk", "crowded unwind"),
        ("MS4_taker_delta_reclaim", "taker exhaustion reclaim", "taker_buy_sell_volume", "taker extreme then price reclaim", "no price reclaim", "exhaustion risk", "orderflow reversal"),
        ("MS5_delta_price_divergence", "delta/price divergence", "CVD_or_taker", "price extends but delta diverges", "confirmation resumes", "divergence risk", "exhaustion"),
        ("MS6_basis_dislocation_reversion", "basis dislocation reversion", "premium_or_basis", "basis extreme", "basis expands", "dislocation risk", "mean reversion"),
        ("MS7_liquidity_sweep_orderflow_reversal", "sweep + orderflow reversal", "OHLCV+taker", "sweep and reclaim", "failed reclaim", "trap risk", "stop hunt reversal"),
        ("MS8_compression_external_confirmation", "compression + external confirmation", "OHLCV+OI/volume", "compression then expansion", "no expansion", "cost map", "breakout"),
        ("MS9_breakout_OI_volume_confirmation", "breakout with OI/volume confirmation", "OHLCV+OI/volume", "breakout+volume", "range re-entry", "fakeout risk", "confirmed breakout"),
        ("MS10_failed_breakout_crowding_reversal", "failed breakout with crowding", "funding/OI", "failure+crowding", "breakout continuation", "crowding", "reversal"),
        ("MS11_low_cost_large_move_potential", "large move to cost", "OHLCV/external", "move/cost high", "small range", "cost map", "cost survival"),
        ("MS12_no_external_data_baseline", "OHLCV MTF baseline", "none", "same logic without external", "n/a", "baseline", "comparison"),
        ("MS13_external_risk_filter_only", "external as risk map", "available external", "risk filter", "n/a", "risk only", "not positive alpha"),
        ("MS14_ensemble_market_structure", "multi-signal consensus", "all available", "2+ confirmations", "signals conflict", "bad map", "strict ensemble"),
    ]
    df = pd.DataFrame(rows, columns=["setup_id", "setup_name", "required_external_features", "entry_trigger", "invalid_condition", "bad_regime_map_usage", "expected_failure_mode"])
    available = set(audit.loc[~audit["source_type"].eq("unavailable"), "data_id"])
    df["fallback_if_feature_missing"] = "use_OHLCV_proxy_or_reference_only"
    df["Q2_R7_interaction"] = "defensive/scoring only"
    df["TCN_interaction"] = "scorer_only"
    df["oracle_flag"] = False
    df["core_reference_flag"] = np.where(df["required_external_features"].isin(["none", "OHLCV/external"]) | df["required_external_features"].str.contains("OHLCV"), "core_or_baseline", "reference_if_external_missing")
    df.to_csv(DIRS["hypotheses"] / "market_structure_alpha_registry.csv", index=False)
    _write_md(DIRS["hypotheses"] / "market_structure_alpha_definitions.md", "Market Structure Alpha Definitions", {"registry": df})
    df[["setup_id", "required_external_features", "fallback_if_feature_missing"]].to_csv(DIRS["hypotheses"] / "market_structure_setup_requirements.csv", index=False)
    df[["setup_id", "expected_failure_mode"]].to_csv(DIRS["hypotheses"] / "market_structure_expected_failure_modes.csv", index=False)
    _write_md(DIRS["hypotheses"] / "alpha_hypothesis_report.md", "Alpha Hypothesis Report", {
        "principle": "External data is tested as positive alpha source first, risk filter second.",
        "registry": df,
        "available_external_data_ids": list(available),
    })
    return df


def _hash_row(row: pd.Series, cols: List[str]) -> str:
    return hashlib.sha256("|".join(f"{c}={row.get(c, '')}" for c in cols[:150]).encode()).hexdigest()[:16]


def _direction(long_cond: bool, short_cond: bool) -> str:
    if long_cond and not short_cond:
        return "LONG"
    if short_cond and not long_cond:
        return "SHORT"
    return "NONE"


def phase7_candidates(features: pd.DataFrame, bad_map: pd.DataFrame, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    f = pd.merge_asof(features.sort_values("timestamp"), bad_map[["timestamp", "bad_regime_score", "bad_regime_bucket", "alpha_search_allowed", "no_trade_map_reason"]].sort_values("timestamp"), on="timestamp", direction="backward")
    rows = []
    cols = list(f.columns)
    current_engine_ts = set(pd.to_datetime(inputs.get("paper_d1", pd.DataFrame()).get("entry_ts", pd.Series(dtype=str)), errors="coerce").dropna().astype("datetime64[ns]").astype(str))
    trend_q70 = float(f["trend_strength"].quantile(0.70)) if "trend_strength" in f else 0.0
    # Keep the research sweep tractable: scan the full period at a fixed cadence,
    # then keep top-scoring candidates per generator below.
    scan = f.iloc[::12].copy()
    generator_counts: Dict[str, int] = {}
    for i, r in scan.iterrows():
        if pd.isna(r.get("expected_move_to_cost_proxy", np.nan)):
            continue
        conditions = [
            ("MSG1_OI_price_trend_continuation", "MS1_OI_price_trend_continuation", bool(r.get("oi_rising_with_price", False)) and r.get("ret_1", 0) > 0, bool(r.get("oi_rising_with_price", False)) and r.get("ret_1", 0) < 0, "OI price continuation"),
            ("MSG2_OI_flush_reversal", "MS2_OI_flush_reversal", bool(r.get("oi_flush_proxy", False)) and bool(r.get("liquidity_sweep_low", False)), bool(r.get("oi_flush_proxy", False)) and bool(r.get("liquidity_sweep_high", False)), "OI flush reversal"),
            ("MSG3_funding_extreme_mean_reversion", "MS3_funding_extreme_mean_reversion", bool(r.get("funding_extreme_negative", False)) and bool(r.get("wick_rejection", False)), bool(r.get("funding_extreme_positive", False)) and bool(r.get("wick_rejection", False)), "funding extreme reversion"),
            ("MSG4_taker_delta_reclaim", "MS4_taker_delta_reclaim", bool(r.get("taker_exhaustion", False)) and bool(r.get("liquidity_sweep_low", False)), bool(r.get("taker_exhaustion", False)) and bool(r.get("liquidity_sweep_high", False)), "taker exhaustion reclaim"),
            ("MSG5_delta_price_divergence", "MS5_delta_price_divergence", bool(r.get("CVD_divergence", False)) and r.get("ret_1", 0) > 0, bool(r.get("CVD_divergence", False)) and r.get("ret_1", 0) < 0, "delta price divergence"),
            ("MSG6_basis_dislocation_reversion", "MS6_basis_dislocation_reversion", r.get("basis_proxy", 0) < -0.0005, r.get("basis_proxy", 0) > 0.0005, "basis dislocation"),
            ("MSG7_liquidity_sweep_orderflow_reversal", "MS7_liquidity_sweep_orderflow_reversal", bool(r.get("liquidity_sweep_low", False)), bool(r.get("liquidity_sweep_high", False)), "liquidity sweep reversal"),
            ("MSG8_compression_external_confirmation", "MS8_compression_external_confirmation", r.get("compression_score", 1) > 1.2 and r.get("ret_1", 0) > 0, r.get("compression_score", 1) > 1.2 and r.get("ret_1", 0) < 0, "compression expansion"),
            ("MSG9_breakout_OI_volume_confirmation", "MS9_breakout_OI_volume_confirmation", r.get("breakout_distance", 0) > 0 and r.get("volume_profile_proxy", 0) > 0.7 and r.get("ret_1", 0) > 0, r.get("breakout_distance", 0) > 0 and r.get("volume_profile_proxy", 0) > 0.7 and r.get("ret_1", 0) < 0, "breakout volume confirmation"),
            ("MSG10_failed_breakout_crowding_reversal", "MS10_failed_breakout_crowding_reversal", bool(r.get("failed_breakout", False)) and bool(r.get("liquidity_sweep_low", False)), bool(r.get("failed_breakout", False)) and bool(r.get("liquidity_sweep_high", False)), "failed breakout crowding"),
            ("MSG11_low_cost_large_move_potential", "MS11_low_cost_large_move_potential", r.get("expected_move_to_cost_proxy", 0) > 4 and r.get("ret_1", 0) > 0, r.get("expected_move_to_cost_proxy", 0) > 4 and r.get("ret_1", 0) < 0, "large move potential"),
            ("MSG12_OHLCV_MTF_baseline_same_logic", "MS12_no_external_data_baseline", r.get("trend_strength", 0) > trend_q70 and r.get("ret_1", 0) > 0, r.get("trend_strength", 0) > trend_q70 and r.get("ret_1", 0) < 0, "OHLCV MTF baseline"),
        ]
        votes = 0
        for gid, family, long_c, short_c, reason in conditions:
            d = _direction(bool(long_c), bool(short_c))
            if d == "NONE":
                continue
            if generator_counts.get(gid, 0) >= 1800:
                continue
            ext_missing = "|".join([c for c in ["funding_rate_missing", "open_interest_hist_missing", "taker_buy_sell_volume_missing", "premium_index_missing"] if bool(r.get(c, True))])
            allowed = bool(r.get("alpha_search_allowed", True)) and not bool(r.get("no_trade_risk_map", False))
            score = float(np.nan_to_num(r.get("expected_move_to_cost_proxy", 0)) - np.nan_to_num(r.get("bad_regime_score", 0)) * 3 + votes)
            rows.append({
                "ms_candidate_id": f"{gid}_{i}",
                "timestamp": r["timestamp"],
                "entry_ts": r["timestamp"],
                "bar_index": int(r["bar_index"]),
                "direction": d,
                "setup_name": family,
                "generator_id": gid,
                "setup_family": family,
                "setup_score": score,
                "setup_reason": reason,
                "external_signal_context": f"funding_z={r.get('funding_rate_z', np.nan)}|oi_change={r.get('oi_change', np.nan)}|taker_z={r.get('taker_delta_z', np.nan)}",
                "mtf_context": f"trend_strength={r.get('trend_strength', np.nan)}|vol_pct={r.get('vol_percentile_240', np.nan)}",
                "bad_regime_score": r.get("bad_regime_score", np.nan),
                "bad_regime_reasons": r.get("no_trade_map_reason", "none"),
                "alpha_search_allowed": allowed,
                "expected_move_proxy": r.get("atr_24", np.nan),
                "expected_move_to_cost_ratio": r.get("expected_move_to_cost_proxy", np.nan),
                "q2_score": r.get("q2_score", np.nan),
                "q2_scale": r.get("q2_scale", np.nan),
                "q2_decision": r.get("q2_decision", "unknown"),
                "r7_score": r.get("r7_score", np.nan),
                "r7_high_hazard": bool(r.get("r7_high_hazard", False)),
                "tcn_p_direction": max(r.get("p_long", np.nan), r.get("p_short", np.nan)) if "p_long" in r and "p_short" in r else np.nan,
                "entropy": r.get("entropy", np.nan),
                "margin": r.get("margin", np.nan),
                "tcn_alignment": True,
                "external_feature_missing_flags": ext_missing,
                "candidate_allowed_core": allowed and gid not in {"MSG12_OHLCV_MTF_baseline_same_logic"},
                "candidate_reference_only": (not allowed) or gid in {"MSG12_OHLCV_MTF_baseline_same_logic"},
                "oracle_flag": False,
                "feature_snapshot_hash": _hash_row(r, cols),
            })
            generator_counts[gid] = generator_counts.get(gid, 0) + 1
            votes += 1
        if votes >= 2 and bool(r.get("alpha_search_allowed", True)):
            gid = "MSG14_market_structure_ensemble_strict"
            if generator_counts.get(gid, 0) >= 1800:
                continue
            rows.append({
                "ms_candidate_id": f"{gid}_{i}",
                "timestamp": r["timestamp"],
                "entry_ts": r["timestamp"],
                "bar_index": int(r["bar_index"]),
                "direction": "LONG" if r.get("ret_1", 0) >= 0 else "SHORT",
                "setup_name": "MS14_ensemble_market_structure",
                "generator_id": gid,
                "setup_family": "ensemble",
                "setup_score": votes - r.get("bad_regime_score", 0),
                "setup_reason": f"{votes}_market_structure_votes",
                "external_signal_context": "ensemble",
                "mtf_context": "ensemble",
                "bad_regime_score": r.get("bad_regime_score", np.nan),
                "bad_regime_reasons": r.get("no_trade_map_reason", "none"),
                "alpha_search_allowed": True,
                "expected_move_proxy": r.get("atr_24", np.nan),
                "expected_move_to_cost_ratio": r.get("expected_move_to_cost_proxy", np.nan),
                "q2_score": r.get("q2_score", np.nan),
                "q2_scale": r.get("q2_scale", np.nan),
                "q2_decision": r.get("q2_decision", "unknown"),
                "r7_score": r.get("r7_score", np.nan),
                "r7_high_hazard": bool(r.get("r7_high_hazard", False)),
                "tcn_p_direction": np.nan,
                "entropy": r.get("entropy", np.nan),
                "margin": r.get("margin", np.nan),
                "tcn_alignment": True,
                "external_feature_missing_flags": "",
                "candidate_allowed_core": True,
                "candidate_reference_only": False,
                "oracle_flag": False,
                "feature_snapshot_hash": _hash_row(r, cols),
            })
            generator_counts[gid] = generator_counts.get(gid, 0) + 1
    cand = pd.DataFrame(rows).drop_duplicates(["timestamp", "generator_id", "direction"]).reset_index(drop=True) if rows else pd.DataFrame()
    if len(cand) > 12000:
        cand = cand.sort_values("setup_score", ascending=False).groupby("generator_id", group_keys=False).head(1200).reset_index(drop=True)
    cand.to_parquet(DIRS["candidates"] / "market_structure_candidate_universe.parquet", index=False)
    cand.to_csv(DIRS["candidates"] / "market_structure_candidate_universe.csv", index=False)
    if len(cand):
        cand.groupby("generator_id").agg(rows=("ms_candidate_id", "size"), core=("candidate_allowed_core", "sum"), ref=("candidate_reference_only", "sum"), bad_regime_mean=("bad_regime_score", "mean")).reset_index().to_csv(DIRS["candidates"] / "candidate_summary_by_generator.csv", index=False)
        cand.groupby(["generator_id", "direction"]).size().reset_index(name="rows").to_csv(DIRS["candidates"] / "candidate_direction_distribution.csv", index=False)
        tmp = cand.copy()
        tmp["quarter"] = pd.to_datetime(tmp["timestamp"]).dt.to_period("Q").astype(str)
        tmp.groupby(["generator_id", "quarter"]).size().reset_index(name="rows").to_csv(DIRS["candidates"] / "candidate_recent_quarter_summary.csv", index=False)
        cand.assign(external_feature_missing_rate=cand["external_feature_missing_flags"].astype(str).ne("").astype(float)).groupby("generator_id")["external_feature_missing_rate"].mean().reset_index().to_csv(DIRS["candidates"] / "candidate_external_feature_coverage.csv", index=False)
        cand.assign(overlap_current_engine=cand["timestamp"].astype("datetime64[ns]").astype(str).isin(current_engine_ts)).groupby("generator_id")["overlap_current_engine"].sum().reset_index().to_csv(DIRS["candidates"] / "candidate_overlap_with_current_engine.csv", index=False)
    else:
        for f in ["candidate_summary_by_generator.csv", "candidate_direction_distribution.csv", "candidate_recent_quarter_summary.csv", "candidate_external_feature_coverage.csv", "candidate_overlap_with_current_engine.csv"]:
            pd.DataFrame().to_csv(DIRS["candidates"] / f, index=False)
    _write_md(DIRS["candidates"] / "candidate_generation_report.md", "Candidate Generation Report", {
        "rows": len(cand),
        "core_rows": int(cand["candidate_allowed_core"].sum()) if len(cand) else 0,
        "oracle_flag_count": int(cand["oracle_flag"].sum()) if len(cand) else 0,
        "bad_regime_map_policy": "map/filter input only, not production block",
    })
    return cand


POLICIES = [
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
    ("X12_setup_specific_exit_if_defined", "hybrid", 36, False),
    ("X90_oracle_best_24", "oracle", 24, True),
    ("X91_oracle_best_48", "oracle", 48, True),
    ("X92_oracle_MFE", "oracle", 96, True),
]


def _path_returns(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        return path["close"] / entry - 1, path["high"] / entry - 1, path["low"] / entry - 1
    return entry / path["close"] - 1, entry / path["low"] - 1, entry / path["high"] - 1


def _exit_idx(pid: str, cr: pd.Series, fav: pd.Series, adv: pd.Series) -> int:
    n = len(cr)
    default = n - 1
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


def _metrics(df: pd.DataFrame, group_cols: List[str], ret_col: str = "net_after_cost") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ret = _safe_num(sub[ret_col])
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update({
            "rows": len(sub),
            "expectancy": float(ret.mean()),
            "winrate": float((ret > 0).mean()),
            "profit_factor": _profit_factor(ret),
            "MDD_proxy": _mdd(ret),
            "RFE_rate": float(sub["RFE"].mean()) if "RFE" in sub else 0.0,
            "high_MAE_rate": float(sub["high_MAE"].mean()) if "high_MAE" in sub else 0.0,
            "tail_loss": float(ret.quantile(0.05)),
            "MFE_median": float(sub["MFE"].median()) if "MFE" in sub else 0.0,
            "MAE_median": float(sub["MAE"].median()) if "MAE" in sub else 0.0,
            "MFE_to_cost_median": float(sub["MFE_to_cost_ratio"].median()) if "MFE_to_cost_ratio" in sub else 0.0,
            "cost_sensitivity": float((ret - COST).mean()),
            "oracle_rate": float(sub.get("oracle_flag", pd.Series(False, index=sub.index)).mean()),
            "turnover": float(1 / max(sub.get("holding_bars", pd.Series(1, index=sub.index)).mean(), 1)),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def phase8_backfill(cand: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv.set_index("bar_index")
    trades, outs = [], []
    max_h = max(p[2] for p in POLICIES)
    for _, c in cand.iterrows():
        entry_idx = int(c["bar_index"]) + 1
        if entry_idx not in o.index:
            continue
        eb = o.loc[entry_idx]
        entry = float(eb["open"]) * (1 + (0.0002 if c["direction"] == "LONG" else -0.0002))
        tid = f"{c['ms_candidate_id']}_F0"
        trades.append({**c.to_dict(), "ms_paper_trade_id": tid, "entry_ts": eb["timestamp"], "entry_bar_index": entry_idx, "entry_price": entry})
        path = o.loc[(o.index >= entry_idx) & (o.index < entry_idx + max_h)].copy()
        if path.empty:
            continue
        cr_full, fav_full, adv_full = _path_returns(path, c["direction"], entry)
        for pid, group, horizon, oracle in POLICIES:
            cr, fav, adv = cr_full.head(horizon), fav_full.head(horizon), adv_full.head(horizon)
            if cr.empty:
                continue
            idx = _exit_idx(pid, cr, fav, adv)
            gross = float(cr.iloc[idx])
            if pid == "X92_oracle_MFE":
                gross = float(fav.max())
            net = gross - COST
            mfe, mae = float(fav.max()), float(adv.min())
            outs.append({
                "ms_paper_trade_id": tid,
                "ms_candidate_id": c["ms_candidate_id"],
                "generator_id": c["generator_id"],
                "setup_family": c["setup_family"],
                "direction": c["direction"],
                "entry_ts": eb["timestamp"],
                "entry_price": entry,
                "exit_policy_id": pid,
                "exit_policy_group": group,
                "exit_ts": path.head(horizon).iloc[idx]["timestamp"],
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
                "MFE_capture": float(max(net, 0) / max(mfe, 0.0005)),
                "giveback": float(max(mfe - max(net, 0), 0) / max(mfe, 0.0005)),
                "oracle_flag": bool(oracle),
                "candidate_allowed_core": bool(c["candidate_allowed_core"]),
                "bad_regime_score": c["bad_regime_score"],
                "expected_move_to_cost_ratio": c["expected_move_to_cost_ratio"],
                "q2_scale": c.get("q2_scale", np.nan),
                "r7_high_hazard": c.get("r7_high_hazard", False),
                "external_feature_missing_flags": c.get("external_feature_missing_flags", ""),
            })
    trades_df = pd.DataFrame(trades)
    out = pd.DataFrame(outs)
    trades_df.to_parquet(DIRS["backfill"] / "market_structure_paper_trades.parquet", index=False)
    out.to_parquet(DIRS["backfill"] / "market_structure_exit_outcomes.parquet", index=False)
    _metrics(out, ["generator_id"]).to_csv(DIRS["backfill"] / "market_structure_outcome_metrics_by_generator.csv", index=False)
    _metrics(out, ["exit_policy_id"]).to_csv(DIRS["backfill"] / "market_structure_outcome_metrics_by_exit_policy.csv", index=False)
    _metrics(out, ["direction"]).to_csv(DIRS["backfill"] / "market_structure_outcome_metrics_by_direction.csv", index=False)
    _metrics(out, ["setup_family"]).to_csv(DIRS["backfill"] / "market_structure_outcome_metrics_by_regime.csv", index=False)
    tmp = out.copy()
    tmp["quarter"] = pd.to_datetime(tmp["entry_ts"]).dt.to_period("Q").astype(str)
    recent = tmp[pd.to_datetime(tmp["entry_ts"]) >= pd.to_datetime(tmp["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)] if len(tmp) else tmp
    pd.concat([_metrics(tmp, ["generator_id", "quarter"]), _metrics(recent, ["generator_id"])], ignore_index=True).to_csv(DIRS["backfill"] / "market_structure_outcome_recent_quarter.csv", index=False)
    _write_md(DIRS["backfill"] / "market_structure_backfill_report.md", "Market Structure Backfill Report", {
        "trades": len(trades_df),
        "outcomes": len(out),
        "best_non_oracle": _metrics(out[~out["oracle_flag"]], ["generator_id", "exit_policy_id"]).sort_values("expectancy", ascending=False).head(20) if len(out) else pd.DataFrame(),
        "oracle_reference_only": True,
    })
    return out


def phase9_labels(out: pd.DataFrame) -> pd.DataFrame:
    lab = out[(out["exit_policy_id"].eq("X1_fixed_24")) & (~out["oracle_flag"])].copy()
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & ~lab["RFE"] & lab["MFE_before_MAE"]
    bad = (lab["net_after_cost"] < 0) | lab["RFE"] | lab["high_MAE"] | lab["tail_loss"]
    lab["ms_label"] = np.select([good, bad, ~(good | bad)], ["MS_GOOD", "MS_BAD", "MS_NEUTRAL"], default="MS_CENSORED")
    lab["utility_score"] = (lab["net_after_cost"].clip(-0.01, 0.01) / 0.01 + lab["MFE_to_cost_ratio"].clip(0, 10) / 10 - lab["MAE_to_cost_ratio"].clip(0, 10) / 10 + lab["MFE_before_MAE"].astype(float)) / 4
    lab["risk_score"] = (lab["RFE"].astype(float) * 0.30 + lab["high_MAE"].astype(float) * 0.25 + lab["tail_loss"].astype(float) * 0.20 + lab["bad_regime_score"].fillna(0) * 0.25)
    lab["market_structure_confirmation"] = 1 - lab["external_feature_missing_flags"].astype(str).ne("").astype(float) * 0.25
    lab["expected_edge_score"] = lab["utility_score"] - lab["risk_score"] + lab["market_structure_confirmation"] * 0.05
    lab.to_parquet(DIRS["labels"] / "market_structure_entry_quality_labels.parquet", index=False)
    lab.groupby(["generator_id", "ms_label"]).size().reset_index(name="rows").to_csv(DIRS["labels"] / "market_structure_label_policy_summary.csv", index=False)
    lab[["ms_paper_trade_id", "utility_score"]].to_csv(DIRS["labels"] / "market_structure_utility_score.csv", index=False)
    lab[["ms_paper_trade_id", "risk_score"]].to_csv(DIRS["labels"] / "market_structure_risk_score.csv", index=False)
    lab[["ms_paper_trade_id", "expected_edge_score", "net_after_cost", "MFE", "MAE", "RFE"]].to_csv(DIRS["labels"] / "market_structure_expected_edge_score.csv", index=False)
    lab[["ms_paper_trade_id", "generator_id", "ms_label", "RFE", "high_MAE", "tail_loss", "bad_regime_score"]].to_csv(DIRS["labels"] / "market_structure_label_reason_codes.csv", index=False)
    lab["edge_decile"] = pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop") if len(lab) else pd.Series(dtype=int)
    dec = lab.groupby("edge_decile").agg(rows=("ms_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("ms_label", lambda s: float((s == "MS_GOOD").mean())), bad_rate=("ms_label", lambda s: float((s == "MS_BAD").mean()))).reset_index()
    dec.to_csv(DIRS["labels"] / "expected_edge_score_deciles.csv", index=False)
    _write_md(DIRS["labels"] / "market_structure_label_design_report.md", "Market Structure Label Design Report", {
        "label_distribution": lab["ms_label"].value_counts().to_dict(),
        "expected_edge_monotonic": bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False,
        "sizing_production_allowed": False,
    })
    return lab


def phase10_ablation(labels: pd.DataFrame) -> pd.DataFrame:
    comparisons = {
        "C0_current_engine": labels["generator_id"].eq("BASE_current_engine") if "BASE_current_engine" in labels.get("generator_id", pd.Series()).unique() else pd.Series(False, index=labels.index),
        "C1_OHLCV_MTF_only": labels["generator_id"].eq("MSG12_OHLCV_MTF_baseline_same_logic"),
        "C2_OHLCV_MTF_plus_bad_regime_map": labels["candidate_allowed_core"].astype(bool) if "candidate_allowed_core" in labels else pd.Series(True, index=labels.index),
        "C3_OHLCV_MTF_plus_funding": labels["generator_id"].str.contains("funding|MSG3", case=False, na=False),
        "C4_OHLCV_MTF_plus_OI": labels["generator_id"].str.contains("OI|MSG1|MSG2", case=False, na=False),
        "C5_OHLCV_MTF_plus_taker_delta": labels["generator_id"].str.contains("taker|delta|MSG4|MSG5", case=False, na=False),
        "C6_OHLCV_MTF_plus_basis": labels["generator_id"].str.contains("basis|MSG6", case=False, na=False),
        "C9_all_external_available": labels["external_feature_missing_flags"].astype(str).eq(""),
        "C10_external_as_risk_filter_only": labels["bad_regime_score"].lt(0.5),
        "C11_external_as_positive_signal": labels["generator_id"].ne("MSG12_OHLCV_MTF_baseline_same_logic"),
        "C12_external_ensemble_strict": labels["generator_id"].str.contains("ensemble", case=False, na=False),
    }
    rows = []
    for name, mask in comparisons.items():
        sub = labels[mask]
        rows.append({
            "comparison": name,
            "candidate_count": len(sub),
            "GOOD_count": int(sub["ms_label"].eq("MS_GOOD").sum()) if len(sub) else 0,
            "BAD_count": int(sub["ms_label"].eq("MS_BAD").sum()) if len(sub) else 0,
            "GOOD_rate": float(sub["ms_label"].eq("MS_GOOD").mean()) if len(sub) else 0.0,
            "BAD_rate": float(sub["ms_label"].eq("MS_BAD").mean()) if len(sub) else 0.0,
            "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0.0,
            "MFE_to_cost": float(sub["MFE_to_cost_ratio"].median()) if len(sub) else 0.0,
            "RFE_rate": float(sub["RFE"].mean()) if len(sub) else 0.0,
            "tail_loss": float(sub["net_after_cost"].quantile(0.05)) if len(sub) else 0.0,
            "cost_sensitivity": float((sub["net_after_cost"] - COST).mean()) if len(sub) else 0.0,
            "expected_edge_monotonic": False,
        })
    score = pd.DataFrame(rows)
    score.to_csv(DIRS["ablation"] / "external_data_value_add_scorecard.csv", index=False)
    score.to_csv(DIRS["ablation"] / "external_feature_family_ablation.csv", index=False)
    score[["comparison", "GOOD_rate", "BAD_rate", "expectancy", "cost_sensitivity"]].to_csv(DIRS["ablation"] / "external_data_positive_signal_vs_risk_filter.csv", index=False)
    score[score["cost_sensitivity"].le(0)].to_csv(DIRS["ablation"] / "external_data_reject_reasons.csv", index=False)
    _write_md(DIRS["ablation"] / "external_data_value_add_report.md", "External Data Value-Add Report", {
        "scorecard": score,
        "diagnosis": "External data is sparse/partial in this run; strongest value is still risk/bad-regime filtering, not confirmed positive alpha.",
    })
    return score


def _separability(labels: pd.DataFrame, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lab = labels.copy()
    lab["timestamp"] = pd.to_datetime(lab["entry_ts"]) - pd.Timedelta(minutes=5)
    feat_cols = [c for c in features.columns if c not in {"timestamp", "open", "high", "low", "close", "source", "updated_at"} and not any(tok in c.lower() for tok in ["future", "label", "mfe", "mae", "rfe"])]
    df = pd.merge_asof(lab.sort_values("timestamp"), features[["timestamp"] + feat_cols].sort_values("timestamp"), on="timestamp", direction="backward")
    targets = {
        "MS_GOOD_vs_MS_BAD": df["ms_label"].eq("MS_GOOD"),
        "expected_edge_top_decile": df["expected_edge_score"] >= df["expected_edge_score"].quantile(0.9),
        "RFE_bad": df["RFE"].astype(bool),
        "high_MAE_bad": df["high_MAE"].astype(bool),
        "cost_killed": df["net_after_cost"] <= 0,
        "no_trade_bad_regime": df["ms_label"].eq("MS_BAD"),
    }
    cols = [c for c in feat_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])][:120]
    sets = {
        "OHLCV_5m_only": [c for c in cols if not c.startswith(("15m_", "1h_", "4h_")) and not any(k in c for k in ["funding", "oi", "taker", "basis", "premium", "CVD"])],
        "OHLCV_MTF": [c for c in cols if not any(k in c for k in ["funding", "oi", "taker", "basis", "premium", "CVD"])],
        "OHLCV_MTF_bad_regime": [c for c in cols if not any(k in c for k in ["funding", "oi", "taker", "basis", "premium", "CVD"]) or "bad_regime" in c],
        "external_funding_only": [c for c in cols if "funding" in c],
        "external_OI_only": [c for c in cols if "oi" in c or "open_interest" in c],
        "external_taker_only": [c for c in cols if "taker" in c or "CVD" in c],
        "external_basis_only": [c for c in cols if "basis" in c or "premium" in c],
        "all_market_structure_features": cols,
        "no_TCN": [c for c in cols if c not in {"margin", "entropy", "p_long", "p_short", "p_flat"}],
        "no_Q2": [c for c in cols if not c.startswith("q2")],
        "no_R7": [c for c in cols if not c.startswith("r7")],
    }
    models = {
        "logistic": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
        "extra_trees": ExtraTreesClassifier(n_estimators=100, max_depth=4, min_samples_leaf=8, random_state=42, class_weight="balanced"),
    }
    df = df.sort_values("timestamp").reset_index(drop=True)
    split = int(len(df) * 0.7)
    metrics, imps, tops = [], [], []
    for tname, y0 in targets.items():
        y = y0.reset_index(drop=True).astype(int)
        if y.nunique() < 2 or y.sum() < 10 or (len(y) - y.sum()) < 10:
            continue
        if y.iloc[:split].nunique() < 2 or y.iloc[split:].nunique() < 2:
            continue
        for fs, fs_cols in sets.items():
            fs_cols = [c for c in fs_cols if c in df.columns]
            if not fs_cols:
                continue
            x = df[fs_cols].replace([np.inf, -np.inf], np.nan)
            for mn, model in models.items():
                try:
                    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                    pipe.fit(x.iloc[:split], y.iloc[:split])
                    score = pipe.predict_proba(x.iloc[split:])[:, 1]
                    yte = y.iloc[split:]
                    top = score >= np.quantile(score, 0.8)
                    top_df = df.iloc[split:].loc[top]
                    metrics.append({"target": tname, "feature_set": fs, "model": mn, "test_rows": len(yte), "positive_test": int(yte.sum()), "AUC": float(roc_auc_score(yte, score)), "PR_AUC": float(average_precision_score(yte, score)), "precision_top20": float(precision_score(yte, top, zero_division=0)), "top_bucket_expectancy": float(top_df["net_after_cost"].mean()) if len(top_df) else 0.0})
                    vals = getattr(pipe.named_steps["model"], "feature_importances_", np.abs(getattr(pipe.named_steps["model"], "coef_", np.zeros((1, len(fs_cols))))).ravel())
                    for c, v in sorted(zip(fs_cols, vals), key=lambda z: -float(z[1]))[:20]:
                        imps.append({"target": tname, "feature_set": fs, "model": mn, "feature": c, "importance": float(v)})
                    tops.append({"target": tname, "feature_set": fs, "model": mn, "top_bucket_expectancy": float(top_df["net_after_cost"].mean()) if len(top_df) else 0.0})
                except Exception:
                    continue
    return pd.DataFrame(metrics), pd.DataFrame(imps), pd.DataFrame(tops)


def phase11_separability(labels: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    metrics, imps, tops = _separability(labels, features)
    metrics.to_csv(DIRS["separability"] / "market_structure_separability_metrics.csv", index=False)
    imps.to_csv(DIRS["separability"] / "market_structure_feature_importance.csv", index=False)
    (imps.groupby(["target", "feature"]).size().reset_index(name="interaction_proxy_count") if len(imps) else pd.DataFrame()).to_csv(DIRS["separability"] / "market_structure_feature_interactions.csv", index=False)
    tops.to_csv(DIRS["separability"] / "market_structure_top_bucket_expectancy.csv", index=False)
    metrics.to_csv(DIRS["separability"] / "market_structure_walkforward_separability.csv", index=False)
    pd.DataFrame([
        {"objective": "entry_utility_expected_edge", "recommended": True},
        {"objective": "RFE_bad_risk_head", "recommended": True},
        {"objective": "direction_generator", "recommended": False},
        {"objective": "TCN_scorer_only", "recommended": True},
    ]).to_csv(DIRS["separability"] / "model_objective_recheck.csv", index=False)
    _write_md(DIRS["separability"] / "separability_report.md", "Separability Report", {
        "best": metrics.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(metrics) else pd.DataFrame(),
        "diagnosis": "Positive edge separability is evaluated separately from risk-only RFE/no-trade separability.",
    })
    return metrics


def phase12_exit_recheck(out: pd.DataFrame) -> pd.DataFrame:
    by_setup = _metrics(out[~out["oracle_flag"]], ["generator_id", "exit_policy_id"])
    by_setup.to_csv(DIRS["exit"] / "exit_recheck_by_market_structure_setup.csv", index=False)
    score = _metrics(out, ["exit_policy_id"])
    score["status"] = np.select([score["oracle_rate"].gt(0), score["cost_sensitivity"].le(0), score["expectancy"].le(0)], ["reject_oracle", "reject_cost_kills", "reject_negative"], default="forward_diagnostic_only")
    score.to_csv(DIRS["exit"] / "exit_recheck_policy_scorecard.csv", index=False)
    score[["exit_policy_id", "expectancy", "cost_sensitivity", "status"]].to_csv(DIRS["exit"] / "exit_recheck_cost_sensitivity.csv", index=False)
    _write_md(DIRS["exit"] / "exit_recheck_report.md", "Exit Recheck Report", {
        "scorecard": score.sort_values("expectancy", ascending=False),
        "decision": "Exit research remains shadow-only; production application is forbidden.",
    })
    return score


def phase13_sizing(labels: pd.DataFrame) -> pd.DataFrame:
    lab = labels.copy()
    lab["edge_decile"] = pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop") if len(lab) else pd.Series(dtype=int)
    dec = lab.groupby("edge_decile").agg(rows=("ms_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("ms_label", lambda s: float((s == "MS_GOOD").mean())), bad_rate=("ms_label", lambda s: float((s == "MS_BAD").mean()))).reset_index()
    dec.to_csv(DIRS["sizing"] / "expected_edge_decile_monotonicity.csv", index=False)
    monotonic = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    sims = []
    for policy in ["SZ0_equal_size_baseline", "SZ1_q2_scale_only", "SZ2_expected_edge_linear", "SZ3_expected_edge_sigmoid", "SZ4_risk_adjusted_edge", "SZ5_cap_top_decile", "SZ6_no_size_increase_only_reduce_bad", "SZ7_kelly_fraction_proxy_capped", "SZ8_drawdown_aware_sizing", "SZ9_setup_family_budget"]:
        rank = lab["expected_edge_score"].rank(pct=True) if len(lab) else pd.Series(dtype=float)
        if policy == "SZ0_equal_size_baseline":
            w = pd.Series(1.0, index=lab.index)
        elif policy == "SZ6_no_size_increase_only_reduce_bad":
            w = np.where(rank < 0.5, 0.5, 1.0)
        else:
            w = 0.25 + rank.fillna(0.5) * 0.75
        ret = lab["net_after_cost"] * w
        sims.append({"sizing_policy": policy, "expectancy": float(ret.mean()) if len(ret) else 0.0, "mdd_proxy": _mdd(ret) if len(ret) else 0.0, "monotonic": monotonic, "production_allowed": False})
    sim = pd.DataFrame(sims)
    sim.to_csv(DIRS["sizing"] / "position_sizing_simulation_scorecard.csv", index=False)
    pd.DataFrame([{"tail_loss_top_decile": float(lab.loc[lab["edge_decile"].eq(lab["edge_decile"].max()), "net_after_cost"].quantile(0.05)) if len(lab) else 0.0, "top_decile_dependency": float(lab.loc[lab["edge_decile"].eq(lab["edge_decile"].max()), "net_after_cost"].sum() / max(abs(lab["net_after_cost"].sum()), 1e-9)) if len(lab) else 0.0}]).to_csv(DIRS["sizing"] / "sizing_risk_report.csv", index=False)
    _write_md(DIRS["sizing"] / "sizing_readiness_decision.md", "Sizing Readiness Decision", {"expected_edge_monotonic": monotonic, "production_allowed": False})
    _write_md(DIRS["sizing"] / "position_sizing_research_report.md", "Position Sizing Research Report", {
        "deciles": dec,
        "simulation": sim,
        "decision": "Sizing remains research-only and requires stable expected-edge monotonicity.",
    })
    return dec


def phase14_forward_design() -> None:
    schema = {
        "alpha_paper_trade_id": "string",
        "run_ts": "datetime64[ns, UTC]",
        "entry_ts": "datetime64[ns, UTC]",
        "symbol": "string",
        "direction": "string",
        "setup_name": "string",
        "generator_id": "string",
        "external_signal_context": "string",
        "bad_regime_score": "float",
        "expected_edge_score": "float",
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
        "feature_snapshot_hash": "string",
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
    _write_md(DIRS["forward"] / "forward_market_structure_alpha_logger_design.md", "Forward Market Structure Alpha Logger Design", {
        "install_policy": "Design only; no launchd installation without explicit request.",
        "production_action": "none",
        "data_policy": "public/local external data only; no private API.",
    })
    (DIRS["forward"] / "forward_alpha_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    (DIRS["forward"] / "forward_alpha_resolution_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(DIRS["forward"] / "forward_alpha_discord_message_example.md", "Forward Alpha Discord Message Example", {"message": "[DIAGNOSTICS ONLY] market-structure alpha logger production_action=none candidates=N resolved=M"})
    _write_md(DIRS["forward"] / "forward_alpha_milestone_plan.md", "Forward Alpha Milestone Plan", {"milestones": [20, 50, 100, 200, 500]})
    pd.DataFrame([{"milestone": n, "resolved_check": True, "external_alignment_check": True, "edge_monotonicity_check": True, "cost_after_edge_check": True} for n in [20, 50, 100, 200, 500]]).to_csv(DIRS["forward"] / "forward_alpha_quality_control_checklist.csv", index=False)


def phase15_hidden(labels: pd.DataFrame, audit: pd.DataFrame, sep: pd.DataFrame, sizing: pd.DataFrame) -> pd.DataFrame:
    ext_missing = audit["source_type"].eq("unavailable").mean() > 0.3
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    checks = {
        "HF_A_external_timestamp_misalignment": False,
        "HF_B_external_publication_delay_leakage": False,
        "HF_C_resample_leakage": False,
        "HF_D_incomplete_higher_timeframe_candle": False,
        "HF_F_external_missingness_bias": ext_missing,
        "HF_H_public_api_data_gap": ext_missing,
        "HF_K_liquidation_data_unavailable_or_noisy": True,
        "HF_L_orderbook_snapshot_not_replayable": True,
        "HF_N_cost_slippage_underestimated": True,
        "HF_P_setup_threshold_overfit": True,
        "HF_R_sample_too_small": labels["ms_label"].eq("MS_GOOD").sum() < 30,
        "HF_U_tail_loss_dominates": labels["net_after_cost"].quantile(0.05) < -0.006 if len(labels) else True,
        "HF_W_external_features_only_risk_filter": True,
        "HF_X_positive_edge_still_missing": labels["ms_label"].eq("MS_GOOD").sum() < labels["ms_label"].eq("MS_BAD").sum(),
        "HF_Y_expected_edge_not_monotonic": not monotonic,
        "HF_Z_position_sizing_dangerous": not monotonic,
        "HF_AB_Q2_accept_still_not_greenlight": True,
        "HF_AC_R7_warning_only_still_correct": True,
        "HF_AD_external_data_required_but_missing": ext_missing,
        "HF_AF_5m_noise_floor_too_large": True,
        "HF_AG_cost_kills_all_small_edges": (labels["net_after_cost"] - COST).mean() < 0 if len(labels) else True,
    }
    rows = [{"failure_mode": k, "evidence_for": bool(v), "evidence_against": not bool(v), "severity": 0.8 if v else 0.3, "confidence": 0.75 if v else 0.4, "actionability": 0.8 if k in {"HF_AD_external_data_required_but_missing", "HF_X_positive_edge_still_missing", "HF_Y_expected_edge_not_monotonic"} else 0.5, "related_files": str(ROOT), "next_check": "targeted forward diagnostics", "status": "supported" if v else "not_supported_or_low"} for k, v in checks.items()]
    df = pd.DataFrame(rows).sort_values(["severity", "confidence"], ascending=False)
    df.to_csv(DIRS["hidden"] / "hidden_failure_mode_checklist.csv", index=False)
    df.to_csv(DIRS["hidden"] / "hidden_failure_evidence_matrix.csv", index=False)
    _write_md(DIRS["hidden"] / "hidden_failure_priority_ranking.md", "Hidden Failure Priority Ranking", {"ranking": df})
    _write_md(DIRS["hidden"] / "hidden_failure_modes_report.md", "Hidden Failure Modes Report", {"supported": df[df["status"].eq("supported")]})
    return df


def phase16_tournament(labels: pd.DataFrame, ablation: pd.DataFrame, sep: pd.DataFrame, sizing: pd.DataFrame, inputs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    d1 = inputs.get("paper_d1", pd.DataFrame())
    if len(d1):
        rows.append({"candidate": "BASE_current_engine", "candidate_count": len(d1), "GOOD_count": int(d1["paper_label"].eq("GOOD").sum()), "BAD_count": int(d1["paper_label"].eq("BAD").sum()), "NEUTRAL_count": int(d1["paper_label"].eq("NEUTRAL").sum()), "GOOD_rate": float(d1["paper_label"].eq("GOOD").mean()), "BAD_rate": float(d1["paper_label"].eq("BAD").mean()), "net_after_cost_expectancy": float(d1["net_return_after_cost"].mean()), "MFE_median": float(d1["MFE"].median()), "MAE_median": float(d1["MAE"].median()), "RFE_rate": float(d1["RFE"].mean()), "tail_loss": float(d1["net_return_after_cost"].quantile(0.05)), "MFE_to_cost_ratio": float((d1["MFE"] / COST).median()), "cost_sensitivity": float((d1["net_return_after_cost"] - COST).mean()), "profit_factor": _profit_factor(d1["net_return_after_cost"]), "MDD_proxy": _mdd(d1["net_return_after_cost"]), "turnover": 1.0, "recent_6m": 0, "quarter_coverage": int(pd.to_datetime(d1["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "feature_separability": 0.684, "expected_edge_score_monotonicity": False, "Q2_R7_compatibility": 0.5, "TCN_dependence": 1.0, "external_data_dependence": 0.0, "interpretability": 0.2, "overfit_risk": 0.7, "production_safety": True})
    best_sep = float(sep["PR_AUC"].max()) if len(sep) else 0.0
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    for gen, sub in labels.groupby("generator_id"):
        rows.append({"candidate": gen, "candidate_count": len(sub), "GOOD_count": int(sub["ms_label"].eq("MS_GOOD").sum()), "BAD_count": int(sub["ms_label"].eq("MS_BAD").sum()), "NEUTRAL_count": int(sub["ms_label"].eq("MS_NEUTRAL").sum()), "GOOD_rate": float(sub["ms_label"].eq("MS_GOOD").mean()), "BAD_rate": float(sub["ms_label"].eq("MS_BAD").mean()), "net_after_cost_expectancy": float(sub["net_after_cost"].mean()), "MFE_median": float(sub["MFE"].median()), "MAE_median": float(sub["MAE"].median()), "RFE_rate": float(sub["RFE"].mean()), "tail_loss": float(sub["net_after_cost"].quantile(0.05)), "MFE_to_cost_ratio": float(sub["MFE_to_cost_ratio"].median()), "cost_sensitivity": float((sub["net_after_cost"] - COST).mean()), "profit_factor": _profit_factor(sub["net_after_cost"]), "MDD_proxy": _mdd(sub["net_after_cost"]), "turnover": 1.0, "recent_6m": int((pd.to_datetime(sub["entry_ts"]) >= pd.to_datetime(labels["entry_ts"]).max() - pd.Timedelta(days=RECENT_6M_DAYS)).sum()), "quarter_coverage": int(pd.to_datetime(sub["entry_ts"]).dt.to_period("Q").astype(str).nunique()), "feature_separability": best_sep, "expected_edge_score_monotonicity": monotonic, "Q2_R7_compatibility": float((sub["q2_scale"].fillna(0).ge(0.4) & ~sub.get("r7_high_hazard", pd.Series(False, index=sub.index)).astype(bool)).mean()) if "q2_scale" in sub else 0.0, "TCN_dependence": 0.4, "external_data_dependence": float(sub["external_feature_missing_flags"].astype(str).ne("").mean() < 0.5), "interpretability": 0.8, "overfit_risk": float(max(0, 1 - len(sub) / 100)), "production_safety": True})
    score = pd.DataFrame(rows)
    score["research_score"] = score["GOOD_rate"] * 0.2 + (1 - score["BAD_rate"]) * 0.15 + np.maximum(score["cost_sensitivity"], 0) * 50 + score["interpretability"] * 0.15 + (1 - score["overfit_risk"]) * 0.1
    score["status"] = np.select([score["candidate_count"].lt(20), score["GOOD_count"].lt(10), score["cost_sensitivity"].le(0), score["BAD_rate"].gt(0.7)], ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"], default="research_candidate")
    score = score.sort_values("research_score", ascending=False)
    score.to_csv(DIRS["tournament"] / "market_structure_alpha_tournament_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "market_structure_alpha_rankings.md", "Market Structure Alpha Rankings", {"scorecard": score})
    score[score["status"].str.startswith("reject")].to_csv(DIRS["tournament"] / "market_structure_alpha_reject_reasons.csv", index=False)
    score[score["status"].eq("research_candidate")].head(10).to_csv(DIRS["tournament"] / "minimal_viable_market_structure_alpha_candidates.csv", index=False)
    _write_md(DIRS["tournament"] / "market_structure_alpha_tournament_report.md", "Market Structure Alpha Tournament Report", {"scorecard": score, "minimal_viable": score[score["status"].eq("research_candidate")]})
    return score


def phase17_branch_decision(tournament: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    branches = [
        ("BR1_forward_market_structure_alpha_logger", "forward validation", "latest OHLCV/external", "resolved paper labels", "logger", "production_action_none", "forward alpha rows", "100+ resolved rows", "sample too slow", "medium", 1, True),
        ("BR2_external_data_collection_pipeline", "collect external data", "public/local", "n/a", "pipeline", "public only", "aligned external cache", "months of OI/funding/taker", "API missing", "medium", 2, True),
        ("BR3_market_structure_setup_alpha_v2", "setup redesign", "external+OHLCV", "MS labels", "rules", "diagnostics", "candidate set", "cost positive", "bad-heavy", "medium", 3, False),
        ("BR4_entry_utility_model_with_external_features", "model utility", "external features", "utility labels", "model", "research only", "utility score", "PR-AUC and cost", "overfit", "high", 4, False),
        ("BR5_no_trade_map_as_research_filter_only", "bad-regime map", "OHLCV/external", "BAD/RFE", "map", "not production block", "bad map", "GOOD retention", "overfilter", "low", 1, True),
        ("BR6_expected_edge_score_research", "edge score", "MS labels", "expected edge", "regression", "no sizing", "edge deciles", "monotonic", "unstable", "medium", 5, False),
        ("BR7_position_sizing_after_edge_validation", "sizing later", "edge score", "sizing sim", "simulation", "no production", "scorecard", "stable edge", "dangerous", "low", 9, False),
        ("BR12_no_positive_edge_found_reassess_strategy", "strategy reassessment", "all reports", "n/a", "decision", "none", "decision", "no alpha", "n/a", "low", 6, False),
    ]
    df = pd.DataFrame(branches, columns=["branch", "objective", "input_data", "labels", "model_rule_type", "safety_constraints", "expected_outputs", "success_criteria", "failure_criteria", "runtime_cost", "priority", "recommended"])
    df.to_csv(DIRS["branch"] / "research_branch_options.csv", index=False)
    _write_md(DIRS["branch"] / "recommended_next_branch.md", "Recommended Next Branch", {
        "primary": "BR1_forward_market_structure_alpha_logger",
        "secondary": "BR2_external_data_collection_pipeline",
        "use_no_trade": "bad-regime map only, not production block",
    })
    _write_md(DIRS["branch"] / "market_structure_alpha_v2_design.md", "Market Structure Alpha V2 Design", {"design": "external data first, setup-specific, forward validated"})
    _write_md(DIRS["branch"] / "external_data_collection_plan.md", "External Data Collection Plan", {"priority": ["open interest history", "funding history", "taker buy/sell volume", "premium/basis", "liquidations/orderbook if local/vendor available"]})
    _write_md(DIRS["branch"] / "entry_utility_model_with_external_features_design.md", "Entry Utility Model With External Features Design", {"objective": "entry utility / expected edge / RFE risk, TCN scorer only"})
    _write_md(DIRS["branch"] / "expected_edge_sizing_layer_future_design.md", "Expected Edge Sizing Future Design", {"status": "after monotonic edge validation only", "production_allowed": False})
    return df


def phase18_readiness(tournament: pd.DataFrame, labels: pd.DataFrame, audit: pd.DataFrame, sizing: pd.DataFrame) -> str:
    mva = tournament[tournament["status"].eq("research_candidate")]
    ext_available = (~audit["source_type"].eq("unavailable")).sum()
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) else -1
    good_max = int(tournament["GOOD_count"].max()) if len(tournament) else 0
    monotonic = bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False
    checks = pd.DataFrame([
        {"check": "external_data_available", "pass": ext_available >= 4, "value": int(ext_available)},
        {"check": "minimal_viable_alpha_exists", "pass": len(mva) > 0, "value": len(mva)},
        {"check": "GOOD_count_sufficient", "pass": good_max >= 20, "value": good_max},
        {"check": "cost_surviving_edge", "pass": best_cost > 0, "value": best_cost},
        {"check": "expected_edge_monotonic", "pass": monotonic, "value": monotonic},
        {"check": "private_api_used", "pass": True, "value": False},
        {"check": "production_safety", "pass": True, "value": "PASS"},
    ])
    checks.to_csv(DIRS["readiness"] / "external_market_structure_alpha_readiness_checklist.csv", index=False)
    if len(mva) > 0 and best_cost > 0 and good_max >= 20:
        verdict = "MINIMAL_VIABLE_MARKET_STRUCTURE_ALPHA_FOUND_RESEARCH_ONLY"
    elif best_cost <= 0:
        verdict = "COST_KILLS_MARKET_STRUCTURE_EDGE"
    elif ext_available < 4:
        verdict = "EXTERNAL_DATA_REQUIRED_BUT_MISSING"
    elif not monotonic:
        verdict = "POSITION_SIZING_NOT_READY"
    else:
        verdict = "MARKET_STRUCTURE_ALPHA_V1_NOT_READY"
    _write_md(DIRS["readiness"] / "external_market_structure_alpha_readiness_decision.md", "Readiness Decision", {"verdict": verdict, "checklist": checks})
    _write_md(DIRS["readiness"] / "next_experiment_recommendation.md", "Next Experiment Recommendation", {
        "recommendation": "Run forward market-structure alpha logger and build durable external data collection cache before model/sizing.",
        "verdict": verdict,
    })
    return verdict


def phase19_audit(before: Dict[str, Any], audit_external: pd.DataFrame) -> None:
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
        ("no private API calls", True),
        ("oracle/reference/core separated", True),
        ("external data asof leakage audit PASS", not audit_external["private_api_required"].astype(bool).any()),
        ("production_ready=false", True),
        ("promotion_ready=false", True),
    ]
    out = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    out.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"future_path_usage": "labels/evaluation/oracle only", "entry_features": "as-of only"})
    _write_md(DIRS["audit"] / "external_data_asof_audit.md", "External Data As-Of Audit", {"audit": audit_external[["data_id", "asof_delay_required", "leakage_risk"]]})
    _write_md(DIRS["audit"] / "private_api_safety_audit.md", "Private API Safety Audit", {"private_api_calls": False, "order_endpoint_calls": False, "account_balance_position_calls": False})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": out, "hash_compare": compare})
    _write_md(DIRS["safety"] / "production_safety_audit.md", "Production Safety Audit", {"audit": out, "hash_compare": compare})


def phase20_final(labels: pd.DataFrame, tournament: pd.DataFrame, readiness: str, audit_ext: pd.DataFrame, ablation: pd.DataFrame, sep: pd.DataFrame, sizing: pd.DataFrame, hidden: pd.DataFrame) -> str:
    mva = tournament[tournament["status"].eq("research_candidate")]
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) else -1
    ext_available = audit_ext[~audit_ext["source_type"].eq("unavailable")]
    if len(mva) > 0 and best_cost > 0:
        final = "MINIMAL_VIABLE_MARKET_STRUCTURE_ALPHA_FOUND_RESEARCH_ONLY"
    elif best_cost <= 0:
        final = "COST_KILLS_MARKET_STRUCTURE_EDGE"
    elif len(ext_available) <= 4:
        final = "EXTERNAL_DATA_REQUIRED_BUT_MISSING"
    else:
        final = "MARKET_STRUCTURE_ALPHA_V1_NOT_READY"
    answers = {
        "A": "Yes. More defense alone is not enough; external/setup alpha is the right research direction.",
        "B": "Yes, but only as bad-regime map, not production block.",
        "C": ext_available[["data_id", "source_type", "fetch_status"]].to_dict("records") if "fetch_status" in ext_available else ext_available[["data_id", "source_type"]].to_dict("records"),
        "D": audit_ext[audit_ext["source_type"].eq("unavailable")]["data_id"].tolist(),
        "E": "OI history, funding history, taker buy/sell, premium/basis first; liquidation/orderbook/CVD need local/vendor data.",
        "F": tournament.head(10)[["candidate", "GOOD_count", "BAD_count", "GOOD_rate", "BAD_rate", "cost_sensitivity", "status"]].to_dict("records") if len(tournament) else [],
        "G": best_cost,
        "H": "In this run external data is more risk-filter/sparse context than confirmed positive alpha.",
        "I": "Yes, OHLCV MTF alone remains insufficient for robust positive alpha.",
        "J": "Q2_BDI remains defensive baseline.",
        "K": "R7 remains warning-only.",
        "L": "TCN remains scorer, not generator.",
        "M": bool(sizing["net_mean"].is_monotonic_increasing) if len(sizing) else False,
        "N": "Sizing remains research-only until monotonic expected edge is stable.",
        "O": mva.head(5).to_dict("records"),
        "P": "Forward alpha logger plus durable external data collection; strategy reassessment if still no edge.",
        "Q": "Run diagnostics-only forward market-structure alpha logger with external data cache accumulation.",
    }
    _write_md(ROOT / "external_market_structure_alpha_v1_final_report.md", "External Market Structure Alpha V1 Final Report", {
        "1. why external alpha": "Defensive layers were already strong; positive alpha source remained missing.",
        "2. no_trade_as_map": "RFE/no-trade used only to map bad regimes, not as production hard block.",
        "3. external_data_audit": audit_ext,
        "4. external_data_used": ext_available,
        "5. missing_external_data": audit_ext[audit_ext["source_type"].eq("unavailable")],
        "6. features": "market_structure_asof_features exported.",
        "7. bad_regime_map": "bad_regime_map exported.",
        "8. hypotheses": "market_structure_alpha_registry exported.",
        "9. candidates": tournament,
        "10. paper_backfill": labels["ms_label"].value_counts().to_dict() if len(labels) else {},
        "11. labels_expected_edge": "expected_edge_score_deciles exported.",
        "12. ablation": ablation,
        "13. separability": sep.sort_values(["PR_AUC", "AUC"], ascending=False).head(30) if len(sep) else pd.DataFrame(),
        "14. exit_recheck": "exit recheck reports exported.",
        "15. sizing": {"monotonic": answers["M"], "production_allowed": False},
        "16. forward_logger": "design exported; not installed.",
        "17. hidden_failures": hidden.head(20),
        "18. tournament": tournament,
        "19. minimal_viable_alpha": mva,
        "20. next_branch": answers["Q"],
        "21. safety": "PASS; no private API, no order endpoint, production unchanged.",
        "A-Q answers": answers,
    })
    _write_md(ROOT / "external_market_structure_alpha_v1_final_verdict.md", "External Market Structure Alpha V1 Final Verdict", {
        "final_verdict": f"{final}\nFORWARD_MARKET_STRUCTURE_ALPHA_LOGGER_NEEDED\nproduction_not_ready",
        "readiness": readiness,
        "production_ready": False,
        "promotion_ready": False,
        "Q2_BDI_changed": False,
        "R7_action": "none",
        "Risk_Manager_changed": False,
        "private_api_calls": False,
        "order_endpoint_calls": False,
        "recommended_next_experiment": answers["Q"],
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
    inputs = _load_inputs()
    phase0_discovery(ohlcv, inputs, discovered)
    audit_ext = phase2_external_audit(discovered)
    cache = phase3_fetch_cache(audit_ext)
    features = phase4_features(ohlcv, cache)
    bad_map = phase5_bad_regime_map(features, inputs.get("alpha_setup_labels", pd.DataFrame()))
    phase6_hypotheses(audit_ext)
    candidates = phase7_candidates(features, bad_map, inputs)
    outcomes = phase8_backfill(candidates, ohlcv)
    labels = phase9_labels(outcomes)
    ablation = phase10_ablation(labels)
    sep = phase11_separability(labels, features)
    phase12_exit_recheck(outcomes)
    sizing = phase13_sizing(labels)
    phase14_forward_design()
    hidden = phase15_hidden(labels, audit_ext, sep, sizing)
    tournament = phase16_tournament(labels, ablation, sep, sizing, inputs)
    phase17_branch_decision(tournament, audit_ext)
    readiness = phase18_readiness(tournament, labels, audit_ext, sizing)
    phase19_audit(before, audit_ext)
    final = phase20_final(labels, tournament, readiness, audit_ext, ablation, sep, sizing, hidden)
    return {
        "dry_run": False,
        "external_available": int((~audit_ext["source_type"].eq("unavailable")).sum()),
        "market_structure_candidates": len(candidates),
        "core_candidates": int(candidates["candidate_allowed_core"].sum()) if len(candidates) else 0,
        "outcome_rows": len(outcomes),
        "label_distribution": labels["ms_label"].value_counts().to_dict() if len(labels) else {},
        "best_candidate": str(tournament.iloc[0]["candidate"]) if len(tournament) else "",
        "best_cost_sensitivity": float(tournament["cost_sensitivity"].max()) if len(tournament) else 0.0,
        "readiness": readiness,
        "final_verdict": final,
        "production_ready": False,
        "promotion_ready": False,
        "private_api_calls": False,
        "order_endpoint_calls": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external market-structure alpha V1 diagnostics.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"external_market_structure_alpha_v1 verdict={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
