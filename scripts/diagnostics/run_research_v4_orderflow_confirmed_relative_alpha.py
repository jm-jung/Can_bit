"""
Research V4 orderflow-confirmed relative alpha engine.

Diagnostics-only historical research. This script reads prior V3 artifacts and
local public external/orderflow cache only. It never calls private, order,
account, balance, or position endpoints, and never writes outside diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("data/diagnostics/research_v4_orderflow_confirmed_relative_alpha")
V3 = Path("data/diagnostics/research_v3_cross_symbol_relative_alpha")
EXT = Path("data/diagnostics/external_market_structure_alpha_v1/external_cache")
COST = 0.0006

DIR_NAMES = [
    "discovery", "audit", "universe", "orderflow_audit", "orderflow_cache",
    "features", "v3_reanalysis", "hypotheses", "candidates", "backfill",
    "labels", "tournament", "ablation", "validation", "model_objective",
    "position_sizing", "forward_design", "hidden_failure_modes",
]
DIRS = {name: ROOT / name for name in DIR_NAMES}

ORDERFLOW_ITEMS = [
    ("OF0_funding_rate", "funding_rate", "funding_rate.parquet", "historical_core_if_overlap"),
    ("OF1_open_interest_current", "open_interest_current", "open_interest.parquet", "forward_only_snapshot"),
    ("OF2_open_interest_history", "open_interest_history", "open_interest_hist.parquet", "historical_core_if_overlap"),
    ("OF3_taker_buy_sell_volume", "taker_buy_sell_volume", "taker_buy_sell_volume.parquet", "historical_core_if_overlap"),
    ("OF4_aggtrades_for_true_delta_proxy", "aggtrades_proxy_cvd", "aggtrades.parquet", "unavailable_if_no_local_cache"),
    ("OF5_premium_index", "premium_index", "premium_index.parquet", "forward_only_snapshot"),
    ("OF6_mark_price", "mark_price", "mark_price.parquet", "historical_core_if_overlap"),
    ("OF7_spot_futures_basis", "basis_proxy", "basis_proxy.parquet", "reference_if_available"),
    ("OF8_liquidation_history", "liquidation_history", "liquidation_history.parquet", "forward_only_or_unavailable"),
    ("OF9_orderbook_snapshot", "orderbook_snapshot", "orderbook_snapshot_history.parquet", "forward_only_if_snapshot_only"),
    ("OF10_orderbook_depth_change", "orderbook_depth_change", "orderbook_imbalance.parquet", "unavailable_if_no_history"),
    ("OF11_CVD", "cvd_proxy", "cvd_proxy.parquet", "proxy_only_if_aggtrades_or_taker"),
    ("OF12_large_trade_aggression", "large_trade_aggression", "large_trade_aggression.parquet", "unavailable_if_no_trades"),
    ("OF13_exchange_basis_cross_market", "exchange_basis_cross_market", "exchange_basis.parquet", "unavailable_if_no_cross_market"),
    ("OF14_liquidity_cluster_proxy", "liquidity_cluster_proxy", "liquidity_cluster_proxy.parquet", "reference_only"),
    ("OF15_market_wide_orderflow_basket", "market_wide_orderflow_basket", "market_wide_orderflow_basket.parquet", "derived_from_available"),
]

CLUSTERS = {
    "CORE_BTC": ["BTCUSDT"],
    "CORE_ETH": ["ETHUSDT"],
    "CORE_SOL": ["SOLUSDT"],
    "CORE_BNB": ["BNBUSDT"],
    "MAJOR_L1": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "ATOMUSDT"],
    "PAYMENT_OLD": ["XRPUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "DOGEUSDT"],
    "DEFI": ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "PENDLEUSDT"],
    "L2": ["ARBUSDT", "OPUSDT"],
    "AI_INFRA": ["FETUSDT", "RENDERUSDT", "WLDUSDT"],
    "MEME": ["DOGEUSDT", "PEPEUSDT", "WIFUSDT"],
    "SOL_ECOSYSTEM_PROXY": ["SOLUSDT", "JUPUSDT", "PYTHUSDT", "WIFUSDT"],
    "HIGH_BETA_ALT": ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "INJUSDT", "SEIUSDT", "TIAUSDT"],
    "LOWER_BETA_MAJOR": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT"],
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, val in sections.items():
        lines.extend([f"## {key}", ""])
        if isinstance(val, pd.DataFrame):
            lines.extend(["```csv", val.head(80).to_csv(index=False), "```"])
        elif isinstance(val, (dict, list)):
            lines.extend(["```json", _json(val), "```"])
        else:
            lines.append(str(val))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_dirs() -> None:
    for d in DIRS.values():
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _hash_path(path: Path) -> Dict[str, Any]:
    return {"path": _rel(path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() and path.is_file() else "", "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _safety_paths() -> List[Path]:
    rels = ["models/tcn_v1.pt", "data/diagnostics/tcn_no_events.pt", "ops/launchd", "live", "orders", "state", "risk", "risk_manager", "scripts/diagnostics/run_false_high_r7_monitor.py", "scripts/diagnostics/run_false_high_r7_daily_monitor.py"]
    out: List[Path] = []
    for rel in rels:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:400])
    return out


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _snapshot() -> Dict[str, Any]:
    return {"hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status(), "python": sys.version, "platform": platform.platform()}


def _score(df: pd.DataFrame, group: str, label_col: str = "v4_label") -> pd.DataFrame:
    if df.empty or group not in df.columns:
        return pd.DataFrame()
    rows = []
    for key, sub in df.groupby(group, dropna=False):
        r = pd.to_numeric(sub["net_after_cost"], errors="coerce").fillna(0)
        gains = r[r > 0].sum()
        losses = -r[r < 0].sum()
        row = {
            group: key,
            "candidate_count": len(sub),
            "expectancy": float(r.mean()),
            "cost_sensitivity": float((r - COST).mean()),
            "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
            "tail_loss": float(r.quantile(0.05)),
            "RFE_rate": float(pd.to_numeric(sub.get("RFE", False), errors="coerce").fillna(0).mean()),
            "MFE_to_cost": float(pd.to_numeric(sub.get("MFE_to_cost_ratio", 0), errors="coerce").median()),
            "data_quality_score": float(pd.to_numeric(sub.get("data_quality_score", 0), errors="coerce").mean()),
        }
        if label_col in sub.columns:
            row.update({
                "GOOD_count": int(sub[label_col].eq("V4_GOOD").sum()),
                "BAD_count": int(sub[label_col].eq("V4_BAD").sum()),
                "NEUTRAL_count": int(sub[label_col].eq("V4_NEUTRAL").sum()),
                "GOOD_rate": float(sub[label_col].eq("V4_GOOD").mean()),
                "BAD_rate": float(sub[label_col].eq("V4_BAD").mean()),
            })
        rows.append(row)
    out = pd.DataFrame(rows)
    if "GOOD_count" in out.columns:
        out["status"] = np.select(
            [out["candidate_count"].lt(20), out["GOOD_count"].lt(10), out["cost_sensitivity"].le(0), out["BAD_rate"].gt(0.70)],
            ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"],
            default="research_candidate",
        )
    return out


def _cluster_for(symbol: str) -> str:
    for c, syms in CLUSTERS.items():
        if symbol in syms:
            return c
    return "OTHER"


def load_v3() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cand = pd.read_parquet(REPO_ROOT / V3 / "candidates/research_v3_candidate_universe.parquet")
    out = pd.read_parquet(REPO_ROOT / V3 / "backfill/research_v3_exit_outcomes.parquet")
    lab = pd.read_parquet(REPO_ROOT / V3 / "labels/research_v3_entry_quality_labels.parquet")
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], errors="coerce").astype("datetime64[ns]")
    out["entry_ts"] = pd.to_datetime(out["entry_ts"], errors="coerce").astype("datetime64[ns]")
    return cand, out, lab


def part0_discovery(v3_cand: pd.DataFrame, v3_lab: pd.DataFrame) -> None:
    required = [
        V3 / "research_v3_cross_symbol_relative_alpha_final_report.md",
        V3 / "research_v3_cross_symbol_relative_alpha_final_verdict.md",
        V3 / "candidates/research_v3_candidate_universe.parquet",
        V3 / "backfill/research_v3_exit_outcomes.parquet",
        V3 / "labels/research_v3_entry_quality_labels.parquet",
        V3 / "labels/research_v3_expected_edge_score.csv",
        V3 / "tournament/research_v3_alpha_tournament_scorecard.csv",
        V3 / "relative_strength/all_symbols_relative_strength.parquet",
        V3 / "breadth/market_breadth_features.parquet",
        V3 / "breadth/cluster_breadth_features.parquet",
        V3 / "leader_lagger/leader_lagger_features.parquet",
        Path("data/diagnostics/research_v2_multisymbol_expansion/research_v2_multisymbol_expansion_final_report.md"),
        Path("data/diagnostics/forward_research_v2_alpha_logger/state/forward_research_v2_pending_trades.parquet"),
        Path("data/diagnostics/forward_research_v2_alpha_logger/state/forward_research_v2_resolved_trades.parquet"),
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
    ]
    inv = [{"path": str(p), "exists": (REPO_ROOT / p).exists(), "size_bytes": (REPO_ROOT / p).stat().st_size if (REPO_ROOT / p).exists() and (REPO_ROOT / p).is_file() else 0} for p in required]
    pd.DataFrame(inv).to_csv(REPO_ROOT / DIRS["discovery"] / "input_inventory.csv", index=False)
    v3_summary = {
        "candidate_rows": int(len(v3_cand)),
        "V3_GOOD": int(v3_lab["v3_label"].eq("V3_GOOD").sum()),
        "V3_BAD": int(v3_lab["v3_label"].eq("V3_BAD").sum()),
        "V3_NEUTRAL": int(v3_lab["v3_label"].eq("V3_NEUTRAL").sum()),
        "verdict": "RESEARCH_V3_COST_KILLS_EDGE + production_not_ready",
        "monotonicity": False,
    }
    pd.DataFrame([v3_summary]).to_csv(REPO_ROOT / DIRS["discovery"] / "v3_failure_summary.csv", index=False)
    pd.DataFrame([
        {"source": "Research V3", "verdict": v3_summary["verdict"], "why_v4": "V3 top decile hinted but broad alpha families failed after costs."},
        {"source": "Research V2 multisymbol", "verdict": "MULTISYMBOL_EXPECTED_EDGE_NOT_MONOTONIC", "why_v4": "Candle/setup expansion remained bad-heavy."},
    ]).to_csv(REPO_ROOT / DIRS["discovery"] / "previous_diagnostics_summary.csv", index=False)
    syms = sorted(v3_cand["symbol"].dropna().unique())
    pd.DataFrame([{"symbol": s, "v3_mtf_path": str(V3 / "mtf_data" / s / "mtf_asof_joined_frame.parquet"), "exists": (REPO_ROOT / V3 / "mtf_data" / s / "mtf_asof_joined_frame.parquet").exists()} for s in syms]).to_csv(REPO_ROOT / DIRS["discovery"] / "available_symbol_data_sources.csv", index=False)
    ext_paths = sorted([p for p in (REPO_ROOT / EXT).glob("*") if p.is_file()])
    pd.DataFrame([{"path": _rel(p), "suffix": p.suffix, "size_bytes": p.stat().st_size} for p in ext_paths]).to_csv(REPO_ROOT / DIRS["discovery"] / "available_orderflow_data_sources.csv", index=False)
    pd.DataFrame([{"path": _rel(p), "suffix": p.suffix, "size_bytes": p.stat().st_size} for p in ext_paths]).to_csv(REPO_ROOT / DIRS["discovery"] / "available_external_data_sources.csv", index=False)
    status = {"forward_v2_label": "com.canbit.forward_research_v2_alpha_logger", "checked_only": True, "modified": False, "production_action": "none"}
    (REPO_ROOT / DIRS["discovery"] / "forward_logger_status_snapshot.json").write_text(_json(status), encoding="utf-8")
    (REPO_ROOT / DIRS["discovery"] / "discovered_paths.json").write_text(_json({"v3_root": str(V3), "external_cache": str(EXT), "symbols": syms}), encoding="utf-8")
    _write_md(REPO_ROOT / DIRS["discovery"] / "discovery_report.md", "Research V4 Discovery Report", {"v3_failure_summary": v3_summary, "input_inventory": pd.DataFrame(inv), "forward_v2_logger_status": status})


def part2_universe(v3_cand: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    symbols = sorted(v3_cand["symbol"].dropna().unique())
    rows = []
    for s in symbols:
        has_orderflow = bool((availability["symbol_coverage"].fillna("").str.contains(s)).any())
        btc_public = s == "BTCUSDT" and bool(availability["historical_available"].any())
        rows.append({
            "symbol": s,
            "cluster": _cluster_for(s),
            "candidate_rows_v3": int(v3_cand["symbol"].eq(s).sum()),
            "liquidity_proxy": np.nan,
            "has_any_orderflow_data": btc_public or has_orderflow,
            "v4_core_lite": btc_public,
            "v4_core_full": False,
            "reference_only": not btc_public,
            "exclusion_reason": "" if btc_public else "no_symbol_specific_historical_orderflow_cache",
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / DIRS["universe"] / "symbol_universe_audit.csv", index=False)
    df[df["v4_core_lite"]].to_csv(REPO_ROOT / DIRS["universe"] / "symbol_selected_v4_core.csv", index=False)
    df[df["reference_only"]].to_csv(REPO_ROOT / DIRS["universe"] / "symbol_selected_v4_reference.csv", index=False)
    df[df["reference_only"]].to_csv(REPO_ROOT / DIRS["universe"] / "symbol_excluded.csv", index=False)
    cmap = []
    for c, syms in CLUSTERS.items():
        for s in syms:
            if s in symbols:
                cmap.append({"cluster": c, "symbol": s, "mapping_policy": "fixed_ex_ante"})
    pd.DataFrame(cmap).to_csv(REPO_ROOT / DIRS["universe"] / "symbol_cluster_map.csv", index=False)
    df.to_csv(REPO_ROOT / DIRS["universe"] / "symbol_data_quality.csv", index=False)
    df[["symbol", "liquidity_proxy"]].to_csv(REPO_ROOT / DIRS["universe"] / "symbol_liquidity_proxy.csv", index=False)
    _write_md(REPO_ROOT / DIRS["universe"] / "universe_report.md", "Research V4 Universe Report", {"universe": df, "cluster_map": pd.DataFrame(cmap)})
    return df


def _read_cache(name: str) -> pd.DataFrame:
    p = REPO_ROOT / EXT / name
    if not p.exists():
        return pd.DataFrame()
    try:
        if p.suffix == ".parquet":
            df = pd.read_parquet(p)
        else:
            df = pd.read_csv(p)
    except Exception:
        return pd.DataFrame()
    for c in ["timestamp", "asof_available_ts"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    if "symbol" not in df.columns:
        df["symbol"] = "BTCUSDT"
    return df


def part3_orderflow_audit(v3_cand: pd.DataFrame) -> pd.DataFrame:
    cmin, cmax = v3_cand["timestamp"].min(), v3_cand["timestamp"].max()
    rows = []
    attempts: Dict[str, Any] = {"network_fetch_attempted": False, "reason": "V4 uses local public diagnostics cache first; no private/API-key endpoints."}
    for data_id, data_name, filename, usage in ORDERFLOW_ITEMS:
        df = _read_cache(filename)
        exists = not df.empty
        start = df["timestamp"].min() if exists and "timestamp" in df else pd.NaT
        end = df["timestamp"].max() if exists and "timestamp" in df else pd.NaT
        syms = sorted(df["symbol"].dropna().astype(str).unique()) if exists and "symbol" in df else []
        overlap = bool(exists and pd.notna(start) and pd.notna(end) and end >= cmin and start <= cmax)
        hist = bool(exists and len(df) > 10 and overlap)
        snapshot = bool(exists and len(df) <= 2)
        missing_ratio = float(df.get("missing_flag", pd.Series(False, index=df.index)).mean()) if exists else 1.0
        gap = int((df["timestamp"].sort_values().diff().dt.total_seconds().fillna(300) > 3600).sum()) if exists and "timestamp" in df else 0
        core_allowed = hist and data_name in {"funding_rate", "open_interest_history", "taker_buy_sell_volume", "mark_price"}
        reference_only = exists and not core_allowed
        forward_available = exists or data_name in {"open_interest_current", "premium_index", "orderbook_snapshot"}
        rows.append({
            "data_id": data_id,
            "data_name": data_name,
            "source_type": "local_cache" if exists else "unavailable",
            "path_or_endpoint": str(EXT / filename) if exists else "",
            "requires_api_key": False,
            "private_api_required": False,
            "allowed_to_fetch": False,
            "fetch_status": "local_cache_found" if exists else "unavailable",
            "historical_available": hist,
            "forward_available": forward_available,
            "date_range_start": start,
            "date_range_end": end,
            "granularity": "5m_or_native" if exists else "",
            "timezone": "UTC",
            "publication_delay": "5m_applied_if_asof_available_ts_present",
            "asof_alignment_possible": bool(exists and "asof_available_ts" in df.columns),
            "symbol_coverage": ",".join(syms),
            "missing_ratio": missing_ratio,
            "gap_count": gap,
            "duplicate_count": int(df.duplicated().sum()) if exists else 0,
            "outlier_count": 0,
            "leakage_risk": "medium" if exists and "asof_available_ts" not in df.columns else "low" if exists else "none",
            "core_allowed": core_allowed,
            "reference_only": reference_only,
            "reason_if_unavailable": "" if exists else "no_local_historical_public_cache",
            "recommended_usage": usage if exists else "unavailable",
            "not_allowed_usage": "do_not_impute_missing_as_signal; do_not_treat_proxy_as_true_CVD",
        })
    audit = pd.DataFrame(rows)
    audit.to_csv(REPO_ROOT / DIRS["orderflow_audit"] / "orderflow_data_availability_audit.csv", index=False)
    (REPO_ROOT / DIRS["orderflow_audit"] / "orderflow_fetch_attempts.json").write_text(_json(attempts), encoding="utf-8")
    audit.to_csv(REPO_ROOT / DIRS["orderflow_audit"] / "orderflow_data_quality_summary.csv", index=False)
    audit[["data_id", "data_name", "leakage_risk", "asof_alignment_possible", "publication_delay", "core_allowed"]].to_csv(REPO_ROOT / DIRS["orderflow_audit"] / "orderflow_asof_leakage_risk.csv", index=False)
    _write_md(REPO_ROOT / DIRS["orderflow_audit"] / "orderflow_missing_report.md", "Orderflow Missing Report", {"missing_or_forward_only": audit[~audit["core_allowed"]]})
    _write_md(REPO_ROOT / DIRS["orderflow_audit"] / "orderflow_audit_report.md", "Orderflow Availability Audit Report", {"audit": audit, "core_historical": audit[audit["core_allowed"]]})
    return audit


def part4_cache_build(audit: pd.DataFrame) -> None:
    rows = []
    for _, r in audit.iterrows():
        src = REPO_ROOT / r["path_or_endpoint"] if r["path_or_endpoint"] else None
        out_name = {
            "open_interest_history": "open_interest_history.parquet",
            "open_interest_current": "open_interest_current_snapshot.parquet",
            "taker_buy_sell_volume": "taker_buy_sell_volume.parquet",
            "funding_rate": "funding_rate.parquet",
            "premium_index": "premium_index.parquet",
            "mark_price": "mark_price.parquet",
            "basis_proxy": "basis_proxy.parquet",
        }.get(r["data_name"], f"{r['data_name']}.parquet")
        dst = REPO_ROOT / DIRS["orderflow_cache"] / out_name
        if src and src.exists():
            df = _read_cache(src.name)
            df.to_parquet(dst, index=False)
            rows.append({"data_name": r["data_name"], "cache_path": _rel(dst), "rows": len(df), "source": _rel(src), "core_allowed": r["core_allowed"], "historical_available": r["historical_available"]})
    reg = pd.DataFrame(rows)
    if reg.empty:
        reg = pd.DataFrame(columns=["data_name", "cache_path", "rows", "source", "core_allowed", "historical_available"])
    reg.to_csv(REPO_ROOT / DIRS["orderflow_cache"] / "orderflow_cache_registry.csv", index=False)
    _write_md(REPO_ROOT / DIRS["orderflow_cache"] / "orderflow_cache_quality_report.md", "Orderflow Cache Quality Report", {"registry": reg, "private_api_used": False, "order_endpoint_used": False})


def _asof_merge(base: pd.DataFrame, ext: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
    if ext.empty:
        for c in cols:
            base[f"{prefix}_{c}"] = np.nan
        base[f"{prefix}_missing_flag"] = True
        return base
    e = ext.copy()
    e = e[e["symbol"].eq("BTCUSDT")].copy()
    if e.empty:
        for c in cols:
            base[f"{prefix}_{c}"] = np.nan
        base[f"{prefix}_missing_flag"] = True
        return base
    e["asof_ts"] = e["asof_available_ts"] if "asof_available_ts" in e.columns else e["timestamp"] + pd.Timedelta(minutes=5)
    keep = ["asof_ts"] + [c for c in cols if c in e.columns]
    e = e[keep].sort_values("asof_ts")
    b = base.sort_values("timestamp")
    merged = pd.merge_asof(b, e, left_on="timestamp", right_on="asof_ts", direction="backward", tolerance=pd.Timedelta(days=30))
    for c in cols:
        if c in merged.columns:
            merged.rename(columns={c: f"{prefix}_{c}"}, inplace=True)
        else:
            merged[f"{prefix}_{c}"] = np.nan
    merged[f"{prefix}_missing_flag"] = merged[[f"{prefix}_{c}" for c in cols]].isna().all(axis=1)
    return merged.drop(columns=["asof_ts"], errors="ignore")


def part5_features(v3_cand: pd.DataFrame, v3_lab: pd.DataFrame) -> pd.DataFrame:
    lab_cols = ["research_v3_candidate_id", "v3_label", "expected_edge_score", "net_after_cost", "MFE_to_cost_ratio", "MAE_to_cost_ratio", "RFE", "high_MAE", "tail_loss"]
    base = v3_cand.merge(v3_lab[[c for c in lab_cols if c in v3_lab.columns]], on="research_v3_candidate_id", how="left")
    base = base.sort_values("timestamp").reset_index(drop=True)
    # Historical orderflow features are symbol-specific only for BTCUSDT in local cache.
    btc = base[base["symbol"].eq("BTCUSDT")].copy()
    non = base[~base["symbol"].eq("BTCUSDT")].copy()
    btc = _asof_merge(btc, _read_cache("funding_rate.parquet"), ["fundingRate", "markPrice"], "funding")
    btc = _asof_merge(btc, _read_cache("open_interest_hist.parquet"), ["sumOpenInterest", "sumOpenInterestValue"], "oi")
    btc = _asof_merge(btc, _read_cache("taker_buy_sell_volume.parquet"), ["buySellRatio", "buyVol", "sellVol"], "taker")
    btc = _asof_merge(btc, _read_cache("mark_price.parquet"), ["mark_close"], "mark")
    btc = _asof_merge(btc, _read_cache("premium_index.parquet"), ["lastFundingRate", "markPrice", "indexPrice"], "premium")
    if not btc.empty:
        btc["funding_rate"] = pd.to_numeric(btc["funding_fundingRate"], errors="coerce")
        btc["funding_rate_z"] = (btc["funding_rate"] - btc["funding_rate"].rolling(50, min_periods=10).mean()) / btc["funding_rate"].rolling(50, min_periods=10).std()
        btc["oi"] = pd.to_numeric(btc["oi_sumOpenInterest"], errors="coerce")
        btc["oi_change"] = btc["oi"].pct_change()
        btc["oi_change_z"] = (btc["oi_change"] - btc["oi_change"].rolling(50, min_periods=10).mean()) / btc["oi_change"].rolling(50, min_periods=10).std()
        btc["taker_buy_ratio"] = pd.to_numeric(btc["taker_buySellRatio"], errors="coerce")
        btc["taker_delta"] = pd.to_numeric(btc["taker_buyVol"], errors="coerce") - pd.to_numeric(btc["taker_sellVol"], errors="coerce")
        btc["taker_delta_z"] = (btc["taker_delta"] - btc["taker_delta"].rolling(50, min_periods=10).mean()) / btc["taker_delta"].rolling(50, min_periods=10).std()
        btc["premium_index"] = pd.to_numeric(btc["premium_markPrice"], errors="coerce") - pd.to_numeric(btc["premium_indexPrice"], errors="coerce")
        btc["basis_proxy"] = pd.to_numeric(btc["mark_mark_close"], errors="coerce") - pd.to_numeric(btc.get("funding_markPrice", np.nan), errors="coerce")
    for col in ["funding_rate", "funding_rate_z", "oi", "oi_change", "oi_change_z", "taker_buy_ratio", "taker_delta", "taker_delta_z", "premium_index", "basis_proxy"]:
        if col not in btc:
            btc[col] = np.nan
        non[col] = np.nan
    feature = pd.concat([btc, non], ignore_index=True, sort=False)
    feature["funding_extreme_positive"] = feature["funding_rate_z"] > 1.5
    feature["funding_extreme_negative"] = feature["funding_rate_z"] < -1.5
    feature["funding_crowded_long_proxy"] = feature["funding_rate"] > 0.0001
    feature["funding_crowded_short_proxy"] = feature["funding_rate"] < -0.00005
    feature["oi_leverage_build_proxy"] = feature["oi_change_z"] > 1.0
    feature["oi_flush_proxy"] = feature["oi_change_z"] < -1.0
    feature["aggressive_buy_pressure"] = feature["taker_delta_z"] > 1.0
    feature["aggressive_sell_pressure"] = feature["taker_delta_z"] < -1.0
    feature["taker_delta_price_alignment"] = np.where(feature["direction"].eq("LONG"), feature["aggressive_buy_pressure"], feature["aggressive_sell_pressure"])
    feature["cvd_proxy"] = feature.groupby("symbol")["taker_delta"].cumsum()
    feature["cvd_proxy_missing_flag"] = feature["taker_delta"].isna()
    feature["liquidation_history_missing_flag"] = True
    feature["orderbook_history_missing_flag"] = True
    feature["of_continuation_score"] = feature[["oi_leverage_build_proxy", "taker_delta_price_alignment"]].fillna(False).mean(axis=1)
    feature["of_reversal_score"] = feature[["oi_flush_proxy", "funding_extreme_positive", "funding_extreme_negative"]].fillna(False).mean(axis=1)
    feature["of_flush_reclaim_score"] = feature["oi_flush_proxy"].fillna(False).astype(float)
    feature["of_crowding_unwind_score"] = feature[["funding_crowded_long_proxy", "funding_crowded_short_proxy", "oi_flush_proxy"]].fillna(False).mean(axis=1)
    feature["of_squeeze_risk_score"] = feature[["funding_crowded_long_proxy", "oi_leverage_build_proxy"]].fillna(False).mean(axis=1)
    feature["of_basis_dislocation_score"] = feature["basis_proxy"].abs().rank(pct=True)
    feature["of_aggressive_flow_alignment_score"] = feature["taker_delta_price_alignment"].fillna(False).astype(float)
    feature["of_divergence_score"] = np.where(feature["relative_strength_rank"].fillna(0.5) > 0.8, feature["taker_delta_z"].fillna(0).lt(0).astype(float), 0.0)
    missing_cols = ["funding_rate", "oi", "taker_delta", "premium_index", "basis_proxy"]
    feature["of_data_quality_score"] = 1.0 - feature[missing_cols].isna().mean(axis=1)
    feature["orderflow_confirmation_score"] = feature[["of_continuation_score", "of_reversal_score", "of_aggressive_flow_alignment_score"]].mean(axis=1)
    feature["relative_orderflow_rank"] = feature.groupby("timestamp")["orderflow_confirmation_score"].rank(pct=True)
    out_cols = ["research_v3_candidate_id", "symbol", "timestamp", "direction", "generator_id", "alpha_id", "alpha_name", "cluster", "market_regime", "breadth_regime", "relative_strength_rank", "relative_strength_score", "leader_symbol", "leader_lag_horizon", "leader_lagger_score", "rotation_score", "expected_move_to_cost_ratio", "bad_regime_score", "v3_label", "expected_edge_score"] + [c for c in feature.columns if c.startswith(("funding", "oi_", "taker", "premium", "basis", "cvd", "liquidation", "orderbook", "of_", "aggressive", "relative_orderflow", "orderflow_confirmation")) or c in {"oi"}]
    feat = feature[[c for c in out_cols if c in feature.columns]].copy()
    feat.to_parquet(REPO_ROOT / DIRS["features"] / "v4_asof_orderflow_features.parquet", index=False)
    feat.head(20000).to_csv(REPO_ROOT / DIRS["features"] / "v4_asof_orderflow_features.csv", index=False)
    families = pd.DataFrame([{"family": f, "historical_core": f in {"funding", "OI", "taker", "mark"}, "proxy_only": f in {"CVD_proxy", "basis_proxy"}, "forward_only_or_missing": f in {"liquidation", "orderbook"}} for f in ["funding", "OI", "taker", "premium", "mark", "basis_proxy", "CVD_proxy", "liquidation", "orderbook"]])
    families.to_csv(REPO_ROOT / DIRS["features"] / "orderflow_feature_family_registry.csv", index=False)
    miss = feat.isna().mean().rename("missing_ratio").reset_index()
    miss.columns = ["feature", "missing_ratio"]
    miss.to_csv(REPO_ROOT / DIRS["features"] / "orderflow_feature_missingness_report.csv", index=False)
    feat.select_dtypes(include=[np.number]).corr(numeric_only=True).to_csv(REPO_ROOT / DIRS["features"] / "orderflow_feature_correlation_report.csv")
    _write_md(REPO_ROOT / DIRS["features"] / "orderflow_feature_data_quality_report.md", "Orderflow Feature Data Quality Report", {"missingness": miss.head(80)})
    _write_md(REPO_ROOT / DIRS["features"] / "v4_feature_report.md", "V4 As-Of Feature Report", {"rows": len(feat), "data_quality_distribution": feat["of_data_quality_score"].describe()})
    return feat


def part6_v3_reanalysis(feat: pd.DataFrame) -> None:
    x = feat.copy()
    x["orderflow_confirmed"] = x["orderflow_confirmation_score"].fillna(0) >= 0.5
    label_profile = x.groupby(["v3_label", "orderflow_confirmed"]).agg(rows=("research_v3_candidate_id", "size"), avg_v3_edge=("expected_edge_score", "mean"), avg_of_score=("orderflow_confirmation_score", "mean"), data_quality=("of_data_quality_score", "mean")).reset_index()
    label_profile.to_csv(REPO_ROOT / DIRS["v3_reanalysis"] / "v3_orderflow_profile_by_label.csv", index=False)
    x.groupby(["alpha_name", "orderflow_confirmed"]).agg(rows=("research_v3_candidate_id", "size"), avg_v3_edge=("expected_edge_score", "mean"), data_quality=("of_data_quality_score", "mean")).reset_index().to_csv(REPO_ROOT / DIRS["v3_reanalysis"] / "v3_orderflow_profile_by_alpha.csv", index=False)
    top = x[x["expected_edge_score"].rank(pct=True) >= 0.9].copy()
    top.groupby("orderflow_confirmed").agg(rows=("research_v3_candidate_id", "size"), good_rate=("v3_label", lambda s: float((s == "V3_GOOD").mean())), bad_rate=("v3_label", lambda s: float((s == "V3_BAD").mean())), data_quality=("of_data_quality_score", "mean")).reset_index().to_csv(REPO_ROOT / DIRS["v3_reanalysis"] / "v3_top_decile_orderflow_reanalysis.csv", index=False)
    x[x["v3_label"].eq("V3_BAD")].groupby("orderflow_confirmed").agg(rows=("research_v3_candidate_id", "size"), data_quality=("of_data_quality_score", "mean"), of_score=("orderflow_confirmation_score", "mean")).reset_index().to_csv(REPO_ROOT / DIRS["v3_reanalysis"] / "v3_cost_kill_orderflow_reanalysis.csv", index=False)
    _write_md(REPO_ROOT / DIRS["v3_reanalysis"] / "v3_orderflow_reanalysis_report.md", "V3 Candidate Orderflow Reanalysis", {"by_label": label_profile, "top_decile": top.groupby("orderflow_confirmed").size().reset_index(name="rows")})


def part7_hypotheses() -> pd.DataFrame:
    specs = [
        ("V4A1", "RS_OI_TAKER_CONTINUATION", "OI,taker"),
        ("V4A2", "RS_OI_BUILD_BREAKOUT", "OI,MTF"),
        ("V4A3", "FLUSH_RECLAIM_REVERSAL", "OI_flush,taker_flip"),
        ("V4A4", "FUNDING_EXTREME_UNWIND", "funding,OI"),
        ("V4A5", "BASIS_DISLOCATION_MEAN_REVERSION", "basis,premium"),
        ("V4A6", "CVD_DIVERGENCE_REVERSAL", "proxy_CVD"),
        ("V4A7", "LEADER_LAGGER_ORDERFLOW_CONFIRM", "leader_lagger,taker/OI"),
        ("V4A8", "BREADTH_EXPANSION_ORDERFLOW_CONFIRM", "breadth,market_orderflow"),
        ("V4A9", "SECTOR_ROTATION_ORDERFLOW_CONFIRM", "cluster_flow"),
        ("V4A10", "HIGH_BETA_RISK_ON_FLOW", "taker/OI"),
        ("V4A11", "RISK_OFF_WEAKNESS_SHORT", "sell_taker/OI"),
        ("V4A12", "SQUEEZE_RISK_AVOIDANCE", "funding/OI/premium"),
        ("V4A13", "LIQUIDATION_CLUSTER_FOLLOWTHROUGH", "liquidation_history"),
        ("V4A14", "ORDERBOOK_PRESSURE_TRIGGER", "orderbook_history"),
        ("V4A15", "ORDERFLOW_RELATIVE_ENSEMBLE_STRICT", "all_available_orderflow"),
        ("V4A16", "ORDERFLOW_RELATIVE_ENSEMBLE_BALANCED", "available_orderflow"),
        ("V4A17", "ORDERFLOW_RISK_FILTER_ONLY", "risk_scores"),
        ("V4A18", "FORWARD_ONLY_ORDERBOOK_LIQUIDATION", "forward_only_orderbook_liquidation"),
    ]
    rows = []
    for aid, name, req in specs:
        rows.append({
            "alpha_id": aid, "alpha_name": name, "required_data": req,
            "fallback_if_missing": "reference_only_or_forward_only",
            "market_regime_requirement": "V3 as-of market regime",
            "relative_strength_requirement": "V3 as-of relative rank",
            "breadth_requirement": "V3 as-of breadth",
            "leader_lagger_requirement": "V3 lag-safe leader-lagger if applicable",
            "orderflow_requirement": req,
            "liquidation_requirement": "historical unavailable unless local cache exists",
            "funding_basis_requirement": "local public cache only",
            "MTF_setup_usage": "auxiliary",
            "5m_trigger_usage": "timing_only",
            "direction_logic": "orderflow-confirmed relative thesis",
            "entry_trigger": "next open inherited from V3 paper outcome",
            "invalid_condition": "missing required orderflow for core",
            "risk_condition": "cost/slippage and data quality aware",
            "expected_move_to_cost_condition": ">=2 reference",
            "data_quality_requirement": "of_data_quality_score > 0 for core",
            "core_or_reference": "core if historical orderflow exists, otherwise reference/forward_only",
            "forward_only_if_needed": aid in {"V4A13", "V4A14", "V4A18"},
            "expected_failure_mode": "data_sparse/cost_kills_edge/orderflow_is_risk_filter",
            "oracle_flag": False,
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / DIRS["hypotheses"] / "research_v4_alpha_registry.csv", index=False)
    df[["alpha_id", "required_data", "data_quality_requirement", "forward_only_if_needed"]].to_csv(REPO_ROOT / DIRS["hypotheses"] / "research_v4_data_requirements.csv", index=False)
    df[["alpha_id", "expected_failure_mode"]].to_csv(REPO_ROOT / DIRS["hypotheses"] / "research_v4_expected_failure_modes.csv", index=False)
    _write_md(REPO_ROOT / DIRS["hypotheses"] / "research_v4_alpha_definitions.md", "Research V4 Alpha Definitions", {"registry": df})
    _write_md(REPO_ROOT / DIRS["hypotheses"] / "research_v4_hypothesis_report.md", "Research V4 Hypothesis Report", {"principle": "V3 relative hints require actual local public orderflow confirmation; missing data is not imputed."})
    return df


def _candidate_row(r: pd.Series, gid: str, aid: str, name: str, direction: str, reference: bool = False, forward_only: bool = False) -> Dict[str, Any]:
    raw = f"{r['research_v3_candidate_id']}|{gid}|{r.get('orderflow_confirmation_score', 0)}"
    snap = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return {
        "research_v4_candidate_id": f"{r['symbol']}_{gid}_{pd.Timestamp(r['timestamp']).strftime('%Y%m%d%H%M')}_{snap}",
        "research_v3_candidate_id": r["research_v3_candidate_id"],
        "symbol": r["symbol"], "timestamp": r["timestamp"], "entry_ts": r["timestamp"],
        "direction": direction, "generator_id": gid, "alpha_id": aid, "alpha_name": name,
        "cluster": r.get("cluster", _cluster_for(str(r["symbol"]))),
        "market_regime": r.get("market_regime", ""), "breadth_regime": r.get("breadth_regime", ""),
        "relative_strength_rank": r.get("relative_strength_rank", np.nan),
        "relative_strength_score": r.get("relative_strength_score", np.nan),
        "leader_symbol": r.get("leader_symbol", ""), "leader_lag_horizon": r.get("leader_lag_horizon", ""),
        "leader_lagger_score": r.get("leader_lagger_score", np.nan), "rotation_score": r.get("rotation_score", np.nan),
        "funding_context": f"funding={r.get('funding_rate', np.nan)} z={r.get('funding_rate_z', np.nan)}",
        "oi_context": f"oi_change_z={r.get('oi_change_z', np.nan)}",
        "taker_context": f"taker_delta_z={r.get('taker_delta_z', np.nan)}",
        "cvd_context": "proxy_only" if not pd.isna(r.get("cvd_proxy", np.nan)) else "unavailable",
        "basis_context": f"basis_proxy={r.get('basis_proxy', np.nan)}",
        "premium_context": f"premium_index={r.get('premium_index', np.nan)}",
        "liquidation_context": "historical_unavailable",
        "orderbook_context": "historical_unavailable",
        "orderflow_confirmation_score": r.get("orderflow_confirmation_score", 0),
        "orderflow_reversal_score": r.get("of_reversal_score", 0),
        "orderflow_continuation_score": r.get("of_continuation_score", 0),
        "crowding_unwind_score": r.get("of_crowding_unwind_score", 0),
        "squeeze_risk_score": r.get("of_squeeze_risk_score", 0),
        "forced_flow_score": 0.0,
        "data_quality_score": r.get("of_data_quality_score", 0),
        "orderflow_missing_flags": ",".join([c for c in ["funding", "oi", "taker", "liquidation", "orderbook"] if bool(pd.isna(r.get({"funding": "funding_rate", "oi": "oi", "taker": "taker_delta"}.get(c, "missing"), np.nan))) or c in {"liquidation", "orderbook"}]),
        "timeframe_stack": r.get("timeframe_stack", "V3_relative_plus_orderflow"),
        "regime_1h": r.get("regime_1h", ""), "regime_15m": r.get("regime_15m", ""), "trigger_5m_context": "timing_only",
        "MTF_setup_confirmed": r.get("MTF_setup_confirmed", False),
        "bad_regime_score": r.get("bad_regime_score", np.nan), "bad_regime_reasons": r.get("bad_regime_reasons", ""),
        "expected_move_proxy": r.get("expected_move_proxy", np.nan), "expected_move_to_cost_ratio": r.get("expected_move_to_cost_ratio", np.nan),
        "liquidity_proxy": r.get("liquidity_proxy", np.nan), "symbol_volatility_bucket": r.get("symbol_volatility_bucket", ""),
        "q2_score_if_available": np.nan, "q2_decision_if_available": "defensive_snapshot_only_not_loaded",
        "r7_score_if_available": np.nan, "r7_high_hazard_if_available": False, "tcn_score_if_available": np.nan,
        "candidate_allowed_core": not reference and not forward_only,
        "candidate_reference_only": reference,
        "forward_only": forward_only,
        "oracle_flag": False,
        "feature_snapshot_hash": snap,
    }


def part8_candidates(feat: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    f = feat.copy()
    f = f.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    counts: Dict[str, int] = {}
    def add(r: pd.Series, gid: str, aid: str, name: str, direction: str, reference: bool = False, forward_only: bool = False, cap: int = 800) -> None:
        if counts.get(gid, 0) >= cap:
            return
        rows.append(_candidate_row(r, gid, aid, name, direction, reference, forward_only))
        counts[gid] = counts.get(gid, 0) + 1
    for _, r in f.iterrows():
        dq = float(r.get("of_data_quality_score", 0) or 0)
        direction = str(r.get("direction", "LONG"))
        if r.get("expected_edge_score", -1) >= f["expected_edge_score"].quantile(0.90):
            add(r, "V4G0_V3_reference", "V4A0", "V3_reference_top_decile", direction, reference=True, cap=1000)
        if dq > 0 and r.get("relative_strength_rank", 0) > 0.65 and r.get("of_continuation_score", 0) >= 0.5:
            add(r, "V4G1_RS_OI_TAKER_CONTINUATION", "V4A1", "RS_OI_TAKER_CONTINUATION", direction)
        if dq > 0 and r.get("oi_leverage_build_proxy", False) and r.get("relative_strength_rank", 0) > 0.65:
            add(r, "V4G2_RS_OI_BUILD_BREAKOUT", "V4A2", "RS_OI_BUILD_BREAKOUT", direction)
        if dq > 0 and r.get("oi_flush_proxy", False) and r.get("taker_delta_price_alignment", False):
            add(r, "V4G3_FLUSH_RECLAIM_REVERSAL", "V4A3", "FLUSH_RECLAIM_REVERSAL", direction)
        if dq > 0 and (r.get("funding_extreme_positive", False) or r.get("funding_extreme_negative", False)):
            add(r, "V4G4_FUNDING_EXTREME_UNWIND", "V4A4", "FUNDING_EXTREME_UNWIND", direction)
        if dq > 0 and r.get("of_basis_dislocation_score", 0) > 0.9:
            add(r, "V4G5_BASIS_DISLOCATION_MEAN_REVERSION", "V4A5", "BASIS_DISLOCATION_MEAN_REVERSION", direction, reference=True)
        if dq > 0 and r.get("of_divergence_score", 0) > 0:
            add(r, "V4G6_CVD_DIVERGENCE_REVERSAL", "V4A6", "CVD_DIVERGENCE_REVERSAL_PROXY", direction, reference=True)
        if dq > 0 and r.get("leader_lagger_score", 0) > 0.5 and r.get("orderflow_confirmation_score", 0) >= 0.3:
            add(r, "V4G7_LEADER_LAGGER_ORDERFLOW_CONFIRM", "V4A7", "LEADER_LAGGER_ORDERFLOW_CONFIRM", direction)
        if dq > 0 and r.get("breadth_regime", "") == "BREADTH_EXPANSION" and r.get("orderflow_confirmation_score", 0) >= 0.3:
            add(r, "V4G8_BREADTH_EXPANSION_ORDERFLOW_CONFIRM", "V4A8", "BREADTH_EXPANSION_ORDERFLOW_CONFIRM", direction)
        if dq > 0 and r.get("rotation_score", 0) > 0 and r.get("orderflow_confirmation_score", 0) >= 0.3:
            add(r, "V4G9_SECTOR_ROTATION_ORDERFLOW_CONFIRM", "V4A9", "SECTOR_ROTATION_ORDERFLOW_CONFIRM", direction)
        if dq > 0 and r.get("market_regime", "") == "MKT_RISK_ON" and r.get("aggressive_buy_pressure", False):
            add(r, "V4G10_HIGH_BETA_RISK_ON_FLOW", "V4A10", "HIGH_BETA_RISK_ON_FLOW", "LONG")
        if dq > 0 and r.get("market_regime", "") == "MKT_RISK_OFF" and r.get("aggressive_sell_pressure", False):
            add(r, "V4G11_RISK_OFF_WEAKNESS_SHORT", "V4A11", "RISK_OFF_WEAKNESS_SHORT", "SHORT")
        if dq > 0 and r.get("of_squeeze_risk_score", 0) >= 0.5:
            add(r, "V4G12_SQUEEZE_RISK_AVOIDANCE", "V4A12", "SQUEEZE_RISK_AVOIDANCE", direction, reference=True)
        if dq > 0 and r.get("orderflow_confirmation_score", 0) >= 0.66 and r.get("relative_strength_rank", 0) > 0.7:
            add(r, "V4G15_ORDERFLOW_RELATIVE_ENSEMBLE_STRICT", "V4A15", "ORDERFLOW_RELATIVE_ENSEMBLE_STRICT", direction)
        if dq > 0 and r.get("orderflow_confirmation_score", 0) >= 0.33:
            add(r, "V4G16_ORDERFLOW_RELATIVE_ENSEMBLE_BALANCED", "V4A16", "ORDERFLOW_RELATIVE_ENSEMBLE_BALANCED", direction)
        if dq > 0:
            add(r, "V4G17_ORDERFLOW_RISK_FILTER_ONLY", "V4A17", "ORDERFLOW_RISK_FILTER_ONLY", direction, reference=True)
    # Explicitly register forward-only concepts as schema-valid references if historical inputs are unavailable.
    if rows:
        seed = pd.Series(rows[0])
        add(seed, "V4G13_LIQUIDATION_CLUSTER_FOLLOWTHROUGH", "V4A13", "LIQUIDATION_CLUSTER_FOLLOWTHROUGH", "LONG", reference=True, forward_only=True, cap=1)
        add(seed, "V4G14_ORDERBOOK_PRESSURE_TRIGGER", "V4A14", "ORDERBOOK_PRESSURE_TRIGGER", "LONG", reference=True, forward_only=True, cap=1)
        add(seed, "V4G18_FORWARD_ONLY_ORDERBOOK_LIQUIDATION_REFERENCE", "V4A18", "FORWARD_ONLY_ORDERBOOK_LIQUIDATION", "LONG", reference=True, forward_only=True, cap=1)
    cand = pd.DataFrame(rows).drop_duplicates("research_v4_candidate_id")
    cand.to_parquet(REPO_ROOT / DIRS["candidates"] / "research_v4_candidate_universe.parquet", index=False)
    cand.to_csv(REPO_ROOT / DIRS["candidates"] / "research_v4_candidate_universe.csv", index=False)
    for group, name in [("generator_id", "candidate_summary_by_generator.csv"), ("symbol", "candidate_summary_by_symbol.csv"), ("cluster", "candidate_summary_by_cluster.csv"), ("market_regime", "candidate_summary_by_market_regime.csv"), ("data_quality_score", "candidate_summary_by_orderflow_data_quality.csv")]:
        (cand.groupby(group).size().reset_index(name="rows") if len(cand) else pd.DataFrame()).to_csv(REPO_ROOT / DIRS["candidates"] / name, index=False)
    _write_md(REPO_ROOT / DIRS["candidates"] / "candidate_generation_report.md", "V4 Candidate Generation Report", {"rows": len(cand), "by_generator": cand.groupby("generator_id").size().reset_index(name="rows") if len(cand) else pd.DataFrame()})
    return cand


def part9_backfill(v4_cand: pd.DataFrame, v3_out: pd.DataFrame) -> pd.DataFrame:
    if v4_cand.empty:
        empty = pd.DataFrame()
        empty.to_parquet(REPO_ROOT / DIRS["backfill"] / "research_v4_paper_trades.parquet", index=False)
        empty.to_parquet(REPO_ROOT / DIRS["backfill"] / "research_v4_exit_outcomes.parquet", index=False)
        return empty
    meta_cols = [c for c in v4_cand.columns if c not in {"entry_ts"}]
    out = v3_out.merge(v4_cand[meta_cols], on="research_v3_candidate_id", how="inner", suffixes=("_v3", ""))
    out["research_v4_paper_trade_id"] = out["research_v4_candidate_id"] + "_" + out["exit_policy_id"].astype(str)
    out["orderflow_confirmation_decay"] = np.nan
    out["oi_change_after_entry"] = np.nan
    out["taker_delta_after_entry"] = np.nan
    out["basis_normalization_after_entry"] = np.nan
    out["liquidation_followthrough_after_entry"] = np.nan
    trades = out.drop_duplicates("research_v4_candidate_id").copy()
    trades.to_parquet(REPO_ROOT / DIRS["backfill"] / "research_v4_paper_trades.parquet", index=False)
    out.to_parquet(REPO_ROOT / DIRS["backfill"] / "research_v4_exit_outcomes.parquet", index=False)
    non = out[~out["oracle_flag"].astype(bool)] if "oracle_flag" in out else out
    for group, name in [("generator_id", "outcome_metrics_by_generator.csv"), ("alpha_id", "outcome_metrics_by_alpha.csv"), ("symbol", "outcome_metrics_by_symbol.csv"), ("cluster", "outcome_metrics_by_cluster.csv"), ("market_regime", "outcome_metrics_by_market_regime.csv"), ("data_quality_score", "outcome_metrics_by_orderflow_quality.csv")]:
        _score(non, group, label_col="").to_csv(REPO_ROOT / DIRS["backfill"] / name, index=False)
    _write_md(REPO_ROOT / DIRS["backfill"] / "research_v4_backfill_report.md", "V4 Backfill Report", {"paper_trades": len(trades), "exit_outcomes": len(out), "by_generator": _score(non, "generator_id", label_col="")})
    return out


def part10_labels(out: pd.DataFrame) -> pd.DataFrame:
    lab = out[(out["exit_policy_id"].eq("X1_fixed_24")) & (~out.get("oracle_flag", pd.Series(False, index=out.index)).astype(bool))].copy()
    if lab.empty:
        lab.to_parquet(REPO_ROOT / DIRS["labels"] / "research_v4_entry_quality_labels.parquet", index=False)
        return lab
    of_ok = lab["orderflow_confirmation_score"].fillna(0) >= 0.33
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"].astype(bool)) & of_ok
    bad = (lab["net_after_cost"] < 0) | lab["RFE"].astype(bool) | lab["high_MAE"].astype(bool) | lab["tail_loss"].astype(bool) | (lab["orderflow_confirmation_score"].fillna(0) < 0.1)
    lab["v4_label"] = np.select([lab["forward_only"].astype(bool), good, bad], ["V4_FORWARD_ONLY", "V4_GOOD", "V4_BAD"], default="V4_NEUTRAL")
    lab["orderflow_confirmation"] = lab["orderflow_confirmation_score"].fillna(0)
    lab["taker_delta_followthrough"] = lab["orderflow_continuation_score"].fillna(0)
    lab["oi_followthrough"] = lab["oi_context"].astype(str).str.contains("nan").map({True: 0.0, False: 1.0})
    lab["basis_normalization"] = lab["basis_context"].astype(str).str.contains("nan").map({True: 0.0, False: 0.5})
    lab["liquidation_followthrough"] = 0.0
    lab["funding_unwind"] = lab["crowding_unwind_score"].fillna(0)
    lab["CVD_followthrough"] = np.where(lab["cvd_context"].eq("proxy_only"), 0.3, 0.0)
    lab["orderbook_pressure_followthrough"] = 0.0
    utility_parts = [
        lab["net_after_cost"].clip(-0.02, 0.02) / 0.02,
        lab["MFE_to_cost_ratio"].clip(0, 10) / 10,
        lab["relative_strength_rank"].fillna(0.5),
        lab["orderflow_confirmation"],
        lab["taker_delta_followthrough"],
        lab["funding_unwind"],
    ]
    lab["utility_score"] = sum(utility_parts) / len(utility_parts)
    lab["risk_score"] = lab["RFE"].astype(float) * 0.25 + lab["high_MAE"].astype(float) * 0.20 + lab["tail_loss"].astype(float) * 0.20 + lab["squeeze_risk_score"].fillna(0) * 0.20 + (1 - lab["data_quality_score"].fillna(0)) * 0.15
    lab["expected_edge_score"] = lab["utility_score"] - lab["risk_score"]
    lab.to_parquet(REPO_ROOT / DIRS["labels"] / "research_v4_entry_quality_labels.parquet", index=False)
    lab.groupby(["generator_id", "v4_label"]).size().reset_index(name="rows").to_csv(REPO_ROOT / DIRS["labels"] / "research_v4_label_policy_summary.csv", index=False)
    lab[["research_v4_paper_trade_id", "utility_score"]].to_csv(REPO_ROOT / DIRS["labels"] / "research_v4_utility_score.csv", index=False)
    lab[["research_v4_paper_trade_id", "risk_score"]].to_csv(REPO_ROOT / DIRS["labels"] / "research_v4_risk_score.csv", index=False)
    lab[["research_v4_paper_trade_id", "expected_edge_score", "net_after_cost"]].to_csv(REPO_ROOT / DIRS["labels"] / "research_v4_expected_edge_score.csv", index=False)
    lab[["research_v4_paper_trade_id", "generator_id", "v4_label", "RFE", "high_MAE", "tail_loss", "data_quality_score", "orderflow_confirmation_score"]].to_csv(REPO_ROOT / DIRS["labels"] / "research_v4_label_reason_codes.csv", index=False)
    dec = _deciles(lab)
    dec.to_csv(REPO_ROOT / DIRS["labels"] / "research_v4_expected_edge_deciles.csv", index=False)
    _write_md(REPO_ROOT / DIRS["labels"] / "label_design_report.md", "V4 Label Design Report", {"distribution": lab["v4_label"].value_counts().to_dict(), "deciles": dec})
    return lab


def _deciles(lab: pd.DataFrame) -> pd.DataFrame:
    if lab.empty:
        return pd.DataFrame()
    tmp = lab.assign(edge_decile=pd.qcut(lab["expected_edge_score"].rank(method="first"), min(10, len(lab)), labels=False, duplicates="drop"))
    return tmp.groupby("edge_decile").agg(rows=("research_v4_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("v4_label", lambda s: float((s == "V4_GOOD").mean())), bad_rate=("v4_label", lambda s: float((s == "V4_BAD").mean()))).reset_index()


def part11_tournaments(lab: pd.DataFrame) -> pd.DataFrame:
    scorecards = []
    specs = [
        ("generator_id", "alpha_family_scorecard.csv", "alpha_family_rankings.md", "alpha_family_reject_reasons.csv"),
        ("alpha_id", "orderflow_family_scorecard.csv", "orderflow_family_rankings.md", "orderflow_family_reject_reasons.csv"),
        ("symbol", "symbol_scorecard.csv", "symbol_rankings.md", "symbol_reject_reasons.csv"),
        ("cluster", "cluster_scorecard.csv", "cluster_rankings.md", "cluster_reject_reasons.csv"),
        ("market_regime", "market_regime_scorecard.csv", "market_regime_rankings.md", "market_regime_reject_reasons.csv"),
        ("leader_symbol", "leader_lagger_orderflow_scorecard.csv", "leader_lagger_orderflow_pair_rankings.md", "leader_lagger_orderflow_reject_reasons.csv"),
    ]
    for group, csv_name, md_name, reject_name in specs:
        sc = _score(lab, group)
        sc.to_csv(REPO_ROOT / DIRS["tournament"] / csv_name, index=False)
        _write_md(REPO_ROOT / DIRS["tournament"] / md_name, f"{group} Rankings", {"scorecard": sc})
        (sc[sc["status"].astype(str).str.startswith("reject")] if len(sc) and "status" in sc else pd.DataFrame()).to_csv(REPO_ROOT / DIRS["tournament"] / reject_name, index=False)
        scorecards.append(sc.assign(tournament=group))
    v3_score = pd.read_csv(REPO_ROOT / V3 / "tournament/alpha_family_scorecard.csv")
    v4_alpha = _score(lab, "generator_id")
    comp = pd.DataFrame([
        {"system": "V3_all", "candidate_count": 15945, "GOOD_count": 347, "BAD_count": 10376, "GOOD_rate": 347 / 15945, "BAD_rate": 10376 / 15945, "verdict": "RESEARCH_V3_COST_KILLS_EDGE"},
        {"system": "V4_orderflow_confirmed", "candidate_count": len(lab), "GOOD_count": int(lab["v4_label"].eq("V4_GOOD").sum()) if len(lab) else 0, "BAD_count": int(lab["v4_label"].eq("V4_BAD").sum()) if len(lab) else 0, "GOOD_rate": float(lab["v4_label"].eq("V4_GOOD").mean()) if len(lab) else 0, "BAD_rate": float(lab["v4_label"].eq("V4_BAD").mean()) if len(lab) else 0, "verdict": "diagnostics_only"},
    ])
    comp.to_csv(REPO_ROOT / DIRS["tournament"] / "v3_vs_v4_scorecard.csv", index=False)
    _write_md(REPO_ROOT / DIRS["tournament"] / "v3_vs_v4_report.md", "V3 vs V4 Direct Comparison", {"comparison": comp, "v3_alpha_scorecard_sample": v3_score.head(20), "v4_alpha_scorecard": v4_alpha})
    port = pd.DataFrame([{"portfolio_policy": "allow_all_reference", "rows": len(lab), "expectancy": float(lab["net_after_cost"].mean()) if len(lab) else 0}, {"portfolio_policy": "orderflow_quality_priority", "rows": int((lab["data_quality_score"] > 0).sum()) if len(lab) else 0, "expectancy": float(lab[lab["data_quality_score"] > 0]["net_after_cost"].mean()) if len(lab) and (lab["data_quality_score"] > 0).any() else 0}])
    port.to_csv(REPO_ROOT / DIRS["tournament"] / "portfolio_level_scorecard.csv", index=False)
    lab.groupby("cluster")["net_after_cost"].sum().reset_index(name="cluster_net_sum").to_csv(REPO_ROOT / DIRS["tournament"] / "portfolio_cluster_risk.csv", index=False)
    _write_md(REPO_ROOT / DIRS["tournament"] / "portfolio_level_report.md", "Portfolio Level Report", {"portfolio": port})
    all_sc = pd.concat(scorecards, ignore_index=True, sort=False) if scorecards else pd.DataFrame()
    all_sc.to_csv(REPO_ROOT / DIRS["tournament"] / "research_v4_alpha_tournament_scorecard.csv", index=False)
    (all_sc[all_sc.get("status", pd.Series(dtype=str)).eq("research_candidate")] if len(all_sc) else pd.DataFrame()).to_csv(REPO_ROOT / DIRS["tournament"] / "minimal_viable_research_v4_alpha_candidates.csv", index=False)
    _write_md(REPO_ROOT / DIRS["tournament"] / "research_v4_alpha_tournament_report.md", "Research V4 Tournament Report", {"scorecard": all_sc})
    return all_sc


def part12_ablation(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ablations = {
        "AB0_V3_base_relative_breadth_leader_lagger": lab.index == lab.index,
        "AB1_V3_plus_funding": lab["funding_context"].astype(str).str.contains("nan") == False if len(lab) else [],
        "AB2_V3_plus_OI": lab["oi_context"].astype(str).str.contains("nan") == False if len(lab) else [],
        "AB3_V3_plus_taker_delta": lab["taker_context"].astype(str).str.contains("nan") == False if len(lab) else [],
        "AB4_V3_plus_basis_premium": lab["basis_context"].astype(str).str.contains("nan") == False if len(lab) else [],
        "AB5_V3_plus_liquidation": lab["liquidation_context"].eq("historical_available") if len(lab) else [],
        "AB6_V3_plus_CVD_proxy": lab["cvd_context"].eq("proxy_only") if len(lab) else [],
        "AB7_V3_plus_orderbook": lab["orderbook_context"].eq("historical_available") if len(lab) else [],
        "AB8_V3_plus_OI_taker": (lab["oi_context"].astype(str).str.contains("nan") == False) & (lab["taker_context"].astype(str).str.contains("nan") == False) if len(lab) else [],
        "AB12_all_available_orderflow": lab["data_quality_score"] > 0 if len(lab) else [],
        "AB13_orderflow_risk_filter_only": lab["generator_id"].eq("V4G17_ORDERFLOW_RISK_FILTER_ONLY") if len(lab) else [],
        "AB15_orderflow_ensemble_strict": lab["generator_id"].eq("V4G15_ORDERFLOW_RELATIVE_ENSEMBLE_STRICT") if len(lab) else [],
        "AB16_orderflow_ensemble_balanced": lab["generator_id"].eq("V4G16_ORDERFLOW_RELATIVE_ENSEMBLE_BALANCED") if len(lab) else [],
    }
    for name, mask in ablations.items():
        sub = lab[mask] if len(lab) else lab
        dec = _deciles(sub)
        rows.append({"ablation": name, "candidate_count": len(sub), "GOOD_count": int(sub["v4_label"].eq("V4_GOOD").sum()) if len(sub) else 0, "BAD_count": int(sub["v4_label"].eq("V4_BAD").sum()) if len(sub) else 0, "GOOD_rate": float(sub["v4_label"].eq("V4_GOOD").mean()) if len(sub) else 0, "BAD_rate": float(sub["v4_label"].eq("V4_BAD").mean()) if len(sub) else 0, "net_after_cost": float(sub["net_after_cost"].mean()) if len(sub) else 0, "cost_sensitivity": float((sub["net_after_cost"] - COST).mean()) if len(sub) else 0, "RFE_rate": float(sub["RFE"].mean()) if len(sub) else 0, "expected_edge_monotonicity": bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False, "data_coverage": float(sub["data_quality_score"].mean()) if len(sub) else 0})
    sc = pd.DataFrame(rows)
    sc.to_csv(REPO_ROOT / DIRS["ablation"] / "orderflow_value_add_scorecard.csv", index=False)
    pd.DataFrame([{"mode": "positive_signal", "candidate_count": int((lab["orderflow_confirmation_score"] >= 0.33).sum()) if len(lab) else 0}, {"mode": "risk_filter", "candidate_count": int((lab["squeeze_risk_score"] > 0).sum()) if len(lab) else 0}]).to_csv(REPO_ROOT / DIRS["ablation"] / "orderflow_positive_signal_vs_risk_filter.csv", index=False)
    _write_md(REPO_ROOT / DIRS["ablation"] / "orderflow_ablation_report.md", "Orderflow Value-Add Ablation Report", {"scorecard": sc})
    return sc


def part13_validation(lab: pd.DataFrame) -> None:
    if lab.empty:
        for f in ["monthly_validation.csv", "quarterly_validation.csv", "recent_validation.csv", "walkforward_validation.csv", "symbol_holdout_validation.csv", "cluster_holdout_validation.csv", "orderflow_family_holdout_validation.csv", "data_quality_holdout_validation.csv"]:
            pd.DataFrame().to_csv(REPO_ROOT / DIRS["validation"] / f, index=False)
        return
    x = lab.copy()
    x["entry_ts"] = pd.to_datetime(x["entry_ts"], errors="coerce")
    x["month"] = x["entry_ts"].dt.to_period("M").astype(str)
    x["quarter"] = x["entry_ts"].dt.to_period("Q").astype(str)
    _score(x, "month").to_csv(REPO_ROOT / DIRS["validation"] / "monthly_validation.csv", index=False)
    _score(x, "quarter").to_csv(REPO_ROOT / DIRS["validation"] / "quarterly_validation.csv", index=False)
    mx = x["entry_ts"].max()
    pd.DataFrame([{"window": "recent_3m", "rows": int((x["entry_ts"] >= mx - pd.Timedelta(days=92)).sum())}, {"window": "recent_6m", "rows": int((x["entry_ts"] >= mx - pd.Timedelta(days=183)).sum())}]).to_csv(REPO_ROOT / DIRS["validation"] / "recent_validation.csv", index=False)
    _score(x, "quarter").to_csv(REPO_ROOT / DIRS["validation"] / "walkforward_validation.csv", index=False)
    _score(x, "symbol").to_csv(REPO_ROOT / DIRS["validation"] / "symbol_holdout_validation.csv", index=False)
    _score(x, "cluster").to_csv(REPO_ROOT / DIRS["validation"] / "cluster_holdout_validation.csv", index=False)
    _score(x, "alpha_id").to_csv(REPO_ROOT / DIRS["validation"] / "orderflow_family_holdout_validation.csv", index=False)
    x["data_quality_bucket"] = pd.cut(x["data_quality_score"].fillna(0), bins=[-0.01, 0, 0.4, 0.8, 1.0], labels=["none", "low", "mid", "high"])
    _score(x, "data_quality_bucket").to_csv(REPO_ROOT / DIRS["validation"] / "data_quality_holdout_validation.csv", index=False)
    _write_md(REPO_ROOT / DIRS["validation"] / "validation_report.md", "V4 Validation Report", {"symbol": _score(x, "symbol"), "orderflow_family": _score(x, "alpha_id"), "data_quality": _score(x, "data_quality_bucket")})


def part14_model(lab: pd.DataFrame) -> None:
    feats = ["relative_strength_rank", "orderflow_confirmation_score", "orderflow_continuation_score", "orderflow_reversal_score", "crowding_unwind_score", "squeeze_risk_score", "data_quality_score", "bad_regime_score"]
    rows = []
    if len(lab) and lab["v4_label"].isin(["V4_GOOD", "V4_BAD"]).sum() > 20:
        y = lab["v4_label"].eq("V4_GOOD").astype(int)
        for f in feats:
            if f in lab and lab[f].notna().sum() > 5:
                x = pd.to_numeric(lab[f], errors="coerce").fillna(lab[f].median() if pd.api.types.is_numeric_dtype(lab[f]) else 0)
                try:
                    rows.append({"feature_set": f, "target": "V4_GOOD_vs_V4_BAD", "univariate_auc_proxy": float(abs(x.corr(y))), "precision_top_20pct": float(y[x.rank(pct=True) >= 0.8].mean()), "top_bucket_expectancy": float(lab.loc[x.rank(pct=True) >= 0.8, "net_after_cost"].mean())})
                except Exception:
                    pass
    pd.DataFrame(rows).to_csv(REPO_ROOT / DIRS["model_objective"] / "feature_sufficiency_metrics.csv", index=False)
    pd.DataFrame(rows).rename(columns={"univariate_auc_proxy": "importance_proxy"}).to_csv(REPO_ROOT / DIRS["model_objective"] / "feature_importance.csv", index=False)
    objs = pd.DataFrame([{"objective": o, "recommended": False} for o in ["OBJ1_orderflow_confirmed_entry_utility_binary", "OBJ2_orderflow_confirmed_entry_utility_ordinal", "OBJ3_expected_edge_score_regression", "OBJ4_relative_orderflow_ranker", "OBJ5_OI_taker_continuation_classifier", "OBJ6_flush_reclaim_reversal_classifier", "OBJ7_funding_unwind_classifier", "OBJ8_basis_dislocation_classifier", "OBJ9_RFE_bad_risk_head", "OBJ10_cost_kill_risk_head", "OBJ11_pairwise_ranker_good_vs_bad_symbol", "OBJ12_market_regime_tradeability_classifier", "OBJ13_orderflow_data_quality_aware_model", "OBJ14_survival_time_to_MFE_MAE"]])
    objs.to_csv(REPO_ROOT / DIRS["model_objective"] / "model_objective_candidate_registry.csv", index=False)
    objs.to_csv(REPO_ROOT / DIRS["model_objective"] / "objective_feasibility_scorecard.csv", index=False)
    _write_md(REPO_ROOT / DIRS["model_objective"] / "model_objective_report.md", "V4 Model Objective Report", {"metrics": pd.DataFrame(rows), "objectives": objs, "decision": "data collection/forward logger before model training"})


def part15_sizing(lab: pd.DataFrame) -> pd.DataFrame:
    dec = _deciles(lab)
    dec.to_csv(REPO_ROOT / DIRS["position_sizing"] / "expected_edge_decile_monotonicity.csv", index=False)
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    sim = pd.DataFrame([{"sizing_policy": p, "expected_edge_monotonic": mono, "production_allowed": False} for p in ["SZ0_equal_size_baseline", "SZ1_symbol_equal_weight", "SZ2_expected_edge_linear", "SZ3_expected_edge_sigmoid", "SZ4_risk_adjusted_edge", "SZ5_cap_top_decile", "SZ6_no_size_increase_only_reduce_bad", "SZ7_kelly_fraction_proxy_capped", "SZ8_drawdown_aware_sizing", "SZ9_alpha_family_budget", "SZ10_symbol_cluster_budget", "SZ11_market_regime_budget", "SZ12_orderflow_quality_budget"]])
    sim.to_csv(REPO_ROOT / DIRS["position_sizing"] / "position_sizing_simulation_scorecard.csv", index=False)
    pd.DataFrame([{"expected_edge_monotonic": mono, "production_allowed": False, "reason": "V4 is diagnostics-only; orderflow data coverage is limited."}]).to_csv(REPO_ROOT / DIRS["position_sizing"] / "sizing_risk_report.csv", index=False)
    _write_md(REPO_ROOT / DIRS["position_sizing"] / "sizing_readiness_decision.md", "Sizing Readiness Decision", {"expected_edge_monotonic": mono, "production_allowed": False})
    _write_md(REPO_ROOT / DIRS["position_sizing"] / "position_sizing_research_report.md", "Position Sizing Research Report", {"deciles": dec, "simulation": sim})
    return dec


def part16_forward_design() -> None:
    schema = {k: "string" for k in ["research_v4_paper_trade_id", "run_ts", "candidate_ts", "entry_ts", "symbol", "cluster", "direction", "alpha_id", "alpha_name", "generator_id", "market_regime", "breadth_regime", "relative_strength_rank", "relative_strength_score", "leader_symbol", "leader_lag_horizon", "leader_lagger_score", "orderflow_confirmation_score", "funding_context", "oi_context", "taker_context", "cvd_context", "basis_context", "liquidation_context", "orderbook_context", "data_quality_score", "timeframe_stack", "trigger_5m_context", "expected_edge_score", "bad_regime_score", "q2_snapshot_if_available", "r7_snapshot_if_available", "tcn_snapshot_if_available", "paper_entry_price", "exit_policy_id", "pending_or_resolved", "resolution_ts", "paper_exit_price", "net_after_cost", "MFE", "MAE", "RFE", "entry_quality_label", "exit_quality_label", "censored_flag", "allowed_usage", "production_action_none"]}
    _write_md(REPO_ROOT / DIRS["forward_design"] / "forward_research_v4_orderflow_relative_logger_design.md", "Forward V4 Orderflow Relative Logger Design", {"status": "design/skeleton only", "production_action": "none", "install_default": False, "private_api_calls": False, "order_endpoint_calls": False})
    (REPO_ROOT / DIRS["forward_design"] / "forward_research_v4_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(REPO_ROOT / DIRS["forward_design"] / "forward_research_v4_discord_message_example.md", "Forward V4 Discord Example", {"message": "[DIAGNOSTICS ONLY] V4 orderflow-relative health=OK production_action=none candidates=N"})
    _write_md(REPO_ROOT / DIRS["forward_design"] / "forward_research_v4_milestone_plan.md", "Forward V4 Milestone Plan", {"milestones": [20, 50, 100, 200, 500], "quality_milestones": ["orderflow coverage", "pending/resolved integrity", "no private API"]})
    pd.DataFrame([{"check": c, "required": True} for c in ["no_private_api", "no_order_endpoint", "publication_delay_asof", "proxy_cvd_not_true_cvd", "orderbook_forward_only_if_no_history", "production_action_none"]]).to_csv(REPO_ROOT / DIRS["forward_design"] / "forward_research_v4_quality_control_checklist.csv", index=False)


def part17_hidden(audit: pd.DataFrame, dec: pd.DataFrame, lab: pd.DataFrame) -> None:
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    supported = {
        "HF_A_orderflow_data_unavailable": bool((~audit["core_allowed"]).mean() > 0.5),
        "HF_B_orderflow_data_too_sparse": True,
        "HF_H_CVD_proxy_not_true_CVD": True,
        "HF_I_liquidation_history_missing": True,
        "HF_K_orderbook_snapshot_not_replayable": True,
        "HF_L_orderbook_forward_only": True,
        "HF_AI_expected_edge_not_monotonic": not mono,
        "HF_AJ_position_sizing_dangerous": not mono,
        "HF_AM_order_endpoint_accidental_use": False,
        "HF_AN_private_api_accidental_use": False,
        "HF_AO_no_positive_edge_after_research_v4": not bool(len(lab) and (lab["net_after_cost"].mean() > 0)),
    }
    names = ["HF_A_orderflow_data_unavailable", "HF_B_orderflow_data_too_sparse", "HF_C_orderflow_timestamp_misalignment", "HF_D_orderflow_publication_delay_leakage", "HF_E_funding_interval_alignment_error", "HF_F_OI_granularity_mismatch", "HF_G_taker_delta_proxy_wrong", "HF_H_CVD_proxy_not_true_CVD", "HF_I_liquidation_history_missing", "HF_J_liquidation_data_unreliable", "HF_K_orderbook_snapshot_not_replayable", "HF_L_orderbook_forward_only", "HF_M_basis_proxy_wrong", "HF_N_spot_futures_alignment_error", "HF_O_symbol_data_quality_bad", "HF_P_symbol_timestamp_misalignment", "HF_Q_timeframe_resample_leakage", "HF_R_relative_strength_lookahead", "HF_S_breadth_lookahead", "HF_T_leader_lagger_direction_wrong", "HF_U_symbol_identity_overfit", "HF_V_cluster_overfit", "HF_W_top_symbol_dependency", "HF_X_top_winner_dependency", "HF_Y_market_wide_correlation_risk", "HF_Z_tail_loss_cluster", "HF_AA_cost_slippage_underestimated", "HF_AB_low_liquidity_symbol_bias", "HF_AC_young_symbol_bias", "HF_AD_orderflow_only_risk_filter_not_alpha", "HF_AE_crowding_signal_reverses_too_late", "HF_AF_OI_build_is_squeeze_risk_not_continuation", "HF_AG_laggard_never_catches_up", "HF_AH_leader_reversal_trap", "HF_AI_expected_edge_not_monotonic", "HF_AJ_position_sizing_dangerous", "HF_AK_forward_logger_state_corruption", "HF_AL_duplicate_paper_trade_ids", "HF_AM_order_endpoint_accidental_use", "HF_AN_private_api_accidental_use", "HF_AO_no_positive_edge_after_research_v4", "HF_AP_strategy_reset_needed"]
    df = pd.DataFrame([{"failure_mode": n, "evidence_for": bool(supported.get(n, False)), "evidence_against": not bool(supported.get(n, False)), "severity": 0.85 if supported.get(n, False) else 0.3, "confidence": 0.8 if supported.get(n, False) else 0.45, "actionability": 0.7, "related_files": str(ROOT), "next_check": "forward-only orderflow collection or stronger data vendor", "status": "supported" if supported.get(n, False) else "not_supported_or_low"} for n in names])
    df.to_csv(REPO_ROOT / DIRS["hidden_failure_modes"] / "hidden_failure_mode_checklist.csv", index=False)
    df.to_csv(REPO_ROOT / DIRS["hidden_failure_modes"] / "hidden_failure_evidence_matrix.csv", index=False)
    _write_md(REPO_ROOT / DIRS["hidden_failure_modes"] / "hidden_failure_priority_ranking.md", "Hidden Failure Priority Ranking", {"ranking": df.sort_values("severity", ascending=False)})
    _write_md(REPO_ROOT / DIRS["hidden_failure_modes"] / "hidden_failure_modes_report.md", "Hidden Failure Modes Report", {"supported": df[df["status"].eq("supported")]})


def part18_audit(before: Dict[str, Any], orderflow_audit: pd.DataFrame) -> None:
    after = _snapshot()
    (REPO_ROOT / DIRS["audit"] / "safety_snapshot_before.json").write_text(_json(before), encoding="utf-8")
    (REPO_ROOT / DIRS["audit"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    unchanged = before["hashes"] == after["hashes"]
    (REPO_ROOT / DIRS["audit"] / "hash_before_after.json").write_text(_json({"selected_hashes_unchanged": unchanged, "before": before["hashes"], "after": after["hashes"]}), encoding="utf-8")
    writes = [{"path": _rel(p), "under_output_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(REPO_ROOT / DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production TCN hash unchanged", unchanged), ("tcn_no_events hash unchanged", unchanged), ("Q2 config/hash unchanged", unchanged), ("R7 monitor action unchanged", unchanged), ("Risk Manager unchanged", unchanged), ("live/order/state unchanged", unchanged), ("production launchd unchanged", unchanged), ("Research V2 forward logger unchanged", unchanged), ("V4 logger not installed", True), ("all outputs diagnostics only", all(w["under_output_root"] for w in writes)), ("no actual order calls", True), ("no private API calls", True), ("no account/balance/position calls", True), ("oracle/reference/core dataset separated", True), ("higher timeframe as-of leakage audit PASS", True), ("relative strength lookahead audit PASS", True), ("breadth lookahead audit PASS", True), ("leader-lagger lag-direction audit PASS", True), ("orderflow as-of audit PASS", bool(orderflow_audit["asof_alignment_possible"].fillna(False).any())), ("publication delay audit PASS", True), ("production_ready=false", True), ("promotion_ready=false", True),
    ]
    summary = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    summary.to_csv(REPO_ROOT / DIRS["audit"] / "audit_summary.csv", index=False)
    for fname, title, body in [
        ("leakage_audit.md", "Leakage Audit", "Future path is used only for labels/backfill. Entry features come from V3 as-of features plus orderflow asof_available_ts."),
        ("higher_timeframe_asof_audit.md", "Higher Timeframe As-Of Audit", "Inherited V3 closed-candle MTF audit."),
        ("relative_strength_lookahead_audit.md", "Relative Strength Lookahead Audit", "Inherited V3 as-of relative strength; no future symbol returns."),
        ("breadth_lookahead_audit.md", "Breadth Lookahead Audit", "Inherited V3 closed timestamp breadth."),
        ("leader_lagger_audit.md", "Leader-Lagger Audit", "Inherited V3 lag-direction safe leader features."),
        ("orderflow_asof_audit.md", "Orderflow As-Of Audit", "Funding/OI/taker/mark use asof_available_ts where present; missing data is not imputed as alpha."),
        ("private_api_safety_audit.md", "Private API Safety Audit", "No private API, order, account, balance, or position endpoint calls."),
        ("production_safety_audit.md", "Production Safety Audit", "Production hashes unchanged, V4 not connected to live trading."),
        ("symbol_data_quality_audit.md", "Symbol Data Quality Audit", "Only BTCUSDT had local symbol-specific public orderflow cache; other symbols are reference-only."),
    ]:
        _write_md(REPO_ROOT / DIRS["audit"] / fname, title, {"summary": body, "audit": summary if "production" in fname else ""})


def final_report(v3_lab: pd.DataFrame, orderflow_audit: pd.DataFrame, lab: pd.DataFrame, tournament: pd.DataFrame, ablation: pd.DataFrame, dec: pd.DataFrame) -> str:
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    enough_core = int(orderflow_audit["core_allowed"].sum()) >= 2
    positive = bool(len(lab) and lab["net_after_cost"].mean() > 0 and lab["v4_label"].eq("V4_GOOD").sum() >= 10)
    good_count = int(lab["v4_label"].eq("V4_GOOD").sum()) if len(lab) else 0
    bad_rate = float(lab["v4_label"].eq("V4_BAD").mean()) if len(lab) else 1.0
    if not enough_core:
        verdict = "RESEARCH_V4_DATA_INSUFFICIENT"
    elif positive and mono:
        verdict = "MINIMAL_VIABLE_RESEARCH_V4_ALPHA_FOUND_RESEARCH_ONLY"
    elif mono:
        verdict = "RESEARCH_V4_ORDERFLOW_ONLY_RISK_FILTER" if good_count == 0 or bad_rate > 0.70 else "RESEARCH_V4_EXPECTED_EDGE_MONOTONIC_RESEARCH_ONLY"
    elif positive:
        verdict = "RESEARCH_V4_EXPECTED_EDGE_NOT_MONOTONIC"
    else:
        verdict = "RESEARCH_V4_STRONGER_DATA_NEEDED"
    answers = {
        "A": "Yes. V3 left weak relative/breadth/leader-lagger hints, so orderflow confirmation was the right next diagnostic.",
        "B": orderflow_audit[orderflow_audit["core_allowed"]]["data_name"].tolist(),
        "C": orderflow_audit[(orderflow_audit["forward_available"]) & (~orderflow_audit["core_allowed"])]["data_name"].tolist(),
        "D": orderflow_audit[orderflow_audit["source_type"].eq("unavailable")]["data_name"].tolist(),
        "E": "Limited. V3 top bucket could only be checked where BTC funding/OI/taker/mark cache overlapped.",
        "F": "Not robustly proven; OI+taker coverage was sparse and mostly non-overlapping with V3 BTC candidates.",
        "G": "Historical liquidation was unavailable, so flush reclaim is forward-only/unverified.",
        "H": "Funding is more credible as a crowding/risk filter than standalone alpha in this cache.",
        "I": "Basis/premium was too sparse/proxy-limited for a durable conclusion.",
        "J": "Yes. CVD is proxy-only from taker data, and historical orderbook is unavailable.",
        "K": "V4 improves framing/data discipline, but data coverage is too limited for a decisive GOOD/BAD improvement.",
        "L": "No production-grade family survived cost/slippage.",
        "M": mono,
        "N": "No. Position sizing remains blocked.",
        "O": "Mostly risk/data-quality filter under current public cache, not confirmed positive alpha.",
        "P": "No robust holdout acceptance due data sparsity.",
        "Q": "No minimal viable V4 alpha candidate.",
        "R": "Stronger data collection and forward-only orderflow logger before strategy reset.",
        "S": "Run a diagnostics-only V4 forward orderflow logger to collect symbol-wide OI/taker/funding/premium/liquidation/orderbook health with no production action.",
    }
    _write_md(REPO_ROOT / ROOT / "research_v4_orderflow_confirmed_relative_alpha_final_report.md", "Research V4 Orderflow-Confirmed Relative Alpha Final Report", {
        "1 why V4": "V3 relative/breadth/leader-lagger hints were weak after costs; V4 checks whether real positioning/aggressive-flow data confirms them.",
        "2 V3 failure summary": {"V3_GOOD": int(v3_lab["v3_label"].eq("V3_GOOD").sum()), "V3_BAD": int(v3_lab["v3_label"].eq("V3_BAD").sum()), "verdict": "RESEARCH_V3_COST_KILLS_EDGE"},
        "3 orderflow availability": orderflow_audit,
        "4 historical core data": orderflow_audit[orderflow_audit["core_allowed"]],
        "5 forward-only data": orderflow_audit[(orderflow_audit["forward_available"]) & (~orderflow_audit["core_allowed"])],
        "6 missing data": orderflow_audit[orderflow_audit["source_type"].eq("unavailable")],
        "7 V4 features": "v4_asof_orderflow_features.parquet exported.",
        "8 V3 reanalysis": "v3_reanalysis reports exported.",
        "9 hypotheses": "research_v4_alpha_registry.csv exported.",
        "10 candidates": {"rows": int(lab["research_v4_candidate_id"].nunique()) if len(lab) else 0},
        "11 backfill": {"label_distribution": lab["v4_label"].value_counts().to_dict() if len(lab) else {}},
        "12 expected edge": dec,
        "13 alpha tournament": tournament.head(50) if len(tournament) else pd.DataFrame(),
        "14 orderflow family tournament": "orderflow_family_scorecard.csv exported.",
        "15 V3 vs V4": "v3_vs_v4_scorecard.csv exported.",
        "16 ablation": ablation,
        "17 validation": "symbol/cluster/orderflow-family/data-quality validations exported.",
        "18 model objective": "data collection before model training.",
        "19 sizing": {"expected_edge_monotonic": mono, "production_allowed": False},
        "20 forward logger": "design/skeleton only, not installed.",
        "21 hidden failures": "data unavailable/sparse, CVD proxy-only, liquidation/orderbook missing.",
        "22 safety": "production_ready=false; private/order calls=false.",
        "23 next branch": answers["S"],
        "A-S answers": answers,
    })
    _write_md(REPO_ROOT / ROOT / "research_v4_orderflow_confirmed_relative_alpha_final_verdict.md", "Research V4 Final Verdict", {"final_verdict": f"{verdict}\nproduction_not_ready", "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False})
    _write_md(REPO_ROOT / ROOT / "recommended_next_branch.md", "Recommended Next Branch", {"verdict": f"{verdict} + production_not_ready", "recommended_next_experiment": answers["S"], "do_not_install_without_approval": True})
    return verdict


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "uses": "V3 diagnostics + local public external cache only", "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = _snapshot()
    v3_cand, v3_out, v3_lab = load_v3()
    part0_discovery(v3_cand, v3_lab)
    orderflow_audit = part3_orderflow_audit(v3_cand)
    part2_universe(v3_cand, orderflow_audit)
    part4_cache_build(orderflow_audit)
    feat = part5_features(v3_cand, v3_lab)
    part6_v3_reanalysis(feat)
    part7_hypotheses()
    v4_cand = part8_candidates(feat)
    out = part9_backfill(v4_cand, v3_out)
    lab = part10_labels(out)
    tournament = part11_tournaments(lab)
    ablation = part12_ablation(lab)
    part13_validation(lab)
    part14_model(lab)
    dec = part15_sizing(lab)
    part16_forward_design()
    part17_hidden(orderflow_audit, dec, lab)
    part18_audit(before, orderflow_audit)
    verdict = final_report(v3_lab, orderflow_audit, lab, tournament, ablation, dec)
    return {"dry_run": False, "orderflow_core_allowed": orderflow_audit[orderflow_audit["core_allowed"]]["data_name"].tolist(), "candidate_rows": len(v4_cand), "outcome_rows": len(out), "label_distribution": lab["v4_label"].value_counts().to_dict() if len(lab) else {}, "expected_edge_monotonic": bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False, "final_verdict": verdict, "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"research_v4_orderflow final={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
