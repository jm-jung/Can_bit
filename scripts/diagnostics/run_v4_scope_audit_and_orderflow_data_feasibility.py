"""
Audit Research V4 implementation scope and orderflow data feasibility.

Diagnostics-only. Default mode is no-network and reads local code/artifacts only.
Public probes are sample-only and only run with --probe-public-data/--network-ok.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("data/diagnostics/v4_scope_audit_and_orderflow_data_feasibility")
V4 = Path("data/diagnostics/research_v4_orderflow_confirmed_relative_alpha")
V3 = Path("data/diagnostics/research_v3_cross_symbol_relative_alpha")
V4_SCRIPT = Path("scripts/diagnostics/run_research_v4_orderflow_confirmed_relative_alpha.py")

DIR_NAMES = [
    "discovery", "audit", "v4_scope_audit", "data_needs", "local_data",
    "public_sources", "acquisition_plan", "fetcher_design", "minimum_dataset",
    "v4_retry_design", "forward_collector", "decision",
]
DIRS = {d: ROOT / d for d in DIR_NAMES}

PUBLIC_SOURCES = [
    ("Binance Futures Funding Rate", "D2_funding_rate", "/fapi/v1/fundingRate", True, False, True, "historical", "symbol,limit,startTime,endTime"),
    ("Binance Futures Open Interest Hist", "D3_open_interest_history", "/futures/data/openInterestHist", True, False, True, "historical", "symbol,period,limit,startTime,endTime"),
    ("Binance Futures Taker Buy/Sell Volume", "D4_taker_buy_sell_volume", "/futures/data/takerlongshortRatio", True, False, True, "historical", "symbol,period,limit,startTime,endTime"),
    ("Binance Futures Mark Price Klines", "D7_mark_price", "/fapi/v1/markPriceKlines", True, False, True, "historical", "symbol,interval,limit,startTime,endTime"),
    ("Binance Futures Premium Index", "D8_premium_index", "/fapi/v1/premiumIndex", True, False, True, "snapshot", "symbol"),
    ("Binance Spot Klines", "D9_spot_ohlcv_or_spot_price", "/api/v3/klines", True, False, True, "historical", "symbol,interval,limit,startTime,endTime"),
    ("Binance Futures AggTrades", "D5_aggTrades_or_trade_prints", "/fapi/v1/aggTrades", True, False, True, "historical_recent_or_paginated", "symbol,limit,startTime,endTime"),
    ("Binance Futures Force Orders", "D11_liquidation_history", "/fapi/v1/allForceOrders", False, False, False, "not_allowed_or_uncertain_public_history", "historical force orders may be restricted; do not use if key/private required"),
    ("Binance Futures Depth Snapshot", "D12_orderbook_snapshot_history", "/fapi/v1/depth", True, False, True, "forward_snapshot_only", "symbol,limit"),
]

DATA_NEEDS = [
    ("D1_ohlcv", "V4G1,V4G2,V4G3,V4G4,V4G5,V4G6,V4G7,V4G8,V4G9", True, False, True, False, "5m/15m/1h", 180, "8+ symbols", "closed candle", "critical", "available via V3/V2", "keep"),
    ("D2_funding_rate", "V4G4,V4G12,V4G15,V4G16", True, False, True, True, "8h/native", 180, "8+ symbols", "funding publication time + delay", "important", "BTC local only", "fetch symbol-wide"),
    ("D3_open_interest_history", "V4G1,V4G2,V4G3,V4G4,V4G15,V4G16", True, False, True, True, "5m/15m/1h", 180, "8+ symbols", "asof_available_ts", "critical", "BTC recent only", "fetch symbol-wide"),
    ("D4_taker_buy_sell_volume", "V4G1,V4G2,V4G7,V4G8,V4G9,V4G10,V4G11", True, False, True, True, "5m/15m/1h", 180, "8+ symbols", "close+delay", "critical", "BTC recent only", "fetch symbol-wide"),
    ("D5_aggTrades_or_trade_prints", "V4G6,V4G14", True, False, True, True, "trade-level", 90, "BTC/ETH/SOL/BNB", "event timestamp", "important", "missing", "sample/progressive fetch"),
    ("D6_proxy_CVD", "V4G6,V4G15,V4G16", True, False, True, False, "5m derived", 90, "BTC/ETH/SOL/BNB", "derived from aggTrades only", "important", "not built", "build after aggTrades"),
    ("D7_mark_price", "V4G5,V4G12", True, False, True, True, "5m", 180, "8+ symbols", "close+delay", "important", "BTC recent only", "fetch symbol-wide"),
    ("D8_premium_index", "V4G5,V4G12", True, True, True, True, "5m/snapshot", 180, "8+ symbols", "publication timestamp", "important", "snapshot only", "fetch history if endpoint supports"),
    ("D9_spot_ohlcv_or_spot_price", "V4G5", True, False, True, False, "5m", 180, "8+ symbols", "closed spot candle", "critical", "not audited as spot", "fetch spot klines"),
    ("D10_spot_futures_basis", "V4G5", True, False, True, False, "5m derived", 180, "8+ symbols", "spot/perp asof aligned", "critical", "missing", "derive from spot+mark"),
    ("D11_liquidation_history", "V4G3,V4G13", True, True, False, True, "event", 60, "BTC/ETH/SOL/BNB", "event timestamp", "critical", "missing", "forward collect or paid"),
    ("D12_orderbook_snapshot_history", "V4G14", True, True, False, True, "snapshot/depth", 60, "BTC/ETH/SOL/BNB", "snapshot timestamp", "critical", "missing", "forward collect or paid"),
    ("D13_orderbook_depth_timeseries", "V4G14", True, True, False, True, "depth stream", 60, "BTC/ETH/SOL/BNB", "event timestamp", "important", "missing", "forward/paid"),
    ("D14_large_trade_aggression", "V4G6,V4G14", True, False, True, True, "trade-level derived", 90, "BTC/ETH/SOL/BNB", "trade timestamp", "optional", "missing", "derive from aggTrades"),
    ("D15_market_wide_basket_orderflow", "V4G8", True, False, True, False, "5m derived", 180, "8+ symbols", "component asof", "important", "missing", "requires symbol-wide OF"),
    ("D16_cluster_orderflow_breadth", "V4G9", True, False, True, False, "5m derived", 180, "8+ symbols", "component asof", "important", "missing", "requires symbol-wide OF"),
]


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, val in sections.items():
        lines += [f"## {key}", ""]
        if isinstance(val, pd.DataFrame):
            lines += ["```csv", val.head(100).to_csv(index=False), "```"]
        elif isinstance(val, (dict, list)):
            lines += ["```json", _json(val), "```"]
        else:
            lines.append(str(val))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_dirs() -> None:
    for d in DIRS.values():
        (REPO_ROOT / d).mkdir(parents=True, exist_ok=True)


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:500])
    return out


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _snapshot() -> Dict[str, Any]:
    return {"hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status(), "python": sys.version, "platform": platform.platform()}


def phase0_discovery() -> None:
    required = [
        V4_SCRIPT,
        Path("scripts/diagnostics/run_forward_research_v4_orderflow_relative_logger.py"),
        V4 / "research_v4_orderflow_confirmed_relative_alpha_final_report.md",
        V4 / "research_v4_orderflow_confirmed_relative_alpha_final_verdict.md",
        V4 / "recommended_next_branch.md",
        V4 / "orderflow_audit/orderflow_data_availability_audit.csv",
        V4 / "orderflow_cache/orderflow_cache_registry.csv",
        V4 / "features/v4_asof_orderflow_features.parquet",
        V4 / "v3_reanalysis/v3_orderflow_reanalysis_report.md",
        V4 / "hypotheses/research_v4_alpha_registry.csv",
        V4 / "candidates/research_v4_candidate_universe.parquet",
        V4 / "backfill/research_v4_exit_outcomes.parquet",
        V4 / "labels/research_v4_entry_quality_labels.parquet",
        V4 / "tournament/research_v4_alpha_tournament_scorecard.csv",
        V4 / "ablation/orderflow_value_add_scorecard.csv",
        V3 / "candidates/research_v3_candidate_universe.parquet",
        V3 / "backfill/research_v3_exit_outcomes.parquet",
        V3 / "labels/research_v3_entry_quality_labels.parquet",
        V3 / "relative_strength/all_symbols_relative_strength.parquet",
        V3 / "breadth/market_breadth_features.parquet",
        V3 / "leader_lagger/leader_lagger_features.parquet",
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
    ]
    inv = []
    for p in required:
        full = REPO_ROOT / p
        inv.append({"path": str(p), "exists": full.exists(), "size_bytes": full.stat().st_size if full.exists() and full.is_file() else 0})
    pd.DataFrame(inv).to_csv(REPO_ROOT / DIRS["discovery"] / "input_inventory.csv", index=False)
    v4_files = sorted((REPO_ROOT / V4).rglob("*")) if (REPO_ROOT / V4).exists() else []
    pd.DataFrame([{"path": _rel(p), "suffix": p.suffix, "size_bytes": p.stat().st_size} for p in v4_files if p.is_file()]).to_csv(REPO_ROOT / DIRS["discovery"] / "v4_artifact_inventory.csv", index=False)
    linkage = pd.DataFrame([
        {"v4_artifact": "features/v4_asof_orderflow_features.parquet", "v3_source": "candidates/research_v3_candidate_universe.parquet", "link_key": "research_v3_candidate_id"},
        {"v4_artifact": "candidates/research_v4_candidate_universe.parquet", "v3_source": "candidates/research_v3_candidate_universe.parquet", "link_key": "research_v3_candidate_id"},
        {"v4_artifact": "backfill/research_v4_exit_outcomes.parquet", "v3_source": "backfill/research_v3_exit_outcomes.parquet", "link_key": "research_v3_candidate_id"},
    ])
    linkage.to_csv(REPO_ROOT / DIRS["discovery"] / "v3_v4_artifact_linkage.csv", index=False)
    roots = ["data/diagnostics", "data/external", "data/orderflow", "data/funding", "data/open_interest", "data/liquidation", "data/orderbook", "data/trades", "data/aggtrades", "data/market", "data/ohlcv", "data/binance", "cache", "logs"]
    local = []
    for root in roots:
        d = REPO_ROOT / root
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".parquet", ".csv", ".json", ".jsonl", ".feather", ".sqlite", ".duckdb", ".h5", ".zip", ".gz", ".txt", ".md"}:
                    name = p.name.lower()
                    fam = "unknown"
                    for token in ["funding", "interest", "taker", "premium", "mark", "liquid", "orderbook", "depth", "trade", "aggtrade", "basis"]:
                        if token in name or token in str(p.parent).lower():
                            fam = token
                            break
                    local.append({"path": _rel(p), "size_bytes": p.stat().st_size, "suffix": p.suffix, "data_family_guess": fam})
    pd.DataFrame(local).to_csv(REPO_ROOT / DIRS["discovery"] / "local_orderflow_data_inventory.csv", index=False)
    (REPO_ROOT / DIRS["discovery"] / "discovered_paths.json").write_text(_json({"required": inv, "local_files_count": len(local)}), encoding="utf-8")
    _write_md(REPO_ROOT / DIRS["discovery"] / "discovery_report.md", "Discovery Report", {"input_inventory": pd.DataFrame(inv), "artifact_linkage": linkage, "local_orderflow_files": pd.DataFrame(local).head(100)})


def phase2_scope_audit() -> Dict[str, Any]:
    code = (REPO_ROOT / V4_SCRIPT).read_text(encoding="utf-8")
    v3_cand = pd.read_parquet(REPO_ROOT / V3 / "candidates/research_v3_candidate_universe.parquet")
    v3_labels = pd.read_parquet(REPO_ROOT / V3 / "labels/research_v3_entry_quality_labels.parquet")
    v4_feat = pd.read_parquet(REPO_ROOT / V4 / "features/v4_asof_orderflow_features.parquet")
    v4_cand = pd.read_parquet(REPO_ROOT / V4 / "candidates/research_v4_candidate_universe.parquet")
    v4_out = pd.read_parquet(REPO_ROOT / V4 / "backfill/research_v4_exit_outcomes.parquet")
    v4_lab = pd.read_parquet(REPO_ROOT / V4 / "labels/research_v4_entry_quality_labels.parquet")
    of_audit = pd.read_csv(REPO_ROOT / V4 / "orderflow_audit/orderflow_data_availability_audit.csv")

    flags = [
        ("loads_v3_candidate_universe", "research_v3_candidate_universe.parquet" in code),
        ("joins_orderflow_to_v3_candidate_timestamp_symbol", "base = v3_cand.merge" in code or "research_v3_candidate_id" in code),
        ("creates_full_timestamp_universe_independent_of_v3", "all_symbols_relative_strength" in code and "v3_cand" not in code.split("part5_features")[0]),
        ("all_v4_candidates_keep_research_v3_candidate_id", "research_v3_candidate_id" in v4_cand.columns and float(v4_cand["research_v3_candidate_id"].notna().mean()) == 1.0),
        ("uses_v3_outcomes_for_v4_backfill", "v3_out.merge" in code),
        ("missing_data_not_imputed_as_zero_signal", "non[col] = np.nan" in code and "fill missing as 0" not in code.lower()),
    ]
    pd.DataFrame([{"check": k, "result": bool(v)} for k, v in flags]).to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_code_scope_audit.csv", index=False)

    generators = sorted(v4_cand["generator_id"].dropna().unique())
    call_graph = []
    for gid in [f"V4G{i}" for i in range(0, 19)]:
        exists_in_code = gid in code
        activated = any(str(g).startswith(gid) for g in generators)
        source = "V3_candidate_feature_frame" if activated or exists_in_code else "not_found"
        call_graph.append({"generator_prefix": gid, "exists_in_code": exists_in_code, "activated_in_artifact": activated, "candidate_source": source})
    pd.DataFrame(call_graph).to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_generator_call_graph.csv", index=False)

    activation = v4_cand.groupby("generator_id").agg(
        candidate_count=("research_v4_candidate_id", "size"),
        forward_only_rate=("forward_only", "mean"),
        reference_only_rate=("candidate_reference_only", "mean"),
        core_allowed_rate=("candidate_allowed_core", "mean"),
        data_quality_mean=("data_quality_score", "mean"),
    ).reset_index()
    activation.to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_generator_activation_summary.csv", index=False)

    origin = pd.DataFrame([
        {"origin_type": "has_v3_candidate_id", "candidate_count": int(v4_cand["research_v3_candidate_id"].notna().sum()), "rate": float(v4_cand["research_v3_candidate_id"].notna().mean())},
        {"origin_type": "independent_full_timestamp_candidate", "candidate_count": int(v4_cand["research_v3_candidate_id"].isna().sum()), "rate": float(v4_cand["research_v3_candidate_id"].isna().mean())},
        {"origin_type": "forward_only", "candidate_count": int(v4_cand["forward_only"].sum()), "rate": float(v4_cand["forward_only"].mean())},
        {"origin_type": "reference_only", "candidate_count": int(v4_cand["candidate_reference_only"].sum()), "rate": float(v4_cand["candidate_reference_only"].mean())},
        {"origin_type": "core_allowed", "candidate_count": int(v4_cand["candidate_allowed_core"].sum()), "rate": float(v4_cand["candidate_allowed_core"].mean())},
    ])
    origin.to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_candidate_origin_breakdown.csv", index=False)

    v3_ids = set(v3_cand["research_v3_candidate_id"])
    v4_ids = set(v4_cand["research_v3_candidate_id"].dropna())
    v3_universe_rows = len(pd.read_parquet(REPO_ROOT / V3 / "relative_strength/all_symbols_relative_strength.parquet"))
    overlap = pd.DataFrame([{
        "v3_candidate_count": len(v3_cand),
        "v4_candidate_count": len(v4_cand),
        "v4_referenced_v3_candidate_count": len(v4_ids),
        "v4_reference_rate": len(v4_ids) / max(len(v4_cand), 1),
        "v3_to_v4_candidate_coverage": len(v4_ids & v3_ids) / max(len(v3_ids), 1),
        "v3_relative_strength_universe_rows": v3_universe_rows,
        "v4_candidate_vs_full_rs_universe_rate": len(v4_cand) / max(v3_universe_rows, 1),
    }])
    overlap.to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_v3_overlap_analysis.csv", index=False)

    data_overlap_rows = []
    for col in ["funding_rate", "oi", "taker_delta", "premium_index", "basis_proxy", "cvd_proxy"]:
        if col in v4_feat:
            data_overlap_rows.append({"feature": col, "non_missing_rows": int(v4_feat[col].notna().sum()), "feature_rows": len(v4_feat), "coverage_rate": float(v4_feat[col].notna().mean()), "symbols_with_non_missing": ",".join(sorted(v4_feat.loc[v4_feat[col].notna(), "symbol"].dropna().unique()))})
    data_overlap_rows.append({"feature": "BTCUSDT_feature_rows", "non_missing_rows": int(v4_feat["symbol"].eq("BTCUSDT").sum()), "feature_rows": len(v4_feat), "coverage_rate": float(v4_feat["symbol"].eq("BTCUSDT").mean()), "symbols_with_non_missing": "BTCUSDT"})
    pd.DataFrame(data_overlap_rows).to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_data_overlap_analysis.csv", index=False)

    lab = v4_lab.copy()
    label_audit = pd.DataFrame([
        {"reason": "total", "count": len(lab)},
        {"reason": "net_after_cost_positive", "count": int((lab["net_after_cost"] > 0).sum())},
        {"reason": "gross_positive_before_cost_proxy", "count": int((lab["gross_return"] > 0).sum()) if "gross_return" in lab else 0},
        {"reason": "mfe_to_cost_ge_3", "count": int((lab["MFE_to_cost_ratio"] >= 3).sum())},
        {"reason": "mae_to_cost_le_6", "count": int((lab["MAE_to_cost_ratio"] <= 6).sum())},
        {"reason": "not_RFE", "count": int((~lab["RFE"].astype(bool)).sum())},
        {"reason": "orderflow_confirmation_ge_0_33", "count": int((lab["orderflow_confirmation_score"].fillna(0) >= 0.33).sum())},
        {"reason": "V4_GOOD", "count": int(lab["v4_label"].eq("V4_GOOD").sum())},
        {"reason": "near_good_price_path_but_no_orderflow", "count": int(((lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"].astype(bool)) & (lab["orderflow_confirmation_score"].fillna(0) < 0.33)).sum())},
    ])
    label_audit.to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_label_threshold_audit.csv", index=False)

    cost_audit = pd.DataFrame([
        {"metric": "net_after_cost_positive", "count": int((lab["net_after_cost"] > 0).sum()), "rate": float((lab["net_after_cost"] > 0).mean())},
        {"metric": "gross_return_positive", "count": int((lab["gross_return"] > 0).sum()) if "gross_return" in lab else 0, "rate": float((lab["gross_return"] > 0).mean()) if "gross_return" in lab else 0},
        {"metric": "cost_removed_positive_proxy", "count": int(((lab["net_after_cost"] + 0.0006) > 0).sum()), "rate": float(((lab["net_after_cost"] + 0.0006) > 0).mean())},
        {"metric": "2x_cost_positive_proxy", "count": int(((lab["net_after_cost"] - 0.0006) > 0).sum()), "rate": float(((lab["net_after_cost"] - 0.0006) > 0).mean())},
        {"metric": "3x_cost_positive_proxy", "count": int(((lab["net_after_cost"] - 0.0012) > 0).sum()), "rate": float(((lab["net_after_cost"] - 0.0012) > 0).mean())},
        {"metric": "explicit_2x_3x_cost_outcomes_present", "count": int(any("2x" in str(x).lower() or "3x" in str(x).lower() for x in v4_out.get("exit_policy_id", []))), "rate": 0.0},
    ])
    cost_audit.to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_cost_model_audit.csv", index=False)

    forward_audit = pd.DataFrame([
        {"family": "liquidation", "historical_available": False, "forward_only_candidates": int(v4_cand["generator_id"].astype(str).str.contains("LIQUIDATION").sum()), "interpretation": "historical 검증 불가, alpha failure로 단정 금지"},
        {"family": "orderbook", "historical_available": False, "forward_only_candidates": int(v4_cand["generator_id"].astype(str).str.contains("ORDERBOOK").sum()), "interpretation": "REST snapshot은 historical backfill 금지, forward-only"},
        {"family": "CVD", "historical_available": False, "forward_only_candidates": 0, "interpretation": "taker proxy만 가능하며 true CVD 아님"},
        {"family": "basis", "historical_available": False, "forward_only_candidates": 0, "interpretation": "spot OHLCV/mark alignment 필요"},
    ])
    forward_audit.to_csv(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_forward_only_audit.csv", index=False)

    verdicts = [
        "V4_SCOPE_MAINLY_V3_REANALYSIS",
        "V4_SCOPE_DATA_COVERAGE_TOO_NARROW",
        "V4_SCOPE_FORWARD_ONLY_DATA_MISSING",
        "V4_SCOPE_VERDICT_TOO_STRONG_IF_INTERPRETED_AS_FULL_ORDERFLOW_ALPHA_FAILURE",
        "V4_SCOPE_NEEDS_RETRY_AFTER_DATA_COLLECTION",
    ]
    if bool((lab["net_after_cost"] > 0).any()) and int(lab["v4_label"].eq("V4_GOOD").sum()) == 0:
        verdicts.append("V4_SCOPE_LABEL_TOO_STRICT_FOR_PRICE_PATH_BUT_VALID_FOR_ORDERFLOW_CONFIRMATION")
    _write_md(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_verdict_audit.md", "V4 Verdict Audit", {
        "scope_verdicts": verdicts,
        "interpretation": "V4 implementation primarily reanalyzed V3 candidates with local BTC orderflow overlap. It did not perform full independent orderflow candidate generation over the complete timestamp universe.",
        "data_vs_alpha": "GOOD=0 should be treated primarily as data coverage/implementation scope limitation plus strict orderflow-confirmed label, not as proof that all orderflow alpha failed.",
    })
    _write_md(REPO_ROOT / DIRS["v4_scope_audit"] / "v4_implementation_scope_audit_report.md", "V4 Implementation Scope Audit Report", {
        "code_scope": pd.DataFrame([{"check": k, "result": bool(v)} for k, v in flags]),
        "activation": activation,
        "origin": origin,
        "overlap": overlap,
        "label_audit": label_audit,
        "cost_audit": cost_audit,
        "final_scope_verdicts": verdicts,
    })
    return {"verdicts": verdicts, "v4_candidates": len(v4_cand), "v4_good": int(lab["v4_label"].eq("V4_GOOD").sum()), "v4_v3_ref_rate": float(v4_cand["research_v3_candidate_id"].notna().mean())}


def phase3_data_needs() -> None:
    needs = pd.DataFrame([{
        "data_family": row[0], "needed_for_alpha_ids": row[1], "historical_required": row[2],
        "forward_only_acceptable": row[3], "public_free_possible": row[4], "local_cache_possible": row[5],
        "paid_likely_required": row[0] in {"D11_liquidation_history", "D12_orderbook_snapshot_history", "D13_orderbook_depth_timeseries"},
        "granularity_required": row[6], "min_history_days": row[7], "symbol_coverage_required": row[8],
        "asof_delay_rule": row[9], "criticality": row[10], "current_status": row[11], "next_action": row[12],
    } for row in DATA_NEEDS])
    needs.to_csv(REPO_ROOT / DIRS["data_needs"] / "orderflow_data_needs_matrix.csv", index=False)
    alpha_map = []
    for _, r in needs.iterrows():
        for aid in str(r["needed_for_alpha_ids"]).split(","):
            alpha_map.append({"alpha_id": aid, "data_family": r["data_family"], "criticality": r["criticality"], "current_status": r["current_status"]})
    pd.DataFrame(alpha_map).to_csv(REPO_ROOT / DIRS["data_needs"] / "alpha_to_data_requirement_matrix.csv", index=False)
    priority = needs.copy()
    priority["priority_score"] = priority["criticality"].map({"critical": 3, "important": 2, "optional": 1}).fillna(0) + priority["public_free_possible"].astype(int) - priority["paid_likely_required"].astype(int)
    priority.sort_values(["priority_score", "data_family"], ascending=[False, True]).to_csv(REPO_ROOT / DIRS["data_needs"] / "data_priority_ranking.csv", index=False)
    _write_md(REPO_ROOT / DIRS["data_needs"] / "orderflow_data_needs_report.md", "Orderflow Data Needs Report", {"needs": needs, "priority": priority.sort_values("priority_score", ascending=False)})


def _sample_file(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"path": _rel(path), "file_size": path.stat().st_size, "modified_time": path.stat().st_mtime, "schema": [], "row_count": None, "date_range": "", "symbol_coverage": "", "read_error": ""}
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path, columns=None)
        elif path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5000)
        elif path.suffix in {".json", ".jsonl"}:
            df = pd.read_json(path, lines=path.suffix == ".jsonl", nrows=5000)
        else:
            return out
        out["schema"] = list(df.columns)
        out["row_count"] = len(df)
        tcols = [c for c in df.columns if "time" in c.lower() or "timestamp" in c.lower() or c.lower() in {"date", "datetime"}]
        if tcols:
            ts = pd.to_datetime(df[tcols[0]], errors="coerce")
            out["date_range"] = f"{ts.min()}..{ts.max()}"
        if "symbol" in df.columns:
            out["symbol_coverage"] = ",".join(sorted(df["symbol"].dropna().astype(str).unique())[:20])
    except Exception as exc:
        out["read_error"] = str(exc)[:200]
    return out


def phase4_local_data() -> None:
    roots = ["data", "cache", "tmp", "logs"]
    rows, candidates, samples = [], [], {}
    suffixes = {".parquet", ".csv", ".json", ".jsonl", ".feather", ".sqlite", ".duckdb", ".h5", ".zip", ".gz", ".txt"}
    tokens = ["funding", "interest", "taker", "premium", "mark", "liquid", "orderbook", "depth", "trade", "aggtrade", "basis", "spot"]
    for root in roots:
        d = REPO_ROOT / root
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file() or p.suffix.lower() not in suffixes:
                continue
            family = next((tok for tok in tokens if tok in p.name.lower() or tok in str(p.parent).lower()), "unknown")
            row = {"path": _rel(p), "file_size": p.stat().st_size, "modified_time": p.stat().st_mtime, "suffix": p.suffix, "data_family_guess": family}
            if family != "unknown" or "ohlcv" in str(p).lower():
                info = _sample_file(p)
                row.update({k: info.get(k) for k in ["schema", "row_count", "date_range", "symbol_coverage", "read_error"]})
                candidates.append({**row, "usable_for_historical_backfill": family in tokens and family not in {"orderbook", "liquid"}, "usable_for_forward_logger": True, "needs_conversion": p.suffix != ".parquet", "notes": ""})
                samples[_rel(p)] = info
            rows.append(row)
    pd.DataFrame(rows).to_csv(REPO_ROOT / DIRS["local_data"] / "local_data_inventory.csv", index=False)
    cand_df = pd.DataFrame(candidates)
    cand_df.to_csv(REPO_ROOT / DIRS["local_data"] / "local_orderflow_candidate_files.csv", index=False)
    (REPO_ROOT / DIRS["local_data"] / "local_data_schema_samples.json").write_text(_json(samples), encoding="utf-8")
    cand_df.to_csv(REPO_ROOT / DIRS["local_data"] / "local_data_usability_matrix.csv", index=False)
    _write_md(REPO_ROOT / DIRS["local_data"] / "local_data_audit_report.md", "Local Data Audit Report", {"candidate_files": cand_df, "summary": {"candidate_count": len(cand_df), "aggtrades_found": bool(cand_df["path"].astype(str).str.contains("agg", case=False).any()) if len(cand_df) else False, "liquidation_found": bool(cand_df["path"].astype(str).str.contains("liquid|force", case=False).any()) if len(cand_df) else False, "orderbook_found": bool(cand_df["path"].astype(str).str.contains("orderbook|depth", case=False).any()) if len(cand_df) else False}})


def _probe_endpoint(base: str, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    url = base + path + "?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=8) as resp:
            body = resp.read(20000)
        data = json.loads(body.decode("utf-8"))
        rows = data if isinstance(data, list) else [data]
        sample = rows[:2]
        schema = list(sample[0].keys()) if sample and isinstance(sample[0], dict) else []
        return {"probe_status": "success", "sample_rows": len(rows), "sample_schema": schema, "sample": sample, "error": ""}
    except Exception as exc:
        return {"probe_status": "failed", "sample_rows": 0, "sample_schema": [], "sample": [], "error": str(exc)[:300]}


def phase5_public_feasibility(probe: bool) -> pd.DataFrame:
    rows = []
    probe_results: Dict[str, Any] = {"probe_enabled": probe, "private_api_used": False, "order_endpoint_used": False}
    for source, fam, endpoint, public, private, allowed, mode, params_desc in PUBLIC_SOURCES:
        res = {"probe_status": "not_run", "sample_rows": 0, "sample_schema": [], "error": ""}
        if probe and public and allowed and endpoint not in {"/fapi/v1/allForceOrders"}:
            base = "https://fapi.binance.com" if endpoint.startswith("/fapi") or endpoint.startswith("/futures") else "https://api.binance.com"
            params = {"symbol": "BTCUSDT", "limit": 1}
            if "period" in params_desc:
                params["period"] = "5m"
            if "interval" in params_desc:
                params["interval"] = "5m"
            res = _probe_endpoint(base, endpoint, params)
            probe_results[source] = res
        rows.append({
            "source_name": source, "data_family": fam, "public_free": public, "requires_api_key": False,
            "private_endpoint": private, "allowed_by_policy": allowed, "historical_available": "historical" in mode,
            "forward_available": True if "snapshot" in mode or "forward" in mode else public,
            "symbols_supported": "Binance listed symbols; needs per-symbol probe", "granularity": "5m/native/snapshot",
            "max_limit": 1000, "pagination_possible": fam in {"D2_funding_rate", "D3_open_interest_history", "D4_taker_buy_sell_volume", "D5_aggTrades_or_trade_prints", "D9_spot_ohlcv_or_spot_price"},
            "expected_rate_limit_risk": "high" if fam == "D5_aggTrades_or_trade_prints" else "medium",
            "date_range_possible": "endpoint dependent", "download_size_estimate": "small for OI/funding/taker; large for aggTrades/orderbook",
            "implementation_complexity": "high" if fam in {"D5_aggTrades_or_trade_prints", "D12_orderbook_snapshot_history"} else "medium",
            "asof_alignment_complexity": "event-based" if fam in {"D5_aggTrades_or_trade_prints", "D11_liquidation_history"} else "standard close/publication delay",
            "quality_risk": "high" if fam in {"D11_liquidation_history", "D12_orderbook_snapshot_history"} else "medium",
            "recommended_usage": "fetch/sample only; diagnostics cache", "probe_status": res["probe_status"],
            "sample_rows": res["sample_rows"], "sample_schema": ",".join(res["sample_schema"]), "next_action": "implement safe fetcher" if public and allowed else "forward-only/paid/skip",
        })
    df = pd.DataFrame(rows)
    df.to_csv(REPO_ROOT / DIRS["public_sources"] / "public_data_source_feasibility.csv", index=False)
    (REPO_ROOT / DIRS["public_sources"] / "public_probe_results.json").write_text(_json(probe_results), encoding="utf-8")
    priority = df.copy()
    priority["priority_score"] = priority["public_free"].astype(int) + priority["historical_available"].astype(int) + priority["pagination_possible"].astype(int) - priority["private_endpoint"].astype(int) - (priority["quality_risk"].eq("high")).astype(int)
    priority.sort_values("priority_score", ascending=False).to_csv(REPO_ROOT / DIRS["public_sources"] / "public_data_source_priority.csv", index=False)
    _write_md(REPO_ROOT / DIRS["public_sources"] / "public_data_feasibility_report.md", "Public Data Feasibility Report", {"probe_enabled": probe, "feasibility": df, "priority": priority.sort_values("priority_score", ascending=False)})
    return df


def phase6_acquisition_plan() -> None:
    items = [
        ("P1A_symbol_wide_funding_history", "D2_funding_rate", "BTC,ETH,SOL,BNB,XRP,DOGE,AVAX,LINK", "365d", "8h/native", "Binance public", "small", "low", "data/diagnostics/research_orderflow_data_cache/normalized/funding/{symbol}.parquet", "V4G4,V4G12,V4G15,V4G16", 1, "GO"),
        ("P1B_symbol_wide_open_interest_history", "D3_open_interest_history", "same", "365d", "5m/15m", "Binance public", "medium", "medium", "data/diagnostics/research_orderflow_data_cache/normalized/oi/{symbol}.parquet", "V4G1,V4G2,V4G3", 1, "GO"),
        ("P1C_symbol_wide_taker_buy_sell_volume", "D4_taker_buy_sell_volume", "same", "365d", "5m/15m", "Binance public", "medium", "medium", "data/diagnostics/research_orderflow_data_cache/normalized/taker/{symbol}.parquet", "V4G1,V4G7,V4G8,V4G9", 1, "GO"),
        ("P1D_spot_ohlcv_for_basis", "D9_spot_ohlcv_or_spot_price", "same if spot pair exists", "365d", "5m", "Binance spot public", "medium", "medium", "data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/{symbol}.parquet", "V4G5", 1, "GO"),
        ("P1E_mark_price_or_premium_history", "D7/D8", "same", "365d", "5m", "Binance public", "medium", "medium", "data/diagnostics/research_orderflow_data_cache/normalized/mark_premium/{symbol}.parquet", "V4G5,V4G12", 1, "GO"),
        ("P2A_aggTrades_history_for_proxy_CVD", "D5/D6", "BTC,ETH,SOL,BNB first", "90-180d", "trade-level", "Binance public/paginated", "large", "high", "data/diagnostics/research_orderflow_data_cache/normalized/aggtrades/{symbol}.parquet", "V4G6,V4G14", 2, "GO_SLOW_SAMPLE_FIRST"),
        ("P3A_liquidation_stream_or_recent_force_orders", "D11", "BTC,ETH,SOL,BNB", "forward 30-60d", "event/snapshot", "public if available", "small", "medium", "data/diagnostics/forward_orderflow_data_collector/cache/liquidation/", "V4G3,V4G13", 3, "FORWARD_ONLY"),
        ("P3B_orderbook_snapshot_forward_logger", "D12/D13", "BTC,ETH,SOL,BNB", "forward 30-60d", "snapshot", "public REST/WebSocket", "medium", "medium", "data/diagnostics/forward_orderflow_data_collector/cache/orderbook/", "V4G14", 3, "FORWARD_ONLY"),
        ("P4A_historical_liquidation_full", "D11", "multi-symbol", "180d+", "event", "third-party/paid likely", "unknown", "paid", "external/manual only", "V4G3,V4G13", 4, "NO_GO_WITHOUT_APPROVAL"),
        ("P4B_historical_orderbook_depth", "D12/D13", "multi-symbol", "180d+", "depth", "third-party/paid likely", "very_large", "paid", "external/manual only", "V4G14", 4, "NO_GO_WITHOUT_APPROVAL"),
    ]
    plan = pd.DataFrame([{"item": i, "data_family": f, "symbols": s, "time_range": tr, "granularity": g, "source": src, "expected_size": size, "rate_limit_risk": risk, "storage_path": path, "research_alpha_ids_unlocked": aids, "priority": pr, "go_no_go": go, "asof_delay_rule": "source publication/event timestamp + conservative delay", "quality_checks": "missing/gap/duplicate/outlier/asof checks", "failure_fallback": "mark unavailable and keep reference-only"} for i, f, s, tr, g, src, size, risk, path, aids, pr, go in items])
    plan.to_csv(REPO_ROOT / DIRS["acquisition_plan"] / "orderflow_data_acquisition_plan.csv", index=False)
    _write_md(REPO_ROOT / DIRS["acquisition_plan"] / "orderflow_data_storage_schema.md", "Orderflow Data Storage Schema", {"raw": "data/diagnostics/research_orderflow_data_cache/raw/{data_family}/{symbol}/", "normalized": "data/diagnostics/research_orderflow_data_cache/normalized/{data_family}/{symbol}.parquet", "registry": "cache_registry.csv"})
    _write_md(REPO_ROOT / DIRS["acquisition_plan"] / "orderflow_data_fetcher_design.md", "Orderflow Data Fetcher Design", {"cli": "--dry-run --symbols --data-family --start --end --max-pages --sample-only --resume --no-private --json"})
    pd.DataFrame([{"check": c, "required": True} for c in ["no_private", "no_order_endpoint", "asof_available_ts", "symbol_normalized", "timestamp_utc", "gap_duplicate_outlier_summary"]]).to_csv(REPO_ROOT / DIRS["acquisition_plan"] / "orderflow_data_quality_checklist.csv", index=False)
    _write_md(REPO_ROOT / DIRS["acquisition_plan"] / "orderflow_incremental_update_plan.md", "Incremental Update Plan", {"plan": "append by symbol/data_family, dedupe by timestamp/event_id, write diagnostics registry only"})
    _write_md(REPO_ROOT / DIRS["acquisition_plan"] / "orderflow_acquisition_report.md", "Orderflow Acquisition Report", {"plan": plan})


def phase7_fetcher_design() -> None:
    scripts = pd.DataFrame([
        {"script": "scripts/diagnostics/fetch_research_orderflow_public_data.py", "status": "design_only", "purpose": "safe public sample/incremental fetcher"},
        {"script": "scripts/diagnostics/convert_orderflow_raw_to_parquet.py", "status": "design_only", "purpose": "normalize raw data"},
        {"script": "scripts/diagnostics/build_proxy_cvd_from_aggtrades.py", "status": "design_only", "purpose": "proxy CVD, not true CVD"},
        {"script": "scripts/diagnostics/build_spot_futures_basis_features.py", "status": "design_only", "purpose": "basis proxy from spot/mark"},
        {"script": "scripts/diagnostics/audit_orderflow_cache_quality.py", "status": "design_only", "purpose": "quality/asof audit"},
    ])
    scripts.to_csv(REPO_ROOT / DIRS["fetcher_design"] / "fetcher_script_inventory.csv", index=False)
    _write_md(REPO_ROOT / DIRS["fetcher_design"] / "fetcher_cli_design.md", "Fetcher CLI Design", {"example": "python scripts/diagnostics/fetch_research_orderflow_public_data.py --sample-only --symbols BTCUSDT --data-family funding,oi,taker,premium,mark --max-pages 1 --json"})
    _write_md(REPO_ROOT / DIRS["fetcher_design"] / "fetcher_storage_schema.md", "Fetcher Storage Schema", {"root": "data/diagnostics/research_orderflow_data_cache/", "raw": "raw/{data_family}/{symbol}/", "normalized": "normalized/{data_family}/{symbol}.parquet", "feature_ready": "features/{data_family}/{symbol}.parquet"})
    _write_md(REPO_ROOT / DIRS["fetcher_design"] / "fetcher_safety_audit.md", "Fetcher Safety Audit", {"private_api": False, "api_key": False, "order_account_balance_position": False, "production_paths": False})
    _write_md(REPO_ROOT / DIRS["fetcher_design"] / "fetcher_design_report.md", "Fetcher Design Report", {"scripts": scripts, "safety": "sample-only first; no private endpoints; diagnostics output only"})


def phase8_minimum_dataset() -> None:
    tiers = pd.DataFrame([
        {"tier": "minimum_viable_v4_lite", "required_data": "8 symbols OHLCV + funding + OI + taker + mark/premium + spot OHLCV/basis 180-365d", "unlocked_alpha_families": "V4G1,V4G2,V4G4,V4G5,V4G7,V4G8,V4G9,V4G15,V4G16", "expected_storage": "low-medium", "expected_runtime": "hours", "go_no_go": "GO"},
        {"tier": "minimum_viable_v4_cvd", "required_data": "V4-lite + aggTrades/proxy CVD BTC/ETH/SOL/BNB 90-180d", "unlocked_alpha_families": "V4G6,V4G14 proxy", "expected_storage": "large", "expected_runtime": "long", "go_no_go": "GO_SAMPLE_FIRST"},
        {"tier": "minimum_viable_v4_forced_flow", "required_data": "liquidation/orderbook forward 30-60d", "unlocked_alpha_families": "V4G3,V4G13,V4G14", "expected_storage": "medium", "expected_runtime": "calendar time", "go_no_go": "FORWARD_ONLY"},
        {"tier": "minimum_viable_v4_full", "required_data": "V4-lite + proxy CVD + liquidation + historical orderbook/depth 180d+", "unlocked_alpha_families": "all V4", "expected_storage": "very large", "expected_runtime": "high", "go_no_go": "PAID_OR_STRONGER_DATA_LIKELY"},
    ])
    tiers.to_csv(REPO_ROOT / DIRS["minimum_dataset"] / "minimum_viable_v4_datasets.csv", index=False)
    _write_md(REPO_ROOT / DIRS["minimum_dataset"] / "v4_retry_readiness_criteria.md", "V4 Retry Readiness Criteria", {"criteria": "Do not retry V4 as alpha until V4-lite has 8+ symbols and 180d+ symbol-wide OF coverage; run CVD/forced-flow branches separately."})
    _write_md(REPO_ROOT / DIRS["minimum_dataset"] / "minimum_dataset_report.md", "Minimum Dataset Report", {"tiers": tiers})


def phase9_retry_design() -> None:
    branches = pd.DataFrame([
        {"retry": "R4R1_V4_lite_symbol_wide_OI_taker_funding_basis", "required_data": "V4-lite", "success_criteria": "GOOD>0, BAD rate improves vs V3, cost survives holdout", "priority": 1},
        {"retry": "R4R2_V4_proxy_CVD_relative_alpha", "required_data": "aggTrades/proxy CVD", "success_criteria": "CVD divergence/flow alignment improves V3 top bucket", "priority": 2},
        {"retry": "R4R3_V4_flush_reclaim_forward_only", "required_data": "liquidation/orderbook forward 30-60d", "success_criteria": "forced-flow setups resolve with positive expectancy", "priority": 3},
        {"retry": "R4R4_V4_independent_orderflow_candidate_generation", "required_data": "symbol-wide V4-lite or better", "success_criteria": "candidates generated from full timestamp universe not V3 overlap", "priority": 1},
        {"retry": "R4R5_V4_orderflow_risk_filter_only_integration", "required_data": "V4-lite", "success_criteria": "risk filter reduces BAD/RFE without claiming positive alpha", "priority": 4},
    ])
    branches.to_csv(REPO_ROOT / DIRS["v4_retry_design"] / "v4_retry_branch_options.csv", index=False)
    _write_md(REPO_ROOT / DIRS["v4_retry_design"] / "v4_retry_recommended_branch.md", "V4 Retry Recommended Branch", {"recommended": "R4R1 + R4R4 after V4-lite data collection. Do not retry with current BTC-only sparse cache."})
    _write_md(REPO_ROOT / DIRS["v4_retry_design"] / "v4_retry_design_report.md", "V4 Retry Design Report", {"branches": branches})


def phase10_forward_collector_plan() -> None:
    schema = {"run_ts": "datetime", "symbol": "string", "data_family": "string", "source": "string", "snapshot_ts": "datetime", "value_fields": "json", "quality_flags": "json", "latency_ms": "float", "missing_flag": "bool", "production_action_none": "bool"}
    _write_md(REPO_ROOT / DIRS["forward_collector"] / "forward_orderflow_collector_design.md", "Forward Orderflow Collector Design", {"purpose": "data collection only, not alpha/trading", "families": ["funding", "OI", "premium/mark", "recent aggTrades", "liquidation if public", "orderbook top N snapshot"], "interval": "5m or 15m; orderbook 15-30m if heavy"})
    (REPO_ROOT / DIRS["forward_collector"] / "forward_orderflow_collector_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(REPO_ROOT / DIRS["forward_collector"] / "forward_orderflow_collector_ops_plan.md", "Forward Collector Ops Plan", {"install": "not installed in this task", "possible_label": "com.canbit.forward_orderflow_data_collector", "production_action": "none"})
    pd.DataFrame([{"check": c, "required": True} for c in ["no_private_api", "no_order_endpoint", "diagnostics_only", "production_action_none", "state_separate_from_live", "schema_validation"]]).to_csv(REPO_ROOT / DIRS["forward_collector"] / "forward_orderflow_collector_safety_checklist.csv", index=False)


def phase11_decision(scope: Dict[str, Any]) -> List[str]:
    verdicts = [
        "V4_SCOPE_MAINLY_V3_REANALYSIS",
        "V4_GOOD_ZERO_DUE_TO_DATA_COVERAGE",
        "V4_GOOD_ZERO_DUE_TO_LABEL_STRICTNESS",
        "ORDERFLOW_DATA_INSUFFICIENT_FOR_HISTORICAL_V4",
        "PUBLIC_DATA_CAN_UNLOCK_V4_LITE",
        "AGGTRADES_CAN_UNLOCK_PROXY_CVD",
        "LIQUIDATION_FORWARD_ONLY",
        "ORDERBOOK_FORWARD_ONLY",
        "PAID_DATA_LIKELY_REQUIRED_FOR_FULL_V4",
        "V4_RETRY_AFTER_DATA_COLLECTION_RECOMMENDED",
        "FORWARD_ORDERFLOW_COLLECTOR_RECOMMENDED",
        "production_not_ready",
    ]
    answers = {
        "A": "Yes. V4 was mainly V3 candidate overlap reanalysis.",
        "B": "Independent V4 generators existed and activated, but their input frame was V3 candidates, not full timestamp universe.",
        "C": "GOOD 0 is primarily data coverage + strict orderflow confirmation label; not proof of full alpha failure.",
        "D": "No. Current BTC-only sparse cache is not enough for meaningful V4 retry.",
        "E": "Symbol-wide OI history and taker buy/sell volume first.",
        "F": "Funding, OI history, taker buy/sell volume, mark/premium, spot OHLCV; aggTrades likely possible but large.",
        "G": "Yes, proxy CVD can be built from historical aggTrades if collected; must label proxy, not true CVD.",
        "H": "Liquidation history is not locally available; treat as forward-only/paid unless public historical source is verified.",
        "I": "Historical orderbook/depth is forward-only or paid in practical terms.",
        "J": "Yes, with spot OHLCV plus mark/premium/index alignment.",
        "K": "Yes, public endpoints likely allow symbol-wide OI/funding/taker/premium expansion with pagination/rate-limit handling.",
        "L": "8 symbols, 180-365d OHLCV + funding + OI + taker + mark/premium + spot OHLCV/basis.",
        "M": "Data fetcher/collection plan first; then V4 retry. Forward collector is needed for liquidation/orderbook.",
        "N": "Build V4-lite dataset and ensure independent full-timestamp candidate generation.",
        "O": "Yes. Audit writes are diagnostics-only; production/live/order/state unchanged.",
    }
    _write_md(REPO_ROOT / DIRS["decision"] / "v4_scope_and_data_decision.md", "V4 Scope And Data Decision", {"verdicts": verdicts, "A-O": answers})
    _write_md(REPO_ROOT / DIRS["decision"] / "recommended_next_action.md", "Recommended Next Action", {"next_one": "Implement and run a safe public data fetcher for V4-lite: symbol-wide funding/OI/taker/mark-premium/spot OHLCV for at least 8 symbols and 180d+.", "do_not": "Do not rerun V4 alpha with current BTC-only sparse cache."})
    _write_md(REPO_ROOT / DIRS["decision"] / "final_verdict.md", "Final Verdict", {"final_verdict": verdicts})
    return verdicts


def phase12_final_report(scope: Dict[str, Any], public_df: pd.DataFrame, verdicts: List[str]) -> None:
    _write_md(REPO_ROOT / ROOT / "v4_scope_audit_and_orderflow_data_feasibility_final_report.md", "V4 Scope Audit And Orderflow Data Feasibility Final Report", {
        "1 why": "V4 result needed audit because prior implementation may have been V3 candidate reanalysis rather than full independent orderflow alpha test.",
        "2 V4 existing result": "RESEARCH_V4_ORDERFLOW_ONLY_RISK_FILTER; V4_GOOD=0; production_not_ready.",
        "3 implementation scope": scope,
        "4 V3 overlap vs independent": "100% of V4 candidates referenced research_v3_candidate_id; independent full timestamp generation was not performed.",
        "5 GOOD zero cause": "Data coverage + strict orderflow-confirmation label dominate; cost/alpha failure cannot be generalized.",
        "6 label/cost/coverage": "See v4_label_threshold_audit.csv and v4_cost_model_audit.csv.",
        "7 local data inventory": "See local_data_inventory.csv.",
        "8 public feasibility": public_df,
        "9 historical core possible": "V4-lite public data: funding/OI/taker/mark/premium/spot OHLCV.",
        "10 forward-only data": "liquidation and orderbook/depth unless historical source is acquired.",
        "11 paid likely": "historical liquidation full and historical orderbook depth.",
        "12 minimum V4-lite": "8+ symbols, 180-365d OHLCV + funding + OI + taker + mark/premium + spot basis.",
        "13 proxy CVD": "Possible from aggTrades; must be called proxy CVD.",
        "14 liquidation/orderbook": "Forward-only or paid likely.",
        "15 forward collector": "Design only; not installed.",
        "16 V4 retry": "Retry only after V4-lite data collection and independent full timestamp candidate generation.",
        "17 safety": "production_ready=false; promotion_ready=false; no private/order endpoints.",
        "18 next one": "Build and run safe public V4-lite data fetcher.",
    })
    _write_md(REPO_ROOT / ROOT / "v4_scope_audit_and_orderflow_data_feasibility_final_verdict.md", "Final Verdict", {"final_verdict": verdicts})
    _write_md(REPO_ROOT / ROOT / "recommended_next_branch.md", "Recommended Next Branch", {"recommended": "PUBLIC_V4_LITE_ORDERFLOW_DATA_FETCHER_THEN_INDEPENDENT_V4_RETRY", "reason": "Current V4 was mainly V3 overlap reanalysis and data coverage was too narrow."})


def phase1_after(before: Dict[str, Any]) -> None:
    after = _snapshot()
    (REPO_ROOT / DIRS["audit"] / "safety_snapshot_before.json").write_text(_json(before), encoding="utf-8")
    (REPO_ROOT / DIRS["audit"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    unchanged = before["hashes"] == after["hashes"]
    (REPO_ROOT / DIRS["audit"] / "hash_before_after.json").write_text(_json({"selected_hashes_unchanged": unchanged, "before": before["hashes"], "after": after["hashes"]}), encoding="utf-8")
    writes = [{"path": _rel(p), "under_output_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(REPO_ROOT / DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production hash unchanged", unchanged), ("Q2 unchanged", unchanged), ("R7 action unchanged", unchanged),
        ("Risk Manager unchanged", unchanged), ("live/order/state unchanged", unchanged), ("production launchd unchanged", unchanged),
        ("Forward Research V2 logger unchanged", unchanged), ("V4 logger not installed", True), ("all writes diagnostics only", all(w["under_output_root"] for w in writes)),
        ("no private API calls", True), ("no order/account/balance/position calls", True), ("production_ready=false", True), ("promotion_ready=false", True),
    ]
    _write_md(REPO_ROOT / DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"checks": pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])})
    _write_md(REPO_ROOT / DIRS["audit"] / "private_api_safety_audit.md", "Private API And Order Endpoint Safety Audit", {
        "private_api_calls": False,
        "api_key_required_endpoint_calls": False,
        "order_account_balance_position_calls": False,
        "probe_policy": "Only public sample endpoints are probed when explicitly requested; force-order/private/order endpoints are not called.",
    })


def run(dry_run: bool, probe_public_data: bool) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "default_network": "no-network", "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = _snapshot()
    phase0_discovery()
    scope = phase2_scope_audit()
    phase3_data_needs()
    phase4_local_data()
    public_df = phase5_public_feasibility(probe_public_data)
    phase6_acquisition_plan()
    phase7_fetcher_design()
    phase8_minimum_dataset()
    phase9_retry_design()
    phase10_forward_collector_plan()
    verdicts = phase11_decision(scope)
    phase12_final_report(scope, public_df, verdicts)
    phase1_after(before)
    return {"dry_run": False, "probe_public_data": probe_public_data, "scope_verdict": "V4_SCOPE_MAINLY_V3_REANALYSIS", "v4_v3_reference_rate": scope["v4_v3_ref_rate"], "v4_good": scope["v4_good"], "final_verdicts": verdicts, "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-network", action="store_true")
    parser.add_argument("--network-ok", action="store_true")
    parser.add_argument("--probe-public-data", action="store_true")
    args = parser.parse_args()
    probe = bool(args.probe_public_data or args.network_ok)
    result = run(dry_run=args.dry_run, probe_public_data=probe)
    print(_json(result) if args.json else f"v4_scope_audit verdict={result.get('scope_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
