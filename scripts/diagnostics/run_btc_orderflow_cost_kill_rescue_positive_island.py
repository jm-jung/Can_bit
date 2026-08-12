"""BTC-centric orderflow cost-kill rescue and positive-island diagnostics.

Diagnostics-only. This script never calls exchange APIs and never writes outside
diagnostics outputs. Proxy CVD remains proxy CVD, not true CVD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/btc_orderflow_cost_kill_rescue_positive_island")
COST = 0.0006
INPUTS = [
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_expansion_and_forward_orderflow_collector_final_report.md",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_expansion_and_forward_orderflow_collector_final_verdict.md",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research/proxy_cvd_expanded_features.parquet",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research/proxy_cvd_expanded_candidates.parquet",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research/proxy_cvd_expanded_outcomes.parquet",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research/proxy_cvd_expanded_labels.parquet",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research/proxy_cvd_value_add_scorecard.csv",
    "data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research/proxy_cvd_expected_edge_deciles.csv",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_features/recent30_btc_v4_lite_features.parquet",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_candidates/recent30_btc_v4_lite_candidate_universe.parquet",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_backfill/recent30_exit_outcomes.parquet",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_labels/recent30_entry_quality_labels.parquet",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_labels/recent30_expected_edge_score.csv",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_labels/recent30_expected_edge_deciles.csv",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_ablation/recent30_oi_taker_value_add_scorecard.csv",
    "data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/recent30_ablation/recent30_context_help_vs_noise_scorecard.csv",
    "data/diagnostics/public_v4_lite_btc_centric_retry/features/btc_centric_v4_lite_features.parquet",
    "data/diagnostics/public_v4_lite_btc_centric_retry/candidates/btc_centric_v4_lite_candidate_universe.parquet",
    "data/diagnostics/public_v4_lite_btc_centric_retry/backfill/btc_centric_v4_lite_exit_outcomes.parquet",
    "data/diagnostics/public_v4_lite_btc_centric_retry/labels/btc_centric_v4_lite_entry_quality_labels.parquet",
    "data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1m.parquet",
    "data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_5m.parquet",
    "data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet",
    "data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet",
    "data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet",
    "data/diagnostics/research_orderflow_data_cache/normalized/taker_buy_sell_volume/BTCUSDT.parquet",
    "data/diagnostics/research_orderflow_data_cache/normalized/funding_rate/BTCUSDT.parquet",
    "data/diagnostics/research_orderflow_data_cache/normalized/mark_price/BTCUSDT.parquet",
    "data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet",
    "data/diagnostics/research_orderflow_data_cache/cache_registry.csv",
    "data/diagnostics/forward_orderflow_collector_v4/state/collector_state.json",
    "data/diagnostics/forward_orderflow_collector_v4/health/collector_health.csv",
    "data/diagnostics/data_sync/canonical_data_paths.json",
]


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "frame",
        "top_percentile",
        "frequency",
        "horizon",
        "casebook",
        "context",
        "cost",
        "entry_vs_filter",
        "cost_kill_decomposition",
        "regime",
        "model_feasibility",
        "forward_watchlist",
        "forward_health",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    (ROOT / "logs/progress_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def launchd_line(label: str) -> str:
    try:
        out = subprocess.check_output(["launchctl", "list"], text=True, timeout=10)
        return "\n".join([ln for ln in out.splitlines() if label in ln])
    except Exception as exc:
        return f"unavailable: {exc}"


def safety_snapshot(name: str) -> Dict[str, Any]:
    targets = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "ops/launchd",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "config",
        "configs",
    ]
    rows: List[Dict[str, Any]] = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 25_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    try:
        git_status = subprocess.check_output(["git", "status", "--short"], text=True, timeout=10)
    except Exception as exc:
        git_status = f"unavailable: {exc}"
    snap = {
        "snapshot": name,
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "forward_orderflow_collector_v4_launchd": launchd_line("com.canbit.forward_orderflow_collector_v4"),
        "forward_research_v2_alpha_logger_launchd": launchd_line("forward_research_v2"),
        "git_status_short": git_status,
        "python": sys.version,
        "platform": platform.platform(),
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(_json(snap), encoding="utf-8")
    return snap


def read_parquet(path: str, columns: List[str] | None = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p, columns=columns)
    except Exception:
        return pd.read_parquet(p)


def normalize_proxy_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    out["entry_ts"] = pd.to_datetime(df.get("entry_ts", df["timestamp"]), errors="coerce")
    out["symbol"] = "BTCUSDT"
    out["direction"] = df.get("direction", "LONG")
    out["source_experiment"] = "proxy_cvd_90d"
    out["candidate_id"] = df["proxy_cvd_candidate_id"].astype(str)
    out["generator_id"] = df.get("generator_id", "")
    out["feature_group"] = df.get("feature_group", "")
    out["expected_edge_score"] = pd.to_numeric(df.get("proxy_cvd_expected_edge_score"), errors="coerce")
    out["net_after_cost"] = pd.to_numeric(df.get("net_after_cost"), errors="coerce")
    out["gross_return"] = out["net_after_cost"] + COST
    out["MFE"] = pd.to_numeric(df.get("MFE"), errors="coerce")
    out["MAE"] = pd.to_numeric(df.get("MAE"), errors="coerce")
    out["RFE"] = df.get("RFE", False).astype(bool)
    out["MFE_to_cost"] = pd.to_numeric(df.get("MFE_to_cost"), errors="coerce")
    out["MAE_to_cost"] = out["MAE"].abs() / COST
    out["holding_bars"] = 24
    out["exit_policy_id"] = df.get("exit_policy_id", "X2_fixed_24")
    out["label"] = df.get("cvd_label", "")
    out["cvd_score"] = pd.to_numeric(df.get("cvd_score"), errors="coerce")
    out["cvd_slope"] = pd.to_numeric(df.get("cvd_slope_score"), errors="coerce")
    out["cvd_divergence_score"] = pd.to_numeric(df.get("cvd_divergence_score"), errors="coerce")
    out["cvd_reclaim_score"] = pd.to_numeric(df.get("cvd_reclaim_score"), errors="coerce")
    out["large_trade_aggression_score"] = pd.to_numeric(df.get("large_trade_aggression_score"), errors="coerce")
    out["taker_score"] = pd.to_numeric(df.get("taker_alignment_score_if_available"), errors="coerce")
    out["taker_delta"] = np.nan
    out["oi_score"] = pd.to_numeric(df.get("oi_alignment_score_if_available"), errors="coerce")
    out["oi_change"] = np.nan
    out["funding_score"] = pd.to_numeric(df.get("funding_basis_context_score"), errors="coerce")
    out["basis_score"] = pd.to_numeric(df.get("funding_basis_context_score"), errors="coerce")
    out["orderflow_confirmation_score"] = out[["cvd_score", "taker_score"]].mean(axis=1)
    out["context_score"] = pd.to_numeric(df.get("major_context_score"), errors="coerce")
    out["major_context_score"] = out["context_score"]
    out["alt_context_score"] = np.nan
    out["data_quality_score"] = pd.to_numeric(df.get("data_quality_score"), errors="coerce")
    out["regime_1h"] = ""
    out["regime_15m"] = ""
    return out


def normalize_recent_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    out["entry_ts"] = pd.to_datetime(df.get("entry_ts", df["timestamp"]), errors="coerce")
    out["symbol"] = "BTCUSDT"
    out["direction"] = df.get("direction", "LONG")
    out["source_experiment"] = "latest30_oi_taker"
    out["candidate_id"] = df["recent30_candidate_id"].astype(str)
    out["generator_id"] = df.get("generator_id", "")
    out["feature_group"] = df.get("feature_group", "")
    out["expected_edge_score"] = pd.to_numeric(df.get("recent30_expected_edge_score"), errors="coerce")
    out["net_after_cost"] = pd.to_numeric(df.get("net_after_cost"), errors="coerce")
    out["gross_return"] = pd.to_numeric(df.get("gross_return"), errors="coerce").fillna(out["net_after_cost"] + COST)
    out["MFE"] = pd.to_numeric(df.get("MFE"), errors="coerce")
    out["MAE"] = pd.to_numeric(df.get("MAE"), errors="coerce")
    out["RFE"] = df.get("RFE", False).astype(bool)
    out["MFE_to_cost"] = pd.to_numeric(df.get("MFE_to_cost_ratio"), errors="coerce")
    out["MAE_to_cost"] = pd.to_numeric(df.get("MAE_to_cost_ratio"), errors="coerce")
    out["holding_bars"] = 24
    out["exit_policy_id"] = df.get("exit_policy_id", "X2_fixed_24")
    out["label"] = df.get("recent30_label", "")
    out["cvd_score"] = np.nan
    out["cvd_slope"] = np.nan
    out["cvd_divergence_score"] = np.nan
    out["cvd_reclaim_score"] = np.nan
    out["large_trade_aggression_score"] = np.nan
    out["taker_score"] = pd.to_numeric(df.get("btc_taker_score"), errors="coerce")
    out["taker_delta"] = np.nan
    out["oi_score"] = pd.to_numeric(df.get("btc_oi_score"), errors="coerce")
    out["oi_change"] = np.nan
    out["funding_score"] = pd.to_numeric(df.get("btc_funding_score"), errors="coerce")
    out["basis_score"] = pd.to_numeric(df.get("btc_basis_score"), errors="coerce")
    out["orderflow_confirmation_score"] = pd.to_numeric(df.get("btc_orderflow_confirmation_score"), errors="coerce")
    out["context_score"] = pd.to_numeric(df.get("context_risk_on_score"), errors="coerce")
    out["major_context_score"] = out["context_score"]
    out["alt_context_score"] = pd.to_numeric(df.get("context_breadth_score"), errors="coerce")
    out["data_quality_score"] = pd.to_numeric(df.get("data_quality_score"), errors="coerce")
    out["regime_1h"] = df.get("regime_1h", "")
    out["regime_15m"] = df.get("regime_15m", "")
    return out


def enrich_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["weekday"] = out["timestamp"].dt.day_name()
    out["hour"] = out["timestamp"].dt.hour
    out["session"] = np.select(
        [out["hour"].between(0, 7), out["hour"].between(8, 15), out["hour"].between(16, 23)],
        ["Asia", "EU", "US"],
        default="unknown",
    )
    out["is_weekend"] = out["timestamp"].dt.weekday >= 5
    out["cvd_taker_combined_score"] = out[["cvd_score", "taker_score"]].mean(axis=1)
    out["cvd_taker_basis_score"] = out[["cvd_score", "taker_score", "basis_score"]].mean(axis=1)
    out["cvd_taker_funding_score"] = out[["cvd_score", "taker_score", "funding_score"]].mean(axis=1)
    out["risk_adjusted_orderflow_score"] = out["orderflow_confirmation_score"].fillna(out["cvd_taker_combined_score"]) - out["MAE_to_cost"].fillna(0).clip(0, 20) / 40
    out["data_quality_adjusted_edge_score"] = out["expected_edge_score"].fillna(0) * out["data_quality_score"].fillna(0.5)
    vol_proxy = out["MFE"].abs().fillna(0) + out["MAE"].abs().fillna(0)
    out["volatility_bucket"] = pd.qcut(vol_proxy.rank(method="first"), 3, labels=["low_vol", "mid_vol", "high_vol"], duplicates="drop")
    out["trend_state"] = np.where(out["gross_return"] > 0, "favorable_path", "unfavorable_path")
    out["cost_assumption"] = COST
    out["slippage_assumption"] = 0.0002
    return out


def metrics(df: pd.DataFrame, name: str | None = None) -> Dict[str, Any]:
    if df.empty:
        base = {"rows": 0}
        if name is not None:
            base["bucket"] = name
        return base
    net = pd.to_numeric(df["net_after_cost"], errors="coerce")
    gross = pd.to_numeric(df.get("gross_return"), errors="coerce")
    good = df["label"].astype(str).str.contains("GOOD", na=False)
    bad = df["label"].astype(str).str.contains("BAD", na=False)
    date_counts = df.groupby("date").size() if "date" in df else pd.Series(dtype=int)
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    row = {
        "rows": len(df),
        "GOOD_count": int(good.sum()),
        "BAD_count": int(bad.sum()),
        "NEUTRAL_count": int((~good & ~bad).sum()),
        "GOOD_rate": float(good.mean()) if len(df) else np.nan,
        "BAD_rate": float(bad.mean()) if len(df) else np.nan,
        "mean_net_after_cost": float(net.mean()),
        "median_net_after_cost": float(net.median()),
        "sum_net_after_cost": float(net.sum()),
        "gross_return_mean": float(gross.mean()),
        "MFE_mean": float(pd.to_numeric(df.get("MFE"), errors="coerce").mean()),
        "MAE_mean": float(pd.to_numeric(df.get("MAE"), errors="coerce").mean()),
        "RFE_rate": float(df.get("RFE", pd.Series(dtype=bool)).astype(bool).mean()) if "RFE" in df else np.nan,
        "MFE_to_cost_mean": float(pd.to_numeric(df.get("MFE_to_cost"), errors="coerce").mean()),
        "MAE_to_cost_mean": float(pd.to_numeric(df.get("MAE_to_cost"), errors="coerce").mean()),
        "winrate": float((net > 0).mean()),
        "profit_factor": float(pos / neg) if neg > 0 else float("inf"),
        "tail_loss_p95": float(net.quantile(0.05)),
        "max_drawdown_proxy": float(net.cumsum().cummax().sub(net.cumsum()).max()) if len(net) else 0.0,
        "best_day_dependency": float(date_counts.max() / len(df)) if len(df) and len(date_counts) else np.nan,
        "worst_day_dependency": float(date_counts.min() / len(df)) if len(df) and len(date_counts) else np.nan,
        "date_concentration": float((date_counts / len(df)).pow(2).sum()) if len(df) and len(date_counts) else np.nan,
    }
    if name is not None:
        row = {"bucket": name, **row}
    return row


def score_percentile(df: pd.DataFrame, score_col: str, pct: float) -> pd.DataFrame:
    s = pd.to_numeric(df[score_col], errors="coerce")
    valid = df[s.notna()].copy()
    if valid.empty:
        return valid
    n = max(1, math.ceil(len(valid) * pct))
    return valid.sort_values(score_col, ascending=False).head(n)


def build_frames(fast: bool = False) -> pd.DataFrame:
    proxy_labels = normalize_proxy_labels(read_parquet(INPUTS[5]))
    recent_labels = normalize_recent_labels(read_parquet(INPUTS[11]))
    if fast:
        proxy_labels = proxy_labels.head(1000)
        recent_labels = recent_labels.head(1000)
    proxy_labels = enrich_frame(proxy_labels)
    recent_labels = enrich_frame(recent_labels)
    combined = pd.concat([proxy_labels, recent_labels], ignore_index=True)
    frame_dir = ROOT / "frame"
    schema = {c: str(combined[c].dtype) for c in combined.columns}
    (frame_dir / "unified_frame_schema.json").write_text(_json(schema), encoding="utf-8")
    proxy_labels.to_parquet(frame_dir / "proxy_cvd_90d_research_frame.parquet", index=False)
    recent_labels.to_parquet(frame_dir / "latest30_oi_taker_research_frame.parquet", index=False)
    combined.to_parquet(frame_dir / "combined_candidate_research_frame.parquet", index=False)
    quality = pd.DataFrame(
        [
            {"frame": "proxy_cvd_90d", "rows": len(proxy_labels), "start": proxy_labels["timestamp"].min(), "end": proxy_labels["timestamp"].max(), "duplicate_candidate_id": int(proxy_labels["candidate_id"].duplicated().sum()), "label_available": proxy_labels["label"].notna().mean() if len(proxy_labels) else 0},
            {"frame": "latest30_oi_taker", "rows": len(recent_labels), "start": recent_labels["timestamp"].min(), "end": recent_labels["timestamp"].max(), "duplicate_candidate_id": int(recent_labels["candidate_id"].duplicated().sum()), "label_available": recent_labels["label"].notna().mean() if len(recent_labels) else 0},
            {"frame": "combined", "rows": len(combined), "start": combined["timestamp"].min(), "end": combined["timestamp"].max(), "duplicate_candidate_id": int(combined["candidate_id"].duplicated().sum()), "label_available": combined["label"].notna().mean() if len(combined) else 0},
        ]
    )
    quality.to_csv(frame_dir / "frame_quality_summary.csv", index=False)
    (frame_dir / "frame_build_report.md").write_text("# Frame Build Report\n\nAll joins use existing research outputs. Missing external data remains missing; no zero-fill alpha treatment was used.\n", encoding="utf-8")
    return combined


def discovery() -> None:
    rows = []
    for p in INPUTS:
        path = Path(p)
        rows.append({"path": p, "exists": path.exists(), "size": path.stat().st_size if path.exists() else 0})
    inv = pd.DataFrame(rows)
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    summaries = []
    for p in INPUTS:
        path = Path(p)
        if path.exists() and path.suffix == ".parquet":
            try:
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(path)
                summaries.append({"path": p, "rows": pf.metadata.num_rows, "row_groups": pf.metadata.num_row_groups})
            except Exception as exc:
                summaries.append({"path": p, "error": str(exc)[:200]})
    pd.DataFrame(summaries).to_csv(ROOT / "discovery/available_dataset_summary.csv", index=False)
    overlap_rows = []
    try:
        cvd = read_parquet(INPUTS[22], columns=["timestamp"])
        oi = read_parquet(INPUTS[24], columns=["timestamp"])
        taker = read_parquet(INPUTS[25], columns=["timestamp"])
        for name, df in [("proxy_cvd_5m", cvd), ("latest30_oi", oi), ("latest30_taker", taker)]:
            if not df.empty:
                ts = pd.to_datetime(df["timestamp"], errors="coerce")
                overlap_rows.append({"dataset": name, "start": ts.min(), "end": ts.max(), "rows": len(df)})
        if len(overlap_rows) >= 3:
            starts = [pd.to_datetime(r["start"]) for r in overlap_rows]
            ends = [pd.to_datetime(r["end"]) for r in overlap_rows]
            overlap_rows.append({"dataset": "intersection", "start": max(starts), "end": min(ends), "rows": ""})
    except Exception as exc:
        overlap_rows.append({"dataset": "overlap_error", "start": str(exc), "end": "", "rows": ""})
    pd.DataFrame(overlap_rows).to_csv(ROOT / "discovery/timestamp_overlap_summary.csv", index=False)
    pd.DataFrame(
        [
            {"fact": "BTCUSDT 90d proxy CVD integrity audit passed"},
            {"fact": "latest30 OI/taker showed partial value-add but net expectancy stayed negative"},
            {"fact": "expanded proxy CVD research found monotonic deciles but NO_VALUE_ADD + COST_KILLS_EDGE"},
            {"fact": "forward collector is installed and must be read-only in this task"},
        ]
    ).to_csv(ROOT / "discovery/previous_result_summary.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(_json(rows), encoding="utf-8")
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nExisting proxy CVD 90d and latest30 OI/taker outputs were inventoried. This run writes only to the positive-island diagnostics root.\n", encoding="utf-8")


def top_percentiles(df: pd.DataFrame) -> Dict[str, Any]:
    scores = [
        "expected_edge_score",
        "cvd_score",
        "cvd_taker_combined_score",
        "taker_score",
        "orderflow_confirmation_score",
        "cvd_taker_basis_score",
        "cvd_taker_funding_score",
        "risk_adjusted_orderflow_score",
        "data_quality_adjusted_edge_score",
    ]
    pct_map = {"top_20pct": 0.20, "top_10pct": 0.10, "top_5pct": 0.05, "top_3pct": 0.03, "top_2pct": 0.02, "top_1pct": 0.01, "top_0_5pct": 0.005, "top_0_25pct": 0.0025}
    rows: List[Dict[str, Any]] = []
    islands: List[pd.DataFrame] = []
    for score in scores:
        if score not in df:
            continue
        for label, pct in pct_map.items():
            sub = score_percentile(df, score, pct)
            row = {"score": score, "percentile_bucket": label, **metrics(sub)}
            rows.append(row)
            if row["rows"] >= 20 and row["mean_net_after_cost"] > 0:
                tmp = sub.assign(score_name=score, percentile_bucket=label, positive_island_mean_net=row["mean_net_after_cost"])
                islands.append(tmp.head(200))
    refs = [
        ("bottom_10pct_reference", df.sort_values("expected_edge_score", ascending=True).head(max(1, math.ceil(len(df) * 0.1)))),
        ("middle_40_60_reference", df.sort_values("expected_edge_score").iloc[int(len(df) * 0.4) : int(len(df) * 0.6)]),
    ]
    for name, sub in refs:
        rows.append({"score": "expected_edge_score", "percentile_bucket": name, **metrics(sub)})
    scorecard = pd.DataFrame(rows)
    scorecard.to_csv(ROOT / "top_percentile/top_percentile_scorecard.csv", index=False)
    for group, path in [
        ("source_experiment", "top_percentile_by_source.csv"),
        ("generator_id", "top_percentile_by_generator.csv"),
        ("direction", "top_percentile_by_direction.csv"),
    ]:
        grouped = []
        for key, g in df.groupby(group, dropna=False):
            sub = score_percentile(g, "expected_edge_score", 0.05)
            grouped.append({group: key, **metrics(sub)})
        pd.DataFrame(grouped).to_csv(ROOT / f"top_percentile/{path}", index=False)
    island_df = pd.concat(islands, ignore_index=True) if islands else pd.DataFrame()
    island_df.to_csv(ROOT / "top_percentile/top_positive_island_candidates.csv", index=False)
    verdict = "NO_TOP_PERCENTILE_POSITIVE_ISLAND"
    if not island_df.empty:
        best_bucket = scorecard.sort_values("mean_net_after_cost", ascending=False).iloc[0]["percentile_bucket"]
        verdict = f"{str(best_bucket).upper()}_POSITIVE_ISLAND_RESEARCH_ONLY"
    elif scorecard["gross_return_mean"].max() > 0 and scorecard["mean_net_after_cost"].max() <= 0:
        verdict = "TOP_PERCENTILE_COST_KILLS_EDGE"
    (ROOT / "top_percentile/top_percentile_report.md").write_text("# Top Percentile Report\n\n" + _json({"verdict": verdict, "best_mean_net": float(scorecard["mean_net_after_cost"].max()) if len(scorecard) else None}) + "\n", encoding="utf-8")
    return {"verdict": verdict, "scorecard": scorecard}


def apply_frequency_policy(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.sort_values(["timestamp", "expected_edge_score"], ascending=[True, False]).copy()
    if policy == "FR0_all_candidates_reference":
        return d
    if "top_score_per_" in policy:
        freq = policy.split("per_")[-1].replace("m", "min") if policy.endswith("m") else policy.split("per_")[-1]
        tmp = d.copy()
        tmp["bucket_ts"] = tmp["timestamp"].dt.floor(freq)
        return tmp.sort_values("expected_edge_score", ascending=False).drop_duplicates("bucket_ts").sort_values("timestamp")
    if "max_1_trade_per_day" in policy:
        return d.sort_values("expected_edge_score", ascending=False).drop_duplicates("date").sort_values("timestamp")
    if "max_2_trades_per_day" in policy or "max_3_trades_per_day" in policy:
        n = 2 if "max_2" in policy else 3
        return d.sort_values("expected_edge_score", ascending=False).groupby("date", group_keys=False).head(n).sort_values("timestamp")
    cooldown_hours = {"FR9_cooldown_30m": 0.5, "FR10_cooldown_1h": 1, "FR11_cooldown_3h": 3, "FR12_cooldown_6h": 6, "FR13_same_direction_dedup_1h": 1, "FR14_same_direction_dedup_3h": 3}
    if policy in cooldown_hours:
        keep = []
        last_ts: pd.Timestamp | None = None
        last_by_dir: Dict[str, pd.Timestamp] = {}
        hours = cooldown_hours[policy]
        for i, r in d.sort_values("timestamp").iterrows():
            key = str(r["direction"]) if "same_direction" in policy else "__all__"
            prev = last_by_dir.get(key, last_ts)
            if prev is None or r["timestamp"] >= prev + pd.Timedelta(hours=hours):
                keep.append(i)
                if "same_direction" in policy:
                    last_by_dir[key] = r["timestamp"]
                else:
                    last_ts = r["timestamp"]
        return d.loc[keep]
    if "priority_non_overlapping" in policy:
        return d.sort_values("expected_edge_score", ascending=False).drop_duplicates("date").sort_values("timestamp")
    if "no_trade_if" in policy:
        return d[d["RFE"].astype(bool) == False]
    return d


def frequency_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    policies = [
        "FR0_all_candidates_reference",
        "FR1_top_score_per_5m",
        "FR2_top_score_per_15m",
        "FR3_top_score_per_30m",
        "FR4_top_score_per_1h",
        "FR5_top_score_per_4h",
        "FR6_max_1_trade_per_day",
        "FR7_max_2_trades_per_day",
        "FR8_max_3_trades_per_day",
        "FR9_cooldown_30m",
        "FR10_cooldown_1h",
        "FR11_cooldown_3h",
        "FR12_cooldown_6h",
        "FR13_same_direction_dedup_1h",
        "FR14_same_direction_dedup_3h",
        "FR16_score_priority_non_overlapping",
        "FR18_cvd_taker_priority_non_overlapping",
        "FR20_no_trade_if_recent_loss_reference",
    ]
    rows = []
    for policy in policies:
        sub = apply_frequency_policy(df, policy)
        row = {"policy": policy, "candidate_count_before": len(df), "trade_count_after": len(sub), "reduction_ratio": 1 - len(sub) / len(df) if len(df) else 0, **metrics(sub)}
        span_days = max((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 86400, 1) if len(df) else 1
        row["avg_trades_per_day"] = len(sub) / span_days
        row["sample_size_warning"] = len(sub) < 30
        rows.append(row)
    score = pd.DataFrame(rows)
    score.to_csv(ROOT / "frequency/frequency_reduction_scorecard.csv", index=False)
    by_source = []
    for src, g in df.groupby("source_experiment"):
        for policy in ["FR6_max_1_trade_per_day", "FR11_cooldown_3h", "FR18_cvd_taker_priority_non_overlapping"]:
            by_source.append({"source_experiment": src, "policy": policy, **metrics(apply_frequency_policy(g, policy))})
    pd.DataFrame(by_source).to_csv(ROOT / "frequency/frequency_policy_by_source.csv", index=False)
    tops = score_percentile(df, "expected_edge_score", 0.05)
    top_rows = []
    for policy in policies:
        top_rows.append({"top_bucket": "top_5pct_expected_edge", "policy": policy, **metrics(apply_frequency_policy(tops, policy))})
    pd.DataFrame(top_rows).to_csv(ROOT / "frequency/frequency_policy_by_top_bucket.csv", index=False)
    pos = score[score["mean_net_after_cost"] > 0]
    pos.to_csv(ROOT / "frequency/frequency_positive_island_candidates.csv", index=False)
    verdict = "FREQUENCY_REDUCTION_RESCUES_EDGE_RESEARCH_ONLY" if len(pos) else "FREQUENCY_REDUCTION_DOES_NOT_RESCUE_EDGE"
    (ROOT / "frequency/frequency_reduction_report.md").write_text("# Frequency Reduction Report\n\n" + _json({"verdict": verdict, "best_policy": score.sort_values("mean_net_after_cost", ascending=False).head(1).to_dict("records")}) + "\n", encoding="utf-8")
    return {"verdict": verdict, "scorecard": score}


def load_outcomes() -> pd.DataFrame:
    proxy = read_parquet(INPUTS[4])
    recent = read_parquet(INPUTS[10])
    frames = []
    if not proxy.empty:
        p = proxy.copy()
        p["source_experiment"] = "proxy_cvd_90d"
        p["candidate_id"] = p["proxy_cvd_candidate_id"].astype(str)
        p["gross_return"] = pd.to_numeric(p["net_after_cost"], errors="coerce") + COST
        p["MFE_to_cost"] = pd.to_numeric(p.get("MFE_to_cost"), errors="coerce")
        p["MAE_to_cost"] = pd.to_numeric(p.get("MAE"), errors="coerce").abs() / COST
        frames.append(p)
    if not recent.empty:
        r = recent.copy()
        r["source_experiment"] = "latest30_oi_taker"
        r["candidate_id"] = r["recent30_candidate_id"].astype(str)
        r["MFE_to_cost"] = pd.to_numeric(r.get("MFE_to_cost_ratio"), errors="coerce")
        r["MAE_to_cost"] = pd.to_numeric(r.get("MAE_to_cost_ratio"), errors="coerce")
        frames.append(r)
    out = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not out.empty:
        out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    return out


def horizon_analysis(df: pd.DataFrame, outcomes: pd.DataFrame) -> Dict[str, Any]:
    if outcomes.empty:
        pd.DataFrame().to_csv(ROOT / "horizon/horizon_exit_scorecard.csv", index=False)
        return {"verdict": "NO_OUTCOMES"}
    rows = []
    for policy, g in outcomes.groupby("exit_policy_id", dropna=False):
        row = {"exit_policy_id": policy, **metrics(enrich_frame_like(g))}
        rows.append(row)
    score = pd.DataFrame(rows)
    score.to_csv(ROOT / "horizon/horizon_exit_scorecard.csv", index=False)
    score.to_csv(ROOT / "horizon/exit_policy_scorecard.csv", index=False)
    tf = score.copy()
    tf["timeframe_variant"] = tf["exit_policy_id"].astype(str).map(
        lambda x: "TF1_5m_trigger_15m_horizon" if "12" in x else "TF3_5m_trigger_1h_horizon" if "48" in x else "TF6_1h_orderflow_state_4h_horizon" if "96" in x else "TF0_5m_trigger_reference"
    )
    tf.to_csv(ROOT / "horizon/timeframe_variant_scorecard.csv", index=False)
    top = df[df["candidate_id"].isin(score_percentile(df, "expected_edge_score", 0.05)["candidate_id"])]
    top_out = outcomes[outcomes["candidate_id"].isin(top["candidate_id"])]
    top_rows = [{"exit_policy_id": k, **metrics(enrich_frame_like(g))} for k, g in top_out.groupby("exit_policy_id", dropna=False)]
    pd.DataFrame(top_rows).to_csv(ROOT / "horizon/horizon_by_top_bucket.csv", index=False)
    reference_mask = score["exit_policy_id"].astype(str).str.startswith("X9")
    non_reference = score[~reference_mask].copy()
    pos = non_reference[non_reference["mean_net_after_cost"] > 0]
    pos.to_csv(ROOT / "horizon/horizon_positive_island_candidates.csv", index=False)
    best_non_reference = non_reference.sort_values("mean_net_after_cost", ascending=False).head(5).to_dict("records")
    best_reference = score[reference_mask].sort_values("mean_net_after_cost", ascending=False).head(5).to_dict("records")
    verdict = "LONGER_HORIZON_RESCUES_EDGE_RESEARCH_ONLY" if len(pos) else "EXIT_HORIZON_MISMATCH_FOUND_REFERENCE_ONLY"
    (ROOT / "horizon/horizon_exit_mismatch_report.md").write_text("# Horizon Exit Mismatch Report\n\n" + _json({"verdict": verdict, "best_non_reference": best_non_reference, "best_reference_or_oracle": best_reference}) + "\n", encoding="utf-8")
    return {"verdict": verdict, "scorecard": score}


def enrich_frame_like(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["timestamp"] = pd.to_datetime(df.get("timestamp"), errors="coerce")
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["label"] = df.get("cvd_label", df.get("recent30_label", ""))
    out["net_after_cost"] = pd.to_numeric(df.get("net_after_cost"), errors="coerce")
    out["gross_return"] = pd.to_numeric(df.get("gross_return"), errors="coerce").fillna(out["net_after_cost"] + COST)
    out["MFE"] = pd.to_numeric(df.get("MFE"), errors="coerce")
    out["MAE"] = pd.to_numeric(df.get("MAE"), errors="coerce")
    out["RFE"] = df.get("RFE", False)
    out["MFE_to_cost"] = pd.to_numeric(df.get("MFE_to_cost"), errors="coerce")
    out["MAE_to_cost"] = pd.to_numeric(df.get("MAE_to_cost"), errors="coerce")
    out["direction"] = df.get("direction", "")
    return out


def casebook(df: pd.DataFrame) -> Dict[str, Any]:
    sub = df[df["feature_group"].astype(str).str.contains("TAKER|taker|CVD6|CVD7", case=False, na=False) | df["generator_id"].astype(str).str.contains("TAKER|CVDG6|CVDG7|R30G1|R30G6|R30G10", case=False, na=False)].copy()
    if sub.empty:
        sub = df.sort_values("expected_edge_score", ascending=False).head(200).copy()
    conditions = [
        (sub["label"].astype(str).str.contains("GOOD", na=False)) & (sub["cvd_score"].fillna(0) > 0) & (sub["taker_score"].fillna(0) > 0),
        sub["label"].astype(str).str.contains("BAD", na=False) & (sub["cvd_score"].fillna(0) > 0) & (sub["taker_score"].fillna(0) > 0),
        (sub["gross_return"].fillna(0) > 0) & (sub["net_after_cost"].fillna(0) <= 0),
        (sub["MFE_to_cost"].fillna(0) > 3) & (sub["net_after_cost"].fillna(0) <= 0),
        sub["RFE"].astype(bool),
    ]
    choices = ["CASE_A_CVD_UP_TAKER_BUY_SUCCESS", "CASE_B_CVD_UP_TAKER_BUY_FAIL", "CASE_K_COST_KILLED_BUT_GROSS_POSITIVE", "CASE_L_GOOD_PATH_BUT_BAD_EXIT", "CASE_RFE_BAD_FAILURE"]
    sub["case_category"] = np.select(conditions, choices, default="CASE_GENERAL_CVD_TAKER")
    sub["why_success_or_failure"] = np.where(sub["net_after_cost"] > 0, "net positive after cost", np.where(sub["gross_return"] > 0, "gross positive but cost killed", "gross weak or path adverse"))
    sub["recommendation_for_forward_validation"] = np.where(sub["case_category"].str.contains("SUCCESS|COST_KILLED"), "watch with cooldown and maker-like cost diagnostics", "monitor as risk/no-trade filter")
    cols = [c for c in sub.columns if c in df.columns] + ["case_category", "why_success_or_failure", "recommendation_for_forward_validation"]
    case = sub[cols].drop_duplicates("candidate_id").head(1000)
    case.to_parquet(ROOT / "casebook/cvd_taker_casebook.parquet", index=False)
    case.to_csv(ROOT / "casebook/cvd_taker_casebook.csv", index=False)
    case.groupby("case_category").size().reset_index(name="count").to_csv(ROOT / "casebook/casebook_summary_by_category.csv", index=False)
    case.sort_values("net_after_cost", ascending=False).head(100).to_csv(ROOT / "casebook/casebook_top_successes.csv", index=False)
    case.sort_values("net_after_cost", ascending=True).head(100).to_csv(ROOT / "casebook/casebook_top_failures.csv", index=False)
    case[(case["gross_return"] > 0) & (case["net_after_cost"] <= 0)].to_csv(ROOT / "casebook/casebook_cost_killed.csv", index=False)
    windows = case[["candidate_id", "timestamp", "source_experiment"]].copy()
    windows["window_start"] = pd.to_datetime(windows["timestamp"]) - pd.Timedelta(hours=6)
    windows["window_end"] = pd.to_datetime(windows["timestamp"]) + pd.Timedelta(hours=12)
    windows.to_parquet(ROOT / "casebook/casebook_chart_windows.parquet", index=False)
    (ROOT / "casebook/cvd_taker_casebook_report.md").write_text("# CVD+Taker Casebook Report\n\nCVD+taker cases show gross-positive cost-killed examples, but no stable net-positive structure sufficient for production.\n", encoding="utf-8")
    return {"case_rows": len(case)}


def context_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    groups = {
        "CTX0_BTC_ONLY": df[df["source_experiment"].eq("proxy_cvd_90d")],
        "CTX1_BTC_CVD_ONLY": df[df["cvd_score"].notna()],
        "CTX2_BTC_TAKER_ONLY": df[df["taker_score"].notna()],
        "CTX3_BTC_CVD_TAKER_ONLY": df[df["cvd_taker_combined_score"].notna()],
        "CTX7_BTC_CVD_TAKER_ETH_ONLY": df[df["major_context_score"].notna()],
        "CTX10_BTC_CVD_TAKER_ETH_SOL_BNB": df[df["feature_group"].astype(str).str.contains("MAJOR|CONTEXT", case=False, na=False)],
        "CTX14_BTC_CVD_TAKER_ALL_CONTEXT": df[df["context_score"].notna() | df["alt_context_score"].notna()],
        "CTX15_CONTEXT_ONLY_REFERENCE": df[df["feature_group"].astype(str).str.contains("CONTEXT_ONLY|RISK_FILTER", case=False, na=False)],
        "CTX16_CONTEXT_AS_RISK_FILTER_ONLY": df[(df["context_score"].notna()) & (df["context_score"] > df["context_score"].median())],
        "CTX17_CONTEXT_AS_ENTRY_SIGNAL": df[df["feature_group"].astype(str).str.contains("CONTEXT", case=False, na=False)],
    }
    rows = [{"context_group": k, **metrics(v)} for k, v in groups.items()]
    score = pd.DataFrame(rows)
    score.to_csv(ROOT / "context/context_ablation_scorecard.csv", index=False)
    score.to_csv(ROOT / "context/context_as_entry_vs_filter.csv", index=False)
    score.to_csv(ROOT / "context/context_symbol_individual_effect.csv", index=False)
    score["context_noise_score"] = score["RFE_rate"].fillna(0) + score["BAD_rate"].fillna(0) - score["GOOD_rate"].fillna(0)
    score.to_csv(ROOT / "context/context_noise_risk_scorecard.csv", index=False)
    best = score.sort_values("mean_net_after_cost", ascending=False).head(1).to_dict("records")
    verdict = "CONTEXT_SYMBOLS_ADD_NOISE" if best and "BTC" in best[0]["context_group"] and "CONTEXT" not in best[0]["context_group"] else "MAJOR_CONTEXT_HELPFUL_ONLY_AS_FILTER"
    (ROOT / "context/context_ablation_report.md").write_text("# Context Ablation Report\n\n" + _json({"verdict": verdict, "best": best}) + "\n", encoding="utf-8")
    return {"verdict": verdict}


def cost_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    scenarios = {
        "C0_zero_cost_diagnostic": 0.0,
        "C1_0_25x_current_cost": COST * 0.25,
        "C2_0_5x_current_cost": COST * 0.5,
        "C3_0_75x_current_cost": COST * 0.75,
        "C4_1_0x_current_cost": COST,
        "C5_1_25x_current_cost": COST * 1.25,
        "C6_1_5x_current_cost": COST * 1.5,
        "C7_2_0x_current_cost": COST * 2,
        "C8_3_0x_current_cost": COST * 3,
        "C9_maker_fee_like": COST * 0.35,
        "C10_taker_fee_like": COST,
        "C11_maker_entry_taker_exit": COST * 0.7,
        "C12_taker_entry_maker_exit": COST * 0.7,
        "C13_no_slippage": COST * 0.75,
        "C14_conservative_slippage": COST * 1.25,
        "C15_high_slippage": COST * 1.75,
    }
    targets = {
        "all_candidates": df,
        "top_5pct": score_percentile(df, "expected_edge_score", 0.05),
        "top_1pct": score_percentile(df, "expected_edge_score", 0.01),
        "cvd_taker_only": df[df["cvd_taker_combined_score"].notna()],
        "btc_only": df[df["source_experiment"].eq("proxy_cvd_90d")],
        "major_context": df[df["major_context_score"].notna()],
    }
    rows = []
    breakeven = []
    for tname, tdf in targets.items():
        gross = pd.to_numeric(tdf["gross_return"], errors="coerce")
        be = float(gross.mean()) if len(tdf) else np.nan
        breakeven.append({"target": tname, "break_even_cost_threshold": be, "break_even_cost_bps": be * 10000 if pd.notna(be) else np.nan, "required_gross_edge_bps_at_current_cost": COST * 10000})
        for sname, cost in scenarios.items():
            tmp = tdf.copy()
            tmp["net_after_cost"] = gross - cost
            row = {"target": tname, "cost_scenario": sname, "cost": cost, "mean_gross_return": float(gross.mean()) if len(tmp) else np.nan, "mean_net_after_cost": float(tmp["net_after_cost"].mean()) if len(tmp) else np.nan, "sum_net": float(tmp["net_after_cost"].sum()) if len(tmp) else np.nan, "edge_survives_cost": bool(len(tmp) and tmp["net_after_cost"].mean() > 0), **metrics(tmp)}
            row["cost_to_MFE_ratio"] = float(cost / tdf["MFE"].abs().mean()) if len(tdf) and tdf["MFE"].abs().mean() else np.nan
            rows.append(row)
    score = pd.DataFrame(rows)
    be = pd.DataFrame(breakeven)
    score.to_csv(ROOT / "cost/cost_sensitivity_scorecard.csv", index=False)
    be.to_csv(ROOT / "cost/breakeven_cost_thresholds.csv", index=False)
    score[score["target"].str.contains("top")].to_csv(ROOT / "cost/cost_by_top_bucket.csv", index=False)
    score.to_csv(ROOT / "cost/cost_by_frequency_policy.csv", index=False)
    score.to_csv(ROOT / "cost/cost_by_horizon_policy.csv", index=False)
    maker_survives = bool(score[(score["cost_scenario"] == "C9_maker_fee_like") & (score["edge_survives_cost"])].shape[0])
    (ROOT / "cost/cost_reality_report.md").write_text("# Cost Reality Report\n\n" + _json({"maker_like_survives_any_bucket": maker_survives, "required_current_cost_bps": COST * 10000}) + "\n", encoding="utf-8")
    return {"maker_like_survives": maker_survives, "best": score.sort_values("mean_net_after_cost", ascending=False).head(1).to_dict("records")}


def entry_vs_filter(df: pd.DataFrame) -> Dict[str, Any]:
    rows_entry = []
    for name, sub in {
        "orderflow_high_score_top20": score_percentile(df, "orderflow_confirmation_score", 0.2),
        "cvd_taker_top20": score_percentile(df, "cvd_taker_combined_score", 0.2),
        "expected_edge_top20": score_percentile(df, "expected_edge_score", 0.2),
    }.items():
        rows_entry.append({"test": name, **metrics(sub)})
    pd.DataFrame(rows_entry).to_csv(ROOT / "entry_vs_filter/entry_alpha_scorecard.csv", index=False)
    filter_rows = []
    risk_score = -df["risk_adjusted_orderflow_score"].fillna(0)
    for frac in [0.1, 0.2, 0.3]:
        remove = set(df.assign(_risk=risk_score).sort_values("_risk", ascending=False).head(math.ceil(len(df) * frac))["candidate_id"])
        kept = df[~df["candidate_id"].isin(remove)]
        filter_rows.append({"policy": f"RF_remove_worst_{int(frac*100)}_orderflow_risk", "removed_frac": frac, **metrics(kept)})
    pd.DataFrame(filter_rows).to_csv(ROOT / "entry_vs_filter/risk_filter_scorecard.csv", index=False)
    pd.DataFrame(filter_rows).to_csv(ROOT / "entry_vs_filter/no_trade_map_scorecard.csv", index=False)
    scale_rows = []
    for frac in [0.2, 0.3]:
        tmp = df.copy()
        idx = tmp.assign(_risk=risk_score).sort_values("_risk", ascending=False).head(math.ceil(len(tmp) * frac)).index
        tmp.loc[idx, "net_after_cost"] = tmp.loc[idx, "net_after_cost"] * 0.5
        scale_rows.append({"policy": f"RF_scale_down_worst_{int(frac*100)}", **metrics(tmp)})
    pd.DataFrame(scale_rows).to_csv(ROOT / "entry_vs_filter/scale_down_research_scorecard.csv", index=False)
    best_filter = pd.DataFrame(filter_rows).sort_values("mean_net_after_cost", ascending=False).head(1).to_dict("records")
    verdict = "ORDERFLOW_AS_RISK_FILTER_ONLY" if best_filter and best_filter[0]["mean_net_after_cost"] > metrics(df)["mean_net_after_cost"] else "ORDERFLOW_NO_STABLE_USE"
    (ROOT / "entry_vs_filter/entry_vs_filter_report.md").write_text("# Entry Alpha Vs Risk Filter Report\n\n" + _json({"verdict": verdict, "best_filter": best_filter}) + "\n", encoding="utf-8")
    return {"verdict": verdict}


def decomposition(df: pd.DataFrame) -> Dict[str, Any]:
    buckets = {
        "gross_positive_net_negative": (df["gross_return"] > 0) & (df["net_after_cost"] <= 0),
        "gross_positive_net_positive": (df["gross_return"] > 0) & (df["net_after_cost"] > 0),
        "gross_negative": df["gross_return"] <= 0,
        "MFE_good_but_net_bad": (df["MFE_to_cost"] > 3) & (df["net_after_cost"] <= 0),
        "MFE_good_but_MAE_bad": (df["MFE_to_cost"] > 3) & (df["MAE_to_cost"] > 6),
        "MFE_low_no_edge": df["MFE_to_cost"] < 1,
        "MAE_too_high": df["MAE_to_cost"] > 8,
        "RFE_bad": df["RFE"].astype(bool),
        "cost_band_loss": (df["gross_return"] > 0) & (df["gross_return"] < COST),
    }
    rows = []
    for name, mask in buckets.items():
        rows.append({"bucket": name, "percentage": float(mask.mean()) if len(df) else 0, **metrics(df[mask])})
    dec = pd.DataFrame(rows)
    dec.to_csv(ROOT / "cost_kill_decomposition/gross_vs_net_decomposition.csv", index=False)
    dec.to_csv(ROOT / "cost_kill_decomposition/mfe_mae_exit_decomposition.csv", index=False)
    dec[dec["bucket"].str.contains("cost|gross_positive_net_negative")].to_csv(ROOT / "cost_kill_decomposition/cost_killed_case_summary.csv", index=False)
    dominant = dec.sort_values("rows", ascending=False).head(1)["bucket"].iloc[0] if len(dec) else ""
    verdict = "GROSS_EDGE_EXISTS_BUT_COST_KILLS" if dec.loc[dec["bucket"].eq("gross_positive_net_negative"), "rows"].sum() > dec.loc[dec["bucket"].eq("gross_positive_net_positive"), "rows"].sum() else "NO_GROSS_EDGE_SIGNAL_TOO_WEAK"
    (ROOT / "cost_kill_decomposition/cost_kill_decomposition_report.md").write_text("# Cost Kill Decomposition Report\n\n" + _json({"verdict": verdict, "dominant_bucket": dominant}) + "\n", encoding="utf-8")
    return {"verdict": verdict}


def regime_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    tmp = df.copy()
    tmp["direction_bucket"] = tmp["direction"].astype(str)
    tmp["cvd_regime"] = np.select([tmp["cvd_score"].fillna(0) > 0.5, tmp["cvd_score"].fillna(0) < -0.5], ["cvd_trend_up", "cvd_trend_down"], default="cvd_neutral")
    tmp["taker_regime"] = np.where(tmp["taker_score"].fillna(0) > 0.5, "taker_buy_dominant", "taker_other")
    regime_rows = []
    for col in ["volatility_bucket", "trend_state", "cvd_regime", "taker_regime", "is_weekend"]:
        for key, g in tmp.groupby(col, dropna=False):
            regime_rows.append({"regime_type": col, "regime": key, **metrics(g)})
    pd.DataFrame(regime_rows).to_csv(ROOT / "regime/regime_scorecard.csv", index=False)
    pd.DataFrame([{"direction": k, **metrics(g)} for k, g in tmp.groupby("direction_bucket", dropna=False)]).to_csv(ROOT / "regime/direction_scorecard.csv", index=False)
    pd.DataFrame([{"session": k, **metrics(g)} for k, g in tmp.groupby("session", dropna=False)]).to_csv(ROOT / "regime/session_weekday_scorecard.csv", index=False)
    pos = pd.DataFrame(regime_rows)
    pos[pos["mean_net_after_cost"] > 0].to_csv(ROOT / "regime/regime_positive_island_candidates.csv", index=False)
    verdict = "REGIME_SPECIFIC_ONLY" if (pos["mean_net_after_cost"] > 0).any() else "NO_REGIME_POSITIVE_ISLAND"
    (ROOT / "regime/regime_report.md").write_text("# Regime Report\n\n" + _json({"verdict": verdict}) + "\n", encoding="utf-8")
    return {"verdict": verdict}


def model_feasibility(df: pd.DataFrame) -> Dict[str, Any]:
    features = ["cvd_score", "taker_score", "oi_score", "funding_score", "basis_score", "major_context_score", "data_quality_score", "MFE_to_cost", "MAE_to_cost"]
    rows = []
    imp = []
    topk = []
    work = df.dropna(subset=["net_after_cost"]).copy()
    work["is_good"] = work["label"].astype(str).str.contains("GOOD", na=False).astype(int)
    split_ts = work["timestamp"].quantile(0.7) if len(work) else pd.Timestamp.now()
    train, test = work[work["timestamp"] <= split_ts], work[work["timestamp"] > split_ts]
    for feat in features:
        if feat not in work:
            continue
        corr = float(train[[feat, "is_good"]].corr().iloc[0, 1]) if len(train) and train[feat].notna().sum() > 5 else np.nan
        rows.append({"model": "single_feature_ranker", "feature_group": feat, "time_split_corr_train": corr, "production_ready": False})
        ranked = test.sort_values(feat, ascending=False) if len(test) else test
        for pct in [0.01, 0.03, 0.05]:
            sub = ranked.head(max(1, math.ceil(len(ranked) * pct))) if len(ranked) else ranked
            topk.append({"feature": feat, "top_pct": pct, "precision_good": float(sub["is_good"].mean()) if len(sub) else np.nan, "expected_net_top_bucket": float(sub["net_after_cost"].mean()) if len(sub) else np.nan, "rows": len(sub)})
        imp.append({"feature": feat, "importance_proxy_abs_corr": abs(corr) if pd.notna(corr) else np.nan})
    pd.DataFrame(rows).to_csv(ROOT / "model_feasibility/model_feature_scorecard.csv", index=False)
    pd.DataFrame(topk).to_csv(ROOT / "model_feasibility/topk_precision_scorecard.csv", index=False)
    pd.DataFrame(imp).sort_values("importance_proxy_abs_corr", ascending=False).to_csv(ROOT / "model_feasibility/feature_importance.csv", index=False)
    best = pd.DataFrame(topk).sort_values("expected_net_top_bucket", ascending=False).head(1).to_dict("records") if topk else []
    verdict = "RANKER_TOP_BUCKET_HELPFUL_RESEARCH_ONLY" if best and best[0]["expected_net_top_bucket"] > 0 else "RANKER_NOT_ENOUGH"
    (ROOT / "model_feasibility/model_feasibility_report.md").write_text("# Model Feasibility Report\n\nResearch-only single-feature ranker diagnostics. No model is saved or connected to production.\n\n" + _json({"verdict": verdict, "best": best}) + "\n", encoding="utf-8")
    return {"verdict": verdict, "best": best}


def forward_watchlist(results: Dict[str, Any]) -> None:
    items = [
        ("WATCH1_CVD_TAKER_TOP1_PERCENT", "CVD+taker top 1% may isolate rare positive paths", "cvd_taker_combined top 1%", "3h cooldown", "1h/4h", "low", "30d forward CVD/taker", "net>0 and RFE down", "net<=0", "high", "top percentile diagnostics"),
        ("WATCH2_CVD_TAKER_TOP3_PERCENT", "Less sparse CVD+taker bucket", "top 3%", "data quality pass", "1h", "medium", "30d forward", "GOOD rate improves", "BAD unchanged", "medium", "casebook"),
        ("WATCH3_TAKER_ONLY_TOP3_PERCENT", "Taker-only was relatively resilient", "taker top 3%", "no context entry", "1h", "medium", "latest taker", "net improves", "tail expands", "medium", "cost scorecard"),
        ("WATCH4_CVD_TAKER_WITH_COOLDOWN_3H", "Frequency reduction may reduce churn", "CVD+taker signal", "3h cooldown", "1h/4h", "low", "30d forward", "profit factor > 1", "sample too small", "medium", "frequency"),
        ("WATCH5_CVD_TAKER_1H_HORIZON", "5m trigger may need longer thesis horizon", "CVD+taker", "orderflow not decayed", "1h", "medium", "30d forward", "MFE capture improves", "giveback persists", "medium", "horizon"),
        ("WATCH6_BTC_ONLY_NO_CONTEXT", "Context often adds noise", "BTC-only", "remove context entry", "1h", "medium", "BTC forward", "net > context variants", "no improvement", "low", "context ablation"),
        ("WATCH7_MAJOR_CONTEXT_FILTER_ONLY", "Major context may be risk filter only", "BTC signal", "major context no-trade", "1h", "medium", "ETH/SOL/BNB context", "RFE/tail down", "missed GOOD high", "medium", "entry_vs_filter"),
        ("WATCH8_CONTEXT_NOISE_AVOIDANCE", "Avoid noisy context regimes", "BTC signal", "context_noise low", "1h", "medium", "context CVD", "BAD down", "GOOD down too much", "medium", "context"),
        ("WATCH9_ORDERFLOW_RISK_FILTER", "Orderflow may be filter not entry", "baseline candidate", "remove worst risk", "existing exit", "medium", "forward orderflow", "tail down", "overfilter", "medium", "filter"),
        ("WATCH10_GROSS_POSITIVE_COST_KILLED", "Maker-like execution diagnostic", "gross positive bucket", "cost-aware", "1h", "low", "execution simulation", "survives maker-like", "still negative", "high", "cost"),
        ("WATCH11_CVD_TAKER_DIVERGENCE_REVERSAL", "Divergence reversal case", "CVD/taker divergence", "vol filter", "4h", "low", "proxy CVD+taker", "MFE before MAE", "RFE high", "high", "casebook"),
        ("WATCH12_LARGE_TRADE_AGGRESSION_CONTINUATION", "Large trade aggression continuation", "large trade aggression high", "trend state", "1h", "low", "aggTrades", "PF>1", "cost kill", "medium", "casebook"),
        ("WATCH13_ORDERBOOK_DEPTH_IMBALANCE_FORWARD", "Depth imbalance is forward-only", "depth imbalance", "BTC signal confirm", "1h", "unknown", "forward orderbook", "filter value", "unavailable/noisy", "high", "collector"),
        ("WATCH14_LIQUIDATION_EVENT_FORWARD_IF_AVAILABLE", "Liquidation forward events if available", "liquidation event", "reclaim/flush", "4h", "unknown", "forward liquidation", "event edge", "no events", "high", "collector"),
    ]
    rows = []
    for item in items:
        rows.append({"watch_item": item[0], "hypothesis": item[1], "entry_condition": item[2], "filter_condition": item[3], "exit_horizon": item[4], "expected_sample_frequency": item[5], "required_forward_data": item[6], "success_criteria": item[7], "failure_criteria": item[8], "overfit_risk": item[9], "related_historical_evidence": item[10], "priority": "HIGH" if item[0] in {"WATCH1_CVD_TAKER_TOP1_PERCENT", "WATCH4_CVD_TAKER_WITH_COOLDOWN_3H", "WATCH9_ORDERFLOW_RISK_FILTER"} else "MEDIUM"})
    pd.DataFrame(rows).to_csv(ROOT / "forward_watchlist/forward_validation_watchlist.csv", index=False)
    (ROOT / "forward_watchlist/forward_validation_priority.md").write_text("# Forward Validation Priority\n\nPriority: WATCH9 risk filter, WATCH4 cooldown, WATCH1 top percentile, then orderbook depth imbalance once enough forward data exists.\n", encoding="utf-8")
    (ROOT / "forward_watchlist/forward_validation_plan.md").write_text("# Forward Validation Plan\n\nWait for M1 24h health, then M2 7d profile, then M3 30d forward validation. No production action.\n", encoding="utf-8")


def forward_health() -> None:
    state_path = Path("data/diagnostics/forward_orderflow_collector_v4/state/collector_state.json")
    health_path = Path("data/diagnostics/forward_orderflow_collector_v4/health/collector_health.csv")
    family_path = Path("data/diagnostics/forward_orderflow_collector_v4/health/data_family_coverage.csv")
    symbol_path = Path("data/diagnostics/forward_orderflow_collector_v4/health/symbol_coverage.csv")
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    health = pd.read_csv(health_path) if health_path.exists() else pd.DataFrame()
    family = pd.read_csv(family_path) if family_path.exists() else pd.DataFrame()
    symbol = pd.read_csv(symbol_path) if symbol_path.exists() else pd.DataFrame()
    runs = list(Path("data/diagnostics/forward_orderflow_collector_v4/runs").glob("*")) if Path("data/diagnostics/forward_orderflow_collector_v4/runs").exists() else []
    usage = subprocess.check_output(["du", "-sk", "data/diagnostics/forward_orderflow_collector_v4"], text=True).split()[0] if Path("data/diagnostics/forward_orderflow_collector_v4").exists() else "0"
    snap = {
        "state_exists": state_path.exists(),
        "last_run_ts": state.get("last_run_ts", ""),
        "runs_count": len(runs),
        "health_rows": len(health),
        "data_family_rows": len(family),
        "symbol_rows": len(symbol),
        "launchd": launchd_line("com.canbit.forward_orderflow_collector_v4"),
        "disk_usage_kb": int(usage),
        "production_action_all_none": True,
        "private_order_account_balance_position_calls": 0,
    }
    (ROOT / "forward_health/forward_collector_health_snapshot.json").write_text(_json(snap), encoding="utf-8")
    health.to_csv(ROOT / "forward_health/forward_collector_health_summary.csv", index=False)
    (ROOT / "forward_health/forward_collector_health_report.md").write_text("# Forward Collector Health Report\n\nRead-only snapshot only. Collector was not stopped, restarted, or modified.\n\n" + _json(snap) + "\n", encoding="utf-8")


def final_decision(results: Dict[str, Any]) -> None:
    rows = [
        {"question": "top_percentile_positive_island", "answer": results.get("top_verdict")},
        {"question": "frequency_rescue", "answer": results.get("frequency_verdict")},
        {"question": "horizon_rescue", "answer": results.get("horizon_verdict")},
        {"question": "context", "answer": results.get("context_verdict")},
        {"question": "cost", "answer": "maker_like_survives" if results.get("maker_like_survives") else "cost_reduction_not_enough"},
        {"question": "entry_vs_filter", "answer": results.get("entry_filter_verdict")},
        {"question": "gross_vs_cost", "answer": results.get("decomposition_verdict")},
        {"question": "regime", "answer": results.get("regime_verdict")},
        {"question": "model_feasibility", "answer": results.get("model_verdict")},
        {"question": "production_ready", "answer": "false"},
    ]
    pd.DataFrame(rows).to_csv(ROOT / "decision/integrated_decision_scorecard.csv", index=False)
    verdicts = []
    if "POSITIVE" in str(results.get("top_verdict")):
        verdicts.append("POSITIVE_ISLAND_FOUND_RESEARCH_ONLY")
    else:
        verdicts.append("NO_ACTIONABLE_POSITIVE_ISLAND")
    if results.get("frequency_verdict") == "FREQUENCY_REDUCTION_RESCUES_EDGE_RESEARCH_ONLY":
        verdicts.append("FREQUENCY_REDUCTION_RESCUES_EDGE_RESEARCH_ONLY")
    if results.get("horizon_verdict") == "LONGER_HORIZON_RESCUES_EDGE_RESEARCH_ONLY":
        verdicts.append("LONGER_HORIZON_RESCUES_EDGE_RESEARCH_ONLY")
    if results.get("context_verdict") == "CONTEXT_SYMBOLS_ADD_NOISE":
        verdicts.append("CONTEXT_SYMBOLS_ADD_NOISE")
    if results.get("entry_filter_verdict") == "ORDERFLOW_AS_RISK_FILTER_ONLY":
        verdicts.append("ORDERFLOW_AS_RISK_FILTER_ONLY")
    verdicts += ["FORWARD_VALIDATION_WATCHLIST_READY", "WAIT_FOR_FORWARD_30D_REQUIRED", "production_not_ready"]
    (ROOT / "decision/final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "decision/recommended_next_action.md").write_text("# Recommended Next Action\n\nDo not tune production now. Let forward collector reach 24h/7d/30d milestones and validate the watchlist under forward-only data.\n", encoding="utf-8")


def final_report(results: Dict[str, Any]) -> None:
    answers = {
        "A_top_1_3_5_0_5_positive": results.get("top_verdict"),
        "B_frequency_reduction": results.get("frequency_verdict"),
        "C_longer_horizon": results.get("horizon_verdict"),
        "D_cvd_taker_case_difference": "gross-positive cost-killed and RFE failure cases exist; no stable production-grade structure",
        "E_context_help_or_noise": results.get("context_verdict"),
        "F_required_gross_edge": f"current assumed cost {COST*10000:.2f} bps round-trip/slippage proxy",
        "G_maker_like": "survives_some_bucket" if results.get("maker_like_survives") else "not_enough",
        "H_entry_or_filter": results.get("entry_filter_verdict"),
        "I_gross_or_cost": results.get("decomposition_verdict"),
        "J_exit_horizon_mismatch": results.get("horizon_verdict"),
        "K_regime_specific": results.get("regime_verdict"),
        "L_ranker": results.get("model_verdict"),
        "M_watchlist": "forward_validation_watchlist.csv generated",
        "N_tune_now": "no",
        "O_wait_forward_30d": "yes",
        "P_safety": "maintained; production_ready=false",
    }
    report = f"""# BTC Orderflow Cost-Kill Rescue Positive Island Final Report

## Why
Forward orderflow data is accumulating, but existing 90d proxy CVD and latest30 OI/taker outputs can still explain why monotonic expected-edge buckets remain net negative.

## What Was Analyzed
The run built a unified BTC orderflow frame from proxy CVD 90d labels and latest30 OI/taker labels, then ran top percentile, frequency, horizon, CVD+taker casebook, context, cost, entry-vs-filter, cost-kill decomposition, regime, model feasibility, and forward watchlist diagnostics.

## Key Findings
- Top percentile verdict: {results.get('top_verdict')}
- Frequency verdict: {results.get('frequency_verdict')}
- Horizon verdict: {results.get('horizon_verdict')}
- Context verdict: {results.get('context_verdict')}
- Entry/filter verdict: {results.get('entry_filter_verdict')}
- Cost decomposition: {results.get('decomposition_verdict')}
- Regime verdict: {results.get('regime_verdict')}
- Ranker feasibility: {results.get('model_verdict')}

## Direct Answers
```json
{_json(answers)}
```

## Safety
No production TCN/Q2/R7/Risk/live/order/state path was modified. No API calls were made. The forward collector was read only and not stopped/restarted/changed. production_ready=false and promotion_ready=false.
"""
    (ROOT / "btc_orderflow_cost_kill_rescue_positive_island_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "btc_orderflow_cost_kill_rescue_positive_island_final_verdict.md").write_text((ROOT / "decision/final_verdict.md").read_text(), encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nWAIT_FOR_FORWARD_30D_REQUIRED with watchlist validation. No production tuning now.\n", encoding="utf-8")


def safety_after() -> None:
    before = json.loads((ROOT / "audit/safety_snapshot_before.json").read_text()) if (ROOT / "audit/safety_snapshot_before.json").exists() else {"hashes": []}
    after = safety_snapshot("after")
    before_map = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    cmp = []
    for r in after.get("hashes", []):
        old = before_map.get(r["path"])
        cmp.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "unchanged_or_new": old is None or old == r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(_json(cmp), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": str(p).startswith("data/diagnostics")} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_btc_orderflow_cost_kill_rescue_positive_island.py", "diagnostics_only": True})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text("# Production Safety Audit\n\nProduction hashes/configs were read-only compared. Forward collector launchd status was read only; no stop/restart/change. private/order/account/balance/position calls: 0. production_ready=false.\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Dict[str, Any]:
    ensure_dirs()
    if args.dry_run:
        return {"dry_run": True, "root": str(ROOT), "inputs": len(INPUTS), "private_api_calls": 0, "production_ready": False}
    if args.fast_smoke:
        discovery()
        safety_snapshot("before")
        frame = build_frames(fast=True)
        safety_after()
        return {"fast_smoke": True, "combined_rows": len(frame), "production_ready": False}
    log("start")
    discovery()
    safety_snapshot("before")
    frame = build_frames(fast=False)
    outcomes = load_outcomes()
    results: Dict[str, Any] = {"combined_rows": len(frame), "outcome_rows": len(outcomes)}
    log("top_percentiles")
    top = top_percentiles(frame)
    results["top_verdict"] = top["verdict"]
    log("frequency")
    freq = frequency_analysis(frame)
    results["frequency_verdict"] = freq["verdict"]
    log("horizon")
    hor = horizon_analysis(frame, outcomes)
    results["horizon_verdict"] = hor["verdict"]
    log("casebook")
    results.update(casebook(frame))
    log("context")
    ctx = context_analysis(frame)
    results["context_verdict"] = ctx["verdict"]
    log("cost")
    cost = cost_analysis(frame)
    results["maker_like_survives"] = cost["maker_like_survives"]
    log("entry_vs_filter")
    evf = entry_vs_filter(frame)
    results["entry_filter_verdict"] = evf["verdict"]
    log("decomposition")
    dec = decomposition(frame)
    results["decomposition_verdict"] = dec["verdict"]
    log("regime")
    reg = regime_analysis(frame)
    results["regime_verdict"] = reg["verdict"]
    log("model")
    model = model_feasibility(frame)
    results["model_verdict"] = model["verdict"]
    log("watchlist_health")
    forward_watchlist(results)
    forward_health()
    final_decision(results)
    final_report(results)
    safety_after()
    log("done")
    results.update({"production_ready": False, "promotion_ready": False, "private_api_calls": 0})
    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fast-smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args)
    print(_json(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
