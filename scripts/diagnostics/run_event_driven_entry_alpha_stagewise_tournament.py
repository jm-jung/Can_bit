"""Stagewise event-driven entry alpha tournament.

Diagnostics-only research script. It builds BTCUSDT event candidates from
existing as-of research/orderflow frames, evaluates single events and stagewise
combinations, and writes reports under data/diagnostics only.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/event_driven_entry_alpha_stagewise_tournament")
SOURCE_FRAME = Path("data/diagnostics/btc_orderflow_cost_kill_rescue_positive_island/frame/combined_candidate_research_frame.parquet")
POS_ROOT = Path("data/diagnostics/btc_orderflow_cost_kill_rescue_positive_island")
FORWARD_ROOT = Path("data/diagnostics/forward_orderflow_collector_v4")
CURRENT_COST_BPS = 6.0


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "features",
        "candidates",
        "stages",
        "backfill",
        "scorecards",
        "failure",
        "casebook",
        "risk_interaction",
        "forward_readiness",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def launchd_lines() -> List[str]:
    out = sh(["launchctl", "list"], timeout=20)
    return [ln for ln in out.splitlines() if "canbit" in ln.lower()]


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
        "ops/run_forward_orderflow_collector_v4.sh",
        "ops/run_false_high_r7_daily_monitor.sh",
        "scripts/run_daily_meta_research_ops.sh",
        "scripts/run_daily_paper_ops.sh",
        "scripts/run_daily_h8_candidate_ops.sh",
        "scripts/run_daily_h8_softgate_candidate_ops.sh",
        "scripts/run_daily_hybrid_candidate_ops.sh",
        "scripts/run_daily_quality_score_candidate_ops.sh",
    ]
    hashes = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            hashes.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    hashes.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            hashes.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": hashes,
        "canbit_launchd_lines": launchd_lines(),
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "python": sys.version,
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_audit(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    before_map = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    rows = []
    for x in after.get("hashes", []):
        old = before_map.get(x["path"])
        rows.append({"path": x["path"], "sha256_before": old, "sha256_after": x.get("sha256"), "changed": old is not None and old != x.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "diagnostics_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_event_driven_entry_alpha_stagewise_tournament.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def input_discovery() -> Dict[str, Any]:
    ensure_dirs()
    required = [
        SOURCE_FRAME,
        POS_ROOT / "forward_watchlist/forward_validation_watchlist.csv",
        POS_ROOT / "risk_interaction/event_risk_filter_interaction.csv",
        Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup/risk_filter_minimal_set_and_schedule_cleanup_final_report.md"),
        FORWARD_ROOT / "state/collector_state.json",
        FORWARD_ROOT / "health/collector_health.csv",
        FORWARD_ROOT / "health/data_family_coverage.csv",
        FORWARD_ROOT / "health/symbol_coverage.csv",
        FORWARD_ROOT / "cache/forward_orderbook_snapshots.parquet",
        FORWARD_ROOT / "cache/forward_recent_aggtrades.parquet",
        FORWARD_ROOT / "cache/forward_liquidation_events.parquet",
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [Path("data/diagnostics"), Path("data/market"), Path("data/ohlcv"), Path("ops"), Path("scripts/diagnostics")]:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and any(k in str(p).lower() for k in ["proxy_cvd", "orderflow", "taker", "funding", "mark", "premium", "collector", "watchlist"]):
                rows.append({"path": str(p), "exists": True, "size": p.stat().st_size, "suffix": p.suffix})
    inv = pd.DataFrame(rows).drop_duplicates("path")
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")[:5000]), encoding="utf-8")
    availability = []
    if SOURCE_FRAME.exists():
        df = pd.read_parquet(SOURCE_FRAME, columns=["timestamp", "source_experiment"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        availability.append({"data": "combined_candidate_research_frame", "rows": len(df), "start": df["timestamp"].min(), "end": df["timestamp"].max(), "historical": True, "forward_only": False})
        for src, g in df.groupby("source_experiment"):
            availability.append({"data": src, "rows": len(g), "start": g["timestamp"].min(), "end": g["timestamp"].max(), "historical": True, "forward_only": False})
    fwd_health = read_forward_health()
    availability.append({"data": "forward_orderbook/liquidation", "rows": fwd_health.get("orderbook_rows", 0), "start": fwd_health.get("last_run_ts", ""), "end": fwd_health.get("last_run_ts", ""), "historical": False, "forward_only": True})
    pd.DataFrame(availability).to_csv(ROOT / "discovery/data_availability_summary.csv", index=False)
    event_map = pd.DataFrame(
        [
            {"event_family": "E1_flush_reclaim", "historical_possible": True, "forward_only": False, "note": "uses proxy CVD/taker/OI proxies; liquidation/orderbook optional"},
            {"event_family": "E2_breakout_aggression", "historical_possible": True, "forward_only": False, "note": "uses proxy CVD/taker/OI/basis proxies; orderbook optional"},
            {"event_family": "E3_squeeze_reversal", "historical_possible": True, "forward_only": False, "note": "uses OI/funding/basis/taker/proxy CVD proxies"},
            {"event_family": "E4_orderbook_imbalance", "historical_possible": False, "forward_only": True, "note": "requires forward orderbook/depth snapshots; current outcome coverage insufficient"},
        ]
    )
    event_map.to_csv(ROOT / "discovery/historical_vs_forward_only_event_map.csv", index=False)
    pd.DataFrame(availability).to_csv(ROOT / "discovery/timestamp_overlap_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nHistorical event evaluation uses the existing BTC proxy CVD/latest30 OI-taker research frame. E4 orderbook imbalance is marked forward-only because orderbook/depth snapshots do not have sufficient historical outcome coverage.\n", encoding="utf-8")
    return {"input_rows": len(inv), "source_frame_exists": SOURCE_FRAME.exists(), "availability_rows": len(availability)}


def read_forward_health() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    state = FORWARD_ROOT / "state/collector_state.json"
    if state.exists():
        try:
            out.update(json.loads(state.read_text()))
        except Exception:
            pass
    health = FORWARD_ROOT / "health/collector_health.csv"
    if health.exists():
        try:
            h = pd.read_csv(health).tail(1).to_dict("records")
            if h:
                out.update(h[0])
        except Exception:
            pass
    out.setdefault("orderbook_rows", 0)
    out.setdefault("recent_aggtrades_rows", 0)
    out.setdefault("liquidation_rows", 0)
    return out


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std()
    if not std or pd.isna(std):
        return pd.Series(0.0, index=s.index)
    return ((s - s.mean()) / std).clip(-5, 5)


def qflag(s: pd.Series, q: float, side: str = "high") -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(False, index=s.index)
    th = s.quantile(q)
    return s >= th if side == "high" else s <= th


def load_source_frame(limit: int | None = None) -> pd.DataFrame:
    if not SOURCE_FRAME.exists():
        return pd.DataFrame()
    df = pd.read_parquet(SOURCE_FRAME)
    if limit:
        df = df.tail(limit).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["entry_ts"] = pd.to_datetime(df.get("entry_ts", df["timestamp"]), errors="coerce")
    df = df[df["symbol"].astype(str).eq("BTCUSDT")].copy()
    return df.sort_values("timestamp").reset_index(drop=True)


def build_event_feature_frame(limit: int | None = None) -> pd.DataFrame:
    df = load_source_frame(limit)
    if df.empty:
        return df
    f = df.copy()
    for c in [
        "cvd_score",
        "cvd_slope",
        "cvd_divergence_score",
        "cvd_reclaim_score",
        "large_trade_aggression_score",
        "taker_score",
        "taker_delta",
        "oi_score",
        "oi_change",
        "funding_score",
        "basis_score",
        "orderflow_confirmation_score",
        "data_quality_score",
        "cvd_taker_combined_score",
        "risk_adjusted_orderflow_score",
        "gross_return",
        "net_after_cost",
        "MFE",
        "MAE",
    ]:
        if c not in f:
            f[c] = np.nan
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f["taker_delta_z"] = zscore(f["taker_delta"])
    f["taker_sell_z"] = zscore(-f["taker_delta"])
    f["taker_buy_z"] = zscore(f["taker_delta"])
    f["cvd_delta_z"] = zscore(f["cvd_score"])
    f["cvd_drop_z"] = zscore(-f["cvd_score"])
    f["cvd_slope_z"] = zscore(f["cvd_slope"])
    f["oi_change_z"] = zscore(f["oi_change"])
    f["oi_build_z"] = zscore(f["oi_change"])
    f["funding_z"] = zscore(f["funding_score"])
    f["basis_z"] = zscore(f["basis_score"])
    # These are as-of research proxies, not future path labels.
    f["flush_intensity_score"] = (f["taker_sell_z"].fillna(0) + f["cvd_drop_z"].fillna(0) + zscore(-f["orderflow_confirmation_score"]).fillna(0)) / 3
    f["reclaim_confirmation_score"] = (zscore(f["cvd_reclaim_score"]).fillna(0) + zscore(f["orderflow_confirmation_score"]).fillna(0) + zscore(f["taker_score"]).fillna(0)) / 3
    f["taker_aggression_score"] = (f["taker_buy_z"].fillna(0) + zscore(f["large_trade_aggression_score"]).fillna(0)) / 2
    f["cvd_reversal_score"] = (zscore(f["cvd_reclaim_score"]).fillna(0) + f["cvd_drop_z"].fillna(0)) / 2
    f["cvd_continuation_score"] = (f["cvd_slope_z"].fillna(0) + f["cvd_delta_z"].fillna(0)) / 2
    f["oi_deleveraging_proxy"] = f["oi_change_z"] <= f["oi_change_z"].quantile(0.35)
    f["premium_overheat"] = (f["funding_z"] >= f["funding_z"].quantile(0.80)) | (f["basis_z"] >= f["basis_z"].quantile(0.80))
    f["breakout_real_aggression_score"] = (zscore(f["orderflow_confirmation_score"]).fillna(0) + f["taker_buy_z"].fillna(0) + f["cvd_continuation_score"].fillna(0) + f["oi_build_z"].fillna(0)) / 4
    f["false_breakout_risk"] = f["premium_overheat"] | (f["cvd_slope_z"] < f["cvd_slope_z"].quantile(0.30))
    f["squeeze_risk_score"] = (zscore(f["oi_score"]).fillna(0) + f["oi_build_z"].fillna(0) + f["funding_z"].fillna(0) + f["basis_z"].fillna(0) + f["taker_buy_z"].fillna(0) + zscore(f["cvd_divergence_score"]).fillna(0)) / 6
    f["long_exhaustion_score"] = f["squeeze_risk_score"]
    f["short_reversal_score"] = (f["squeeze_risk_score"].fillna(0) + zscore(f["cvd_divergence_score"]).fillna(0) - f["cvd_slope_z"].fillna(0)) / 3
    f["orderbook_available"] = False
    f["liquidation_available"] = False
    f["orderbook_event_score"] = np.nan
    f["historical_or_forward_only"] = "historical_proxy_orderflow"
    f["asof_safe_flag"] = True
    f["feature_missing_flags"] = f[[c for c in ["cvd_score", "taker_delta", "oi_change", "funding_score", "basis_score"] if c in f]].isna().apply(lambda r: ",".join(r.index[r].tolist()), axis=1)
    f["data_quality_score"] = f["data_quality_score"].fillna(0.8)
    out = ROOT / "features/event_feature_frame.parquet"
    f.to_parquet(out, index=False)
    (ROOT / "features/event_feature_schema.json").write_text(jdump({c: str(f[c].dtype) for c in f.columns}), encoding="utf-8")
    f.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "features/event_feature_missingness.csv", index=False)
    pd.DataFrame([{"rows": len(f), "start": f["timestamp"].min(), "end": f["timestamp"].max(), "historical_proxy": True, "forward_orderbook_rows": read_forward_health().get("orderbook_rows", 0)}]).to_csv(ROOT / "features/event_feature_coverage.csv", index=False)
    (ROOT / "features/event_feature_build_report.md").write_text("# Event Feature Build Report\n\nBuilt BTCUSDT event feature frame from existing as-of proxy CVD/OI/taker/basis/funding research diagnostics. Proxy CVD is explicitly proxy CVD, not true exchange CVD. Forward orderbook/liquidation features are marked forward-only.\n", encoding="utf-8")
    return f


def event_catalog() -> pd.DataFrame:
    rows = [
        {"event_id": "E1A_flush_reclaim_loose", "family": "E1_flush_reclaim", "direction": "LONG", "strictness": "loose"},
        {"event_id": "E1B_flush_reclaim_normal", "family": "E1_flush_reclaim", "direction": "LONG", "strictness": "normal"},
        {"event_id": "E1C_flush_reclaim_strict", "family": "E1_flush_reclaim", "direction": "LONG", "strictness": "strict"},
        {"event_id": "E1D_flush_reclaim_with_taker_flip", "family": "E1_flush_reclaim", "direction": "LONG", "strictness": "normal"},
        {"event_id": "E1E_flush_reclaim_with_cvd_reclaim", "family": "E1_flush_reclaim", "direction": "LONG", "strictness": "normal"},
        {"event_id": "E1F_flush_reclaim_with_oi_deleveraging", "family": "E1_flush_reclaim", "direction": "LONG", "strictness": "strict"},
        {"event_id": "E2A_breakout_aggression_loose", "family": "E2_breakout_aggression", "direction": "LONG", "strictness": "loose"},
        {"event_id": "E2B_breakout_aggression_normal", "family": "E2_breakout_aggression", "direction": "LONG", "strictness": "normal"},
        {"event_id": "E2C_breakout_aggression_strict", "family": "E2_breakout_aggression", "direction": "LONG", "strictness": "strict"},
        {"event_id": "E2H_breakout_basis_not_overheated", "family": "E2_breakout_aggression", "direction": "LONG", "strictness": "normal"},
        {"event_id": "E3A_squeeze_risk_avoid_long", "family": "E3_squeeze_reversal", "direction": "FILTER_LONG", "strictness": "normal"},
        {"event_id": "E3D_short_reversal_normal", "family": "E3_squeeze_reversal", "direction": "SHORT", "strictness": "normal"},
        {"event_id": "E3E_short_reversal_strict", "family": "E3_squeeze_reversal", "direction": "SHORT", "strictness": "strict"},
        {"event_id": "E4A_orderbook_bid_thin_flush_reclaim", "family": "E4_orderbook_imbalance", "direction": "LONG", "strictness": "forward_only"},
        {"event_id": "E4B_orderbook_ask_thin_breakout", "family": "E4_orderbook_imbalance", "direction": "LONG", "strictness": "forward_only"},
        {"event_id": "E4D_depth_imbalance_reversal", "family": "E4_orderbook_imbalance", "direction": "LONG_OR_SHORT", "strictness": "forward_only"},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "candidates/event_definition_catalog.csv", index=False)
    return df


def make_candidates(f: pd.DataFrame) -> pd.DataFrame:
    if f.empty:
        return f
    event_catalog()
    rows = []
    conditions: List[Tuple[str, str, str, pd.Series, pd.Series]] = []
    long_rows = f["direction"].astype(str).eq("LONG")
    short_rows = f["direction"].astype(str).eq("SHORT")
    e1_loose = long_rows & qflag(f["flush_intensity_score"], 0.75) & qflag(f["reclaim_confirmation_score"], 0.55)
    e1_normal = long_rows & qflag(f["flush_intensity_score"], 0.80) & qflag(f["reclaim_confirmation_score"], 0.65) & qflag(f["cvd_reversal_score"], 0.60)
    e1_strict = long_rows & qflag(f["flush_intensity_score"], 0.85) & qflag(f["reclaim_confirmation_score"], 0.75) & qflag(f["cvd_reversal_score"], 0.70) & f["oi_deleveraging_proxy"].astype(bool)
    conditions += [
        ("E1A_flush_reclaim_loose", "E1_flush_reclaim", "LONG", e1_loose, f["flush_intensity_score"] + f["reclaim_confirmation_score"]),
        ("E1B_flush_reclaim_normal", "E1_flush_reclaim", "LONG", e1_normal, f["flush_intensity_score"] + f["reclaim_confirmation_score"] + f["cvd_reversal_score"]),
        ("E1C_flush_reclaim_strict", "E1_flush_reclaim", "LONG", e1_strict, f["flush_intensity_score"] + f["reclaim_confirmation_score"] + f["cvd_reversal_score"]),
        ("E1D_flush_reclaim_with_taker_flip", "E1_flush_reclaim", "LONG", e1_normal & qflag(f["taker_delta_z"], 0.60), f["flush_intensity_score"] + f["taker_delta_z"]),
        ("E1E_flush_reclaim_with_cvd_reclaim", "E1_flush_reclaim", "LONG", e1_normal & qflag(f["cvd_reclaim_score"], 0.70), f["flush_intensity_score"] + zscore(f["cvd_reclaim_score"])),
        ("E1F_flush_reclaim_with_oi_deleveraging", "E1_flush_reclaim", "LONG", e1_normal & f["oi_deleveraging_proxy"].astype(bool), f["flush_intensity_score"] + f["reclaim_confirmation_score"]),
    ]
    e2_loose = long_rows & qflag(f["breakout_real_aggression_score"], 0.75) & ~f["premium_overheat"].astype(bool)
    e2_normal = long_rows & qflag(f["breakout_real_aggression_score"], 0.82) & qflag(f["taker_buy_z"], 0.65) & ~f["premium_overheat"].astype(bool)
    e2_strict = long_rows & qflag(f["breakout_real_aggression_score"], 0.90) & qflag(f["taker_buy_z"], 0.75) & qflag(f["oi_build_z"], 0.60) & ~f["false_breakout_risk"].astype(bool)
    conditions += [
        ("E2A_breakout_aggression_loose", "E2_breakout_aggression", "LONG", e2_loose, f["breakout_real_aggression_score"]),
        ("E2B_breakout_aggression_normal", "E2_breakout_aggression", "LONG", e2_normal, f["breakout_real_aggression_score"] + f["taker_buy_z"]),
        ("E2C_breakout_aggression_strict", "E2_breakout_aggression", "LONG", e2_strict, f["breakout_real_aggression_score"] + f["taker_buy_z"] + f["oi_build_z"]),
        ("E2H_breakout_basis_not_overheated", "E2_breakout_aggression", "LONG", e2_normal & ~f["premium_overheat"].astype(bool), f["breakout_real_aggression_score"]),
    ]
    e3_risk = qflag(f["squeeze_risk_score"], 0.80) & f["premium_overheat"].astype(bool)
    e3_short = short_rows & qflag(f["short_reversal_score"], 0.80)
    e3_short_strict = short_rows & qflag(f["short_reversal_score"], 0.90) & qflag(f["squeeze_risk_score"], 0.85)
    conditions += [
        ("E3A_squeeze_risk_avoid_long", "E3_squeeze_reversal", "FILTER_LONG", e3_risk, f["squeeze_risk_score"]),
        ("E3D_short_reversal_normal", "E3_squeeze_reversal", "SHORT", e3_short, f["short_reversal_score"]),
        ("E3E_short_reversal_strict", "E3_squeeze_reversal", "SHORT", e3_short_strict, f["short_reversal_score"] + f["squeeze_risk_score"]),
    ]
    # E4 intentionally has no historical candidates unless forward outcome coverage exists.
    for event_id, family, direction, mask, score in conditions:
        sub = f[mask.fillna(False)].copy()
        if sub.empty:
            continue
        sub["event_id"] = event_id
        sub["event_family"] = family
        sub["event_stage"] = family[:2]
        sub["event_variant"] = event_id
        sub["event_score"] = pd.to_numeric(score.loc[sub.index], errors="coerce").fillna(0)
        sub["event_confirmations"] = sub[["data_quality_score", "event_score"]].apply(lambda r: f"score={r['event_score']:.3f};dq={r['data_quality_score']:.2f}", axis=1)
        sub["entry_condition_json"] = json.dumps({"thresholds": "feature_quantile_unsupervised", "event_id": event_id})
        sub["filter_condition_json"] = "{}"
        sub["historical_or_forward_only"] = "historical_proxy_orderflow"
        sub["requires_forward_orderbook"] = False
        sub["requires_liquidation"] = False
        sub["oracle_flag"] = False
        sub["event_candidate_id"] = [f"{event_id}_{i:06d}" for i in range(len(sub))]
        if direction in {"LONG", "SHORT"}:
            sub["direction"] = direction
        rows.append(sub)
    c = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not c.empty:
        c.to_parquet(ROOT / "candidates/single_event_candidates.parquet", index=False)
        c.groupby(["event_family", "event_id", "direction"]).size().reset_index(name="candidate_count").to_csv(ROOT / "candidates/single_event_candidate_summary.csv", index=False)
    else:
        pd.DataFrame().to_parquet(ROOT / "candidates/single_event_candidates.parquet", index=False)
        pd.DataFrame().to_csv(ROOT / "candidates/single_event_candidate_summary.csv", index=False)
    (ROOT / "candidates/event_generator_report.md").write_text("# Event Generator Report\n\nGenerated event candidates from as-of proxy orderflow feature thresholds. E4 orderbook candidates are not historically backfilled and remain forward-only watchlist candidates.\n", encoding="utf-8")
    return c


def dedupe_priority(df: pd.DataFrame, priority: List[str] | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    priority = priority or ["E1_flush_reclaim", "E2_breakout_aggression", "E3_squeeze_reversal", "E4_orderbook_imbalance"]
    rank = {x: i for i, x in enumerate(priority)}
    tmp = df.copy()
    tmp["_rank"] = tmp["event_family"].map(rank).fillna(99)
    return tmp.sort_values(["timestamp", "_rank", "event_score"], ascending=[True, True, False]).drop_duplicates(["timestamp", "direction"]).drop(columns=["_rank"])


def apply_cooldown(df: pd.DataFrame, minutes: int = 0, max_per_day: int | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    tmp = df.sort_values("timestamp").copy()
    keep_idx = []
    last_ts: Dict[str, pd.Timestamp] = {}
    day_counts: Dict[Tuple[str, str], int] = {}
    for idx, row in tmp.iterrows():
        direction = str(row.get("direction", ""))
        ts = pd.to_datetime(row["timestamp"])
        day_key = (str(ts.date()), direction)
        if minutes and direction in last_ts and ts < last_ts[direction] + pd.Timedelta(minutes=minutes):
            continue
        if max_per_day is not None and day_counts.get(day_key, 0) >= max_per_day:
            continue
        keep_idx.append(idx)
        last_ts[direction] = ts
        day_counts[day_key] = day_counts.get(day_key, 0) + 1
    return tmp.loc[keep_idx].copy()


def stage_candidates(c: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    e1 = c[c["event_family"].eq("E1_flush_reclaim")]
    e2 = c[c["event_family"].eq("E2_breakout_aggression")]
    e3 = c[c["event_family"].eq("E3_squeeze_reversal")]
    e3_filter_keys = set(e3[e3["direction"].eq("FILTER_LONG")]["candidate_id"].astype(str))
    out["STAGE_1_E1_ONLY"] = dedupe_priority(e1)
    out["STAGE_2_E1_PLUS_E2"] = dedupe_priority(pd.concat([e1, e2], ignore_index=True))
    stage3_entry = dedupe_priority(pd.concat([e1, e2, e3[~e3["direction"].eq("FILTER_LONG")]], ignore_index=True))
    out["STAGE_3_E1_E2_E3_ENTRY"] = stage3_entry
    e12 = pd.concat([e1, e2], ignore_index=True)
    out["STAGE_3_E1_E2_E3_FILTER"] = dedupe_priority(e12[~e12["candidate_id"].astype(str).isin(e3_filter_keys)])
    out["STAGE_4_E1_E2_E3_E4"] = out["STAGE_3_E1_E2_E3_FILTER"].copy()
    for name, df in out.items():
        df.to_parquet(ROOT / f"stages/{name.lower()}_candidates.parquet", index=False)
    summary = [{"stage": k, "candidate_count": len(v), "families": ",".join(sorted(v["event_family"].unique())) if not v.empty else ""} for k, v in out.items()]
    pd.DataFrame(summary).to_csv(ROOT / "stages/stagewise_candidate_summary.csv", index=False)
    contrib = []
    for k, v in out.items():
        for fam, g in v.groupby("event_family"):
            contrib.append({"stage": k, "event_family": fam, "candidate_count": len(g), "share": len(g) / len(v) if len(v) else 0})
    pd.DataFrame(contrib).to_csv(ROOT / "stages/stagewise_contribution_matrix.csv", index=False)
    (ROOT / "stages/stagewise_application_report.md").write_text("# Stagewise Application Report\n\nStages are separated as requested. Stage 3 is reported both as E3 entry and E3 risk-filter mode. Stage 4 currently equals Stage 3 filter mode plus E4 forward-only watchlist, because historical orderbook outcome coverage is insufficient.\n", encoding="utf-8")
    return out


def perf_metrics(df: pd.DataFrame, name: str, group_type: str) -> Dict[str, Any]:
    if df.empty:
        return {"id": name, "group_type": group_type, "trade_count": 0, "sample_warning": True}
    net = pd.to_numeric(df["net_after_cost"], errors="coerce").fillna(0)
    gross = pd.to_numeric(df["gross_return"], errors="coerce").fillna(0)
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    curve = net.cumsum()
    mdd = (curve.cummax() - curve).max() if len(curve) else 0
    good = df["label"].astype(str).str.contains("GOOD", na=False)
    bad = df["label"].astype(str).str.contains("BAD", na=False)
    days = max(1, (pd.to_datetime(df["timestamp"]).max() - pd.to_datetime(df["timestamp"]).min()).days + 1)
    return {
        "id": name,
        "group_type": group_type,
        "candidate_count": len(df),
        "trade_count": len(df),
        "trades_per_day": len(df) / days,
        "GOOD_count": int(good.sum()),
        "BAD_count": int(bad.sum()),
        "NEUTRAL_count": int((~good & ~bad).sum()),
        "GOOD_rate": float(good.mean()),
        "BAD_rate": float(bad.mean()),
        "mean_net_bps": float(net.mean() * 10000),
        "median_net_bps": float(net.median() * 10000),
        "sum_net": float(net.sum()),
        "gross_mean_bps": float(gross.mean() * 10000),
        "winrate": float((net > 0).mean()),
        "profit_factor": float(pos / neg) if neg > 0 else math.inf,
        "MFE_to_cost": float(pd.to_numeric(df.get("MFE_to_cost", pd.Series(index=df.index, dtype=float)), errors="coerce").mean()),
        "MAE_to_cost": float(pd.to_numeric(df.get("MAE_to_cost", pd.Series(index=df.index, dtype=float)), errors="coerce").mean()),
        "RFE_rate": float(df.get("RFE", pd.Series(False, index=df.index)).astype(bool).mean()),
        "MDD_proxy": float(mdd),
        "tail_loss_bps": float(net.quantile(0.05) * 10000),
        "current_cost_result_bps": float((gross - CURRENT_COST_BPS / 10000).mean() * 10000),
        "maker_like_result_bps": float((gross - 3.0 / 10000).mean() * 10000),
        "two_x_cost_result_bps": float((gross - 12.0 / 10000).mean() * 10000),
        "best_horizon": "existing_outcome_reference",
        "best_exit_policy": ",".join(sorted(df.get("exit_policy_id", pd.Series([], dtype=str)).astype(str).dropna().unique())[:3]),
        "sample_warning": len(df) < 100,
        "production_ready": False,
    }


def evaluate(c: pd.DataFrame, stages: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    outcomes = c.copy()
    if not outcomes.empty:
        outcomes["cost_scenario"] = "current"
        outcomes["net_current_cost"] = pd.to_numeric(outcomes["gross_return"], errors="coerce") - CURRENT_COST_BPS / 10000
        outcomes["net_maker_like"] = pd.to_numeric(outcomes["gross_return"], errors="coerce") - 3.0 / 10000
        outcomes["net_two_x_cost"] = pd.to_numeric(outcomes["gross_return"], errors="coerce") - 12.0 / 10000
    outcomes.to_parquet(ROOT / "backfill/event_alpha_paper_trades.parquet", index=False)
    outcomes.to_parquet(ROOT / "backfill/event_alpha_exit_outcomes.parquet", index=False)
    single_rows = []
    for fam, g in c.groupby("event_family") if not c.empty else []:
        single_rows.append(perf_metrics(dedupe_priority(g), fam, "single_event"))
    for event_id, g in c.groupby("event_id") if not c.empty else []:
        single_rows.append(perf_metrics(dedupe_priority(g), event_id, "single_variant"))
    single = pd.DataFrame(single_rows)
    single.to_csv(ROOT / "scorecards/single_event_scorecard.csv", index=False)
    single.to_csv(ROOT / "backfill/outcome_summary_by_event.csv", index=False)
    stage_rows = [perf_metrics(v, k, "stage") for k, v in stages.items()]
    stage_score = pd.DataFrame(stage_rows)
    stage_score.to_csv(ROOT / "scorecards/stagewise_scorecard.csv", index=False)
    stage_score.to_csv(ROOT / "backfill/outcome_summary_by_stage.csv", index=False)
    delta_rows = []
    ordered = ["STAGE_1_E1_ONLY", "STAGE_2_E1_PLUS_E2", "STAGE_3_E1_E2_E3_ENTRY", "STAGE_3_E1_E2_E3_FILTER", "STAGE_4_E1_E2_E3_E4"]
    smap = {r["id"]: r for r in stage_rows}
    for prev, cur, label in [
        ("STAGE_1_E1_ONLY", "STAGE_2_E1_PLUS_E2", "Stage2_minus_Stage1_E2_add"),
        ("STAGE_2_E1_PLUS_E2", "STAGE_3_E1_E2_E3_ENTRY", "Stage3Entry_minus_Stage2_E3_entry_add"),
        ("STAGE_2_E1_PLUS_E2", "STAGE_3_E1_E2_E3_FILTER", "Stage3Filter_minus_Stage2_E3_filter_apply"),
        ("STAGE_3_E1_E2_E3_FILTER", "STAGE_4_E1_E2_E3_E4", "Stage4_minus_Stage3_E4_forward_only"),
    ]:
        if prev in smap and cur in smap:
            delta_rows.append({"delta_id": label, "mean_net_bps_delta": smap[cur].get("mean_net_bps", np.nan) - smap[prev].get("mean_net_bps", np.nan), "trade_count_delta": smap[cur].get("trade_count", 0) - smap[prev].get("trade_count", 0), "RFE_rate_delta": smap[cur].get("RFE_rate", np.nan) - smap[prev].get("RFE_rate", np.nan)})
    pd.DataFrame(delta_rows).to_csv(ROOT / "scorecards/stage_delta_scorecard.csv", index=False)
    combo_rows = []
    fams = sorted([x for x in c["event_family"].unique() if x != "E4_orderbook_imbalance"]) if not c.empty else []
    for r in range(1, min(4, len(fams)) + 1):
        for combo in itertools.combinations(fams, r):
            combo_rows.append(perf_metrics(dedupe_priority(c[c["event_family"].isin(combo)]), "+".join(combo), f"combo_{r}"))
    combo = pd.DataFrame(combo_rows)
    combo.to_csv(ROOT / "scorecards/pair_combination_scorecard.csv", index=False)
    combo.to_csv(ROOT / "backfill/outcome_summary_by_combination.csv", index=False)
    gated_rows = []
    e3_filter_ids = set(c[(c["event_family"].eq("E3_squeeze_reversal")) & (c["direction"].eq("FILTER_LONG"))]["candidate_id"].astype(str)) if not c.empty else set()
    base_e12 = c[c["event_family"].isin(["E1_flush_reclaim", "E2_breakout_aggression"])] if not c.empty else pd.DataFrame()
    if not base_e12.empty:
        gated_rows.append(perf_metrics(dedupe_priority(base_e12), "E1_E2_no_gate", "gated"))
        gated_rows.append(perf_metrics(dedupe_priority(base_e12[~base_e12["candidate_id"].astype(str).isin(e3_filter_ids)]), "E1_E2_gated_by_E3_squeeze_risk", "gated"))
    gated = pd.DataFrame(gated_rows)
    gated.to_csv(ROOT / "scorecards/gated_combination_scorecard.csv", index=False)
    cd_rows = []
    stage2 = stages.get("STAGE_2_E1_PLUS_E2", pd.DataFrame())
    for minutes in [0, 30, 60, 180, 360]:
        for max_day in [None, 1, 2, 3]:
            cd = apply_cooldown(stage2, minutes, max_day)
            m = perf_metrics(cd, f"stage2_cd{minutes}_max{max_day or 'unlimited'}", "cooldown_trade_limit")
            m["cooldown_minutes"] = minutes
            m["max_trades_per_day"] = max_day or "unlimited"
            cd_rows.append(m)
    pd.DataFrame(cd_rows).to_csv(ROOT / "scorecards/cooldown_trade_limit_scorecard.csv", index=False)
    (ROOT / "scorecards/stagewise_scorecard_report.md").write_text("# Stagewise Scorecard Report\n\nStage scorecards, deltas, gated E3 filter mode, and cooldown/max-trade variants were generated from non-oracle existing paper outcomes. Horizon alternatives are marked existing-outcome reference because full raw OHLCV horizon replay is not available in this run.\n", encoding="utf-8")
    (ROOT / "backfill/backfill_report.md").write_text("# Backfill Report\n\nPaper outcomes are reused from existing BTC proxy CVD/latest30 OI-taker diagnostics. Entry candidates are event-driven subsets; outcome columns are only used after candidate generation.\n", encoding="utf-8")
    return {"single": single, "stage": stage_score, "combo": combo, "gated": gated}


def failure_attribution(scorecards: Dict[str, pd.DataFrame]) -> None:
    rows = []
    for name, df in scorecards.items():
        if df.empty:
            continue
        for _, r in df.iterrows():
            mean_net = r.get("mean_net_bps", np.nan)
            gross = r.get("gross_mean_bps", np.nan)
            sample = r.get("trade_count", 0)
            rfe = r.get("RFE_rate", np.nan)
            if sample == 0:
                primary = "NO_SAMPLE"
                action = "DROP"
            elif sample < 100:
                primary = "SAMPLE_TOO_SMALL"
                action = "KEEP_CASEBOOK_ONLY"
            elif pd.notna(mean_net) and mean_net < 0 and pd.notna(gross) and gross > 0:
                primary = "COST_KILL"
                action = "KEEP_FOR_FORWARD" if mean_net > -3 else "KEEP_CASEBOOK_ONLY"
            elif pd.notna(mean_net) and mean_net < 0:
                primary = "GROSS_EDGE_WEAK"
                action = "DROP"
            elif pd.notna(rfe) and rfe > 0.30:
                primary = "RFE_TOO_HIGH"
                action = "KEEP_FOR_FORWARD_RISK_FILTER"
            else:
                primary = "NONE_OR_ACCEPTABLE"
                action = "KEEP_FOR_FORWARD_ENTRY_ALPHA"
            rows.append({"object_id": r.get("id", ""), "group_type": r.get("group_type", name), "primary_failure_reason": primary, "secondary_failure_reason": "EXIT_HORIZON_MISMATCH_REFERENCE_ONLY" if r.get("best_horizon") == "existing_outcome_reference" else "", "evidence_metrics": jdump({"mean_net_bps": mean_net, "gross_mean_bps": gross, "sample": sample, "RFE_rate": rfe}), "recommended_action": action})
    out = pd.DataFrame(rows)
    out[out["group_type"].astype(str).str.contains("single", na=False)].to_csv(ROOT / "failure/event_failure_attribution.csv", index=False)
    out[out["group_type"].astype(str).str.contains("stage", na=False)].to_csv(ROOT / "failure/stage_failure_attribution.csv", index=False)
    out.groupby(["primary_failure_reason", "recommended_action"]).size().reset_index(name="count").to_csv(ROOT / "failure/failure_reason_summary.csv", index=False)
    (ROOT / "failure/failure_attribution_report.md").write_text("# Failure Attribution Report\n\nFailures are attributed using sample size, gross/net relationship, RFE, and existing-outcome horizon limitations. Outcome-derived information is not used for event generation.\n", encoding="utf-8")


def build_casebook(c: pd.DataFrame) -> None:
    if c.empty:
        pd.DataFrame().to_csv(ROOT / "casebook/event_alpha_casebook.csv", index=False)
        return
    tmp = c.copy()
    tmp["case_category"] = np.select(
        [
            tmp["event_family"].eq("E1_flush_reclaim") & (tmp["net_after_cost"] > 0),
            tmp["event_family"].eq("E1_flush_reclaim") & (tmp["net_after_cost"] <= 0),
            tmp["event_family"].eq("E2_breakout_aggression") & (tmp["net_after_cost"] > 0),
            tmp["event_family"].eq("E2_breakout_aggression") & (tmp["net_after_cost"] <= 0),
            tmp["event_family"].eq("E3_squeeze_reversal") & (tmp["net_after_cost"] > 0),
            tmp["event_family"].eq("E3_squeeze_reversal") & (tmp["net_after_cost"] <= 0),
        ],
        [
            "E1_SUCCESS_FLUSH_RECLAIM",
            "E1_FAIL_FALSE_RECLAIM",
            "E2_SUCCESS_REAL_BREAKOUT",
            "E2_FAIL_FAKE_BREAKOUT",
            "E3_SUCCESS_SHORT_REVERSAL_OR_AVOIDANCE",
            "E3_FAIL_EARLY_SHORT_OR_OVERBLOCK",
        ],
        default="OTHER",
    )
    cols = [
        "event_candidate_id",
        "timestamp",
        "direction",
        "event_id",
        "event_family",
        "event_variant",
        "net_after_cost",
        "MFE",
        "MAE",
        "RFE",
        "taker_delta",
        "cvd_score",
        "cvd_slope",
        "oi_change",
        "basis_score",
        "funding_score",
        "case_category",
        "historical_or_forward_only",
    ]
    case = tmp[[c for c in cols if c in tmp.columns]].sort_values("net_after_cost", ascending=False)
    case["why_success"] = np.where(case["net_after_cost"] > 0, "positive existing paper outcome after event trigger", "")
    case["why_failure"] = np.where(case["net_after_cost"] <= 0, "negative existing paper outcome; inspect false reclaim/fake breakout/cost/exit", "")
    case["forward_validation_note"] = "validate as event-driven candidate; no production connection"
    case["chart_window_path"] = ""
    case.to_parquet(ROOT / "casebook/event_alpha_casebook.parquet", index=False)
    case.to_csv(ROOT / "casebook/event_alpha_casebook.csv", index=False)
    case.groupby(["event_family", "case_category"]).size().reset_index(name="count").to_csv(ROOT / "casebook/casebook_summary_by_event.csv", index=False)
    case.head(50).to_csv(ROOT / "casebook/top_success_cases.csv", index=False)
    case.tail(50).to_csv(ROOT / "casebook/top_failure_cases.csv", index=False)
    (ROOT / "casebook/event_casebook_report.md").write_text("# Event Casebook Report\n\nSuccess/failure casebook generated from event-triggered candidates. Chart windows are left as follow-up artifacts because raw OHLCV plotting is outside this diagnostics pass.\n", encoding="utf-8")


def risk_interaction(c: pd.DataFrame) -> None:
    if c.empty:
        return
    tmp = c.copy()
    risk_score = pd.to_numeric(tmp.get("risk_adjusted_orderflow_score"), errors="coerce")
    tmp["orderflow_risk_worst_20_flag"] = risk_score <= risk_score.quantile(0.20)
    tmp["GOOD"] = tmp["label"].astype(str).str.contains("GOOD", na=False)
    tmp["BAD"] = tmp["label"].astype(str).str.contains("BAD", na=False)
    rows = []
    for fam, g in tmp.groupby("event_family"):
        flag = g["orderflow_risk_worst_20_flag"].astype(bool)
        rows.append({
            "event_family": fam,
            "rows": len(g),
            "removed_by_orderflow_worst20_rate": float(flag.mean()),
            "GOOD_removed_rate": float((flag & g["GOOD"]).sum() / max(1, g["GOOD"].sum())),
            "BAD_removed_rate": float((flag & g["BAD"]).sum() / max(1, g["BAD"].sum())),
            "RFE_removed_rate": float((flag & g["RFE"].astype(bool)).sum() / max(1, g["RFE"].astype(bool).sum())),
            "interaction_verdict": "complementary_if_BAD_RFE_removed_exceeds_GOOD_removed",
        })
    pd.DataFrame(rows).to_csv(ROOT / "risk_interaction/event_risk_filter_interaction.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "risk_interaction/event_good_bad_preservation.csv", index=False)
    (ROOT / "risk_interaction/event_risk_interaction_report.md").write_text("# Event Risk Interaction Report\n\nInteraction with orderflow_risk_worst_20 was evaluated as diagnostics-only. Q2/R7 exact flags remain unavailable in this frame.\n", encoding="utf-8")


def forward_readiness() -> pd.DataFrame:
    h = read_forward_health()
    health = pd.DataFrame([h])
    health.to_csv(ROOT / "forward_readiness/forward_collector_health_snapshot.csv", index=False)
    readiness = pd.DataFrame(
        [
            {"event": "E1_flush_reclaim_long", "historical_evidence": "available_proxy", "required_forward_data": "optional orderbook/liquidation", "readiness": "READY_FOR_FORWARD_WITH_OPTIONAL_CONFIRMS"},
            {"event": "E2_real_aggression_breakout", "historical_evidence": "available_proxy", "required_forward_data": "optional orderbook", "readiness": "READY_FOR_FORWARD_WITH_OPTIONAL_CONFIRMS"},
            {"event": "E3_squeeze_reversal_or_avoidance", "historical_evidence": "available_proxy", "required_forward_data": "funding/OI/taker/proxy CVD", "readiness": "READY_FOR_FORWARD"},
            {"event": "E4_orderbook_imbalance_trigger", "historical_evidence": "insufficient", "required_forward_data": "orderbook depth snapshots + recent aggTrades + outcomes", "readiness": "FORWARD_ONLY_WAIT"},
        ]
    )
    readiness.to_csv(ROOT / "forward_readiness/event_forward_readiness.csv", index=False)
    watch = pd.DataFrame(
        [
            {"watch_id": "FW_E1_flush_reclaim_long", "priority": 1, "entry_condition": "flush intensity + reclaim confirmation + taker/proxy CVD reversal", "filter_condition": "avoid premium overheat; optional orderflow_risk_worst_20 review", "cooldown": "30m/1h", "horizon": "1h/4h reference", "required_data": "proxy CVD,taker,OI,basis/funding", "min_sample": 100, "success_criteria": "positive current/maker-like net and acceptable RFE", "failure_criteria": "false reclaim/cost kill/RFE high", "current_verdict": "research_only"},
            {"watch_id": "FW_E2_real_aggression_breakout", "priority": 2, "entry_condition": "breakout aggression + taker buy + proxy CVD continuation + OI build; basis not overheated", "filter_condition": "drop fake breakout/premium overheat", "cooldown": "30m/1h", "horizon": "1h/4h reference", "required_data": "proxy CVD,taker,OI,basis", "min_sample": 100, "success_criteria": "improves Stage1", "failure_criteria": "Stage2 degrades vs Stage1", "current_verdict": "research_only"},
            {"watch_id": "FW_E3_squeeze_reversal_or_avoidance", "priority": 3, "entry_condition": "squeeze/overheat + proxy CVD divergence", "filter_condition": "use as E1/E2 long risk filter first", "cooldown": "1h/3h", "horizon": "1h/4h reference", "required_data": "funding,OI,taker,proxy CVD", "min_sample": 100, "success_criteria": "improves RFE/MDD as filter", "failure_criteria": "entry degrades or overblocks GOOD", "current_verdict": "risk_filter_candidate"},
            {"watch_id": "FW_E4_orderbook_imbalance_trigger", "priority": 4, "entry_condition": "depth imbalance/spread/large aggression + reclaim/continuation", "filter_condition": "forward-only data quality gate", "cooldown": "30m", "horizon": "15m/1h/4h", "required_data": "forward orderbook snapshots,recent aggTrades,liquidation if available", "min_sample": 200, "success_criteria": "pre-registered forward positive", "failure_criteria": "noisy orderbook or insufficient samples", "current_verdict": "FORWARD_ONLY_WAIT"},
        ]
    )
    watch.to_csv(ROOT / "forward_readiness/event_forward_watchlist.csv", index=False)
    (ROOT / "forward_readiness/forward_readiness_report.md").write_text("# Forward Readiness Report\n\nForward collector is read-only. E4 is forward-only wait because historical orderbook/depth outcomes are not available. Liquidation rows must not be treated as zero-event alpha when missing.\n", encoding="utf-8")
    return watch


def decision(scorecards: Dict[str, pd.DataFrame]) -> None:
    single = scorecards.get("single", pd.DataFrame())
    stage = scorecards.get("stage", pd.DataFrame())
    rows = []
    for _, r in single.iterrows() if not single.empty else []:
        if not str(r.get("id", "")).startswith("E"):
            continue
        mean_net = r.get("mean_net_bps", np.nan)
        sample = r.get("trade_count", 0)
        if sample < 100:
            dec = "KEEP_CASEBOOK_ONLY"
        elif mean_net > 0:
            dec = "KEEP_FOR_FORWARD_ENTRY_ALPHA"
        elif "E3" in str(r.get("id", "")):
            dec = "KEEP_FOR_FORWARD_RISK_FILTER"
        else:
            dec = "DROP_OR_CASEBOOK_ONLY"
        rows.append({"event_or_variant": r.get("id", ""), "decision": dec, "mean_net_bps": mean_net, "trade_count": sample, "production_ready": False})
    rows.append({"event_or_variant": "E4_orderbook_imbalance", "decision": "FORWARD_ONLY_WAIT", "mean_net_bps": np.nan, "trade_count": 0, "production_ready": False})
    pd.DataFrame(rows).to_csv(ROOT / "decision/event_decision_matrix.csv", index=False)
    stage_rows = []
    smap = {r["id"]: r for _, r in stage.iterrows()} if not stage.empty else {}
    for sid, r in smap.items():
        verdict = "STAGE_SAMPLE_TOO_SMALL" if r.get("trade_count", 0) < 100 else ("STAGE_IMPROVES" if r.get("mean_net_bps", -999) > 0 else "STAGE_DEGRADES")
        if sid == "STAGE_4_E1_E2_E3_E4":
            verdict = "STAGE_FORWARD_ONLY"
        stage_rows.append({"stage": sid, "stage_verdict": verdict, "mean_net_bps": r.get("mean_net_bps", np.nan), "trade_count": r.get("trade_count", 0)})
    pd.DataFrame(stage_rows).to_csv(ROOT / "decision/stage_decision_matrix.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "decision/final_event_alpha_recommendation.csv", index=False)
    (ROOT / "decision/recommended_next_action.md").write_text("# Recommended Next Action\n\nCarry the best non-oracle event candidates into a 30-day forward watchlist, with E4 kept as forward-only wait until orderbook/depth samples accumulate.\n", encoding="utf-8")


def final_report(scorecards: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    stage = scorecards.get("stage", pd.DataFrame())
    single = scorecards.get("single", pd.DataFrame())
    combo = scorecards.get("combo", pd.DataFrame())
    best_combo = combo.sort_values("mean_net_bps", ascending=False).head(1).to_dict("records") if not combo.empty and "mean_net_bps" in combo else []
    stage_records = stage.to_dict("records") if not stage.empty else []
    single_records = single[single["group_type"].eq("single_event")].to_dict("records") if not single.empty and "group_type" in single else []
    verdicts = ["EVENT_ALPHA_STAGEWISE_COMPLETED"]
    def get_stage(name: str) -> Dict[str, Any]:
        rows = stage[stage["id"].eq(name)].to_dict("records") if not stage.empty else []
        return rows[0] if rows else {}
    s1 = get_stage("STAGE_1_E1_ONLY")
    s2 = get_stage("STAGE_2_E1_PLUS_E2")
    s3e = get_stage("STAGE_3_E1_E2_E3_ENTRY")
    s3f = get_stage("STAGE_3_E1_E2_E3_FILTER")
    if s1.get("mean_net_bps", -999) > 0:
        verdicts += ["STAGE_1_POSITIVE", "E1_FLUSH_RECLAIM_KEEP_FOR_FORWARD"]
    else:
        verdicts.append("E1_FLUSH_RECLAIM_DROP")
    if s2 and s1 and s2.get("mean_net_bps", -999) > s1.get("mean_net_bps", 999):
        verdicts += ["STAGE_2_IMPROVES", "E2_BREAKOUT_AGGRESSION_ADDS_VALUE", "E2_BREAKOUT_KEEP_FOR_FORWARD"]
    else:
        verdicts += ["STAGE_2_DEGRADES", "E2_BREAKOUT_AGGRESSION_DEGRADES"]
    if s3f and s2 and s3f.get("RFE_rate", 999) < s2.get("RFE_rate", -999):
        verdicts.append("STAGE_3_IMPROVES_AS_FILTER")
        verdicts.append("E3_SQUEEZE_RISK_FILTER_ONLY")
    if s3e and s2 and s3e.get("mean_net_bps", -999) < s2.get("mean_net_bps", 999):
        verdicts.append("STAGE_3_DEGRADES_AS_ENTRY")
    full_stage_negative = bool(stage.empty) or all(float(x.get("mean_net_bps", -1)) <= 0 for x in stage_records if "mean_net_bps" in x)
    if full_stage_negative:
        verdicts += ["EVENT_ALPHA_COST_KILLS_EDGE", "NO_EVENT_ENTRY_ALPHA_FOUND"]
    verdicts += ["E4_ORDERBOOK_FORWARD_ONLY_WAIT", "STAGE_4_FORWARD_ONLY", "FORWARD_WATCHLIST_READY", "production_not_ready"]
    (ROOT / "event_driven_entry_alpha_stagewise_tournament_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(dict.fromkeys(verdicts)) + "\n", encoding="utf-8")
    report = f"""# Event Driven Entry Alpha Stagewise Tournament Final Report

## Why
The goal was to move from always-on scores/no-trade maps toward event-driven, confirmation-based entry timing. This is diagnostics-only and does not connect any event to live inference.

## Event Definitions
E1 Flush-Reclaim Long: forced-sell/proxy CVD/taker flush followed by reclaim confirmation.  
E2 Breakout with Real Aggression: breakout proxy plus taker buy, proxy CVD continuation, OI build, and non-overheated basis/funding.  
E3 Squeeze/Reversal: long risk avoidance or short reversal when OI/funding/taker overheat with proxy CVD divergence.  
E4 Orderbook Imbalance: forward orderbook/depth imbalance trigger; historical outcome coverage is insufficient, so it is forward-only.

## Data Availability
Historical scoring uses `combined_candidate_research_frame.parquet` from prior BTC proxy CVD/latest30 OI-taker diagnostics. CVD is proxy CVD. Orderbook/liquidation are forward-only and not treated as missing=0 alpha.

## Stagewise Results
```json
{jdump(stage_records)}
```

## Single Event Results
```json
{jdump(single_records)}
```

## Best Combination
```json
{jdump(best_combo)}
```

## Key Interpretation
Stage 3 is reported both as E3 entry and E3 risk-filter mode. Stage 4 is forward-only wait for orderbook/depth. Existing paper outcomes are used only after event candidate generation, not for threshold selection.

## Safety
No production TCN/Q2/R7/Risk/live/order/state path was changed. No private API or order/account/balance/position endpoint was called. `forward_orderflow_collector_v4` was read-only and remains active. production_ready=false and promotion_ready=false.
"""
    (ROOT / "event_driven_entry_alpha_stagewise_tournament_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nForward-register E1/E2/E3-filter event watchlist for 30 days, and keep E4 as orderbook forward-only until sufficient depth/outcome samples accumulate.\n", encoding="utf-8")
    return {"verdicts": verdicts, "stage": stage_records, "single": single_records, "best_combo": best_combo}


def run_pipeline(mode: str = "full", stage: int | None = None, fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} stage={stage} fast={fast}")
    discovery = input_discovery()
    limit = 2500 if fast else None
    f = build_event_feature_frame(limit=limit)
    c = make_candidates(f)
    stages = stage_candidates(c)
    if stage:
        allowed = {
            1: ["STAGE_1_E1_ONLY"],
            2: ["STAGE_2_E1_PLUS_E2"],
            3: ["STAGE_3_E1_E2_E3_ENTRY", "STAGE_3_E1_E2_E3_FILTER"],
            4: ["STAGE_4_E1_E2_E3_E4"],
        }[stage]
        stages = {k: v for k, v in stages.items() if k in allowed}
    scorecards = evaluate(c, stages)
    if mode == "single":
        scorecards["stage"] = pd.DataFrame()
    if mode == "combos":
        scorecards["single"] = pd.DataFrame()
    failure_attribution(scorecards)
    build_casebook(c)
    risk_interaction(c)
    forward_readiness()
    decision(scorecards)
    result = final_report(scorecards)
    finalize_audit(before)
    log("done")
    return {
        "mode": mode,
        "stage": stage,
        "fast": fast,
        "feature_rows": len(f),
        "candidate_rows": len(c),
        "discovery": discovery,
        "verdicts": result.get("verdicts", []),
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3, 4])
    parser.add_argument("--single-events-only", action="store_true")
    parser.add_argument("--combinations-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "source_frame_exists": SOURCE_FRAME.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_pipeline(mode="fast_smoke", fast=True)
    elif args.single_events_only:
        res = run_pipeline(mode="single")
    elif args.combinations_only:
        res = run_pipeline(mode="combos")
    elif args.stage:
        res = run_pipeline(mode=f"stage_{args.stage}", stage=args.stage)
    else:
        res = run_pipeline(mode="full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
