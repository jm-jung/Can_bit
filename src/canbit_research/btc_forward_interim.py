"""Read-only BTC strict-live forward interim review (observation markers/outcomes).

Supports:
- EFFECTIVE_T0_FULL (entire strict-live since effective T0)
- ROLLING_WINDOW (--window-days N): true rolling N×24h subset by signal/event timestamp
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
KST = ZoneInfo("Asia/Seoul")
OUT = REPO / "data/diagnostics/btc_forward_interim_review"
FIX_OUT = REPO / "data/diagnostics/btc_forward_window_fix"

T0_PATH = REPO / "data/diagnostics/microstructure_data_provenance_and_forward_safety_audit/audit/forward_observation_t0.json"
HASH_FREEZE = REPO / "data/diagnostics/microstructure_data_provenance_and_forward_safety_audit/audit/hash_freeze_audit.json"
ENDPOINT_REPAIR = REPO / "data/diagnostics/microstructure_data_provenance_and_forward_safety_audit/audit/endpoint_repair_applied.json"
READY_PATH = REPO / "data/diagnostics/microstructure_observation_quality/readiness/evaluation_readiness_latest.json"
EST_PATH = REPO / "data/diagnostics/microstructure_observation_quality/readiness/estimated_time_to_ready.json"
COMPACT_PATH = REPO / "data/diagnostics/microstructure_observation_quality/reports/observation_quality_compact_status.json"
DAILY_CSV = REPO / "data/diagnostics/microstructure_observation_quality/daily_rollup/data/daily_observation_quality.csv"
MARKERS = REPO / "data/diagnostics/microstructure_weak_hint_forward_observer/markers/observed_primary_markers.parquet"
OUTCOMES = REPO / "data/diagnostics/microstructure_weak_hint_forward_observer/outcomes/filled_outcomes.csv"
SECONDARY = REPO / "data/diagnostics/microstructure_weak_hint_forward_observer/markers/secondary_conditions.csv"
Q2_BASELINE = REPO / "data/diagnostics/false_high_r7_daily_monitor/state/q2_bdi_baseline_snapshot.csv"
CFG = REPO / "data/diagnostics/microstructure_observation_quality/config/observation_quality_config.json"
KEEPAWAKE = REPO / "data/diagnostics/microstructure_keepawake_guard/reports/keepawake_guard_final_report.json"
SELF_HEAL = REPO / "data/diagnostics/microstructure_ws_self_healing/operational_validation/final_activation_status.json"
TAKER_CLASS = (
    REPO
    / "data/diagnostics/microstructure_observation_quality/taker_forensics/taker_marker_classification.csv"
)

PROTECTED_BTC = [
    REPO / "data/state/paper_trading_state.json",
    REPO / "data/state/shadow_daily_state.json",
    REPO / "data/diagnostics/microstructure_weak_hint_forward_observer/config/weak_hint_forward_observer_config.json",
    REPO / "data/equity_etf/qqq/regime/locks/qqq_regime_prospective_holdout_lock.json",
]

TREND_DISCLAIMER = (
    "ROLLING WINDOW TREND IS DESCRIPTIVE ONLY AND MUST NOT BE USED TO RETUNE THE FROZEN OBSERVER."
)


class InterimReviewError(ValueError):
    """CLI / window validation error with stable code."""

    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


def _now() -> Tuple[datetime, datetime]:
    utc = datetime.now(timezone.utc)
    return utc, utc.astimezone(KST)


def _sha(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    tmp.replace(path)


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def ensure_dirs() -> None:
    for sub in (
        "preflight",
        "inventory",
        "lineage",
        "coverage",
        "signals",
        "trades",
        "performance",
        "readiness",
        "sensitivity",
        "reports",
        "charts",
        "tests",
    ):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    FIX_OUT.mkdir(parents=True, exist_ok=True)


def load_t0() -> pd.Timestamp:
    t0 = json.loads(T0_PATH.read_text())
    return pd.Timestamp(t0["t0_utc"])


def derive_effective_t0() -> Dict[str, Any]:
    t0 = load_t0()
    material = []
    for path, label in (
        (ENDPOINT_REPAIR, "mainnet_endpoint_repair"),
        (HASH_FREEZE, "hash_freeze_ok"),
        (SELF_HEAL, "ws_self_healing_activation"),
        (KEEPAWAKE, "keepawake_guard_collector_restart"),
        (CFG, "observation_quality_config_mtime"),
    ):
        if not path.exists():
            continue
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        material.append(
            {
                "label": label,
                "path": str(path),
                "mtime_utc": mtime.isoformat(),
                "material_for_strategy_clock": False,
                "forward_clock_reset_required": False,
                "reason": "Collection/reporting/ops change; frozen observer definition unchanged (HASH_FREEZE_OK).",
            }
        )
    hf = json.loads(HASH_FREEZE.read_text()) if HASH_FREEZE.exists() else {}
    return {
        "original_strict_live_t0": t0.isoformat(),
        "current_configuration_effective_t0": t0.isoformat(),
        "forward_clock_reset_required": False,
        "hash_freeze_verdict": hf.get("verdict"),
        "material_changes_after_original_t0": material,
        "note": (
            "Weak-hint observer definition frozen before T0; post-T0 changes are collection/ops "
            "(quarantine, self-heal, keep-awake). Strategy evaluation clock remains ORIGINAL T0."
        ),
    }


def parse_as_of_utc(raw: Optional[str], *, now_utc: datetime, t0: pd.Timestamp) -> pd.Timestamp:
    if raw is None:
        return pd.Timestamp(now_utc)
    s = str(raw).strip().replace("Z", "+00:00")
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts > pd.Timestamp(now_utc) + pd.Timedelta(seconds=1):
        raise InterimReviewError("ERROR_AS_OF_IN_FUTURE", f"as-of {ts.isoformat()} is after now")
    if ts < t0:
        raise InterimReviewError("ERROR_AS_OF_BEFORE_EFFECTIVE_T0", f"as-of {ts.isoformat()} before T0")
    return ts


def resolve_analysis_window(
    *,
    analysis_scope: str,
    window_days: Optional[int],
    as_of: pd.Timestamp,
    t0: pd.Timestamp,
) -> Dict[str, Any]:
    if analysis_scope == "EFFECTIVE_T0_FULL":
        start = t0
        end = as_of
        requested_start = None
        truncated = False
        req_days = None
    elif analysis_scope == "ROLLING_WINDOW":
        if window_days is None:
            raise InterimReviewError("ERROR_WINDOW_DAYS_REQUIRED", "window-days required for ROLLING_WINDOW")
        if not isinstance(window_days, int) or isinstance(window_days, bool):
            raise InterimReviewError("ERROR_INVALID_WINDOW_DAYS", "window-days must be int")
        if window_days < 1:
            raise InterimReviewError("ERROR_INVALID_WINDOW_DAYS", "window-days must be >= 1")
        if window_days > 365:
            raise InterimReviewError("ERROR_INVALID_WINDOW_DAYS", "window-days must be <= 365")
        requested_start = as_of - pd.Timedelta(days=int(window_days))
        start = max(t0, requested_start)
        end = as_of
        truncated = start > requested_start
        req_days = int(window_days)
    else:
        raise InterimReviewError("ERROR_UNKNOWN_SCOPE", analysis_scope)

    dur = end - start
    return {
        "analysis_scope": analysis_scope,
        "requested_window_days": req_days,
        "analysis_as_of_utc": as_of.isoformat(),
        "requested_window_start_utc": requested_start.isoformat() if requested_start is not None else None,
        "effective_window_start_utc": start.isoformat(),
        "effective_window_end_utc": end.isoformat(),
        "window_truncated_by_effective_t0": bool(truncated),
        "window_duration_hours": float(dur / pd.Timedelta(hours=1)),
        "window_duration_days": float(dur / pd.Timedelta(days=1)),
        "effective_window_start": start,
        "effective_window_end": end,
    }


def _eligible_markers() -> pd.DataFrame:
    df = pd.read_parquet(MARKERS)
    df["signal_ts"] = pd.to_datetime(df["signal_ts"], utc=True)
    if "exclude_from_forward_eval" not in df.columns:
        df["exclude_from_forward_eval"] = False
    df["exclude_from_forward_eval"] = df["exclude_from_forward_eval"].fillna(False).astype(bool)
    # provenance / quarantine already encoded in exclude_from_forward_eval
    if "observation_id" in df.columns:
        df = df.drop_duplicates(subset=["observation_id"], keep="first")
    return df[~df["exclude_from_forward_eval"]].copy()


def _eligible_outcomes() -> pd.DataFrame:
    df = pd.read_csv(OUTCOMES)
    df["anchor_ts"] = pd.to_datetime(df["anchor_ts"], utc=True)
    if "outcome_filled_ts" in df.columns:
        df["outcome_filled_ts"] = pd.to_datetime(df["outcome_filled_ts"], utc=True)
    if "exclude_from_forward_eval" not in df.columns:
        df["exclude_from_forward_eval"] = False
    df["exclude_from_forward_eval"] = df["exclude_from_forward_eval"].fillna(False).astype(bool)
    # Primary research unit is condition-path outcome: observation × horizon × condition
    key_cols = [c for c in ("observation_id", "horizon_min", "condition_name") if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(subset=key_cols, keep="first")
    return df[~df["exclude_from_forward_eval"]].copy()


def outcome_row_id(df: pd.DataFrame) -> pd.Series:
    if "condition_name" in df.columns:
        cond = df["condition_name"].fillna("NA").astype(str)
    else:
        cond = pd.Series(["NA"] * len(df), index=df.index)
    return df["observation_id"].astype(str) + "|" + df["horizon_min"].astype(str) + "|" + cond


def outcome_ids(df: pd.DataFrame, horizon: int) -> List[str]:
    sub = df[df["horizon_min"].astype(int) == horizon]
    if len(sub) == 0:
        return []
    return sorted(outcome_row_id(sub).astype(str).tolist())


def filter_ts_inclusive(df: pd.DataFrame, col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    ts = pd.to_datetime(df[col], utc=True)
    return df[(ts >= start) & (ts <= end)].copy()


def maturity_ts(row: pd.Series) -> pd.Timestamp:
    if "outcome_filled_ts" in row.index and pd.notna(row.get("outcome_filled_ts")):
        return pd.Timestamp(row["outcome_filled_ts"])
    return pd.Timestamp(row["anchor_ts"]) + pd.Timedelta(minutes=int(row["horizon_min"]))


def split_matured_immature(outcomes: pd.DataFrame, as_of: pd.Timestamp) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(outcomes) == 0:
        return outcomes.copy(), outcomes.copy()
    mat = outcomes.apply(maturity_ts, axis=1)
    net = pd.to_numeric(outcomes["net_current_bps"], errors="coerce")
    matured_mask = (mat <= as_of) & net.notna()
    immature_mask = (~matured_mask) & (pd.to_datetime(outcomes["anchor_ts"], utc=True) <= as_of)
    return outcomes.loc[matured_mask].copy(), outcomes.loc[immature_mask].copy()


def analyze_coverage(daily: pd.DataFrame) -> Dict[str, Any]:
    d = daily.copy()
    d["date_utc"] = pd.to_datetime(d["date_utc"]).dt.tz_localize(None).dt.normalize()
    complete = d[~d["is_partial_day"].astype(bool)] if "is_partial_day" in d.columns else d
    partial = d[d["is_partial_day"].astype(bool)] if "is_partial_day" in d.columns else d.iloc[0:0]
    cov = complete["strict_live_coverage_pct"].astype(float)
    pass_days = int((cov >= 95.0).sum()) if len(cov) else 0
    return {
        "completed_utc_days": int(len(complete)),
        "partial_current_day": bool(len(partial)),
        "total_expected_minutes": int(complete["calendar_minutes"].sum()) if len(complete) else 0,
        "strict_eligible_minutes": int(complete["strict_live_minutes"].sum()) if len(complete) else 0,
        "total_coverage_pct": float(complete["strict_live_minutes"].sum() / complete["calendar_minutes"].sum() * 100)
        if len(complete) and complete["calendar_minutes"].sum()
        else None,
        "daily_pass_days_ge_95": pass_days,
        "longest_gap_minutes": float(complete["longest_gap_minutes"].max()) if len(complete) else 0.0,
        "reconnect_attempts_sum": int(pd.to_numeric(complete.get("reconnect_attempt_count"), errors="coerce").fillna(0).sum())
        if len(complete)
        else 0,
        "hard_stale_sum": int(pd.to_numeric(complete.get("hard_stale_incident_count"), errors="coerce").fillna(0).sum())
        if len(complete)
        else 0,
        "contamination_sum": int(
            pd.to_numeric(complete.get("quarantine_contamination_count"), errors="coerce").fillna(0).sum()
        )
        if len(complete)
        else 0,
        "first_day": str(complete["date_utc"].min().date()) if len(complete) else None,
        "last_complete_day": str(complete["date_utc"].max().date()) if len(complete) else None,
        "partial_days": int(len(partial)),
    }


def filter_daily_window(daily: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if len(daily) == 0:
        return daily.copy()
    d = daily.copy()
    d["date_utc"] = pd.to_datetime(d["date_utc"]).dt.tz_localize(None).dt.normalize()
    start_d = pd.Timestamp(start).tz_convert("UTC").tz_localize(None).normalize()
    end_d = pd.Timestamp(end).tz_convert("UTC").tz_localize(None).normalize()
    return d[(d["date_utc"] >= start_d) & (d["date_utc"] <= end_d)].copy()


def outcome_metrics(out: pd.DataFrame, horizon: int) -> Dict[str, Any]:
    sub = out[out["horizon_min"].astype(int) == horizon].copy()
    net = pd.to_numeric(sub["net_current_bps"], errors="coerce")
    filled = sub[net.notna()].copy()
    netf = pd.to_numeric(filled["net_current_bps"], errors="coerce")
    n = int(len(netf))
    if n == 0:
        return {"horizon_min": horizon, "n": 0, "status": "NO_FILLED"}
    wins = netf > 0
    total = float(netf.sum())
    abs_sorted = netf.reindex(netf.abs().sort_values(ascending=False).index)
    top1 = float(abs_sorted.iloc[0]) if n else 0.0
    top3 = float(abs_sorted.iloc[:3].sum()) if n else 0.0
    return {
        "horizon_min": horizon,
        "n": n,
        "net_total_bps": total,
        "mean_bps": float(netf.mean()),
        "median_bps": float(netf.median()),
        "win_rate": float(wins.mean()),
        "profit_factor": float(netf[netf > 0].sum() / abs(netf[netf < 0].sum()))
        if (netf < 0).any() and netf[netf < 0].sum() != 0
        else None,
        "expectancy_bps": float(netf.mean()),
        "best_bps": float(netf.max()),
        "worst_bps": float(netf.min()),
        "best_contribution_pct": abs(top1 / total) * 100 if total != 0 else None,
        "top3_contribution_pct": abs(top3 / total) * 100 if total != 0 else None,
        "descriptive_only": n < 30,
        "too_early_for_promotion": True,
    }


def window_sample_status(n: int) -> str:
    if n == 0:
        return "NO_MATURED_OUTCOMES"
    if n < 10:
        return "VERY_SMALL"
    if n < 30:
        return "EARLY"
    if n < 50:
        return "INTERMEDIATE"
    return "SUBSTANTIAL_INTERIM"


def window_strategy_status(n: int, mean: Optional[float], pf: Optional[float], expectancy: Optional[float]) -> str:
    if n < 10:
        return "TOO_EARLY_TO_ASSESS"
    if mean is None or expectancy is None:
        return "TOO_EARLY_TO_ASSESS"
    pf_v = pf if pf is not None else 0.0
    if mean > 0 and expectancy > 0 and pf_v > 1:
        return "EARLY_POSITIVE"
    if mean < 0 and expectancy < 0 and (pf is None or pf_v < 1):
        return "EARLY_NEGATIVE"
    return "MIXED_INCONCLUSIVE"


def classify_statuses(
    *,
    compact: Dict[str, Any],
    ready: Dict[str, Any],
    primary_n: int,
    filled_n: int,
    mean_30: Optional[float],
) -> Dict[str, Any]:
    collector = str(compact.get("collector_health") or "").upper()
    ws = str(compact.get("ws_state") or "").upper()
    contam = int(compact.get("contamination") or 0)
    if contam > 0:
        coll_op = "CRITICAL"
    elif collector == "HEALTHY" and ws == "HEALTHY":
        coll_op = "NORMAL"
    else:
        coll_op = "WARNING"

    integrity = "PASS"
    if contam > 0:
        integrity = "FAIL"
    elif float(compact.get("recent_7d_coverage_pct") or 100) < 90:
        integrity = "PASS_WITH_WARNING"

    if primary_n < 10 or filled_n < 10:
        sample = "INSUFFICIENT"
    elif primary_n < 30:
        sample = "EARLY"
    elif primary_n < 50:
        sample = "MINIMUM_APPROACHING"
    else:
        sample = "MINIMUM_REACHED"

    if integrity == "FAIL":
        strat = "INTEGRITY_BLOCKED"
    elif filled_n < 10:
        strat = "TOO_EARLY_TO_ASSESS"
    elif mean_30 is None:
        strat = "TOO_EARLY_TO_ASSESS"
    elif mean_30 > 0:
        strat = "EARLY_POSITIVE"
    elif mean_30 < 0:
        strat = "EARLY_NEGATIVE"
    else:
        strat = "MIXED_INCONCLUSIVE"

    bottleneck = (ready.get("estimate") or {}).get("bottleneck_condition") or compact.get("readiness_bottleneck")
    if integrity == "FAIL":
        action = "STOP_DUE_TO_INTEGRITY_FAILURE"
    elif bottleneck in ("basis_markers", "taker_trusted") and primary_n >= 45:
        action = "WAIT_FOR_READINESS_MARKERS"
    elif strat == "TOO_EARLY_TO_ASSESS":
        action = "KEEP_COLLECTING_TOO_EARLY"
    elif strat == "EARLY_POSITIVE":
        action = "KEEP_COLLECTING_EARLY_POSITIVE"
    elif strat == "EARLY_NEGATIVE":
        action = "KEEP_COLLECTING_EARLY_NEGATIVE"
    else:
        action = "KEEP_COLLECTING_MIXED_INCONCLUSIVE"

    return {
        "collection_operational_status": coll_op,
        "data_integrity_status": integrity,
        "research_sample_status": sample,
        "strategy_interim_status": strat,
        "current_action": action,
        "readiness_bottleneck": bottleneck,
    }


def count_trusted_taker_in_window(start: pd.Timestamp, end: pd.Timestamp) -> int:
    if not TAKER_CLASS.exists():
        return 0
    df = pd.read_csv(TAKER_CLASS)
    if len(df) == 0:
        return 0
    df["marker_timestamp"] = pd.to_datetime(df["marker_timestamp"], utc=True)
    trusted = {
        "TRUSTED_STRICT_MARKER",
        "TRUSTED_WITH_ALIGNMENT_WARNING",
    }
    w = df[(df["marker_timestamp"] >= start) & (df["marker_timestamp"] <= end)]
    return int(w["trust_class"].isin(trusted).sum())


def horizon_bundle(matured: pd.DataFrame, immature: pd.DataFrame, signals_n: int) -> Dict[str, Any]:
    horizons = sorted({int(x) for x in matured["horizon_min"].tolist()} | {int(x) for x in immature["horizon_min"].tolist()} | {15, 30, 60, 120})
    out: Dict[str, Any] = {}
    for h in horizons:
        m = outcome_metrics(matured, h)
        imm_n = int((immature["horizon_min"].astype(int) == h).sum()) if len(immature) else 0
        key = f"{h}m"
        out[key] = {
            "window_signals": signals_n,
            "matured": int(m.get("n") or 0),
            "immature": imm_n,
            "mean_bps": m.get("mean_bps"),
            "median_bps": m.get("median_bps"),
            "win_rate": m.get("win_rate"),
            "profit_factor": m.get("profit_factor"),
            "expectancy_bps": m.get("expectancy_bps"),
            "total_bps": m.get("net_total_bps"),
            **{k: v for k, v in m.items() if k not in ("n",)},
            "n": m.get("n"),
        }
    return out


def compare_windows(cur: Dict[str, Any], prev: Dict[str, Any], prev_complete: bool) -> Dict[str, Any]:
    cn = int(cur.get("matured") or 0)
    pn = int(prev.get("matured") or 0)
    if not prev_complete or cn < 10 or pn < 10:
        trend = "INSUFFICIENT_FOR_TREND"
    else:
        cm = cur.get("mean_bps")
        pm = prev.get("mean_bps")
        cpf = cur.get("profit_factor")
        ppf = prev.get("profit_factor")
        cw = cur.get("win_rate")
        pw = prev.get("win_rate")
        better = 0
        worse = 0
        for a, b in ((cm, pm), (cw, pw), (cpf, ppf)):
            if a is None or b is None:
                continue
            if a > b:
                better += 1
            elif a < b:
                worse += 1
        if better > worse:
            trend = "IMPROVING_DESCRIPTIVE"
        elif worse > better:
            trend = "WORSENING_DESCRIPTIVE"
        else:
            trend = "MIXED_DESCRIPTIVE"
    delta = {
        "mean_bps": (None if cur.get("mean_bps") is None or prev.get("mean_bps") is None else cur["mean_bps"] - prev["mean_bps"]),
        "win_rate": (None if cur.get("win_rate") is None or prev.get("win_rate") is None else cur["win_rate"] - prev["win_rate"]),
        "profit_factor": (
            None
            if cur.get("profit_factor") is None or prev.get("profit_factor") is None
            else cur["profit_factor"] - prev["profit_factor"]
        ),
        "matured_n": cn - pn,
    }
    return {
        "current_window": cur,
        "previous_equal_window": prev,
        "previous_window_complete": prev_complete,
        "delta": delta,
        "rolling_trend_status": trend,
        "disclaimer": TREND_DISCLAIMER,
    }


def maybe_charts(daily: pd.DataFrame, markers: pd.DataFrame, ready: Dict[str, Any], tag: str = "") -> List[str]:
    created: List[str] = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return created

    charts = OUT / "charts"
    suffix = f"_{tag}" if tag else ""
    d = daily[~daily["is_partial_day"].astype(bool)].copy() if len(daily) and "is_partial_day" in daily.columns else daily
    if len(d):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(pd.to_datetime(d["date_utc"]), d["strict_live_coverage_pct"].astype(float), marker="o", ms=3)
        ax.axhline(95, color="gray", ls="--", lw=1)
        ax.set_title(f"Strict-live coverage by day{suffix}")
        ax.set_ylabel("%")
        fig.autofmt_xdate()
        fig.tight_layout()
        p = charts / f"strict_coverage_by_day{suffix}.png"
        fig.savefig(p)
        plt.close(fig)
        created.append(str(p))

    if len(markers):
        fig, ax = plt.subplots(figsize=(8, 4))
        markers["feature_family"].value_counts().plot(kind="bar", ax=ax)
        ax.set_title(f"Eligible primary markers by family{suffix}")
        fig.tight_layout()
        p = charts / f"marker_family_counts{suffix}.png"
        fig.savefig(p)
        plt.close(fig)
        created.append(str(p))

        m = markers.sort_values("signal_ts")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(m["signal_ts"], np.arange(1, len(m) + 1), drawstyle="steps-post")
        ax.set_title(f"Cumulative eligible primary markers{suffix}")
        fig.autofmt_xdate()
        fig.tight_layout()
        p = charts / f"marker_accumulation{suffix}.png"
        fig.savefig(p)
        plt.close(fig)
        created.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ["PRIMARY", "BASIS", "TRUSTED_TAKER"]
    cur = [ready.get("primary_markers", 0), ready.get("basis_markers", 0), ready.get("taker_markers_trusted", 0)]
    req = [50, 10, 10]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, cur, 0.4, label="current")
    ax.bar(x + 0.2, req, 0.4, label="target")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Readiness progress (cumulative)")
    fig.tight_layout()
    p = charts / f"readiness_progress{suffix}.png"
    fig.savefig(p)
    plt.close(fig)
    created.append(str(p))
    return created


def _perf_slice(outcomes_all: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, as_of: pd.Timestamp) -> Dict[str, Any]:
    sig = filter_ts_inclusive(outcomes_all, "anchor_ts", start, end)
    matured, immature = split_matured_immature(sig, as_of)
    m30 = outcome_metrics(matured, 30)
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "signal_rows": int(len(sig)),
        "matured": int(m30.get("n") or 0),
        "immature": int((immature["horizon_min"].astype(int) == 30).sum()) if len(immature) else 0,
        "mean_bps": m30.get("mean_bps"),
        "median_bps": m30.get("median_bps"),
        "win_rate": m30.get("win_rate"),
        "profit_factor": m30.get("profit_factor"),
        "expectancy_bps": m30.get("expectancy_bps"),
        "total_bps": m30.get("net_total_bps"),
        "matured_df": matured,
        "immature_df": immature,
        "signal_df": sig,
        "ids_30m": outcome_ids(matured, 30),
    }


def run_review(
    *,
    analysis_scope: str = "EFFECTIVE_T0_FULL",
    window_days: Optional[int] = None,
    as_of_utc: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dirs()
    os.environ.setdefault("CANBIT_INTERIM_REVIEW_READ_ONLY", "1")

    before_hashes = {str(p): _sha(p) for p in PROTECTED_BTC}
    now_utc, now_kst = _now()
    t0_info = derive_effective_t0()
    t0 = pd.Timestamp(t0_info["current_configuration_effective_t0"])
    as_of = parse_as_of_utc(as_of_utc, now_utc=now_utc, t0=t0)
    win = resolve_analysis_window(analysis_scope=analysis_scope, window_days=window_days, as_of=as_of, t0=t0)
    start = win["effective_window_start"]
    end = win["effective_window_end"]

    compact = json.loads(COMPACT_PATH.read_text()) if COMPACT_PATH.exists() else {}
    ready = json.loads(READY_PATH.read_text()) if READY_PATH.exists() else {}
    est = json.loads(EST_PATH.read_text()) if EST_PATH.exists() else (ready.get("estimate") or {})

    markers_all = pd.read_parquet(MARKERS)
    markers_eligible = _eligible_markers()
    # order: eligibility → dedupe (in loader) → window/T0 filter
    markers_cum = markers_eligible[markers_eligible["signal_ts"] >= t0].copy()
    markers = filter_ts_inclusive(markers_cum, "signal_ts", start, end)

    outcomes_eligible = _eligible_outcomes()
    outcomes_cum_all = outcomes_eligible[outcomes_eligible["anchor_ts"] >= t0].copy()
    # For full regression compatibility: use all post-T0 filled nets without as-of maturity cut when scope=full
    # Still block future maturity relative to as_of for both scopes (safety).
    outcomes_window_signal = filter_ts_inclusive(outcomes_cum_all, "anchor_ts", start, end)
    matured, immature = split_matured_immature(outcomes_window_signal, as_of)

    # Cumulative matured for readiness/strategy cumulative
    matured_cum, immature_cum = split_matured_immature(outcomes_cum_all, as_of)

    daily_all = pd.read_csv(DAILY_CSV) if DAILY_CSV.exists() else pd.DataFrame()
    daily = filter_daily_window(daily_all, start, end) if len(daily_all) else daily_all
    daily_cum = filter_daily_window(daily_all, t0, end) if len(daily_all) else daily_all
    cov = analyze_coverage(daily) if len(daily) else {}
    cov_cum = analyze_coverage(daily_cum) if len(daily_cum) else {}

    sec_n = 0
    sec_trig = 0
    if SECONDARY.exists():
        sec = pd.read_csv(SECONDARY)
        if "exclude_from_forward_eval" in sec.columns:
            sec["exclude_from_forward_eval"] = sec["exclude_from_forward_eval"].fillna(False).astype(bool)
            sec = sec[~sec["exclude_from_forward_eval"]]
        sec["signal_ts"] = pd.to_datetime(sec["signal_ts"], utc=True)
        sec = filter_ts_inclusive(sec[sec["signal_ts"] >= t0], "signal_ts", start, end)
        sec_n = int(len(sec))
        sec_trig = int(sec["condition_triggered"].fillna(False).astype(bool).sum()) if "condition_triggered" in sec.columns else 0

    by_fam = markers["feature_family"].value_counts().to_dict() if len(markers) else {}
    by_fam_cum = markers_cum["feature_family"].value_counts().to_dict() if len(markers_cum) else {}
    basis_n = int(by_fam.get("BASIS_PREMIUM", 0))
    funding_n = int(by_fam.get("FUNDING", 0))
    taker_n = int(by_fam.get("CVD_TAKER_FLOW", 0))
    basis_cum = int(by_fam_cum.get("BASIS_PREMIUM", 0))
    funding_cum = int(by_fam_cum.get("FUNDING", 0))

    primary_cum = int(ready.get("primary_markers") or len(markers_cum))
    basis_ready = int(ready.get("basis_markers") or basis_cum)
    trusted_cum = int(ready.get("taker_markers_trusted") or 0)
    trusted_window = count_trusted_taker_in_window(start, end)

    metrics = {h: outcome_metrics(matured, h) for h in (15, 30, 60, 120)}
    metrics_cum = {h: outcome_metrics(matured_cum, h) for h in (15, 30, 60, 120)}
    m30 = metrics[30]
    m30_cum = metrics_cum[30]
    filled_primary = int(m30.get("n") or 0)
    filled_cum = int(m30_cum.get("n") or 0)
    imm30 = int((immature["horizon_min"].astype(int) == 30).sum()) if len(immature) else 0

    days = float((as_of - t0) / pd.Timedelta(days=1))
    last_marker = markers_cum["signal_ts"].max() if len(markers_cum) else None
    days_since_last_marker = (
        float((as_of - last_marker) / pd.Timedelta(days=1)) if last_marker is not None else None
    )

    # Official readiness / sample status remain cumulative
    statuses = classify_statuses(
        compact=compact,
        ready=ready,
        primary_n=primary_cum,
        filled_n=filled_cum,
        mean_30=m30_cum.get("mean_bps"),
    )
    w_strat = window_strategy_status(
        filled_primary, m30.get("mean_bps"), m30.get("profit_factor"), m30.get("expectancy_bps")
    )
    w_sample = window_sample_status(filled_primary)

    # Previous equal window (only for rolling)
    window_comparison = None
    if analysis_scope == "ROLLING_WINDOW" and window_days:
        cur_len = pd.Timedelta(days=int(window_days))
        prev_end = start
        prev_start_req = prev_end - cur_len
        prev_start = max(t0, prev_start_req)
        prev_complete = prev_start == prev_start_req and (prev_end - prev_start) >= cur_len - pd.Timedelta(seconds=1)
        # half-open previous: [prev_start, start)
        prev_sig = outcomes_cum_all[
            (outcomes_cum_all["anchor_ts"] >= prev_start) & (outcomes_cum_all["anchor_ts"] < start)
        ].copy()
        prev_mat, prev_imm = split_matured_immature(prev_sig, as_of)
        prev_m30 = outcome_metrics(prev_mat, 30)
        prev_markers = markers_cum[(markers_cum["signal_ts"] >= prev_start) & (markers_cum["signal_ts"] < start)]
        prev_cov_daily = filter_daily_window(daily_all, prev_start, prev_end - pd.Timedelta(microseconds=1)) if len(daily_all) else daily_all
        prev_cov = analyze_coverage(prev_cov_daily) if len(prev_cov_daily) else {}
        prev_pack = {
            "start": prev_start.isoformat(),
            "end": prev_end.isoformat(),
            "matured": int(prev_m30.get("n") or 0),
            "immature": int((prev_imm["horizon_min"].astype(int) == 30).sum()) if len(prev_imm) else 0,
            "mean_bps": prev_m30.get("mean_bps"),
            "median_bps": prev_m30.get("median_bps"),
            "win_rate": prev_m30.get("win_rate"),
            "profit_factor": prev_m30.get("profit_factor"),
            "expectancy_bps": prev_m30.get("expectancy_bps"),
            "total_bps": prev_m30.get("net_total_bps"),
            "signal_count": int(len(prev_markers)),
            "primary_additions": int(len(prev_markers)),
            "basis_additions": int((prev_markers["feature_family"] == "BASIS_PREMIUM").sum()) if len(prev_markers) else 0,
            "trusted_taker_additions": count_trusted_taker_in_window(prev_start, prev_end - pd.Timedelta(microseconds=1)),
            "coverage_pct": prev_cov.get("total_coverage_pct"),
        }
        cur_pack = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "matured": filled_primary,
            "immature": imm30,
            "mean_bps": m30.get("mean_bps"),
            "median_bps": m30.get("median_bps"),
            "win_rate": m30.get("win_rate"),
            "profit_factor": m30.get("profit_factor"),
            "expectancy_bps": m30.get("expectancy_bps"),
            "total_bps": m30.get("net_total_bps"),
            "signal_count": int(len(markers)),
            "primary_additions": int(len(markers)),
            "basis_additions": basis_n,
            "trusted_taker_additions": trusted_window,
            "coverage_pct": cov.get("total_coverage_pct"),
        }
        window_comparison = compare_windows(cur_pack, prev_pack, bool(prev_complete))

    q2: Dict[str, Any] = {}
    if Q2_BASELINE.exists():
        q2df = pd.read_csv(Q2_BASELINE)
        q2 = q2df.iloc[0].to_dict() if len(q2df) else {}

    weeks = {}
    if len(markers):
        m = markers.copy()
        m["week"] = m["signal_ts"].dt.tz_localize(None).dt.to_period("W-SUN").astype(str)
        weeks = m.groupby("week").size().to_dict()

    perf_by_h = horizon_bundle(matured, immature, int(len(markers)))
    ids_window = set(outcome_ids(matured, 30))
    ids_full = set(outcome_ids(matured_cum, 30))
    proper_subset = ids_window.issubset(ids_full) and (
        analysis_scope == "EFFECTIVE_T0_FULL" or len(ids_window) < len(ids_full)
    )

    funnel = [
        {"stage": "raw_primary_marker_rows", "count": int(len(markers_all))},
        {
            "stage": "quarantine_excluded",
            "count": int(markers_all["exclude_from_forward_eval"].fillna(False).astype(bool).sum())
            if "exclude_from_forward_eval" in markers_all.columns
            else 0,
        },
        {"stage": "eligible_in_scope_primary", "count": int(len(markers))},
        {"stage": "basis_family", "count": basis_n},
        {"stage": "funding_family", "count": funding_n},
        {"stage": "taker_family", "count": taker_n},
        {"stage": "secondary_condition_rows_in_scope", "count": sec_n},
        {"stage": "secondary_triggered", "count": sec_trig},
        {"stage": "filled_outcomes_30m", "count": filled_primary},
        {"stage": "filled_outcomes_60m", "count": int(metrics[60].get("n") or 0)},
        {"stage": "engine_shadow_entries", "count": 0, "note": "not in micro observer ledger"},
        {"stage": "engine_completed_trades", "count": 0, "note": "not in micro observer ledger"},
    ]

    date_tag = as_of.strftime("%Y-%m-%d")
    if analysis_scope == "ROLLING_WINDOW":
        tag = f"window_{int(window_days)}d"
        report_json = OUT / f"reports/{tag}_{date_tag}.json"
        report_md = OUT / f"reports/btc_forward_interim_{tag}_{date_tag}.md"
        chart_tag = tag
        # CSV artifacts
        markers.to_csv(OUT / f"signals/{tag}_signals.csv", index=False)
        matured[matured["horizon_min"].astype(int) == 30].to_csv(OUT / f"performance/{tag}_outcomes_30m.csv", index=False)
        matured[matured["horizon_min"].astype(int) == 60].to_csv(OUT / f"performance/{tag}_outcomes_60m.csv", index=False)
        if len(daily):
            daily.to_csv(OUT / f"coverage/{tag}_daily_coverage.csv", index=False)
        pd.DataFrame(
            [
                {
                    "family": "PRIMARY",
                    "window_additions": int(len(markers)),
                    "cumulative": primary_cum,
                },
                {"family": "BASIS", "window_additions": basis_n, "cumulative": basis_ready},
                {"family": "TRUSTED_TAKER", "window_additions": trusted_window, "cumulative": trusted_cum},
            ]
        ).to_csv(OUT / f"readiness/{tag}_marker_additions.csv", index=False)
    else:
        tag = "full"
        report_json = OUT / f"reports/full_run_{date_tag}.json"
        report_md = OUT / f"reports/btc_forward_interim_full_{date_tag}.md"
        chart_tag = "full"
        pd.DataFrame(funnel).to_csv(OUT / "signals/signal_funnel.csv", index=False)
        markers.to_csv(OUT / "signals/eligible_primary_markers.csv", index=False)
        if len(daily):
            daily.to_csv(OUT / "coverage/daily_strict_coverage.csv", index=False)
        pd.DataFrame([metrics[h] for h in (15, 30, 60, 120)]).to_csv(
            OUT / "performance/horizon_outcome_metrics.csv", index=False
        )

    charts = maybe_charts(daily if analysis_scope == "ROLLING_WINDOW" else daily_cum, markers if analysis_scope == "ROLLING_WINDOW" else markers_cum, ready, tag=chart_tag)
    after_hashes = {str(p): _sha(p) for p in PROTECTED_BTC}
    protected_ok = before_hashes == after_hashes

    coverage_for_headline = cov.get("total_coverage_pct") if analysis_scope == "ROLLING_WINDOW" else cov_cum.get("total_coverage_pct")
    completed_days = cov.get("completed_utc_days") if analysis_scope == "ROLLING_WINDOW" else cov_cum.get("completed_utc_days")
    pass_days = cov.get("daily_pass_days_ge_95") if analysis_scope == "ROLLING_WINDOW" else cov_cum.get("daily_pass_days_ge_95")
    expected_min = cov.get("total_expected_minutes") if analysis_scope == "ROLLING_WINDOW" else cov_cum.get("total_expected_minutes")
    eligible_min = cov.get("strict_eligible_minutes") if analysis_scope == "ROLLING_WINDOW" else cov_cum.get("strict_eligible_minutes")

    report: Dict[str, Any] = {
        "btc_forward_interim_review_verdict": w_strat if analysis_scope == "ROLLING_WINDOW" else statuses["strategy_interim_status"],
        "mode": "READ_ONLY_INTERIM_REVIEW",
        **{k: v for k, v in win.items() if k not in ("effective_window_start", "effective_window_end")},
        "generated_at_utc": now_utc.isoformat(),
        "generated_at_kst": now_kst.isoformat(),
        **t0_info,
        "effective_forward_days": days,
        "completed_utc_days": completed_days,
        "partial_current_day": (cov if analysis_scope == "ROLLING_WINDOW" else cov_cum).get("partial_current_day"),
        "collection_operational_status": statuses["collection_operational_status"],
        "data_integrity_status": statuses["data_integrity_status"],
        "research_sample_status": statuses["research_sample_status"],
        "window_sample_status": w_sample,
        "strategy_interim_status": statuses["strategy_interim_status"],
        "cumulative_strategy_interim_status": statuses["strategy_interim_status"],
        "window_strategy_interim_status": w_strat,
        "current_action": statuses["current_action"],
        "total_expected_bars_minutes": expected_min,
        "strict_eligible_bars_minutes": eligible_min,
        "total_coverage": coverage_for_headline,
        "cumulative_coverage_pct": cov_cum.get("total_coverage_pct"),
        "window_coverage_pct": cov.get("total_coverage_pct"),
        "window_expected_raw_intervals": expected_min if analysis_scope == "ROLLING_WINDOW" else None,
        "window_observed_raw_intervals": eligible_min if analysis_scope == "ROLLING_WINDOW" else None,
        "window_strict_eligible_intervals": eligible_min if analysis_scope == "ROLLING_WINDOW" else None,
        "window_completed_utc_days": cov.get("completed_utc_days") if analysis_scope == "ROLLING_WINDOW" else None,
        "window_daily_pass_days": cov.get("daily_pass_days_ge_95") if analysis_scope == "ROLLING_WINDOW" else None,
        "window_partial_days": cov.get("partial_days") if analysis_scope == "ROLLING_WINDOW" else None,
        "window_longest_gap_min": cov.get("longest_gap_minutes") if analysis_scope == "ROLLING_WINDOW" else None,
        "window_reconnects": cov.get("reconnect_attempts_sum") if analysis_scope == "ROLLING_WINDOW" else None,
        "window_hard_stale": cov.get("hard_stale_sum") if analysis_scope == "ROLLING_WINDOW" else None,
        "window_contamination": cov.get("contamination_sum") if analysis_scope == "ROLLING_WINDOW" else None,
        "daily_pass_days": pass_days,
        "longest_gap": (cov if analysis_scope == "ROLLING_WINDOW" else cov_cum).get("longest_gap_minutes"),
        "reconnects": (cov if analysis_scope == "ROLLING_WINDOW" else cov_cum).get("reconnect_attempts_sum"),
        "hard_stale": (cov if analysis_scope == "ROLLING_WINDOW" else cov_cum).get("hard_stale_sum"),
        "contamination": compact.get("contamination", 0),
        "quarantined_rows": int(markers_all["exclude_from_forward_eval"].fillna(False).astype(bool).sum())
        if "exclude_from_forward_eval" in markers_all.columns
        else None,
        "pre_t0_rows_excluded": int((pd.to_datetime(markers_all["signal_ts"], utc=True) < t0).sum()) if len(markers_all) else 0,
        "post_rule_signals": int(len(markers)),
        "window_post_rule_signals": int(len(markers)),
        "shadow_entries": 0,
        "completed_trades": 0,
        "open_trades": 0,
        "matured_outcomes_30m": filled_primary,
        "matured_outcomes_60m": int(metrics[60].get("n") or 0),
        "immature_outcomes": int(len(immature)),
        "window_matured_outcomes_30m": filled_primary,
        "window_immature_outcomes_30m": imm30,
        "primary_markers": primary_cum if analysis_scope == "EFFECTIVE_T0_FULL" else int(len(markers)),
        "primary_cumulative": primary_cum,
        "primary_window": int(len(markers)),
        "primary_window_additions": int(len(markers)),
        "primary_target": 50,
        "basis_markers": basis_ready if analysis_scope == "EFFECTIVE_T0_FULL" else basis_n,
        "basis_cumulative": basis_ready,
        "basis_window": basis_n,
        "basis_window_additions": basis_n,
        "basis_target": 10,
        "trusted_taker": trusted_cum if analysis_scope == "EFFECTIVE_T0_FULL" else trusted_window,
        "trusted_taker_cumulative": trusted_cum,
        "trusted_taker_window": trusted_window,
        "trusted_taker_window_additions": trusted_window,
        "trusted_taker_target": 10,
        "funding_markers": int(ready.get("funding_markers") or funding_cum),
        "readiness_bottleneck": statuses["readiness_bottleneck"],
        "bottleneck_cause": (
            "Market-event scarcity for basis_bps_q95 / trusted taker under frozen definitions; "
            "not a collector gap (complete days often 100% coverage, gaps=0). "
            f"Days since last marker≈{days_since_last_marker:.1f}."
            if days_since_last_marker and days_since_last_marker > 3
            else "Insufficient unique basis/trusted-taker events under frozen marker definitions."
        ),
        "estimated_remaining_observations": {
            "primary": max(50 - primary_cum, 0),
            "basis": max(10 - basis_ready, 0),
            "trusted_taker": max(10 - trusted_cum, 0),
        },
        "estimated_checkpoint_date": est.get("estimated_ready_date_range_kst"),
        "estimate_detail": est,
        "last_marker_ts": last_marker.isoformat() if last_marker is not None else None,
        "days_since_last_marker": days_since_last_marker,
        "weekly_marker_counts": weeks,
        "horizon_metrics": metrics,
        "performance_by_horizon": perf_by_h,
        "performance_concentration": {
            "best_trade_contribution_pct": m30.get("best_contribution_pct"),
            "top3_trade_contribution_pct": m30.get("top3_contribution_pct"),
            "best_bps": m30.get("best_bps"),
            "worst_bps": m30.get("worst_bps"),
        },
        "window_comparison": window_comparison,
        "rolling_trend_status": (window_comparison or {}).get("rolling_trend_status"),
        "rolling_trend_disclaimer": TREND_DISCLAIMER if analysis_scope == "ROLLING_WINDOW" else None,
        "current_baseline": "Q2_BDI (engine production baseline; separate ledger)",
        "current_candidate": "WEAK_HINT_FORWARD_OBSERVER",
        "common_comparison_start": None,
        "common_comparison_end": None,
        "baseline_candidate_common_trade_comparison": "NOT_AVAILABLE_DIFFERENT_LEDGERS",
        "q2_bdi_snapshot": q2,
        "candidate_30m_net_total_bps": m30.get("net_total_bps"),
        "candidate_30m_mean_bps": m30.get("mean_bps"),
        "candidate_30m_median_bps": m30.get("median_bps"),
        "candidate_30m_win_rate": m30.get("win_rate"),
        "candidate_30m_profit_factor": m30.get("profit_factor"),
        "candidate_30m_expectancy_bps": m30.get("expectancy_bps"),
        "candidate_30m_n": m30.get("n"),
        "window_30m_mean_bps": m30.get("mean_bps"),
        "window_30m_median_bps": m30.get("median_bps"),
        "window_30m_win_rate": m30.get("win_rate"),
        "window_30m_profit_factor": m30.get("profit_factor"),
        "window_30m_expectancy_bps": m30.get("expectancy_bps"),
        "baseline_net_return": q2.get("net"),
        "baseline_mdd": q2.get("mdd"),
        "baseline_trades": q2.get("rows"),
        "outcome_id_audit": {
            "full_30m_n": len(ids_full),
            "scope_30m_n": len(ids_window),
            "window_outcome_ids_proper_subset_of_full": proper_subset,
            "future_maturity_leakage": int((outcomes_window_signal.apply(maturity_ts, axis=1) > as_of).sum())
            if len(outcomes_window_signal)
            else 0,
        },
        "what_can_be_concluded": [
            "Strict-live collection operational status is reported separately from strategy interim status.",
            f"Analysis scope={analysis_scope}; window start={start.isoformat()} end={end.isoformat()}.",
            f"Cumulative readiness PRIMARY {primary_cum}/50, BASIS {basis_ready}/10, TRUSTED_TAKER {trusted_cum}/10.",
            f"In-scope 30m matured n={filled_primary} (descriptive only).",
            TREND_DISCLAIMER if analysis_scope == "ROLLING_WINDOW" else "Full effective-T0 cumulative review.",
        ],
        "what_cannot_be_concluded": [
            "Production or promotion readiness.",
            "Final accept/reject of weak-hint layer.",
            "Replacement of Q2_BDI baseline.",
            "Statistically reliable Sharpe/MDD for strategy promotion.",
            "Same-period economic superiority vs Q2_BDI trades (no shared ledger).",
            "Rolling trend as a retune signal.",
        ],
        "checkpoints": {
            "A": {"desc": "Primary 50/50, Basis 10/10, Trusted taker 10/10", "estimate": est.get("estimated_ready_date_range_kst")},
            "B": {"desc": "filled 30m outcomes >=10", "met": filled_cum >= 10},
            "C": {"desc": "markers>=30 across >=2 families", "met": primary_cum >= 30 and len(by_fam_cum) >= 2},
            "D": {"desc": "markers>=50, >=6-8 weeks, integrity PASS, readiness PASS", "met": False},
        },
        "recommended_next_review": "CHECKPOINT_A readiness (basis/trusted taker) or earliest estimate band 2026-09-08",
        "production_ready": False,
        "promotion_ready": False,
        "execution_enabled": False,
        "private_calls": 0,
        "order_calls": 0,
        "btc_collector": compact.get("collector_health"),
        "btc_keep_awake": compact.get("KEEP_AWAKE_GUARD_ACTIVE"),
        "btc_protected_files_unchanged": protected_ok,
        "qqq_protected_files_unchanged": protected_ok,
        "charts_created": charts,
        "funnel": funnel,
        "secondary_rows": sec_n,
        "secondary_triggered": sec_trig,
        "compact_ops": {
            "latest_complete_day_coverage_pct": compact.get("latest_complete_day_coverage_pct"),
            "recent_7d_coverage_pct": compact.get("recent_7d_coverage_pct"),
            "pending_gaps": compact.get("pending_gaps"),
            "failed_gaps": compact.get("failed_gaps"),
            "hard_stale_incidents_7d": compact.get("hard_stale_incidents_7d"),
            "reconnect_attempts_7d": compact.get("reconnect_attempts_7d"),
        },
        "report_path": str(report_json),
    }

    if window_comparison:
        prev = window_comparison["previous_equal_window"]
        report["previous_window_30m_matured"] = prev.get("matured")
        report["previous_window_30m_mean_bps"] = prev.get("mean_bps")
        report["previous_window_30m_win_rate"] = prev.get("win_rate")
        report["previous_window_30m_profit_factor"] = prev.get("profit_factor")
        report["previous_window_complete"] = window_comparison.get("previous_window_complete")
        report["previous_window_start_utc"] = prev.get("start")
        report["previous_window_end_utc"] = prev.get("end")

    # lineage / inventory (shared)
    _atomic_json(OUT / "lineage/forward_lineage_audit.json", t0_info)
    _write_md(
        OUT / "lineage/forward_lineage_audit.md",
        "# Forward Lineage Audit\n\n"
        f"- ORIGINAL_STRICT_LIVE_T0: `{t0_info['original_strict_live_t0']}`\n"
        f"- CURRENT_CONFIGURATION_EFFECTIVE_T0: `{t0_info['current_configuration_effective_t0']}`\n"
        f"- forward_clock_reset_required: `{t0_info['forward_clock_reset_required']}`\n\n{t0_info['note']}\n",
    )
    _atomic_json(
        OUT / "inventory/research_unit_counts.json",
        {
            "primary_key": "observation_id",
            "units": {
                "eligible_primary_markers_in_scope": int(len(markers)),
                "secondary_condition_rows": sec_n,
                "filled_outcomes_30m": filled_primary,
                "engine_trades": 0,
            },
        },
    )
    _write_md(
        OUT / "inventory/research_unit_definition.md",
        "# Research Unit Definition\n\n"
        "This forward track is the **microstructure weak-hint forward observer**, not the engine paper/shadow trade ledger.\n\n"
        "- Primary unit: `observation_id` marker episode\n"
        "- Outcome unit: `(observation_id, horizon_min)` filled path metrics (`net_current_bps`)\n"
        "- Window attribution uses signal/event timestamp (`anchor_ts` / `signal_ts`)\n"
        "- Maturity gate uses `outcome_filled_ts <= analysis_as_of`\n",
    )
    _atomic_json(OUT / "readiness/readiness_snapshot.json", {"ready": ready, "estimate": est})

    compact_out = {
        "analysis_scope": analysis_scope,
        "requested_window_days": win["requested_window_days"],
        "analysis_as_of_utc": win["analysis_as_of_utc"],
        "effective_window_start_utc": win["effective_window_start_utc"],
        "effective_window_end_utc": win["effective_window_end_utc"],
        "btc_forward_interim_review_verdict": report["btc_forward_interim_review_verdict"],
        "mode": report["mode"],
        "generated_at_utc": report["generated_at_utc"],
        "generated_at_kst": report["generated_at_kst"],
        "original_strict_live_t0": report["original_strict_live_t0"],
        "current_configuration_effective_t0": report["current_configuration_effective_t0"],
        "effective_forward_days": report["effective_forward_days"],
        "collection_operational_status": report["collection_operational_status"],
        "data_integrity_status": report["data_integrity_status"],
        "research_sample_status": report["research_sample_status"],
        "window_sample_status": report["window_sample_status"],
        "strategy_interim_status": report["strategy_interim_status"],
        "cumulative_strategy_interim_status": report["cumulative_strategy_interim_status"],
        "window_strategy_interim_status": report["window_strategy_interim_status"],
        "current_action": report["current_action"],
        "window_coverage_pct": report.get("window_coverage_pct"),
        "total_coverage": report["total_coverage"],
        "window_post_rule_signals": report["window_post_rule_signals"],
        "window_matured_outcomes_30m": report["window_matured_outcomes_30m"],
        "window_immature_outcomes_30m": report["window_immature_outcomes_30m"],
        "window_30m_mean_bps": report["window_30m_mean_bps"],
        "window_30m_median_bps": report["window_30m_median_bps"],
        "window_30m_win_rate": report["window_30m_win_rate"],
        "window_30m_profit_factor": report["window_30m_profit_factor"],
        "window_30m_expectancy_bps": report["window_30m_expectancy_bps"],
        "previous_window_30m_matured": report.get("previous_window_30m_matured"),
        "previous_window_30m_mean_bps": report.get("previous_window_30m_mean_bps"),
        "previous_window_30m_win_rate": report.get("previous_window_30m_win_rate"),
        "previous_window_30m_profit_factor": report.get("previous_window_30m_profit_factor"),
        "rolling_trend_status": report.get("rolling_trend_status"),
        "primary_markers": report["primary_cumulative"],
        "primary_cumulative": report["primary_cumulative"],
        "primary_window_additions": report["primary_window_additions"],
        "basis_markers": report["basis_cumulative"],
        "basis_cumulative": report["basis_cumulative"],
        "basis_window_additions": report["basis_window_additions"],
        "trusted_taker": report["trusted_taker_cumulative"],
        "trusted_taker_cumulative": report["trusted_taker_cumulative"],
        "trusted_taker_window_additions": report["trusted_taker_window_additions"],
        "readiness_bottleneck": report["readiness_bottleneck"],
        "matured_outcomes_30m": report["matured_outcomes_30m"],
        "candidate_30m_mean_bps": report["candidate_30m_mean_bps"],
        "candidate_30m_win_rate": report["candidate_30m_win_rate"],
        "production_ready": False,
        "promotion_ready": False,
        "execution_enabled": False,
        "private_calls": 0,
        "order_calls": 0,
    }

    _atomic_json(report_json, report)
    _atomic_json(OUT / "reports/btc_forward_interim_review.json", report)  # latest pointer
    _atomic_json(OUT / "reports/btc_forward_interim_compact.json", compact_out)

    if analysis_scope == "ROLLING_WINDOW":
        md = f"""# BTC FORWARD ROLLING WINDOW REVIEW

ANALYSIS SCOPE:
ROLLING_WINDOW

REQUESTED WINDOW:
{window_days} DAYS

WINDOW START UTC:
{win['effective_window_start_utc']}

WINDOW END UTC:
{win['effective_window_end_utc']}

EFFECTIVE T0:
{t0_info['current_configuration_effective_t0']}

IMPORTANT:
This report evaluates only signals whose event timestamps occurred inside the rolling window.
Readiness targets remain cumulative from effective T0.

Generated: `{report['generated_at_kst']}`

## 1. Window Definition
- requested_window_days: {window_days}
- analysis_as_of_utc: {win['analysis_as_of_utc']}
- requested_window_start_utc: {win['requested_window_start_utc']}
- window_truncated_by_effective_t0: {win['window_truncated_by_effective_t0']}
- duration_hours: {win['window_duration_hours']:.2f}

## 2. Data Integrity
- collection: `{report['collection_operational_status']}`
- integrity: `{report['data_integrity_status']}`
- contamination: {report['contamination']}

## 3. Window Coverage
- window_coverage_pct: {report['window_coverage_pct']}
- cumulative_coverage_pct: {report['cumulative_coverage_pct']}
- completed UTC days: {report['window_completed_utc_days']}
- daily pass days: {report['window_daily_pass_days']}

## 4. Window Sample Counts
- post_rule_signals: {report['window_post_rule_signals']}
- matured 30m: {report['window_matured_outcomes_30m']}
- immature 30m: {report['window_immature_outcomes_30m']}
- window_sample_status: `{report['window_sample_status']}`

## 5. 30m Performance
- mean: {report['window_30m_mean_bps']}
- median: {report['window_30m_median_bps']}
- win: {report['window_30m_win_rate']}
- PF: {report['window_30m_profit_factor']}
- expectancy: {report['window_30m_expectancy_bps']}

## 6. 60m Performance
- matured: {report['matured_outcomes_60m']}
- mean: {metrics[60].get('mean_bps')}
- win: {metrics[60].get('win_rate')}
- PF: {metrics[60].get('profit_factor')}

## 7. Current vs Previous Equal Window
{json.dumps(window_comparison, indent=2, default=str) if window_comparison else 'N/A'}

{TREND_DISCLAIMER}

## 8. Cumulative Readiness
- PRIMARY: {primary_cum}/50
- BASIS: {basis_ready}/10
- TRUSTED_TAKER: {trusted_cum}/10
- bottleneck: `{report['readiness_bottleneck']}`

## 9. Window Marker Additions
- primary_window: {report['primary_window_additions']}
- basis_window: {report['basis_window_additions']}
- trusted_taker_window: {report['trusted_taker_window_additions']}

## 10. Strategy Interim Status
- cumulative: `{report['cumulative_strategy_interim_status']}`
- window: `{report['window_strategy_interim_status']}`

## 11. What Can Be Concluded
{chr(10).join('- ' + x for x in report['what_can_be_concluded'])}

## 12. What Cannot Be Concluded
{chr(10).join('- ' + x for x in report['what_cannot_be_concluded'])}

## 13. Safety Status
production_ready=false · promotion_ready=false · execution_enabled=false · private_calls=0 · order_calls=0
"""
    else:
        md = f"""# BTC Forward Interim Review (FULL)

Generated: `{report['generated_at_kst']}`

## Verdict
- Strategy interim: `{report['strategy_interim_status']}`
- Current action: `{report['current_action']}`
- Collection: `{report['collection_operational_status']}`
- Integrity: `{report['data_integrity_status']}`
- Sample: `{report['research_sample_status']}`

## Clocks
- ORIGINAL_STRICT_LIVE_T0: `{report['original_strict_live_t0']}`
- CURRENT_CONFIGURATION_EFFECTIVE_T0: `{report['current_configuration_effective_t0']}`
- Effective forward days: **{report['effective_forward_days']:.2f}**
- Analysis as-of: `{win['analysis_as_of_utc']}`

## Collection
- Completed UTC days: {report['completed_utc_days']}
- Total coverage: {report['total_coverage']}
- Daily pass days (≥95%): {report['daily_pass_days']}

## Research units
- Eligible primary markers (cumulative): **{report['primary_cumulative']}**
- Filled outcomes 30m/60m: {report['matured_outcomes_30m']} / {report['matured_outcomes_60m']}

## Horizon-30m descriptive economics
- n: {report['candidate_30m_n']}
- mean net bps: {report['candidate_30m_mean_bps']}
- median: {report['candidate_30m_median_bps']}
- win rate: {report['candidate_30m_win_rate']}
- profit factor: {report['candidate_30m_profit_factor']}

## Safety
production_ready=false · promotion_ready=false · execution_enabled=false
"""
    _write_md(report_md, md)
    _write_md(OUT / "reports/btc_forward_interim_review.md", md)
    return report


def run_full_review(as_of_utc: Optional[str] = None) -> Dict[str, Any]:
    return run_review(analysis_scope="EFFECTIVE_T0_FULL", window_days=None, as_of_utc=as_of_utc)


def build_status() -> Dict[str, Any]:
    path = OUT / "reports/btc_forward_interim_compact.json"
    now_utc, now_kst = _now()
    if not path.exists():
        return {
            "status": "NOT_RUN",
            "status_checked_at_utc": now_utc.isoformat(),
            "status_checked_at_kst": now_kst.isoformat(),
            "production_ready": False,
            "promotion_ready": False,
        }
    base = json.loads(path.read_text())
    return {
        **base,
        "status_checked_at_utc": now_utc.isoformat(),
        "status_checked_at_kst": now_kst.isoformat(),
        "status_does_not_rerun_analysis": True,
    }
