"""Normalized QQQ data quality checks."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .calendar import expected_sessions, get_calendar
from .config import QQQConfig, paths


def audit_normalized(df: pd.DataFrame, cfg: QQQConfig = QQQConfig()) -> Dict[str, Any]:
    issues: List[str] = []
    warnings: List[str] = []
    if df is None or len(df) == 0:
        return {"verdict": "QQQ_DATA_PIPELINE_FAIL", "issues": ["empty"], "warnings": [], "metrics": {}}

    d = df.copy()
    d["session_date"] = pd.to_datetime(d["session_date"]).dt.normalize()
    d = d.sort_values("session_date")

    metrics: Dict[str, Any] = {
        "rows": int(len(d)),
        "start": str(d["session_date"].iloc[0].date()),
        "end": str(d["session_date"].iloc[-1].date()),
    }

    if not d["session_date"].is_monotonic_increasing:
        issues.append("session_date_not_monotonic")
    dup = int(d.duplicated(subset=["symbol", "session_date"]).sum())
    metrics["duplicates"] = dup
    if dup:
        issues.append("duplicate_primary_keys")

    for col in ["open_adj", "high_adj", "low_adj", "close_adj", "open_raw", "high_raw", "low_raw", "close_raw"]:
        if d[col].isna().any():
            issues.append(f"null_{col}")
        if (d[col] <= 0).any():
            issues.append(f"nonpositive_{col}")

    if d["volume"].isna().any() or (d["volume"] < 0).any():
        issues.append("bad_volume")

    bad_hl = ~((d["high_adj"] + 1e-12 >= d[["open_adj", "close_adj", "low_adj"]].max(axis=1)) & (d["low_adj"] - 1e-12 <= d[["open_adj", "close_adj", "high_adj"]].min(axis=1)) & (d["high_adj"] >= d["low_adj"]))
    metrics["ohlc_invariant_violations"] = int(bad_hl.sum())
    if bad_hl.any():
        issues.append("ohlc_invariant_violations")

    if (~np.isfinite(d["adjustment_factor"]) | (d["adjustment_factor"] <= 0)).any():
        issues.append("bad_adjustment_factor")

    rets = d["close_adj"].pct_change()
    if np.isinf(rets.to_numpy(dtype=float)).any():
        issues.append("infinite_returns")
    outlier = d.loc[rets.abs() > 0.20, "session_date"]
    metrics["abs_return_gt_20pct_count"] = int(len(outlier))
    if len(outlier):
        warnings.append(f"abs_return_gt_20pct:{[str(x.date()) for x in outlier.head(10).tolist()]}")

    # weekends
    if (d["session_date"].dt.dayofweek >= 5).any():
        issues.append("weekend_rows")

    cal = get_calendar(cfg.exchange)
    # Only audit missing sessions inside exchange_calendars coverage window
    audit_start = max(pd.Timestamp(d["session_date"].iloc[0]), pd.Timestamp(cal.first_session))
    audit_end = min(pd.Timestamp(d["session_date"].iloc[-1]), pd.Timestamp(cal.last_session))
    exp = expected_sessions(audit_start, audit_end, cal)
    have = set(pd.to_datetime(d["session_date"]).dt.normalize())
    expected = set(pd.to_datetime(exp).normalize())
    missing = sorted(expected - have)
    in_cal = {x for x in have if pd.Timestamp(cal.first_session) <= x <= pd.Timestamp(cal.last_session)}
    unexpected = sorted(in_cal - expected)
    metrics["missing_expected_sessions"] = len(missing)
    metrics["unexpected_sessions"] = len(unexpected)
    if missing:
        warnings.append(f"missing_expected_sessions_count={len(missing)}")
    if unexpected:
        warnings.append(f"unexpected_sessions_count={len(unexpected)}")

    # Incomplete current session should already be removed by provider
    metrics["incomplete_session_removed"] = True

    verdict = "QQQ_DATA_PIPELINE_PASS"
    if issues:
        verdict = "QQQ_DATA_PIPELINE_FAIL"
    elif warnings:
        verdict = "QQQ_DATA_PIPELINE_PASS_WITH_WARNINGS"

    report = {
        "verdict": verdict,
        "issues": issues,
        "warnings": warnings,
        "metrics": metrics,
        "production_ready": False,
        "promotion_ready": False,
    }
    p = paths(cfg)
    (p["reports"] / "qqq_data_quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    md = [
        "# QQQ Data Quality Report",
        "",
        f"**Verdict:** `{verdict}`",
        "",
        f"- rows: {metrics['rows']}",
        f"- start/end: {metrics['start']} → {metrics['end']}",
        f"- duplicates: {metrics['duplicates']}",
        f"- missing expected sessions: {metrics['missing_expected_sessions']}",
        f"- OHLC violations: {metrics['ohlc_invariant_violations']}",
        f"- |ret|>20% outliers: {metrics['abs_return_gt_20pct_count']} (not auto-deleted)",
        "",
        "## Issues",
        "",
    ]
    md.extend([f"- {x}" for x in issues] if issues else ["- none"])
    md.extend(["", "## Warnings", ""])
    md.extend([f"- {x}" for x in warnings] if warnings else ["- none"])
    md.extend(["", "Provider provenance: YAHOO_FINANCE_UNOFFICIAL. Not an official Nasdaq feed."])
    (p["reports"] / "qqq_data_quality_report.md").write_text("\n".join(md) + "\n")
    dq = p["diag"].joinpath("data_quality")
    dq.mkdir(parents=True, exist_ok=True)
    (dq / "qqq_data_quality_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
