"""
Meta dataset audit helpers (diagnostics only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    HISTORY_DIR,
    LABEL_COLS,
    MANIFEST_PATH,
    MASTER_PATH,
    OUT_DIR,
    _dataset_statistics,
    _dedupe,
    _regime_summary,
    load_v2_dataset,
)

MILESTONES = [200, 300, 500]


def audit_dataset(dataset: pd.DataFrame, ds_path: Path) -> Dict[str, Any]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8")) if MANIFEST_PATH.exists() else {}
    stats = _dataset_statistics(dataset)

    dup_ids = int(dataset["trade_id"].duplicated().sum()) if "trade_id" in dataset.columns else 0
    deduped = _dedupe(dataset.copy())
    dup_after = int(deduped["trade_id"].duplicated().sum()) if "trade_id" in deduped.columns else 0

    expected = set(ENTRY_FEATURE_COLS_V2 + LABEL_COLS + [
        "trade_id", "df_idx", "direction", "net_return", "vol_bucket", "trend_state",
        "replay_timestamp", "meta_v2_version",
    ])
    missing = sorted(expected - set(dataset.columns))

    missing_features = [
        c for c in ENTRY_FEATURE_COLS_V2
        if c not in dataset.columns or float(dataset[c].isna().mean()) > 0.05
    ]

    long_ratio = float((dataset["direction"] == "LONG").mean()) if "direction" in dataset.columns else 0.0
    high_vol_ratio = float((dataset["vol_bucket"] == "high").mean()) if "vol_bucket" in dataset.columns else 0.0

    n = len(dataset)
    next_milestone = next((m for m in MILESTONES if n < m), MILESTONES[-1])

    audit_pass = (
        dup_after == 0
        and not missing
        and not missing_features
    )

    return {
        "total_rows": n,
        "unique_trade_ids": int(dataset["trade_id"].nunique()) if "trade_id" in dataset.columns else 0,
        "duplicate_trade_ids": dup_ids,
        "duplicate_after_dedupe": dup_after,
        "schema_drift": bool(missing),
        "missing_columns": missing,
        "missing_features": missing_features,
        "append_only_integrity": dup_after == 0,
        "long_short_ratio": stats.get("long_short_ratio", {}),
        "long_ratio": long_ratio,
        "high_vol_ratio": high_vol_ratio,
        "label_distribution": stats.get("label_distribution", {}),
        "vol_distribution": stats.get("vol_distribution", {}),
        "regime_distribution": stats.get("regime_distribution", {}),
        "history_snapshots": len(list(HISTORY_DIR.glob("meta_dataset_v2_*.parquet"))) if HISTORY_DIR.exists() else 0,
        "manifest_exists": MANIFEST_PATH.exists(),
        "master_path": str(ds_path),
        "milestone_current": n,
        "milestone_next": next_milestone,
        "milestone_progress_300": f"{n}/300",
        "audit_pass": audit_pass,
        "audit_status": "PASS" if audit_pass else "FAIL",
        "last_replay_timestamp": manifest.get("last_replay_timestamp"),
    }


def append_growth_history(
    row: Dict[str, Any],
    path: Path = OUT_DIR / "meta_dataset_growth_history.csv",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if path.exists():
        hist = pd.read_csv(path)
        hist = pd.concat([hist, df], ignore_index=True)
    else:
        hist = df
    hist.to_csv(path, index=False)


def write_integrity_report(audit: Dict[str, Any], path: Path = OUT_DIR / "meta_dataset_integrity_report.md") -> None:
    path.write_text(
        "# Meta Dataset Integrity Report\n\n"
        f"**Audit:** {audit.get('audit_status', 'UNKNOWN')}\n\n"
        f"```json\n{json.dumps(audit, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )


def write_daily_audit_md(audit: Dict[str, Any], ts: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Meta Dataset Daily Audit — {ts}\n\n"
        f"- total_rows: {audit.get('total_rows')}\n"
        f"- duplicate_trade_ids: {audit.get('duplicate_trade_ids')}\n"
        f"- schema_drift: {audit.get('schema_drift')}\n"
        f"- missing_features: {audit.get('missing_features')}\n"
        f"- long_ratio: {audit.get('long_ratio', 0):.1%}\n"
        f"- high_vol_ratio: {audit.get('high_vol_ratio', 0):.1%}\n"
        f"- milestone: {audit.get('milestone_progress_300')}\n"
        f"- audit: **{audit.get('audit_status')}**\n\n"
        f"## Label distribution\n```json\n{json.dumps(audit.get('label_distribution', {}), indent=2)}\n```\n",
        encoding="utf-8",
    )
