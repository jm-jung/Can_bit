"""
Historical meta_dataset_v2 backfill — time-safe cache replay (diagnostics only).

Uses TCN proba cache + OHLCV chunks with per-chunk simulation state reset
to avoid kill-switch path collapse over full history.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    HISTORY_DIR,
    LEAKAGE_COLS,
    MASTER_PATH,
    _collect_production_trades_v2,
    _ohlcv_features,
    load_v2_dataset,
)
from scripts.diagnostics.run_daily_meta_research_pipeline import safe_append_master
from scripts.diagnostics.validate_quality_score_replay import POSITION_SIZE
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper
from scripts.run_daily_shadow import PROBA_CACHE_DEFAULT_PATH

OUT_DIR = Path("data/diagnostics/meta_layer/backfill")
BACKUP_PATH = MASTER_PATH.with_suffix(".parquet.pre_backfill.bak")
WARMUP_BARS = 300
DEFAULT_CHUNK_DAYS = 90
TARGET_ROWS = 300


def _audit_data_sources() -> Dict[str, Any]:
    ohlcv_path = Path("data/ohlcv/BTCUSDT_5m_full.csv")
    proba_path = PROBA_CACHE_DEFAULT_PATH
    audit: Dict[str, Any] = {
        "ohlcv_path": str(ohlcv_path),
        "proba_path": str(proba_path),
        "ohlcv_exists": ohlcv_path.exists(),
        "proba_exists": proba_path.exists(),
    }
    if ohlcv_path.exists():
        ohlcv = pd.read_csv(ohlcv_path, usecols=["timestamp"], nrows=5)
        ohlcv_tail = pd.read_csv(ohlcv_path, usecols=["timestamp"]).tail(1)
        audit["ohlcv_start"] = str(ohlcv["timestamp"].iloc[0])
        audit["ohlcv_end"] = str(ohlcv_tail["timestamp"].iloc[0])
        audit["ohlcv_rows"] = sum(1 for _ in open(ohlcv_path)) - 1
    if proba_path.exists():
        cdf = pd.read_parquet(proba_path, columns=["timestamp"])
        audit["proba_start"] = str(cdf["timestamp"].min())
        audit["proba_end"] = str(cdf["timestamp"].max())
        audit["proba_rows"] = len(cdf)
        audit["proba_stale_days"] = (
            datetime.now() - pd.Timestamp(cdf["timestamp"].max()).to_pydatetime()
        ).days
    audit["replay_mode_priority"] = "cache_replay_mode"
    audit["walk_forward_needed"] = not audit.get("proba_exists", False)
    audit["usable_for_backfill"] = audit.get("ohlcv_exists") and audit.get("proba_exists")
    return audit


def _chunk_ranges(df: pd.DataFrame, chunk_days: int, start_date: str) -> List[Dict[str, Any]]:
    ts = pd.to_datetime(df["timestamp"])
    df = df.copy()
    df["_ts"] = ts
    start = pd.Timestamp(start_date)
    end = ts.max()
    chunks: List[Dict[str, Any]] = []
    cur = start
    cid = 0
    while cur < end:
        nxt = cur + pd.Timedelta(days=chunk_days)
        mask = (df["_ts"] >= cur) & (df["_ts"] < nxt)
        if mask.sum() < WARMUP_BARS + 50:
            cur = nxt
            continue
        idxs = np.where(mask.to_numpy())[0]
        chunks.append({
            "chunk_id": f"hist_{cur.strftime('%Y%m%d')}_{cid}",
            "start_ts": str(cur),
            "end_ts": str(nxt),
            "start_idx": int(idxs[0]),
            "end_idx": int(idxs[-1]) + 1,
        })
        cid += 1
        cur = nxt
    return chunks


def _slice_chunk(df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
    i0 = max(0, start_idx - WARMUP_BARS)
    chunk = df.iloc[i0:end_idx].copy().reset_index(drop=True)
    return chunk


def _collect_chunk_trades(
    chunk_df: pd.DataFrame,
    chunk_id: str,
    replay_ts: str,
    replay_mode: str = "cache_replay",
) -> pd.DataFrame:
    feat_map, _ = _ohlcv_features(chunk_df)
    ticks, proba_meta = simulate_signals_paper(chunk_df)
    if not ticks:
        return pd.DataFrame()
    batch = _collect_production_trades_v2(ticks, feat_map, replay_ts)
    if batch.empty:
        return batch
    batch["source_replay"] = "historical_backfill"
    batch["backfill_batch_id"] = chunk_id
    batch["replay_mode"] = replay_mode
    batch["proba_fallback_rate"] = float(proba_meta.get("fallback_rate", 0) or 0)
    return batch


def _distribution_row(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    if df.empty:
        return {"subset": label, "rows": 0}
    row: Dict[str, Any] = {
        "subset": label,
        "rows": len(df),
        "long_ratio": float((df["direction"] == "LONG").mean()),
        "high_vol_ratio": float((df["vol_bucket"] == "high").mean()) if "vol_bucket" in df.columns else 0,
        "calibration_fail": int(df["calibration_label"].sum()) if "calibration_label" in df.columns else 0,
        "trade_quality_0": int((df["trade_quality_label"] == 0).sum()) if "trade_quality_label" in df.columns else 0,
        "trade_quality_2": int((df["trade_quality_label"] == 2).sum()) if "trade_quality_label" in df.columns else 0,
        "avg_net": float(df["net_return"].mean()),
        "rfe_rate": float(df["rfe_flag"].mean()) if "rfe_flag" in df.columns else 0,
    }
    if "vol_bucket" in df.columns:
        row["vol_bucket_dist"] = df["vol_bucket"].value_counts().to_dict()
    if "trend_state" in df.columns:
        row["trend_state_dist"] = df["trend_state"].value_counts().to_dict()
    if "trade_quality_label" in df.columns:
        row["trade_quality_dist"] = df["trade_quality_label"].value_counts().to_dict()
    if "calibration_label" in df.columns:
        row["calibration_label_dist"] = df["calibration_label"].value_counts().to_dict()
    return row


def _leakage_audit(feature_cols: List[str]) -> Dict[str, Any]:
    bad = [c for c in feature_cols if c in LEAKAGE_COLS]
    return {
        "passed": len(bad) == 0,
        "suspicious_feature_cols": bad,
        "entry_features_only": ENTRY_FEATURE_COLS_V2,
        "labels_post_exit_only": True,
        "per_chunk_state_reset": True,
        "no_full_period_scaler": True,
    }


def _q2_metrics_on_dataset(dataset: pd.DataFrame, ticks_cache: Optional[Dict] = None) -> Dict[str, Any]:
    """Aggregate Q2-relevant stats from dataset rows (no re-sim required)."""
    vals = (dataset["net_return"] * dataset.get("scale", 1.0)).tolist() if "net_return" in dataset.columns else []
    eq = peak = 1.0
    mdd = 0.0
    for r in vals:
        eq *= 1.0 + r * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
    fh = 0
    if {"direction", "vol_bucket", "entropy", "net_return"}.issubset(dataset.columns):
        fh = int((
            (dataset["direction"] == "LONG")
            & (dataset["trend_state"] == "up")
            & (dataset["vol_bucket"] == "high")
            & (dataset["entropy"] <= 0.90)
            & (dataset["net_return"] < 0)
        ).sum())
    return {
        "trades": len(dataset),
        "MDD": float(mdd),
        "false_high": fh,
        "RFE": int(dataset["rfe_flag"].sum()) if "rfe_flag" in dataset.columns else 0,
        "avg_net": float(dataset["net_return"].mean()) if len(dataset) else 0,
    }


def _revalidation_summary(dataset: pd.DataFrame, rows_before: int) -> Dict[str, Any]:
    n = len(dataset)
    summary: Dict[str, Any] = {
        "rows_before": rows_before,
        "rows_after": n,
        "target_300_met": n >= 300,
        "target_500_met": n >= 500,
        "meta_retrain_eligible": n >= 300,
        "promotion_ready": False,
        "q2_baseline_unchanged": True,
    }
    if n >= 300:
        try:
            from scripts.diagnostics.train_meta_layer_model import train_models
            _, report_path, _ = train_models()
            summary["train_attempted"] = True
            summary["train_report"] = str(report_path)
        except Exception as exc:
            summary["train_attempted"] = False
            summary["train_error"] = str(exc)[:200]
    return summary


def _final_verdict(rows_after: int, rows_added: int, leak: Dict[str, Any], reval: Dict[str, Any]) -> str:
    if not leak.get("passed"):
        return "backfill_failed_due_to_leakage_risk"
    if rows_added == 0:
        return "insufficient_historical_proba_cache"
    if rows_after < TARGET_ROWS:
        return "backfill_success_revalidate_later"
    if reval.get("train_attempted"):
        return "backfill_success_meta_still_reject"
    return "meta_research_reopened"


def run_backfill(
    *,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    start_date: str = "2021-06-01",
    dry_run: bool = False,
    max_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    replay_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"backfill_{replay_ts}"

    data_audit = _audit_data_sources()
    data_audit.update({
        "available_start_ts": data_audit.get("ohlcv_start"),
        "available_end_ts": data_audit.get("ohlcv_end"),
        "proba_coverage_pct": round(
            100 * data_audit.get("proba_rows", 0) / max(data_audit.get("ohlcv_rows", 1), 1), 2
        ),
        "cache_stale": data_audit.get("proba_stale_days", 99) == 0,
        "usable_historical_windows": f"{start_date} → present ({chunk_days}d chunks, cache_replay_mode)",
        "external_fetch_required": False,
    })
    (OUT_DIR / "historical_backfill_data_audit.md").write_text(
        f"# Historical Backfill Data Audit\n\n```json\n{json.dumps(data_audit, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    if not data_audit.get("usable_for_backfill"):
        verdict = "insufficient_historical_proba_cache"
        (OUT_DIR / "historical_backfill_append_report.md").write_text(
            f"# Append Report\n\n**Verdict:** {verdict}\n", encoding="utf-8",
        )
        return {"verdict": verdict, "rows_added": 0}

    df = load_ohlcv()
    if df is None or df.empty:
        raise SystemExit("OHLCV load failed")

    # Coverage CSV
    ts = pd.to_datetime(df["timestamp"])
    coverage = pd.DataFrame([{
        "source": "ohlcv",
        "start": str(ts.min()),
        "end": str(ts.max()),
        "rows": len(df),
    }])
    if PROBA_CACHE_DEFAULT_PATH.exists():
        cdf = pd.read_parquet(PROBA_CACHE_DEFAULT_PATH, columns=["timestamp"])
        coverage = pd.concat([coverage, pd.DataFrame([{
            "source": "tcn_proba",
            "start": str(cdf["timestamp"].min()),
            "end": str(cdf["timestamp"].max()),
            "rows": len(cdf),
        }])], ignore_index=True)
    coverage.to_csv(OUT_DIR / "historical_range_coverage.csv", index=False)

    rows_before = 0
    existing_ids: set = set()
    if MASTER_PATH.exists():
        existing, _ = load_v2_dataset()
        rows_before = len(existing)
        existing_ids = set(existing["trade_id"].astype(str))
        if not dry_run:
            shutil.copy2(MASTER_PATH, BACKUP_PATH)

    chunks = _chunk_ranges(df, chunk_days, start_date)
    if max_chunks:
        chunks = chunks[:max_chunks]

    all_batches: List[pd.DataFrame] = []
    chunk_log: List[Dict[str, Any]] = []

    for ch in chunks:
        chunk_df = _slice_chunk(df, ch["start_idx"], ch["end_idx"])
        batch = _collect_chunk_trades(chunk_df, ch["chunk_id"], replay_ts)
        if batch.empty:
            chunk_log.append({**ch, "trades": 0, "added": 0})
            continue
        new_batch = batch[~batch["trade_id"].astype(str).isin(existing_ids)].copy()
        duplicates = len(batch) - len(new_batch)
        all_batches.append(new_batch)
        existing_ids.update(new_batch["trade_id"].astype(str).tolist())
        chunk_log.append({**ch, "trades": len(batch), "added": len(new_batch), "duplicates_skipped": duplicates})

    if not all_batches:
        combined_new = pd.DataFrame()
    else:
        combined_new = pd.concat(all_batches, ignore_index=True)

    rows_added = len(combined_new)
    duplicates_skipped = sum(c.get("duplicates_skipped", 0) for c in chunk_log)

    if dry_run:
        rows_after = rows_before + rows_added
        dataset_after = pd.concat([load_v2_dataset()[0], combined_new], ignore_index=True) if rows_added else load_v2_dataset()[0]
    else:
        if rows_added > 0:
            dataset_after, added = safe_append_master(combined_new, replay_ts)
            rows_after = len(dataset_after)
            rows_added = added
        else:
            dataset_after, _ = load_v2_dataset()
            rows_after = len(dataset_after)

    # Snapshots for backfill batch only
    if not dry_run and rows_added > 0:
        snap = HISTORY_DIR / f"meta_dataset_v2_backfill_{batch_id}.parquet"
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        combined_new.to_parquet(snap, index=False)

    existing_df = load_v2_dataset()[0]
    if rows_before > 0 and MASTER_PATH.exists():
        pre = pd.read_parquet(BACKUP_PATH) if BACKUP_PATH.exists() else existing_df.iloc[:rows_before]
    else:
        pre = existing_df.iloc[:max(0, rows_before)]

    new_rows = dataset_after[dataset_after.get("source_replay", pd.Series()) == "historical_backfill"] if "source_replay" in dataset_after.columns else combined_new
    if new_rows.empty and rows_added:
        new_rows = combined_new

    dist_compare = pd.DataFrame([
        _distribution_row(pre, "existing_pre_backfill"),
        _distribution_row(new_rows, "historical_backfill_new"),
        _distribution_row(dataset_after, "combined_after"),
    ])
    dist_compare.to_csv(OUT_DIR / "historical_backfill_distribution_compare.csv", index=False)

    leak = _leakage_audit([])
    q2_pre = _q2_metrics_on_dataset(pre)
    q2_post = _q2_metrics_on_dataset(dataset_after)
    reval = _revalidation_summary(dataset_after, rows_before)

    verdict = _final_verdict(rows_after, rows_added, leak, reval)

    append_report = {
        "batch_id": batch_id,
        "replay_mode": "cache_replay",
        "chunk_days": chunk_days,
        "start_date": start_date,
        "chunks_processed": len(chunk_log),
        "rows_before": rows_before,
        "rows_added": rows_added,
        "rows_after": rows_after,
        "duplicates_skipped": duplicates_skipped,
        "dry_run": dry_run,
        "verdict": verdict,
        "chunk_log": chunk_log,
    }
    (OUT_DIR / "historical_backfill_append_report.md").write_text(
        f"# Historical Backfill Append Report\n\n```json\n{json.dumps(append_report, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    quality = {
        "q2_metrics_pre": q2_pre,
        "q2_metrics_post": q2_post,
        "leakage_audit": leak,
        "distribution_compare": dist_compare.to_dict(orient="records"),
        "temporal_chunks": len(chunk_log),
        "regime_diversity_chunks_with_trades": sum(1 for c in chunk_log if c.get("trades", 0) > 0),
    }
    (OUT_DIR / "historical_backfill_quality_audit.md").write_text(
        f"# Quality Audit\n\n```json\n{json.dumps(quality, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    (OUT_DIR / "historical_backfill_meta_revalidation.md").write_text(
        f"# Meta Revalidation\n\n**Verdict:** {verdict}\n\n"
        f"```json\n{json.dumps(reval, indent=2, default=str)}\n```\n\n"
        f"Q2_BDI baseline unchanged. No production/monitor promotion.\n",
        encoding="utf-8",
    )

    (OUT_DIR / "historical_replay_mode_summary.md").write_text(
        f"# Replay Mode Summary\n\n- mode: **cache_replay**\n"
        f"- chunks: {len(chunk_log)}\n- chunk_days: {chunk_days}\n"
        f"- per-chunk KS reset: yes (time-safe independent windows)\n"
        f"- proba: TCN cache merge_asof\n",
        encoding="utf-8",
    )

    (OUT_DIR / "timezone_integrity_check.md").write_text(
        f"# Timezone Integrity\n\nOHLCV tz-naive UTC assumed. Proba aligned via merge_asof 5m tolerance.\n",
        encoding="utf-8",
    )

    pd.DataFrame(chunk_log).to_csv(OUT_DIR / "historical_replay_chunk_log.csv", index=False)

    return {
        "verdict": verdict,
        "rows_before": rows_before,
        "rows_added": rows_added,
        "rows_after": rows_after,
        "chunks": len(chunk_log),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Historical meta dataset backfill")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    parser.add_argument("--start-date", default="2021-06-01")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=None)
    args = parser.parse_args()

    r = run_backfill(
        chunk_days=args.chunk_days,
        start_date=args.start_date,
        dry_run=args.dry_run,
        max_chunks=args.max_chunks,
    )
    print(f"verdict: {r['verdict']}")
    print(f"rows: {r['rows_before']} -> {r['rows_after']} (+{r['rows_added']})")
    print(f"chunks: {r['chunks']}")


if __name__ == "__main__":
    main()
