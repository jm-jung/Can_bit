"""
Daily Meta research pipeline — dataset accumulation + audit + Discord (diagnostics only).

Does NOT modify production trading, execution, or existing production launchd jobs.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import shutil
import sys
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.build_meta_label_dataset import (
    HISTORY_DIR,
    MANIFEST_PATH,
    MASTER_PATH,
    META_V2_VERSION,
    MODEL_ID,
    OUT_DIR,
    SOURCE_REPLAY,
    _collect_production_trades_v2,
    _dataset_statistics,
    _dedupe,
    _ohlcv_features,
    _regime_summary,
    _write_summary_report,
    load_v2_dataset,
)
from scripts.diagnostics.meta_research_audit import (
    append_growth_history,
    audit_dataset,
    write_daily_audit_md,
    write_integrity_report,
)
from scripts.diagnostics.meta_research_notifier import send_failure_alert, send_success_alert
from scripts.diagnostics.run_forward_meta_shadow_monitor import run_shadow_monitor
from scripts.diagnostics.validate_cwce_shadow_candidate import run_cwce_daily_shadow
from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _entropy, _simulate_variant
from scripts.diagnostics.validate_q2_penalty_tournament import apply_penalties
from scripts.diagnostics.validate_quality_score_replay import (
    FEE_RATE,
    POSITION_SIZE,
    SLIPPAGE_RATE,
    _risk_routing,
    _simulate_quality,
    map_m3,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("meta_research_pipeline")

DAILY_REPORTS_DIR = OUT_DIR / "daily_reports"
OPS_LOG_DIR = Path("data/ops_logs")
LOCK_PATH = OUT_DIR / ".meta_research_pipeline.lock"
STATUS_LOCK_PATH = OUT_DIR / "meta_research_status_lock.md"
BACKUP_PATH = MASTER_PATH.with_suffix(".parquet.bak")

PENALTY_D = frozenset({"D"})
PENALTY_BDI = frozenset({"B", "D", "I"})
SCORE_Q2_PD = lambda t, fm: apply_penalties(t, fm, set(PENALTY_D))
SCORE_Q2_BDI = lambda t, fm: apply_penalties(t, fm, set(PENALTY_BDI))
BASELINE = Variant("Baseline", fail_scale=1.0, hard_block=False)

RETRAIN_ROW_DELTA = 50
WEEKLY_RETRAIN_NOTE = "Meta retrain: daily forbidden; weekly or +50 rows only"


@contextmanager
def pipeline_lock() -> Generator[None, None, None]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lock_file = open(LOCK_PATH, "w", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file.write(f"pid={__import__('os').getpid()} ts={datetime.now().isoformat()}\n")
        lock_file.flush()
        yield
    except BlockingIOError as exc:
        raise RuntimeError("Another meta research pipeline instance is running") from exc
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        finally:
            lock_file.close()


def _mdd(vals: List[float]) -> float:
    eq = peak = 1.0
    mdd = 0.0
    for r in vals:
        eq *= 1.0 + r * POSITION_SIZE
        peak = max(peak, eq)
        mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0)
    return float(mdd)


def _false_high_count(tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> int:
    n = 0
    for _, tr in tdf.iterrows():
        tt = ticks[int(tr["entry_idx"])]
        if (
            tr.get("direction") == "LONG"
            and str(tt.get("trend_label")) == "up"
            and str(tt.get("vol_bucket")) == "high"
            and _entropy(tt) <= 0.90
            and float(tr.get("scaled_return", 0)) < 0
        ):
            n += 1
    return n


def _run_q2_baseline_metrics(ticks: List[Dict[str, Any]], feat_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    _, prod_tdf = _simulate_variant(ticks, BASELINE, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, pd_tdf, _ = _simulate_quality(ticks, SCORE_Q2_PD, map_m3, feat_map)
    _, bdi_tdf, _ = _simulate_quality(ticks, SCORE_Q2_BDI, map_m3, feat_map)

    def _row(name: str, tdf: pd.DataFrame) -> Dict[str, Any]:
        vals = tdf["scaled_return"].tolist() if not tdf.empty else []
        rr = _risk_routing(tdf, ticks)
        routing_valid = rr["avg_scale_winners"] > rr["avg_scale_losers"] > rr["avg_scale_rfe"] if not tdf.empty else False
        return {
            "candidate": name,
            "trades": len(tdf),
            "preservation": len(tdf) / max(len(tdf), 1),
            "net_return": float(sum(vals)),
            "MDD": _mdd(vals),
            "routing_valid": routing_valid,
            "false_high": _false_high_count(tdf, ticks),
            "RFE": int((tdf["exit_reason"] == "risk_force_exit").sum()) if not tdf.empty else 0,
            "avg_scale": float(tdf["scale"].mean()) if not tdf.empty else 0,
        }

    return {
        "production": _row("Production", prod_tdf),
        "q2_pd": _row("Q2_PD", pd_tdf),
        "q2_bdi": _row("Q2_BDI", bdi_tdf),
    }


def safe_append_master(batch: pd.DataFrame, replay_ts: str) -> Tuple[pd.DataFrame, int]:
    """Append-only with dedupe, backup, and rollback on failure."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    rows_before = 0
    if MASTER_PATH.exists():
        master = pd.read_parquet(MASTER_PATH)
        rows_before = len(master)
        shutil.copy2(MASTER_PATH, BACKUP_PATH)
    else:
        master = pd.DataFrame()

    try:
        if master.empty:
            new_ids = set(batch["trade_id"]) if "trade_id" in batch.columns else set()
            combined = batch.copy()
        else:
            existing = set(master["trade_id"])
            new_batch = batch[~batch["trade_id"].isin(existing)].copy()
            if new_batch.empty:
                combined = master
            else:
                combined = pd.concat([master, new_batch], ignore_index=True)

        combined = _dedupe(combined)
        combined = combined.sort_values(["df_idx", "entry_ts"], na_position="last").reset_index(drop=True)

        if combined["trade_id"].duplicated().any():
            raise RuntimeError("duplicate trade_id after dedupe")

        combined.to_parquet(MASTER_PATH, index=False)
        snap = HISTORY_DIR / f"meta_dataset_v2_{replay_ts}.parquet"
        batch.to_parquet(snap, index=False)

        stats = _dataset_statistics(combined)
        manifest = {
            "meta_v2_version": META_V2_VERSION,
            "last_replay_timestamp": replay_ts,
            "source_replay": SOURCE_REPLAY,
            "model_id": MODEL_ID,
            "master_path": str(MASTER_PATH),
            "history_snapshot": str(snap),
            "total_rows": stats["rows"],
            "batch_rows": int(len(batch)),
            "new_rows_appended": int(len(combined) - rows_before),
            "regime_summary": _regime_summary(combined),
            "statistics": stats,
        }
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

        added = len(combined) - rows_before
        return combined, added
    except Exception:
        if BACKUP_PATH.exists() and MASTER_PATH.exists():
            shutil.copy2(BACKUP_PATH, MASTER_PATH)
            logger.error("Append failed — restored master from backup")
        raise


def _next_action(audit: Dict[str, Any], rows_added: int) -> str:
    n = audit.get("total_rows", 0)
    if audit.get("audit_status") != "PASS":
        return "fix_integrity_before_accumulation"
    if n < 200:
        return f"accumulate_forensic_rows ({n}/200 checkpoint)"
    if n < 300:
        return f"accumulate_until_300 ({n}/300 Meta re-eval gate)"
    if n < 500:
        return f"accumulate_until_500 ({n}/500 continuous re-eval gate)"
    if rows_added >= RETRAIN_ROW_DELTA:
        return "weekly_or_delta_retrain_eligible — run OOS re-eval manually"
    return "maintain_baseline_Q2_BDI — research archive only"


def _write_status_lock(audit: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    STATUS_LOCK_PATH.write_text(
        "# Meta Research Status Lock\n\n"
        "## OFFICIAL STATUS\n\n"
        "**Production:** unchanged\n\n"
        "**Official forensic baseline:** Q2_BDI discrete M3\n\n"
        "**Research archive:** Meta_V2, Continuous allocation, Hybrid adjustment\n\n"
        "**Research status:** accumulating forensic dataset\n\n"
        f"**Rows:** {audit.get('total_rows')} (milestone {audit.get('milestone_progress_300')})\n\n"
        f"**Q2_BDI MDD:** {metrics['q2_bdi']['MDD']:.6f}\n\n"
        f"**Verdict:** Q2_BDI baseline retained — promotion_ready forbidden until 300+ rows\n\n"
        f"**Retrain policy:** {WEEKLY_RETRAIN_NOTE}\n",
        encoding="utf-8",
    )


def _write_daily_summary(
    ts: str,
    audit: Dict[str, Any],
    metrics: Dict[str, Any],
    rows_added: int,
    append_ok: bool,
) -> Path:
    DAILY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = DAILY_REPORTS_DIR / f"meta_research_daily_{ts}.md"
    bdi = metrics["q2_bdi"]
    prod = metrics["production"]
    path.write_text(
        f"# Meta Research Daily Summary — {ts}\n\n"
        f"## Dataset\n"
        f"- total_rows: {audit.get('total_rows')}\n"
        f"- rows_added: {rows_added}\n"
        f"- append_success: {append_ok}\n"
        f"- milestone: {audit.get('milestone_progress_300')}\n"
        f"- integrity: {audit.get('audit_status')}\n\n"
        f"## Q2_BDI baseline\n"
        f"- MDD: {bdi['MDD']:.6f}\n"
        f"- false_high: {bdi['false_high']}\n"
        f"- routing_valid: {bdi['routing_valid']}\n"
        f"- preservation: 100%\n\n"
        f"## Production (reference)\n"
        f"- MDD: {prod['MDD']:.6f}\n"
        f"- false_high: {prod['false_high']}\n\n"
        f"## Regime\n"
        f"- long_ratio: {audit.get('long_ratio', 0):.1%}\n"
        f"- high_vol_ratio: {audit.get('high_vol_ratio', 0):.1%}\n"
        f"- LONG bias watch: {'YES' if audit.get('long_ratio', 0) > 0.85 else 'no'}\n\n"
        f"## Next action\n{ _next_action(audit, rows_added) }\n",
        encoding="utf-8",
    )
    return path


def run_pipeline(*, dry_run: bool = False, no_discord: bool = False) -> Dict[str, Any]:
    replay_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = OPS_LOG_DIR / f"meta_research_pipeline_{replay_ts}.log"
    OPS_LOG_DIR.mkdir(parents=True, exist_ok=True)

    report: Dict[str, Any] = {
        "timestamp": replay_ts,
        "status": "FAIL",
        "append_success": False,
        "rows_added": 0,
    }

    try:
        with pipeline_lock():
            logger.info("Pipeline lock acquired")

            ohlcv = load_ohlcv()
            if ohlcv is None or ohlcv.empty:
                raise RuntimeError("OHLCV load failed")
            feat_map, _ = _ohlcv_features(ohlcv)
            ticks, _ = simulate_signals_paper(ohlcv)
            if not ticks:
                raise RuntimeError("No ticks from replay")

            metrics = _run_q2_baseline_metrics(ticks, feat_map)
            report["q2_metrics"] = metrics

            batch = _collect_production_trades_v2(ticks, feat_map, replay_ts)
            if batch.empty:
                raise RuntimeError("No production trades collected")

            if dry_run:
                dataset, _ = load_v2_dataset()
                added = 0
                append_ok = True
                logger.info("Dry-run: skip append")
            else:
                dataset, added = safe_append_master(batch, replay_ts)
                append_ok = True
                _write_summary_report(OUT_DIR / f"meta_dataset_v2_summary_{replay_ts}.md", dataset, replay_ts)

            audit = audit_dataset(dataset if not dry_run else load_v2_dataset()[0], MASTER_PATH)
            report.update({
                "append_success": append_ok,
                "rows_added": added,
                "total_rows": audit["total_rows"],
                "q2_bdi_mdd": metrics["q2_bdi"]["MDD"],
                "false_high": metrics["q2_bdi"]["false_high"],
                "schema_drift": audit["schema_drift"],
                "duplicate_trade_ids": audit["duplicate_trade_ids"],
                "audit_status": audit["audit_status"],
                "milestone": {"current": audit["total_rows"], "target_300": 300, "next": audit["milestone_next"]},
                "next_action": _next_action(audit, added),
            })

            write_integrity_report(audit)
            write_daily_audit_md(audit, replay_ts, OUT_DIR / f"meta_dataset_daily_audit_{replay_ts}.md")
            _write_daily_summary(replay_ts, audit, metrics, added, append_ok)
            _write_status_lock(audit, metrics)

            if not dry_run:
                append_growth_history({
                    "timestamp": replay_ts,
                    "rows_added": added,
                    "total_rows": audit["total_rows"],
                    "long_ratio": audit.get("long_ratio"),
                    "high_vol_ratio": audit.get("high_vol_ratio"),
                    "audit_status": audit["audit_status"],
                    "q2_bdi_mdd": metrics["q2_bdi"]["MDD"],
                    "false_high": metrics["q2_bdi"]["false_high"],
                })

            if audit["audit_status"] == "PASS" and append_ok:
                report["status"] = "PASS"
            else:
                report["status"] = "FAIL"
                raise RuntimeError(f"Audit failed: {audit.get('audit_status')}")

            try:
                shadow = run_shadow_monitor(dry_run=dry_run)
                report["meta_shadow"] = shadow
            except Exception as shadow_exc:
                logger.error("Meta shadow monitor failed: %s", shadow_exc)
                report["meta_shadow"] = {
                    "status": "FAIL",
                    "reason": str(shadow_exc),
                    "promotion_ready": False,
                }

            try:
                cwce_shadow = run_cwce_daily_shadow(dry_run=True)
                cwce_shadow["status"] = "PASS"
                report["risk_aware_tcn_shadow"] = cwce_shadow
            except Exception as cwce_exc:
                logger.error("CWCE shadow monitor failed: %s", cwce_exc)
                report["risk_aware_tcn_shadow"] = {
                    "status": "FAIL",
                    "reason": str(cwce_exc),
                    "promotion_ready": False,
                }

            if not no_discord:
                send_success_alert(report, dry_run=dry_run)

            log_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
            logger.info("Pipeline complete: rows=%s added=%s", audit["total_rows"], added)
            return report

    except Exception as exc:
        tb = traceback.format_exc()
        logger.error("Pipeline failed: %s", exc)
        log_path.write_text(tb, encoding="utf-8")
        if not no_discord:
            send_failure_alert(str(exc), tb[-800:], dry_run=dry_run)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Meta research daily pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Skip append, test flow")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    report = run_pipeline(dry_run=args.dry_run, no_discord=args.no_discord)
    print(f"status: {report['status']}")
    print(f"rows_total: {report.get('total_rows')}")
    print(f"rows_added: {report.get('rows_added')}")
    print(f"audit: {report.get('audit_status')}")


if __name__ == "__main__":
    main()
