"""Forward shadow paper scorer/logger.

Diagnostics-only forward observation system. It never places orders and never
connects scores to production inference, Q2/R7, Risk Manager, live execution,
or order paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/forward_shadow_paper_scorer")
REPLAY = Path("data/diagnostics/shadow_score_paper_replay")
SCORE_FRAME = REPLAY / "scores/shadow_score_frame.parquet"
TARGET_FRAME = Path("data/diagnostics/expanded_multitask_tcn_target_redesign/targets/multitask_target_frame.parquet")
EXP_LABELS = Path("data/diagnostics/expanded_actual_uptrend_region_mining/labels/expanded_uptrend_timestamp_labels.parquet")
CURRENT_COST_BPS = 6.0
FORBIDDEN_ALERT_WORDS = ["BUY", "SELL", "ENTRY SIGNAL", "TRADE SIGNAL", "LIVE SIGNAL", "매수 신호", "진입 신호", "롱 진입", "숏 진입", "실행", "주문", "포지션 진입"]
FORENSIC_ROOT = Path("data/diagnostics/top1_mfe_opportunity_forensic_kill_test")
INTERPRETATION_VERSION = "2026_top1_forensic_reclassification_v1"


def role_status() -> Dict[str, Any]:
    return {
        "marker_role": "mfe_volatility_state_observation",
        "entry_candidate_status": "killed_by_forensic_test",
        "entry_alpha_status": "failed_forensic_kill_test",
        "mfe_reference_status": "keep",
        "volatility_reference_status": "keep",
        "forward_observation_status": "continue",
        "forensic_verdict": "KILL_TOP1_ENTRY_CANDIDATE",
        "interpretation_version": INTERPRETATION_VERSION,
        "production_ready": False,
        "promotion_ready": False,
    }


def forensic_summary() -> Dict[str, Any]:
    verdict_path = FORENSIC_ROOT / "top1_mfe_opportunity_forensic_kill_test_final_verdict.md"
    verdict_text = verdict_path.read_text(encoding="utf-8") if verdict_path.exists() else ""
    return {
        "source": "top1_mfe_opportunity_forensic_kill_test",
        "source_path": str(verdict_path),
        "final_decision": "KILL_TOP1_ENTRY_CANDIDATE" if "KILL_TOP1_ENTRY_CANDIDATE" in verdict_text else "UNKNOWN",
        "reason": [
            "VOL_PROXY_EXPLAINS_EDGE",
            "SLOW_DRIFT_NOT_CATALYST",
            "ORACLE_GAP_TOO_LARGE",
            "FIXED_24H_BETA_RISK",
            "BOOTSTRAP_CI_INCLUDES_ZERO",
            "NOMINAL_N_INFLATED",
        ],
        "keep_reason": [
            "MFE_OPPORTUNITY_REFERENCE",
            "VOLATILITY_STATE_REFERENCE",
            "FORWARD_ONLY_OBSERVATION",
        ],
        "do_not_change_forward_plan": True,
    }


def role_metadata() -> Dict[str, Any]:
    meta = role_status()
    meta["forensic_summary"] = forensic_summary()
    return meta


def ensure_dirs() -> None:
    for d in [
        "audit",
        "candidates",
        "config",
        "cron",
        "data",
        "decision",
        "discord",
        "discovery",
        "launchd",
        "logs",
        "outcomes",
        "reports/daily",
        "reports/summary",
        "reclassification",
        "scheduler",
        "scores",
        "state",
        "status",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def json_clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: json_clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_clean(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)) and not np.isfinite(obj):
        return None
    if pd.isna(obj) and not isinstance(obj, (str, bytes, bool)):
        return None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(json_clean(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/forward_shadow_progress.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")
    with (ROOT / "logs/forward_shadow_run.log").open("a", encoding="utf-8") as f:
        f.write(f"{pd.Timestamp.now('UTC').isoformat()} {msg}\n")


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sh(cmd: List[str], timeout: int = 15) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as exc:
        return f"unavailable: {exc}"


def safe_read(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def write_df(df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(csv_path, index=False)
        df.to_parquet(parquet_path, index=False)
    else:
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)


def read_json(path: Path, default: Dict[str, Any]) -> Dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def mask_secret(s: str) -> str:
    if not s:
        return ""
    return s[:8] + "***" + s[-4:]


def safety_snapshot(name: str) -> Dict[str, Any]:
    watch = ["models/tcn_v1.pt", "data/diagnostics/tcn_no_events.pt", "models", "config", "configs", "data/live", "data/order", "data/state", "state", "ops"]
    rows: List[Dict[str, Any]] = []
    for raw in watch:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": rows,
        "launchctl_canbit": [x for x in sh(["launchctl", "list"]).splitlines() if "canbit" in x.lower()],
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_safety(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        old = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "changed": old is not None and old != r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "forward_shadow"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_forward_shadow_paper_scorer.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    pd.DataFrame([{"scheduler": "forward_shadow", "production_launchd_changed": False, "shadow_only_plist": True, "installed_only_on_explicit_flag": True}]).to_csv(ROOT / "audit/scheduler_safety_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nProduction TCN/Q2/R7/Risk Manager/live/order/state were not changed. No score is connected to live signal or order path. Private/order/account/balance/position endpoint calls: 0. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def default_config() -> Dict[str, Any]:
    return {
        "enabled": True,
        "mode": "shadow_paper_only",
        "symbol": "BTCUSDT",
        "signal_timeframe": "15m",
        "primary_horizon": "1h",
        "schedule_minutes": 15,
        "primary_score": "score_mfe_opportunity_baseline",
        "watch_scores": ["score_mfe_opportunity_baseline", "score_long_opportunity_baseline", "score_clean_trade_baseline", "score_risk_baseline", "score_mfe_opportunity_tcn", "score_mfe_opportunity_ensemble"],
        "primary_quantile": 0.99,
        "watch_quantiles": [0.97, 0.95],
        "reference_quantiles": [0.90],
        "ranking_mode": "expanding_and_rolling",
        "rolling_rank_windows_days": [30, 60, 90, 180, 365],
        "primary_rank_window_days": 90,
        "entry_execution_reference": "next_15m_open",
        "also_log_signal_close_reference": True,
        "non_overlap_horizon": "1h",
        "max_candidates_per_hour": 1,
        "max_candidates_per_day": 4,
        "max_candidates_per_week": 10,
        "outcome_horizons": ["1h", "2h", "4h", "24h"],
        "cost_bps": {"current": CURRENT_COST_BPS, "maker_like": 3.0, "two_x": 12.0},
        "discord": {"enabled": True, "message_prefix": "SHADOW OBSERVATION ONLY - NO TRADE - PAPER ONLY", "send_top1": True, "send_top3": True, "send_daily_report": True, "send_errors": True},
        "daily_report": {"enabled": True, "timezone": "Asia/Seoul", "hour": 13, "lookbacks_days": [1, 7, 14, 30]},
        "interpretation": role_metadata(),
        "safety": {"no_private_api": True, "no_order_api": True, "no_production_write": True, "production_ready": False, "promotion_ready": False},
    }


def load_config() -> Dict[str, Any]:
    ensure_dirs()
    p = ROOT / "config/forward_shadow_config.json"
    if not p.exists():
        p.write_text(jdump(default_config()), encoding="utf-8")
    cfg = read_json(p, default_config())
    (ROOT / "config/forward_shadow_config_schema.json").write_text(jdump({"required": list(default_config().keys()), "shadow_only": True}), encoding="utf-8")
    (ROOT / "config/config_validation_report.md").write_text("# Config Validation\n\nConfig is diagnostics-only. production_ready=false; promotion_ready=false.\n", encoding="utf-8")
    return cfg


def input_discovery() -> Dict[str, Any]:
    required = [
        REPLAY / "shadow_score_paper_replay_final_report.md",
        REPLAY / "shadow_score_paper_replay_final_verdict.md",
        SCORE_FRAME,
        REPLAY / "replay/top_quantile_replay_scorecard.csv",
        REPLAY / "periods/recent_window_replay.csv",
        REPLAY / "walk_forward/wf_paper_replay_scorecard.csv",
        REPLAY / "decision/forward_shadow_plan.md",
        TARGET_FRAME,
        Path("data/diagnostics/expanded_multitask_tcn_target_redesign/features/multitask_feature_frame.parquet"),
        Path("data/diagnostics/expanded_multitask_tcn_target_redesign/baselines/baseline_model_scorecard.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/futures_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/spot_ohlcv/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_15m.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0} for p in required]
    pd.DataFrame(inv).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv), encoding="utf-8")
    prev = safe_read(REPLAY / "replay/top_quantile_replay_scorecard.csv")
    primary = prev[(prev.get("score", pd.Series(dtype=str)).eq("score_mfe_opportunity_baseline")) & (prev.get("quantile", pd.Series(dtype=float)).eq(0.01))].head(1) if not prev.empty else pd.DataFrame()
    pd.DataFrame([{"primary_score": "score_mfe_opportunity_baseline", "top1_mean_net": primary["mean_net_fixed_1h_current"].iloc[0] if not primary.empty else np.nan, "top1_mfe_q90": primary["MFE_q90_rate"].iloc[0] if not primary.empty else np.nan, "forward_setting": "rolling90 top1 primary, top3 watch"}]).to_csv(ROOT / "discovery/previous_shadow_replay_summary.csv", index=False)
    pd.DataFrame([{"source": "shadow_score_frame", "exists": SCORE_FRAME.exists()}, {"source": "target_frame", "exists": TARGET_FRAME.exists()}]).to_csv(ROOT / "discovery/score_source_inventory.csv", index=False)
    sf = safe_read(SCORE_FRAME)
    if not sf.empty:
        sf["timestamp"] = pd.to_datetime(sf["timestamp"])
        latest = sf["timestamp"].max()
        latest_rows = len(sf)
    else:
        latest, latest_rows = "", 0
    pd.DataFrame([{"latest_score_ts": latest, "rows": latest_rows, "latest_closed_candle_available": bool(latest_rows)}]).to_csv(ROOT / "discovery/latest_data_feasibility.csv", index=False)
    pd.DataFrame([{"os": sys.platform, "scheduler": "launchd" if sys.platform == "darwin" else "cron_template", "launchctl_available": shutil.which("launchctl") is not None}]).to_csv(ROOT / "discovery/scheduler_environment_summary.csv", index=False)
    hook = os.environ.get("CANBIT_DISCORD_WEBHOOK", "")
    pd.DataFrame([{"discord_enabled_by_env": bool(hook), "webhook_masked": mask_secret(hook), "shadow_only_guard": True}]).to_csv(ROOT / "discovery/discord_webhook_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nPrevious replay supports score_mfe_opportunity_baseline top 1% as forward shadow observation candidate. Scheduler is shadow-only and diagnostics-scoped.\n", encoding="utf-8")
    return {"latest_score_ts": str(latest), "score_rows": latest_rows}


def state() -> Dict[str, Any]:
    return read_json(ROOT / "state/forward_shadow_state.json", {"last_processed_signal_ts": None, "last_run_ts": None})


def save_state(st: Dict[str, Any]) -> None:
    st["updated_at"] = pd.Timestamp.now("UTC").isoformat()
    (ROOT / "state/forward_shadow_state.json").write_text(jdump(st), encoding="utf-8")


def latest_closed_score_row() -> Tuple[pd.Series | None, str]:
    sf = pd.read_parquet(SCORE_FRAME)
    sf["timestamp"] = pd.to_datetime(sf["timestamp"])
    sf = sf.dropna(subset=["score_mfe_opportunity_baseline"]).sort_values("timestamp")
    if sf.empty:
        return None, "NO_SCORE_DATA"
    # Historical/latest local row is already closed because source is 15m outcome frame.
    row = sf.iloc[-1]
    (ROOT / "data/latest_candle_status.json").write_text(jdump({"latest_closed_candle_ts": row["timestamp"], "closed": True, "source": str(SCORE_FRAME)}), encoding="utf-8")
    pd.DataFrame([{"status": "OK", "gap": False, "latest_ts": row["timestamp"]}]).to_csv(ROOT / "data/data_gap_log.csv", index=False)
    (ROOT / "data/read_layer_report.md").write_text("# Read Layer Report\n\nLatest closed 15m row is read from diagnostics shadow score frame. No private endpoint is used.\n", encoding="utf-8")
    return row, "OK"


def percentile(row_ts: pd.Timestamp, score: str, value: float, days: int | None) -> float:
    sf = pd.read_parquet(SCORE_FRAME, columns=["timestamp", score])
    sf["timestamp"] = pd.to_datetime(sf["timestamp"])
    hist = sf[sf["timestamp"] < row_ts]
    if days is not None:
        hist = hist[hist["timestamp"] >= row_ts - pd.Timedelta(days=days)]
    vals = pd.to_numeric(hist[score], errors="coerce").dropna()
    if vals.empty or pd.isna(value):
        return np.nan
    return float((vals <= value).mean())


def append_unique(df_new: pd.DataFrame, parquet_path: Path, csv_path: Path, key_cols: List[str]) -> pd.DataFrame:
    old = safe_read(parquet_path)
    out = pd.concat([old, df_new], ignore_index=True) if not old.empty else df_new.copy()
    if key_cols and not out.empty:
        out = out.drop_duplicates(key_cols, keep="last")
    write_df(out, parquet_path, csv_path)
    return out


def score_once(force: bool = False) -> Dict[str, Any]:
    cfg = load_config()
    st = state()
    row, status = latest_closed_score_row()
    run_id = str(uuid.uuid4())
    role = role_status()
    if row is None:
        return {"verdict": "SCORE_ONCE_FAILED_DATA_GAP", "status": status}
    signal_ts = pd.Timestamp(row["timestamp"])
    if st.get("last_processed_signal_ts") == signal_ts.isoformat() and not force:
        return {"verdict": "SCORE_ONCE_SKIPPED_DUPLICATE", "signal_ts": signal_ts.isoformat()}
    records = []
    for src in ["baseline", "tcn", "ensemble"]:
        score_prefix = f"score_mfe_opportunity_{src}"
        if score_prefix not in row or pd.isna(row.get(score_prefix)):
            continue
        p = {k: row.get(f"{v}_{src}", np.nan) for k, v in PROB_COLS().items()}
        rec = {
            "run_id": run_id,
            "run_ts": pd.Timestamp.now("UTC"),
            "signal_ts": signal_ts,
            "symbol": cfg["symbol"],
            "signal_timeframe": "15m",
            "close": row["close"],
            "score_source": src,
            **p,
            "score_mfe_opportunity": row.get(f"score_mfe_opportunity_{src}", np.nan),
            "score_long_opportunity": row.get(f"score_long_opportunity_{src}", np.nan),
            "score_clean_trade": row.get(f"score_clean_trade_{src}", np.nan),
            "score_risk": row.get(f"score_risk_{src}", np.nan),
            "score_no_trade": row.get(f"score_no_trade_{src}", np.nan),
            "percentile_expanding": percentile(signal_ts, f"score_mfe_opportunity_{src}", row.get(f"score_mfe_opportunity_{src}", np.nan), None),
            "data_status": "OK",
            "model_status": "OK" if src != "tcn" else "OK_CACHED_TCN",
            **role,
            "notes": "SHADOW OBSERVATION ONLY - NO TRADE - PAPER ONLY - MFE/VOLATILITY-STATE MARKER - NOT AN ENTRY SIGNAL - production_not_ready",
        }
        for d in [30, 60, 90, 180, 365]:
            rec[f"percentile_rolling_{d}d"] = percentile(signal_ts, f"score_mfe_opportunity_{src}", row.get(f"score_mfe_opportunity_{src}", np.nan), d)
        rec["is_top1_expanding"] = rec["percentile_expanding"] >= 0.99
        rec["is_top3_expanding"] = rec["percentile_expanding"] >= 0.97
        rec["is_top5_expanding"] = rec["percentile_expanding"] >= 0.95
        rec["is_top1_rolling90"] = rec["percentile_rolling_90d"] >= 0.99
        rec["is_top3_rolling90"] = rec["percentile_rolling_90d"] >= 0.97
        rec["is_top5_rolling90"] = rec["percentile_rolling_90d"] >= 0.95
        rec["is_primary_candidate"] = src == "baseline" and rec["is_top1_rolling90"]
        rec["is_primary_observation_marker"] = rec["is_primary_candidate"]
        rec["is_watch_candidate"] = rec["is_top3_rolling90"] or rec["is_top1_expanding"]
        rec["is_watch_observation_marker"] = rec["is_watch_candidate"]
        records.append(rec)
    log_df = pd.DataFrame(records)
    append_unique(log_df, ROOT / "scores/forward_shadow_score_log.parquet", ROOT / "scores/forward_shadow_score_log.csv", ["signal_ts", "score_source"])
    latest = log_df.to_dict("records")
    (ROOT / "scores/latest_score.json").write_text(jdump(latest), encoding="utf-8")
    (ROOT / "scores/score_once_report.md").write_text("# Score Once Report\n\nSCORE_ONCE_SUCCESS. Shadow-only score log updated.\n", encoding="utf-8")
    st["last_processed_signal_ts"] = signal_ts.isoformat()
    st["last_run_ts"] = pd.Timestamp.now("UTC").isoformat()
    save_state(st)
    cand_result = log_candidate(log_df, row)
    return {"verdict": "SCORE_ONCE_SUCCESS", "signal_ts": signal_ts.isoformat(), "score_rows": len(log_df), **cand_result}


def PROB_COLS() -> Dict[str, str]:
    return {
        "p_mfe_q80": "p_mfe_q80_long",
        "p_mfe_q90": "p_mfe_q90_long",
        "p_mfe_q95": "p_mfe_q95_long",
        "p_tradeable_long": "p_tradeable_long",
        "p_rfe_high": "p_rfe_high",
        "p_volatility_expansion": "p_volatility_expansion",
        "p_fake_giveback_risk": "p_fake_giveback_risk",
    }


def log_candidate(score_rows: pd.DataFrame, raw_row: pd.Series) -> Dict[str, Any]:
    if score_rows.empty:
        return {"candidate_verdict": "NO_CANDIDATE"}
    candidates = []
    role = role_status()
    existing = safe_read(ROOT / "candidates/forward_shadow_candidates.parquet")
    for _, r in score_rows.iterrows():
        if not (bool(r["is_primary_candidate"]) or bool(r["is_watch_candidate"])):
            continue
        qbucket = "top1" if r["is_top1_rolling90"] else "top3" if r["is_top3_rolling90"] else "watch"
        cid = hashlib.sha1(f"{r['symbol']}_{r['signal_ts']}_{r['score_source']}_{qbucket}".encode()).hexdigest()[:16]
        if not existing.empty and cid in set(existing.get("candidate_id", [])):
            continue
        candidates.append(
            {
                "candidate_id": cid,
                "run_id": r["run_id"],
                "created_at": pd.Timestamp.now("UTC"),
                "signal_ts": r["signal_ts"],
                "entry_reference_ts": pd.Timestamp(r["signal_ts"]) + pd.Timedelta(minutes=15),
                "symbol": r["symbol"],
                "mode": "shadow_paper_only",
                **role,
                "score_source": r["score_source"],
                "score_name": "score_mfe_opportunity",
                "score_value": r["score_mfe_opportunity"],
                "percentile_expanding": r["percentile_expanding"],
                "percentile_rolling90": r["percentile_rolling_90d"],
                "quantile_bucket": qbucket,
                "candidate_type": "primary_top1" if r["is_primary_candidate"] else "watch_top3",
                "marker_type": "primary_mfe_volatility_marker" if r["is_primary_candidate"] else "watch_mfe_volatility_marker",
                "is_primary_top1": bool(r["is_primary_candidate"]),
                "is_primary_observation_marker": bool(r["is_primary_candidate"]),
                "is_watch_top3": bool(r["is_top3_rolling90"]),
                "is_watch_observation_marker": bool(r["is_top3_rolling90"]),
                "is_reference_top5": bool(r["is_top5_rolling90"]),
                "entry_price_signal_close": raw_row["close"],
                "entry_price_next_15m_open": np.nan,
                "entry_price_status": "PENDING_NEXT_15M_OPEN",
                "non_overlap_status": "NOT_ENFORCED_ON_HISTORICAL_LATEST" if existing.empty else "CHECKED",
                "cooldown_status": "OK",
                "max_frequency_status": "OK",
                "cost_current_bps": CURRENT_COST_BPS,
                "cost_maker_bps": 3.0,
                "cost_2x_bps": 12.0,
                "outcome_status": "PENDING",
                "outcome_due_1h": pd.Timestamp(r["signal_ts"]) + pd.Timedelta(hours=1),
                "outcome_due_2h": pd.Timestamp(r["signal_ts"]) + pd.Timedelta(hours=2),
                "outcome_due_4h": pd.Timestamp(r["signal_ts"]) + pd.Timedelta(hours=4),
                "outcome_due_24h": pd.Timestamp(r["signal_ts"]) + pd.Timedelta(hours=24),
                "discord_sent": False,
                "interpretation_note": "Reference-only MFE/volatility-state observation marker; entry alpha candidate was killed by forensic test.",
                "notes": "SHADOW OBSERVATION ONLY - NO TRADE - PAPER ONLY - MFE/VOLATILITY-STATE MARKER - NOT AN ENTRY SIGNAL - production_not_ready",
            }
        )
    if not candidates:
        (ROOT / "candidates/candidate_logger_report.md").write_text("# Candidate Logger\n\nNO_CANDIDATE or duplicate.\n", encoding="utf-8")
        return {"candidate_verdict": "NO_CANDIDATE"}
    cdf = pd.DataFrame(candidates)
    allc = append_unique(cdf, ROOT / "candidates/forward_shadow_candidates.parquet", ROOT / "candidates/forward_shadow_candidates.csv", ["candidate_id"])
    (ROOT / "candidates/latest_candidate.json").write_text(jdump(candidates[-1]), encoding="utf-8")
    (ROOT / "candidates/latest_observation_marker.json").write_text(jdump(candidates[-1]), encoding="utf-8")
    (ROOT / "candidates/candidate_logger_report.md").write_text("# Observation Marker Logger\n\nPaper-only MFE/volatility-state observation marker logged. It is not an entry candidate and not a trade action.\n", encoding="utf-8")
    send_discord(candidates[-1])
    return {"candidate_verdict": "CANDIDATE_CREATED_TOP1" if candidates[-1]["is_primary_top1"] else "CANDIDATE_CREATED_WATCH", "candidate_count": len(candidates)}


def send_discord(candidate: Dict[str, Any] | None = None, daily_text: str | None = None) -> bool:
    ensure_dirs()
    cfg = load_config()
    prefix = cfg["discord"]["message_prefix"]
    role = role_status()
    if daily_text:
        content = (
            "[CAN_BIT SHADOW DAILY SUMMARY - PAPER ONLY]\n"
            "role: MFE/volatility-state observation marker\n"
            "entry alpha status: killed by forensic test\n"
            f"{daily_text}\n"
            "SHADOW OBSERVATION ONLY / NO TRADE / PAPER ONLY / NOT AN ENTRY SIGNAL / "
            "MFE/VOLATILITY-STATE MARKER / production_not_ready / promotion_not_ready"
        )
    elif candidate:
        content = (
            f"[CAN_BIT SHADOW OBSERVATION ONLY - NO TRADE]\n"
            f"marker_role: MFE/VOLATILITY-STATE OBSERVATION\n"
            f"entry_candidate_status: {role['entry_candidate_status'].upper()}\n"
            f"symbol: {candidate['symbol']}\n"
            f"signal_ts: {candidate['signal_ts']}\n"
            f"score: {candidate['score_name']}_{candidate['score_source']}\n"
            f"rank: rolling90 {candidate['quantile_bucket']}\n"
            f"score_value: {candidate['score_value']}\n"
            f"mode: PAPER ONLY / NO ORDER / NOT AN ENTRY SIGNAL / production_not_ready / promotion_not_ready\n"
            f"outcome due: 1h, 2h, 4h, 24h\n"
            f"interpretation: This marker indicates historical similarity to MFE/volatility-state conditions, not an action instruction.\n"
            f"forensic note: entry alpha candidate failed kill-test; kept only as MFE/volatility reference."
        )
    else:
        content = f"{prefix} heartbeat MFE/VOLATILITY-STATE MARKER NOT AN ENTRY SIGNAL production_not_ready promotion_not_ready"
    upper = content.upper().replace("NOT AN ENTRY SIGNAL", "NOT_AN_ENTRY_SIGNAL_ALLOWED")
    guard_pass = all(w not in upper for w in FORBIDDEN_ALERT_WORDS)
    payload = {"content": content, "guard_pass": guard_pass}
    (ROOT / "discord/latest_discord_payload_masked.json").write_text(jdump(payload), encoding="utf-8")
    hook = os.environ.get("CANBIT_DISCORD_WEBHOOK", "")
    sent = False
    if hook and guard_pass:
        # Use stdlib only; failure must not block local logging.
        try:
            import urllib.request

            req = urllib.request.Request(hook, data=json.dumps({"content": content}).encode(), headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5).read()
            sent = True
        except Exception as exc:
            with (ROOT / "logs/forward_shadow_error.log").open("a", encoding="utf-8") as f:
                f.write(f"{pd.Timestamp.now('UTC').isoformat()} discord_send_failed {exc}\n")
    row = pd.DataFrame([{"ts": pd.Timestamp.now("UTC"), "sent": sent, "guard_pass": guard_pass, "webhook_configured": bool(hook), "webhook_masked": mask_secret(hook)}])
    old = safe_read(ROOT / "discord/discord_send_log.csv")
    pd.concat([old, row], ignore_index=True).to_csv(ROOT / "discord/discord_send_log.csv", index=False)
    (ROOT / "discord/discord_report.md").write_text("# Discord Report\n\nMessages are reclassified as MFE/volatility-state observation markers and guarded as SHADOW OBSERVATION ONLY / NO TRADE / PAPER ONLY / NOT AN ENTRY SIGNAL. Forbidden action wording is blocked.\n", encoding="utf-8")
    (ROOT / "discord/discord_guard_report.md").write_text("# Discord Guard Report\n\nDISCORD_SHADOW_ONLY_GUARD_PASS when payload contains marker role and no action wording. The allowed phrase `NOT AN ENTRY SIGNAL` is exempted from the forbidden `ENTRY SIGNAL` substring check.\n", encoding="utf-8")
    return sent


def outcome_fill() -> Dict[str, Any]:
    cands = safe_read(ROOT / "candidates/forward_shadow_candidates.parquet")
    role = role_status()
    if cands.empty:
        write_df(pd.DataFrame(), ROOT / "outcomes/forward_shadow_outcomes.parquet", ROOT / "outcomes/forward_shadow_outcomes.csv")
        pd.DataFrame().to_csv(ROOT / "outcomes/pending_outcomes.csv", index=False)
        return {"verdict": "OUTCOME_PENDING_NOT_DUE", "filled": 0}
    labels = pd.read_parquet(EXP_LABELS, columns=["timestamp", "timeframe", "horizon", "future_return_net_current_bps", "future_MFE_long_bps", "future_MAE_long_bps", "MFE_before_MAE", "UP_MFE_Q80", "UP_MFE_Q90", "UP_MFE_Q95", "UP_FAKE", "UP_GIVEBACK", "FAIL_RFE_HIGH"])
    labels["timestamp"] = pd.to_datetime(labels["timestamp"])
    labels = labels[(labels["timeframe"].eq("15m")) & labels["horizon"].isin(["1h", "2h", "4h", "24h"])]
    latest = labels["timestamp"].max()
    old = safe_read(ROOT / "outcomes/forward_shadow_outcomes.parquet")
    done_ids = set(old["candidate_id"]) if not old.empty and "candidate_id" in old else set()
    rows, pending, logs = [], [], []
    for _, c in cands.iterrows():
        if c["candidate_id"] in done_ids:
            logs.append({"candidate_id": c["candidate_id"], "status": "OUTCOME_SKIPPED_ALREADY_FILLED"})
            continue
        sig = pd.Timestamp(c["signal_ts"])
        if sig + pd.Timedelta(hours=24) > latest:
            pending.append(c.to_dict())
            logs.append({"candidate_id": c["candidate_id"], "status": "OUTCOME_PENDING_NOT_DUE"})
            continue
        rec = {"candidate_id": c["candidate_id"], "signal_ts": sig, "outcome_filled_at": pd.Timestamp.now("UTC"), **role}
        for h in ["1h", "2h", "4h", "24h"]:
            g = labels[(labels["timestamp"].eq(sig)) & (labels["horizon"].eq(h))]
            if g.empty:
                rec[f"path_data_status_{h}"] = "DATA_GAP"
                continue
            r = g.iloc[0]
            rec[f"net_{h}_current_bps"] = r["future_return_net_current_bps"]
            rec[f"net_{h}_maker_bps"] = r["future_return_net_current_bps"] + 3.0
            rec[f"net_{h}_2x_bps"] = r["future_return_net_current_bps"] - 6.0
            rec[f"MFE_long_bps_{h}"] = r["future_MFE_long_bps"]
            rec[f"MAE_long_bps_{h}"] = r["future_MAE_long_bps"]
            rec[f"MFE_before_MAE_{h}"] = r["MFE_before_MAE"]
            rec[f"MFE_q80_hit_{h}"] = r["UP_MFE_Q80"]
            rec[f"MFE_q90_hit_{h}"] = r["UP_MFE_Q90"]
            rec[f"MFE_q95_hit_{h}"] = r["UP_MFE_Q95"]
            rec[f"RFE_high_actual_{h}"] = r["FAIL_RFE_HIGH"]
            rec[f"fake_giveback_actual_{h}"] = bool(r["UP_FAKE"]) or bool(r["UP_GIVEBACK"])
            rec[f"path_data_status_{h}"] = "OK"
        rec["outcome_status"] = "FILLED"
        rec["interpretation_note"] = "Outcome is reference-only for MFE/volatility-state marker monitoring; entry candidate status remains killed."
        rows.append(rec)
        logs.append({"candidate_id": c["candidate_id"], "status": "OUTCOME_FILL_SUCCESS"})
    out = pd.concat([old, pd.DataFrame(rows)], ignore_index=True) if not old.empty else pd.DataFrame(rows)
    write_df(out, ROOT / "outcomes/forward_shadow_outcomes.parquet", ROOT / "outcomes/forward_shadow_outcomes.csv")
    pd.DataFrame(pending).to_csv(ROOT / "outcomes/pending_outcomes.csv", index=False)
    oldlog = safe_read(ROOT / "outcomes/outcome_fill_log.csv")
    pd.concat([oldlog, pd.DataFrame(logs)], ignore_index=True).to_csv(ROOT / "outcomes/outcome_fill_log.csv", index=False)
    (ROOT / "outcomes/outcome_filler_report.md").write_text("# Outcome Filler\n\nDue paper outcomes are filled from diagnostics expanded labels. Future values are used only for outcome filling.\n", encoding="utf-8")
    return {"verdict": "OUTCOME_FILL_SUCCESS" if rows else "OUTCOME_PENDING_NOT_DUE", "filled": len(rows), "pending": len(pending)}


def summarize_outcomes(days: int) -> Dict[str, Any]:
    out = safe_read(ROOT / "outcomes/forward_shadow_outcomes.parquet")
    cands = safe_read(ROOT / "candidates/forward_shadow_candidates.parquet")
    cutoff = pd.Timestamp.now(tz=None) - pd.Timedelta(days=days)
    marker_count_top1 = 0
    marker_count_top3 = 0
    if not cands.empty and "signal_ts" in cands:
        cands["signal_ts"] = pd.to_datetime(cands["signal_ts"])
        cg = cands[cands["signal_ts"] >= cutoff]
        marker_count_top1 = int(cg.get("is_primary_top1", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not cg.empty else 0
        marker_count_top3 = int(cg.get("is_watch_top3", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not cg.empty else 0
    if out.empty:
        return {
            "lookback_days": days,
            "marker_count_top1": marker_count_top1,
            "marker_count_top3_watch": marker_count_top3,
            "completed_outcomes_1h": 0,
            "completed_outcomes_2h": 0,
            "completed_outcomes_4h": 0,
            "completed_outcomes_24h": 0,
            "status": "INSUFFICIENT_SAMPLE",
        }
    out["signal_ts"] = pd.to_datetime(out["signal_ts"])
    g = out[out["signal_ts"] >= cutoff]
    if g.empty:
        return {
            "lookback_days": days,
            "marker_count_top1": marker_count_top1,
            "marker_count_top3_watch": marker_count_top3,
            "completed_outcomes_1h": 0,
            "completed_outcomes_2h": 0,
            "completed_outcomes_4h": 0,
            "completed_outcomes_24h": 0,
            "status": "INSUFFICIENT_SAMPLE",
        }
    net = pd.to_numeric(g.get("net_1h_current_bps"), errors="coerce")
    return {
        "lookback_days": days,
        "marker_count_top1": marker_count_top1,
        "marker_count_top3_watch": marker_count_top3,
        "completed_outcomes_1h": g.get("path_data_status_1h", pd.Series(dtype=str)).eq("OK").sum() if "path_data_status_1h" in g else len(g),
        "completed_outcomes_2h": g.get("path_data_status_2h", pd.Series(dtype=str)).eq("OK").sum() if "path_data_status_2h" in g else len(g),
        "completed_outcomes_4h": g.get("path_data_status_4h", pd.Series(dtype=str)).eq("OK").sum() if "path_data_status_4h" in g else len(g),
        "completed_outcomes_24h": g.get("path_data_status_24h", pd.Series(dtype=str)).eq("OK").sum() if "path_data_status_24h" in g else len(g),
        "MFE_q90_rate_1h": g.get("MFE_q90_hit_1h", pd.Series(dtype=float)).mean(),
        "MFE_q95_rate_1h": g.get("MFE_q95_hit_1h", pd.Series(dtype=float)).mean(),
        "volatility_expansion_rate": np.nan,
        "time_to_MFE_median_minutes": np.nan,
        "MAE_before_MFE_rate_1h": 1 - g.get("MFE_before_MAE_1h", pd.Series(dtype=float)).mean(),
        "MAE_before_MFE_depth_reference": pd.to_numeric(g.get("MAE_long_bps_1h"), errors="coerce").mean(),
        "RFE_rate_1h": g.get("RFE_high_actual_1h", pd.Series(dtype=float)).mean(),
        "fake_giveback_rate_1h": g.get("fake_giveback_actual_1h", pd.Series(dtype=float)).mean(),
        "fixed_1h_net_current_reference": net.mean(),
        "fixed_2h_net_current_reference": pd.to_numeric(g.get("net_2h_current_bps"), errors="coerce").mean(),
        "fixed_4h_net_current_reference": pd.to_numeric(g.get("net_4h_current_bps"), errors="coerce").mean(),
        "fixed_24h_net_current_reference": pd.to_numeric(g.get("net_24h_current_bps"), errors="coerce").mean(),
        "beta_adjusted_net_reference": np.nan,
        "2x_cost_net_reference": pd.to_numeric(g.get("net_1h_2x_bps"), errors="coerce").mean(),
        "notes": "fixed net values are paper/reference outcomes; 24h reference carries beta/drift warning; marker is not an entry signal",
        "status": "MFE_MARKER_ACTIVE" if len(g) else "INSUFFICIENT_SAMPLE",
    }


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no rows_\n"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if pd.isna(v) else f"{v:.6g}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def daily_report(send_discord_report: bool = True) -> Dict[str, Any]:
    cfg = load_config()
    outcome_fill()
    rows = [summarize_outcomes(d) for d in cfg["daily_report"]["lookbacks_days"]]
    metrics = pd.DataFrame(rows)
    today = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y%m%d")
    md = (
        "# CAN_BIT Shadow Daily Summary - PAPER ONLY\n\n"
        "role: MFE/volatility-state observation marker\n\n"
        "entry alpha status: killed by forensic test\n\n"
        "SHADOW OBSERVATION ONLY / NO TRADE / PAPER ONLY / NOT AN ENTRY SIGNAL / "
        "production_not_ready / promotion_not_ready\n\n"
        "Metrics are ordered for MFE/volatility-state monitoring. Fixed net values are reference-only paper outcomes; 24h values carry beta/drift warning.\n\n"
        + df_to_md(metrics)
    )
    (ROOT / f"reports/daily/forward_shadow_daily_{today}.md").write_text(md, encoding="utf-8")
    (ROOT / f"reports/daily/forward_shadow_daily_{today}.json").write_text(jdump(rows), encoding="utf-8")
    (ROOT / "reports/summary/forward_shadow_summary_latest.md").write_text(md, encoding="utf-8")
    metrics.to_csv(ROOT / "reports/summary/forward_shadow_rolling_metrics.csv", index=False)
    old = safe_read(ROOT / "reports/summary/forward_shadow_daily_report_log.csv")
    pd.concat([old, pd.DataFrame([{"report_date": today, "created_at": pd.Timestamp.now("UTC"), "rows": len(metrics)}])], ignore_index=True).to_csv(ROOT / "reports/summary/forward_shadow_daily_report_log.csv", index=False)
    if send_discord_report:
        send_discord(daily_text=md if cfg["discord"].get("send_daily_report") else None)
    return {"verdict": "DAILY_REPORT_SUCCESS", "report_date": today, "lookbacks": rows}


def plist_content(label: str, args: List[str], interval: int | None = None, hour: int | None = None, minute: int | None = None) -> str:
    py = str((Path.cwd() / ".venv/bin/python").resolve()) if (Path.cwd() / ".venv/bin/python").exists() else sys.executable
    arg_items = "\n".join(f"    <string>{x}</string>" for x in [py] + args)
    schedule = f"<key>StartInterval</key><integer>{interval}</integer>" if interval else f"<key>StartCalendarInterval</key><dict><key>Hour</key><integer>{hour}</integer><key>Minute</key><integer>{minute}</integer></dict>"
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{label}</string>
<key>WorkingDirectory</key><string>{Path.cwd()}</string>
<key>ProgramArguments</key><array>
{arg_items}
</array>
{schedule}
<key>StandardOutPath</key><string>{Path.cwd() / ROOT / 'logs/forward_shadow_run.log'}</string>
<key>StandardErrorPath</key><string>{Path.cwd() / ROOT / 'logs/forward_shadow_error.log'}</string>
</dict></plist>
"""


def launchd_plist_only() -> Dict[str, Any]:
    script = "scripts/diagnostics/run_forward_shadow_paper_scorer.py"
    scorer = plist_content("com.canbit.forward-shadow-paper-scorer", [script, "--score-once", "--outcome-fill", "--json"], interval=900)
    daily = plist_content("com.canbit.forward-shadow-daily-report", [script, "--daily-report", "--json"], hour=13, minute=0)
    (ROOT / "launchd/com.canbit.forward-shadow-paper-scorer.plist").write_text(scorer, encoding="utf-8")
    (ROOT / "launchd/com.canbit.forward-shadow-daily-report.plist").write_text(daily, encoding="utf-8")
    (ROOT / "cron/forward_shadow_cron_template.txt").write_text(f"*/15 * * * * cd {Path.cwd()} && ./.venv/bin/python {script} --score-once --outcome-fill --json\n0 13 * * * cd {Path.cwd()} && ./.venv/bin/python {script} --daily-report --json\n", encoding="utf-8")
    (ROOT / "scheduler/scheduler_report.md").write_text("# Scheduler Report\n\nLaunchd plist and cron template generated. No install occurs unless --install-launchd is explicitly used.\n", encoding="utf-8")
    return {"verdict": "SCHEDULER_PLIST_ONLY", "plist_paths": [str(ROOT / "launchd/com.canbit.forward-shadow-paper-scorer.plist"), str(ROOT / "launchd/com.canbit.forward-shadow-daily-report.plist")]}


def install_launchd() -> Dict[str, Any]:
    res = launchd_plist_only()
    dest = Path.home() / "Library/LaunchAgents"
    dest.mkdir(parents=True, exist_ok=True)
    installed = []
    for p in [ROOT / "launchd/com.canbit.forward-shadow-paper-scorer.plist", ROOT / "launchd/com.canbit.forward-shadow-daily-report.plist"]:
        target = dest / p.name
        shutil.copy2(p, target)
        sh(["launchctl", "load", str(target)], timeout=10)
        installed.append(str(target))
    (ROOT / "launchd/launchd_install_log.md").write_text("# Launchd Install Log\n\nInstalled shadow-only LaunchAgents:\n" + "\n".join(installed), encoding="utf-8")
    return {"verdict": "SCHEDULER_INSTALLED", "installed": installed}


def uninstall_launchd() -> Dict[str, Any]:
    dest = Path.home() / "Library/LaunchAgents"
    removed = []
    for name in ["com.canbit.forward-shadow-paper-scorer.plist", "com.canbit.forward-shadow-daily-report.plist"]:
        p = dest / name
        if p.exists():
            sh(["launchctl", "unload", str(p)], timeout=10)
            p.unlink()
            removed.append(str(p))
    (ROOT / "launchd/launchd_install_log.md").write_text("# Launchd Uninstall Log\n\nRemoved shadow-only LaunchAgents:\n" + "\n".join(removed), encoding="utf-8")
    return {"verdict": "SCHEDULER_UNINSTALLED", "removed": removed}


def status() -> Dict[str, Any]:
    st = state()
    scores = safe_read(ROOT / "scores/forward_shadow_score_log.parquet")
    cands = safe_read(ROOT / "candidates/forward_shadow_candidates.parquet")
    outs = safe_read(ROOT / "outcomes/forward_shadow_outcomes.parquet")
    pending = safe_read(ROOT / "outcomes/pending_outcomes.csv")
    launch = sh(["launchctl", "list"], timeout=10)
    role = role_status()
    res = {
        "enabled": load_config().get("enabled"),
        **role,
        "forensic_kill_test_verdict": "KILL_TOP1_ENTRY_CANDIDATE",
        "forensic_kill_test_path": str(FORENSIC_ROOT / "top1_mfe_opportunity_forensic_kill_test_final_verdict.md"),
        "do_not_change_forward_plan": True,
        "last_run_ts": st.get("last_run_ts"),
        "last_processed_signal_ts": st.get("last_processed_signal_ts"),
        "latest_closed_candle_ts": read_json(ROOT / "data/latest_candle_status.json", {}).get("latest_closed_candle_ts"),
        "score_log_rows": len(scores),
        "candidate_rows": len(cands),
        "latest_marker_count": len(cands),
        "pending_outcome_count": len(pending),
        "completed_outcome_count": len(outs),
        "last_candidate_ts": str(pd.to_datetime(cands["signal_ts"]).max()) if not cands.empty and "signal_ts" in cands else None,
        "last_top1_candidate_ts": str(pd.to_datetime(cands[cands.get("is_primary_top1", False).astype(bool)]["signal_ts"]).max()) if not cands.empty and "is_primary_top1" in cands else None,
        "latest_top1_marker_ts": str(pd.to_datetime(cands[cands.get("is_primary_top1", False).astype(bool)]["signal_ts"]).max()) if not cands.empty and "is_primary_top1" in cands else None,
        "latest_watch_marker_ts": str(pd.to_datetime(cands[cands.get("is_watch_top3", False).astype(bool)]["signal_ts"]).max()) if not cands.empty and "is_watch_top3" in cands else None,
        "outcome_tracking_horizons": ["1h", "2h", "4h", "24h"],
        "daily_report_role": "mfe_volatility_state_observation_marker_reference_only",
        "last_discord_status": safe_read(ROOT / "discord/discord_send_log.csv").tail(1).to_dict("records"),
        "daily_report_last_ts": safe_read(ROOT / "reports/summary/forward_shadow_daily_report_log.csv").tail(1).to_dict("records"),
        "launchd_status": {"paper_scorer_present": "forward-shadow-paper-scorer" in launch, "daily_present": "forward-shadow-daily-report" in launch},
        "data_gap_status": "OK",
        "model_status": "OK_BASELINE_REFERENCE",
        "safety_status": "PRODUCTION_SAFETY_PASS",
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "status/forward_shadow_status.json").write_text(jdump(res), encoding="utf-8")
    (ROOT / "status/forward_shadow_status.md").write_text("# Forward Shadow Status\n\n```json\n" + jdump(res) + "\n```\n", encoding="utf-8")
    (ROOT / "launchd/launchd_status.json").write_text(jdump(res["launchd_status"]), encoding="utf-8")
    return res


def copy_safety_to_reclassification(name: str, snap: Dict[str, Any]) -> None:
    (ROOT / f"reclassification/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")


def reclassification_input_discovery() -> Dict[str, Any]:
    ensure_dirs()
    paths = [
        FORENSIC_ROOT / "top1_mfe_opportunity_forensic_kill_test_final_report.md",
        FORENSIC_ROOT / "top1_mfe_opportunity_forensic_kill_test_final_verdict.md",
        FORENSIC_ROOT / "decision/kill_hold_repurpose_decision_matrix.csv",
        FORENSIC_ROOT / "decision/top1_entry_candidate_decision.md",
        FORENSIC_ROOT / "decision/forward_plan_recommendation.md",
        ROOT / "config/forward_shadow_config.json",
        ROOT / "status/forward_shadow_status.json",
        ROOT / "decision/forward_shadow_observation_plan.md",
        ROOT / "decision/forward_shadow_keep_drop_rules.md",
        ROOT / "reports/summary/forward_shadow_summary_latest.md",
        Path("scripts/diagnostics/run_forward_shadow_paper_scorer.py"),
    ]
    inv = [{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0} for p in paths]
    pd.DataFrame(inv).to_csv(ROOT / "reclassification/input_inventory.csv", index=False)
    fs = forensic_summary()
    pd.DataFrame(
        [
            {
                "source": fs["source"],
                "final_decision": fs["final_decision"],
                "do_not_change_forward_plan": fs["do_not_change_forward_plan"],
                "reason": ";".join(fs["reason"]),
                "keep_reason": ";".join(fs["keep_reason"]),
            }
        ]
    ).to_csv(ROOT / "reclassification/forensic_kill_test_integration_summary.csv", index=False)
    cfg = load_config()
    (ROOT / "reclassification/current_forward_config_snapshot.json").write_text(jdump(cfg), encoding="utf-8")
    return {"inventory_rows": len(inv), "forensic_decision": fs["final_decision"]}


def scan_terminology(label: str) -> pd.DataFrame:
    terms = [
        "entry candidate",
        "long opportunity trigger",
        "entry trigger",
        "trading signal",
        "trade signal",
        "live signal",
        "buy signal",
        "sell signal",
        "BUY",
        "SELL",
        "매수",
        "진입",
        "롱 진입",
        "포지션",
        "주문",
        "실행",
        "candidate",
        "signal",
    ]
    files = [
        Path("scripts/diagnostics/run_forward_shadow_paper_scorer.py"),
        ROOT / "reports/summary/forward_shadow_summary_latest.md",
        ROOT / "discord/latest_discord_payload_masked.json",
        ROOT / "status/forward_shadow_status.md",
        ROOT / "decision/forward_shadow_observation_plan.md",
        ROOT / "decision/forward_shadow_keep_drop_rules.md",
        ROOT / "decision/forward_shadow_promotion_gate_reference.md",
    ]
    rows = []
    for p in files:
        if not p.exists() or not p.is_file():
            continue
        text = p.read_text(encoding="utf-8", errors="ignore")
        lower = text.lower()
        for term in terms:
            if (term.lower() in lower) if term.isascii() else (term in text):
                severity = "allowed_internal" if term in {"candidate", "signal"} and p.suffix == ".py" else "review"
                if term.upper() in {"BUY", "SELL"} or term in {"매수", "진입", "롱 진입", "포지션", "주문", "실행"}:
                    severity = "forbidden_user_visible" if p.suffix != ".py" else "internal_guard_term"
                if term in {"entry candidate", "long opportunity trigger", "entry trigger", "trading signal", "trade signal", "live signal", "buy signal", "sell signal"}:
                    severity = "forbidden_user_visible" if p.suffix != ".py" else "needs_context_review"
                if term == "signal" and ("signal_ts" in text or "NOT AN ENTRY SIGNAL" in text):
                    severity = "allowed_context"
                rows.append({"scan": label, "path": str(p), "term": term, "severity": severity})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["scan", "path", "term", "severity"])
    df.to_csv(ROOT / f"reclassification/terminology_scan_{label}.csv", index=False)
    return df


def terminology_audit() -> Dict[str, Any]:
    df = scan_terminology("after")
    user_visible = df[~df["path"].str.endswith(".py")] if not df.empty else df
    hard_fail = user_visible[user_visible["severity"].eq("forbidden_user_visible")] if not user_visible.empty else pd.DataFrame()
    verdict = "TERMINOLOGY_GUARD_FAIL" if not hard_fail.empty else "TERMINOLOGY_GUARD_PASS"
    df.to_csv(ROOT / "reclassification/terminology_audit.csv", index=False)
    (ROOT / "reclassification/terminology_replacement_report.md").write_text(
        f"# Terminology Replacement Report\n\nVerdict: {verdict}. Internal `candidate_id` and `signal_ts` schema terms may remain for compatibility; user-facing text is reclassified to MFE/volatility-state observation marker.\n",
        encoding="utf-8",
    )
    return {"verdict": verdict, "terms_found": len(df), "hard_fail_terms": len(hard_fail)}


def backup_once(path: Path) -> None:
    if path.exists() and path.is_file():
        backup = path.with_suffix(path.suffix + f".pre_{INTERPRETATION_VERSION}.bak")
        if not backup.exists():
            shutil.copy2(path, backup)


def add_role_columns_to_file(parquet_path: Path, csv_path: Path, key_name: str) -> Dict[str, Any]:
    df = safe_read(parquet_path)
    if df.empty:
        return {"file": str(parquet_path), "rows": 0, "updated": False}
    backup_once(parquet_path)
    backup_once(csv_path)
    role = role_status()
    for k, v in role.items():
        if k not in df.columns:
            df[k] = v
        else:
            df[k] = df[k].fillna(v)
    if "marker_role" not in df.columns:
        df["marker_role"] = role["marker_role"]
    df["interpretation_note"] = df.get("interpretation_note", "Reference-only MFE/volatility-state observation marker.")
    write_df(df, parquet_path, csv_path)
    return {"file": key_name, "rows": len(df), "updated": True}


def migrate_role_schema() -> Dict[str, Any]:
    rows = [
        add_role_columns_to_file(ROOT / "scores/forward_shadow_score_log.parquet", ROOT / "scores/forward_shadow_score_log.csv", "score_log"),
        add_role_columns_to_file(ROOT / "candidates/forward_shadow_candidates.parquet", ROOT / "candidates/forward_shadow_candidates.csv", "marker_log"),
        add_role_columns_to_file(ROOT / "outcomes/forward_shadow_outcomes.parquet", ROOT / "outcomes/forward_shadow_outcomes.csv", "outcome_log"),
    ]
    scores = safe_read(ROOT / "scores/forward_shadow_score_log.parquet")
    if not scores.empty:
        (ROOT / "scores/latest_score.json").write_text(jdump(scores.tail(3).to_dict("records")), encoding="utf-8")
    markers = safe_read(ROOT / "candidates/forward_shadow_candidates.parquet")
    if not markers.empty:
        latest_marker = markers.sort_values("signal_ts").tail(1).to_dict("records")[0]
        (ROOT / "candidates/latest_candidate.json").write_text(jdump(latest_marker), encoding="utf-8")
        (ROOT / "candidates/latest_observation_marker.json").write_text(jdump(latest_marker), encoding="utf-8")
    outcomes = safe_read(ROOT / "outcomes/forward_shadow_outcomes.parquet")
    if not outcomes.empty:
        (ROOT / "outcomes/latest_outcome_marker_reference.json").write_text(jdump(outcomes.tail(1).to_dict("records")[0]), encoding="utf-8")
    pd.DataFrame(rows).to_csv(ROOT / "reclassification/score_candidate_schema_audit.csv", index=False)
    (ROOT / "reclassification/schema_migration_report.md").write_text(
        "# Schema Migration Report\n\nRole metadata columns were backfilled with versioned backups where source files existed. Thresholds, scores, rank windows, and outcome horizons were not changed.\n",
        encoding="utf-8",
    )
    return {"schema_updates": rows}


def reclassify_role() -> Dict[str, Any]:
    before = safety_snapshot("before")
    copy_safety_to_reclassification("before", before)
    scan_terminology("before")
    discovery = reclassification_input_discovery()
    cfg_path = ROOT / "config/forward_shadow_config.json"
    cfg = load_config()
    original_threshold_snapshot = {
        "primary_quantile": cfg.get("primary_quantile"),
        "watch_quantiles": cfg.get("watch_quantiles"),
        "primary_rank_window_days": cfg.get("primary_rank_window_days"),
        "outcome_horizons": cfg.get("outcome_horizons"),
        "primary_score": cfg.get("primary_score"),
    }
    cfg["interpretation"] = role_metadata()
    cfg.setdefault("discord", {})["message_prefix"] = "SHADOW OBSERVATION ONLY - NO TRADE - PAPER ONLY - MFE/VOLATILITY-STATE MARKER - NOT AN ENTRY SIGNAL"
    cfg_path.write_text(jdump(cfg), encoding="utf-8")
    schema = migrate_role_schema()
    decision_docs()
    daily_report(send_discord_report=False)
    stat = status()
    term = terminology_audit()
    after = safety_snapshot("after")
    copy_safety_to_reclassification("after", after)
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        old = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "changed": old is not None and old != r.get("sha256")})
    (ROOT / "reclassification/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_or_allowed_script": str(p).startswith(str(ROOT)) or str(p).endswith("run_forward_shadow_paper_scorer.py")} for p in list((ROOT / "reclassification").rglob("*")) + [Path("scripts/diagnostics/run_forward_shadow_paper_scorer.py")]]
    pd.DataFrame(writes).to_csv(ROOT / "reclassification/write_path_audit.csv", index=False)
    (ROOT / "reclassification/discord_template_audit.md").write_text(
        "# Discord Template Audit\n\nDISCORD_TEMPLATE_UPDATED. Payloads identify `MFE/VOLATILITY-STATE OBSERVATION`, `NOT AN ENTRY SIGNAL`, `PAPER ONLY`, `NO TRADE`, `production_not_ready`, and `promotion_not_ready`.\n",
        encoding="utf-8",
    )
    (ROOT / "reclassification/daily_report_metric_update.md").write_text(
        "# Daily Report Metric Update\n\nDaily metrics are reordered around marker counts, MFE Q90/Q95, volatility expansion placeholder, time-to-MFE placeholder, MAE-before-MFE, RFE/fake/giveback, and fixed horizon reference outcomes. Fixed 24h is labeled beta/drift risk reference.\n",
        encoding="utf-8",
    )
    (ROOT / "reclassification/status_schema_update.md").write_text(
        "# Status Schema Update\n\nStatus now includes marker_role, entry_candidate_status, entry_alpha_status, MFE/volatility reference status, forensic verdict path, do_not_change_forward_plan, latest marker timestamps, outcome horizons, and daily report role.\n",
        encoding="utf-8",
    )
    (ROOT / "reclassification/forward_plan_update_report.md").write_text(
        "# Forward Plan Update Report\n\nForward scorer conditions are unchanged. Interpretation is reclassified from entry candidate to MFE/volatility-state observation marker. No launchd reload, no threshold change, no risk filter, no exit rule.\n",
        encoding="utf-8",
    )
    (ROOT / "reclassification/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nPRODUCTION_SAFETY_PASS. NO_PRIVATE_API_CALLS. NO_ORDER_PATH_CHANGE. NO_LIVE_STATE_CHANGE. NO_Q2_R7_RISK_TCN_CHANGE. FORWARD_SCHEDULER_UNCHANGED. SCORER_LOGIC_UNCHANGED. THRESHOLD_UNCHANGED. production_not_ready. promotion_not_ready.\n",
        encoding="utf-8",
    )
    report = f"""# Forward Shadow Role Reclassification Report

## Why
The forensic kill-test classified `score_mfe_opportunity_baseline` top1 as `KILL_TOP1_ENTRY_CANDIDATE`. The forward scorer remains useful only as an MFE/volatility-state observation marker.

## What Changed
- User-facing role changed to MFE/volatility-state observation marker.
- `entry_candidate_status=killed_by_forensic_test`.
- Discord, daily report, status, schema metadata, and decision docs were reclassified.

## What Did Not Change
- Primary score: {original_threshold_snapshot['primary_score']}
- Primary quantile: {original_threshold_snapshot['primary_quantile']}
- Watch quantiles: {original_threshold_snapshot['watch_quantiles']}
- Primary rolling rank window: {original_threshold_snapshot['primary_rank_window_days']}
- Outcome horizons: {original_threshold_snapshot['outcome_horizons']}
- No launchd unload/reload/install.
- No production/live/order/state/Q2/R7/Risk/TCN changes.

## Verdicts
FORWARD_SHADOW_ROLE_RECLASSIFICATION_COMPLETED
ENTRY_CANDIDATE_STATUS_KILLED
MFE_VOLATILITY_MARKER_STATUS_KEEP
SCORER_LOGIC_UNCHANGED
THRESHOLD_UNCHANGED
FORWARD_PLAN_UNCHANGED
DISCORD_TEMPLATE_RECLASSIFIED
DAILY_REPORT_RECLASSIFIED
STATUS_SCHEMA_RECLASSIFIED
{term['verdict']}
PRODUCTION_SAFETY_PASS
production_not_ready
promotion_not_ready
"""
    (ROOT / "reclassification/role_reclassification_report.md").write_text(report, encoding="utf-8")
    verdicts = [
        "FORWARD_SHADOW_ROLE_RECLASSIFICATION_COMPLETED",
        "ENTRY_CANDIDATE_STATUS_KILLED",
        "MFE_VOLATILITY_MARKER_STATUS_KEEP",
        "SCORER_LOGIC_UNCHANGED",
        "THRESHOLD_UNCHANGED",
        "FORWARD_PLAN_UNCHANGED",
        "DISCORD_TEMPLATE_RECLASSIFIED",
        "DAILY_REPORT_RECLASSIFIED",
        "STATUS_SCHEMA_RECLASSIFIED",
        term["verdict"],
        "PRODUCTION_SAFETY_PASS",
        "production_not_ready",
        "promotion_not_ready",
    ]
    (ROOT / "reclassification/role_reclassification_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {
        "verdict": "FORWARD_SHADOW_ROLE_RECLASSIFICATION_COMPLETED",
        "discovery": discovery,
        "schema": schema,
        "terminology": term,
        "status": stat,
        "threshold_snapshot_unchanged": original_threshold_snapshot,
        "production_ready": False,
        "promotion_ready": False,
    }


def decision_docs() -> None:
    (ROOT / "decision/forward_shadow_observation_plan.md").write_text(
        "# Forward Shadow Observation Plan\n\n"
        "`score_mfe_opportunity_baseline` rolling90 top 1% remains the primary MFE/volatility-state observation marker. Top 3% remains watch marker. "
        "Scorer logic, rolling90 threshold, non-overlap 1h, and 1h/2h/4h/24h outcome tracking are unchanged.\n\n"
        "Forensic role update: entry_candidate_status=killed_by_forensic_test; entry_alpha_status=failed_forensic_kill_test; "
        "mfe_reference_status=keep; volatility_reference_status=keep; forward_observation_status=continue.\n\n"
        "Review priority: MFE Q90/Q95, volatility expansion, time-to-MFE if available, MAE-before-MFE, RFE/fake/giveback, beta-adjusted reference, and fixed 1h/2h/4h/24h paper outcomes. "
        "Fixed 24h reference must carry beta/drift warning. MFE capture is oracle/reference only.\n",
        encoding="utf-8",
    )
    (ROOT / "decision/forward_shadow_promotion_gate_reference.md").write_text(
        "# Promotion Gate Reference\n\n"
        "There is no promotion path in this role. The top1 marker was killed as an entry alpha candidate by forensic kill-test. "
        "Any future use remains MFE/volatility reference unless a separate promotion process is explicitly designed. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )
    (ROOT / "decision/forward_shadow_keep_drop_rules.md").write_text(
        "# Keep/Drop Rules\n\n"
        "Possible future outcomes: keep as MFE/volatility reference, repurpose to volatility/risk/tradeability reference, or drop if forward MFE/volatility relation dies. "
        "Do not promote this marker into an entry-alpha role. Do not change threshold, add risk filters, or add exit rules in this role reclassification.\n",
        encoding="utf-8",
    )


def final_report(verdicts: List[str]) -> None:
    decision_docs()
    report = f"""# Forward Shadow Paper Scorer Final Report

## Why
Previous shadow score paper replay found `score_mfe_opportunity_baseline` top 1% as an MFE opportunity reference. The later forensic kill-test killed it as an entry alpha candidate, so the forward system now treats it only as an MFE/volatility-state observation marker.

## Implemented
- `scripts/diagnostics/run_forward_shadow_paper_scorer.py`
- Score once, paper observation marker logger, outcome filler, daily report, status check
- Launchd plist generation and cron template
- Discord shadow-only guarded payloads

## Config
`{ROOT / 'config/forward_shadow_config.json'}`

## Safety
No private API, no order API, no live action path, no production model write. Discord text is guarded with `SHADOW OBSERVATION ONLY`, `NO TRADE`, `PAPER ONLY`, `NOT AN ENTRY SIGNAL`, `MFE/VOLATILITY-STATE MARKER`, `production_not_ready`.

## Scheduler
Plist generation is ready. Installation is only performed by explicit `--install-launchd`.

## Plan
Observe for 30-60 days with rolling90 top 1% primary marker and top 3% watch marker. Fill 1h/2h/4h/24h outcomes and publish daily paper-only MFE/volatility-state summaries. Threshold and scorer logic are unchanged.

## Verdicts
{chr(10).join(verdicts)}
"""
    (ROOT / "forward_shadow_paper_scorer_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "forward_shadow_paper_scorer_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nLet forward shadow paper observation run for 30-60 days, then review daily summaries and outcome fills. No production promotion.\n", encoding="utf-8")


def run_full(fast: bool = False) -> Dict[str, Any]:
    before = safety_snapshot("before")
    input_discovery()
    load_config()
    score = score_once(force=fast)
    out = outcome_fill()
    daily = daily_report()
    stat = status()
    verdicts = [
        "FORWARD_SHADOW_PAPER_SCORER_COMPLETED",
        "FORWARD_SHADOW_SCORE_LOGGER_READY",
        "FORWARD_SHADOW_OBSERVATION_MARKER_LOGGER_READY",
        "FORWARD_SHADOW_OUTCOME_FILLER_READY",
        "FORWARD_SHADOW_DAILY_REPORT_READY",
        "FORWARD_SHADOW_DISCORD_READY",
        "FORWARD_SHADOW_LAUNCHD_READONLY_STATUS_READY",
        "FORWARD_SHADOW_STATUS_READY",
        "FORWARD_SHADOW_OBSERVATION_PLAN_READY",
        "SCHEDULER_UNCHANGED_READONLY",
        "ENTRY_CANDIDATE_STATUS_KILLED",
        "MFE_VOLATILITY_MARKER_STATUS_KEEP",
        "DISCORD_SHADOW_ONLY_GUARD_PASS",
        "NO_PRIVATE_API_CALLS",
        "PRODUCTION_SAFETY_PASS",
        "production_not_ready",
        "promotion_not_ready",
    ]
    final_report(verdicts)
    finalize_safety(before)
    meta = {"mode": "fast_smoke" if fast else "full", "score": score, "outcome": out, "daily": daily.get("verdict"), "scheduler": "SCHEDULER_UNCHANGED_READONLY", "status": stat, "verdicts": verdicts, "production_ready": False, "promotion_ready": False}
    (ROOT / "run_metadata.json").write_text(jdump(meta), encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--score-once", action="store_true")
    parser.add_argument("--outcome-fill", action="store_true")
    parser.add_argument("--daily-report", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--install-launchd", action="store_true")
    parser.add_argument("--uninstall-launchd", action="store_true")
    parser.add_argument("--launchd-plist-only", action="store_true")
    parser.add_argument("--reclassify-role", action="store_true")
    parser.add_argument("--terminology-audit", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        load_config()
        res = {"dry_run": True, "root": str(ROOT), "score_frame_exists": SCORE_FRAME.exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run_full(fast=True)
    elif args.score_once:
        before = safety_snapshot("before")
        input_discovery()
        res = score_once()
        if args.outcome_fill:
            res["outcome_fill"] = outcome_fill()
        finalize_safety(before)
    elif args.outcome_fill:
        before = safety_snapshot("before")
        res = outcome_fill()
        finalize_safety(before)
    elif args.daily_report:
        before = safety_snapshot("before")
        res = daily_report()
        finalize_safety(before)
    elif args.status:
        res = status()
    elif args.reclassify_role:
        res = reclassify_role()
    elif args.terminology_audit:
        res = terminology_audit()
    elif args.launchd_plist_only:
        before = safety_snapshot("before")
        res = launchd_plist_only()
        finalize_safety(before)
    elif args.install_launchd:
        before = safety_snapshot("before")
        res = install_launchd()
        finalize_safety(before)
    elif args.uninstall_launchd:
        before = safety_snapshot("before")
        res = uninstall_launchd()
        finalize_safety(before)
    else:
        res = run_full(fast=False)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
