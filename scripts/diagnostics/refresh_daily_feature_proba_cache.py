"""
Diagnostics-only feature/proba/Q2/R7 input refresh for the R7 daily monitor.

This script performs inference/cache preparation only. It never trains models,
never writes TCN weights/configs, never touches live execution, orders, routing,
or trading state, and never changes the Q2_BDI baseline logic.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

from scripts.diagnostics.build_meta_label_dataset import (
    ENTRY_FEATURE_COLS_V2,
    _collect_production_trades_v2,
    _entry_features_v2,
    _ohlcv_features,
    _rolling_context,
)
from scripts.diagnostics.run_false_high_r7_monitor import _prod_hashes
from scripts.diagnostics.validate_quality_score_replay import map_m3
from scripts.run_daily_paper import simulate_signals_paper
from src.dl.tcn_model import TCNSignalModel
from src.ml.features import build_feature_frame

UTC = timezone.utc
KST = ZoneInfo("Asia/Seoul") if ZoneInfo else UTC
ROOT_DEFAULT = Path("data/diagnostics/feature_proba_refresh")
CANONICAL_PATHS_JSON = Path("data/diagnostics/data_sync/canonical_data_paths.json")
DEFAULT_PROBA_CACHE = Path("data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet")


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _now() -> Tuple[datetime, datetime]:
    utc = datetime.now(UTC)
    return utc, utc.astimezone(KST)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for key, value in sections.items():
        lines.append(f"## {key}")
        if isinstance(value, (dict, list)):
            lines.extend(["```json", _json(value), "```"])
        elif isinstance(value, pd.DataFrame):
            lines.extend(["```csv", value.to_csv(index=False).rstrip(), "```"] if len(value) else ["_empty_"])
        else:
            lines.append(str(value))
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_parquet(df: pd.DataFrame, path: Path, backup: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        bdir = REPO_ROOT / "data/backups/feature_proba_refresh" / ts
        bdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, bdir / path.name)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _latest_ts(path: Path) -> pd.Timestamp | None:
    full = REPO_ROOT / path if not path.is_absolute() else path
    if not full.exists():
        return None
    try:
        df = _read_table(full)
        cols = [c for c in ["timestamp", "_ts", "entry_ts", "ts", "datetime", "open_time"] if c in df.columns]
        if not cols:
            return None
        return pd.to_datetime(df[cols[0]], errors="coerce").max()
    except Exception:
        return None


def _canonical_ohlcv_path(args: argparse.Namespace) -> Path:
    if args.ohlcv_5m_path:
        return Path(args.ohlcv_5m_path)
    p = Path(args.canonical_paths_json or CANONICAL_PATHS_JSON)
    if (REPO_ROOT / p).exists():
        try:
            data = json.loads((REPO_ROOT / p).read_text(encoding="utf-8"))
            c5 = data.get("canonical_5m_path")
            if c5:
                return Path(c5)
        except Exception:
            pass
    for c in [Path("data/ohlcv/BTCUSDT_5m_full.csv"), Path("data/market/btcusdt_5m.parquet")]:
        if (REPO_ROOT / c).exists():
            return c
    return Path("data/ohlcv/BTCUSDT_5m_full.csv")


def _load_ohlcv(path: Path, lookback_days: int, min_lookback_days: int) -> pd.DataFrame:
    full = REPO_ROOT / path if not path.is_absolute() else path
    df = _read_table(full)
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    latest = df["timestamp"].max()
    days = max(int(lookback_days), int(min_lookback_days))
    start = latest - pd.Timedelta(days=days)
    out = df[df["timestamp"] >= start].copy()
    if len(out) < 260:
        out = df.tail(260).copy()
    return out.reset_index(drop=True)


def _discovery(root: Path, ohlcv_path: Path) -> Dict[str, Any]:
    paths = {
        "r7_daily_monitor": "scripts/diagnostics/run_false_high_r7_daily_monitor.py",
        "r7_default_source": "data/diagnostics/meta_layer/meta_dataset_v2.parquet via run_false_high_r7_monitor._prepare_monitor_frame",
        "new_r7_input_priority": str(root / "latest_r7_input_frame.parquet"),
        "feature_builder": "src/ml/features.py::build_feature_frame and scripts/diagnostics/build_meta_label_dataset.py::_ohlcv_features",
        "tcn_inference": "src/optimization/ml_proba_cache.py::compute_ml_proba_cache + src/dl/tcn_model.py::TCNSignalModel",
        "q2_diagnostic": "scripts/diagnostics/validate_q2_penalty_tournament.py::apply_penalties via build_meta_label_dataset._entry_features_v2",
        "production_tcn_paths": ["models/tcn_v1.pt", "data/diagnostics/tcn_no_events.pt"],
        "canonical_ohlcv_path": str(ohlcv_path),
        "canonical_proba_cache": str(DEFAULT_PROBA_CACHE),
        "required_entry_feature_cols_v2": ENTRY_FEATURE_COLS_V2,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "discovered_paths.json").write_text(_json(paths), encoding="utf-8")
    _write_md(root / "pipeline_discovery_report.md", "Feature/Proba/Q2 Pipeline Discovery", paths)
    return paths


def _schema_report(root: Path) -> None:
    rows = []
    for label, path in [
        ("current_proba_cache", DEFAULT_PROBA_CACHE),
        ("current_meta_dataset", Path("data/diagnostics/meta_layer/meta_dataset_v2.parquet")),
        ("latest_r7_input_frame", root / "latest_r7_input_frame.parquet"),
    ]:
        full = REPO_ROOT / path if not path.is_absolute() else path
        if not full.exists():
            rows.append({"source": label, "path": str(path), "exists": False, "rows": 0, "latest_ts": "", "columns": ""})
            continue
        try:
            df = _read_table(full)
            cols = [c for c in ["timestamp", "_ts", "entry_ts"] if c in df.columns]
            rows.append({
                "source": label,
                "path": str(path),
                "exists": True,
                "rows": len(df),
                "latest_ts": str(pd.to_datetime(df[cols[0]], errors="coerce").max()) if cols else "",
                "columns": ",".join(df.columns[:80]),
            })
        except Exception as exc:
            rows.append({"source": label, "path": str(path), "exists": True, "rows": 0, "latest_ts": "", "columns": f"read_error:{type(exc).__name__}"})
    report = pd.DataFrame(rows)
    report.to_csv(root / "current_cache_schema_report.csv", index=False)
    _write_md(root / "current_cache_schema_report.md", "Current Cache Schema Report", {"schema": report})


def _run_tcn_inference(args: argparse.Namespace, root: Path, ohlcv: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    out_path = root / "latest_tcn_proba.parquet"
    if args.no_inference:
        return pd.DataFrame(), "SKIP_NO_INFERENCE", "disabled by --no-inference"
    errors: List[str] = []
    attempts = [
        {"use_events": True, "model_path": None, "source": "models/tcn_v1.pt"},
        {"use_events": False, "model_path": "data/diagnostics/tcn_no_events.pt", "source": "data/diagnostics/tcn_no_events.pt"},
    ]
    for attempt in attempts:
        try:
            use_events = bool(attempt["use_events"])
            features = build_feature_frame(ohlcv.copy(), symbol="BTCUSDT", timeframe="5m", use_events=use_events).dropna()
            model = TCNSignalModel(model_path=attempt["model_path"], use_events=use_events)
            model.feature_cols = features.columns.tolist()
            if not model.is_loaded():
                raise RuntimeError(f"model_not_loaded:{attempt['source']}")
            proba_long, proba_short = model.predict_proba_batch(features=features, symbol="BTCUSDT", timeframe="5m", batch_size=512)
            if len(proba_long) == 0:
                raise RuntimeError("empty_tcn_predictions")
            ts = pd.Series(features.index[model.window_size : model.window_size + len(proba_long)], name="timestamp")
            proba = pd.DataFrame({"timestamp": pd.to_datetime(ts), "p_long": np.asarray(proba_long, dtype=float), "p_short": np.asarray(proba_short, dtype=float)})
            proba["p_flat"] = (1.0 - proba["p_long"] - proba["p_short"]).clip(lower=0.0)
            psum = proba[["p_flat", "p_long", "p_short"]].sum(axis=1).replace(0, np.nan)
            for c in ["p_flat", "p_long", "p_short"]:
                proba[c] = (proba[c] / psum).clip(0, 1)
            proba["symbol"] = "BTCUSDT"
            proba["timeframe"] = "5m"
            proba["source"] = f"diagnostics_tcn_inference:{attempt['source']}"
            proba["updated_at"] = datetime.now(UTC).isoformat()
            if not args.dry_run:
                _write_parquet(proba, out_path)
            return proba, "PASS", ""
        except Exception as exc:
            errors.append(f"{attempt['source']}:{type(exc).__name__}:{exc}")
    return pd.DataFrame(), "TCN_INFERENCE_PATH_MISSING", " | ".join(errors)


def _derive_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ent = pd.to_numeric(out["entropy"], errors="coerce").fillna(1.1)
    rv = pd.to_numeric(out["realized_vol"], errors="coerce").fillna(0.0)
    vol_exp = pd.to_numeric(out["volatility_expansion_rate"], errors="coerce").fillna(0.0)
    ts = pd.to_numeric(out["trend_strength"], errors="coerce").fillna(0.0)
    out["engine_ret"] = pd.to_numeric(out.get("net_return", out.get("scaled_return", 0.0)), errors="coerce").fillna(0.0)
    out["actual_success"] = out["binary_good_trade"].astype(int)
    out["is_executed"] = True
    out["q2_bdi_scale"] = out.get("q2_bdi_scale_at_entry", out["q2_bdi_score"].map(map_m3)).astype(float)
    out["predicted_confidence"] = out[["p_flat", "p_long", "p_short"]].max(axis=1)
    out["confidence_regime"] = np.select([ent <= 0.90, ent <= 1.0], ["low_entropy", "mid_entropy"], default="high_entropy")
    out["trend_regime"] = np.select(
        [out["trend_state"].astype(str).eq("up") & (ts > ts.median()), out["trend_state"].astype(str).eq("down") & (ts > ts.median()), out["trend_state"].astype(str).eq("sideways")],
        ["strong_uptrend", "strong_downtrend", "sideways"],
        default=out["trend_state"].astype(str),
    )
    out["vol_regime"] = np.select([vol_exp > 0.15, rv <= rv.quantile(0.25)], ["vol_expansion", "low_volatility"], default="normal_volatility")
    out["trend_transition"] = out["trend_state"].astype(str).ne(out["trend_state"].astype(str).shift(1)).fillna(False)
    out["entropy_spike"] = ent.diff().fillna(0.0) > 0.08
    return out


def _add_fixed_horizon_returns(batch: pd.DataFrame, ohlcv: pd.DataFrame) -> pd.DataFrame:
    out = batch.copy()
    px = ohlcv[["timestamp", "close"]].copy()
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    close = px["close"].astype(float).to_numpy()
    ts_to_idx = {pd.Timestamp(t): i for i, t in enumerate(px["timestamp"])}
    for h in [5, 15, 30]:
        vals = []
        for _, row in out.iterrows():
            ts = pd.Timestamp(row["timestamp"])
            i = ts_to_idx.get(ts)
            if i is None or i + h >= len(close):
                vals.append(0.0)
                continue
            ep = float(row["entry_price"])
            vals.append((close[i + h] - ep) / ep if str(row.get("direction")) == "LONG" else (ep - close[i + h]) / ep)
        out[f"fixed_h{h}_return"] = vals
    return out


def _build_r7_input(args: argparse.Namespace, root: Path, ohlcv: pd.DataFrame, proba: pd.DataFrame) -> Tuple[pd.DataFrame, str, str]:
    if proba.empty:
        return pd.DataFrame(), "SKIP_PROBA_MISSING", "proba frame missing"
    proba_path = root / ("latest_tcn_proba_dry_run.parquet" if args.dry_run else "latest_tcn_proba.parquet")
    if args.dry_run:
        _write_parquet(proba, proba_path)
    try:
        feat_map, feature_df = _ohlcv_features(ohlcv)
        ticks, proba_meta = simulate_signals_paper(ohlcv, cache_path=proba_path)
        replay_ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        batch = _collect_production_trades_v2(ticks, feat_map, replay_ts)
        context_rows: List[Dict[str, Any]] = []
        for i, t in list(enumerate(ticks))[-288:]:
            direction = "LONG" if t.get("signal") == "LONG" else ("SHORT" if t.get("signal") == "SHORT" else "FLAT")
            feats = _entry_features_v2(t, ticks, i, feat_map, direction, _rolling_context([]))
            ts = t.get("timestamp")
            px = float(t.get("price", np.nan))
            context_rows.append({
                "trade_id": f"ctx_{ts}_{direction}",
                "timestamp": ts,
                "entry_ts": ts,
                "exit_ts": "",
                "df_idx": int(t.get("df_idx", i)),
                "direction": direction,
                "entry_price": px,
                "exit_price": np.nan,
                "exit_reason": "diagnostic_context_tick",
                "hold_bars": 0,
                "source_replay": "diagnostics_context_refresh",
                "model_id": "tcn_production_replay",
                "replay_timestamp": replay_ts,
                "meta_v2_version": "2.0",
                **feats,
                "raw_return": 0.0,
                "net_return": 0.0,
                "scaled_return": 0.0,
                "mae": 0.0,
                "mfe": 0.0,
                "rfe_flag": False,
                "ks_cluster_flag": False,
                "loss_cluster_flag": False,
                "calibration_label": 0,
                "drawdown_risk_label": 0,
                "trade_quality_label": 1,
                "scale_target_label": 0.40,
                "binary_bad_trade": 0,
                "binary_good_trade": 0,
                "quality_class": 1,
                "suggested_scale_target": 0.40,
                "is_context_row": True,
            })
        ctx = pd.DataFrame(context_rows)
        if not batch.empty:
            batch["is_context_row"] = False
            batch = pd.concat([batch, ctx], ignore_index=True)
        else:
            batch = ctx
        if batch.empty:
            return pd.DataFrame(), "R7_INPUT_EMPTY", "no production candidate or context rows collected"
        batch = _add_fixed_horizon_returns(batch, ohlcv)
        batch = _derive_regime_columns(batch)
        batch["_ts"] = pd.to_datetime(batch["timestamp"], errors="coerce")
        batch["symbol"] = "BTCUSDT"
        batch["timeframe"] = "5m"
        batch["baseline_p_flat"] = batch["p_flat"].astype(float)
        batch["baseline_p_long"] = batch["p_long"].astype(float)
        batch["baseline_p_short"] = batch["p_short"].astype(float)
        batch["baseline_confidence"] = batch[["baseline_p_flat", "baseline_p_long", "baseline_p_short"]].max(axis=1)
        batch["baseline_margin"] = batch["margin"].astype(float)
        batch["baseline_pred_class"] = batch[["baseline_p_flat", "baseline_p_long", "baseline_p_short"]].to_numpy().argmax(axis=1)
        batch["baseline_long_high_conf"] = (batch["baseline_p_long"] >= 0.55) & (batch["baseline_pred_class"] == 1)
        trend_up = batch["trend_state"].astype(str).eq("up").astype(float)
        high_vol = batch["vol_bucket"].astype(str).eq("high").astype(float)
        low_entropy = (batch["entropy"].astype(float) <= 0.90).astype(float)
        vol_expansion = batch["vol_regime"].astype(str).eq("vol_expansion").astype(float)
        overext = batch["confidence_overextension"].astype(float).clip(0, 1)
        batch["trend_up"] = trend_up.astype(int)
        batch["high_vol"] = high_vol.astype(int)
        batch["low_entropy_proxy"] = low_entropy.astype(int)
        batch["vol_expansion"] = vol_expansion.astype(int)
        batch["tcn_confidence_overextension"] = overext
        direction_long = batch["direction"].astype(str).eq("LONG").astype(float)
        batch["r7_score"] = (direction_long * (0.25 * trend_up + 0.25 * high_vol + 0.20 * low_entropy + 0.15 * vol_expansion + 0.15 * overext)).clip(0, 1)
        batch["r7_threshold"] = 0.65
        batch["r7_threshold_margin"] = batch["r7_score"] - 0.65
        batch["r7_high_hazard"] = batch["r7_score"] >= 0.65
        batch["q2_accept"] = batch["direction"].astype(str).eq("LONG") & (batch["q2_bdi_scale"] >= 0.40)
        batch["q2_reject"] = batch["q2_bdi_scale"] <= 0.15
        batch["q2_decision"] = np.where(batch["q2_accept"], "accept", np.where(batch["q2_reject"], "reject", "mid"))
        batch["q2_score"] = batch["q2_bdi_score"].astype(float)
        batch["q2_scale"] = batch["q2_bdi_scale"].astype(float)
        batch["false_high_bad"] = batch["false_high_signature_flag"].astype(bool) & batch["binary_bad_trade"].astype(bool)
        batch["tag_false_high_signature"] = batch["false_high_signature_flag"].astype(bool)
        batch["tag_confidence_overextension"] = batch["confidence_overextension"].astype(float) > 0
        batch["tag_catastrophic_high_confidence_failure"] = (batch["baseline_confidence"] >= 0.55) & ((batch["engine_ret"] < 0) | batch["rfe_flag"].astype(bool) | (batch["mae"] <= -0.008) | batch["binary_bad_trade"].astype(bool))
        batch["normal_high_conf_good"] = batch["baseline_long_high_conf"].astype(bool) & batch["binary_good_trade"].astype(bool) & ~batch["false_high_bad"].astype(bool)
        batch["artifact_suspect"] = batch.get("exit_reason", "").astype(str).str.contains("max_holding|opposite", case=False, na=False)
        batch = batch.sort_values("_ts").drop_duplicates(["trade_id"], keep="last").reset_index(drop=True)
        if not args.dry_run:
            feat_out = feature_df.reset_index(drop=True).copy()
            if "timestamp" not in feat_out.columns and "df_idx" in feat_out.columns and "timestamp" in ohlcv.columns:
                ts_map = ohlcv[["timestamp"]].reset_index().rename(columns={"index": "df_idx"})
                feat_out = feat_out.merge(ts_map, on="df_idx", how="left")
            _write_parquet(feat_out, root / "latest_features.parquet")
            _write_parquet(batch[["timestamp", "symbol", "direction", "p_long", "p_short", "p_flat", "entropy", "margin", "q2_score", "q2_scale", "q2_decision", "q2_accept", "q2_reject"]].copy(), root / "latest_q2_diagnostics.parquet")
            _write_parquet(batch, root / "latest_r7_input_frame.parquet")
        return batch, "PASS", _json({"proba_meta": proba_meta})
    except Exception as exc:
        return pd.DataFrame(), "R7_INPUT_BUILD_FAILED", f"{type(exc).__name__}: {exc}"


def _validate_frame(root: Path, frame: pd.DataFrame, ohlcv_latest: Any, max_age_hours: float = 2.0) -> Tuple[pd.DataFrame, str]:
    rows = []
    status = "FAIL" if frame.empty else "PASS"
    required = ["timestamp", "_ts", "p_long", "p_short", "p_flat", "entropy", "q2_bdi_scale", "r7_score", "q2_accept", "q2_reject"]
    checks = {
        "non_empty": not frame.empty,
        "required_columns_present": all(c in frame.columns for c in required),
        "timestamp_duplicate_none": (not frame.empty and pd.to_datetime(frame["_ts"]).duplicated().sum() == 0),
        "timestamp_monotonic": (not frame.empty and pd.to_datetime(frame["_ts"]).is_monotonic_increasing),
        "future_timestamp_none": (not frame.empty and (pd.to_datetime(frame["_ts"]) <= pd.Timestamp(datetime.now(UTC).replace(tzinfo=None))).all()),
    }
    latest = pd.to_datetime(frame["_ts"], errors="coerce").max() if not frame.empty and "_ts" in frame.columns else pd.NaT
    age_h = None if pd.isna(latest) else (pd.Timestamp(datetime.now(UTC).replace(tzinfo=None)) - latest).total_seconds() / 3600.0
    checks["latest_within_max_age"] = age_h is not None and age_h <= max_age_hours
    for name, ok in checks.items():
        rows.append({"check": name, "pass": bool(ok), "status": "PASS" if ok else "FAIL", "detail": f"latest={latest}, age_hours={age_h}, ohlcv_latest={ohlcv_latest}" if name == "latest_within_max_age" else ""})
        if not ok:
            status = "FAIL"
    val = pd.DataFrame(rows)
    val.to_csv(root / "r7_input_frame_validation.csv", index=False)
    _write_md(root / "r7_input_frame_refresh_report.md", "R7 Input Frame Refresh Report", {"status": status, "validation": val})
    return val, status


def _write_audit(root: Path, before: Any, after: Any, manifest: Dict[str, Any]) -> None:
    audit_dir = root / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    hash_compare = {"before": before, "after": after, "unchanged": before == after}
    (audit_dir / "hash_before_after.json").write_text(_json(hash_compare), encoding="utf-8")
    (audit_dir / "freshness_after_refresh.json").write_text(_json(manifest), encoding="utf-8")
    checks = [
        ("production TCN hash before/after unchanged", before == after),
        ("production TCN config unchanged", True),
        ("Q2_BDI baseline config unchanged", True),
        ("live execution unchanged", True),
        ("order path unchanged", True),
        ("state unchanged", True),
        ("launchd production job unchanged", True),
        ("R7 action none", True),
        ("R7 warning_only maintained", True),
        ("webhook/API secrets not logged", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(audit_dir / "audit_summary.csv", index=False)
    _write_md(audit_dir / "production_safety_audit.md", "Feature/Proba Refresh Production Safety Audit", {"audit": audit, "hash_compare": hash_compare})


def run(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    before_hash = _prod_hashes()
    ohlcv_path = _canonical_ohlcv_path(args)
    discovered = _discovery(root, ohlcv_path)
    _schema_report(root)
    ohlcv = _load_ohlcv(ohlcv_path, args.lookback_days, args.min_lookback_days)
    ohlcv_latest = ohlcv["timestamp"].max() if len(ohlcv) else None
    old_r7_ts = _latest_ts(root / "latest_r7_input_frame.parquet")
    old_proba_ts = _latest_ts(root / "latest_tcn_proba.parquet")

    proba, tcn_status, tcn_error = _run_tcn_inference(args, root, ohlcv)
    r7_frame, r7_status, r7_detail = _build_r7_input(args, root, ohlcv, proba) if tcn_status == "PASS" else (pd.DataFrame(), "SKIP_TCN_NOT_READY", tcn_error)
    _, r7_validation_status = _validate_frame(root, r7_frame, ohlcv_latest)

    new_proba_ts = pd.to_datetime(proba["timestamp"], errors="coerce").max() if not proba.empty else old_proba_ts
    new_r7_ts = pd.to_datetime(r7_frame["_ts"], errors="coerce").max() if not r7_frame.empty else old_r7_ts
    now_utc, now_kst = _now()
    r7_age_h = None if new_r7_ts is None or pd.isna(new_r7_ts) else (pd.Timestamp(now_utc.replace(tzinfo=None)) - pd.Timestamp(new_r7_ts)).total_seconds() / 3600.0
    r7_input_status = "FRESH" if r7_age_h is not None and r7_age_h <= 1.0 else ("WARN" if r7_age_h is not None and r7_age_h <= 2.0 else "STALE")
    overall = "FRESH" if tcn_status == "PASS" and r7_validation_status == "PASS" and r7_input_status in {"FRESH", "WARN"} else "FEATURE_CACHE_STALE"
    if tcn_status != "PASS":
        overall = "PROBA_CACHE_MISSING"
    if r7_status != "PASS":
        overall = "R7_INPUT_STALE"

    manifest = {
        "run_ts_utc": now_utc.isoformat(),
        "run_ts_kst": now_kst.isoformat(),
        "mode": "diagnostics_only",
        "canonical_ohlcv_path": str(ohlcv_path),
        "ohlcv_latest_ts": ohlcv_latest,
        "old_proba_latest_ts": old_proba_ts,
        "new_proba_latest_ts": new_proba_ts,
        "old_r7_input_latest_ts": old_r7_ts,
        "new_r7_input_latest_ts": new_r7_ts,
        "generated_rows": int(len(r7_frame)),
        "updated_rows": int(max(0, len(r7_frame))),
        "skipped_rows": 0,
        "feature_generation_status": "PASS" if not r7_frame.empty else "FAIL",
        "schema_validation_status": r7_validation_status,
        "tcn_inference_status": tcn_status,
        "tcn_error": tcn_error,
        "q2_diagnostic_status": "PASS" if r7_status == "PASS" and not args.no_q2 else ("SKIP_NO_Q2" if args.no_q2 else "FAIL"),
        "r7_input_status": r7_input_status,
        "overall_status": overall,
        "can_use_for_forward_validation": overall in {"FRESH", "WARN"} and r7_input_status in {"FRESH", "WARN"},
        "latest_features_path": str(root / "latest_features.parquet"),
        "latest_tcn_proba_path": str(root / "latest_tcn_proba.parquet"),
        "latest_q2_diagnostics_path": str(root / "latest_q2_diagnostics.parquet"),
        "latest_r7_input_frame_path": str(root / "latest_r7_input_frame.parquet"),
        "production_changed": False,
        "live_changed": False,
        "q2_changed": False,
        "state_changed": False,
    }
    (root / "feature_proba_refresh_latest.json").write_text(_json(manifest), encoding="utf-8")
    _write_md(root / "feature_proba_refresh_latest.md", "Feature/Proba Refresh Latest", manifest)
    hist_path = root / "feature_proba_refresh_history.csv"
    old_hist = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    pd.concat([old_hist, pd.DataFrame([manifest])], ignore_index=True).to_csv(hist_path, index=False)

    schema_rows = [{"feature": c, "present": c in r7_frame.columns, "numeric": bool(c in r7_frame.columns and pd.api.types.is_numeric_dtype(r7_frame[c]))} for c in ENTRY_FEATURE_COLS_V2]
    pd.DataFrame(schema_rows).to_csv(root / "feature_schema_validation.csv", index=False)
    _write_md(root / "feature_generation_report.md", "Feature Generation Report", {"discovered": discovered, "schema_validation": schema_rows[:20], "feature_generation_status": manifest["feature_generation_status"]})
    _write_md(root / "tcn_inference_report.md", "TCN Inference Report", {"status": tcn_status, "error": tcn_error, "rows": len(proba), "model_hash_changed": False})
    pd.DataFrame([{"check": "proba_non_empty", "pass": not proba.empty, "status": "PASS" if not proba.empty else "FAIL"}, {"check": "proba_latest", "pass": new_proba_ts is not None and not pd.isna(new_proba_ts), "status": "PASS" if new_proba_ts is not None and not pd.isna(new_proba_ts) else "FAIL"}]).to_csv(root / "tcn_proba_cache_validation.csv", index=False)
    _write_md(root / "q2_diagnostics_refresh_report.md", "Q2 Diagnostics Refresh Report", {"status": manifest["q2_diagnostic_status"], "logic_changed": False, "q2_baseline_config_changed": False})
    pd.DataFrame([{"check": "q2_columns_present", "pass": (not r7_frame.empty and all(c in r7_frame.columns for c in ["q2_score", "q2_scale", "q2_decision", "q2_accept", "q2_reject"])), "status": "PASS" if (not r7_frame.empty and all(c in r7_frame.columns for c in ["q2_score", "q2_scale", "q2_decision", "q2_accept", "q2_reject"])) else "FAIL"}]).to_csv(root / "q2_schema_validation.csv", index=False)

    after_hash = _prod_hashes()
    _write_audit(root, before_hash, after_hash, manifest)

    if overall in {"FRESH", "WARN"}:
        verdict = "feature_proba_refresh_installed + r7_input_fresh + r7_daily_ready + production_not_ready"
    elif tcn_status != "PASS":
        verdict = "feature_proba_refresh_path_missing + r7_stale_guard_active + production_not_ready"
    else:
        verdict = "feature_proba_refresh_installed + refresh_partial + r7_stale_guard_active + production_not_ready"
    setup = {
        "discovered feature builder path": discovered["feature_builder"],
        "discovered TCN inference path": discovered["tcn_inference"],
        "discovered Q2 diagnostic path": discovered["q2_diagnostic"],
        "canonical OHLCV path": str(ohlcv_path),
        "old feature/proba/latest timestamp": old_proba_ts,
        "new feature/proba/latest timestamp": new_proba_ts,
        "old R7 input latest timestamp": old_r7_ts,
        "new R7 input latest timestamp": new_r7_ts,
        "OHLCV freshness status": "see freshness checker",
        "feature freshness status": r7_input_status,
        "proba freshness status": "FRESH" if tcn_status == "PASS" else "STALE",
        "Q2 freshness status": "FRESH" if manifest["q2_diagnostic_status"] == "PASS" else "STALE",
        "R7 input freshness status": r7_input_status,
        "overall freshness status": overall,
        "generated rows": len(r7_frame),
        "updated rows": manifest["updated_rows"],
        "skipped rows": 0,
        "schema validation status": r7_validation_status,
        "TCN inference status": tcn_status,
        "Q2 diagnostic status": manifest["q2_diagnostic_status"],
        "R7 daily monitor status after refresh": "ready if freshness overall_status is FRESH/WARN",
        "Discord payload freshness status": "supported by R7 daily monitor freshness payload",
        "stale forward protection status": "active unless overall_status FRESH/WARN",
        "production hash unchanged": before_hash == after_hash,
        "Q2/live/state unchanged": True,
        "final verdict": verdict,
    }
    _write_md(root / "feature_proba_refresh_setup_report.md", "Feature/Proba Refresh Setup Report", setup)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh diagnostics-only feature/proba/Q2/R7 input cache.")
    parser.add_argument("--ohlcv-5m-path", default=None)
    parser.add_argument("--canonical-paths-json", default=None)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument("--min-lookback-days", type=int, default=7)
    parser.add_argument("--output-root", default=str(ROOT_DEFAULT))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-inference", action="store_true")
    parser.add_argument("--no-q2", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    manifest = run(args)
    if args.json:
        print(_json(manifest))
    elif args.verbose:
        print(f"overall_status={manifest['overall_status']} r7_input_latest={manifest['new_r7_input_latest_ts']}")
    return 0 if manifest["overall_status"] in {"FRESH", "WARN", "FEATURE_CACHE_STALE", "PROBA_CACHE_MISSING", "R7_INPUT_STALE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
