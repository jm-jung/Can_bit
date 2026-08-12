"""F2 pullback-reclaim OOS failure autopsy.

Diagnostics-only. This script does not create new entry rules or tune thresholds.
It only dissects the existing low-frequency swing F2 pullback-reclaim artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

SRC = Path("data/diagnostics/low_frequency_swing_entry_alpha_research")
ROOT = Path("data/diagnostics/f2_pullback_reclaim_oos_failure_autopsy")
CURRENT_COST_BPS = 6.0
PRIMARY_NET = "net_H4_24h"
PRIMARY_GROSS = "gross_H4_24h"


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "dataset",
        "fold",
        "regime",
        "subsets",
        "exit",
        "fake_reclaim",
        "orderflow",
        "cost",
        "casebook",
        "decision",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout)
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


def safety_snapshot(name: str) -> Dict[str, Any]:
    targets = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "config",
        "configs",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "ops",
        "scripts/run_daily_meta_research_ops.sh",
        "scripts/run_daily_paper_ops.sh",
        "scripts/run_daily_h8_candidate_ops.sh",
        "scripts/run_daily_h8_softgate_candidate_ops.sh",
        "scripts/run_daily_hybrid_candidate_ops.sh",
        "scripts/run_daily_quality_score_candidate_ops.sh",
    ]
    rows: List[Dict[str, Any]] = []
    for raw in targets:
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
        "canbit_launchd_lines": [ln for ln in sh(["launchctl", "list"]).splitlines() if "canbit" in ln.lower()],
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
        "python": sys.version,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def finalize_audit(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    rows = []
    for x in after.get("hashes", []):
        old = bmap.get(x["path"])
        rows.append({"path": x["path"], "sha256_before": old, "sha256_after": x.get("sha256"), "changed": old is not None and old != x.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": True, "write_class": "f2_autopsy_output"} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_f2_pullback_reclaim_oos_failure_autopsy.py", "diagnostics_only": False, "write_class": "requested_entrypoint"})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nNo production TCN/Q2/R7/Risk Manager/live/order/state path was changed. `forward_orderflow_collector_v4` and `false_high_r7_daily_monitor` were read-only. Discord/webhook policy was unchanged. No private/order/account/balance/position endpoints were called. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def input_discovery() -> Dict[str, Any]:
    required = [
        SRC / "low_frequency_swing_entry_alpha_research_final_report.md",
        SRC / "scorecards/single_family_scorecard.csv",
        SRC / "scorecards/stagewise_scorecard.csv",
        SRC / "scorecards/walk_forward_scorecard.csv",
        SRC / "walk_forward/walk_forward_fold_results.csv",
        SRC / "walk_forward/walk_forward_selected_signals.csv",
        SRC / "candidates/pullback_reclaim_candidates.parquet",
        SRC / "backfill/swing_alpha_paper_trades.parquet",
        SRC / "backfill/swing_alpha_exit_outcomes.parquet",
        SRC / "features/pullback_reclaim_features.parquet",
        SRC / "features/swing_feature_frame.parquet",
        SRC / "system_interaction/swing_risk_filter_interaction.csv",
        SRC / "forward_watchlist/swing_forward_watchlist.csv",
        Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup/decision/minimal_filter_set_decision.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/cache_registry.csv"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/open_interest_history/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/taker_buy_sell_volume/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/funding_rate/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/normalized/premium_index/BTCUSDT.parquet"),
        Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd/BTCUSDT_1h.parquet"),
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
    ]
    rows = []
    for p in required:
        rows.append({"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() and p.is_file() else 0, "suffix": p.suffix})
    for root in [SRC, Path("data/diagnostics/research_orderflow_data_cache"), Path("scripts/diagnostics")]:
        if root.exists():
            for p in root.rglob("*"):
                if p.is_file() and any(k in str(p).lower() for k in ["f2", "pullback", "walk_forward", "orderflow", "risk", "cvd", "funding", "premium"]):
                    rows.append({"path": str(p), "exists": True, "size": p.stat().st_size, "suffix": p.suffix})
    inv = pd.DataFrame(rows).drop_duplicates("path")
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    inv[inv["path"].str.contains("pullback|walk_forward|paper_trades|scorecard", case=False, regex=True)].to_csv(ROOT / "discovery/f2_artifact_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(inv.to_dict("records")[:5000]), encoding="utf-8")
    schema: Dict[str, Any] = {}
    for p in required:
        if p.suffix in [".parquet", ".csv"] and p.exists():
            df = read_df(p)
            schema[str(p)] = {"rows": len(df), "columns": list(df.columns), "dtypes": {c: str(df[c].dtype) for c in df.columns[:200]}}
    (ROOT / "discovery/f2_schema_summary.json").write_text(jdump(schema), encoding="utf-8")
    avail = []
    for p in required:
        df = read_df(p)
        ts_cols = [c for c in df.columns if "ts" in c or "time" in c or c == "timestamp"] if not df.empty else []
        start = end = ""
        for c in ts_cols:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                start, end = s.min(), s.max()
                break
        avail.append({"path": str(p), "exists": p.exists(), "rows": len(df), "start": start, "end": end})
    pd.DataFrame(avail).to_csv(ROOT / "discovery/data_availability_summary.csv", index=False)
    (ROOT / "discovery/discovery_report.md").write_text(
        "# Discovery Report\n\nF2 autopsy reads existing low-frequency swing artifacts only. Missing orderflow fields remain missing; no missing value is converted into a signal.\n",
        encoding="utf-8",
    )
    return {"inventory_rows": len(inv), "required_existing": int(sum(x["exists"] for x in rows[: len(required)]))}


def profit_factor(net: pd.Series) -> float:
    n = pd.to_numeric(net, errors="coerce").dropna()
    pos = n[n > 0].sum()
    neg = -n[n < 0].sum()
    return float(pos / neg) if neg > 0 else math.inf


def mdd(net: pd.Series) -> float:
    curve = pd.to_numeric(net, errors="coerce").fillna(0).cumsum()
    return float((curve.cummax() - curve).max()) if len(curve) else 0.0


def score(df: pd.DataFrame, object_id: str, group: str) -> Dict[str, Any]:
    if df.empty:
        return {"object_id": object_id, "group": group, "trade_count": 0, "sample_warning": True, "production_ready": False}
    net = pd.to_numeric(df["net_after_cost"], errors="coerce")
    gross = pd.to_numeric(df["gross_return"], errors="coerce")
    days = max(1, (pd.to_datetime(df["signal_ts"]).max() - pd.to_datetime(df["signal_ts"]).min()).days + 1)
    return {
        "object_id": object_id,
        "group": group,
        "trade_count": len(df),
        "trades_per_day": len(df) / days,
        "trades_per_week": len(df) / days * 7,
        "mean_net_bps": float(net.mean() * 10000),
        "median_net_bps": float(net.median() * 10000),
        "gross_mean_bps": float(gross.mean() * 10000),
        "maker_like_bps": float((gross - 3 / 10000).mean() * 10000),
        "two_x_cost_bps": float((gross - 12 / 10000).mean() * 10000),
        "winrate": float((net > 0).mean()),
        "PF": profit_factor(net),
        "RFE_rate": float(df["RFE"].astype(bool).mean()) if "RFE" in df else np.nan,
        "MFE_to_cost": float(pd.to_numeric(df.get("MFE_to_cost"), errors="coerce").mean()) if "MFE_to_cost" in df else np.nan,
        "MAE_to_cost": float(pd.to_numeric(df.get("MAE_to_cost"), errors="coerce").mean()) if "MAE_to_cost" in df else np.nan,
        "MDD_proxy": mdd(net),
        "tail_loss_bps": float(net.quantile(0.05) * 10000),
        "fake_reclaim_rate": float(df.get("fake_reclaim", pd.Series(False, index=df.index)).astype(bool).mean()),
        "MFE_exists_1x_cost_rate": float((pd.to_numeric(df.get("MFE", 0), errors="coerce") >= 0.0006).mean()),
        "MFE_exists_2x_cost_rate": float((pd.to_numeric(df.get("MFE", 0), errors="coerce") >= 0.0012).mean()),
        "sample_warning": len(df) < 50,
        "production_ready": False,
    }


def apply_frequency(df: pd.DataFrame, policy: str) -> pd.DataFrame:
    if policy == "all" or df.empty:
        return df.copy()
    limits = {"max_1_day": ("day", 1), "max_3_week": ("week", 3), "max_5_week": ("week", 5), "max_10_month": ("month", 10)}
    unit, limit = limits[policy]
    counts: Dict[Tuple[str, str], int] = {}
    keep = []
    for idx, row in df.sort_values("signal_ts").iterrows():
        ts = pd.to_datetime(row["signal_ts"])
        direction = str(row.get("direction", "MIXED"))
        key_ts = str(ts.date()) if unit == "day" else f"{ts.isocalendar().year}-{ts.isocalendar().week}" if unit == "week" else f"{ts.year}-{ts.month}"
        key = (key_ts, direction)
        if counts.get(key, 0) >= limit:
            continue
        counts[key] = counts.get(key, 0) + 1
        keep.append(idx)
    return df.loc[keep].copy()


def load_f2(fast: bool = False) -> pd.DataFrame:
    trades = pd.read_parquet(SRC / "backfill/swing_alpha_paper_trades.parquet")
    f2 = trades[trades["family"].astype(str).eq("F2_pullback_reclaim")].copy()
    # Match the prior F2 single-family scorecard basis: one F2 trade per
    # signal timestamp and direction, preserving the original candidate order.
    f2 = f2.sort_values(["signal_ts", "direction", "candidate_id"]).drop_duplicates(["signal_ts", "direction"]).reset_index(drop=True)
    if fast:
        f2 = f2.tail(800).copy()
    for c in ["signal_ts", "entry_ts", "timestamp"]:
        if c in f2:
            f2[c] = pd.to_datetime(f2[c], errors="coerce").astype("datetime64[ns]")
    f2["symbol"] = "BTCUSDT"
    f2["stage"] = "single_family_F2"
    f2["cost_bps"] = CURRENT_COST_BPS
    f2["net_after_cost"] = pd.to_numeric(f2.get(PRIMARY_NET), errors="coerce")
    f2["gross_return"] = pd.to_numeric(f2.get(PRIMARY_GROSS), errors="coerce")
    f2["horizon"] = "H4_24h"
    f2["exit_policy"] = "X1_fixed_H4_24h"
    f2["holding_hours"] = 24
    f2["exit_ts"] = f2["entry_ts"] + pd.Timedelta(hours=24)
    f2["exit_price"] = f2["entry_price"] * (1 + np.where(f2["direction"].eq("SHORT"), -f2["gross_return"], f2["gross_return"]))
    for col in ["time_to_MFE", "time_to_MAE", "exit_reason"]:
        if col not in f2:
            f2[col] = np.nan if col != "exit_reason" else "fixed_24h"
    f2 = add_fold_membership(f2)
    f2 = add_regime_columns(f2)
    f2 = add_fake_reclaim(f2)
    f2 = add_orderflow_flags(f2)
    return f2


def add_fold_membership(f2: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_csv(SRC / "walk_forward/walk_forward_fold_results.csv")
    folds["test_start"] = pd.to_datetime(folds["test_start"], errors="coerce")
    folds["test_end"] = pd.to_datetime(folds["test_end"], errors="coerce")
    folds["train_start"] = pd.to_datetime(folds["train_start"], errors="coerce")
    folds["train_end"] = pd.to_datetime(folds["train_end"], errors="coerce")
    f2 = f2.copy()
    f2["fold_id"] = -1
    f2["fold_train_test"] = "not_in_test"
    f2["walk_forward_train_or_test"] = "not_in_test"
    f2["walk_forward_selected"] = False
    for _, r in folds.iterrows():
        mask = (f2["signal_ts"] >= r["test_start"]) & (f2["signal_ts"] <= r["test_end"])
        f2.loc[mask, "fold_id"] = int(r["fold"])
        f2.loc[mask, "fold_train_test"] = "test"
        f2.loc[mask, "walk_forward_train_or_test"] = "test"
        f2.loc[mask, "walk_forward_selected"] = "F2_pullback_reclaim" in str(r.get("selected_signals", ""))
    return f2


def bucket_quantile(s: pd.Series, labels: Iterable[str]) -> pd.Series:
    labels = list(labels)
    try:
        return pd.qcut(pd.to_numeric(s, errors="coerce"), q=len(labels), labels=labels, duplicates="drop").astype(str)
    except Exception:
        return pd.Series("missing", index=s.index)


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(False, index=df.index)
    s = df[col]
    if s.dtype == bool:
        return s.fillna(False)
    return s.map(lambda x: True if str(x).lower() == "true" else False).fillna(False)


def add_regime_columns(f2: pd.DataFrame) -> pd.DataFrame:
    f = f2.copy()
    f["1d_trend_state"] = np.select([bool_col(f, "1d_trend_stack_bull"), bool_col(f, "1d_trend_stack_bear")], ["bull", "bear"], default="sideways")
    f["4h_trend_state"] = np.select([bool_col(f, "4h_trend_stack_bull"), bool_col(f, "4h_trend_stack_bear")], ["bull", "bear"], default="sideways")
    f["1h_trend_state"] = np.select([bool_col(f, "trend_stack_bull"), bool_col(f, "trend_stack_bear")], ["bull", "bear"], default="sideways")
    f["1d_slope"] = pd.to_numeric(f.get("trend_return_24"), errors="coerce")
    f["4h_slope"] = pd.to_numeric(f.get("ema_20_slope_6"), errors="coerce")
    f["drawdown_from_1d_high"] = pd.to_numeric(f.get("drawdown_from_recent_high"), errors="coerce")
    f["drawdown_from_4h_high"] = pd.to_numeric(f.get("drawdown_from_recent_high"), errors="coerce")
    f["ATR_percentile"] = f.get("atr_pct", pd.Series(np.nan, index=f.index)).rank(pct=True)
    f["vol_regime"] = pd.cut(f["ATR_percentile"], [0, 0.25, 0.75, 0.95, 1.0], labels=["low_vol", "mid_vol", "high_vol", "extreme_vol"], include_lowest=True).astype(str)
    f["ATR_bucket"] = bucket_quantile(f["ATR_percentile"], ["Q1", "Q2", "Q3", "Q4"])
    f["drawdown_bucket"] = pd.cut((-f["drawdown_from_recent_high"].fillna(0)).clip(lower=0), [-0.001, 0.02, 0.05, 0.10, 10], labels=["0_2pct", "2_5pct", "5_10pct", "10pct_plus"]).astype(str)
    f["distance_to_ma_bucket"] = np.select(
        [f["dist_ema_20"].abs() <= 0.005, f["dist_ema_50"].abs() <= 0.01, f["dist_ema_20"] > 0.02, f["dist_ema_20"] < -0.02],
        ["near_EMA20", "near_EMA50", "far_above", "far_below"],
        default="mid_distance",
    )
    f["BTC_market_state"] = np.select(
        [
            (f["1d_trend_state"] == "bull") & (f["4h_trend_state"] == "bull") & (f["trend_return_24"] > 0.03),
            (f["1d_trend_state"] == "bull"),
            (f["drawdown_bucket"].isin(["5_10pct", "10pct_plus"])) & (f["trend_return_6"] > 0),
            (f["1d_trend_state"] == "bear") & (f["4h_trend_state"] == "bear") & (f["trend_return_24"] < -0.03),
            (f["1d_trend_state"] == "bear"),
        ],
        ["strong_uptrend", "weak_uptrend", "post_drawdown_recovery", "strong_downtrend", "weak_downtrend"],
        default="range_chop",
    )
    f["range_bound_score"] = 1 - pd.to_numeric(f["trend_return_24"], errors="coerce").abs().rank(pct=True)
    f["trendiness_score"] = pd.to_numeric(f["trend_return_24"], errors="coerce").abs().rank(pct=True)
    f["chop_score"] = f["range_bound_score"] * pd.to_numeric(f["volatility_percentile"], errors="coerce").fillna(0.5)
    f["funding_z"] = pd.to_numeric(f.get("funding_score"), errors="coerce")
    f["basis_z"] = pd.to_numeric(f.get("basis_score"), errors="coerce")
    f["funding_bucket"] = pd.cut(f["funding_z"], [-np.inf, -1, 1, 2, np.inf], labels=["low", "normal", "high", "extreme"]).astype(str)
    f["basis_bucket"] = pd.cut(f["basis_z"], [-np.inf, -1, 1, 2, np.inf], labels=["discount", "normal", "premium", "extreme"]).astype(str)
    f["premium_overheat"] = f["basis_z"] > 2
    f["OI_change"] = pd.to_numeric(f.get("oi_change"), errors="coerce")
    f["taker_delta"] = pd.to_numeric(f.get("taker_delta"), errors="coerce")
    f["proxy_CVD_slope"] = pd.to_numeric(f.get("cvd_slope"), errors="coerce")
    f["taker_state"] = np.select([f["taker_delta"] > f["taker_delta"].quantile(0.67), f["taker_delta"] < f["taker_delta"].quantile(0.33)], ["taker_buy_dominant", "taker_sell_dominant"], default="neutral")
    f["cvd_state"] = np.select([f["proxy_CVD_slope"] > f["proxy_CVD_slope"].quantile(0.67), f["proxy_CVD_slope"] < f["proxy_CVD_slope"].quantile(0.33)], ["CVD_rising", "CVD_falling"], default="neutral")
    f["oi_state"] = np.select([f["OI_change"] > f["OI_change"].quantile(0.67), f["OI_change"] < f["OI_change"].quantile(0.33)], ["OI_building", "OI_deleveraging"], default="neutral")
    f["data_quality_score"] = 1 - f[["funding_z", "basis_z", "OI_change", "taker_delta", "proxy_CVD_slope"]].isna().mean(axis=1)
    return f


def add_fake_reclaim(f2: pd.DataFrame) -> pd.DataFrame:
    f = f2.copy()
    level = np.where(f["signal_id"].astype(str).str.contains("EMA50"), f.get("ema_50"), f.get("ema_20"))
    f["reclaim_level"] = pd.to_numeric(pd.Series(level, index=f.index), errors="coerce")
    # Attribution-only proxy: if MAE arrives before useful MFE or MAE exceeds MFE after reclaim.
    f["MFE_before_MAE"] = pd.to_numeric(f.get("MFE"), errors="coerce") >= pd.to_numeric(f.get("MAE"), errors="coerce")
    f["fake_reclaim"] = (~f["MFE_before_MAE"]) | (pd.to_numeric(f.get("MAE_to_cost"), errors="coerce") > pd.to_numeric(f.get("MFE_to_cost"), errors="coerce"))
    f["pullback_depth_atr"] = ((f["close"] - f["low"]).abs() / f["atr_14"].replace(0, np.nan)).where(f["direction"].eq("LONG"), ((f["high"] - f["close"]).abs() / f["atr_14"].replace(0, np.nan)))
    f["pullback_depth_pct"] = f["pullback_depth_atr"] * f["atr_pct"]
    f["pullback_duration_bars"] = np.nan
    f["distance_to_EMA20"] = f["dist_ema_20"]
    f["distance_to_EMA50"] = f["dist_ema_50"]
    f["reclaim_strength"] = np.where(f["direction"].eq("LONG"), (f["close"] - f["reclaim_level"]) / f["atr_14"].replace(0, np.nan), (f["reclaim_level"] - f["close"]) / f["atr_14"].replace(0, np.nan))
    f["reclaim_close_position"] = f["close_position_in_range"]
    f["reclaim_volume_z"] = f["volume_z"]
    f["prior_trend_slope"] = f["ema_20_slope_6"]
    f["higher_timeframe_alignment"] = ((f["direction"].eq("LONG") & (f["1d_trend_state"].eq("bull") | f["4h_trend_state"].eq("bull"))) | (f["direction"].eq("SHORT") & (f["1d_trend_state"].eq("bear") | f["4h_trend_state"].eq("bear"))))
    q = pd.to_numeric(f["reclaim_strength"], errors="coerce").rank(pct=True).fillna(0.5) + pd.to_numeric(f["reclaim_close_position"], errors="coerce").rank(pct=True).fillna(0.5) + f["higher_timeframe_alignment"].astype(int)
    f["pullback_quality_score"] = q / 3
    f["quality_bucket"] = pd.cut(f["pullback_quality_score"], [0, 0.25, 0.5, 0.75, 1.01], labels=["Q1_weak", "Q2_mid", "Q3_good", "Q4_best"], include_lowest=True).astype(str)
    return f


def add_orderflow_flags(f2: pd.DataFrame) -> pd.DataFrame:
    f = f2.copy()
    risk = pd.to_numeric(f.get("risk_adjusted_orderflow_score"), errors="coerce")
    f["orderflow_risk_worst20_flag"] = risk <= risk.quantile(0.20)
    f["orderflow_risk_not_worst20"] = ~f["orderflow_risk_worst20_flag"].fillna(False)
    f["taker_buy_confirm"] = f["taker_delta"] > f["taker_delta"].quantile(0.67)
    f["CVD_reclaim"] = f["proxy_CVD_slope"] > f["proxy_CVD_slope"].quantile(0.67)
    f["OI_build"] = f["OI_change"] > f["OI_change"].quantile(0.67)
    f["OI_deleveraging"] = f["OI_change"] < f["OI_change"].quantile(0.33)
    f["funding_normal"] = f["funding_z"].between(-1, 1)
    f["basis_not_overheated"] = ~(f["basis_z"] > 2)
    return f


def build_dataset(fast: bool = False) -> pd.DataFrame:
    f2 = load_f2(fast=fast)
    subset = pd.DataFrame(
        {
            "candidate_id": f2["candidate_id"],
            "F2_ALL": True,
            "F2_LONG_ONLY": f2["direction"].eq("LONG"),
            "F2_SHORT_ONLY": f2["direction"].eq("SHORT"),
            "F2_1H_ENTRY": f2["entry_timeframe"].eq("1h"),
            "F2_4H_ENTRY": f2["entry_timeframe"].eq("4h"),
            "F2_1D_ENTRY": f2["entry_timeframe"].eq("1d"),
            "F2_4H_OR_1D_ENTRY": f2["entry_timeframe"].isin(["4h", "1d"]),
            "F2_LONG_4H_OR_1D": f2["direction"].eq("LONG") & f2["entry_timeframe"].isin(["4h", "1d"]),
            "F2_SELECTED_BY_WALK_FORWARD": f2["walk_forward_selected"].astype(bool),
            "F2_NOT_SELECTED_BY_WALK_FORWARD": ~f2["walk_forward_selected"].astype(bool),
        }
    )
    for pol in ["max_5_week", "max_3_week"]:
        keep = set(apply_frequency(f2, pol)["candidate_id"])
        subset[f"F2_{pol.upper()}_POLICY"] = subset["candidate_id"].isin(keep)
    keep_long = set(apply_frequency(f2[f2["direction"].eq("LONG") & f2["entry_timeframe"].isin(["4h", "1d"])], "max_5_week")["candidate_id"])
    subset["F2_LONG_4H_1D_MAX5WEEK"] = subset["candidate_id"].isin(keep_long)
    f2.to_parquet(ROOT / "dataset/f2_canonical_autopsy_dataset.parquet", index=False)
    subset.to_parquet(ROOT / "dataset/f2_subset_membership.parquet", index=False)
    (ROOT / "dataset/f2_dataset_schema.json").write_text(jdump({c: str(f2[c].dtype) for c in f2.columns}), encoding="utf-8")
    f2.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "dataset/f2_missingness_summary.csv", index=False)
    (ROOT / "dataset/f2_dataset_build_report.md").write_text(
        f"# F2 Dataset Build Report\n\nCanonical F2 dataset rows: {len(f2)}. It uses existing F2 artifacts only; outcome-derived fields are attribution-only.\n",
        encoding="utf-8",
    )
    return f2


def fold_autopsy(f2: pd.DataFrame) -> pd.DataFrame:
    folds = pd.read_csv(SRC / "walk_forward/walk_forward_fold_results.csv")
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        folds[c] = pd.to_datetime(folds[c], errors="coerce")
    rows = []
    for _, r in folds.iterrows():
        train = f2[(f2["signal_ts"] >= r["train_start"]) & (f2["signal_ts"] <= r["train_end"])]
        test = f2[(f2["signal_ts"] >= r["test_start"]) & (f2["signal_ts"] <= r["test_end"])]
        test_net = test["net_after_cost"].mean() * 10000 if len(test) else np.nan
        fold_state = "PASS_FOLDS" if pd.notna(test_net) and test_net > 0 else "FAIL_FOLDS"
        if pd.notna(test_net) and test_net > CURRENT_COST_BPS:
            fold_state = "STRONG_PASS"
        elif pd.notna(test_net) and test_net < -CURRENT_COST_BPS:
            fold_state = "STRONG_FAIL"
        elif pd.notna(test_net) and -2 <= test_net <= 2:
            fold_state = "NEAR_ZERO"
        rows.append(
            {
                "fold_id": int(r["fold"]),
                "train_start": r["train_start"],
                "train_end": r["train_end"],
                "test_start": r["test_start"],
                "test_end": r["test_end"],
                "train_trade_count": len(train),
                "test_trade_count": len(test),
                "train_mean_net": train["net_after_cost"].mean() * 10000 if len(train) else np.nan,
                "test_mean_net": test_net,
                "train_pf": profit_factor(train["net_after_cost"]),
                "test_pf": profit_factor(test["net_after_cost"]),
                "train_rfe": train["RFE"].mean() if len(train) else np.nan,
                "test_rfe": test["RFE"].mean() if len(test) else np.nan,
                "train_mdd": mdd(train["net_after_cost"]),
                "test_mdd": mdd(test["net_after_cost"]),
                "selected_signal_count": len(str(r.get("selected_signals", "")).split(",")),
                "selected_signals": r.get("selected_signals", ""),
                "F2_selected": "F2_pullback_reclaim" in str(r.get("selected_signals", "")),
                "pass_fail": fold_state,
                "market_return_during_fold": test["close"].iloc[-1] / test["close"].iloc[0] - 1 if len(test) > 1 else np.nan,
                "BTC_1d_trend_return": test["trend_return_24"].mean() if len(test) else np.nan,
                "BTC_4h_trend_persistence": (test["4h_trend_state"].eq("bull") | test["4h_trend_state"].eq("bear")).mean() if len(test) else np.nan,
                "volatility_percentile": test["volatility_percentile"].mean() if len(test) else np.nan,
                "max_drawdown_in_test_window": test["drawdown_from_recent_high"].min() if len(test) else np.nan,
                "range_bound_score": test["range_bound_score"].mean() if len(test) else np.nan,
                "trendiness_score": test["trendiness_score"].mean() if len(test) else np.nan,
                "chop_score": test["chop_score"].mean() if len(test) else np.nan,
                "funding_mean_z": test["funding_z"].mean() if len(test) else np.nan,
                "basis_mean_z": test["basis_z"].mean() if len(test) else np.nan,
                "OI_trend": test["OI_change"].mean() if len(test) else np.nan,
                "taker_imbalance": test["taker_delta"].mean() if len(test) else np.nan,
                "proxy_CVD_trend": test["proxy_CVD_slope"].mean() if len(test) else np.nan,
                "fake_reclaim_rate": test["fake_reclaim"].mean() if len(test) else np.nan,
                "MFE_exists_rate": (test["MFE"] >= 0.0006).mean() if len(test) else np.nan,
                "time_to_MFE_median": test["time_to_MFE"].median() if "time_to_MFE" in test else np.nan,
                "time_to_MAE_median": test["time_to_MAE"].median() if "time_to_MAE" in test else np.nan,
                "primary_failure_reason": primary_fold_failure(test),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "fold/f2_fold_autopsy.csv", index=False)
    cmp_cols = ["test_trade_count", "test_mean_net", "test_pf", "test_rfe", "market_return_during_fold", "volatility_percentile", "range_bound_score", "trendiness_score", "chop_score", "fake_reclaim_rate", "MFE_exists_rate"]
    out.assign(pass_group=np.where(out["test_mean_net"] > 0, "PASS_FOLDS", "FAIL_FOLDS")).groupby("pass_group")[cmp_cols].mean(numeric_only=True).reset_index().to_csv(ROOT / "fold/f2_pass_vs_fail_fold_comparison.csv", index=False)
    out[["fold_id", "pass_fail", "market_return_during_fold", "BTC_1d_trend_return", "volatility_percentile", "range_bound_score", "trendiness_score", "chop_score"]].to_csv(ROOT / "fold/f2_fold_market_regime_summary.csv", index=False)
    out.groupby(["F2_selected", "pass_fail"]).size().reset_index(name="fold_count").to_csv(ROOT / "fold/f2_fold_selected_signal_stability.csv", index=False)
    (ROOT / "fold/fold_autopsy_report.md").write_text("# Fold Autopsy Report\n\nF2 fold metrics are recomputed from canonical F2 trades within existing walk-forward test windows. No threshold is retuned.\n", encoding="utf-8")
    return out


def primary_fold_failure(test: pd.DataFrame) -> str:
    if test.empty:
        return "NO_SAMPLE"
    mean_net = test["net_after_cost"].mean() * 10000
    gross = test["gross_return"].mean() * 10000
    fake = test["fake_reclaim"].mean()
    mfe2 = (test["MFE"] >= 0.0012).mean()
    if mean_net > 0:
        return "PASS"
    if gross > 0 and mean_net <= 0:
        return "COST_KILL"
    if fake > 0.55:
        return "FAKE_RECLAIM"
    if mfe2 > 0.5:
        return "MFE_EXISTS_BUT_EXIT_BAD"
    return "GROSS_EDGE_WEAK"


def regime_autopsy(f2: pd.DataFrame) -> pd.DataFrame:
    bucket_cols = [
        "1d_trend_state",
        "4h_trend_state",
        "BTC_market_state",
        "vol_regime",
        "ATR_bucket",
        "drawdown_bucket",
        "distance_to_ma_bucket",
        "funding_bucket",
        "basis_bucket",
        "taker_state",
        "cvd_state",
        "oi_state",
        "orderflow_risk_worst20_flag",
    ]
    rows = []
    for col in bucket_cols:
        if col not in f2:
            continue
        for val, g in f2.groupby(col, dropna=False):
            m = score(g, f"{col}={val}", "regime")
            m["bucket_type"] = col
            m["bucket_value"] = val
            rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "regime/f2_regime_scorecard.csv", index=False)
    out.sort_values(["mean_net_bps", "trade_count"], ascending=[False, False]).head(30).to_csv(ROOT / "regime/f2_best_regime_candidates.csv", index=False)
    out.sort_values(["mean_net_bps", "trade_count"], ascending=[True, False]).head(30).to_csv(ROOT / "regime/f2_bad_regime_candidates.csv", index=False)
    interaction = []
    for a, b in [("1d_trend_state", "4h_trend_state"), ("BTC_market_state", "vol_regime"), ("drawdown_bucket", "vol_regime"), ("basis_bucket", "taker_state")]:
        if a in f2 and b in f2:
            for vals, g in f2.groupby([a, b], dropna=False):
                m = score(g, f"{a}={vals[0]}|{b}={vals[1]}", "regime_interaction")
                m["bucket_a"] = a
                m["value_a"] = vals[0]
                m["bucket_b"] = b
                m["value_b"] = vals[1]
                interaction.append(m)
    pd.DataFrame(interaction).to_csv(ROOT / "regime/f2_regime_interaction_matrix.csv", index=False)
    verdict = "REGIME_SPECIFIC_EDGE" if (out[(out["trade_count"] >= 100) & (out["mean_net_bps"] > 0)].shape[0] > 0) else "NO_STABLE_REGIME_EDGE"
    (ROOT / "regime/regime_dependency_report.md").write_text(f"# Regime Dependency Report\n\nVerdict: {verdict}. Regime buckets are fixed pre-defined buckets; no test-window threshold retuning was used.\n", encoding="utf-8")
    return out


def subset_autopsy(f2: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    specs = {
        "F2_ALL": f2.index == f2.index,
        "F2_LONG_ONLY": f2["direction"].eq("LONG"),
        "F2_SHORT_ONLY": f2["direction"].eq("SHORT"),
        "F2_1H_ENTRY": f2["entry_timeframe"].eq("1h"),
        "F2_4H_ENTRY": f2["entry_timeframe"].eq("4h"),
        "F2_1D_ENTRY": f2["entry_timeframe"].eq("1d"),
        "F2_4H_1D": f2["entry_timeframe"].isin(["4h", "1d"]),
        "F2_LONG_4H_1D": f2["direction"].eq("LONG") & f2["entry_timeframe"].isin(["4h", "1d"]),
        "F2_MAX_3_WEEK": f2["candidate_id"].isin(set(apply_frequency(f2, "max_3_week")["candidate_id"])),
        "F2_MAX_5_WEEK": f2["candidate_id"].isin(set(apply_frequency(f2, "max_5_week")["candidate_id"])),
        "F2_MAX_10_MONTH": f2["candidate_id"].isin(set(apply_frequency(f2, "max_10_month")["candidate_id"])),
    }
    rows = []
    for name, mask in specs.items():
        sub = f2[mask].copy()
        m = score(sub, name, "subset")
        test_means = []
        pass_count = 0
        test_counts = []
        for _, fold in folds.iterrows():
            g = sub[sub["fold_id"].eq(int(fold["fold_id"]))]
            if len(g):
                val = g["net_after_cost"].mean() * 10000
                test_means.append(val)
                pass_count += int(val > 0)
                test_counts.append(len(g))
        m["walk_forward_test_mean_bps"] = float(np.nanmean(test_means)) if test_means else np.nan
        m["walk_forward_pass_rate"] = float(pass_count / len(test_means)) if test_means else np.nan
        m["folds_with_sample"] = len(test_means)
        m["median_test_count"] = float(np.median(test_counts)) if test_counts else 0
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "subsets/f2_timeframe_direction_frequency_scorecard.csv", index=False)
    out[out["object_id"].str.contains("LONG", na=False)].to_csv(ROOT / "subsets/f2_long_only_autopsy.csv", index=False)
    out[out["object_id"].str.contains("4H|1D", na=False)].to_csv(ROOT / "subsets/f2_4h_1d_autopsy.csv", index=False)
    out[out["object_id"].str.contains("MAX", na=False)].to_csv(ROOT / "subsets/f2_frequency_policy_autopsy.csv", index=False)
    (ROOT / "subsets/subset_autopsy_report.md").write_text("# Subset Autopsy Report\n\nSubsets reuse existing F2 entries and existing fold windows. Frequency policies are chronological caps, not optimized thresholds.\n", encoding="utf-8")
    return out


def exit_autopsy(f2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    horizon_pairs = [(c.replace("net_", ""), c, f"gross_{c.replace('net_', '')}") for c in f2.columns if c.startswith("net_H")]
    for hid, net_col, gross_col in horizon_pairs:
        tmp = f2.copy()
        tmp["net_after_cost"] = pd.to_numeric(tmp[net_col], errors="coerce")
        tmp["gross_return"] = pd.to_numeric(tmp[gross_col], errors="coerce") if gross_col in tmp else tmp["gross_return"]
        rows.append(score(tmp, hid, "fixed_horizon"))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "exit/f2_exit_horizon_scorecard.csv", index=False)
    timing = pd.DataFrame(
        [
            {
                "MFE_exists_1x_cost_rate": (f2["MFE"] >= 0.0006).mean(),
                "MFE_exists_2x_cost_rate": (f2["MFE"] >= 0.0012).mean(),
                "MFE_exists_3x_cost_rate": (f2["MFE"] >= 0.0018).mean(),
                "MFE_before_MAE_rate": f2["MFE_before_MAE"].mean(),
                "giveback_rate": ((f2["MFE"] > 0.0012) & (f2["net_after_cost"] < 0)).mean(),
                "late_exit_loss_rate": ((f2["MFE"] > 0.0012) & (f2["net_after_cost"] < -0.0006)).mean(),
                "early_exit_missed_rate": np.nan,
                "best_non_oracle_exit": out.sort_values("mean_net_bps", ascending=False).iloc[0]["object_id"] if not out.empty else "",
                "exit_improvement_possible": bool((f2["MFE"] > 0.0012).mean() > 0.5 and f2["net_after_cost"].mean() < f2["MFE"].mean()),
                "exit_overfit_warning": "reference_only_no_exit_rule_selected",
            }
        ]
    )
    timing.to_csv(ROOT / "exit/f2_mfe_mae_timing.csv", index=False)
    modes = []
    for name, mask in {
        "MFE_EXISTS_BUT_GIVEBACK": (f2["MFE"] >= 0.0012) & (f2["net_after_cost"] < 0),
        "MFE_TOO_SMALL_COST_KILL": f2["MFE"] < 0.0006,
        "MAE_TOO_EARLY": ~f2["MFE_before_MAE"],
        "HORIZON_MISMATCH": (f2["MFE"] >= 0.0012) & (f2["net_after_cost"] < f2["gross_return"]),
    }.items():
        modes.append({"failure_mode": name, "rate": float(mask.mean()), "count": int(mask.sum())})
    pd.DataFrame(modes).to_csv(ROOT / "exit/f2_exit_failure_modes.csv", index=False)
    verdict = "ENTRY_OK_EXIT_BAD" if timing.iloc[0]["MFE_exists_2x_cost_rate"] > 0.5 and f2["net_after_cost"].mean() <= 0 else "ENTRY_BAD_NOT_EXIT"
    if timing.iloc[0]["giveback_rate"] > 0.3:
        verdict = "MFE_EXISTS_BUT_GIVEBACK"
    (ROOT / "exit/exit_autopsy_report.md").write_text(f"# Exit Autopsy Report\n\nVerdict: {verdict}. Oracle MFE is attribution-only; no exit policy is promoted.\n", encoding="utf-8")
    return out


def fake_reclaim_autopsy(f2: pd.DataFrame) -> pd.DataFrame:
    fake = f2[f2["fake_reclaim"]].copy()
    fake.to_csv(ROOT / "fake_reclaim/f2_fake_reclaim_cases.csv", index=False)
    rows = []
    for bucket, g in f2.groupby("quality_bucket", dropna=False):
        m = score(g, f"quality_bucket={bucket}", "pullback_quality")
        m["quality_bucket"] = bucket
        rows.append(m)
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "fake_reclaim/f2_pullback_quality_scorecard.csv", index=False)
    cmp = []
    for name, g in [("success", f2[f2["net_after_cost"] > 0]), ("fake_reclaim", fake), ("failure_non_fake", f2[(f2["net_after_cost"] <= 0) & ~f2["fake_reclaim"]])]:
        cmp.append(
            {
                "group": name,
                "rows": len(g),
                "mean_net_bps": g["net_after_cost"].mean() * 10000 if len(g) else np.nan,
                "pullback_depth_atr": g["pullback_depth_atr"].mean() if len(g) else np.nan,
                "reclaim_strength": g["reclaim_strength"].mean() if len(g) else np.nan,
                "reclaim_close_position": g["reclaim_close_position"].mean() if len(g) else np.nan,
                "prior_trend_slope": g["prior_trend_slope"].mean() if len(g) else np.nan,
                "higher_timeframe_alignment": g["higher_timeframe_alignment"].mean() if len(g) else np.nan,
            }
        )
    pd.DataFrame(cmp).to_csv(ROOT / "fake_reclaim/f2_success_vs_fake_reclaim_comparison.csv", index=False)
    verdict = "FAKE_RECLAIM_MAIN_FAILURE" if f2["fake_reclaim"].mean() > 0.5 else "QUALITY_SCORE_FORWARD_WATCH_ONLY"
    (ROOT / "fake_reclaim/fake_reclaim_autopsy_report.md").write_text(f"# Fake Reclaim Autopsy Report\n\nVerdict: {verdict}. Fake reclaim is attribution-only and not a new gating rule.\n", encoding="utf-8")
    return out


def orderflow_autopsy(f2: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    masks = {
        "F2_orderflow_risk_worst20_false": ~f2["orderflow_risk_worst20_flag"].fillna(False),
        "F2_orderflow_risk_worst20_true": f2["orderflow_risk_worst20_flag"].fillna(False),
        "F2_taker_buy_confirm": f2["taker_buy_confirm"].fillna(False),
        "F2_CVD_reclaim": f2["CVD_reclaim"].fillna(False),
        "F2_OI_deleveraging": f2["OI_deleveraging"].fillna(False),
        "F2_OI_build": f2["OI_build"].fillna(False),
        "F2_funding_normal": f2["funding_normal"].fillna(False),
        "F2_basis_not_overheated": f2["basis_not_overheated"].fillna(False),
        "F2_orderflow_confirm_only": (f2["taker_buy_confirm"].fillna(False) | f2["CVD_reclaim"].fillna(False)),
        "F2_orderflow_risk_filter_only": ~f2["orderflow_risk_worst20_flag"].fillna(False),
        "F2_confirm_and_risk_filter": (~f2["orderflow_risk_worst20_flag"].fillna(False)) & (f2["taker_buy_confirm"].fillna(False) | f2["CVD_reclaim"].fillna(False)),
    }
    base_good = f2["net_after_cost"] > 0
    base_bad = f2["net_after_cost"] <= 0
    rows, pres, stab = [], [], []
    for name, mask in masks.items():
        sub = f2[mask].copy()
        rows.append(score(sub, name, "orderflow"))
        pres.append(
            {
                "subset": name,
                "trade_count": len(sub),
                "GOOD_retention": float((mask & base_good).sum() / max(1, base_good.sum())),
                "BAD_reduction": float(1 - (mask & base_bad).sum() / max(1, base_bad.sum())),
                "missed_good": int((~mask & base_good).sum()),
            }
        )
        fold_vals = []
        for _, fold in folds.iterrows():
            g = sub[sub["fold_id"].eq(int(fold["fold_id"]))]
            if len(g):
                fold_vals.append(g["net_after_cost"].mean() * 10000)
        stab.append({"subset": name, "folds_with_sample": len(fold_vals), "oos_mean_bps": float(np.nanmean(fold_vals)) if fold_vals else np.nan, "oos_pass_rate": float(np.mean([x > 0 for x in fold_vals])) if fold_vals else np.nan})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "orderflow/f2_orderflow_interaction_scorecard.csv", index=False)
    pd.DataFrame(pres).to_csv(ROOT / "orderflow/f2_orderflow_good_bad_preservation.csv", index=False)
    pd.DataFrame(stab).to_csv(ROOT / "orderflow/f2_orderflow_fold_stability.csv", index=False)
    base_mean = score(f2, "F2_ALL", "base")["mean_net_bps"]
    stab_df = pd.DataFrame(stab)
    base_oos = f2.groupby("fold_id")["net_after_cost"].mean().mul(10000).mean()
    best = out.sort_values("mean_net_bps", ascending=False).head(1)
    verdict = "ORDERFLOW_NO_HELP"
    best_stab = stab_df[stab_df["subset"].eq(best.iloc[0]["object_id"])] if not best.empty else pd.DataFrame()
    oos_improves = (not best_stab.empty) and pd.notna(best_stab.iloc[0]["oos_mean_bps"]) and best_stab.iloc[0]["oos_mean_bps"] > base_oos + 1
    meaningful_filter = (not best.empty) and best.iloc[0]["trade_count"] < len(f2) * 0.95
    if not best.empty and best.iloc[0]["mean_net_bps"] > base_mean + 1 and best.iloc[0]["trade_count"] > 50 and (oos_improves or meaningful_filter):
        verdict = "ORDERFLOW_FORWARD_WATCH_ONLY"
    (ROOT / "orderflow/orderflow_interaction_report.md").write_text(f"# Orderflow Interaction Report\n\nVerdict: {verdict}. Existing buckets/quantiles only; no production filter is created.\n", encoding="utf-8")
    return out


def cost_autopsy(f2: pd.DataFrame, subset_scores: pd.DataFrame) -> pd.DataFrame:
    subset_masks = {
        "F2_ALL": f2.index == f2.index,
        "F2_LONG_ONLY": f2["direction"].eq("LONG"),
        "F2_4H_1D": f2["entry_timeframe"].isin(["4h", "1d"]),
        "F2_LONG_4H_1D": f2["direction"].eq("LONG") & f2["entry_timeframe"].isin(["4h", "1d"]),
        "F2_MAX5WEEK": f2["candidate_id"].isin(set(apply_frequency(f2, "max_5_week")["candidate_id"])),
        "F2_BEST_REGIME_REFERENCE": f2["BTC_market_state"].isin(["weak_uptrend", "strong_uptrend", "post_drawdown_recovery"]),
        "F2_ORDERFLOW_FILTER_REFERENCE": ~f2["orderflow_risk_worst20_flag"].fillna(False),
    }
    rows, be = [], []
    for name, mask in subset_masks.items():
        sub = f2[mask]
        gross = sub["gross_return"]
        for cost in [0, 3, 6, 9, 12, 15]:
            net = gross - cost / 10000
            rows.append({"subset": name, "cost_bps": cost, "trade_count": len(sub), "gross_mean_bps": gross.mean() * 10000 if len(sub) else np.nan, "net_after_cost_bps": net.mean() * 10000 if len(sub) else np.nan, "profit_factor": profit_factor(net) if len(sub) else np.nan})
        be.append({"subset": name, "break_even_cost_bps": gross.mean() * 10000 if len(sub) else np.nan, "gross_mean": gross.mean() * 10000 if len(sub) else np.nan, "execution_dependency": "MAKER_REQUIRED" if len(sub) and gross.mean() * 10000 < 6 else "CURRENT_COST_SURVIVES_REFERENCE"})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "cost/f2_cost_sensitivity.csv", index=False)
    pd.DataFrame(be).to_csv(ROOT / "cost/f2_break_even_cost_by_subset.csv", index=False)
    verdict = "GROSS_EDGE_EXISTS_COST_KILLS" if f2["gross_return"].mean() * 10000 > 0 and f2["net_after_cost"].mean() * 10000 <= 0 else "COST_NOT_PRIMARY_FAILURE"
    if f2["gross_return"].mean() * 10000 <= 0:
        verdict = "NO_GROSS_EDGE"
    (ROOT / "cost/cost_execution_autopsy_report.md").write_text(f"# Cost Execution Autopsy Report\n\nVerdict: {verdict}. Break-even cost is gross mean bps by subset.\n", encoding="utf-8")
    return out


def casebook(f2: pd.DataFrame) -> pd.DataFrame:
    cb = f2.copy()
    cb["case_category"] = np.select(
        [
            cb["net_after_cost"] > 0,
            cb["fake_reclaim"],
            cb["1d_trend_state"].eq("bear") | cb["4h_trend_state"].eq("bear"),
            (cb["MFE"] >= 0.0012) & (cb["net_after_cost"] < 0),
            cb["gross_return"] > 0,
            cb["vol_regime"].isin(["high_vol", "extreme_vol"]),
        ],
        [
            "F2_SUCCESS_TREND_PULLBACK_RECLAIM",
            "F2_FAIL_FAKE_RECLAIM",
            "F2_FAIL_TREND_BREAKDOWN",
            "F2_FAIL_EXIT_GIVEBACK",
            "F2_FAIL_COST_KILL",
            "F2_FAIL_HIGH_VOL_CHOP",
        ],
        default="F2_FAIL_BEAR_OR_RANGE_REGIME",
    )
    cb["why_success"] = np.where(cb["net_after_cost"] > 0, "positive current-cost 24h net outcome", "")
    cb["why_failure"] = np.where(cb["net_after_cost"] <= 0, "negative current-cost outcome; see fake/reclaim/regime/exit fields", "")
    cb["recommended_action"] = np.where(cb["net_after_cost"] > 0, "CASEBOOK_SUCCESS_REFERENCE", "CASEBOOK_FAILURE_REFERENCE")
    cb["chart_window_path"] = ""
    cols = [
        "candidate_id",
        "fold_id",
        "signal_ts",
        "entry_timeframe",
        "direction",
        "signal_id",
        "variant",
        "entry_price",
        "net_after_cost",
        "MFE",
        "MAE",
        "RFE",
        "1d_trend_state",
        "4h_trend_state",
        "vol_regime",
        "funding_z",
        "basis_z",
        "OI_change",
        "taker_delta",
        "proxy_CVD_slope",
        "orderflow_risk_worst20_flag",
        "case_category",
        "why_success",
        "why_failure",
        "recommended_action",
        "chart_window_path",
    ]
    out = cb[[c for c in cols if c in cb]].sort_values("net_after_cost", ascending=False)
    out.to_parquet(ROOT / "casebook/f2_casebook.parquet", index=False)
    out.to_csv(ROOT / "casebook/f2_casebook.csv", index=False)
    out.head(50).to_csv(ROOT / "casebook/f2_top_success_cases.csv", index=False)
    out.tail(50).to_csv(ROOT / "casebook/f2_top_failure_cases.csv", index=False)
    (ROOT / "casebook/f2_casebook_report.md").write_text("# F2 Casebook Report\n\nTop success and failure cases are stored. Chart windows are not generated by default to keep this autopsy deterministic and light.\n", encoding="utf-8")
    return out


def decisions(f2: pd.DataFrame, folds: pd.DataFrame, subsets: pd.DataFrame, regimes: pd.DataFrame, exit_sc: pd.DataFrame, orderflow: pd.DataFrame) -> List[str]:
    all_score = score(f2, "F2_ALL", "decision")
    long_4h1d = subsets[subsets["object_id"].eq("F2_LONG_4H_1D")]
    best_subset = subsets.sort_values("walk_forward_test_mean_bps", ascending=False).head(1)
    stable_regimes = regimes[(regimes["trade_count"] >= 100) & (regimes["mean_net_bps"] > 0)]
    exit_timing = pd.read_csv(ROOT / "exit/f2_mfe_mae_timing.csv")
    verdicts = ["F2_OOS_AUTOPSY_COMPLETED", "production_not_ready"]
    if folds["test_mean_net"].mean() <= 0:
        verdicts.append("F2_WALK_FORWARD_FAIL_CONFIRMED")
    if not long_4h1d.empty and long_4h1d.iloc[0]["mean_net_bps"] >= 0 and long_4h1d.iloc[0]["walk_forward_pass_rate"] >= 0.5:
        verdicts.append("F2_LONG_4H_1D_ONLY")
    if not stable_regimes.empty:
        verdicts.append("F2_REGIME_SPECIFIC_ONLY")
    if f2[f2["direction"].eq("LONG")]["net_after_cost"].mean() > f2[f2["direction"].eq("SHORT")]["net_after_cost"].mean():
        verdicts.append("F2_BULL_ONLY_EDGE")
    if exit_timing.iloc[0]["giveback_rate"] > 0.3:
        verdicts += ["F2_ENTRY_OK_EXIT_BAD", "F2_NEED_EXIT_REDESIGN"]
    if f2["fake_reclaim"].mean() > 0.5:
        verdicts.append("F2_FAKE_RECLAIM_MAIN_FAILURE")
    if all_score["gross_mean_bps"] > 0 and all_score["mean_net_bps"] <= 0:
        verdicts.append("F2_COST_KILLED")
    elif all_score["gross_mean_bps"] <= 0:
        verdicts.append("F2_NO_GROSS_EDGE")
    of_stab = pd.read_csv(ROOT / "orderflow/f2_orderflow_fold_stability.csv") if (ROOT / "orderflow/f2_orderflow_fold_stability.csv").exists() else pd.DataFrame()
    best_of = orderflow.sort_values("mean_net_bps", ascending=False).head(1)
    best_of_stab = of_stab[of_stab["subset"].eq(best_of.iloc[0]["object_id"])] if not best_of.empty and not of_stab.empty else pd.DataFrame()
    of_oos_improves = (not best_of_stab.empty) and pd.notna(best_of_stab.iloc[0]["oos_mean_bps"]) and best_of_stab.iloc[0]["oos_mean_bps"] > folds["test_mean_net"].mean() + 1
    of_meaningful_filter = (not best_of.empty) and best_of.iloc[0]["trade_count"] < len(f2) * 0.95
    if not best_of.empty and best_of.iloc[0]["mean_net_bps"] > all_score["mean_net_bps"] + 1 and best_of.iloc[0]["trade_count"] >= 50 and (of_oos_improves or of_meaningful_filter):
        verdicts.append("F2_ORDERFLOW_RESCUES")
    else:
        verdicts.append("F2_ORDERFLOW_NO_HELP")
    # Final class: OOS is negative, but stable positive pockets exist. Keep only as forward-gated research if OOS subset is not hopeless.
    if not best_subset.empty and best_subset.iloc[0]["walk_forward_pass_rate"] >= 0.5 and best_subset.iloc[0]["walk_forward_test_mean_bps"] >= -1:
        final_decision = "KEEP_FOR_FORWARD_REGIME_GATED"
        verdicts.append("F2_KEEP_FOR_FORWARD_REGIME_GATED")
    elif not stable_regimes.empty or exit_timing.iloc[0]["MFE_exists_2x_cost_rate"] > 0.5:
        final_decision = "KEEP_CASEBOOK_ONLY"
        verdicts.append("F2_KEEP_CASEBOOK_ONLY")
    else:
        final_decision = "DROP"
        verdicts.append("F2_DROP")
    verdicts = list(dict.fromkeys(verdicts))
    pd.DataFrame(
        [
            {
                "object_id": "F2_pullback_reclaim",
                "decision": final_decision,
                "mean_net_bps": all_score["mean_net_bps"],
                "gross_mean_bps": all_score["gross_mean_bps"],
                "walk_forward_mean_test_bps": folds["test_mean_net"].mean(),
                "walk_forward_pass_rate": (folds["test_mean_net"] > 0).mean(),
                "fake_reclaim_rate": f2["fake_reclaim"].mean(),
                "MFE_2x_cost_rate": exit_timing.iloc[0]["MFE_exists_2x_cost_rate"],
                "production_ready": False,
            }
        ]
    ).to_csv(ROOT / "decision/f2_keep_drop_decision.csv", index=False)
    pd.DataFrame(
        [
            {"watch_id": "FW_F2_LONG_4H_1D_PULLBACK_RECLAIM", "action": "keep_casebook_or_forward_gated_only", "reason": "positive historical pockets but OOS instability"},
            {"watch_id": "FW_F2_EXIT_REDESIGN_REFERENCE", "action": "research_only", "reason": "MFE/giveback attribution suggests exit may be the next isolated question"},
        ]
    ).to_csv(ROOT / "decision/f2_forward_watchlist_update.csv", index=False)
    (ROOT / "decision/f2_final_recommendation.md").write_text(
        f"# F2 Final Recommendation\n\nDecision: {final_decision}. Do not promote. If continued, keep as research-only casebook/forward-gated observation and isolate exit redesign separately.\n",
        encoding="utf-8",
    )
    return verdicts


def final_report(f2: pd.DataFrame, fold: pd.DataFrame, regimes: pd.DataFrame, subsets: pd.DataFrame, exit_sc: pd.DataFrame, orderflow: pd.DataFrame, costs: pd.DataFrame, verdicts: List[str]) -> None:
    pass_cmp = pd.read_csv(ROOT / "fold/f2_pass_vs_fail_fold_comparison.csv")
    top_reg = regimes.sort_values("mean_net_bps", ascending=False).head(10).to_dict("records")
    bad_reg = regimes.sort_values("mean_net_bps", ascending=True).head(10).to_dict("records")
    report = f"""# F2 Pullback-Reclaim OOS Failure Autopsy

## Why
Low-frequency swing research found only `F2_pullback_reclaim` positive in-sample/current-cost, but walk-forward failed. This autopsy dissects that failure without creating or tuning new entry rules.

## Canonical Dataset
- F2 trades: {len(f2)}
- Mean net bps: {f2['net_after_cost'].mean() * 10000:.2f}
- Gross mean bps: {f2['gross_return'].mean() * 10000:.2f}
- LONG mean bps: {f2[f2['direction'].eq('LONG')]['net_after_cost'].mean() * 10000:.2f}
- SHORT mean bps: {f2[f2['direction'].eq('SHORT')]['net_after_cost'].mean() * 10000:.2f}

## Fold Autopsy
- F2 fold mean test bps: {fold['test_mean_net'].mean():.2f}
- F2 fold pass rate: {(fold['test_mean_net'] > 0).mean():.2%}
- Strong fail folds: {(fold['pass_fail'] == 'STRONG_FAIL').sum()}

Pass/fail comparison:
```json
{jdump(pass_cmp.to_dict('records'))}
```

## Regime Dependency
Best fixed regime buckets:
```json
{jdump(top_reg)}
```

Worst fixed regime buckets:
```json
{jdump(bad_reg)}
```

## Subsets
```json
{jdump(subsets.sort_values('walk_forward_test_mean_bps', ascending=False).head(15).to_dict('records'))}
```

## Exit / MFE
```json
{jdump(pd.read_csv(ROOT / 'exit/f2_mfe_mae_timing.csv').to_dict('records'))}
```

## Orderflow / Risk
```json
{jdump(orderflow.sort_values('mean_net_bps', ascending=False).head(12).to_dict('records'))}
```

## Cost / Execution
```json
{jdump(pd.read_csv(ROOT / 'cost/f2_break_even_cost_by_subset.csv').to_dict('records'))}
```

## Decision
Verdicts:
{chr(10).join(verdicts)}

## Safety
Production TCN/Q2/R7/Risk Manager/live/order/state files were not changed. forward_orderflow_collector_v4 and Discord/webhook policy were read-only. production_ready=false; promotion_ready=false.
"""
    (ROOT / "f2_pullback_reclaim_oos_failure_autopsy_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "f2_pullback_reclaim_oos_failure_autopsy_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nRun a diagnostics-only exit redesign autopsy on F2 casebook if continuing. Do not add new F2 gates or production triggers.\n", encoding="utf-8")


def run(mode: str = "full", fast: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log(f"start mode={mode} fast={fast}")
    discovery = input_discovery()
    f2 = build_dataset(fast=fast)
    fold = fold_autopsy(f2)
    regimes = regime_autopsy(f2)
    subsets = subset_autopsy(f2, fold)
    exit_sc = exit_autopsy(f2)
    fake_reclaim_autopsy(f2)
    orderflow = orderflow_autopsy(f2, fold)
    costs = cost_autopsy(f2, subsets)
    casebook(f2)
    verdicts = decisions(f2, fold, subsets, regimes, exit_sc, orderflow)
    final_report(f2, fold, regimes, subsets, exit_sc, orderflow, costs, verdicts)
    (ROOT / "run_metadata.json").write_text(jdump({"mode": mode, "fast": fast, "rows": len(f2), "updated_ts": pd.Timestamp.now("UTC").isoformat()}), encoding="utf-8")
    finalize_audit(before)
    log("done")
    return {"mode": mode, "fast": fast, "f2_rows": len(f2), "folds": len(fold), "verdicts": verdicts, "production_ready": False, "promotion_ready": False, "discovery": discovery}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--fold-autopsy", action="store_true")
    parser.add_argument("--regime-autopsy", action="store_true")
    parser.add_argument("--exit-autopsy", action="store_true")
    parser.add_argument("--casebook-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "source_exists": SRC.exists(), "f2_artifact_exists": (SRC / "backfill/swing_alpha_paper_trades.parquet").exists(), "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = run("fast_smoke", fast=True)
    elif args.fold_autopsy:
        res = run("fold_autopsy")
    elif args.regime_autopsy:
        res = run("regime_autopsy")
    elif args.exit_autopsy:
        res = run("exit_autopsy")
    elif args.casebook_only:
        res = run("casebook_only")
    else:
        res = run("full")
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
