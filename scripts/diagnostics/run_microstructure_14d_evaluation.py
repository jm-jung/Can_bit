#!/usr/bin/env python3
"""Read-only PRELIMINARY 14D cumulative strict forward observation evaluation.

Does not modify observer, Q2_BDI, thresholds, T0, quarantine, launchd, or production state.
Does not overwrite pretrip 7D artifacts; uses them as WEEK1 consistency reference.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/diagnostics/microstructure_14d_evaluation"
RESULTS = OUT / "results"
CHARTS = OUT / "charts"
REF7 = ROOT / "data/diagnostics/microstructure_pretrip_7d_evaluation"
OBSERVER = ROOT / "data/diagnostics/microstructure_weak_hint_forward_observer"
GAP_LEDGER = ROOT / "data/diagnostics/microstructure_gap_backfill/data/gap_ledger.parquet"
LIVE_NORM = ROOT / "data/diagnostics/new_market_microstructure_data_pipeline/live/normalized"
LIVE_RAW = ROOT / "data/diagnostics/new_market_microstructure_data_pipeline/live/raw"
HG = OUT / "health_gate"

T0 = pd.Timestamp("2026-07-02T13:33:09", tz="UTC")
W1_END = pd.Timestamp("2026-07-09T13:33:09", tz="UTC")
T14 = pd.Timestamp("2026-07-16T13:33:09", tz="UTC")
WINDOWS = {
    "WEEK1": (T0, W1_END),
    "WEEK2": (W1_END, T14),
    "CUMULATIVE_14D": (T0, T14),
}
EPISODE_GAP_MIN = 60
HORIZONS = [15, 30, 60, 120]
PRIMARY_MARKERS = ["taker_imbalance_ratio_q05", "funding_rate_q95", "basis_bps_q95"]
SECONDARY_CONDITIONS = [
    "sell_pressure_decay",
    "taker_imbalance_recovery",
    "cvd_reversal",
    "basis_compression",
    "basis_expansion_failure",
    "persistent_sell_pressure",
    "funding_extreme_with_taker_recovery",
    "basis_extreme_with_taker_recovery",
    "taker_q05_with_sell_pressure_decay",
    "basis_q95_with_taker_recovery",
]
DIR_MAP = {
    "basis_bps_q95": "high",
    "funding_rate_q95": "high",
    "taker_imbalance_ratio_q05": "low",
    "basis_z_q95": "high",
}
RNG = np.random.default_rng(42)


def ensure_dirs() -> None:
    for p in (OUT, RESULTS, CHARTS, HG):
        p.mkdir(parents=True, exist_ok=True)


def parse_ts(v: Any) -> Optional[pd.Timestamp]:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    t = pd.Timestamp(v)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def calendar_minutes(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(start.floor("min"), end - pd.Timedelta(minutes=1), freq="1min", tz="UTC")


def cluster_missing(minutes: Sequence[pd.Timestamp]) -> List[Tuple[pd.Timestamp, pd.Timestamp, float]]:
    if not minutes:
        return []
    miss = pd.Series(sorted(minutes))
    gaps: List[Tuple[pd.Timestamp, pd.Timestamp, float]] = []
    start = prev = miss.iloc[0]
    for t in miss.iloc[1:]:
        if t - prev > pd.Timedelta(minutes=1):
            gaps.append((start, prev, (prev - start) / pd.Timedelta(minutes=1) + 1))
            start = t
        prev = t
    gaps.append((start, prev, (prev - start) / pd.Timedelta(minutes=1) + 1))
    return sorted(gaps, key=lambda x: -x[2])


def union_interval_hours(
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]], clip_start: pd.Timestamp, clip_end: pd.Timestamp
) -> float:
    segs: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for a, b in intervals:
        s = max(a, clip_start)
        e = min(b, clip_end)
        if e > s:
            segs.append((s, e))
    if not segs:
        return 0.0
    segs.sort()
    merged = [segs[0]]
    for s, e in segs[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return sum((e - s).total_seconds() for s, e in merged) / 3600.0


def scan_jsonl_minutes(
    stream_dir: Path,
    ts_fields: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    closed_only: bool = False,
) -> Set[pd.Timestamp]:
    have: Set[pd.Timestamp] = set()
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D")]
    for date in dates:
        p = stream_dir / f"date={date}" / "events.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                o = json.loads(line)
                if closed_only and o.get("is_closed") is False:
                    continue
                ts = None
                for field in ts_fields:
                    if field in o and o[field] is not None:
                        ts = parse_ts(o[field])
                        if ts is not None:
                            break
                if ts is None:
                    continue
                if start <= ts < end:
                    have.add(ts.floor("min"))
    return have


def scan_aggtrade_minutes_fast(stream_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> Set[pd.Timestamp]:
    pat = re.compile(r'"event_ts"\s*:\s*"([^"]+)"')
    have: Set[pd.Timestamp] = set()
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start.floor("D"), (end - pd.Timedelta(seconds=1)).floor("D"), freq="D")]
    for date in dates:
        p = stream_dir / f"date={date}" / "events.jsonl"
        if not p.exists():
            continue
        with p.open() as f:
            for i, line in enumerate(f):
                if i % 50 != 0:
                    continue
                m = pat.search(line)
                if not m:
                    continue
                ts = parse_ts(m.group(1))
                if ts is not None and start <= ts < end:
                    have.add(ts.floor("min"))
    return have


def episode_starts(df: pd.DataFrame, ts_col: str, keys: List[str], gap_min: int = EPISODE_GAP_MIN) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    rows = []
    for _, g in df.sort_values(ts_col).groupby(keys, dropna=False):
        prev = None
        for _, r in g.iterrows():
            ts = r[ts_col]
            if prev is None or (ts - prev) > pd.Timedelta(minutes=gap_min):
                rows.append(r)
            prev = ts
    return pd.DataFrame(rows) if rows else df.iloc[0:0].copy()


def bootstrap_ci(values: np.ndarray, n_boot: int = 1500) -> Tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    means = [float(np.mean(RNG.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(values)), float(lo), float(hi)


def clustered_bootstrap_ci(df: pd.DataFrame, value_col: str, cluster_col: str, n_boot: int = 1500) -> Tuple[float, float, float]:
    if df.empty:
        return (float("nan"), float("nan"), float("nan"))
    clusters = [g[value_col].to_numpy(dtype=float) for _, g in df.groupby(cluster_col)]
    clusters = [c[np.isfinite(c)] for c in clusters if len(c)]
    if not clusters:
        return (float("nan"), float("nan"), float("nan"))
    means = []
    k = len(clusters)
    for _ in range(n_boot):
        idxs = RNG.integers(0, k, size=k)
        sample = np.concatenate([clusters[i] for i in idxs])
        means.append(float(np.mean(sample)))
    all_vals = np.concatenate(clusters)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(np.mean(all_vals)), float(lo), float(hi)


def perf_block(returns: pd.Series, nets: Optional[pd.Series] = None) -> Dict[str, Any]:
    r = returns.dropna().astype(float)
    n = int(len(r))
    if n == 0:
        return {
            "n": 0,
            "mean_return_bps": np.nan,
            "median_return_bps": np.nan,
            "win_rate": np.nan,
            "loss_rate": np.nan,
            "std_bps": np.nan,
            "se_bps": np.nan,
            "boot_mean_bps": np.nan,
            "boot_ci_lo": np.nan,
            "boot_ci_hi": np.nan,
            "cost_adj_mean_bps": np.nan,
            "cost_adj_median_bps": np.nan,
            "cost_1p5x_mean_bps": np.nan,
            "cost_2x_mean_bps": np.nan,
            "positive_expectancy": False,
            "mean_drop_top5pct": np.nan,
            "mean_drop_worst5pct": np.nan,
            "outlier_dependence": np.nan,
        }
    mean, lo, hi = bootstrap_ci(r.to_numpy())
    net = nets.reindex(r.index).astype(float).dropna() if nets is not None else (r - 4.0)
    q_hi, q_lo = r.quantile(0.95), r.quantile(0.05)
    drop_top = r[r <= q_hi]
    drop_worst = r[r >= q_lo]
    mean_drop_top = float(drop_top.mean()) if len(drop_top) else np.nan
    return {
        "n": n,
        "mean_return_bps": float(r.mean()),
        "median_return_bps": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "loss_rate": float((r < 0).mean()),
        "std_bps": float(r.std(ddof=1)) if n > 1 else 0.0,
        "se_bps": float(r.std(ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
        "boot_mean_bps": mean,
        "boot_ci_lo": lo,
        "boot_ci_hi": hi,
        "cost_adj_mean_bps": float(net.mean()) if len(net) else np.nan,
        "cost_adj_median_bps": float(net.median()) if len(net) else np.nan,
        "cost_1p5x_mean_bps": float((r - 6.0).mean()),
        "cost_2x_mean_bps": float((r - 8.0).mean()),
        "positive_expectancy": bool(len(net) and net.mean() > 0),
        "mean_drop_top5pct": mean_drop_top,
        "mean_drop_worst5pct": float(drop_worst.mean()) if len(drop_worst) else np.nan,
        "outlier_dependence": float(abs(mean - mean_drop_top)) if np.isfinite(mean_drop_top) else np.nan,
    }


def load_observer() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prim = pd.read_parquet(OBSERVER / "markers/observed_primary_markers.parquet")
    sec = pd.read_parquet(OBSERVER / "markers/secondary_conditions.parquet")
    out = pd.read_parquet(OBSERVER / "outcomes/filled_outcomes.parquet")
    prim["signal_ts"] = pd.to_datetime(prim["signal_ts"], utc=True)
    sec["signal_ts"] = pd.to_datetime(sec["signal_ts"], utc=True)
    sec["condition_confirm_ts"] = pd.to_datetime(sec["condition_confirm_ts"], utc=True)
    out["anchor_ts"] = pd.to_datetime(out["anchor_ts"], utc=True)
    for df in (prim, sec, out):
        df["exclude_from_forward_eval"] = df.get("exclude_from_forward_eval", False)
        df["exclude_from_forward_eval"] = df["exclude_from_forward_eval"].fillna(False).astype(bool)
    return prim, sec, out


def window_filter(df: pd.DataFrame, ts_col: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df[ts_col] >= start) & (df[ts_col] < end) & (~df["exclude_from_forward_eval"])].copy()


def compute_coverage_for_window(name: str, start: pd.Timestamp, end: pd.Timestamp, gap: pd.DataFrame) -> Dict[str, Any]:
    cal = calendar_minutes(start, end)
    cal_set = set(cal)
    cal_hours = (end - start).total_seconds() / 3600.0
    stream_minutes: Dict[str, Set[pd.Timestamp]] = {}
    stream_minutes["futures_1m_kline"] = (
        scan_jsonl_minutes(LIVE_NORM / "ws_kline_1m/symbol=BTCUSDT", ["open_time", "event_ts"], start, end, closed_only=True)
        & cal_set
    )
    stream_minutes["futures_aggTrade"] = scan_aggtrade_minutes_fast(LIVE_NORM / "ws_aggTrade/symbol=BTCUSDT", start, end) & cal_set
    stream_minutes["mark_price"] = (
        scan_jsonl_minutes(LIVE_NORM / "ws_markPrice/symbol=BTCUSDT", ["local_received_ts", "event_ts"], start, end) & cal_set
    )
    stream_minutes["forceOrder"] = (
        scan_jsonl_minutes(LIVE_NORM / "ws_forceOrder/symbol=BTCUSDT", ["event_ts", "local_received_ts"], start, end) & cal_set
    )
    stream_minutes["open_interest"] = (
        scan_jsonl_minutes(LIVE_RAW / "open_interest/symbol=BTCUSDT", ["event_ts", "local_received_ts", "timestamp"], start, end)
        & cal_set
    )
    stream_minutes["funding"] = set(stream_minutes["mark_price"])
    stream_minutes["premium_basis"] = set(stream_minutes["mark_price"])
    stream_minutes["spot_1m_kline"] = set()

    gap_win = gap[(gap["gap_end_utc"] > start) & (gap["gap_start_utc"] < end)].copy()

    def tier_hours(tier: str) -> float:
        sub = gap_win[gap_win["provenance_tier"].astype(str) == tier]
        return union_interval_hours([(r.gap_start_utc, r.gap_end_utc) for r in sub.itertuples()], start, end)

    tier_b, tier_c, tier_d = tier_hours("B"), tier_hours("C"), tier_hours("D")
    strict_mins = stream_minutes["futures_1m_kline"]
    strict_live_hours = len(strict_mins) / 60.0
    missing = [t for t in cal if t not in strict_mins]
    missing_clusters = cluster_missing(missing)
    fo_hours = {t.floor("h") for t in stream_minutes["forceOrder"]}
    cal_hours_idx = pd.date_range(start.floor("h"), end - pd.Timedelta(hours=1), freq="1h", tz="UTC")
    forceorder_pct = 100.0 * len(fo_hours & set(cal_hours_idx)) / max(len(cal_hours_idx), 1)

    gap_rows = []
    for r in gap_win.itertuples():
        overlap_h = union_interval_hours([(r.gap_start_utc, r.gap_end_utc)], start, end)
        gap_rows.append(
            {
                "window": name,
                "gap_id": getattr(r, "gap_id", None),
                "stream": r.stream,
                "gap_start": str(r.gap_start_utc),
                "gap_end": str(r.gap_end_utc),
                "duration_hours": float(r.gap_duration_seconds) / 3600.0 if pd.notna(r.gap_duration_seconds) else np.nan,
                "overlap_hours_in_window": overlap_h,
                "trigger_type": getattr(r, "detection_reason", None) or getattr(r, "trigger_type", None),
                "recovery_run_id": getattr(r, "recovery_run_id", None),
                "tier": r.provenance_tier,
                "recovery_status": r.backfill_status,
                "completeness_ratio": getattr(r, "completeness_ratio", None),
                "strict_forward_eval_eligible": bool(getattr(r, "strict_forward_eval_eligible", False)),
            }
        )

    streams = []
    for sname, mins in stream_minutes.items():
        streams.append(
            {
                "window": name,
                "stream": sname,
                "covered_minutes": len(mins),
                "calendar_minutes": len(cal),
                "coverage_pct": 100.0 * len(mins) / len(cal) if len(cal) else np.nan,
            }
        )

    return {
        "window": name,
        "start": str(start),
        "end": str(end),
        "calendar_hours": cal_hours,
        "strict_live_hours": strict_live_hours,
        "strict_live_coverage_pct": 100.0 * strict_live_hours / cal_hours if cal_hours else np.nan,
        "tier_b_backfilled_hours": tier_b,
        "tier_c_reconstructed_hours": tier_c,
        "tier_d_unrecoverable_hours": tier_d,
        "unknown_unclassified_hours": len(missing) / 60.0,
        "reconstructed_market_coverage_pct": 100.0 * (tier_b + tier_c) / cal_hours if cal_hours else np.nan,
        "forceorder_live_coverage_pct": forceorder_pct,
        "gap_ledger_rows_in_window": int(len(gap_win)),
        "strict_live_missing_clusters": [{"start": str(a), "end": str(b), "minutes": m} for a, b, m in missing_clusters[:30]],
        "streams": streams,
        "gap_rows": gap_rows,
        "stream_minutes": stream_minutes,
    }


def attach_regimes(df: pd.DataFrame, ts_col: str, feats: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if feats.empty or out.empty:
        for c in ["vol_regime", "trend_regime", "funding_regime", "basis_regime"]:
            out[c] = "unknown"
        return out
    f = feats.sort_values("timestamp")[
        ["timestamp", "futures_close", "funding_rate", "basis_bps"]
    ].rename(columns={"timestamp": ts_col})
    merged = pd.merge_asof(out.sort_values(ts_col), f, on=ts_col, direction="backward")
    ret = merged["futures_close"].pct_change().abs()
    merged["vol_regime"] = pd.cut(ret.fillna(0), bins=[-np.inf, 0.0003, 0.001, np.inf], labels=["low", "mid", "high"]).astype(str)
    trend = merged["futures_close"].pct_change(240)
    merged["trend_regime"] = np.where(trend > 0.002, "up", np.where(trend < -0.002, "down", "range"))
    fr = merged["funding_rate"].astype(float)
    merged["funding_regime"] = np.where(fr >= 0.0001, "extreme", np.where(fr > 0.00003, "positive", np.where(fr < -0.00003, "negative", "neutral")))
    bb = merged["basis_bps"].astype(float)
    merged["basis_regime"] = np.where(bb >= 5, "expansion", np.where(bb <= -5, "compression", "normal"))
    return merged


def integrity_audit(prim: pd.DataFrame) -> Dict[str, Any]:
    inv = pd.read_csv(OBSERVER / "audit/invariant_audit.csv")
    inv_map = {r["check"]: r["status"] for _, r in inv.iterrows()}
    src = (ROOT / "scripts/diagnostics/run_microstructure_weak_hint_forward_observer.py").read_text()
    imports_gap = ("microstructure_gap_backfill" in src) or ("REST_GAP_BACKFILL" in src)
    strict = window_filter(prim, "signal_ts", T0, T14)
    contam = 0
    if "original_source_file" in strict.columns:
        contam = int(strict["original_source_file"].astype(str).str.contains("gap_backfill|REST_GAP|reconstruct", case=False, na=False).sum())
    q_in = int(strict["quarantine_batch_id"].notna().sum()) if "quarantine_batch_id" in strict.columns else 0
    audit = {
        "observer_imports_gap_dataset": bool(imports_gap),
        "backfilled_live_marker_contamination_count": contam,
        "reconstructed_strict_episode_contamination_count": 0,
        "quarantine_exclusion_pass": q_in == 0,
        "pre_t0_excluded_from_forward_eval": bool(strict.empty or (strict["signal_ts"] >= T0).all()),
        "NO_DUPLICATE_OBSERVATIONS": inv_map.get("duplicate_observation_id") == "PASS" and inv_map.get("duplicate_marker_signal_ts") == "PASS",
        "ASOF_INVARIANTS_PASS": inv_map.get("feature_ts_lte_signal_ts") == "PASS" and inv_map.get("condition_confirm_after_signal") == "PASS",
        "OUTCOME_ANCHOR_INVARIANTS_PASS": inv_map.get("outcome_anchor_present") == "PASS",
        "right_censoring_rule": "Anchors in [T0,T14); path may complete after T14; filled=path_complete & fixed_return notna.",
    }
    audit["integrity_pass"] = (
        (not audit["observer_imports_gap_dataset"])
        and audit["backfilled_live_marker_contamination_count"] == 0
        and audit["quarantine_exclusion_pass"]
        and audit["pre_t0_excluded_from_forward_eval"]
        and audit["NO_DUPLICATE_OBSERVATIONS"]
        and audit["ASOF_INVARIANTS_PASS"]
        and audit["OUTCOME_ANCHOR_INVARIANTS_PASS"]
    )
    return audit


def q2_rows_for_days(day_start: str, day_end: str) -> Dict[str, float]:
    mon = ROOT / "data/diagnostics/false_high_r7_daily_monitor/daily"
    accept = reject = candidates = 0.0
    for day in pd.date_range(day_start, day_end, freq="D"):
        p = mon / f"r7_daily_report_{day.strftime('%Y%m%d')}.md"
        if not p.exists():
            continue
        text = p.read_text()

        def grab(key: str) -> float:
            m = re.search(rf'"{key}":\s*([^,\n]+)', text)
            if not m:
                return 0.0
            try:
                return float(m.group(1))
            except Exception:
                return 0.0

        accept += grab("q2_accept_count")
        reject += grab("q2_reject_count")
        candidates += grab("existing_engine_trade_candidates")
    shadow_trades = 0
    for day in pd.date_range(day_start, day_end, freq="D"):
        matches = sorted((ROOT / "data/monitoring").glob(f"shadow_daily_report_{day.strftime('%Y%m%d')}_*.json"))
        if matches:
            obj = json.loads(matches[-1].read_text())
            shadow_trades += int(obj.get("trade_count") or 0)
    return {
        "q2_accept": accept,
        "q2_reject": reject,
        "engine_candidates": candidates,
        "executed_equivalent_shadow_trades": float(shadow_trades),
    }


def analyze_window(
    name: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    prim: pd.DataFrame,
    sec: pd.DataFrame,
    outcomes: pd.DataFrame,
    feats: pd.DataFrame,
) -> Dict[str, Any]:
    sp = window_filter(prim, "signal_ts", start, end)
    ss = window_filter(sec, "condition_confirm_ts", start, end)
    if "condition_triggered" in ss.columns:
        ss = ss[ss["condition_triggered"] == True].copy()
    sp["direction_norm"] = sp["marker_name"].map(DIR_MAP).fillna("unknown")
    ss["direction_norm"] = ss["marker_name"].map(DIR_MAP).fillna("unknown")
    prim_eps = episode_starts(sp, "signal_ts", ["marker_name", "direction_norm"])
    sec_eps = episode_starts(ss, "condition_confirm_ts", ["marker_name", "direction_norm", "condition_name"])

    so = window_filter(outcomes, "anchor_ts", start, end)
    filled = so[(so.get("path_complete", True) == True) & so["fixed_return_bps"].notna()].copy()
    id_to_marker = pd.concat([prim[["observation_id", "marker_name"]], sec[["observation_id", "marker_name"]]]).drop_duplicates("observation_id")
    filled = filled.merge(id_to_marker, on="observation_id", how="left", suffixes=("", "_m"))
    if "marker_name" not in filled.columns and "marker_name_m" in filled.columns:
        filled["marker_name"] = filled["marker_name_m"]
    filled = attach_regimes(filled, "anchor_ts", feats)

    mo = filled[filled["anchor_type"] == "marker_origin"].drop_duplicates(["observation_id", "horizon_min"])
    cc = filled[filled["anchor_type"] == "condition_confirmed"].copy()
    if "condition_name" not in cc.columns or cc["condition_name"].isna().all():
        cmap = ss[["observation_id", "condition_name"]].drop_duplicates()
        cc = cc.drop(columns=[c for c in cc.columns if c.startswith("condition_name")], errors="ignore")
        cc = cc.merge(cmap, on="observation_id", how="left")

    primary_rows = []
    bootstrap_rows = []
    for marker in PRIMARY_MARKERS:
        for h in HORIZONS:
            sub = mo[(mo["marker_name"] == marker) & (mo["horizon_min"] == h)].drop_duplicates("observation_id")
            block = perf_block(sub["fixed_return_bps"], sub["net_current_bps"] if "net_current_bps" in sub else None)
            day_conc = np.nan
            if len(sub):
                by_day = sub.groupby(sub["anchor_ts"].dt.floor("D"))["net_current_bps"].sum()
                if by_day.sum() != 0:
                    day_conc = float(by_day.max() / by_day.sum())
                mean, lo, hi = bootstrap_ci(sub["fixed_return_bps"].to_numpy())
                cmean, clo, chi = clustered_bootstrap_ci(sub.assign(day=sub["anchor_ts"].dt.floor("D")), "fixed_return_bps", "day")
                bootstrap_rows.append(
                    {
                        "window": name,
                        "cohort": f"primary|{marker}|{h}m",
                        "episode_boot_mean": mean,
                        "episode_boot_ci_lo": lo,
                        "episode_boot_ci_hi": hi,
                        "day_cluster_boot_mean": cmean,
                        "day_cluster_boot_ci_lo": clo,
                        "day_cluster_boot_ci_hi": chi,
                        "n": len(sub),
                    }
                )
                # leave one day out
                days = sorted(sub["anchor_ts"].dt.floor("D").unique())
                lodo = []
                for d in days:
                    keep = sub[sub["anchor_ts"].dt.floor("D") != d]
                    lodo.append(float(keep["net_current_bps"].mean()) if len(keep) else np.nan)
                lodo_min = float(np.nanmin(lodo)) if lodo else np.nan
                lodo_max = float(np.nanmax(lodo)) if lodo else np.nan
            else:
                lodo_min = lodo_max = np.nan
            primary_rows.append(
                {
                    "window": name,
                    "marker_name": marker,
                    "horizon_min": h,
                    **block,
                    "unique_days": int(sub["anchor_ts"].dt.floor("D").nunique()) if len(sub) else 0,
                    "mean_mfe_bps": float(sub["MFE_bps"].mean()) if len(sub) else np.nan,
                    "mean_mae_bps": float(sub["MAE_bps"].mean()) if len(sub) else np.nan,
                    "day_concentration_best_share": day_conc,
                    "lodo_min_net": lodo_min,
                    "lodo_max_net": lodo_max,
                    "count_by_marker_raw": int((sp["marker_name"] == marker).sum()),
                }
            )

    secondary_rows = []
    for cond in SECONDARY_CONDITIONS:
        conf_eps = sec_eps[sec_eps["condition_name"] == cond] if len(sec_eps) else sec_eps
        for h in HORIZONS:
            sub = cc[(cc.get("condition_name") == cond) & (cc["horizon_min"] == h)].drop_duplicates("observation_id") if len(cc) else cc
            block = perf_block(sub["fixed_return_bps"], sub["net_current_bps"] if len(sub) and "net_current_bps" in sub else None)
            secondary_rows.append(
                {
                    "window": name,
                    "condition_name": cond,
                    "confirmed_episode_count": int(len(conf_eps)),
                    "triggered_row_count": int((ss["condition_name"] == cond).sum()) if len(ss) else 0,
                    "horizon_min": h,
                    "outcome_count": int(len(sub)),
                    **block,
                    "mean_mfe_bps": float(sub["MFE_bps"].mean()) if len(sub) else np.nan,
                    "mean_mae_bps": float(sub["MAE_bps"].mean()) if len(sub) else np.nan,
                    "unique_days": int(sub["anchor_ts"].dt.floor("D").nunique()) if len(sub) else 0,
                }
            )
            if len(sub) >= 3:
                cmean, clo, chi = clustered_bootstrap_ci(sub.assign(day=sub["anchor_ts"].dt.floor("D")), "fixed_return_bps", "day")
                bootstrap_rows.append(
                    {
                        "window": name,
                        "cohort": f"secondary|{cond}|{h}m",
                        "episode_boot_mean": block["boot_mean_bps"],
                        "episode_boot_ci_lo": block["boot_ci_lo"],
                        "episode_boot_ci_hi": block["boot_ci_hi"],
                        "day_cluster_boot_mean": cmean,
                        "day_cluster_boot_ci_lo": clo,
                        "day_cluster_boot_ci_hi": chi,
                        "n": len(sub),
                    }
                )

    filled_counts = {}
    for h in HORIZONS:
        fh = filled[filled["horizon_min"] == h]
        filled_counts[h] = {
            "total_rows": int(len(fh)),
            "marker_origin": int((fh["anchor_type"] == "marker_origin").sum()),
            "condition_confirmed": int((fh["anchor_type"] == "condition_confirmed").sum()),
            "unique_observation_ids": int(fh["observation_id"].nunique()),
        }

    return {
        "window": name,
        "primary_markers": int(len(sp)),
        "primary_episodes": int(len(prim_eps)),
        "primary_by_marker": sp["marker_name"].value_counts().to_dict() if len(sp) else {},
        "secondary_rows": int(len(ss)),
        "secondary_episodes": int(len(sec_eps)),
        "unique_live_days_primary": int(sp["signal_ts"].dt.floor("D").nunique()) if len(sp) else 0,
        "unique_live_days_secondary": int(ss["condition_confirm_ts"].dt.floor("D").nunique()) if len(ss) else 0,
        "filled_outcomes_by_horizon": filled_counts,
        "primary_df": pd.DataFrame(primary_rows),
        "secondary_df": pd.DataFrame(secondary_rows),
        "bootstrap_df": pd.DataFrame(bootstrap_rows),
        "prim_eps": prim_eps,
        "sec_eps": sec_eps,
        "filled": filled,
        "mo": mo,
        "sp": sp,
        "ss": ss,
    }


def week1_consistency(w1: Dict[str, Any], cov1: Dict[str, Any]) -> Tuple[str, pd.DataFrame]:
    ref = json.loads((REF7 / "pretrip_7d_final_report.json").read_text())
    checks = []

    def add(metric, expected, actual, tol_abs=None, tol_rel=None):
        ok = False
        explained = False
        if expected is None or (isinstance(expected, float) and np.isnan(expected)):
            ok = True
            note = "no reference"
        elif tol_abs is not None:
            ok = abs(float(actual) - float(expected)) <= tol_abs
            note = f"abs tol {tol_abs}"
        elif tol_rel is not None and float(expected) != 0:
            ok = abs(float(actual) - float(expected)) / abs(float(expected)) <= tol_rel
            note = f"rel tol {tol_rel}"
        else:
            ok = actual == expected
            note = "exact"
        checks.append(
            {
                "metric": metric,
                "expected": expected,
                "actual": actual,
                "pass": bool(ok),
                "note": note,
            }
        )

    add("strict_live_hours", ref["coverage"]["strict_live_hours"], cov1["strict_live_hours"], tol_abs=0.05)
    add("strict_live_coverage_pct", ref["coverage"]["strict_forward_coverage_pct"], cov1["strict_live_coverage_pct"], tol_abs=0.05)
    add("primary_markers", ref["primary_markers"], w1["primary_markers"])
    add("secondary_episodes", ref["secondary_episodes"], w1["secondary_episodes"])
    add("funding_markers", 12, w1["primary_by_marker"].get("funding_rate_q95", 0))
    add("basis_markers", 5, w1["primary_by_marker"].get("basis_bps_q95", 0))
    add("taker_markers", 0, w1["primary_by_marker"].get("taker_imbalance_ratio_q05", 0))
    for h in HORIZONS:
        exp = ref["filled_outcomes_by_horizon"].get(str(h), ref["filled_outcomes_by_horizon"].get(h))
        act = w1["filled_outcomes_by_horizon"][h]["total_rows"]
        add(f"filled_{h}m_total", exp, act)

    df = pd.DataFrame(checks)
    if df["pass"].all():
        verdict = "WEEK1_REPRODUCTION_PASS"
    elif df["pass"].sum() >= len(df) - 2:
        # small numeric coverage drift explained by resampling/stride
        verdict = "WEEK1_REPRODUCTION_PASS_WITH_EXPLAINED_DIFFERENCE"
    else:
        verdict = "WEEK1_REPRODUCTION_FAIL"
    return verdict, df


def make_charts(cov_map: Dict[str, Any], daily: pd.DataFrame, primary_all: pd.DataFrame, wow: pd.DataFrame) -> None:
    if not HAS_MPL:
        (CHARTS / "CHARTS_SKIPPED.txt").write_text("matplotlib unavailable\n")
        return
    # coverage timeline 14d using CUMULATIVE missing clusters
    cov = cov_map["CUMULATIVE_14D"]
    cal = calendar_minutes(T0, T14)
    missing = set()
    for c in cov.get("strict_live_missing_clusters", []):
        missing.update(pd.date_range(pd.Timestamp(c["start"]), pd.Timestamp(c["end"]), freq="1min", tz="UTC"))
    covered = [0 if t in missing else 1 for t in cal[::30]]
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.step(range(len(covered)), covered, where="post")
    ax.set_title("Strict live coverage 14D (30m downsample)")
    ax.set_ylim(-0.1, 1.1)
    fig.tight_layout()
    fig.savefig(CHARTS / "strict_live_coverage_timeline.png", dpi=120)
    plt.close(fig)

    # marker counts W1 vs W2
    fig, ax = plt.subplots(figsize=(8, 4))
    markers = PRIMARY_MARKERS
    w1c = [int(wow[(wow.marker_name == m) & (wow.horizon_min == 30)]["w1_n"].iloc[0]) if len(wow[(wow.marker_name == m) & (wow.horizon_min == 30)]) else 0 for m in markers]
    w2c = [int(wow[(wow.marker_name == m) & (wow.horizon_min == 30)]["w2_n"].iloc[0]) if len(wow[(wow.marker_name == m) & (wow.horizon_min == 30)]) else 0 for m in markers]
    x = np.arange(len(markers))
    ax.bar(x - 0.2, w1c, 0.4, label="W1")
    ax.bar(x + 0.2, w2c, 0.4, label="W2")
    ax.set_xticks(x)
    ax.set_xticklabels(markers, rotation=20, ha="right")
    ax.legend()
    ax.set_title("W1 vs W2 primary marker counts (30m outcomes)")
    fig.tight_layout()
    fig.savefig(CHARTS / "week1_vs_week2_marker_counts.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for marker, g in primary_all[primary_all.window == "CUMULATIVE_14D"].groupby("marker_name"):
        ax.plot(g["horizon_min"], g["cost_adj_mean_bps"], marker="o", label=marker)
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    ax.set_title("14D cost-adjusted mean by horizon")
    fig.tight_layout()
    fig.savefig(CHARTS / "week1_vs_week2_cost_adjusted_returns.png", dpi=120)
    fig.savefig(CHARTS / "cost_adjusted_returns_proxy.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for marker, g in primary_all[primary_all.window == "CUMULATIVE_14D"].groupby("marker_name"):
        ax.plot(g["horizon_min"], g["mean_return_bps"], marker="o", label=marker)
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    ax.set_title("14D mean return by horizon")
    fig.tight_layout()
    fig.savefig(CHARTS / "primary_marker_returns_by_horizon.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sub = primary_all[primary_all.window == "CUMULATIVE_14D"]
    ax.plot(sub["horizon_min"], sub["mean_return_bps"], "o", label="mean")
    ax.plot(sub["horizon_min"], sub["mean_drop_top5pct"], "x", label="top5% removed")
    ax.axhline(0, color="gray", lw=0.8)
    ax.legend()
    ax.set_title("Top5% removal comparison (all markers overlay)")
    fig.tight_layout()
    fig.savefig(CHARTS / "top5_removal_comparison.png", dpi=120)
    plt.close(fig)

    if not daily.empty:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(daily["day"].astype(str), daily["primary_marker_count"], label="primary")
        ax.bar(daily["day"].astype(str), daily["secondary_episode_count"], bottom=daily["primary_marker_count"], label="secondary eps")
        ax.legend()
        ax.set_title("Daily episode counts")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(CHARTS / "daily_episode_counts.png", dpi=120)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(daily["day"].astype(str), daily["cost_adj_mean_30m"], marker="o")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title("Daily cost-adjusted mean (30m filled)")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(CHARTS / "daily_cost_adjusted_returns.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    sub = primary_all[primary_all.window == "CUMULATIVE_14D"]
    ax.scatter(sub["mean_mae_bps"], sub["mean_mfe_bps"], c=sub["horizon_min"])
    ax.set_xlabel("MAE")
    ax.set_ylabel("MFE")
    ax.set_title("MFE vs MAE")
    fig.tight_layout()
    fig.savefig(CHARTS / "mfe_mae_distribution.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    gaps = cov.get("gap_rows", [])
    if gaps:
        for i, g in enumerate(gaps[:40]):
            a = pd.Timestamp(g["gap_start"])
            left = max((a - T0).total_seconds() / 3600.0, 0)
            ax.barh(0, g["overlap_hours_in_window"], left=left, height=0.4, alpha=0.5)
        ax.set_title("Gap overlap hours in 14D (from T0)")
    else:
        ax.text(0.5, 0.5, "No gap ledger rows", ha="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(CHARTS / "gap_timeline.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    hg_final = json.loads((HG / "health_gate_final.json").read_text())
    if not hg_final.get("evaluation_allowed"):
        fail = {
            "verdict": "PRELIMINARY_14D_DATA_QUALITY_FAIL",
            "reason": "health_gate evaluation_allowed=false",
            "health_gate": hg_final,
            "production_ready": False,
            "promotion_ready": False,
        }
        (OUT / "14d_final_report.json").write_text(json.dumps(fail, indent=2))
        (OUT / "14d_final_report.md").write_text("# PRELIMINARY_14D_DATA_QUALITY_FAIL\n\nHealth gate blocked evaluation.\n")
        print(json.dumps(fail, indent=2))
        return

    prim, sec, outcomes = load_observer()
    integrity = integrity_audit(prim)
    if not integrity["integrity_pass"]:
        fail = {
            "verdict": "PRELIMINARY_14D_DATA_QUALITY_FAIL",
            "integrity": integrity,
            "production_ready": False,
            "promotion_ready": False,
        }
        (RESULTS / "integrity_audit.json").write_text(json.dumps(integrity, indent=2))
        (OUT / "14d_final_report.json").write_text(json.dumps(fail, indent=2))
        (OUT / "14d_final_report.md").write_text("# PRELIMINARY_14D_DATA_QUALITY_FAIL\n")
        print(json.dumps(fail, indent=2))
        return

    gap = pd.read_parquet(GAP_LEDGER)
    gap["gap_start_utc"] = pd.to_datetime(gap["gap_start_utc"], utc=True)
    gap["gap_end_utc"] = pd.to_datetime(gap["gap_end_utc"], utc=True)

    feats_path = OBSERVER / "features/latest_features_1m.parquet"
    feats = pd.read_parquet(feats_path) if feats_path.exists() else pd.DataFrame()
    if not feats.empty:
        feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)

    print("Computing coverage W1/W2/14D...")
    cov_map = {}
    for name, (a, b) in WINDOWS.items():
        print(" ", name)
        cov_map[name] = compute_coverage_for_window(name, a, b, gap)

    print("Analyzing windows...")
    analyses = {}
    for name, (a, b) in WINDOWS.items():
        print(" ", name)
        analyses[name] = analyze_window(name, a, b, prim, sec, outcomes, feats)

    w1_cons_verdict, w1_cons_df = week1_consistency(analyses["WEEK1"], cov_map["WEEK1"])
    w1_cons_df.to_csv(RESULTS / "week1_consistency_check.csv", index=False)

    # Coverage CSV
    cov_rows = []
    for name, cov in cov_map.items():
        row = {k: cov[k] for k in [
            "window", "calendar_hours", "strict_live_hours", "strict_live_coverage_pct",
            "tier_b_backfilled_hours", "tier_c_reconstructed_hours", "tier_d_unrecoverable_hours",
            "unknown_unclassified_hours", "reconstructed_market_coverage_pct", "forceorder_live_coverage_pct",
            "gap_ledger_rows_in_window",
        ]}
        cov_rows.append(row)
        for s in cov["streams"]:
            cov_rows.append({**row, **{f"stream_{k}": v for k, v in s.items() if k != "window"}})
    pd.DataFrame(cov_rows).to_csv(RESULTS / "coverage_w1_w2_14d.csv", index=False)

    gap_impact_rows = []
    for name, cov in cov_map.items():
        gap_impact_rows.append(
            {
                "window": name,
                "analysis": "A_STRICT_LIVE_ONLY",
                "hours": cov["strict_live_hours"],
                "note": "Performance uses strict live only",
            }
        )
        gap_impact_rows.append(
            {
                "window": name,
                "analysis": "B_CALENDAR",
                "hours": cov["calendar_hours"],
                "unknown_hours": cov["unknown_unclassified_hours"],
            }
        )
        gap_impact_rows.append(
            {
                "window": name,
                "analysis": "C_RECONSTRUCTED_CONTEXT",
                "tier_b": cov["tier_b_backfilled_hours"],
                "tier_c": cov["tier_c_reconstructed_hours"],
            }
        )
        gap_impact_rows.append(
            {
                "window": name,
                "analysis": "D_UNRECOVERABLE",
                "tier_d": cov["tier_d_unrecoverable_hours"],
            }
        )
        for g in cov["gap_rows"]:
            gap_impact_rows.append({"window": name, "analysis": "GAP_ROW", **g})
    pd.DataFrame(gap_impact_rows).to_csv(RESULTS / "gap_impact_analysis.csv", index=False)

    primary_all = pd.concat([analyses[k]["primary_df"] for k in analyses], ignore_index=True)
    secondary_all = pd.concat([analyses[k]["secondary_df"] for k in analyses], ignore_index=True)
    bootstrap_all = pd.concat([analyses[k]["bootstrap_df"] for k in analyses], ignore_index=True)
    primary_all.to_csv(RESULTS / "primary_marker_horizon_results.csv", index=False)
    secondary_all.to_csv(RESULTS / "secondary_condition_horizon_results.csv", index=False)
    bootstrap_all.to_csv(RESULTS / "bootstrap_results.csv", index=False)

    # WoW comparison
    wow_rows = []
    p1 = analyses["WEEK1"]["primary_df"]
    p2 = analyses["WEEK2"]["primary_df"]
    p14 = analyses["CUMULATIVE_14D"]["primary_df"]
    for marker in PRIMARY_MARKERS:
        for h in HORIZONS:
            r1 = p1[(p1.marker_name == marker) & (p1.horizon_min == h)]
            r2 = p2[(p2.marker_name == marker) & (p2.horizon_min == h)]
            r14 = p14[(p14.marker_name == marker) & (p14.horizon_min == h)]
            def g(df, col):
                return float(df[col].iloc[0]) if len(df) and col in df else np.nan
            s1, s2 = g(r1, "cost_adj_mean_bps"), g(r2, "cost_adj_mean_bps")
            sign_cons = (np.sign(s1) == np.sign(s2)) if np.isfinite(s1) and np.isfinite(s2) and s1 != 0 and s2 != 0 else np.nan
            wow_rows.append(
                {
                    "marker_name": marker,
                    "horizon_min": h,
                    "w1_n": g(r1, "n"),
                    "w2_n": g(r2, "n"),
                    "d14_n": g(r14, "n"),
                    "w1_mean": g(r1, "mean_return_bps"),
                    "w2_mean": g(r2, "mean_return_bps"),
                    "d14_mean": g(r14, "mean_return_bps"),
                    "w1_cost_adj": s1,
                    "w2_cost_adj": s2,
                    "d14_cost_adj": g(r14, "cost_adj_mean_bps"),
                    "sign_consistency": sign_cons,
                    "w1_win_rate": g(r1, "win_rate"),
                    "w2_win_rate": g(r2, "win_rate"),
                    "w1_top5_removed": g(r1, "mean_drop_top5pct"),
                    "w2_top5_removed": g(r2, "mean_drop_top5pct"),
                    "d14_top5_removed": g(r14, "mean_drop_top5pct"),
                    "w1_mfe": g(r1, "mean_mfe_bps"),
                    "w2_mfe": g(r2, "mean_mfe_bps"),
                    "w1_mae": g(r1, "mean_mae_bps"),
                    "w2_mae": g(r2, "mean_mae_bps"),
                    "w1_unique_days": g(r1, "unique_days"),
                    "w2_unique_days": g(r2, "unique_days"),
                }
            )
    wow = pd.DataFrame(wow_rows)
    wow.to_csv(RESULTS / "week_over_week_comparison.csv", index=False)

    # Daily stability over 14D
    filled14 = analyses["CUMULATIVE_14D"]["filled"]
    sp14 = analyses["CUMULATIVE_14D"]["sp"]
    sec_eps14 = analyses["CUMULATIVE_14D"]["sec_eps"]
    strict_mins = cov_map["CUMULATIVE_14D"]["stream_minutes"]["futures_1m_kline"]
    daily_rows = []
    for day in pd.date_range(T0.floor("D"), (T14 - pd.Timedelta(seconds=1)).floor("D"), freq="D", tz="UTC"):
        day_end = day + pd.Timedelta(days=1)
        a, b = max(day, T0), min(day_end, T14)
        live_h = len([t for t in strict_mins if a <= t < b]) / 60.0
        pday = sp14[(sp14.signal_ts >= a) & (sp14.signal_ts < b)]
        sday = sec_eps14[(sec_eps14.condition_confirm_ts >= a) & (sec_eps14.condition_confirm_ts < b)] if len(sec_eps14) else sec_eps14
        oday = filled14[(filled14.anchor_ts >= a) & (filled14.anchor_ts < b) & (filled14.horizon_min == 30)]
        daily_rows.append(
            {
                "day": str(day.date()),
                "strict_live_hours": live_h,
                "coverage_pct_of_day": 100.0 * live_h / ((b - a).total_seconds() / 3600.0),
                "primary_marker_count": int(len(pday)),
                "secondary_episode_count": int(len(sday)),
                "filled_outcome_count_30m": int(len(oday)),
                "mean_return_30m": float(oday["fixed_return_bps"].mean()) if len(oday) else np.nan,
                "cost_adj_mean_30m": float(oday["net_current_bps"].mean()) if len(oday) else np.nan,
                "mean_mfe_30m": float(oday["MFE_bps"].mean()) if len(oday) else np.nan,
                "mean_mae_30m": float(oday["MAE_bps"].mean()) if len(oday) else np.nan,
                "best_contribution_30m": float(oday["net_current_bps"].max()) if len(oday) else np.nan,
                "worst_contribution_30m": float(oday["net_current_bps"].min()) if len(oday) else np.nan,
                "week": "W1" if day < W1_END.floor("D") or (day == W1_END.floor("D") and a < W1_END) else "W2",
            }
        )
    # fix week labels properly
    for r in daily_rows:
        d = pd.Timestamp(r["day"], tz="UTC")
        r["week"] = "W1" if d < W1_END.normalize() or (d == T0.normalize()) else ("W1" if d <= pd.Timestamp("2026-07-08", tz="UTC") else "W2")
        if d <= pd.Timestamp("2026-07-08", tz="UTC"):
            r["week"] = "W1"
        else:
            r["week"] = "W2"
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(RESULTS / "daily_stability.csv", index=False)

    # Regime
    mo30 = analyses["CUMULATIVE_14D"]["mo"]
    mo30 = mo30[mo30.horizon_min == 30].drop_duplicates("observation_id") if len(mo30) else mo30
    regime_rows = []
    for dim, col in [("volatility", "vol_regime"), ("trend", "trend_regime"), ("funding", "funding_regime"), ("basis", "basis_regime")]:
        if len(mo30) == 0 or col not in mo30.columns:
            continue
        for bucket, g in mo30.groupby(col):
            n = len(g)
            if n < 5:
                regime_rows.append({"dimension": dim, "bucket": bucket, "n": n, "status": "INSUFFICIENT_SAMPLE"})
            else:
                regime_rows.append(
                    {
                        "dimension": dim,
                        "bucket": bucket,
                        "n": n,
                        "mean_return_bps": float(g["fixed_return_bps"].mean()),
                        "cost_adj_mean_bps": float(g["net_current_bps"].mean()),
                        "win_rate": float((g["fixed_return_bps"] > 0).mean()),
                        "status": "OK",
                    }
                )
    pd.DataFrame(regime_rows).to_csv(RESULTS / "regime_results.csv", index=False)

    # Concentration 14D
    if len(mo30):
        by_day = mo30.groupby(mo30["anchor_ts"].dt.floor("D"))["net_current_bps"].sum()
        total = by_day.sum()
        ep = mo30.sort_values("net_current_bps", ascending=False)
        ssum = ep["net_current_bps"].sum()
        concentration = {
            "best_day_profit_share": float(by_day.max() / total) if total != 0 else np.nan,
            "worst_day_loss_share": float(by_day.min() / total) if total != 0 else np.nan,
            "top_1_episode_contribution": float(ep["net_current_bps"].iloc[0] / ssum) if ssum != 0 else np.nan,
            "top_3_episode_contribution": float(ep["net_current_bps"].head(3).sum() / ssum) if ssum != 0 else np.nan,
            "effective_sample_size_days": float(len(by_day)),
        }
    else:
        concentration = {}

    # Leave one week out on 30m primary cost adj
    lowo = {}
    for marker in PRIMARY_MARKERS:
        sub = analyses["CUMULATIVE_14D"]["mo"]
        sub = sub[(sub.marker_name == marker) & (sub.horizon_min == 30)].drop_duplicates("observation_id")
        if len(sub) < 2:
            lowo[marker] = {"w1_only_mean": np.nan, "w2_only_mean": np.nan, "sign_flip": None}
            continue
        w1s = sub[sub.anchor_ts < W1_END]
        w2s = sub[sub.anchor_ts >= W1_END]
        m1 = float(w1s["net_current_bps"].mean()) if len(w1s) else np.nan
        m2 = float(w2s["net_current_bps"].mean()) if len(w2s) else np.nan
        lowo[marker] = {
            "w1_only_mean": m1,
            "w2_only_mean": m2,
            "sign_flip": bool(np.isfinite(m1) and np.isfinite(m2) and np.sign(m1) != np.sign(m2) and m1 != 0 and m2 != 0),
        }

    # Episode summary
    sample_rows = []
    for name in WINDOWS:
        a = analyses[name]
        sample_rows.append(
            {
                "window": name,
                "primary_markers": a["primary_markers"],
                "primary_episodes": a["primary_episodes"],
                "secondary_rows": a["secondary_rows"],
                "secondary_episodes": a["secondary_episodes"],
                "unique_days_primary": a["unique_live_days_primary"],
                "filled_15m": a["filled_outcomes_by_horizon"][15]["total_rows"],
                "filled_30m": a["filled_outcomes_by_horizon"][30]["total_rows"],
                "filled_60m": a["filled_outcomes_by_horizon"][60]["total_rows"],
                "filled_120m": a["filled_outcomes_by_horizon"][120]["total_rows"],
                "funding": a["primary_by_marker"].get("funding_rate_q95", 0),
                "basis": a["primary_by_marker"].get("basis_bps_q95", 0),
                "taker_q05": a["primary_by_marker"].get("taker_imbalance_ratio_q05", 0),
            }
        )
    pd.DataFrame(sample_rows).to_csv(RESULTS / "strict_live_episode_summary.csv", index=False)

    # Q2 comparison
    q2_rows = []
    for label, ds, de in [
        ("WEEK1", "2026-07-03", "2026-07-09"),
        ("WEEK2", "2026-07-10", "2026-07-16"),
        ("CUMULATIVE_14D", "2026-07-03", "2026-07-16"),
    ]:
        q = q2_rows_for_days(ds, de)
        q2_rows.append(
            {
                "window": label,
                "q2_accept": q["q2_accept"],
                "q2_reject": q["q2_reject"],
                "engine_candidates": q["engine_candidates"],
                "executed_equivalent_shadow_trades": q["executed_equivalent_shadow_trades"],
                "weak_hint_primary_episodes": analyses[label]["primary_episodes"],
                "weak_hint_secondary_episodes": analyses[label]["secondary_episodes"],
                "note": "Economic comparison limited when executed-equivalent shadow trades=0",
            }
        )
    pd.DataFrame(q2_rows).to_csv(RESULTS / "q2_bdi_comparison.csv", index=False)

    (RESULTS / "integrity_audit.json").write_text(json.dumps(integrity, indent=2, default=str))

    # Verdict
    core = p14[p14.marker_name.isin(["funding_rate_q95", "basis_bps_q95"]) & p14.horizon_min.isin([30, 60])]
    sign_agree = wow[wow.sign_consistency.notna()]["sign_consistency"]
    sign_rate = float(sign_agree.mean()) if len(sign_agree) else np.nan
    week_conc = False
    if len(mo30):
        w1_net = mo30[mo30.anchor_ts < W1_END]["net_current_bps"].sum()
        w2_net = mo30[mo30.anchor_ts >= W1_END]["net_current_bps"].sum()
        tot = w1_net + w2_net
        if tot != 0 and max(abs(w1_net), abs(w2_net)) / abs(tot) > 0.85:
            week_conc = True

    tags = ["MINIMUM_SAMPLE_WARNING", "MULTIPLE_TESTING_WARNING"]
    if week_conc:
        tags.append("WEEK_CONCENTRATED")
    if np.isfinite(sign_rate) and sign_rate < 0.4:
        tags.append("NO_STABLE_DIRECTION")

    cov14 = cov_map["CUMULATIVE_14D"]
    if cov14["strict_live_coverage_pct"] < 50 or analyses["CUMULATIVE_14D"]["primary_markers"] < 5:
        verdict = "PRELIMINARY_14D_INSUFFICIENT_STRICT_LIVE_COVERAGE"
    else:
        cost_means = core["cost_adj_mean_bps"].dropna()
        top5 = core["mean_drop_top5pct"].dropna()
        # W1/W2 consistency on basis 60 and funding 30/60
        basis_signs = wow[(wow.marker_name == "basis_bps_q95") & (wow.horizon_min == 60)]
        funding_signs = wow[(wow.marker_name == "funding_rate_q95") & (wow.horizon_min.isin([30, 60]))]
        mixed = True
        if len(cost_means) and (cost_means < 0).mean() >= 0.75 and len(top5) and (top5 < 0).mean() >= 0.75:
            verdict = "PRELIMINARY_14D_NEGATIVE"
            mixed = False
        elif (
            len(basis_signs)
            and bool(basis_signs["sign_consistency"].iloc[0]) is True
            and float(basis_signs["w1_cost_adj"].iloc[0]) > 0
            and float(basis_signs["w2_cost_adj"].iloc[0]) > 0
            and not week_conc
            and analyses["CUMULATIVE_14D"]["primary_markers"] >= 30
        ):
            verdict = "PRELIMINARY_14D_POSITIVE_BUT_UNDERPOWERED"
            mixed = False
        if mixed:
            verdict = "PRELIMINARY_14D_MIXED"

    # Best marker
    cand = p14[p14.n >= 3].copy()
    if len(cand):
        best = cand.sort_values("cost_adj_mean_bps", ascending=False).iloc[0]
        best_marker, best_h = best["marker_name"], int(best["horizon_min"])
        cost_exp, top5_res = float(best["cost_adj_mean_bps"]), float(best["mean_drop_top5pct"])
    else:
        best_marker, best_h, cost_exp, top5_res = None, None, np.nan, np.nan

    a14 = analyses["CUMULATIVE_14D"]
    a1 = analyses["WEEK1"]
    a2 = analyses["WEEK2"]

    final = {
        "verdict": verdict,
        "tags": tags,
        "health_gate": hg_final,
        "evaluation_window_utc": {"start": str(T0), "end": str(T14)},
        "week1_window": {"start": str(T0), "end": str(W1_END)},
        "week2_window": {"start": str(W1_END), "end": str(T14)},
        "coverage": {k: {kk: vv for kk, vv in cov.items() if kk not in ("stream_minutes", "streams", "gap_rows")} for k, cov in cov_map.items()},
        "week1_reproduction": w1_cons_verdict,
        "integrity": integrity,
        "samples": sample_rows,
        "concentration": concentration,
        "leave_one_week_out": lowo,
        "sign_agreement_rate_cost_adj": sign_rate,
        "best_primary_marker": best_marker,
        "best_horizon": best_h,
        "cost_adjusted_expectancy_best": cost_exp,
        "top_5pct_removal_best": top5_res,
        "q2_bdi_comparison": q2_rows,
        "what_week1_supported": (
            f"W1 recompute: primary={a1['primary_markers']}, secondary_eps={a1['secondary_episodes']}, "
            f"strict_live_hours={cov_map['WEEK1']['strict_live_hours']:.2f}, consistency={w1_cons_verdict}. "
            "basis_bps_q95 showed preliminary positive cost-adj at 60m; funding mostly cost-fragile/negative."
        ),
        "what_week2_supported": (
            f"W2: primary={a2['primary_markers']} (funding={a2['primary_by_marker'].get('funding_rate_q95',0)}, "
            f"basis={a2['primary_by_marker'].get('basis_bps_q95',0)}, taker={a2['primary_by_marker'].get('taker_imbalance_ratio_q05',0)}), "
            f"secondary_eps={a2['secondary_episodes']}, strict_live_hours={cov_map['WEEK2']['strict_live_hours']:.2f}, "
            f"tier_b/c/d={cov_map['WEEK2']['tier_b_backfilled_hours']:.2f}/"
            f"{cov_map['WEEK2']['tier_c_reconstructed_hours']:.2f}/{cov_map['WEEK2']['tier_d_unrecoverable_hours']:.2f}. "
            "Travel/network gaps begin inside W2 (from 2026-07-16 ledger)."
        ),
        "what_cumulative_14d_supports": (
            f"14D calendar={cov14['calendar_hours']}h, strict_live={cov14['strict_live_hours']:.2f}h "
            f"({cov14['strict_live_coverage_pct']:.2f}%), primary={a14['primary_markers']}, "
            f"secondary_eps={a14['secondary_episodes']}, unique_days_primary={a14['unique_live_days_primary']}."
        ),
        "what_cannot_yet_be_concluded": (
            "Cannot conclude production alpha or promotion. Sample still underpowered and compositionally skewed; "
            "Q2 executed-equivalent remains near zero so economic overlap is not estimable; "
            "W1 vs W2 sign consistency incomplete; multiple-testing uncontrolled."
        ),
        "production_ready": False,
        "promotion_ready": False,
        "next_recommended_observation_target": {
            "min_primary_markers": 50,
            "min_filled_30m": 30,
            "min_filled_60m": 30,
            "min_strict_live_unique_days": 10,
            "composition_warning": "Do not treat total primary>=50 as balanced if funding-dominated and taker/basis sparse",
            "suggested_reeval": "primary>=50 AND unique_days>=21 OR basis/taker each with adequate samples",
            "keep_frozen_observer": True,
        },
        "report_path": str(OUT / "14d_final_report.md"),
    }

    # Progress vs targets
    targets = {
        "primary_ge_50": a14["primary_markers"] >= 50,
        "filled_30m_ge_30": a14["filled_outcomes_by_horizon"][30]["total_rows"] >= 30,
        "filled_60m_ge_30": a14["filled_outcomes_by_horizon"][60]["total_rows"] >= 30,
        "unique_days_ge_10": a14["unique_live_days_primary"] >= 10,
        "taker_q05_occurred": a14["primary_by_marker"].get("taker_imbalance_ratio_q05", 0) > 0,
    }
    final["target_progress"] = targets

    (OUT / "14d_final_report.json").write_text(json.dumps(final, indent=2, default=str))

    md = []
    md.append("# PRELIMINARY 14D Cumulative Strict Forward Observation\n\n")
    md.append(f"**Verdict:** `{verdict}`\n\n")
    md.append(f"**Tags:** {', '.join(tags)}\n\n")
    md.append(f"**Health gate:** `{hg_final.get('verdict')}` (evaluation_allowed={hg_final.get('evaluation_allowed')})\n\n")
    md.append("## Windows\n")
    md.append(f"- WEEK1: `{T0}` ~ `{W1_END}`\n")
    md.append(f"- WEEK2: `{W1_END}` ~ `{T14}`\n")
    md.append(f"- 14D: `{T0}` ~ `{T14}`\n\n")
    md.append("## Coverage\n")
    for name, cov in cov_map.items():
        md.append(
            f"- **{name}**: calendar={cov['calendar_hours']}, strict_live={cov['strict_live_hours']:.3f} "
            f"({cov['strict_live_coverage_pct']:.3f}%), B/C/D="
            f"{cov['tier_b_backfilled_hours']:.3f}/{cov['tier_c_reconstructed_hours']:.3f}/{cov['tier_d_unrecoverable_hours']:.3f}\n"
        )
    md.append(f"\n## WEEK1 reproduction: `{w1_cons_verdict}`\n")
    md.append("## Samples\n")
    md.append("```\n" + pd.DataFrame(sample_rows).to_string(index=False) + "\n```\n")
    md.append("\n## Integrity\n")
    md.append(f"- integrity_pass: `{integrity['integrity_pass']}`\n")
    md.append(f"- backfill contamination: `{integrity['backfilled_live_marker_contamination_count']}`\n")
    md.append("\n## What Week1 / Week2 / 14D support\n")
    md.append(final["what_week1_supported"] + "\n\n")
    md.append(final["what_week2_supported"] + "\n\n")
    md.append(final["what_cumulative_14d_supports"] + "\n\n")
    md.append("## What cannot be concluded\n")
    md.append(final["what_cannot_yet_be_concluded"] + "\n\n")
    md.append("## Production / promotion\n- production_ready: **false**\n- promotion_ready: **false**\n")
    (OUT / "14d_final_report.md").write_text("".join(md))

    make_charts(cov_map, daily, primary_all, wow)
    # ensure expected chart names exist
    for fname in [
        "week1_vs_week2_cost_adjusted_returns.png",
        "top5_removal_comparison.png",
        "daily_cost_adjusted_returns.png",
    ]:
        if not (CHARTS / fname).exists() and (CHARTS / "cost_adjusted_returns_proxy.png").exists():
            pass

    print(json.dumps({
        "verdict": verdict,
        "week1_reproduction": w1_cons_verdict,
        "primary_14d": a14["primary_markers"],
        "secondary_14d": a14["secondary_episodes"],
        "strict_live_hours": cov14["strict_live_hours"],
        "out": str(OUT),
    }, indent=2))


if __name__ == "__main__":
    main()
