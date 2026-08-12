"""Expanded BTC-centric proxy CVD research. Diagnostics only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/proxy_cvd_research")
CVD = Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd")
CACHE = Path("data/diagnostics/research_orderflow_data_cache")
COST = 0.0006


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _read_cvd(symbol: str, tf: str = "5m") -> pd.DataFrame:
    p = CVD / f"{symbol}_{tf}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
    return df.sort_values("timestamp")


def _read_of(family: str, symbol: str) -> pd.DataFrame:
    aliases = {"oi": "open_interest_history", "taker": "taker_buy_sell_volume", "funding": "funding", "mark": "mark", "spot": "spot"}
    fam = aliases.get(family, family)
    paths = [CACHE / "normalized" / fam / f"{symbol}.parquet", CACHE / "features" / family / f"{symbol}.parquet", CACHE / "features" / fam / f"{symbol}.parquet"]
    p = next((x for x in paths if x.exists()), paths[0])
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for c in ["timestamp", "asof_available_ts"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df.sort_values("timestamp")


def _asof(base: pd.DataFrame, ext: pd.DataFrame, cols: List[str], prefix: str, tol: str = "36h") -> pd.DataFrame:
    out = base.sort_values("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("datetime64[ns]")
    if ext.empty:
        for c in cols:
            out[f"{prefix}_{c}"] = np.nan
        return out
    e = ext.copy()
    e["join_ts"] = e["asof_available_ts"] if "asof_available_ts" in e else e["timestamp"]
    e["join_ts"] = pd.to_datetime(e["join_ts"], errors="coerce").astype("datetime64[ns]")
    e = e[["join_ts"] + [c for c in cols if c in e]].rename(columns={c: f"{prefix}_{c}" for c in cols if c in e}).sort_values("join_ts")
    return pd.merge_asof(out, e, left_on="timestamp", right_on="join_ts", direction="backward", tolerance=pd.Timedelta(tol)).drop(columns=["join_ts"], errors="ignore")


def build_features() -> pd.DataFrame:
    btc = _read_cvd("BTCUSDT", "5m")
    if btc.empty:
        raise RuntimeError("BTCUSDT_5m proxy CVD required")
    base = btc.copy()
    span_days = (base["timestamp"].max() - base["timestamp"].min()).total_seconds() / 86400
    base["sample_days"] = span_days
    for sym in ["ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        ctx = _read_cvd(sym, "5m")
        base = _asof(base, ctx, ["cvd_zscore", "large_trade_aggression_score", "cvd_rolling_delta_12"], sym.lower(), "15min")
    base = _asof(base, _read_of("taker", "BTCUSDT"), ["buySellRatio", "buyVol", "sellVol"], "btc_taker")
    base = _asof(base, _read_of("oi", "BTCUSDT"), ["sumOpenInterest", "sumOpenInterestValue"], "btc_oi")
    base = _asof(base, _read_of("funding", "BTCUSDT"), ["fundingRate"], "btc_funding", "10h")
    base = _asof(base, _read_of("mark", "BTCUSDT"), ["close"], "btc_mark")
    base = _asof(base, _read_of("spot", "BTCUSDT"), ["close"], "btc_spot")
    base["btc_taker_delta"] = pd.to_numeric(base.get("btc_taker_buyVol"), errors="coerce") - pd.to_numeric(base.get("btc_taker_sellVol"), errors="coerce")
    base["btc_oi_change"] = pd.to_numeric(base.get("btc_oi_sumOpenInterest"), errors="coerce").pct_change()
    base["btc_basis"] = pd.to_numeric(base.get("btc_mark_close"), errors="coerce") / pd.to_numeric(base.get("btc_spot_close"), errors="coerce") - 1
    ctx_cols = [c for c in base.columns if c.endswith("_cvd_zscore") and not c.startswith("btc")]
    base["major_context_score"] = base[ctx_cols].mean(axis=1) if ctx_cols else np.nan
    base["cvd_score"] = base["cvd_zscore"].fillna(0).clip(-3, 3) / 3
    base["cvd_slope_score"] = np.sign(base["cvd_slope"].fillna(0))
    base["cvd_divergence_score"] = base["cvd_divergence_vs_price"].fillna(0).abs().clip(0, 2) / 2
    base["cvd_reclaim_score"] = base["cvd_reclaim"].fillna(0)
    base["taker_alignment_score_if_available"] = np.sign(base["btc_taker_delta"].fillna(0)) == np.sign(base["delta_qty_proxy"].fillna(0))
    base["taker_alignment_score_if_available"] = base["taker_alignment_score_if_available"].astype(float)
    base["oi_alignment_score_if_available"] = (np.sign(base["btc_oi_change"].fillna(0)) == np.sign(base["price_return"].fillna(0))).astype(float)
    base["funding_basis_context_score"] = (base["btc_basis"].fillna(0).abs() + pd.to_numeric(base.get("btc_funding_fundingRate"), errors="coerce").fillna(0).abs()).clip(0, 1)
    base["data_quality_score"] = 1 - base[["cvd_zscore", "delta_qty_proxy", "close_proxy_price", "trade_count"]].isna().mean(axis=1)
    base.to_parquet(ROOT / "proxy_cvd_expanded_features.parquet", index=False)
    return base


def _cid(ts: Any, gid: str) -> str:
    return hashlib.sha256(f"{ts}|{gid}".encode()).hexdigest()[:14]


def generate_candidates(f: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("CVDG0_REFERENCE_NO_CVD", "CVD0_BTC_NO_CVD_REFERENCE", f["price_return"] > 0.0005, "LONG", True),
        ("CVDG1_BTC_CVD_CONTINUATION", "CVD1_BTC_PROXY_CVD_ONLY", (f["cvd_zscore"] > 1) & (f["price_return"] > 0), "LONG", False),
        ("CVDG2_BTC_CVD_DIVERGENCE_REVERSAL", "CVD1_BTC_PROXY_CVD_ONLY", f["cvd_divergence_vs_price"].abs() >= 2, "LONG", False),
        ("CVDG3_BTC_CVD_RECLAIM", "CVD2_BTC_PROXY_CVD_PLUS_OHLCV", f["cvd_reclaim"] > 0, "LONG", False),
        ("CVDG4_BTC_LARGE_TRADE_AGGRESSION_CONTINUATION", "CVD2_BTC_PROXY_CVD_PLUS_OHLCV", f["large_trade_aggression_score"] > 1.5, "LONG", False),
        ("CVDG5_BTC_ABSORPTION_REVERSAL_PROXY", "CVD1_BTC_PROXY_CVD_ONLY", f["cvd_absorption_proxy"] > 0, "LONG", False),
        ("CVDG6_BTC_CVD_TAKER_ALIGNMENT", "CVD4_BTC_PROXY_CVD_PLUS_RECENT_OI_TAKER", (f["cvd_zscore"] > 0.8) & (f["taker_alignment_score_if_available"] > 0), "LONG", False),
        ("CVDG7_BTC_CVD_TAKER_DIVERGENCE", "CVD6_BTC_PROXY_CVD_PLUS_TAKER_ONLY", (f["cvd_zscore"].abs() > 1) & (f["taker_alignment_score_if_available"] <= 0), "LONG", False),
        ("CVDG8_BTC_CVD_ORDERFLOW_ENSEMBLE_STRICT", "CVD7_BTC_PROXY_CVD_PLUS_ORDERFLOW_ENSEMBLE", (f["cvd_zscore"] > 1) & (f["taker_alignment_score_if_available"] > 0) & (f["oi_alignment_score_if_available"] > 0), "LONG", False),
        ("CVDG9_BTC_CVD_ORDERFLOW_ENSEMBLE_BALANCED", "CVD7_BTC_PROXY_CVD_PLUS_ORDERFLOW_ENSEMBLE", (f["cvd_zscore"] > 0.5) & (f["taker_alignment_score_if_available"] > 0), "LONG", False),
        ("CVDG10_BTC_CVD_RISK_FILTER_ONLY", "CVD5_BTC_PROXY_CVD_PLUS_MAJOR_CONTEXT", (f["major_context_score"].fillna(0) < -1), "SHORT", True),
    ]
    rows: List[Dict[str, Any]] = []
    for gid, group, mask, direction, ref in specs:
        for _, r in f[mask.fillna(False)].head(2000).iterrows():
            cid = _cid(r["timestamp"], gid)
            rows.append({"proxy_cvd_candidate_id": f"BTCUSDT_{gid}_{pd.Timestamp(r['timestamp']).strftime('%Y%m%d%H%M')}_{cid}", "timestamp": r["timestamp"], "entry_ts": r["timestamp"], "symbol": "BTCUSDT", "direction": direction, "generator_id": gid, "feature_group": group, "cvd_score": r.get("cvd_score"), "cvd_slope_score": r.get("cvd_slope_score"), "cvd_divergence_score": r.get("cvd_divergence_score"), "cvd_reclaim_score": r.get("cvd_reclaim_score"), "large_trade_aggression_score": r.get("large_trade_aggression_score"), "taker_alignment_score_if_available": r.get("taker_alignment_score_if_available"), "oi_alignment_score_if_available": r.get("oi_alignment_score_if_available"), "funding_basis_context_score": r.get("funding_basis_context_score"), "major_context_score": r.get("major_context_score"), "data_quality_score": r.get("data_quality_score"), "expected_move_to_cost_ratio": abs(r.get("price_return", 0)) / COST, "candidate_allowed_core": not ref, "oracle_flag": False, "source": "proxy_cvd_independent_full_timestamp", "research_v3_candidate_id": pd.NA})
    c = pd.DataFrame(rows).drop_duplicates("proxy_cvd_candidate_id")
    c.to_parquet(ROOT / "proxy_cvd_expanded_candidates.parquet", index=False)
    return c


def backfill(c: pd.DataFrame, f: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    px = f[["timestamp", "close_proxy_price", "cvd_zscore", "btc_taker_delta"]].dropna(subset=["close_proxy_price"]).set_index("timestamp")
    policies = {"X1_fixed_12": 12, "X2_fixed_24": 24, "X3_fixed_48": 48, "X4_fixed_96": 96, "X5_first_cost_plus_move": 24, "X6_first_2x_cost_plus_move": 48, "X7_vol_adjusted_stop": 48, "X8_cvd_decay_exit": 48, "X9_cvd_flip_exit": 48, "X10_taker_cvd_flip_exit": 48, "X90_oracle_best_24": 24, "X91_oracle_best_48": 48, "X92_oracle_MFE": 96}
    rows = []
    for _, r in c.iterrows():
        path = px[(px.index > r["entry_ts"]) & (px.index <= r["entry_ts"] + pd.Timedelta(minutes=5 * 96))]
        if len(path) < 6:
            continue
        entry = float(path.iloc[0]["close_proxy_price"])
        long = r["direction"] == "LONG"
        ret = path["close_proxy_price"] / entry - 1 if long else entry / path["close_proxy_price"] - 1
        for pid, bars in policies.items():
            s = ret.head(bars)
            if len(s) < 3:
                continue
            idx = int(s.values.argmax()) if pid.startswith("X9") else len(s) - 1
            if "first_cost" in pid:
                hit = np.where(s.values >= COST * 2)[0]
                idx = int(hit[0]) if len(hit) else idx
            if "first_2x" in pid:
                hit = np.where(s.values >= COST * 3)[0]
                idx = int(hit[0]) if len(hit) else idx
            gross = float(s.max()) if pid.endswith("MFE") else float(s.iloc[idx])
            rows.append({**r.to_dict(), "exit_policy_id": pid, "oracle_exit_policy": pid.startswith("X9"), "net_after_cost": gross - COST, "MFE": float(s.max()), "MAE": float(s.min()), "MFE_to_cost": float(s.max() / COST), "RFE": bool(s.min() < -0.006), "tail_loss": bool(gross - COST < -0.006)})
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "proxy_cvd_expanded_outcomes.parquet", index=False)
    lab = out[(out["exit_policy_id"] == "X2_fixed_24") & (~out["oracle_exit_policy"].astype(bool))].copy()
    if len(lab):
        lab["cvd_label"] = np.select([(lab["net_after_cost"] > 0.001) & (lab["MFE_to_cost"] > 2), lab["net_after_cost"] < -0.001], ["CVD_GOOD", "CVD_BAD"], default="CVD_NEUTRAL")
        lab["proxy_cvd_expected_edge_score"] = lab["net_after_cost"].clip(-0.02, 0.02) / 0.02 + lab["cvd_score"].fillna(0) * 0.2 - lab["RFE"].astype(float) * 0.3
    lab.to_parquet(ROOT / "proxy_cvd_expanded_labels.parquet", index=False)
    dec = lab.assign(edge_decile=pd.qcut(lab["proxy_cvd_expected_edge_score"].rank(method="first"), min(10, len(lab)), labels=False, duplicates="drop")).groupby("edge_decile").agg(rows=("proxy_cvd_candidate_id", "size"), net_mean=("net_after_cost", "mean"), GOOD_rate=("cvd_label", lambda x: float((x == "CVD_GOOD").mean())), BAD_rate=("cvd_label", lambda x: float((x == "CVD_BAD").mean()))).reset_index() if len(lab) else pd.DataFrame()
    dec.to_csv(ROOT / "proxy_cvd_expected_edge_deciles.csv", index=False)
    score = lab.groupby("feature_group").agg(rows=("proxy_cvd_candidate_id", "size"), net_after_cost=("net_after_cost", "mean"), profit_factor=("net_after_cost", lambda x: float(x[x > 0].sum() / -x[x < 0].sum()) if (x < 0).any() else float("inf")), GOOD_rate=("cvd_label", lambda x: float((x == "CVD_GOOD").mean())), BAD_rate=("cvd_label", lambda x: float((x == "CVD_BAD").mean())), tail_loss=("tail_loss", "mean"), RFE_rate=("RFE", "mean")).reset_index() if len(lab) else pd.DataFrame()
    score.to_csv(ROOT / "proxy_cvd_value_add_scorecard.csv", index=False)
    return out, lab, dec


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "proxy_cvd": True, "production_ready": False}
    ROOT.mkdir(parents=True, exist_ok=True)
    f = build_features()
    c = generate_candidates(f)
    out, lab, dec = backfill(c, f)
    sample_days = float(f["sample_days"].max()) if len(f) else 0
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    if sample_days < 30:
        verdict = "PROXY_CVD_SAMPLE_TOO_SMALL"
    elif len(lab) and lab.groupby("feature_group")["net_after_cost"].mean().max() > 0:
        verdict = "PROXY_CVD_90D_VALUE_ADD_FOUND_RESEARCH_ONLY" if sample_days >= 90 else "PROXY_CVD_30D_VALUE_ADD_FOUND_RESEARCH_ONLY"
    else:
        verdict = "PROXY_CVD_NO_VALUE_ADD"
    if len(lab) and lab["net_after_cost"].mean() < 0:
        verdict += " + PROXY_CVD_COST_KILLS_EDGE"
    if mono:
        verdict += " + PROXY_CVD_EXPECTED_EDGE_MONOTONIC_RESEARCH_ONLY"
    else:
        verdict += " + PROXY_CVD_EXPECTED_EDGE_NOT_MONOTONIC"
    verdict += " + production_not_ready"
    (ROOT / "proxy_cvd_expanded_research_report.md").write_text("# Proxy CVD Expanded Research Report\n\nProxy CVD is not true CVD. BTCUSDT only target; context symbols are reference/as-of sensors.\n\n```json\n" + _json({"sample_days": sample_days, "features": len(f), "candidates": len(c), "labels": lab["cvd_label"].value_counts().to_dict() if len(lab) else {}, "expected_edge_monotonic": mono, "verdict": verdict}) + "\n```\n", encoding="utf-8")
    return {"sample_days": sample_days, "feature_rows": len(f), "candidate_rows": len(c), "outcome_rows": len(out), "label_rows": len(lab), "expected_edge_monotonic": mono, "verdict": verdict, "production_ready": False, "promotion_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args.dry_run)
    print(_json(res) if args.json else res.get("verdict", "dry_run"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
