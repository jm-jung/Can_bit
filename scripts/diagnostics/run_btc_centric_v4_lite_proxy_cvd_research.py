"""Preliminary BTC-centric proxy CVD research; diagnostics only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd/proxy_cvd_research")
CVD = Path("data/diagnostics/research_orderflow_data_cache/features/proxy_cvd")


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def run(dry_run: bool = False) -> dict:
    if dry_run:
        return {"dry_run": True, "proxy_cvd": True, "true_cvd": False, "production_ready": False}
    ROOT.mkdir(parents=True, exist_ok=True)
    p = CVD / "BTCUSDT_5m.parquet"
    if not p.exists():
        report = {"status": "PROXY_CVD_BUILD_REQUIRED", "reason": "BTCUSDT_5m proxy CVD parquet not found", "production_ready": False}
        pd.DataFrame().to_parquet(ROOT / "proxy_cvd_candidates.parquet", index=False)
        pd.DataFrame().to_parquet(ROOT / "proxy_cvd_outcomes.parquet", index=False)
        pd.DataFrame([report]).to_csv(ROOT / "proxy_cvd_scorecard.csv", index=False)
        (ROOT / "proxy_cvd_research_report.md").write_text("# Proxy CVD Research Report\n\nProxy CVD data is not available yet. This remains design/skeleton only.\n", encoding="utf-8")
        print(_json(report))
        return report
    cvd = pd.read_parquet(p).sort_values("timestamp")
    span_days = (pd.to_datetime(cvd["timestamp"]).max() - pd.to_datetime(cvd["timestamp"]).min()).total_seconds() / 86400 if len(cvd) else 0
    specs = [
        ("CVDG1_BTC_CVD_CONTINUATION", "CVD1_BTC_PROXY_CVD_ONLY", (cvd["cvd_zscore"] > 1) & (cvd["price_return"] > 0), "LONG"),
        ("CVDG2_BTC_CVD_DIVERGENCE_REVERSAL", "CVD1_BTC_PROXY_CVD_ONLY", (cvd["cvd_divergence_vs_price"].abs() >= 2), "LONG"),
        ("CVDG3_BTC_CVD_RECLAIM", "CVD2_BTC_PROXY_CVD_PLUS_OHLCV", cvd["cvd_reclaim"] > 0, "LONG"),
        ("CVDG4_BTC_LARGE_TRADE_AGGRESSION_CONTINUATION", "CVD2_BTC_PROXY_CVD_PLUS_OHLCV", cvd["large_trade_aggression_score"] > 1.5, "LONG"),
        ("CVDG5_BTC_ABSORPTION_REVERSAL_PROXY", "CVD1_BTC_PROXY_CVD_ONLY", cvd["cvd_absorption_proxy"] > 0, "LONG"),
        ("CVDG6_BTC_CVD_ORDERFLOW_ENSEMBLE", "CVD5_BTC_PROXY_CVD_PLUS_MAJOR_CONTEXT", (cvd["cvd_zscore"] > 1) & (cvd["large_trade_aggression_score"] > 0), "LONG"),
    ]
    rows = []
    for gid, group, mask, direction in specs:
        take = cvd[mask.fillna(False)].head(500)
        for _, r in take.iterrows():
            rows.append({"proxy_cvd_candidate_id": f"BTCUSDT_{gid}_{pd.Timestamp(r['timestamp']).strftime('%Y%m%d%H%M')}", "timestamp": r["timestamp"], "symbol": "BTCUSDT", "direction": direction, "generator_id": gid, "feature_group": group, "cvd_zscore": r.get("cvd_zscore"), "cvd_slope": r.get("cvd_slope"), "large_trade_aggression_score": r.get("large_trade_aggression_score"), "source": "proxy_cvd_independent_timestamp", "oracle_flag": False, "production_ready": False})
    cand = pd.DataFrame(rows)
    cand.to_parquet(ROOT / "proxy_cvd_candidates.parquet", index=False)
    outcomes = cand.copy()
    if len(outcomes):
        fut = cvd[["timestamp", "close_proxy_price"]].rename(columns={"close_proxy_price": "close"}).set_index("timestamp")
        nets = []
        for _, r in outcomes.iterrows():
            path = fut[(fut.index > r["timestamp"]) & (fut.index <= r["timestamp"] + pd.Timedelta(hours=4))]
            if len(path) < 3:
                nets.append(np.nan)
            else:
                entry = path.iloc[0]["close"]
                nets.append(float(path.iloc[-1]["close"] / entry - 1 - 0.0006))
        outcomes["net_after_cost"] = nets
        outcomes["proxy_cvd_label"] = np.select([outcomes["net_after_cost"] > 0.001, outcomes["net_after_cost"] < -0.001], ["CVD_GOOD", "CVD_BAD"], default="CVD_NEUTRAL")
    outcomes.to_parquet(ROOT / "proxy_cvd_outcomes.parquet", index=False)
    score = outcomes.groupby("feature_group").agg(rows=("proxy_cvd_candidate_id", "size"), net_after_cost=("net_after_cost", "mean"), GOOD_rate=("proxy_cvd_label", lambda x: float((x == "CVD_GOOD").mean())), BAD_rate=("proxy_cvd_label", lambda x: float((x == "CVD_BAD").mean()))).reset_index() if len(outcomes) else pd.DataFrame()
    score["sample_days"] = span_days if len(score) else []
    score.to_csv(ROOT / "proxy_cvd_scorecard.csv", index=False)
    status = "PROXY_CVD_VALUE_ADD_FOUND_RESEARCH_ONLY" if len(score) and score["net_after_cost"].max() > 0 and span_days >= 7 else "PROXY_CVD_NEXT_REQUIRED"
    (ROOT / "proxy_cvd_research_report.md").write_text("# Proxy CVD Research Report\n\nPreliminary only. Proxy CVD is based on aggTrades `is_buyer_maker`; it is not true CVD. production_ready=false.\n\n```json\n" + _json({"status": status, "sample_days": span_days, "candidates": len(cand)}) + "\n```\n", encoding="utf-8")
    return {"status": status, "sample_days": span_days, "candidate_rows": len(cand), "outcome_rows": len(outcomes), "production_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args.dry_run)
    print(_json(res) if args.json else res.get("status", "dry_run"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
