"""
PHASE 2 - Hold survival analysis (analysis-only).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from scripts.diagnostics.lifecycle_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    align_to_state_window,
    compute_net_return,
    ensure_out_dir,
    load_tick_events,
    reconstruct_actual_trades,
)


def _future_return(direction: str, entry_price: float, future_price: float) -> float:
    if direction == "LONG":
        return (future_price - entry_price) / entry_price
    return (entry_price - future_price) / entry_price


def main() -> None:
    out_dir = ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lifecycle_hold_survival_{ts}.csv"
    md_path = out_dir / f"lifecycle_hold_survival_{ts}.md"

    events = load_tick_events()
    trades = align_to_state_window(reconstruct_actual_trades(events))
    prices = {e.idx: e.price for e in events}

    horizon_rows: List[Dict[str, Any]] = []
    for h in range(1, 13):
        vals = []
        risk_vals = []
        survive_count = 0
        forced_count = 0
        force_and_recover = 0
        for _, tr in trades.iterrows():
            fut_idx = int(tr["entry_idx"]) + h
            if fut_idx not in prices:
                continue
            raw = _future_return(str(tr["direction"]), float(tr["entry_price"]), float(prices[fut_idx]))
            net = compute_net_return(raw, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE)
            vals.append(net)
            if int(tr["hold_bars"]) >= h:
                survive_count += 1
            if str(tr["exit_reason"]) == "risk_force_exit" and int(tr["hold_bars"]) <= h:
                forced_count += 1
                if net > 0:
                    force_and_recover += 1
                risk_vals.append(net)
        horizon_rows.append(
            {
                "horizon_bar": h,
                "sample_n": len(vals),
                "cumulative_expectancy": (sum(vals) / len(vals)) if vals else 0.0,
                "survival_wr": (sum(1 for x in vals if x > 0) / len(vals)) if vals else 0.0,
                "avg_future_return": (sum(vals) / len(vals)) if vals else 0.0,
                "forced_exit_ratio": (forced_count / max(len(trades), 1)),
                "profitable_continuation_ratio": (force_and_recover / forced_count) if forced_count else 0.0,
                "risk_cluster_future_mean": (sum(risk_vals) / len(risk_vals)) if risk_vals else 0.0,
                "survive_count": survive_count,
            }
        )

    hz = pd.DataFrame(horizon_rows)
    bucket_defs = {"1~3": (1, 3), "4~6": (4, 6), "7~9": (7, 9), "10~12": (10, 12)}
    bucket_rows = []
    for name, (a, b) in bucket_defs.items():
        seg = hz[(hz["horizon_bar"] >= a) & (hz["horizon_bar"] <= b)]
        bucket_rows.append(
            {
                "bucket": name,
                "avg_cumulative_expectancy": float(seg["cumulative_expectancy"].mean()),
                "avg_survival_wr": float(seg["survival_wr"].mean()),
                "avg_future_return": float(seg["avg_future_return"].mean()),
                "avg_forced_exit_ratio": float(seg["forced_exit_ratio"].mean()),
                "avg_profitable_continuation_ratio": float(seg["profitable_continuation_ratio"].mean()),
            }
        )
    bucket_df = pd.DataFrame(bucket_rows)

    out = hz.copy()
    out["section"] = "horizon"
    b2 = bucket_df.rename(columns={"bucket": "horizon_bar"}).copy()
    b2["section"] = "bucket"
    out = pd.concat([out, b2], ignore_index=True, sort=False)
    out.to_csv(csv_path, index=False)

    best_h = hz.sort_values("cumulative_expectancy", ascending=False).iloc[0]
    md = []
    md.append("# Lifecycle Hold Survival")
    md.append("")
    md.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- trades: {len(trades)}")
    md.append("")
    md.append("## Horizon(1~12) Summary")
    for _, r in hz.iterrows():
        md.append(
            f"- h={int(r['horizon_bar'])}: exp={r['cumulative_expectancy']:.6f}, WR={r['survival_wr']*100:.2f}%, "
            f"forced_ratio={r['forced_exit_ratio']*100:.2f}%, profitable_cont={r['profitable_continuation_ratio']*100:.2f}%"
        )
    md.append("")
    md.append("## Bucket Summary")
    for _, r in bucket_df.iterrows():
        md.append(
            f"- {r['bucket']}: exp={r['avg_cumulative_expectancy']:.6f}, WR={r['avg_survival_wr']*100:.2f}%, "
            f"forced_ratio={r['avg_forced_exit_ratio']*100:.2f}%, profitable_cont={r['avg_profitable_continuation_ratio']*100:.2f}%"
        )
    md.append("")
    md.append("## Key Finding")
    md.append(
        f"- best horizon by expectancy: h={int(best_h['horizon_bar'])}, exp={best_h['cumulative_expectancy']:.6f}"
    )

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
