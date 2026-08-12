"""
PHASE 3 - Fee/slippage sensitivity analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from scripts.diagnostics.lifecycle_common import (
    DEFAULT_POSITION_SIZE,
    align_to_state_window,
    build_equity_curve,
    ensure_out_dir,
    expectancy_stats,
    max_drawdown_from_curve,
    reconstruct_actual_trades,
    load_tick_events,
)


def main() -> None:
    out_dir = ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lifecycle_fee_sensitivity_{ts}.csv"
    md_path = out_dir / f"lifecycle_fee_sensitivity_{ts}.md"

    events = load_tick_events()
    trades = align_to_state_window(reconstruct_actual_trades(events))
    raw = trades["raw_return"].astype(float).tolist()

    fee_grid = [0.0001, 0.0002, 0.0004, 0.0006, 0.0010]
    slip_grid = [0.0, 0.0001, 0.0002, 0.0005]
    rows: List[Dict[str, Any]] = []

    gross_sum = float(sum(raw))
    gross_stats = expectancy_stats(raw)
    for f in fee_grid:
        for s in slip_grid:
            net = [r - 2.0 * (f + s) for r in raw]
            st = expectancy_stats(net)
            eq = build_equity_curve(net, position_size=DEFAULT_POSITION_SIZE, initial_equity=1.0)
            rows.append(
                {
                    "fee_rate": f,
                    "slippage_rate": s,
                    "gross_return_sum": gross_sum,
                    "net_return_sum": float(sum(net)),
                    "gross_expectancy": gross_stats["expectancy"],
                    "net_expectancy": st["expectancy"],
                    "gross_pf": gross_stats["pf"],
                    "net_pf": st["pf"],
                    "net_wr": st["wr"],
                    "net_avg_return": st["avg_return"],
                    "equity_return": (eq[-1] - 1.0) if eq else 0.0,
                    "mdd_proxy": max_drawdown_from_curve(eq),
                    "round_trip_cost": 2.0 * (f + s),
                }
            )

    out = pd.DataFrame(rows).sort_values(["fee_rate", "slippage_rate"]).reset_index(drop=True)
    out.to_csv(csv_path, index=False)

    best = out.sort_values("net_expectancy", ascending=False).iloc[0]
    worst = out.sort_values("net_expectancy", ascending=True).iloc[0]
    zero_cross = out[out["net_expectancy"] >= 0].sort_values("round_trip_cost")

    md = []
    md.append("# Lifecycle Fee Sensitivity")
    md.append("")
    md.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- trades: {len(raw)}")
    md.append(f"- gross_expectancy: {gross_stats['expectancy']:.6f}")
    md.append("")
    md.append("## Best/Worst Grid Point")
    md.append(
        f"- best: fee={best['fee_rate']:.4f}, slip={best['slippage_rate']:.4f}, "
        f"net_exp={best['net_expectancy']:.6f}, eq_ret={best['equity_return']:.6f}"
    )
    md.append(
        f"- worst: fee={worst['fee_rate']:.4f}, slip={worst['slippage_rate']:.4f}, "
        f"net_exp={worst['net_expectancy']:.6f}, eq_ret={worst['equity_return']:.6f}"
    )
    if len(zero_cross):
        z = zero_cross.iloc[0]
        md.append(
            f"- breakeven-like minimum cost point: fee={z['fee_rate']:.4f}, slip={z['slippage_rate']:.4f}, "
            f"round_trip_cost={z['round_trip_cost']:.6f}"
        )
    else:
        md.append("- no non-negative net_expectancy in provided grid")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
