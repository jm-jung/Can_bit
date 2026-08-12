"""
PHASE 4 - LONG/SHORT decomposition.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from scripts.diagnostics.lifecycle_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_POSITION_SIZE,
    DEFAULT_SLIPPAGE_RATE,
    align_to_state_window,
    build_equity_curve,
    compute_net_return,
    ensure_out_dir,
    expectancy_stats,
    load_tick_events,
    max_drawdown_from_curve,
    reconstruct_actual_trades,
    streak_stats,
)


def _summarize_mode(name: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        return {
            "mode": name,
            "trades": 0,
            "wr": 0.0,
            "expectancy": 0.0,
            "pf": 0.0,
            "avg_return": 0.0,
            "avg_hold": 0.0,
            "equity_return": 0.0,
            "mdd_proxy": 0.0,
            "risk_force_ratio": 0.0,
            "exit_signal_ratio": 0.0,
            "ks_like_triggers": 0,
            "max_consecutive_losses": 0,
        }
    net = df["net_return"].astype(float).tolist()
    st = expectancy_stats(net)
    eq = build_equity_curve(net, position_size=DEFAULT_POSITION_SIZE, initial_equity=1.0)
    sk = streak_stats(net)
    return {
        "mode": name,
        "trades": len(df),
        "wr": st["wr"],
        "expectancy": st["expectancy"],
        "pf": st["pf"],
        "avg_return": st["avg_return"],
        "avg_hold": float(df["hold_bars"].mean()),
        "equity_return": (eq[-1] - 1.0) if eq else 0.0,
        "mdd_proxy": max_drawdown_from_curve(eq),
        "risk_force_ratio": float((df["exit_reason"] == "risk_force_exit").mean()),
        "exit_signal_ratio": float((df["exit_reason"] == "exit_signal").mean()),
        "ks_like_triggers": sk["ks_like_triggers"],
        "max_consecutive_losses": sk["max_consecutive_losses"],
    }


def main() -> None:
    out_dir = ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lifecycle_directional_edge_{ts}.csv"
    md_path = out_dir / f"lifecycle_directional_edge_{ts}.md"

    events = load_tick_events()
    trades = align_to_state_window(reconstruct_actual_trades(events))
    trades["net_return"] = trades["raw_return"].apply(
        lambda x: compute_net_return(x, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE)
    )

    long_df = trades[trades["direction"] == "LONG"].copy()
    short_df = trades[trades["direction"] == "SHORT"].copy()
    both_df = trades.copy()

    rows = [
        _summarize_mode("LONG_only", long_df),
        _summarize_mode("SHORT_only", short_df),
        _summarize_mode("LONG+SHORT", both_df),
    ]
    out = pd.DataFrame(rows)
    out.to_csv(csv_path, index=False)

    md = []
    md.append("# Lifecycle Directional Edge")
    md.append("")
    md.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- trades_total: {len(trades)} (LONG={len(long_df)}, SHORT={len(short_df)})")
    md.append("")
    for _, r in out.iterrows():
        md.append(
            f"- {r['mode']}: trades={int(r['trades'])}, WR={r['wr']*100:.2f}%, exp={r['expectancy']:.6f}, "
            f"PF={r['pf']:.4f}, avg_hold={r['avg_hold']:.2f}, risk_force_ratio={r['risk_force_ratio']*100:.2f}%, "
            f"ks_like={int(r['ks_like_triggers'])}"
        )
    md.append("")
    md.append("## Interpretation")
    best_mode = out.sort_values("expectancy", ascending=False).iloc[0]
    md.append(f"- best_expectancy_mode: {best_mode['mode']} ({best_mode['expectancy']:.6f})")
    md.append(
        "- question target: SHORT가 전체 전략 저하에 기여하는지 LONG_only vs LONG+SHORT 기대값 차이로 확인"
    )

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
