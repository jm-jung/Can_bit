"""
PHASE 5 - Kill switch forensics.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from scripts.diagnostics.lifecycle_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    KS_STREAK_THRESHOLD,
    align_to_state_window,
    compute_net_return,
    ensure_out_dir,
    load_state,
    load_tick_events,
    reconstruct_actual_trades,
)


def main() -> None:
    out_dir = ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lifecycle_kill_switch_forensics_{ts}.csv"
    md_path = out_dir / f"lifecycle_kill_switch_forensics_{ts}.md"
    timeline_csv = out_dir / f"lifecycle_kill_switch_timeline_{ts}.csv"

    state = load_state()
    events = load_tick_events()
    trades = align_to_state_window(reconstruct_actual_trades(events))
    trades["net_return"] = trades["raw_return"].apply(
        lambda x: compute_net_return(x, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE)
    )
    trades["is_loss"] = trades["net_return"] < 0

    # entry idx 기반 근사 volatility (12 bars std of tick returns)
    px = pd.Series([e.price for e in events], dtype=float)
    tick_ret = px.pct_change()
    roll_vol = tick_ret.rolling(12, min_periods=5).std()
    trades["entry_vol_12"] = trades["entry_idx"].apply(
        lambda i: float(roll_vol.iloc[int(i)]) if 0 <= int(i) < len(roll_vol) else float("nan")
    )

    # consecutive losses trace
    consec = 0
    consec_trace: List[int] = []
    for r in trades["net_return"].tolist():
        if r < 0:
            consec += 1
        else:
            consec = 0
        consec_trace.append(consec)
    trades["consecutive_losses_after_trade"] = consec_trace

    trigger_idx = None
    for i, c in enumerate(consec_trace):
        if c >= KS_STREAK_THRESHOLD:
            trigger_idx = i
            break

    last20 = trades.tail(20).copy()
    last20.to_csv(timeline_csv, index=False)

    seq_df = pd.DataFrame()
    if trigger_idx is not None:
        seq_df = trades.iloc[max(0, trigger_idx - 10) : trigger_idx + 1].copy()
    else:
        seq_df = last20.copy()
    seq_df.to_csv(csv_path, index=False)

    # rebound test: 과민반응 여부 보조지표
    rebs = []
    nr = trades["net_return"].tolist()
    for i in range(len(nr) - KS_STREAK_THRESHOLD - 3):
        if all(x < 0 for x in nr[i : i + KS_STREAK_THRESHOLD]):
            nxt = nr[i + KS_STREAK_THRESHOLD : i + KS_STREAK_THRESHOLD + 3]
            rebs.append(sum(nxt) / len(nxt))
    rebound_mean = float(sum(rebs) / len(rebs)) if rebs else 0.0

    md = []
    md.append("# Lifecycle Kill Switch Forensics")
    md.append("")
    md.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- state_kill_switch_active: {state.get('kill_switch_active')}")
    md.append(f"- state_kill_switch_reason: {state.get('kill_switch_reason')}")
    md.append(f"- trigger_trade_index_estimated: {trigger_idx}")
    md.append("")
    md.append("## Trigger Sequence (estimated)")
    if len(seq_df):
        for _, r in seq_df.iterrows():
            md.append(
                f"- idx={int(r.name)} time={r['exit_time']} dir={r['direction']} net={r['net_return']:.6f} "
                f"hold={int(r['hold_bars'])} exit={r['exit_reason']} entropy={r['entropy'] if pd.notna(r['entropy']) else 'NA'} "
                f"activation={r['activation_state']} vol12={r['entry_vol_12']:.6f}"
            )
    else:
        md.append("- no sequence available")
    md.append("")
    md.append("## Rebound Probe")
    md.append(
        f"- mean return of next 3 trades after historical {KS_STREAK_THRESHOLD}-loss clusters: {rebound_mean:.6f}"
    )
    if rebound_mean > 0:
        md.append("- interpretation: KS가 일부 상황에서 과도 조기 차단일 가능성")
    else:
        md.append("- interpretation: KS가 손실 구간 차단에 기여했을 가능성")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"TIMELINE_CSV: {timeline_csv}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
