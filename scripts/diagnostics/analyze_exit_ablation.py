"""
PHASE 1 - Exit ablation analysis (analysis-only).

Output:
  data/diagnostics/lifecycle/lifecycle_exit_ablation_<timestamp>.csv
  data/diagnostics/lifecycle/lifecycle_exit_ablation_<timestamp>.md
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from scripts.diagnostics.lifecycle_common import (
    DEFAULT_FEE_RATE,
    DEFAULT_SLIPPAGE_RATE,
    align_to_state_window,
    compute_net_return,
    ensure_out_dir,
    load_tick_events,
    reconstruct_actual_trades,
    summarize_case,
)


def _direction_from_reason(reason: str) -> str:
    return "SHORT" if str(reason).startswith("SHORT") else "LONG"


def _raw_return(direction: str, entry_price: float, exit_price: float) -> float:
    if direction == "LONG":
        return (exit_price - entry_price) / entry_price
    return (entry_price - exit_price) / entry_price


def _simulate_case(
    events,
    max_hold_bars: int,
    ignore_risk_exit: bool = False,
    delay_risk_to_hold: Optional[int] = None,
    trailing_delay_bars: Optional[int] = None,
    trailing_giveback: float = 0.006,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    pos: Optional[Dict[str, Any]] = None

    for ev in events:
        if pos is not None:
            pos["hold_bars"] += 1
            cur_raw = _raw_return(pos["direction"], pos["entry_price"], ev.price)
            pos["best_raw"] = max(pos["best_raw"], cur_raw)
            pos["last_raw"] = cur_raw

            if pos.get("risk_armed"):
                pos["risk_armed_bars"] += 1

        should_exit = False
        exit_reason = ""

        if pos is None and ev.decision == "enter":
            pos = {
                "entry_idx": ev.idx,
                "entry_time": ev.ts,
                "entry_price": ev.price,
                "direction": _direction_from_reason(ev.reason),
                "hold_bars": 0,
                "best_raw": 0.0,
                "last_raw": 0.0,
                "risk_armed": False,
                "risk_armed_bars": 0,
            }
            continue

        if pos is None:
            continue

        if ev.decision == "exit":
            if ev.reason == "exit_signal":
                should_exit = True
                exit_reason = "exit_signal"
            elif ev.reason == "risk_force_exit":
                if ignore_risk_exit:
                    pass
                elif delay_risk_to_hold is not None:
                    if pos["hold_bars"] >= int(delay_risk_to_hold):
                        should_exit = True
                        exit_reason = "risk_force_exit_delayed_hold"
                elif trailing_delay_bars is not None:
                    pos["risk_armed"] = True
                else:
                    should_exit = True
                    exit_reason = "risk_force_exit"

        if trailing_delay_bars is not None and pos.get("risk_armed"):
            giveback = pos["best_raw"] - pos["last_raw"]
            if giveback >= trailing_giveback:
                should_exit = True
                exit_reason = "risk_force_trailing_giveback"
            elif pos["risk_armed_bars"] >= trailing_delay_bars:
                should_exit = True
                exit_reason = "risk_force_trailing_timeout"

        if not should_exit and pos["hold_bars"] >= max_hold_bars:
            should_exit = True
            exit_reason = "max_hold_exit"

        if should_exit:
            raw = _raw_return(pos["direction"], pos["entry_price"], ev.price)
            rows.append(
                {
                    "entry_idx": pos["entry_idx"],
                    "exit_idx": ev.idx,
                    "entry_time": pos["entry_time"],
                    "exit_time": ev.ts,
                    "direction": pos["direction"],
                    "hold_bars": pos["hold_bars"],
                    "exit_reason": exit_reason,
                    "raw_return": raw,
                    "net_return": compute_net_return(raw, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE),
                }
            )
            pos = None

    return pd.DataFrame(rows)


def main() -> None:
    out_dir = ensure_out_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"lifecycle_exit_ablation_{ts}.csv"
    md_path = out_dir / f"lifecycle_exit_ablation_{ts}.md"

    events = load_tick_events()
    actual = align_to_state_window(reconstruct_actual_trades(events))
    actual["net_return"] = actual["raw_return"].apply(
        lambda x: compute_net_return(x, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE)
    )
    # 최신 state와 정합된 구간 내에서만 ablation을 비교한다.
    start_idx = int(actual["entry_idx"].min()) if not actual.empty else 0
    end_idx = int(actual["exit_idx"].max()) if not actual.empty else (events[-1].idx if events else 0)
    sim_events = [e for e in events if start_idx <= e.idx <= end_idx]

    cases: List[Dict[str, Any]] = []
    case_tables: Dict[str, pd.DataFrame] = {}

    case_tables["A_current"] = actual.copy()
    case_tables["B_no_risk_force_exit"] = _simulate_case(sim_events, max_hold_bars=12, ignore_risk_exit=True)
    case_tables["C_risk_delay_to_hold12"] = _simulate_case(sim_events, max_hold_bars=12, delay_risk_to_hold=12)
    case_tables["D_risk_trailing_delayed"] = _simulate_case(
        sim_events,
        max_hold_bars=12,
        trailing_delay_bars=3,
        trailing_giveback=0.006,
    )
    for hold in (12, 15, 18, 24):
        case_tables[f"E_max_hold_{hold}"] = _simulate_case(sim_events, max_hold_bars=hold, ignore_risk_exit=False)

    for name, df in case_tables.items():
        returns = df["net_return"].astype(float).tolist() if not df.empty else []
        row = summarize_case(name, returns)
        row["avg_hold_bars"] = float(df["hold_bars"].mean()) if not df.empty else 0.0
        row["risk_force_ratio"] = float((df["exit_reason"].astype(str).str.contains("risk_force")).mean()) if not df.empty else 0.0
        row["exit_signal_ratio"] = float((df["exit_reason"] == "exit_signal").mean()) if not df.empty else 0.0
        cases.append(row)

    out_df = pd.DataFrame(cases).sort_values("case").reset_index(drop=True)
    out_df.to_csv(csv_path, index=False)

    best_exp = out_df.sort_values("expectancy", ascending=False).iloc[0]
    best_eq = out_df.sort_values("equity_return", ascending=False).iloc[0]
    base = out_df[out_df["case"] == "A_current"].iloc[0]

    md = []
    md.append("# Lifecycle Exit Ablation")
    md.append("")
    md.append(f"- generated_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    md.append(f"- events_total: {len(events)}")
    md.append(f"- events_used_for_ablation: {len(sim_events)} (idx {start_idx}~{end_idx})")
    md.append("")
    md.append("## Case Summary")
    for _, r in out_df.iterrows():
        md.append(
            f"- {r['case']}: trades={int(r['trades'])}, WR={r['wr']*100:.2f}%, exp={r['expectancy']:.6f}, "
            f"PF={r['pf']:.4f}, eq_ret={r['equity_return']:.6f}, mdd={r['mdd_proxy']:.6f}, "
            f"max_consec_loss={int(r['max_consecutive_losses'])}"
        )
    md.append("")
    md.append("## Key Findings")
    md.append(
        f"- baseline(A_current) expectancy={base['expectancy']:.6f}, equity_return={base['equity_return']:.6f}"
    )
    md.append(
        f"- best_expectancy={best_exp['case']} ({best_exp['expectancy']:.6f}), "
        f"delta_vs_base={best_exp['expectancy']-base['expectancy']:.6f}"
    )
    md.append(
        f"- best_equity_return={best_eq['case']} ({best_eq['equity_return']:.6f}), "
        f"delta_vs_base={best_eq['equity_return']-base['equity_return']:.6f}"
    )

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"CSV: {csv_path}")
    print(f"MD:  {md_path}")


if __name__ == "__main__":
    main()
