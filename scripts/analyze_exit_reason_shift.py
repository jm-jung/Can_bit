#!/usr/bin/env python3
"""
C9 Sanity check: baseline vs BE variant exit_reason 분포 비교.
입력: trade log CSV (--emit-trade-log on 으로 생성). 365d 로그 사용.
출력: exit_reason별 count/avg_pnl, break_even holding_bars 분포, break_even 샘플 10개.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADE_LOG_DIR = PROJECT_ROOT / "data" / "diagnostics" / "trade_logs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run-id", type=str, default="phase_c9_base")
    ap.add_argument("--other-run-id", type=str, default="phase_c9_be004")
    ap.add_argument("--trade-log-dir", type=str, default=None)
    ap.add_argument("--days", type=int, default=365)
    args = ap.parse_args()

    log_dir = Path(args.trade_log_dir or TRADE_LOG_DIR)
    suffix = f"{args.days}d"
    base_path = log_dir / f"{args.baseline_run_id}_{suffix}.csv"
    other_path = log_dir / f"{args.other_run_id}_{suffix}.csv"

    if not base_path.exists():
        print(f"Baseline trade log not found: {base_path}", file=sys.stderr)
        print("Run validation with --emit-trade-log on for baseline and other.", file=sys.stderr)
        return 1
    if not other_path.exists():
        print(f"Other trade log not found: {other_path}", file=sys.stderr)
        return 1

    import pandas as pd
    df_base = pd.read_csv(base_path)
    df_other = pd.read_csv(other_path)

    def summary(df: pd.DataFrame, label: str) -> None:
        print(f"\n=== {label} (n={len(df)}) ===")
        if df.empty:
            print("  (no exits)")
            return
        if "exit_reason" not in df.columns:
            print("  (no exit_reason column)")
            return
        reason_counts = df["exit_reason"].value_counts()
        print("exit_reason count:")
        for r, c in reason_counts.items():
            print(f"  {r}: {c}")
        print("exit_reason avg net_return:")
        for r in reason_counts.index:
            sub = df[df["exit_reason"] == r]
            net = sub.get("net_return")
            if net is not None and len(sub) > 0:
                avg = net.mean()
                print(f"  {r}: {avg:.6f} (n={len(sub)})")
        # break_even holding_bars 분포
        be = df[df["exit_reason"] == "break_even"]
        if not be.empty and "holding_bars" in be.columns:
            print("break_even holding_bars: min={:.0f} max={:.0f} mean={:.1f}".format(
                be["holding_bars"].min(), be["holding_bars"].max(), be["holding_bars"].mean()))
        # break_even 샘플 10개
        if not be.empty:
            print("break_even 샘플 (최대 10개): entry_ts, exit_ts, entry_price, exit_price, holding_bars, net_return, be_armed, be_arm_ts")
            sample = be.head(10)
            for _, row in sample.iterrows():
                print(f"  {row.get('entry_ts')} | {row.get('exit_ts')} | entry={row.get('entry_price')} exit={row.get('exit_price')} | bars={row.get('holding_bars')} net={row.get('net_return')} | be_armed={row.get('be_armed')} be_arm_ts={row.get('be_arm_ts')}")

    print("=" * 60)
    print("Exit reason shift: baseline vs BE variant (365d trade log)")
    print("=" * 60)
    summary(df_base, args.baseline_run_id)
    summary(df_other, args.other_run_id)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
