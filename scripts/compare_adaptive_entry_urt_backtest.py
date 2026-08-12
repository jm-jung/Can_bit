#!/usr/bin/env python3
"""
Formal A/B: baseline vs adaptive ENTRY gate (same URT snapshot for both arms).

Uses load_latest_entry_urt_snapshot (B_with_meta only) once at start; passes
entry_urt_state_log_snapshot_override so baseline/adaptive share the same gate URT.
load_latest_window_stats is unchanged and printed only for reference.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loss_aux_signal_exit_common import (  # noqa: E402
    load_fr2_override_ensemble,
    run_loss_aux_window,
    to_utc_ts,
)
from src.strategies.signal_exit_th_activation import (  # noqa: E402
    load_latest_entry_urt_snapshot,
    load_latest_window_stats,
)


def _sharpe_trades(res: dict) -> float | None:
    trades = list(res.get("trades") or [])
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if len(vals) < 2:
        return None
    a = np.asarray(vals, dtype=float)
    sd = float(a.std(ddof=1))
    if sd <= 1e-12:
        return None
    return float((a.mean() / sd) * math.sqrt(len(a)))


def _row(label: str, res: dict) -> dict[str, float | int | bool | str | None]:
    tr = int(res.get("total_trades") or 0)
    tp = float(res.get("total_return") or 0.0)
    return {
        "label": label,
        "total_trades": tr,
        "win_rate": float(res.get("win_rate") or 0.0) if tr else 0.0,
        "total_profit": tp,
        "mean_profit": float(res.get("avg_profit") or 0.0) if tr else 0.0,
        "max_drawdown": float(res.get("max_drawdown") or 0.0),
        "sharpe": _sharpe_trades(res),
        "blocked_urt": res.get("entries_blocked_by_urt_gate"),
        "entry_urt_gate_active": res.get("entry_urt_gate_active"),
        "entry_urt_from_state_log": res.get("entry_urt_from_state_log"),
        "entry_urt_gate_value_source": res.get("entry_urt_gate_value_source"),
        "entry_urt_gate_block_samples": res.get("entry_urt_gate_block_samples"),
        "entry_snapshot_selected_timestamp": res.get("entry_snapshot_selected_timestamp"),
        "entry_snapshot_selected_strategy_name": res.get("entry_snapshot_selected_strategy_name"),
        "entry_snapshot_selected_trades_60d_raw": res.get("entry_snapshot_selected_trades_60d_raw"),
        "entry_snapshot_selected_unique_round_trips_raw": res.get(
            "entry_snapshot_selected_unique_round_trips_raw"
        ),
    }


def _print_csv_urt_histogram(state_log_path: Path) -> None:
    if not state_log_path.exists():
        return
    df = pd.read_csv(state_log_path, low_memory=False)
    for col in ("unique_round_trips", "trades_60d"):
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        print(
            f"  csv column {col}: n={len(s)} min={float(s.min()):.0f} max={float(s.max()):.0f} "
            f"p50={float(s.quantile(0.5)):.0f} p90={float(s.quantile(0.9)):.0f}",
            flush=True,
        )


def _parse_fixed_snapshot(ws: dict) -> int | None:
    raw = ws.get("unique_round_trips")
    if raw is None:
        return None
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    p.add_argument("--days-full", type=int, default=420)
    p.add_argument("--threshold", type=float, default=0.60)
    p.add_argument("--urt-th", type=int, default=582)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--state-log",
        type=Path,
        default=PROJECT_ROOT / "data/diagnostics/fr2/state_log.csv",
        help="CSV for entry snapshot (B_with_meta filter) and reference load_latest_window_stats",
    )
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    state_log_path = args.state_log
    if not state_log_path.is_absolute():
        state_log_path = (PROJECT_ROOT / state_log_path).resolve()

    ws_ref = load_latest_window_stats(state_log_path)
    entry_snapshot_once = load_latest_entry_urt_snapshot(state_log_path, strategy_name_filter="B_with_meta")
    fixed_snapshot = _parse_fixed_snapshot(entry_snapshot_once)

    t_anchor = to_utc_ts(args.t_max)
    df_bt, pl, ps, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
    print(f"anchor={t_anchor.isoformat()} ts_range={ts_min}..{ts_max} rows={len(df_bt)}", flush=True)
    print("--- compare: reference load_latest_window_stats (unchanged) ---", flush=True)
    print(f"load_latest_window_stats({state_log_path}): {ws_ref}", flush=True)
    print("--- compare: fixed entry URT snapshot (B_with_meta rows only) ---", flush=True)
    print(
        f"load_latest_entry_urt_snapshot({state_log_path}, strategy_name_filter='B_with_meta'): "
        f"{entry_snapshot_once}",
        flush=True,
    )
    print(f"compare_fixed_entry_urt_snapshot={fixed_snapshot!r}", flush=True)
    if fixed_snapshot is None:
        print(
            "[warn] fixed snapshot is None; adaptive entry gate falls back to B_with_meta read per run.",
            flush=True,
        )
    _print_csv_urt_histogram(state_log_path)

    windows = [("fixed_180d", 180), ("recent_7d", 7)]
    for name, wdays in windows:
        print(f"\n=== {name} (window_days={wdays}) ===", flush=True)
        print(
            f"summary_fields compare_fixed_entry_urt_snapshot={fixed_snapshot!r} urt_th={int(args.urt_th)}",
            flush=True,
        )
        base = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl,
            ps_primary=ps,
            t_end=t_anchor,
            window_days=wdays,
            threshold=float(args.threshold),
            emit_trade_log=False,
            signal_exit_th_loss_aux_delta_p=None,
            signal_exit_th_loss_aux_mae_cut=None,
            entry_min_unique_round_trips=None,
            entry_urt_state_log_path=state_log_path,
            entry_urt_state_log_snapshot_override=fixed_snapshot,
        )
        adp = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl,
            ps_primary=ps,
            t_end=t_anchor,
            window_days=wdays,
            threshold=float(args.threshold),
            emit_trade_log=False,
            signal_exit_th_loss_aux_delta_p=None,
            signal_exit_th_loss_aux_mae_cut=None,
            entry_min_unique_round_trips=int(args.urt_th),
            entry_urt_state_log_path=state_log_path,
            entry_urt_state_log_snapshot_override=fixed_snapshot,
        )
        rb, ra = _row("baseline", base), _row(f"adaptive_urt>={args.urt_th}", adp)
        for r in (rb, ra):
            print(r, flush=True)
        print(
            "delta",
            {
                "d_total_trades": int(ra["total_trades"]) - int(rb["total_trades"]),
                "d_total_profit": float(ra["total_profit"]) - float(rb["total_profit"]),
                "d_mean_profit": float(ra["mean_profit"]) - float(rb["mean_profit"]),
                "d_max_drawdown": float(ra["max_drawdown"]) - float(rb["max_drawdown"]),
                "d_win_rate": float(ra["win_rate"]) - float(rb["win_rate"]),
            },
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
