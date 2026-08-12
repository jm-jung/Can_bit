#!/usr/bin/env python3
"""
손실 보조 후보(B: delta_unreal_4bars>0 보호 포함)를 "언제 켤지" 게이트 비교.

게이트:
- trades_gate: unique_round_trips >= 100 일 때 ON
- rolling90_gate: window_days == 90 일 때 ON
- hybrid_gate: (window_days == 90) AND (unique_round_trips >= 100) 일 때 ON

검증 구간:
- fixed 180d
- roll90
- roll60 (참고)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loss_aux_signal_exit_common import load_fr2_override_ensemble, run_loss_aux_window, to_utc_ts

DELTA_P = 0.05
MAE_CUT = -0.007


def _sth_exits(res: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in res.get("trade_events") or []:
        if not str(e.get("event", "")).startswith("EXIT"):
            continue
        if e.get("exit_reason") != "signal_exit_th":
            continue
        out.append(e)
    return out


def _trade_key(e: dict[str, Any]) -> tuple[str, str, str]:
    return (str(e.get("entry_ts", "")), str(e.get("exit_ts", "")), str(e.get("direction", "")))


def _pnl(e: dict[str, Any]) -> float:
    if e.get("pnl_change") is not None:
        return float(e["pnl_change"])
    return float(e.get("profit", 0.0) or 0.0)


def profit_damaged_count_and_damage(base: dict[str, Any], combo: dict[str, Any]) -> tuple[int, float]:
    eb = {_trade_key(e): e for e in _sth_exits(base)}
    ec = {_trade_key(e): e for e in _sth_exits(combo)}
    common = set(eb.keys()) & set(ec.keys())
    n = 0
    dmg = 0.0
    for k in common:
        pb, pc = _pnl(eb[k]), _pnl(ec[k])
        if pb > 0 and pc < pb:
            n += 1
            dmg += pc - pb
    return n, dmg


def mean_profit_roundtrip(res: dict[str, Any]) -> float | None:
    from src.backtest.engine import dedupe_trades_round_trips

    trades = dedupe_trades_round_trips(list(res.get("trades") or []))
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(np.mean(vals))


def max_loss_trade(res: dict[str, Any]) -> float | None:
    from src.backtest.engine import dedupe_trades_round_trips

    trades = dedupe_trades_round_trips(list(res.get("trades") or []))
    if not trades:
        return None
    vals = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    if not vals:
        return None
    return float(min(vals))


def _rolling_end_dates(
    ts_min: pd.Timestamp,
    ts_max: pd.Timestamp,
    t_anchor: pd.Timestamp,
    span_days: int,
    step_days: int,
) -> list[pd.Timestamp]:
    ends: list[pd.Timestamp] = []
    t_end = min(t_anchor, ts_max)
    while True:
        w_start = t_end - pd.Timedelta(days=span_days)
        if w_start < ts_min:
            break
        ends.append(t_end)
        t_end = t_end - pd.Timedelta(days=step_days)
    return ends


def gate_on(gate_name: str, window_days: int, unique_trips: int) -> bool:
    if gate_name == "trades_gate":
        return unique_trips >= 100
    if gate_name == "rolling90_gate":
        return window_days == 90
    if gate_name == "hybrid_gate":
        return window_days == 90 and unique_trips >= 100
    raise ValueError(f"Unknown gate_name: {gate_name}")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=420)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rolling-step-days", type=int, default=30)
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_GATE_COMPARE_REPORT.md",
    )
    parser.add_argument(
        "--out-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_GATE_COMPARE_REPORT.csv",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)

    print(f"[act-gate] load ensemble days_full={args.days_full} ...", flush=True)
    df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
    print(f"[act-gate] ts range {ts_min} ~ {ts_max}", flush=True)

    window_specs: list[tuple[str, int, pd.Timestamp]] = []
    if t_anchor - pd.Timedelta(days=180) >= ts_min:
        window_specs.append(("fixed_180d", 180, t_anchor))
    for span in (90, 60):
        for te in _rolling_end_dates(ts_min, ts_max, t_anchor, span, int(args.rolling_step_days)):
            window_specs.append((f"rolling_{span}d", span, pd.Timestamp(te)))

    rows_detail: list[dict[str, Any]] = []
    gates = ("trades_gate", "rolling90_gate", "hybrid_gate")

    for segment, wdays, t_end in window_specs:
        tag = f"{segment} end={pd.Timestamp(t_end).date()}"
        print(f"[act-gate] {tag} baseline/B ...", flush=True)

        base = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            window_days=wdays,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=None,
            signal_exit_th_loss_aux_mae_cut=None,
        )
        b = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            window_days=wdays,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=DELTA_P,
            signal_exit_th_loss_aux_mae_cut=MAE_CUT,
            signal_exit_th_loss_aux_gate_delta_unreal_positive=True,
            signal_exit_th_loss_aux_gate_mfe_min=None,
        )
        pd_n_b, pd_d_b = profit_damaged_count_and_damage(base, b)
        ut = int(base.get("unique_round_trips") or 0)

        for gate in gates:
            on = gate_on(gate, wdays, ut)
            sel = b if on else base
            pd_n = pd_n_b if on else 0
            pd_d = pd_d_b if on else 0.0
            rows_detail.append(
                {
                    "segment": segment,
                    "window_days": wdays,
                    "t_end": str(t_end),
                    "gate": gate,
                    "gate_on": int(on),
                    "unique_round_trips": int(sel.get("unique_round_trips") or 0),
                    "total_return": float(sel.get("total_return", 0.0) or 0.0),
                    "mean_profit_roundtrip": mean_profit_roundtrip(sel),
                    "signal_exit_th_total_profit": float(sel.get("signal_exit_th_total_profit", 0.0) or 0.0),
                    "max_loss_trade": max_loss_trade(sel),
                    "profit_damaged_count": int(pd_n),
                    "profit_damaged_total_damage": float(pd_d),
                    "applied_trade_count": int(sel.get("signal_exit_th_loss_aux_applied", 0) or 0),
                }
            )
        # baseline reference row
        rows_detail.append(
            {
                "segment": segment,
                "window_days": wdays,
                "t_end": str(t_end),
                "gate": "baseline_ref",
                "gate_on": 0,
                "unique_round_trips": int(base.get("unique_round_trips") or 0),
                "total_return": float(base.get("total_return", 0.0) or 0.0),
                "mean_profit_roundtrip": mean_profit_roundtrip(base),
                "signal_exit_th_total_profit": float(base.get("signal_exit_th_total_profit", 0.0) or 0.0),
                "max_loss_trade": max_loss_trade(base),
                "profit_damaged_count": 0,
                "profit_damaged_total_damage": 0.0,
                "applied_trade_count": 0,
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows_detail[0].keys()) if rows_detail else []
    if keys:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=keys)
            cw.writeheader()
            cw.writerows(rows_detail)

    df = pd.DataFrame(rows_detail)
    base_df = df[df["gate"] == "baseline_ref"].copy()
    gate_df = df[df["gate"].isin(gates)].copy()

    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH 활성화 게이트 비교")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| t_anchor | `{t_anchor.isoformat()}` |")
    lines.append(f"| threshold | {thr} |")
    lines.append(f"| 손실 보조 기본 | δp={DELTA_P}, mae_cut={MAE_CUT}, 보호게이트(B: delta_unreal_4bars>0 skip) |")
    lines.append("")

    lines.append("## A. 게이트별 비교 표")
    lines.append("")
    for seg_name, title in (
        ("fixed_180d", "fixed 180d"),
        ("rolling_90d", "roll90 (평균)"),
        ("rolling_60d", "roll60 (평균, 참고)"),
    ):
        gseg = gate_df[gate_df["segment"] == seg_name]
        bseg = base_df[base_df["segment"] == seg_name]
        lines.append(f"### {title}")
        lines.append("")
        if gseg.empty or bseg.empty:
            lines.append("*(데이터 없음)*")
            lines.append("")
            continue
        b_tr = float(bseg["total_return"].mean())
        b_sth = float(bseg["signal_exit_th_total_profit"].mean())
        lines.append(
            "| gate | ON ratio | unique_round_trips | total_return | Δtr vs baseline | sth total_profit | Δsth vs baseline | profit_damaged n | pd total damage | applied_trade_count |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for gate in gates:
            part = gseg[gseg["gate"] == gate]
            if part.empty:
                continue
            on_ratio = float(part["gate_on"].mean())
            tr = float(part["total_return"].mean())
            sth = float(part["signal_exit_th_total_profit"].mean())
            lines.append(
                f"| {gate} | {on_ratio:.3f} | {_fmt(float(part['unique_round_trips'].mean()))} | "
                f"{tr:.6f} | {(tr - b_tr):.6f} | {sth:.6f} | {(sth - b_sth):.6f} | "
                f"{_fmt(float(part['profit_damaged_count'].mean()))} | {_fmt(float(part['profit_damaged_total_damage'].mean()))} | "
                f"{_fmt(float(part['applied_trade_count'].mean()))} |"
            )
        lines.append("")

    # select best gate (stability + improvement heuristic)
    score_rows: list[tuple[str, float]] = []
    for gate in gates:
        g = gate_df[gate_df["gate"] == gate]
        if g.empty:
            continue
        b90 = base_df[base_df["segment"] == "rolling_90d"]
        g90 = g[g["segment"] == "rolling_90d"]
        b60 = base_df[base_df["segment"] == "rolling_60d"]
        g60 = g[g["segment"] == "rolling_60d"]
        b180 = base_df[base_df["segment"] == "fixed_180d"]
        g180 = g[g["segment"] == "fixed_180d"]
        score = 0.0
        if not b180.empty and not g180.empty:
            score += 4.0 * (float(g180["total_return"].mean()) - float(b180["total_return"].mean()))
            score += 3.0 * (float(g180["signal_exit_th_total_profit"].mean()) - float(b180["signal_exit_th_total_profit"].mean()))
            score -= 1.5 * float(g180["profit_damaged_count"].mean())
        if not b90.empty and not g90.empty:
            score += 2.0 * (float(g90["total_return"].mean()) - float(b90["total_return"].mean()))
            score += 1.5 * (float(g90["signal_exit_th_total_profit"].mean()) - float(b90["signal_exit_th_total_profit"].mean()))
            score -= 0.5 * float(g90["profit_damaged_count"].mean())
        if not b60.empty and not g60.empty:
            score += 1.0 * (float(g60["total_return"].mean()) - float(b60["total_return"].mean()))
            score += 0.8 * (float(g60["signal_exit_th_total_profit"].mean()) - float(b60["signal_exit_th_total_profit"].mean()))
            score -= 0.4 * float(g60["profit_damaged_count"].mean())
            score -= 0.3 * float(g60["total_return"].std(ddof=1) or 0.0)
        score_rows.append((gate, score))
    best_gate = sorted(score_rows, key=lambda x: x[1], reverse=True)[0][0] if score_rows else "N/A"

    lines.append("## B. 최적 게이트 1개")
    lines.append("")
    lines.append(f"- **{best_gate}**")
    lines.append("- 선정 기준: fixed 180d 개선 유지 + roll90 개선 지속 + roll60 흔들림(분산/손상) 페널티를 함께 반영한 휴리스틱 점수.")
    lines.append("")

    lines.append("## C. 운영 반영 가능 여부")
    lines.append("")
    verdict = "Conditional"
    reason = []
    if best_gate != "N/A":
        g = gate_df[gate_df["gate"] == best_gate]
        b = base_df
        def _m(seg: str, col: str) -> tuple[float, float]:
            gs = g[g["segment"] == seg]
            bs = b[b["segment"] == seg]
            if gs.empty or bs.empty:
                return (float("nan"), float("nan"))
            return (float(gs[col].mean()), float(bs[col].mean()))

        g180_tr, b180_tr = _m("fixed_180d", "total_return")
        g180_sth, b180_sth = _m("fixed_180d", "signal_exit_th_total_profit")
        g90_tr, b90_tr = _m("rolling_90d", "total_return")
        g60_pd, _ = _m("rolling_60d", "profit_damaged_count")
        g60 = g[g["segment"] == "rolling_60d"]
        roll60_vol = float(g60["total_return"].std(ddof=1)) if len(g60) > 1 else float("nan")

        cond_180 = np.isfinite(g180_tr) and np.isfinite(b180_tr) and g180_tr > b180_tr and g180_sth > b180_sth
        cond_90 = np.isfinite(g90_tr) and np.isfinite(b90_tr) and g90_tr >= b90_tr
        cond_60 = np.isfinite(g60_pd) and g60_pd <= 8.0  # 평균 pd_count 경험적 안전선

        if cond_180 and cond_90 and cond_60:
            verdict = "Conditional"
            reason.append("180d/roll90 개선 유지 + roll60 손상지표 완화 확인. 다만 roll60 변동성으로 전역 ON은 보수적으로.")
        else:
            verdict = "No"
            reason.append("한 개 이상 핵심 조건(180d 개선, roll90 유지, roll60 손상 완화) 미충족.")

        reason.append(f"best_gate={best_gate}, roll60 total_return std={_fmt(roll60_vol)}")

    lines.append(f"- **{verdict}**")
    for r in reason:
        lines.append(f"- {r}")
    lines.append("")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[act-gate] wrote {out_md} rows={len(rows_detail)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
