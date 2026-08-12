#!/usr/bin/env python3
"""
손실 보조(δp=0.05, mae_cut=-0.007) + 보호 게이트 B/C 검증.

BASELINE / A(보조만) / B(Δunreal>0 스킵) / C(B + MFE≥winners median 스킵)

금지: 파라미터 스윕, threshold, 모델, SHORT OFF, time_stop/early_exit 변경.
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

# 이전 180d 분석: signal_exit_th 중 profit>0 (winners) MFE 중앙값 — 고정 1개 (스윕 아님)
MFE_WINNERS_MEDIAN_FIXED = 0.00912629931357865

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


def _row_metrics(
    label: str,
    segment: str,
    res: dict[str, Any],
    *,
    pd_count: int | None,
    pd_damage: float | None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ut = res.get("unique_round_trips")
    wr = res.get("win_rate")
    row: dict[str, Any] = {
        "label": label,
        "segment": segment,
        "unique_round_trips": int(ut) if ut is not None else None,
        "win_rate": float(wr) if wr is not None else None,
        "mean_profit_roundtrip": mean_profit_roundtrip(res),
        "total_return": float(res.get("total_return", 0.0) or 0.0),
        "signal_exit_th_total_profit": float(res.get("signal_exit_th_total_profit", 0.0) or 0.0),
        "max_loss_trade": max_loss_trade(res),
        "profit_damaged_count": pd_count,
        "profit_damaged_total_damage": pd_damage,
        "aux_applied": int(res.get("signal_exit_th_loss_aux_applied", 0) or 0),
    }
    if extra:
        row.update(extra)
    return row


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=420)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rolling-step-days", type=int, default=30)
    parser.add_argument("--only-fixed", action="store_true", help="180d 고정만 (롤링 생략)")
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_LOSS_AUX_GATE_VALIDATION_REPORT.md",
    )
    parser.add_argument(
        "--out-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_LOSS_AUX_GATE_VALIDATION_REPORT.csv",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)

    print(f"[gate] load ensemble days_full={args.days_full} ...", flush=True)
    df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
    print(f"[gate] ts range {ts_min} ~ {ts_max}", flush=True)

    rows: list[dict[str, Any]] = []

    def run_combo(
        *,
        t_end: pd.Timestamp,
        window_days: int,
        tag: str,
        segment: str,
        extra: dict[str, Any],
    ) -> None:
        base = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            window_days=window_days,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=None,
            signal_exit_th_loss_aux_mae_cut=None,
        )
        a = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            window_days=window_days,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=DELTA_P,
            signal_exit_th_loss_aux_mae_cut=MAE_CUT,
            signal_exit_th_loss_aux_gate_delta_unreal_positive=False,
            signal_exit_th_loss_aux_gate_mfe_min=None,
        )
        b = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            window_days=window_days,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=DELTA_P,
            signal_exit_th_loss_aux_mae_cut=MAE_CUT,
            signal_exit_th_loss_aux_gate_delta_unreal_positive=True,
            signal_exit_th_loss_aux_gate_mfe_min=None,
        )
        c = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            window_days=window_days,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=DELTA_P,
            signal_exit_th_loss_aux_mae_cut=MAE_CUT,
            signal_exit_th_loss_aux_gate_delta_unreal_positive=True,
            signal_exit_th_loss_aux_gate_mfe_min=MFE_WINNERS_MEDIAN_FIXED,
        )

        pda_n, pda_d = profit_damaged_count_and_damage(base, a)
        pdb_n, pdb_d = profit_damaged_count_and_damage(base, b)
        pdc_n, pdc_d = profit_damaged_count_and_damage(base, c)

        rows.append(
            _row_metrics(f"BASELINE · {tag}", segment, base, pd_count=0, pd_damage=0.0, extra=extra)
        )
        rows.append(_row_metrics(f"A · {tag}", segment, a, pd_count=pda_n, pd_damage=pda_d, extra=extra))
        rows.append(_row_metrics(f"B · {tag}", segment, b, pd_count=pdb_n, pd_damage=pdb_d, extra=extra))
        rows.append(_row_metrics(f"C · {tag}", segment, c, pd_count=pdc_n, pd_damage=pdc_d, extra=extra))

    # --- fixed 180d
    w = 180
    if t_anchor - pd.Timedelta(days=w) >= ts_min:
        tag = f"fixed {w}d end={t_anchor.date()}"
        print(f"[gate] {tag} ...", flush=True)
        run_combo(t_end=t_anchor, window_days=w, tag=tag, segment="fixed_180d", extra={"window_days": w, "t_end": str(t_anchor)})
    else:
        print(f"[gate] SKIP fixed 180d: window start < ts_min", flush=True)

    if not args.only_fixed:
        for span in (90, 60):
            ends = _rolling_end_dates(ts_min, ts_max, t_anchor, span, int(args.rolling_step_days))
            print(f"[gate] rolling {span}d: {len(ends)} windows", flush=True)
            for t_end in ends:
                te = pd.Timestamp(t_end)
                tag = f"roll{span}d end={te.date()}"
                print(f"[gate] {tag} ...", flush=True)
                run_combo(
                    t_end=te,
                    window_days=span,
                    tag=tag,
                    segment=f"rolling_{span}d",
                    extra={"window_days": span, "t_end": str(te)},
                )

    # --- CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        keys = list(rows[0].keys())
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            cw = csv.DictWriter(f, fieldnames=keys)
            cw.writeheader()
            cw.writerows(rows)

    # --- MD: 집계 표 (fixed 180d 한 줄 + 롤링 요약)
    def _fmt(x: Any) -> str:
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.6f}"
        return str(x)

    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH 손실 보조 — 보호 게이트 검증")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| t_anchor | `{t_anchor.isoformat()}` |")
    lines.append(f"| threshold | {thr} |")
    lines.append(f"| δ_p / mae_cut | {DELTA_P} / {MAE_CUT} |")
    lines.append(f"| MFE 게이트 (C) | >= **{MFE_WINNERS_MEDIAN_FIXED}** (180d winners MFE 중앙값 고정) |")
    lines.append(f"| B | `delta_unreal_4bars > 0` 이면 보조 미적용 |")
    lines.append(f"| C | B + `MFE >= {MFE_WINNERS_MEDIAN_FIXED}` 이면 미적용 |")
    lines.append("")

    df_rows = pd.DataFrame(rows)
    fixed = df_rows[df_rows["segment"] == "fixed_180d"] if len(df_rows) else pd.DataFrame()

    lines.append("## A. BASELINE / A / B / C 비교 (fixed 180d)")
    lines.append("")
    if fixed.empty:
        lines.append("*(fixed 180d 행 없음 — 데이터 범위 확인)*")
    else:
        lines.append(
            "| 실험 | unique_round_trips | total_return | mean_profit_roundtrip | sth total_profit | max_loss_trade | profit_damaged n | profit_damaged Σdamage | aux_applied |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for lab in ("BASELINE", "A", "B", "C"):
            sub = fixed[fixed["label"].str.startswith(lab + " ·")]
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"| {lab} | {r['unique_round_trips']} | {_fmt(r['total_return'])} | {_fmt(r['mean_profit_roundtrip'])} | "
                f"{_fmt(r['signal_exit_th_total_profit'])} | {_fmt(r['max_loss_trade'])} | {r['profit_damaged_count']} | "
                f"{_fmt(r['profit_damaged_total_damage'])} | {r['aux_applied']} |"
            )
    lines.append("")

    for seg_name, span in (("rolling_90d", 90), ("rolling_60d", 60)):
        sub = df_rows[df_rows["segment"] == seg_name] if len(df_rows) else pd.DataFrame()
        lines.append(f"### 참고: roll{span} 요약 (창당 평균)")
        lines.append("")
        if sub.empty:
            lines.append("*(데이터 없음)*")
            lines.append("")
            continue
        agg = []
        for lab in ("BASELINE", "A", "B", "C"):
            part = sub[sub["label"].str.startswith(lab + " ·")]
            if part.empty:
                continue
            agg.append(
                {
                    "exp": lab,
                    "mean_total_return": float(part["total_return"].mean()),
                    "mean_sth_tp": float(part["signal_exit_th_total_profit"].mean()),
                    "mean_pd_count": float(part["profit_damaged_count"].mean()),
                    "mean_pd_damage": float(part["profit_damaged_total_damage"].mean()),
                    "mean_aux": float(part["aux_applied"].mean()),
                    "n": len(part),
                }
            )
        lines.append(
            "| 실험 | n_windows | mean total_return | mean sth total_profit | mean pd_count | mean pd_damage | mean aux_applied |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for a in agg:
            lines.append(
                f"| {a['exp']} | {a['n']} | {_fmt(a['mean_total_return'])} | {_fmt(a['mean_sth_tp'])} | "
                f"{_fmt(a['mean_pd_count'])} | {_fmt(a['mean_pd_damage'])} | {_fmt(a['mean_aux'])} |"
            )
        lines.append("")

    # 판정
    lines.append("## B. profit_damaged 변화 요약")
    lines.append("")
    if not fixed.empty:
        ra = fixed[fixed["label"].str.startswith("A ·")].iloc[0]
        rb = fixed[fixed["label"].str.startswith("B ·")].iloc[0]
        rc = fixed[fixed["label"].str.startswith("C ·")].iloc[0]
        lines.append(
            f"- **A:** profit_damaged **{int(ra['profit_damaged_count'])}**, Σdamage **{_fmt(ra['profit_damaged_total_damage'])}**"
        )
        lines.append(
            f"- **B:** **{int(rb['profit_damaged_count'])}**, Σdamage **{_fmt(rb['profit_damaged_total_damage'])}**"
        )
        lines.append(
            f"- **C:** **{int(rc['profit_damaged_count'])}**, Σdamage **{_fmt(rc['profit_damaged_total_damage'])}**"
        )
    lines.append("")

    lines.append("## C. 손실 완화 유지 여부 (sth total_profit, fixed 180d)")
    lines.append("")
    if not fixed.empty:
        base = fixed[fixed["label"].str.startswith("BASELINE ·")].iloc[0]
        sth_b = float(base["signal_exit_th_total_profit"])
        for lab in ("A", "B", "C"):
            r = fixed[fixed["label"].str.startswith(lab + " ·")].iloc[0]
            sth = float(r["signal_exit_th_total_profit"])
            ok = sth > sth_b
            lines.append(f"- **{lab}:** sth total {sth:.6f} vs BASELINE {sth_b:.6f} → {'개선' if ok else '악화/동일'}")
    lines.append("")

    lines.append("## D. 최종 판정 (숫자 기반)")
    lines.append("")
    verdict = "연구 후보 유지"
    reason: list[str] = []
    if not fixed.empty:
        base = fixed[fixed["label"].str.startswith("BASELINE ·")].iloc[0]
        ra = fixed[fixed["label"].str.startswith("A ·")].iloc[0]
        rb = fixed[fixed["label"].str.startswith("B ·")].iloc[0]
        rc = fixed[fixed["label"].str.startswith("C ·")].iloc[0]
        sth_b = float(base["signal_exit_th_total_profit"])
        pd_a = int(ra["profit_damaged_count"])
        dmg_a = float(ra["profit_damaged_total_damage"] or 0)

        def _pd_improved(r: pd.Series) -> bool:
            n, d = int(r["profit_damaged_count"]), float(r["profit_damaged_total_damage"] or 0)
            if n < pd_a:
                return True
            if n == pd_a and d >= dmg_a:
                return True
            return False

        def _sth_improved(r: pd.Series) -> bool:
            return float(r["signal_exit_th_total_profit"]) > sth_b

        b_hit = _pd_improved(rb) and _sth_improved(rb)
        c_hit = _pd_improved(rc) and _sth_improved(rc)

        roll60 = df_rows[df_rows["segment"] == "rolling_60d"] if len(df_rows) else pd.DataFrame()
        r60_volatile = False
        if not roll60.empty:
            a60 = roll60[roll60["label"].str.startswith("A ·")]
            if len(a60) > 1:
                tr_std = float(a60["total_return"].std(ddof=1))
                tr_m = abs(float(a60["total_return"].mean()))
                if np.isfinite(tr_std) and tr_m > 1e-12 and tr_std > 0.25 * tr_m:
                    r60_volatile = True
                    reason.append(
                        f"roll60에서 A의 total_return 변동성(std={tr_std:.6f})이 평균(|{tr_m:.6f}|)의 25% 초과 → 짧은 창 흔들림."
                    )

        roll60_ran = bool(len(df_rows) and not df_rows[df_rows["segment"] == "rolling_60d"].empty)

        if not (b_hit or c_hit):
            verdict = "연구 후보 유지"
            reason.append("게이트가 profit_damaged를 줄이지 못하거나 sth 합 개선을 깨뜨림.")
        elif b_hit and c_hit and not r60_volatile and roll60_ran:
            verdict = "운영 반영 가능"
            reason.append("B·C 모두: A 대비 손상 완화 + sth 개선 + roll60 과도 변동 아님(휴리스틱).")
        elif b_hit or c_hit:
            verdict = "추가 검증 필요"
            if not roll60_ran:
                reason.append("180d에서 B/C 조건 충족. roll60·roll90 미산출 — 롤링 후 재판정 권장.")
            elif not r60_volatile:
                reason.append("180d에서 B/C 조건 충족. roll60 변동성 휴리스틱 통과 시 운영 검토 가능.")

        if r60_volatile and verdict == "운영 반영 가능":
            verdict = "추가 검증 필요"
            reason.append("roll60 흔들림으로 운영 등급 하향.")

        if verdict == "운영 반영 가능" and not roll60_ran:
            verdict = "추가 검증 필요"
            reason.append("roll60·roll90 미산출 — 전체 스크립트 실행 후 재판정 권장.")

    lines.append(f"**판정: {verdict}**")
    for r in reason:
        lines.append(f"- {r}")
    lines.append("")
    lines.append("### 질문 답")
    lines.append("")
    lines.append(
        "1. **게이트가 profit_damaged를 줄이는가?** — fixed 180d에서 B/C의 `profit_damaged_count`·Σdamage와 A를 비교."
    )
    lines.append(
        "2. **손실 완화 유지?** — `signal_exit_th total_profit`이 BASELINE보다 나은지(A/B/C 각각)."
    )
    lines.append(
        "3. **운영 후보 승급?** — 180d에서 게이트가 손상 감소 + sth 개선을 동시에 만족하고, roll60이 과도하지 않을 때만."
    )
    lines.append(
        "4. **roll60 흔들림** — 참고 표에서 A의 `total_return` 분산이 크면 연구 후보 유지가 안전."
    )
    lines.append("")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")
    lines.append("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"[gate] wrote {args.out_md} rows={len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
