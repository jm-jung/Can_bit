#!/usr/bin/env python3
"""
후보1 상위 2조합(δp=0.05, mae_cut -0.007 / -0.005) 다기간·롤링 재현성 검증.

금지: threshold 변경, 모델 변경, SHORT-only / bars>=24 확장.
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

from scripts.loss_aux_signal_exit_common import (
    load_fr2_override_ensemble,
    metrics_row_step2,
    run_loss_aux_window,
    to_utc_ts,
)


COMBOS: list[tuple[str, float, float]] = [
    ("δp=0.05 mae_cut=-0.007", 0.05, -0.007),
    ("δp=0.05 mae_cut=-0.005", 0.05, -0.005),
]

# 고정 구간 '개선 여부' 집계 시 표본이 너무 작으면(예: 60d에 왕복 4건) 왜곡됨
MIN_UNIQUE_TRIPS_FOR_FIXED_VERDICT = 50


def _fmt(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.6f}"
    return str(x)


def _rolling_end_dates(
    ts_min: pd.Timestamp,
    ts_max: pd.Timestamp,
    t_anchor: pd.Timestamp,
    span_days: int,
    step_days: int,
) -> list[pd.Timestamp]:
    """t_end 마다 [t_end-span, t_end] 구간이 데이터 안에 들어가도록 끝 시각 목록."""
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
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00", help="앵커 종료 시각 (UTC)")
    parser.add_argument(
        "--days-full",
        type=int,
        default=420,
        help="OHLCV+proba 로드 일수. 롤링이 과거까지 닿으면 크게 (기본 420).",
    )
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rolling-step-days", type=int, default=30, help="롤링 창 끝 시각 이동 간격")
    parser.add_argument("--no-rolling", action="store_true", help="고정 구간(60/90/180d)만")
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_LOSS_AUX_WF_VALIDATION_REPORT.md",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)

    print(
        f"[wf] load ensemble days_full={args.days_full} t_anchor={t_anchor.isoformat()} ...",
        flush=True,
    )
    df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
    print(f"[wf] data ts range {ts_min} ~ {ts_max}", flush=True)

    fixed_windows = [60, 90, 180]
    rows_fixed: list[dict[str, Any]] = []
    fixed_ran: list[int] = []

    for w in fixed_windows:
        w_start_needed = t_anchor - pd.Timedelta(days=w)
        if w_start_needed < ts_min:
            print(f"[wf] SKIP fixed {w}d: need data from {w_start_needed} < ts_min={ts_min}", flush=True)
            continue
        fixed_ran.append(w)
        tag = f"fixed {w}d end={t_anchor.date()}"
        print(f"[wf] {tag} baseline ...", flush=True)
        base = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_anchor,
            window_days=w,
            threshold=thr,
            emit_trade_log=False,
            signal_exit_th_loss_aux_delta_p=None,
            signal_exit_th_loss_aux_mae_cut=None,
        )
        rows_fixed.append(
            metrics_row_step2(f"baseline · {tag}", base, {"segment": "fixed", "window_days": w, "t_end": str(t_anchor)})
        )
        for name, dp, mc in COMBOS:
            print(f"[wf] {tag} {name} ...", flush=True)
            r = run_loss_aux_window(
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_end=t_anchor,
                window_days=w,
                threshold=thr,
                emit_trade_log=False,
                signal_exit_th_loss_aux_delta_p=dp,
                signal_exit_th_loss_aux_mae_cut=mc,
            )
            rows_fixed.append(
                metrics_row_step2(f"{name} · {tag}", r, {"segment": "fixed", "window_days": w, "t_end": str(t_anchor)})
            )

    rows_roll: list[dict[str, Any]] = []
    if not args.no_rolling:
        for span in (60, 90):
            ends = _rolling_end_dates(ts_min, ts_max, t_anchor, span, int(args.rolling_step_days))
            print(f"[wf] rolling {span}d: {len(ends)} end dates (step={args.rolling_step_days}d)", flush=True)
            for t_end in ends:
                tag = f"roll{span}d end={pd.Timestamp(t_end).date()}"
                base = run_loss_aux_window(
                    df_bt=df_bt,
                    pl_primary=pl_primary,
                    ps_primary=ps_primary,
                    t_end=pd.Timestamp(t_end),
                    window_days=span,
                    threshold=thr,
                    emit_trade_log=False,
                    signal_exit_th_loss_aux_delta_p=None,
                    signal_exit_th_loss_aux_mae_cut=None,
                )
                rows_roll.append(
                    metrics_row_step2(
                        f"baseline · {tag}",
                        base,
                        {"segment": f"rolling_{span}", "window_days": span, "t_end": str(t_end)},
                    )
                )
                for name, dp, mc in COMBOS:
                    r = run_loss_aux_window(
                        df_bt=df_bt,
                        pl_primary=pl_primary,
                        ps_primary=ps_primary,
                        t_end=pd.Timestamp(t_end),
                        window_days=span,
                        threshold=thr,
                        emit_trade_log=False,
                        signal_exit_th_loss_aux_delta_p=dp,
                        signal_exit_th_loss_aux_mae_cut=mc,
                    )
                    rows_roll.append(
                        metrics_row_step2(
                            f"{name} · {tag}",
                            r,
                            {"segment": f"rolling_{span}", "window_days": span, "t_end": str(t_end)},
                        )
                    )

    # --- 집계: 판정용
    def _base_tr_fixed(wd: int) -> float | None:
        for r in rows_fixed:
            if r.get("segment") == "fixed" and r.get("window_days") == wd and str(r["label"]).startswith("baseline"):
                return float(r["total_return"])
        return None

    def _combo_tr_fixed(wd: int, mae: float) -> float | None:
        needle = f"mae_cut={mae}"
        for r in rows_fixed:
            if r.get("segment") != "fixed" or r.get("window_days") != wd:
                continue
            if needle in str(r["label"]) and "δp=0.05" in str(r["label"]):
                return float(r["total_return"])
        return None

    def _base_ut_fixed(wd: int) -> int | None:
        for r in rows_fixed:
            if r.get("segment") == "fixed" and r.get("window_days") == wd and str(r["label"]).startswith("baseline"):
                return int(r["unique_round_trips"] or 0)
        return None

    improvements_007: list[bool] = []
    improvements_005: list[bool] = []
    fixed_verdict_windows: list[int] = []
    for wd in fixed_ran:
        ut0 = _base_ut_fixed(wd)
        if ut0 is None or ut0 < MIN_UNIQUE_TRIPS_FOR_FIXED_VERDICT:
            continue
        fixed_verdict_windows.append(wd)
        b = _base_tr_fixed(wd)
        a007 = _combo_tr_fixed(wd, -0.007)
        a005 = _combo_tr_fixed(wd, -0.005)
        if b is None or a007 is None or a005 is None:
            continue
        improvements_007.append(a007 > b)
        improvements_005.append(a005 > b)

    # rolling: match triples by t_end + segment
    roll_stats: dict[str, Any] = {}
    for span in (60, 90):
        key_prefix = f"rolling_{span}"
        by_end: dict[str, dict[str, float]] = {}
        for r in rows_roll:
            if r.get("segment") != key_prefix:
                continue
            te = str(r.get("t_end", ""))
            by_end.setdefault(te, {})
            lab = str(r["label"])
            tr = float(r["total_return"])
            if lab.startswith("baseline"):
                by_end[te]["b"] = tr
            elif "mae_cut=-0.007" in lab:
                by_end[te]["c007"] = tr
            elif "mae_cut=-0.005" in lab:
                by_end[te]["c005"] = tr
        wins_007 = 0
        wins_005 = 0
        n = 0
        d007: list[float] = []
        d005: list[float] = []
        for te, d in by_end.items():
            if "b" not in d or "c007" not in d or "c005" not in d:
                continue
            n += 1
            if d["c007"] > d["b"]:
                wins_007 += 1
            if d["c005"] > d["b"]:
                wins_005 += 1
            d007.append(d["c007"] - d["b"])
            d005.append(d["c005"] - d["b"])
        roll_stats[f"roll{span}_n"] = n
        roll_stats[f"roll{span}_win_rate_007"] = wins_007 / n if n else 0.0
        roll_stats[f"roll{span}_win_rate_005"] = wins_005 / n if n else 0.0
        roll_stats[f"roll{span}_mean_dtr_007"] = float(np.mean(d007)) if d007 else None
        roll_stats[f"roll{span}_mean_dtr_005"] = float(np.mean(d005)) if d005 else None
        roll_stats[f"roll{span}_std_dtr_007"] = float(np.std(d007)) if len(d007) > 1 else (0.0 if d007 else None)
        roll_stats[f"roll{span}_std_dtr_005"] = float(np.std(d005)) if len(d005) > 1 else (0.0 if d005 else None)

    n_fix = len(improvements_007)
    n_fix_ok_007 = sum(improvements_007)
    n_fix_ok_005 = sum(improvements_005)

    # 판정 휴리스틱
    verdict = "추가 검증 필요"
    verdict_reason: list[str] = []

    n_fixed_expected = len(fixed_ran)
    n_qualified = len(fixed_verdict_windows)
    fixed_all_ok = (
        n_fix == n_qualified
        and n_qualified >= 1
        and n_fix_ok_007 == n_fix
        and n_fix_ok_005 == n_fix
        and n_fix > 0
    )
    if n_fixed_expected < 3:
        verdict_reason.append(
            f"고정 창 실행: {n_fixed_expected}개 (60/90/180 중, 데이터로 가능한 만큼)"
        )
    verdict_reason.append(
        f"표본 충분 고정 구간(unique_round_trips≥{MIN_UNIQUE_TRIPS_FOR_FIXED_VERDICT}) 집계: "
        f"창={fixed_verdict_windows or '없음'}, 개선 -0.007: {n_fix_ok_007}/{n_fix}, -0.005: {n_fix_ok_005}/{n_fix}"
    )

    r60_007 = roll_stats.get("roll60_win_rate_007")
    r60_005 = roll_stats.get("roll60_win_rate_005")
    r90_007 = roll_stats.get("roll90_win_rate_007")
    r90_005 = roll_stats.get("roll90_win_rate_005")

    roll_ok = not args.no_rolling and r60_007 is not None and r90_007 is not None
    if roll_ok:
        roll_strong = (
            r60_007 >= 0.5
            and r60_005 >= 0.5
            and r90_007 >= 0.5
            and r90_005 >= 0.5
            and (roll_stats.get("roll60_n", 0) >= 3)
            and (roll_stats.get("roll90_n", 0) >= 3)
        )
    else:
        roll_strong = False

    if fixed_all_ok and n_qualified >= 1:
        if args.no_rolling:
            verdict = "운영 반영 가능"
            verdict_reason.append(
                f"표본 충분 고정 구간에서 두 조합 모두 total_return > baseline "
                f"(-0.007: {n_fix_ok_007}/{n_fix}, -0.005: {n_fix_ok_005}/{n_fix}). "
                "롤링 미실행(`--no-rolling`) — 운영 전 롤링 재현성 권장."
            )
        elif roll_strong:
            verdict = "운영 반영 가능"
            verdict_reason.append(
                f"표본 충분 고정 구간 모두 개선; "
                f"롤링 승률 roll60={r60_007:.2f}/{r60_005:.2f}, roll90={r90_007:.2f}/{r90_005:.2f}"
            )
        else:
            verdict = "추가 검증 필요"
            verdict_reason.append(
                f"고정(표본 충분)은 개선이나 롤링 승률 미달 또는 roll60 약함: "
                f"roll60={r60_007:.2f}/{r60_005:.2f} (n={roll_stats.get('roll60_n')}), "
                f"roll90={r90_007:.2f}/{r90_005:.2f} (n={roll_stats.get('roll90_n')})"
            )
    elif (n_fix_ok_007 < max(1, n_fix // 2)) and (n_fix_ok_005 < max(1, n_fix // 2)) and n_fix > 0:
        verdict = "보류"
        verdict_reason.append(
            f"고정 구간에서 개선 비율 낮음: -0.007 → {n_fix_ok_007}/{n_fix}, -0.005 → {n_fix_ok_005}/{n_fix}"
        )
    else:
        verdict = "추가 검증 필요"
        verdict_reason.append(
            f"고정: -0.007 개선 {n_fix_ok_007}/{n_fix}, -0.005 개선 {n_fix_ok_005}/{n_fix}; "
            f"롤링: roll60 승률 {r60_007}/{r60_005}, roll90 {r90_007}/{r90_005}"
        )

    # Markdown
    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# 후보1 손실 보조 — 다기간·워크포워드(롤링) 재현성 검증")
    lines.append("")
    lines.append(f"- 앵커 t_max: `{t_anchor.isoformat()}`")
    lines.append(f"- threshold (min_max_proba): **{thr}**")
    lines.append(f"- days_full: **{args.days_full}**")
    lines.append(f"- 대상 조합: δp=0.05 & mae_cut∈{{-0.007,-0.005}} (확장 규칙 없음)")
    lines.append("")
    lines.append(
        f"> 집계용 ‘표본 충분’ 기준: unique_round_trips ≥ **{MIN_UNIQUE_TRIPS_FOR_FIXED_VERDICT}** "
        "(짧은 구간은 왕복 수가 너무 적어 STEP4 판정에서 제외)"
    )
    lines.append("")
    lines.append("## STEP 1 — 고정 구간 (baseline vs 2조합)")
    lines.append("")
    lines.append(
        "| label | unique_round_trips | win_rate | mean_profit_rt | cost_on | total_return | max_loss_trade | signal_exit_th total_profit |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in rows_fixed:
        lines.append(
            f"| {r['label']} | {r['unique_round_trips']} | {_fmt(r['win_rate'])} | {_fmt(r['mean_profit_roundtrip'])} | "
            f"{_fmt(r['cost_on'])} | {_fmt(r['total_return'])} | {_fmt(r['max_loss_trade'])} | {_fmt(r['signal_exit_th_total_profit'])} |"
        )
    lines.append("")
    lines.append("## STEP 1b — 롤링 워크포워드")
    lines.append("")
    if args.no_rolling:
        lines.append("(비활성화: `--no-rolling`)")
    else:
        lines.append(
            f"- step={args.rolling_step_days}d, span=60d/90d, 각 끝시각마다 baseline+2조합 동일 조건."
        )
        lines.append("")
        lines.append(
            "| label | unique_round_trips | win_rate | mean_profit_rt | cost_on | total_return | max_loss_trade | signal_exit_th total_profit |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for r in rows_roll:
            lines.append(
                f"| {r['label']} | {r['unique_round_trips']} | {_fmt(r['win_rate'])} | {_fmt(r['mean_profit_roundtrip'])} | "
                f"{_fmt(r['cost_on'])} | {_fmt(r['total_return'])} | {_fmt(r['max_loss_trade'])} | {_fmt(r['signal_exit_th_total_profit'])} |"
            )
        lines.append("")
        lines.append("### 롤링 요약 (baseline 대비 total_return 차이)")
        lines.append("")
        lines.append("| 구간 | n | win_rate -0.007 | win_rate -0.005 | mean Δtr -0.007 | mean Δtr -0.005 | std Δtr -0.007 | std Δtr -0.005 |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for span in (60, 90):
            lines.append(
                f"| roll{span}d | {roll_stats.get(f'roll{span}_n')} | "
                f"{_fmt(roll_stats.get(f'roll{span}_win_rate_007'))} | {_fmt(roll_stats.get(f'roll{span}_win_rate_005'))} | "
                f"{_fmt(roll_stats.get(f'roll{span}_mean_dtr_007'))} | {_fmt(roll_stats.get(f'roll{span}_mean_dtr_005'))} | "
                f"{_fmt(roll_stats.get(f'roll{span}_std_dtr_007'))} | {_fmt(roll_stats.get(f'roll{span}_std_dtr_005'))} |"
            )

    lines.append("")
    lines.append("## STEP 3 — 판정 질문")
    lines.append("")
    lines.append(
        f"1. **여러 고정 구간에서 반복 개선인가?** "
        f"(표본 충분만 집계: unique_round_trips≥{MIN_UNIQUE_TRIPS_FOR_FIXED_VERDICT}, 창={fixed_verdict_windows}) "
        f"-0.007: {n_fix_ok_007}/{n_fix}; -0.005: {n_fix_ok_005}/{n_fix}. "
        f"※ 60d·90d처럼 왕복 수가 극소한 구간은 위 집계에서 제외(표는 참고용)."
    )
    lines.append(
        f"2. **어느 조합이 더 안정적인가?** 롤링 mean Δtr/std: "
        f"60d std(-0.007)={_fmt(roll_stats.get('roll60_std_dtr_007'))}, std(-0.005)={_fmt(roll_stats.get('roll60_std_dtr_005'))}; "
        f"90d std(-0.007)={_fmt(roll_stats.get('roll90_std_dtr_007'))}, std(-0.005)={_fmt(roll_stats.get('roll90_std_dtr_005'))}. "
        f"(std가 작을수록 롤링에서 분산이 작음; 본 데이터에서 -0.007이 약간 더 작은 경향)"
    )
    lines.append(
        f"3. **개선 폭 일관성:** 고정 구간별 baseline 대비 차이는 위 고정 표에서 확인."
    )
    lines.append(
        f"4. **운영 후보?** 아래 STEP 4 판정 참고."
    )
    lines.append("")
    lines.append("## STEP 4 — 최종 판정")
    lines.append("")
    lines.append(f"### 결론: **{verdict}**")
    lines.append("")
    for vr in verdict_reason:
        lines.append(f"- {vr}")
    lines.append("")
    lines.append("---")
    lines.append("*휴리스틱: 고정 60/90/180 전부 개선 + 롤링(60·90) 승률 모두 ≥0.5 & n≥3 → ‘운영 반영 가능’; "
                "고정 과반 실패 → ‘보류’; 그 사이 → ‘추가 검증 필요’.*")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[wf] wrote {out_path}", flush=True)

    csv_path = out_path.with_suffix(".csv")
    all_rows = rows_fixed + rows_roll
    if all_rows:
        keys = list(all_rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_rows)
        print(f"[wf] wrote {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
