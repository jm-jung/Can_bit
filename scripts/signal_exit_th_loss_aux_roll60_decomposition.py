#!/usr/bin/env python3
"""
δp=0.05, mae_cut=-0.007 단일 후보만 — roll60 창별 baseline 대비 개선/악화 분해.

금지: 파라미터 스윕, threshold 변경, 모델 변경.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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


def _rolling_ends(
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


def _fmt_pct_dist(c: Counter, keys: list[str]) -> str:
    tot = sum(c.get(k, 0) for k in keys)
    if tot == 0:
        return "—"
    parts = [f"{k}:{100 * c.get(k, 0) / tot:.0f}%" for k in keys]
    return ", ".join(parts)


def _lookup_regime_row(df_reg: pd.DataFrame, t: pd.Timestamp) -> pd.Series | None:
    ts = pd.to_datetime(df_reg["timestamp"], utc=True)
    et = pd.Timestamp(t)
    if et.tzinfo is None:
        et = et.tz_localize("UTC")
    else:
        et = et.tz_convert("UTC")
    m = ts == et
    if m.any():
        return df_reg.loc[m].iloc[0]
    j = int((ts - et).abs().argmin())
    return df_reg.iloc[j]


def aggregate_exit_events(
    res: dict[str, Any],
    df_reg_window: pd.DataFrame,
) -> dict[str, Any]:
    """EXIT 이벤트 기준: 방향 비중, MAE/MFE 평균, 진입 시점 trend/vol 레짐 비중."""
    events = [e for e in (res.get("trade_events") or []) if str(e.get("event", "")).startswith("EXIT")]
    if not events:
        return {
            "n_exit_events": 0,
            "long_share": None,
            "short_share": None,
            "mean_mae": None,
            "mean_mfe": None,
            "trend_dist_str": "—",
            "vol_dist_str": "—",
            "trend_counter": Counter(),
            "vol_counter": Counter(),
        }

    directions = [str(e.get("direction", "")) for e in events]
    n = len(directions)
    n_long = sum(1 for d in directions if d == "LONG")
    n_short = sum(1 for d in directions if d == "SHORT")

    maes: list[float] = []
    mfes: list[float] = []
    trend_c = Counter()
    vol_c = Counter()

    for e in events:
        mae = e.get("max_adverse_excursion")
        mfe = e.get("max_favorable_excursion")
        if mae is not None and np.isfinite(float(mae)):
            maes.append(float(mae))
        if mfe is not None and np.isfinite(float(mfe)):
            mfes.append(float(mfe))

        entry_ts = e.get("entry_ts")
        if entry_ts is None:
            continue
        try:
            row = _lookup_regime_row(df_reg_window, pd.Timestamp(entry_ts))
        except Exception:
            continue
        if row is None:
            continue
        tr = row.get("trend_regime")
        vr = row.get("vol_regime")
        if tr is not None and pd.notna(tr):
            trend_c[str(tr)] += 1
        if vr is not None and pd.notna(vr):
            vol_c[str(vr)] += 1

    return {
        "n_exit_events": n,
        "long_share": n_long / n if n else None,
        "short_share": n_short / n if n else None,
        "mean_mae": float(np.mean(maes)) if maes else None,
        "mean_mfe": float(np.mean(mfes)) if mfes else None,
        "trend_dist_str": _fmt_pct_dist(trend_c, ["uptrend", "downtrend"]),
        "vol_dist_str": _fmt_pct_dist(vol_c, ["low_vol", "mid_vol", "high_vol"]),
        "trend_counter": trend_c,
        "vol_counter": vol_c,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=420)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rolling-step-days", type=int, default=30)
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_LOSS_AUX_ROLL60_DECOMPOSITION.md",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)

    from scripts.run_fr2_regime_conditioning import add_regime_columns

    print(f"[decomp] load data days_full={args.days_full} ...", flush=True)
    df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
    df_reg_full = add_regime_columns(int(args.days_full), df_bt.copy())

    ends = _rolling_ends(ts_min, ts_max, t_anchor, span_days=60, step_days=int(args.rolling_step_days))
    print(f"[decomp] roll60 end dates: {len(ends)}", flush=True)

    rows: list[dict[str, Any]] = []

    for t_end in ends:
        window_start = t_end - pd.Timedelta(days=60)
        tser = pd.to_datetime(df_bt["timestamp"], utc=True)
        mask = (tser >= window_start) & (tser <= t_end)
        df_reg_w = df_reg_full.loc[mask].reset_index(drop=True)

        tag = f"roll60d end={pd.Timestamp(t_end).date()}"
        print(f"[decomp] {tag} baseline+combo (emit_trade_log) ...", flush=True)

        base = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=pd.Timestamp(t_end),
            window_days=60,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=None,
            signal_exit_th_loss_aux_mae_cut=None,
        )
        combo = run_loss_aux_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=pd.Timestamp(t_end),
            window_days=60,
            threshold=thr,
            emit_trade_log=True,
            signal_exit_th_loss_aux_delta_p=DELTA_P,
            signal_exit_th_loss_aux_mae_cut=MAE_CUT,
        )

        tr_b = float(base.get("total_return", 0.0) or 0.0)
        tr_c = float(combo.get("total_return", 0.0) or 0.0)
        sth_b = float(base.get("signal_exit_th_total_profit", 0.0) or 0.0)
        sth_c = float(combo.get("signal_exit_th_total_profit", 0.0) or 0.0)
        d_tr = tr_c - tr_b
        d_sth = sth_c - sth_b

        ut = int(base.get("unique_round_trips") or 0)

        ag_b = aggregate_exit_events(base, df_reg_w)
        ag_c = aggregate_exit_events(combo, df_reg_w)

        if d_tr > 1e-12:
            bucket = "개선"
        elif d_tr < -1e-12:
            bucket = "악화"
        else:
            bucket = "동일"

        rows.append(
            {
                "bucket": bucket,
                "t_end": str(t_end),
                "unique_round_trips": ut,
                "delta_total_return": d_tr,
                "delta_signal_exit_th_profit": d_sth,
                "baseline_total_return": tr_b,
                "combo_total_return": tr_c,
                "long_share_combo": ag_c["long_share"],
                "short_share_combo": ag_c["short_share"],
                "trend_dist_combo": ag_c["trend_dist_str"],
                "vol_dist_combo": ag_c["vol_dist_str"],
                "mean_mae_combo": ag_c["mean_mae"],
                "mean_mfe_combo": ag_c["mean_mfe"],
                "n_exits_combo": ag_c["n_exit_events"],
                "long_share_base": ag_b["long_share"],
                "short_share_base": ag_b["short_share"],
                "trend_dist_base": ag_b["trend_dist_str"],
                "vol_dist_base": ag_b["vol_dist_str"],
                "mean_mae_base": ag_b["mean_mae"],
                "mean_mfe_base": ag_b["mean_mfe"],
                "_trend_c_combo": ag_c["trend_counter"],
                "_vol_c_combo": ag_c["vol_counter"],
            }
        )

    improved = [r for r in rows if r["bucket"] == "개선"]
    worsened = [r for r in rows if r["bucket"] == "악화"]
    flat = [r for r in rows if r["bucket"] == "동일"]

    def _pool(rs: list[dict], key: str) -> Counter:
        out = Counter()
        for r in rs:
            c = r.get(key)
            if isinstance(c, Counter):
                out.update(c)
        return out

    trend_imp = _pool(improved, "_trend_c_combo")
    trend_worse = _pool(worsened, "_trend_c_combo")
    vol_imp = _pool(improved, "_vol_c_combo")
    vol_worse = _pool(worsened, "_vol_c_combo")

    def _mean_key(rs: list[dict], k: str) -> float | None:
        vals = [r[k] for r in rs if r.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    # Markdown
    out = Path(args.out_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# roll60 창 분해 — δp=0.05, mae_cut=-0.007 (단일 후보)")
    lines.append("")
    lines.append(f"- 앵커: `{t_anchor.isoformat()}` · threshold={thr} · roll60 step={args.rolling_step_days}d")
    lines.append(f"- 창 수: **{len(rows)}** (개선 **{len(improved)}** / 악화 **{len(worsened)}** / 동일 **{len(flat)}**)")
    lines.append("")

    lines.append("## STEP 1 — 창 분류")
    lines.append("")
    lines.append("| t_end | 분류 | Δ total_return | Δ signal_exit_th profit |")
    lines.append("| --- | --- | ---: | ---: |")
    for r in sorted(rows, key=lambda x: str(x["t_end"])):
        lines.append(
            f"| {r['t_end'][:10]} | {r['bucket']} | {r['delta_total_return']:.6f} | {r['delta_signal_exit_th_profit']:.6f} |"
        )
    lines.append("")

    def table_group(title: str, rs: list[dict]) -> None:
        lines.append(f"### {title} ({len(rs)}창)")
        lines.append("")
        if not rs:
            lines.append("(해당 없음)")
            lines.append("")
            return
        lines.append(
            "| 창 종료 | 왕복수 | Δ total_return | Δ sth profit | SHORT% | LONG% | trend(콤보) | vol(콤보) | mean MAE | mean MFE |"
        )
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |")

        def _pct(x: float | None) -> str:
            return f"{100 * x:.1f}%" if x is not None else ""

        for r in sorted(rs, key=lambda x: str(x["t_end"])):
            ss = r["short_share_combo"]
            ls = r["long_share_combo"]
            mae = r["mean_mae_combo"]
            mfe = r["mean_mfe_combo"]
            mae_s = f"{mae:.6f}" if mae is not None else ""
            mfe_s = f"{mfe:.6f}" if mfe is not None else ""
            lines.append(
                f"| {r['t_end'][:10]} | {r['unique_round_trips']} | {r['delta_total_return']:.6f} | "
                f"{r['delta_signal_exit_th_profit']:.6f} | "
                f"{_pct(ss)} | {_pct(ls)} | "
                f"{r['trend_dist_combo']} | {r['vol_dist_combo']} | "
                f"{mae_s} | {mfe_s} |"
            )
        lines.append("")
        lines.append("**그룹 평균 (가능한 항목)**")
        lines.append("")
        def _mf(x: float | None, nd: int = 6) -> str:
            return f"{x:.{nd}f}" if x is not None else "—"

        lines.append(
            f"- mean(Δ total_return): **{_mf(_mean_key(rs, 'delta_total_return'))}** ; "
            f"mean(Δ sth profit): **{_mf(_mean_key(rs, 'delta_signal_exit_th_profit'))}**"
        )
        lines.append(
            f"- mean(SHORT% 콤보): **{_mf(_mean_key(rs, 'short_share_combo'), 4)}** ; "
            f"mean(LONG% 콤보): **{_mf(_mean_key(rs, 'long_share_combo'), 4)}**"
        )
        lines.append(
            f"- mean(mean MAE 콤보): **{_mf(_mean_key(rs, 'mean_mae_combo'))}** ; "
            f"mean(mean MFE 콤보): **{_mf(_mean_key(rs, 'mean_mfe_combo'))}**"
        )
        lines.append("")

    lines.append("## STEP 2 — 개선 vs 악화 창 비교 표")
    lines.append("")
    table_group("개선 창", improved)
    table_group("악화 창", worsened)
    if flat:
        table_group("동일 창", flat)

    lines.append("## STEP 2b — 풀링 (개선 vs 악화, 진입 레짐 합산)")
    lines.append("")
    lines.append("**trend (콤보 청산 기준)**")
    lines.append("")
    lines.append(f"- 개선 창 합: {_fmt_pct_dist(trend_imp, ['uptrend', 'downtrend'])}")
    lines.append(f"- 악화 창 합: {_fmt_pct_dist(trend_worse, ['uptrend', 'downtrend'])}")
    lines.append("")
    lines.append("**vol (콤보)**")
    lines.append("")
    lines.append(f"- 개선 창 합: {_fmt_pct_dist(vol_imp, ['low_vol', 'mid_vol', 'high_vol'])}")
    lines.append(f"- 악화 창 합: {_fmt_pct_dist(vol_worse, ['low_vol', 'mid_vol', 'high_vol'])}")
    lines.append("")

    lines.append("## STEP 3 — 질문 답변 (본 run 기준)")
    lines.append("")
    lines.append(
        "1. **특정 레짐에서만 먹히는가?** "
        f"풀링 trend: 개선 { _fmt_pct_dist(trend_imp, ['uptrend', 'downtrend']) } vs "
        f"악화 { _fmt_pct_dist(trend_worse, ['uptrend', 'downtrend']) }. "
        "차이가 크지 않으면 **단일 trend 필터로만 켜기/끄기는 근거가 약함**."
    )
    lines.append("")
    lines.append(
        "2. **SHORT 중심으로만 먹히는가?** "
        f"개선 창 평균 SHORT 비중 **{_mean_key(improved, 'short_share_combo')}**, "
        f"악화 창 **{_mean_key(worsened, 'short_share_combo')}** — "
        "차이가 작으면 ‘SHORT 전용’ 규칙은 **아직 데이터로 확정 불가**."
    )
    lines.append("")
    lines.append(
        "3. **변동성 구간에 따라 성과 차이가 큰가?** "
        f"풀링 vol: 개선 { _fmt_pct_dist(vol_imp, ['low_vol', 'mid_vol', 'high_vol']) } vs "
        f"악화 { _fmt_pct_dist(vol_worse, ['low_vol', 'mid_vol', 'high_vol']) }. "
        f"평균 MAE(콤보): 개선 **{_mean_key(improved, 'mean_mae_combo')}**, 악화 **{_mean_key(worsened, 'mean_mae_combo')}** ; "
        f"평균 MFE: 개선 **{_mean_key(improved, 'mean_mfe_combo')}**, 악화 **{_mean_key(worsened, 'mean_mfe_combo')}**."
    )
    lines.append("")
    lines.append(
        "4. **roll60 악화 창 공통 패턴?** "
        f"악화 {len(worsened)}창: 평균 Δtr **{_mean_key(worsened, 'delta_total_return')}**, "
        f"평균 왕복 **{_mean_key(worsened, 'unique_round_trips')}**. "
        "거래 수가 매우 적은 창이 섞이면 **표본 편차** 가능. "
        "선행 WF에서 roll90 승률이 더 나은 편이므로 **60d만의 잡음** 가능성 유지."
    )
    lines.append("")

    lines.append("## STEP 4 — 결론")
    lines.append("")
    lines.append(
        "- **조건부 운영 규칙 가능성:** "
        "레짐/방향 단일 조건으로 ‘켜기/끄기’를 정하기엔 이번 분해만으로는 **근거가 부족**. "
        "다만 **roll90·180d에서는 개선이 더 잘 맞는 편**이므로, **짧은 구간(60d) 롤링만으로 on/off 판단하지 말 것**."
    )
    lines.append(
        "- **전역 적용:** roll60 승률이 0.5 미만이었던 선행 WF 결과와 합치면 **아직 전역 플래그 ON은 이르다** — "
        "**추가 검증 필요** 유지, 필요 시 **롤링 90d 또는 최소 왕복 수 조건**을 만족할 때만 활성화하는 식의 **게이트**는 후속 과제."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 부록 A — 외부(GPT 등) 전달 시 같이 넣으면 좋은 맥락")
    lines.append("")
    lines.append("이 MD만 보내도 되지만, 오해를 줄이려면 아래를 **2~4줄이라도** 붙이는 것을 권장한다.")
    lines.append("")
    lines.append("| 항목 | 내용 |")
    lines.append("| --- | --- |")
    lines.append(
        "| 신호·데이터 | FR2 C4 게이트 + baseline/FR2 **override** 앙상블 proba, "
        "`min_max_proba`(threshold 역할)=**0.6**, 그 외 백테스트 설정은 그리드·WF 스크립트와 동일 (early_exit / time_stop 등) |"
    )
    lines.append(
        "| 선행 판정 | `SIGNAL_EXIT_TH_LOSS_AUX_WF_VALIDATION_REPORT.md` 기준 **「추가 검증 필요」** 유지 (roll60 승률 0.5 미만 등) |"
    )
    lines.append("| 본 문서 후보 | **δp=0.05, mae_cut=-0.007** 단일 조합만 분해 |")
    lines.append(
        "| 재현 | `python scripts/signal_exit_th_loss_aux_roll60_decomposition.py --days-full 420` (프로젝트 루트, `.venv` 권장) |"
    )
    lines.append("| 산출물 | 동명 `.csv`에 창별 행 저장 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 부록 B — 추가 해석·평가 (로컬 자료 기준, 비공식)")
    lines.append("")
    lines.append("> 주의: 아래는 표·수치에 대한 **추가 의견**이며, STEP 1~4의 자동 집계와 별개다.")
    lines.append("")
    lines.append(
        "1. **roll60 해석의 한계** — 일부 창은 왕복 수가 매우 적다. Δ는 노이즈에 가깝고 악화 창 공통 패턴을 **과대해석하지 않는 것**이 타당."
    )
    lines.append(
        "2. **레짐/SHORT 풀링** — trend가 개선·악화 동일하면 “특정 레짐에서만 켠다”는 내러티브가 **이 데이터만으로는 약함**."
    )
    lines.append(
        "3. **SHORT 비중** — 악화 창의 SHORT 비중이 개선보다 높게 나올 수 있음. 본 지표는 **창 내 전체 청산** 기준이며 **sth 손실만**이 아니므로 **직접 대조 불가**."
    )
    lines.append(
        "4. **180d / roll90** — 긴 창에서는 상대적으로 잘 맞고 roll60은 흔들림 → **짧은 표본·짧은 창에 민감**할 가능성. 전역 ON보다 **긴 창 평가·최소 왕복 게이트**가 더 잘 맞는 그림."
    )
    lines.append(
        "5. **다음 단계** — 창 요약만으로는 “악화가 항상 sth에서 왔다”까지 단정 어려움. **exit_reason·sth 건별 비교**는 본 MD 범위 밖 후속 과제."
    )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[decomp] wrote {out}", flush=True)

    import csv

    csv_path = out.with_suffix(".csv")
    if rows:
        keys = [k for k in rows[0].keys() if not k.startswith("_")]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in keys})
        print(f"[decomp] wrote {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
