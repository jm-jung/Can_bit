#!/usr/bin/env python3
"""
loss_aux(B: delta_unreal_4bars>0 보호 포함) activation logic 탐색.

목표:
- roll60 / roll90 창별 개선/악화 패턴 분석
- 단순 activation 조건 1~2개 제안
- 조건 ON/OFF 성능 비교
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Callable

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


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if np.isnan(x) or np.isinf(x):
            return "—"
        return f"{x:.6f}"
    return str(x)


def _window_mask(df_bt: pd.DataFrame, t_end: pd.Timestamp, window_days: int) -> np.ndarray:
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    w_start = t_end - pd.Timedelta(days=window_days)
    return ((t >= w_start) & (t <= t_end)).to_numpy()


def _quantile_bucket(values: pd.Series) -> pd.Series:
    q1 = float(values.quantile(1 / 3))
    q2 = float(values.quantile(2 / 3))

    def f(v: float) -> str:
        if v <= q1:
            return "low"
        if v <= q2:
            return "mid"
        return "high"

    return values.map(f)


def _group_compare(df: pd.DataFrame, col: str, mask_a: pd.Series, mask_b: pd.Series) -> tuple[float, float]:
    a = df.loc[mask_a, col]
    b = df.loc[mask_b, col]
    return (float(a.mean()) if len(a) else float("nan"), float(b.mean()) if len(b) else float("nan"))


def _simulate_activation(
    df: pd.DataFrame,
    cond: Callable[[pd.Series], bool],
) -> dict[str, Any]:
    sim = df.copy()
    sim["on"] = sim.apply(cond, axis=1).astype(int)
    sim["ret_sel"] = np.where(sim["on"] == 1, sim["ret_b"], sim["ret_base"])
    sim["sth_sel"] = np.where(sim["on"] == 1, sim["sth_b"], sim["sth_base"])
    sim["pd_sel"] = np.where(sim["on"] == 1, sim["profit_damaged_count_b"], 0.0)
    sim["pd_damage_sel"] = np.where(sim["on"] == 1, sim["profit_damaged_damage_b"], 0.0)
    sim["applied_sel"] = np.where(sim["on"] == 1, sim["aux_applied_b"], 0.0)
    sim["dret_vs_base"] = sim["ret_sel"] - sim["ret_base"]
    sim["dsth_vs_base"] = sim["sth_sel"] - sim["sth_base"]

    out: dict[str, Any] = {"all": sim}
    for seg in ("rolling_90d", "rolling_60d"):
        s = sim[sim["segment"] == seg]
        if s.empty:
            out[seg] = {}
            continue
        out[seg] = {
            "n_windows": int(len(s)),
            "on_ratio": float(s["on"].mean()),
            "mean_ret_sel": float(s["ret_sel"].mean()),
            "mean_dret_vs_base": float(s["dret_vs_base"].mean()),
            "mean_sth_sel": float(s["sth_sel"].mean()),
            "mean_dsth_vs_base": float(s["dsth_vs_base"].mean()),
            "mean_pd_count": float(s["pd_sel"].mean()),
            "mean_pd_damage": float(s["pd_damage_sel"].mean()),
            "mean_applied": float(s["applied_sel"].mean()),
            "ret_std_sel": float(s["ret_sel"].std(ddof=1)) if len(s) > 1 else 0.0,
            "ret_std_base": float(s["ret_base"].std(ddof=1)) if len(s) > 1 else 0.0,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=420)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rolling-step-days", type=int, default=30)
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_LOGIC_REPORT.md",
    )
    parser.add_argument(
        "--out-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_LOGIC_REPORT.csv",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)

    print(f"[act-logic] load ensemble days_full={args.days_full} ...", flush=True)
    df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
    print(f"[act-logic] ts range {ts_min} ~ {ts_max}", flush=True)

    rows: list[dict[str, Any]] = []
    for span in (90, 60):
        ends = _rolling_end_dates(ts_min, ts_max, t_anchor, span, int(args.rolling_step_days))
        print(f"[act-logic] rolling {span}d windows={len(ends)}", flush=True)
        for te in ends:
            te = pd.Timestamp(te)
            tag = f"roll{span}d end={te.date()}"
            print(f"[act-logic] {tag} ...", flush=True)

            base = run_loss_aux_window(
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_end=te,
                window_days=span,
                threshold=thr,
                emit_trade_log=True,
                signal_exit_th_loss_aux_delta_p=None,
                signal_exit_th_loss_aux_mae_cut=None,
            )
            b = run_loss_aux_window(
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_end=te,
                window_days=span,
                threshold=thr,
                emit_trade_log=True,
                signal_exit_th_loss_aux_delta_p=DELTA_P,
                signal_exit_th_loss_aux_mae_cut=MAE_CUT,
                signal_exit_th_loss_aux_gate_delta_unreal_positive=True,
                signal_exit_th_loss_aux_gate_mfe_min=None,
            )

            mask = _window_mask(df_bt, te, span)
            df_w = df_bt.loc[mask].reset_index(drop=True)
            close = df_w["close"].astype(float)
            ret = close.pct_change().dropna()
            vol = float(ret.std()) if len(ret) else 0.0

            ex = _sth_exits(base)
            if ex:
                mae = [float(e.get("max_adverse_excursion")) for e in ex if e.get("max_adverse_excursion") is not None]
                mfe = [float(e.get("max_favorable_excursion")) for e in ex if e.get("max_favorable_excursion") is not None]
                short_share = float(np.mean([1.0 if str(e.get("direction")) == "SHORT" else 0.0 for e in ex]))
                mean_mae = float(np.mean(mae)) if mae else float("nan")
                mean_mfe = float(np.mean(mfe)) if mfe else float("nan")
            else:
                mean_mae = float("nan")
                mean_mfe = float("nan")
                short_share = float("nan")

            eb = { (str(e.get("entry_ts")), str(e.get("exit_ts")), str(e.get("direction"))): e for e in _sth_exits(base)}
            ec = { (str(e.get("entry_ts")), str(e.get("exit_ts")), str(e.get("direction"))): e for e in _sth_exits(b)}
            common = set(eb.keys()) & set(ec.keys())
            pd_count = 0
            pd_damage = 0.0
            for k in common:
                pb = float(eb[k].get("pnl_change", eb[k].get("profit", 0.0)) or 0.0)
                pc = float(ec[k].get("pnl_change", ec[k].get("profit", 0.0)) or 0.0)
                if pb > 0 and pc < pb:
                    pd_count += 1
                    pd_damage += pc - pb

            ret_base = float(base.get("total_return", 0.0) or 0.0)
            ret_b = float(b.get("total_return", 0.0) or 0.0)
            sth_base = float(base.get("signal_exit_th_total_profit", 0.0) or 0.0)
            sth_b = float(b.get("signal_exit_th_total_profit", 0.0) or 0.0)
            dtr = ret_b - ret_base
            dsth = sth_b - sth_base
            if dtr > 1e-12 and dsth > 1e-12:
                cls = "improved"
            elif dtr < -1e-12 and dsth < -1e-12:
                cls = "worsened"
            else:
                cls = "mixed"

            rows.append(
                {
                    "segment": f"rolling_{span}d",
                    "window_days": span,
                    "t_end": str(te),
                    "ret_base": ret_base,
                    "ret_b": ret_b,
                    "dret": dtr,
                    "sth_base": sth_base,
                    "sth_b": sth_b,
                    "dsth": dsth,
                    "unique_round_trips": int(base.get("unique_round_trips") or 0),
                    "mean_mae": mean_mae,
                    "mean_mfe": mean_mfe,
                    "short_share": short_share,
                    "volatility_std": vol,
                    "profit_damaged_count_b": int(pd_count),
                    "profit_damaged_damage_b": float(pd_damage),
                    "aux_applied_b": int(b.get("signal_exit_th_loss_aux_applied", 0) or 0),
                    "class": cls,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No rows produced.")

    df["vol_bucket"] = _quantile_bucket(df["volatility_std"])

    improved = df["class"] == "improved"
    worsened = df["class"] == "worsened"

    # activation 후보 1~2개 (단순)
    x_trips = int(round(float(df.loc[improved, "unique_round_trips"].median()))) if improved.any() else 100
    # 후보1: 거래수 게이트
    cond1 = lambda r: int(r["unique_round_trips"]) >= x_trips
    cond1_name = f"cond1: unique_round_trips >= {x_trips}"
    # 후보2: 거래수 + 변동성(mid/high)
    cond2 = lambda r: (int(r["unique_round_trips"]) >= x_trips) and (str(r["vol_bucket"]) in ("mid", "high"))
    cond2_name = f"cond2: unique_round_trips >= {x_trips} AND vol_bucket in {{mid,high}}"

    sim1 = _simulate_activation(df, cond1)
    sim2 = _simulate_activation(df, cond2)

    # write csv
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    # markdown report
    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH activation logic 탐색")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| t_anchor | `{t_anchor.isoformat()}` |")
    lines.append(f"| threshold | {thr} |")
    lines.append(f"| loss_aux | δp={DELTA_P}, mae_cut={MAE_CUT}, 보호게이트(B) 포함 |")
    lines.append("")

    lines.append("## 창 특성 요약 (개선 vs 악화)")
    lines.append("")
    lines.append("| 항목 | 개선 창 mean | 악화 창 mean |")
    lines.append("| --- | ---: | ---: |")
    for col in ("unique_round_trips", "mean_mae", "mean_mfe", "short_share", "volatility_std"):
        a, b = _group_compare(df, col, improved, worsened)
        lines.append(f"| {col} | {_fmt(a)} | {_fmt(b)} |")
    lines.append("")
    vb_cmp = (
        df[df["class"] == "improved"]["vol_bucket"].value_counts(normalize=True).to_dict(),
        df[df["class"] == "worsened"]["vol_bucket"].value_counts(normalize=True).to_dict(),
    )
    lines.append(
        f"- 개선 창 volatility bucket 비중: low={vb_cmp[0].get('low', 0):.2f}, mid={vb_cmp[0].get('mid', 0):.2f}, high={vb_cmp[0].get('high', 0):.2f}"
    )
    lines.append(
        f"- 악화 창 volatility bucket 비중: low={vb_cmp[1].get('low', 0):.2f}, mid={vb_cmp[1].get('mid', 0):.2f}, high={vb_cmp[1].get('high', 0):.2f}"
    )
    lines.append("")

    lines.append("## A. activation 조건별 결과")
    lines.append("")
    lines.append("| 조건 | segment | ON ratio | mean Δtotal_return vs base | mean Δsth vs base | mean pd_count | mean pd_damage | mean applied | ret std (sel/base) |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for cname, sim in ((cond1_name, sim1), (cond2_name, sim2)):
        for seg in ("rolling_90d", "rolling_60d"):
            s = sim.get(seg) or {}
            if not s:
                continue
            lines.append(
                f"| {cname} | {seg} | {_fmt(s['on_ratio'])} | {_fmt(s['mean_dret_vs_base'])} | {_fmt(s['mean_dsth_vs_base'])} | "
                f"{_fmt(s['mean_pd_count'])} | {_fmt(s['mean_pd_damage'])} | {_fmt(s['mean_applied'])} | "
                f"{_fmt(s['ret_std_sel'])} / {_fmt(s['ret_std_base'])} |"
            )
    lines.append("")

    lines.append("## B. baseline 대비 개선 여부")
    lines.append("")
    for cname, sim in ((cond1_name, sim1), (cond2_name, sim2)):
        s90 = sim.get("rolling_90d") or {}
        s60 = sim.get("rolling_60d") or {}
        ok90 = bool(s90) and s90["mean_dret_vs_base"] > 0 and s90["mean_dsth_vs_base"] > 0
        ok60 = bool(s60) and s60["mean_dret_vs_base"] > 0 and s60["mean_dsth_vs_base"] > 0
        lines.append(f"- **{cname}**: roll90 개선={ok90}, roll60 개선={ok60}")
    lines.append("")

    lines.append("## C. roll60 안정성 변화")
    lines.append("")
    for cname, sim in ((cond1_name, sim1), (cond2_name, sim2)):
        s60 = sim.get("rolling_60d") or {}
        if not s60:
            continue
        lines.append(
            f"- **{cname}**: roll60 ret std {s60['ret_std_sel']:.6f} (base {s60['ret_std_base']:.6f}), "
            f"mean pd_count {s60['mean_pd_count']:.3f}"
        )
    lines.append("")

    # choose recommendation
    # 우선순위: roll90 개선 유지 + roll60 pd_count 낮고 std 증가 억제
    def score(sim: dict[str, Any]) -> float:
        s90 = sim.get("rolling_90d") or {}
        s60 = sim.get("rolling_60d") or {}
        if not s90 or not s60:
            return -1e9
        return (
            3.0 * float(s90["mean_dret_vs_base"])
            + 2.5 * float(s90["mean_dsth_vs_base"])
            + 1.5 * float(s60["mean_dret_vs_base"])
            + 1.2 * float(s60["mean_dsth_vs_base"])
            - 0.2 * float(s60["mean_pd_count"])
            - 0.4 * max(0.0, float(s60["ret_std_sel"]) - float(s60["ret_std_base"]))
        )

    best_name, best_sim = max(((cond1_name, sim1), (cond2_name, sim2)), key=lambda x: score(x[1]))

    lines.append("## D. 최종 추천 activation 조건")
    lines.append("")
    lines.append(f"- **추천 조건:** `{best_name}`")
    lines.append("- 이유: roll90 개선 유지와 roll60 안정성(손상/분산) 균형 점수가 가장 높음.")
    lines.append("")

    verdict = "추가 검증 필요"
    s90 = best_sim.get("rolling_90d") or {}
    s60 = best_sim.get("rolling_60d") or {}
    if s90 and s60:
        cond_main = s90["mean_dret_vs_base"] > 0 and s90["mean_dsth_vs_base"] > 0
        cond_60 = s60["mean_dret_vs_base"] >= 0 and s60["mean_pd_count"] <= 8.0
        std_not_worse = s60["ret_std_sel"] <= s60["ret_std_base"] * 1.05
        if cond_main and cond_60 and std_not_worse:
            verdict = "조건부 운영 가능"
        if cond_main and cond_60 and std_not_worse and s60["mean_pd_count"] <= 5.0:
            verdict = "운영 가능"

    lines.append("## 최종 판정")
    lines.append("")
    lines.append(f"- **{verdict}**")
    lines.append("")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[act-logic] wrote {out_md} rows={len(df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

