#!/usr/bin/env python3
"""
SIGNAL_EXIT_TH deployment decision automation

입력:
  data/diagnostics/fr2/SIGNAL_EXIT_TH_CONDITIONAL_DEPLOYMENT_REPORT.csv

출력:
  data/diagnostics/fr2/SIGNAL_EXIT_TH_DEPLOYMENT_DECISION_REPORT.md
  data/diagnostics/fr2/SIGNAL_EXIT_TH_DEPLOYMENT_DECISION_REPORT.csv
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ===== 판정 기준 상수 =====
LOOKBACK_WINDOWS = (14, 28)

UPGRADE_PD_COUNT_MAX = 2.0
UPGRADE_ON_RATIO_MIN = 0.2
UPGRADE_ON_RATIO_MAX = 0.8
UPGRADE_RET_STD_WORSE_MAX = 0.002

HOLD_PD_COUNT_MAX = 8.0
HOLD_ON_RATIO_MIN = 0.05
HOLD_ON_RATIO_MAX = 0.95
HOLD_RET_STD_WORSE_MAX = 0.010


@dataclass
class Decision:
    decision: str
    note: str


def _pick_col(df: pd.DataFrame, choices: list[str], required: bool = True) -> str | None:
    for c in choices:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column. expected one of: {choices}")
    return None


def _safe_to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True, errors="coerce")


def decide_candidate_status(stats: dict[str, Any]) -> Decision:
    mean_dret = float(stats.get("mean_delta_total_return", np.nan))
    mean_dsth = float(stats.get("mean_delta_sth_total_profit", np.nan))
    pd_count = float(stats.get("total_profit_damaged_count", np.nan))
    on_ratio = float(stats.get("on_ratio_mean", np.nan))
    ret_std_delta = float(stats.get("return_std_delta", np.nan))

    reasons: list[str] = []
    if np.isnan(mean_dret) or np.isnan(mean_dsth):
        return Decision("유지", "delta 계산 불가(필수 값 누락)로 조건부 유지")

    if mean_dret < 0:
        reasons.append("mean Δtotal_return 음수")
    if mean_dsth < 0:
        reasons.append("mean Δsth 음수")
    if not np.isnan(pd_count) and pd_count > HOLD_PD_COUNT_MAX:
        reasons.append("profit_damaged count 과다")
    if not np.isnan(ret_std_delta) and ret_std_delta > HOLD_RET_STD_WORSE_MAX:
        reasons.append("return std 악화")
    if not np.isnan(on_ratio) and (on_ratio < HOLD_ON_RATIO_MIN or on_ratio > HOLD_ON_RATIO_MAX):
        reasons.append("ON ratio 비정상 범위")
    if reasons:
        return Decision("탈락", "; ".join(reasons))

    upgrade_ok = (
        mean_dret >= 0
        and mean_dsth >= 0
        and (np.isnan(pd_count) or pd_count <= UPGRADE_PD_COUNT_MAX)
        and (np.isnan(on_ratio) or (UPGRADE_ON_RATIO_MIN <= on_ratio <= UPGRADE_ON_RATIO_MAX))
        and (np.isnan(ret_std_delta) or ret_std_delta <= UPGRADE_RET_STD_WORSE_MAX)
    )
    if upgrade_ok:
        return Decision("승급", "핵심 5축 기준 충족(수익·sth·손상·ON ratio·std)")

    return Decision("유지", "일부 개선이나 기준 일부 미충족(조건부 유지)")


def _calc_stats_for_candidate(
    detail_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    candidate: str,
) -> dict[str, float]:
    cand = detail_df[detail_df["strategy_norm"] == candidate].copy()
    base = baseline_df.copy()
    if cand.empty or base.empty:
        return {
            "mean_delta_total_return": np.nan,
            "mean_delta_sth_total_profit": np.nan,
            "total_profit_damaged_count": np.nan,
            "total_profit_damaged_damage": np.nan,
            "on_ratio_mean": np.nan,
            "return_std_delta": np.nan,
            "n_rows": 0.0,
        }

    key_cols = ["segment", "t_end_ts"]
    merged = cand.merge(
        base[key_cols + ["total_return", "signal_exit_th_total_profit"]],
        on=key_cols,
        how="left",
        suffixes=("", "_base"),
    )
    merged["delta_total_return"] = merged["total_return"] - merged["total_return_base"]
    merged["delta_sth_total_profit"] = (
        merged["signal_exit_th_total_profit"] - merged["signal_exit_th_total_profit_base"]
    )

    cand_ret_std = float(merged["total_return"].std(ddof=1)) if len(merged) > 1 else 0.0
    base_ret_std = float(merged["total_return_base"].std(ddof=1)) if len(merged) > 1 else 0.0
    return {
        "mean_delta_total_return": float(merged["delta_total_return"].mean()),
        "mean_delta_sth_total_profit": float(merged["delta_sth_total_profit"].mean()),
        "total_profit_damaged_count": float(merged["profit_damaged_count"].sum()),
        "total_profit_damaged_damage": float(merged["profit_damaged_total_damage"].sum()),
        "on_ratio_mean": float(merged["on"].mean()),
        "return_std_delta": cand_ret_std - base_ret_std,
        "n_rows": float(len(merged)),
    }


def _norm_strategy(v: str) -> str:
    x = str(v).strip().lower()
    if x in {"rt-a", "rt_a", "rta"}:
        return "rt_a"
    if x in {"vol-a", "vol_a", "vola"}:
        return "vol_a"
    if x in {"baseline", "base"}:
        return "baseline"
    return x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_CONDITIONAL_DEPLOYMENT_REPORT.csv",
    )
    ap.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_DEPLOYMENT_DECISION_REPORT.md",
    )
    ap.add_argument(
        "--out-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_DEPLOYMENT_DECISION_REPORT.csv",
    )
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    df = pd.read_csv(input_csv)
    fallback_notes: list[str] = []

    # robust column mapping
    row_type_col = _pick_col(df, ["row_type"], required=False)
    strategy_col = _pick_col(df, ["strategy", "candidate", "gate", "mode"], required=True)
    t_end_col = _pick_col(df, ["t_end", "date", "timestamp", "window_end"], required=True)
    seg_col = _pick_col(df, ["segment", "window", "window_name"], required=False)
    on_col = _pick_col(df, ["on", "gate_on", "activation_on"], required=False)
    tr_col = _pick_col(df, ["total_return"], required=True)
    sth_col = _pick_col(df, ["signal_exit_th_total_profit", "sth_total_profit"], required=True)
    pd_count_col = _pick_col(df, ["profit_damaged_count"], required=False)
    pd_dmg_col = _pick_col(df, ["profit_damaged_total_damage", "profit_damaged_damage"], required=False)

    if row_type_col is None:
        fallback_notes.append("`row_type` 컬럼 없음: 전체를 detail로 간주")
        detail_df = df.copy()
        trend_df = pd.DataFrame()
    else:
        detail_df = df[df[row_type_col] == "detail"].copy()
        trend_df = df[df[row_type_col] == "on_ratio_trend"].copy()

    if detail_df.empty:
        fallback_notes.append("detail row 없음: 판정 불가")

    # normalize shape
    detail_df["strategy_norm"] = detail_df[strategy_col].map(_norm_strategy)
    detail_df["t_end_ts"] = _safe_to_datetime(detail_df[t_end_col])
    if seg_col is None:
        detail_df["segment"] = "unknown"
        fallback_notes.append("segment 컬럼 없음: segment=unknown 사용")
    else:
        detail_df["segment"] = detail_df[seg_col].astype(str)

    if on_col is None:
        detail_df["on"] = np.nan
        fallback_notes.append("on 컬럼 없음: on_ratio 계산은 NaN")
    else:
        detail_df["on"] = pd.to_numeric(detail_df[on_col], errors="coerce")

    detail_df["total_return"] = pd.to_numeric(detail_df[tr_col], errors="coerce")
    detail_df["signal_exit_th_total_profit"] = pd.to_numeric(detail_df[sth_col], errors="coerce")
    if pd_count_col is None:
        detail_df["profit_damaged_count"] = np.nan
        fallback_notes.append("profit_damaged_count 컬럼 없음")
    else:
        detail_df["profit_damaged_count"] = pd.to_numeric(detail_df[pd_count_col], errors="coerce")
    if pd_dmg_col is None:
        detail_df["profit_damaged_total_damage"] = np.nan
        fallback_notes.append("profit_damaged_total_damage 컬럼 없음")
    else:
        detail_df["profit_damaged_total_damage"] = pd.to_numeric(detail_df[pd_dmg_col], errors="coerce")

    detail_df = detail_df[detail_df["t_end_ts"].notna()].copy()
    if detail_df.empty:
        fallback_notes.append("유효한 날짜 행이 없음")

    anchor_ts = detail_df["t_end_ts"].max() if not detail_df.empty else pd.Timestamp.now("UTC")
    candidates = ("rt_a", "vol_a")
    baseline_all = detail_df[detail_df["strategy_norm"] == "baseline"].copy()

    rows: list[dict[str, Any]] = []
    for lb in LOOKBACK_WINDOWS:
        start_ts = anchor_ts - pd.Timedelta(days=int(lb))
        d = detail_df[detail_df["t_end_ts"] >= start_ts].copy()
        b = baseline_all[baseline_all["t_end_ts"] >= start_ts].copy()
        for cand in candidates:
            stats = _calc_stats_for_candidate(d, b, cand)
            dec = decide_candidate_status(stats)
            rows.append(
                {
                    "candidate": "RT-A" if cand == "rt_a" else "VOL-A",
                    "lookback_days": int(lb),
                    "mean_delta_total_return": stats["mean_delta_total_return"],
                    "mean_delta_sth_total_profit": stats["mean_delta_sth_total_profit"],
                    "total_profit_damaged_count": stats["total_profit_damaged_count"],
                    "total_profit_damaged_damage": stats["total_profit_damaged_damage"],
                    "on_ratio_mean": stats["on_ratio_mean"],
                    "return_std_delta": stats["return_std_delta"],
                    "decision": dec.decision,
                    "note": dec.note,
                    "n_rows": int(stats["n_rows"]),
                }
            )

    out_df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    # MD output
    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH Deployment Decision Report")
    lines.append("")
    lines.append("## A. 메타")
    lines.append("")
    lines.append(f"- 생성시각(UTC): `{pd.Timestamp.now('UTC').isoformat()}`")
    lines.append(f"- 입력 파일: `{input_csv.as_posix()}`")
    lines.append(f"- anchor 시각: `{anchor_ts.isoformat()}`")
    lines.append(f"- 분석 기간: `{LOOKBACK_WINDOWS[0]}d`, `{LOOKBACK_WINDOWS[1]}d`")
    lines.append("- 기준 후보: RT-A(Primary), VOL-A(Shadow)")
    if fallback_notes:
        lines.append("- fallback 처리:")
        for n in fallback_notes:
            lines.append(f"  - {n}")
    lines.append("")

    def _sec_for_candidate(title: str, cand_label: str) -> list[str]:
        sub = out_df[out_df["candidate"] == cand_label].copy()
        out: list[str] = []
        out.append(f"## {title}")
        out.append("")
        out.append("| lookback | mean ΔTR | mean Δsth | pd_count(합) | pd_damage(합) | on_ratio | return_std_delta | 판정 | note |")
        out.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
        for _, r in sub.sort_values("lookback_days").iterrows():
            out.append(
                f"| {int(r['lookback_days'])}d | {r['mean_delta_total_return']:.6f} | {r['mean_delta_sth_total_profit']:.6f} | "
                f"{r['total_profit_damaged_count']:.3f} | {r['total_profit_damaged_damage']:.6f} | "
                f"{r['on_ratio_mean']:.4f} | {r['return_std_delta']:.6f} | **{r['decision']}** | {r['note']} |"
            )
        out.append("")
        latest = sub[sub["lookback_days"] == max(LOOKBACK_WINDOWS)]
        if not latest.empty:
            rr = latest.iloc[0]
            out.append("- 사유 요약:")
            out.append(f"  - mean ΔTR={rr['mean_delta_total_return']:.6f}, mean Δsth={rr['mean_delta_sth_total_profit']:.6f}")
            out.append(
                f"  - pd_count={rr['total_profit_damaged_count']:.3f}, on_ratio={rr['on_ratio_mean']:.4f}, std_delta={rr['return_std_delta']:.6f}"
            )
        out.append("")
        return out

    lines.extend(_sec_for_candidate("B. RT-A 운영 판정", "RT-A"))
    lines.extend(_sec_for_candidate("C. VOL-A shadow 판정", "VOL-A"))

    # D. direct comparison (28d)
    lines.append("## D. 후보 간 비교")
    lines.append("")
    d28 = out_df[out_df["lookback_days"] == 28].set_index("candidate")
    if {"RT-A", "VOL-A"} <= set(d28.index):
        rt = d28.loc["RT-A"]
        vl = d28.loc["VOL-A"]
        better_delta = "RT-A" if rt["mean_delta_total_return"] >= vl["mean_delta_total_return"] else "VOL-A"
        better_stability = "RT-A" if rt["return_std_delta"] <= vl["return_std_delta"] else "VOL-A"
        lines.append(f"- 개선폭(ΔTR) 우위: **{better_delta}**")
        lines.append(f"- 안정성(std_delta) 우위: **{better_stability}**")
        lines.append(
            f"- 28d 기준 RT-A vs VOL-A ΔTR 차이: {(rt['mean_delta_total_return'] - vl['mean_delta_total_return']):.6f}, "
            f"Δsth 차이: {(rt['mean_delta_sth_total_profit'] - vl['mean_delta_sth_total_profit']):.6f}"
        )
    else:
        lines.append("- 28d 비교 불가(후보 데이터 부족)")
    lines.append("")

    # E. recommendation
    lines.append("## E. 운영 권고")
    lines.append("")
    rec = "둘 다 유지 관찰"
    if {"RT-A", "VOL-A"} <= set(d28.index):
        rt_dec = str(d28.loc["RT-A", "decision"])
        vl_dec = str(d28.loc["VOL-A", "decision"])
        if rt_dec == "승급" and vl_dec != "승급":
            rec = "RT-A 유지"
        elif rt_dec != "승급" and vl_dec == "승급":
            rec = "VOL-A 교체 검토"
        elif rt_dec == "승급" and vl_dec == "승급":
            rec = "둘 다 유지 관찰"
        elif rt_dec == "탈락" and vl_dec == "탈락":
            rec = "둘 다 보류"
    lines.append(f"- 권고: **{rec}**")
    lines.append("- 본 스크립트는 과거 백테스트 결과(히스토리컬)에도 동일하게 적용되며, 미래 운영 로그에도 재사용 가능.")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[deploy-decision] wrote {out_md} and {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
