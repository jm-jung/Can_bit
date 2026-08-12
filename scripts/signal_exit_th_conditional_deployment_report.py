#!/usr/bin/env python3
"""
SIGNAL_EXIT_TH activation v2
Conditional Deployment + Shadow Monitoring report

- Primary applied: RT-A (unique_round_trips >= 225)
- Shadow only: VOL-A (vol_bucket in {mid, high})
- Optional research flag: MFE-B (does not affect selection)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RT_A_THRESHOLD = 225

# fixed_180d baseline/B from existing validation outputs (unchanged setup)
FIXED180_BASELINE_TR = -0.07665713566110521
FIXED180_BASELINE_STH = -0.03632222787580999
FIXED180_B_TR = -0.07406819002651444
FIXED180_B_STH = -0.03352406333693331
FIXED180_PD_B = 0.0
FIXED180_PD_DMG_B = 0.0
FIXED180_AUX_B = 19.0
FIXED180_UT = 335


def _volatility_std_from_ohlc(ohlc_path: Path, t_end: pd.Timestamp, window_days: int) -> float:
    dfo = pd.read_csv(ohlc_path, usecols=["timestamp", "close"])
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"], utc=True)
    w0 = t_end - pd.Timedelta(days=window_days)
    m = (dfo["timestamp"] >= w0) & (dfo["timestamp"] <= t_end)
    close = dfo.loc[m, "close"].astype(float)
    ret = close.pct_change().dropna()
    return float(ret.std()) if len(ret) else 0.0


def _bucket(vol: float, q1: float, q2: float) -> str:
    if vol <= q1:
        return "low"
    if vol <= q2:
        return "mid"
    return "high"


def _compute_on_runs(seg_df: pd.DataFrame) -> tuple[float, float]:
    if seg_df.empty:
        return 0.0, 0.0
    arr = seg_df["on"].to_numpy(dtype=int)
    runs: list[int] = []
    cur = 0
    for x in arr:
        if x == 1:
            cur += 1
        elif cur > 0:
            runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    if not runs:
        return 0.0, 0.0
    return float(np.mean(runs)), float(np.max(runs))


def _strategy_rows(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    x = df.copy()
    if strategy == "baseline":
        x["on"] = 0
    elif strategy == "rt_a":
        x["on"] = (x["unique_round_trips"].astype(int) >= RT_A_THRESHOLD).astype(int)
    elif strategy == "vol_a":
        x["on"] = x["vol_bucket"].isin(["mid", "high"]).astype(int)
    else:
        raise ValueError(strategy)

    x["strategy"] = strategy
    x["total_return"] = np.where(x["on"] == 1, x["ret_b"], x["ret_base"])
    x["signal_exit_th_total_profit"] = np.where(x["on"] == 1, x["sth_b"], x["sth_base"])
    x["profit_damaged_count"] = np.where(x["on"] == 1, x["profit_damaged_count_b"], 0.0)
    x["profit_damaged_total_damage"] = np.where(x["on"] == 1, x["profit_damaged_damage_b"], 0.0)
    x["applied_trade_count"] = np.where(x["on"] == 1, x["aux_applied_b"], 0.0)
    return x


def _summarize_vs_base(df_s: pd.DataFrame, df_b: pd.DataFrame, segment: str) -> dict[str, float]:
    s = df_s[df_s["segment"] == segment].copy()
    b = df_b[df_b["segment"] == segment].copy()
    if s.empty or b.empty:
        return {}
    s = s.sort_values("t_end")
    b = b.sort_values("t_end")
    dret = s["total_return"].to_numpy() - b["total_return"].to_numpy()
    dsth = s["signal_exit_th_total_profit"].to_numpy() - b["signal_exit_th_total_profit"].to_numpy()
    on_mean_run, on_max_run = _compute_on_runs(s)
    return {
        "n": float(len(s)),
        "on_ratio": float(s["on"].mean()),
        "mean_total_return": float(s["total_return"].mean()),
        "mean_dret_vs_base": float(np.mean(dret)),
        "std_dret_vs_base": float(np.std(dret, ddof=1)) if len(dret) > 1 else 0.0,
        "mean_sth": float(s["signal_exit_th_total_profit"].mean()),
        "mean_dsth_vs_base": float(np.mean(dsth)),
        "ret_std": float(np.std(s["total_return"].to_numpy(), ddof=1)) if len(s) > 1 else 0.0,
        "ret_std_base": float(np.std(b["total_return"].to_numpy(), ddof=1)) if len(s) > 1 else 0.0,
        "mean_pd_count": float(s["profit_damaged_count"].mean()),
        "sum_pd_damage": float(s["profit_damaged_total_damage"].sum()),
        "mean_applied": float(s["applied_trade_count"].mean()),
        "on_mean_run": on_mean_run,
        "on_max_run": on_max_run,
    }


def _judge(sm: dict[str, float]) -> str:
    if not sm:
        return "유지 (조건부)"
    ok = (
        sm["mean_dret_vs_base"] >= 0
        and sm["mean_dsth_vs_base"] >= 0
        and sm["mean_pd_count"] <= 8.0
        and 0.2 <= sm["on_ratio"] <= 0.8
    )
    weak = sm["mean_dret_vs_base"] >= 0 and sm["mean_dsth_vs_base"] >= 0
    bad = sm["mean_dret_vs_base"] < 0 or sm["ret_std"] > sm["ret_std_base"] * 1.05 or sm["mean_pd_count"] > 12.0
    if ok:
        return "승급 (운영 반영 가능)"
    if bad:
        return "탈락"
    if weak:
        return "유지 (조건부)"
    return "유지 (조건부)"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    p.add_argument("--activation-csv", default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_LOGIC_REPORT.csv")
    p.add_argument("--ohlc-csv", default="data/ohlcv/BTCUSDT_5m_full.csv")
    p.add_argument("--out-md", default="data/diagnostics/fr2/SIGNAL_EXIT_TH_CONDITIONAL_DEPLOYMENT_REPORT.md")
    p.add_argument("--out-csv", default="data/diagnostics/fr2/SIGNAL_EXIT_TH_CONDITIONAL_DEPLOYMENT_REPORT.csv")
    args = p.parse_args()

    t_anchor = pd.Timestamp(args.t_max)
    if t_anchor.tzinfo is None:
        t_anchor = t_anchor.tz_localize("UTC")
    t_anchor = t_anchor.tz_convert("UTC")

    df = pd.read_csv(args.activation_csv)
    if "class" in df.columns:
        df = df.drop(columns=["class"])

    # append fixed_180d row
    roll = df[df["segment"].str.startswith("rolling_")].copy()
    q1 = float(roll["volatility_std"].quantile(1 / 3))
    q2 = float(roll["volatility_std"].quantile(2 / 3))
    vol180 = _volatility_std_from_ohlc(Path(args.ohlc_csv), t_anchor, 180)
    fixed = {
        "segment": "fixed_180d",
        "window_days": 180,
        "t_end": str(t_anchor),
        "ret_base": FIXED180_BASELINE_TR,
        "ret_b": FIXED180_B_TR,
        "sth_base": FIXED180_BASELINE_STH,
        "sth_b": FIXED180_B_STH,
        "unique_round_trips": FIXED180_UT,
        "mean_mae": np.nan,
        "mean_mfe": np.nan,
        "short_share": np.nan,
        "volatility_std": vol180,
        "profit_damaged_count_b": FIXED180_PD_B,
        "profit_damaged_damage_b": FIXED180_PD_DMG_B,
        "aux_applied_b": FIXED180_AUX_B,
        "vol_bucket": _bucket(vol180, q1, q2),
    }
    for c in df.columns:
        if c not in fixed:
            fixed[c] = np.nan
    df = pd.concat([df, pd.DataFrame([fixed])], ignore_index=True)

    # optional research flag only
    df["mfe_b_on"] = 0
    for seg in ("rolling_90d", "rolling_60d"):
        m = df["segment"] == seg
        sub = df.loc[m].copy()
        sub["_ts"] = pd.to_datetime(sub["t_end"], utc=True)
        sub = sub.sort_values("_ts")
        idx = sub.index.to_numpy()
        mfe = sub["mean_mfe"].to_numpy(dtype=float)
        for i, ix in enumerate(idx):
            if i == 0 or np.isnan(mfe[i]):
                continue
            start = max(0, i - 12)
            hist = mfe[start:i]
            hist = hist[~np.isnan(hist)]
            if len(hist) == 0:
                continue
            thr = float(np.percentile(hist, 60))
            df.loc[ix, "mfe_b_on"] = int(float(mfe[i]) >= thr)

    b0 = _strategy_rows(df, "baseline")
    rt = _strategy_rows(df, "rt_a")
    vl = _strategy_rows(df, "vol_a")
    merged = pd.concat([b0, rt, vl], ignore_index=True)

    # cumulative on ratio trend
    trend_rows: list[dict[str, Any]] = []
    for sname, sdf in (("rt_a", rt), ("vol_a", vl)):
        for seg in ("rolling_90d", "rolling_60d"):
            z = sdf[sdf["segment"] == seg].copy()
            if z.empty:
                continue
            z["_ts"] = pd.to_datetime(z["t_end"], utc=True)
            z = z.sort_values("_ts").reset_index(drop=True)
            z["cum_on_ratio"] = z["on"].expanding().mean()
            for _, r in z.iterrows():
                trend_rows.append(
                    {
                        "strategy": sname,
                        "segment": seg,
                        "t_end": r["t_end"],
                        "on": int(r["on"]),
                        "cum_on_ratio": float(r["cum_on_ratio"]),
                        "mfe_b_on": int(r.get("mfe_b_on", 0)),
                    }
                )
    trend_df = pd.DataFrame(trend_rows)

    # summary
    segs = ("fixed_180d", "rolling_90d", "rolling_60d")
    sm_rt = {seg: _summarize_vs_base(rt, b0, seg) for seg in segs}
    sm_vl = {seg: _summarize_vs_base(vl, b0, seg) for seg in segs}
    sm_rv = {
        seg: _summarize_vs_base(rt, vl.rename(columns={"total_return": "total_return", "signal_exit_th_total_profit": "signal_exit_th_total_profit"}), seg)
        for seg in segs
    }

    # write csv: detailed + trend
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "strategy",
        "segment",
        "window_days",
        "t_end",
        "on",
        "unique_round_trips",
        "vol_bucket",
        "mfe_b_on",
        "total_return",
        "signal_exit_th_total_profit",
        "profit_damaged_count",
        "profit_damaged_total_damage",
        "applied_trade_count",
    ]
    merged_csv = merged[base_cols].copy()
    merged_csv["row_type"] = "detail"
    if not trend_df.empty:
        t = trend_df.copy()
        t["row_type"] = "on_ratio_trend"
        t["window_days"] = np.nan
        t["unique_round_trips"] = np.nan
        t["vol_bucket"] = ""
        t["total_return"] = np.nan
        t["signal_exit_th_total_profit"] = np.nan
        t["profit_damaged_count"] = np.nan
        t["profit_damaged_total_damage"] = np.nan
        t["applied_trade_count"] = np.nan
        t = t[merged_csv.columns.tolist() + ["cum_on_ratio"]]
        merged_csv["cum_on_ratio"] = np.nan
        all_csv = pd.concat([merged_csv, t], ignore_index=True)
    else:
        merged_csv["cum_on_ratio"] = np.nan
        all_csv = merged_csv
    all_csv.to_csv(out_csv, index=False)

    # markdown
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            if np.isnan(v) or np.isinf(v):
                return "—"
            return f"{v:.6f}"
        return str(v)

    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH Conditional Deployment Report")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| t_anchor | `{t_anchor.isoformat()}` |")
    lines.append("| threshold | 0.6 |")
    lines.append("| 실제 적용 | RT-A (`unique_round_trips >= 225`) |")
    lines.append("| shadow | VOL-A (`vol_bucket in {mid, high}`) |")
    lines.append("| research only | MFE-B (`mfe_b_on` 계산만, 로직 미적용) |")
    lines.append("")

    lines.append("## A. RT-A vs baseline")
    lines.append("")
    lines.append("| segment | ON ratio | mean ΔTR | mean Δsth | ret std (rt/base) | mean pd_count | mean applied | mean ON run / max ON run |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seg in segs:
        s = sm_rt.get(seg) or {}
        if not s:
            continue
        lines.append(
            f"| {seg} | {fmt(s['on_ratio'])} | {fmt(s['mean_dret_vs_base'])} | {fmt(s['mean_dsth_vs_base'])} | "
            f"{fmt(s['ret_std'])} / {fmt(s['ret_std_base'])} | {fmt(s['mean_pd_count'])} | {fmt(s['mean_applied'])} | "
            f"{fmt(s['on_mean_run'])} / {fmt(s['on_max_run'])} |"
        )
    lines.append("")

    lines.append("## B. VOL-A vs baseline (shadow)")
    lines.append("")
    lines.append("| segment | ON ratio | mean ΔTR | mean Δsth | ret std (vol/base) | mean pd_count | mean applied | mean ON run / max ON run |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seg in segs:
        s = sm_vl.get(seg) or {}
        if not s:
            continue
        lines.append(
            f"| {seg} | {fmt(s['on_ratio'])} | {fmt(s['mean_dret_vs_base'])} | {fmt(s['mean_dsth_vs_base'])} | "
            f"{fmt(s['ret_std'])} / {fmt(s['ret_std_base'])} | {fmt(s['mean_pd_count'])} | {fmt(s['mean_applied'])} | "
            f"{fmt(s['on_mean_run'])} / {fmt(s['on_max_run'])} |"
        )
    lines.append("")

    lines.append("## C. RT-A vs VOL-A 직접 비교 (RT-A - VOL-A)")
    lines.append("")
    lines.append("| segment | mean ΔTR (rt-vol) | mean Δsth (rt-vol) | pd_count diff (rt-vol) | ON ratio rt/vol |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for seg in segs:
        srt = sm_rt.get(seg) or {}
        svl = sm_vl.get(seg) or {}
        if not srt or not svl:
            continue
        lines.append(
            f"| {seg} | {fmt(srt['mean_dret_vs_base'] - svl['mean_dret_vs_base'])} | "
            f"{fmt(srt['mean_dsth_vs_base'] - svl['mean_dsth_vs_base'])} | "
            f"{fmt(srt['mean_pd_count'] - svl['mean_pd_count'])} | "
            f"{fmt(srt['on_ratio'])} / {fmt(svl['on_ratio'])} |"
        )
    lines.append("")

    lines.append("## D. ON ratio / applied_trade_count 변화")
    lines.append("")
    if trend_df.empty:
        lines.append("- trend rows 없음")
    else:
        lines.append("- `CSV`의 `row_type=on_ratio_trend`에서 시계열 추이 확인 가능")
        for seg in ("rolling_90d", "rolling_60d"):
            r = trend_df[(trend_df["strategy"] == "rt_a") & (trend_df["segment"] == seg)]
            v = trend_df[(trend_df["strategy"] == "vol_a") & (trend_df["segment"] == seg)]
            if len(r) and len(v):
                lines.append(
                    f"- {seg}: RT-A cum_on_ratio {r['cum_on_ratio'].iloc[-1]:.4f}, VOL-A cum_on_ratio {v['cum_on_ratio'].iloc[-1]:.4f}"
                )
    lines.append("")

    lines.append("## E. profit_damaged 추이")
    lines.append("")
    lines.append("| strategy | segment | mean pd_count | sum pd_damage |")
    lines.append("| --- | --- | ---: | ---: |")
    for name, sms in (("RT-A", sm_rt), ("VOL-A", sm_vl)):
        for seg in segs:
            s = sms.get(seg) or {}
            if not s:
                continue
            lines.append(f"| {name} | {seg} | {fmt(s['mean_pd_count'])} | {fmt(s['sum_pd_damage'])} |")
    lines.append("")

    lines.append("## 운영 판정 (고정 기준 적용)")
    lines.append("")
    lines.append(f"- RT-A: **{_judge(sm_rt.get('rolling_90d') or {})}**")
    lines.append(f"- VOL-A (shadow): **{_judge(sm_vl.get('rolling_90d') or {})}**")
    lines.append("")
    lines.append("- 주의: 본 리포트는 pre-production 데이터 수집용이며, VOL-A는 shadow 계산만 수행(실제 로직 미반영).")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[cond-deploy] wrote {out_md} and {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
