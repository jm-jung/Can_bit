#!/usr/bin/env python3
"""
SIGNAL_EXIT_TH activation v2 — 단일 후보 검증 (RT-A, VOL-A, MFE-B만).

- 결합(CB-1) 없음
- baseline: loss_aux OFF / B: loss_aux + 보호게이트(B)
"""

from __future__ import annotations

import argparse
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
RT_A_THRESHOLD = 225
MFE_B_K = 12
MFE_B_P = 0.60

# LOSS_AUX_GATE_VALIDATION_REPORT.csv (B / BASELINE · fixed 180d end=2026-03-20) — 오프라인 재현용
FIXED180_BASELINE_TR = -0.07665713566110521
FIXED180_BASELINE_STH = -0.03632222787580999
FIXED180_B_TR = -0.07406819002651444
FIXED180_B_STH = -0.03352406333693331
FIXED180_PD_B = 0
FIXED180_PD_DMG_B = 0.0
FIXED180_AUX_B = 19
FIXED180_UT = 335


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


def _bucket_from_thresholds(vol: float, q1: float, q2: float) -> str:
    if vol <= q1:
        return "low"
    if vol <= q2:
        return "mid"
    return "high"


def _volatility_std_from_ohlc(ohlc_path: Path, t_end: pd.Timestamp, window_days: int) -> float:
    """close pct_change 표준편차 (activation_logic과 동일 정의)."""
    dfo = pd.read_csv(ohlc_path, usecols=["timestamp", "close"])
    dfo["timestamp"] = pd.to_datetime(dfo["timestamp"], utc=True)
    w0 = t_end - pd.Timedelta(days=window_days)
    m = (dfo["timestamp"] >= w0) & (dfo["timestamp"] <= t_end)
    close = dfo.loc[m, "close"].astype(float)
    ret = close.pct_change().dropna()
    return float(ret.std()) if len(ret) else 0.0


def _apply_activation_features(df: pd.DataFrame) -> pd.DataFrame:
    """vol_bucket(세그먼트별), fixed용 전역 q1/q2, MFE-B 플래그."""
    df = df.copy()
    df["vol_bucket"] = ""
    for seg in ("rolling_90d", "rolling_60d"):
        m = df["segment"] == seg
        if not m.any():
            continue
        df.loc[m, "vol_bucket"] = _quantile_bucket(df.loc[m, "volatility_std"])

    roll_only = df[df["segment"].str.startswith("rolling_")].copy()
    vpool = roll_only["volatility_std"]
    q1_g = float(vpool.quantile(1 / 3))
    q2_g = float(vpool.quantile(2 / 3))

    df["mfe_b_threshold"] = np.nan
    df["mfe_b_eligible"] = 0
    df["mfe_b_on"] = 0
    for seg in ("rolling_90d", "rolling_60d"):
        m = df["segment"] == seg
        sub = df.loc[m].copy()
        sub["_ts"] = pd.to_datetime(sub["t_end"], utc=True)
        sub = sub.sort_values("_ts")
        idx = sub.index.to_numpy()
        mfe = sub["mean_mfe"].to_numpy(dtype=float)
        thr_arr, on_arr = _mfe_b_threshold_series(mfe)
        for j, ix in enumerate(idx):
            df.loc[ix, "mfe_b_threshold"] = thr_arr[j]
            df.loc[ix, "mfe_b_eligible"] = int(j > 0)
            df.loc[ix, "mfe_b_on"] = int(on_arr[j])

    fmask = df["segment"] == "fixed_180d"
    if fmask.any():
        ix = df.index[fmask][0]
        v180 = float(df.loc[ix, "volatility_std"])
        df.loc[ix, "vol_bucket"] = _bucket_from_thresholds(v180, q1_g, q2_g)
        r90 = df[df["segment"] == "rolling_90d"].copy()
        r90["_ts"] = pd.to_datetime(r90["t_end"], utc=True)
        r90 = r90.sort_values("_ts")
        mfe_hist = r90["mean_mfe"].dropna().to_numpy()
        if len(mfe_hist) >= 1:
            last12 = mfe_hist[-min(MFE_B_K, len(mfe_hist)) :]
            thr180 = float(np.percentile(last12, MFE_B_P * 100.0))
            df.loc[ix, "mfe_b_threshold"] = thr180
            df.loc[ix, "mfe_b_eligible"] = 1
            mf = df.loc[ix, "mean_mfe"]
            if pd.isna(mf):
                df.loc[ix, "mfe_b_on"] = 0
            else:
                df.loc[ix, "mfe_b_on"] = int(float(mf) >= thr180)

    df["mfe_b_on"] = df["mfe_b_on"].fillna(0).astype(int)
    return df


def _build_df_offline(
    *,
    activation_csv: Path,
    t_anchor: pd.Timestamp,
    ohlc_csv: Path,
    fixed_mean_mfe: float | None,
    fixed_mean_mfe_proxy: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(activation_csv)
    drop_cols = [c for c in ("class", "vol_bucket", "dret", "dsth") if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    vol180 = _volatility_std_from_ohlc(ohlc_csv, t_anchor, 180)
    r90 = df[df["segment"] == "rolling_90d"]
    if fixed_mean_mfe is not None:
        mf180 = float(fixed_mean_mfe)
    elif fixed_mean_mfe_proxy == "roll90_median":
        mf180 = float(r90["mean_mfe"].median())
    else:
        mf180 = float("nan")

    fixed_row = {
        "segment": "fixed_180d",
        "window_days": 180,
        "t_end": str(t_anchor),
        "ret_base": FIXED180_BASELINE_TR,
        "ret_b": FIXED180_B_TR,
        "sth_base": FIXED180_BASELINE_STH,
        "sth_b": FIXED180_B_STH,
        "unique_round_trips": FIXED180_UT,
        "mean_mae": np.nan,
        "mean_mfe": mf180,
        "volatility_std": vol180,
        "profit_damaged_count_b": FIXED180_PD_B,
        "profit_damaged_damage_b": FIXED180_PD_DMG_B,
        "aux_applied_b": FIXED180_AUX_B,
    }
    return pd.concat([df, pd.DataFrame([fixed_row])], ignore_index=True)


def _profit_damaged(base: dict[str, Any], b: dict[str, Any]) -> tuple[int, float]:
    eb = {
        (str(e.get("entry_ts")), str(e.get("exit_ts")), str(e.get("direction"))): e
        for e in _sth_exits(base)
    }
    ec = {
        (str(e.get("entry_ts")), str(e.get("exit_ts")), str(e.get("direction"))): e
        for e in _sth_exits(b)
    }
    common = set(eb.keys()) & set(ec.keys())
    pd_count = 0
    pd_damage = 0.0
    for k in common:
        pb = float(eb[k].get("pnl_change", eb[k].get("profit", 0.0)) or 0.0)
        pc = float(ec[k].get("pnl_change", ec[k].get("profit", 0.0)) or 0.0)
        if pb > 0 and pc < pb:
            pd_count += 1
            pd_damage += pc - pb
    return pd_count, pd_damage


def _simulate(
    df: pd.DataFrame,
    cond: Callable[[pd.Series], bool],
    name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sim = df.copy()
    sim["on"] = sim.apply(cond, axis=1).astype(int)
    sim["ret_sel"] = np.where(sim["on"] == 1, sim["ret_b"], sim["ret_base"])
    sim["sth_sel"] = np.where(sim["on"] == 1, sim["sth_b"], sim["sth_base"])
    sim["pd_sel"] = np.where(sim["on"] == 1, sim["profit_damaged_count_b"], 0.0)
    sim["pd_damage_sel"] = np.where(sim["on"] == 1, sim["profit_damaged_damage_b"], 0.0)
    sim["applied_sel"] = np.where(sim["on"] == 1, sim["aux_applied_b"], 0.0)
    sim["dret_vs_base"] = sim["ret_sel"] - sim["ret_base"]
    sim["dsth_vs_base"] = sim["sth_sel"] - sim["sth_base"]
    sim["candidate"] = name
    return sim, {"name": name}


def _segment_summary(sim: pd.DataFrame, seg: str) -> dict[str, Any]:
    s = sim[sim["segment"] == seg]
    if s.empty:
        return {}
    dret = s["dret_vs_base"]
    return {
        "n_windows": int(len(s)),
        "on_ratio": float(s["on"].mean()),
        "mean_total_return": float(s["ret_sel"].mean()),
        "mean_dret_vs_base": float(dret.mean()),
        "std_dret_vs_base": float(dret.std(ddof=1)) if len(s) > 1 else 0.0,
        "mean_sth": float(s["sth_sel"].mean()),
        "mean_dsth_vs_base": float(s["dsth_vs_base"].mean()),
        "ret_std_sel": float(s["ret_sel"].std(ddof=1)) if len(s) > 1 else 0.0,
        "ret_std_base": float(s["ret_base"].std(ddof=1)) if len(s) > 1 else 0.0,
        "mean_pd_count": float(s["pd_sel"].mean()),
        "mean_pd_damage": float(s["pd_damage_sel"].mean()),
        "mean_applied": float(s["applied_sel"].mean()),
    }


def _fixed180_summary(sim: pd.DataFrame) -> dict[str, Any]:
    s = sim[sim["segment"] == "fixed_180d"]
    if s.empty or len(s) != 1:
        return {}
    r = s.iloc[0]
    return {
        "on": int(r["on"]),
        "total_return": float(r["ret_sel"]),
        "dret_vs_base": float(r["dret_vs_base"]),
        "signal_exit_th_total_profit": float(r["sth_sel"]),
        "dsth_vs_base": float(r["dsth_vs_base"]),
        "profit_damaged_count": float(r["pd_sel"]),
        "profit_damaged_total_damage": float(r["pd_damage_sel"]),
        "applied_trade_count": float(r["applied_sel"]),
        "ret_std_sel": float("nan"),
        "ret_std_base": float("nan"),
    }


def _mfe_b_threshold_series(mfe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """직전 최대 K개 창의 mean_mfe에 대해 p 분위 임계; i<1이면 nan."""
    n = len(mfe)
    thr = np.full(n, np.nan)
    on = np.zeros(n, dtype=bool)
    for i in range(n):
        if i == 0:
            continue
        start = max(0, i - MFE_B_K)
        hist = mfe[start:i]
        if len(hist) < 1:
            continue
        thr[i] = float(np.percentile(hist, MFE_B_P * 100.0))
        on[i] = bool(mfe[i] >= thr[i])
    return thr, on


def _verdict(s90: dict[str, Any], s60: dict[str, Any], f180: dict[str, Any]) -> str:
    if not s90 or not s60:
        return "FAIL"
    ok90 = s90["mean_dret_vs_base"] > 0 and s90["mean_dsth_vs_base"] > 0
    ok60 = s60["mean_dret_vs_base"] >= 0 and s60["mean_pd_count"] <= 8.0
    std_ok = s60["ret_std_sel"] <= s60["ret_std_base"] * 1.05 + 1e-12
    stable90 = s90["std_dret_vs_base"] <= abs(s90["mean_dret_vs_base"]) * 3.0 + 1e-9 or s90["std_dret_vs_base"] < 0.02
    # 평균 개선 폭이 매우 작으면(스파스 ON·약한 Δ) 설명력 관점에서 WEAK
    small_lift = s90["mean_dret_vs_base"] < 0.0005 or s90["mean_dsth_vs_base"] < 0.0005
    sparse_on = s90["on_ratio"] < 0.35

    if ok90 and ok60 and std_ok and s90["mean_dret_vs_base"] > 1e-6:
        if small_lift or sparse_on:
            return "WEAK PASS"
        if stable90 and s60["mean_pd_count"] <= 5.0:
            return "PASS"
        return "WEAK PASS"
    if ok90 or (s90["mean_dret_vs_base"] > 0 and s90["mean_dsth_vs_base"] > 0):
        return "WEAK PASS"
    return "FAIL"


def _narrative_lines(cname: str, on: pd.DataFrame, off: pd.DataFrame) -> list[str]:
    """ON vs OFF 평균 비교 기반 짧은 서술."""

    def _m(col: str, d: pd.DataFrame) -> float:
        if d.empty or col not in d.columns:
            return float("nan")
        return float(pd.to_numeric(d[col], errors="coerce").mean())

    vo, voff = _m("volatility_std", on), _m("volatility_std", off)
    to, toff = _m("unique_round_trips", on), _m("unique_round_trips", off)
    mo, moff = _m("mean_mfe", on), _m("mean_mfe", off)
    lines: list[str] = []
    if not on.empty and not off.empty and not (np.isnan(vo) or np.isnan(voff)):
        if vo > voff * 1.02:
            lines.append(f"- **먹히는 구간**: ON 창의 평균 변동성(volatility_std)이 OFF보다 **높음** (ON {vo:.6f} vs OFF {voff:.6f}).")
        elif vo < voff * 0.98:
            lines.append(f"- **먹히는 구간**: ON 창의 평균 변동성이 OFF보다 **낮음** (ON {vo:.6f} vs OFF {voff:.6f}) — `{cname}` 축은 ‘저변동’과도 겹칠 수 있음.")
        else:
            lines.append(f"- **먹히는 구간**: ON/OFF 간 평균 변동성 차이는 **크지 않음** (ON {vo:.6f} vs OFF {voff:.6f}).")
    if not on.empty and not off.empty and not (np.isnan(to) or np.isnan(toff)):
        if to > toff:
            lines.append(f"- **거래 밀도**: ON 창의 `unique_round_trips` 평균이 OFF보다 **큼** (ON {to:.1f} vs OFF {toff:.1f}).")
        elif to < toff:
            lines.append(f"- **거래 밀도**: ON 창의 `unique_round_trips` 평균이 OFF보다 **작음** (ON {to:.1f} vs OFF {toff:.1f}) — ‘항상 많을수록’은 아님.")
    if not on.empty and not off.empty and not (np.isnan(mo) or np.isnan(moff)):
        if mo > moff:
            lines.append(f"- **MFE 구조**: ON 창의 평균 mean_mfe가 OFF보다 **큼** (ON {mo:.6f} vs OFF {moff:.6f}).")
        else:
            lines.append(f"- **MFE 구조**: ON 창의 평균 mean_mfe가 OFF보다 **작거나 비슷** — 이 축 단독으로는 유리한 청산 MFE와 완전히 정렬되지 않을 수 있음.")
    return lines


def _compare_on_off(sim: pd.DataFrame, seg: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = sim[sim["segment"] == seg]
    on = s[s["on"] == 1]
    off = s[s["on"] == 0]
    return on, off


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=420)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--rolling-step-days", type=int, default=30)
    parser.add_argument(
        "--offline-from-csv",
        default=None,
        help="백테스트 생략: ACTIVATION_LOGIC_REPORT 등 동일 스키마 CSV + OHLC + LOSS_AUX 180d 고정치",
    )
    parser.add_argument(
        "--ohlc-csv",
        default="data/ohlcv/BTCUSDT_5m_full.csv",
        help="fixed_180d volatility_std 계산용",
    )
    parser.add_argument(
        "--fixed-mean-mfe",
        type=float,
        default=None,
        help="fixed_180d baseline mean_mfe (미지정 시 --fixed-mean-mfe-proxy)",
    )
    parser.add_argument(
        "--fixed-mean-mfe-proxy",
        choices=("roll90_median", "none"),
        default="roll90_median",
        help="none이면 mean_mfe NaN → fixed MFE-B는 OFF",
    )
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_V2_SINGLE_VALIDATION.md",
    )
    parser.add_argument(
        "--out-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_ACTIVATION_V2_SINGLE_VALIDATION.csv",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)
    offline_csv = args.offline_from_csv

    if offline_csv:
        print(f"[act-v2] OFFLINE mode from {offline_csv}", flush=True)
        df = _build_df_offline(
            activation_csv=Path(offline_csv),
            t_anchor=t_anchor,
            ohlc_csv=Path(args.ohlc_csv),
            fixed_mean_mfe=args.fixed_mean_mfe,
            fixed_mean_mfe_proxy=args.fixed_mean_mfe_proxy,
        )
        df = _apply_activation_features(df)
    else:
        print(f"[act-v2] load ensemble days_full={args.days_full} ...", flush=True)
        df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)
        print(f"[act-v2] ts range {ts_min} ~ {ts_max}", flush=True)

        rows: list[dict[str, Any]] = []

        def add_window(segment: str, window_days: int, te: pd.Timestamp) -> None:
            tag = f"{segment} end={te.date()}"
            print(f"[act-v2] {tag} ...", flush=True)
            base = run_loss_aux_window(
                df_bt=df_bt,
                pl_primary=pl_primary,
                ps_primary=ps_primary,
                t_end=te,
                window_days=window_days,
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
                window_days=window_days,
                threshold=thr,
                emit_trade_log=True,
                signal_exit_th_loss_aux_delta_p=DELTA_P,
                signal_exit_th_loss_aux_mae_cut=MAE_CUT,
                signal_exit_th_loss_aux_gate_delta_unreal_positive=True,
                signal_exit_th_loss_aux_gate_mfe_min=None,
            )
            mask = _window_mask(df_bt, te, window_days)
            df_w = df_bt.loc[mask].reset_index(drop=True)
            close = df_w["close"].astype(float)
            ret = close.pct_change().dropna()
            vol = float(ret.std()) if len(ret) else 0.0

            ex = _sth_exits(base)
            if ex:
                mae = [float(e.get("max_adverse_excursion")) for e in ex if e.get("max_adverse_excursion") is not None]
                mfe = [float(e.get("max_favorable_excursion")) for e in ex if e.get("max_favorable_excursion") is not None]
                mean_mae = float(np.mean(mae)) if mae else float("nan")
                mean_mfe = float(np.mean(mfe)) if mfe else float("nan")
            else:
                mean_mae = float("nan")
                mean_mfe = float("nan")

            pd_n, pd_d = _profit_damaged(base, b)
            rows.append(
                {
                    "segment": segment,
                    "window_days": window_days,
                    "t_end": str(te),
                    "ret_base": float(base.get("total_return", 0.0) or 0.0),
                    "ret_b": float(b.get("total_return", 0.0) or 0.0),
                    "sth_base": float(base.get("signal_exit_th_total_profit", 0.0) or 0.0),
                    "sth_b": float(b.get("signal_exit_th_total_profit", 0.0) or 0.0),
                    "unique_round_trips": int(base.get("unique_round_trips") or 0),
                    "mean_mae": mean_mae,
                    "mean_mfe": mean_mfe,
                    "volatility_std": vol,
                    "profit_damaged_count_b": int(pd_n),
                    "profit_damaged_damage_b": float(pd_d),
                    "aux_applied_b": int(b.get("signal_exit_th_loss_aux_applied", 0) or 0),
                }
            )

        if t_anchor - pd.Timedelta(days=180) >= ts_min:
            add_window("fixed_180d", 180, t_anchor)

        for span in (90, 60):
            for te in _rolling_end_dates(ts_min, ts_max, t_anchor, span, int(args.rolling_step_days)):
                add_window(f"rolling_{span}d", span, pd.Timestamp(te))

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No rows produced.")
        df = _apply_activation_features(df)

    candidates: list[tuple[str, str, Callable[[pd.Series], bool]]] = [
        ("RT-A", f"unique_round_trips >= {RT_A_THRESHOLD}", lambda r: int(r["unique_round_trips"]) >= RT_A_THRESHOLD),
        ("VOL-A", "vol_bucket in {mid, high}", lambda r: str(r["vol_bucket"]) in ("mid", "high")),
        (
            "MFE-B",
            f"mean_mfe >= rolling {MFE_B_K}-window p={MFE_B_P} (시간순 직전 창들)",
            lambda r: bool(int(r.get("mfe_b_on", 0)) == 1),
        ),
    ]

    all_detail: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []

    for cname, cdesc, cond in candidates:
        sim, _ = _simulate(df, cond, cname)
        all_detail.append(sim)
        s90 = _segment_summary(sim, "rolling_90d")
        s60 = _segment_summary(sim, "rolling_60d")
        f180 = _fixed180_summary(sim)
        summaries.append(
            {
                "candidate": cname,
                "description": cdesc,
                "roll90": s90,
                "roll60": s60,
                "fixed180": f180,
                "verdict": _verdict(s90, s60, f180),
            }
        )

    out_detail = pd.concat(all_detail, ignore_index=True)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_detail.to_csv(out_csv, index=False)

    # markdown
    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH activation v2 — 단일 후보 검증")
    lines.append("")
    if offline_csv:
        lines.append("## 실행 모드")
        lines.append("")
        lines.append(
            f"- **OFFLINE**: `{offline_csv}` + OHLC(`{args.ohlc_csv}`) + LOSS_AUX fixed180 고정치. "
            f"fixed_180d `mean_mfe`는 `--fixed-mean-mfe` 또는 proxy(`{args.fixed_mean_mfe_proxy}`) 사용."
        )
        lines.append("")
    lines.append("## 공통 설정")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| t_anchor | `{t_anchor.isoformat()}` |")
    lines.append(f"| threshold (min_max_proba) | {thr} |")
    lines.append(f"| loss_aux | δp={DELTA_P}, mae_cut={MAE_CUT}, 보호게이트(B): delta_unreal_4bars>0 → skip |")
    lines.append(f"| RT-A | unique_round_trips >= {RT_A_THRESHOLD} |")
    lines.append("| VOL-A | vol_bucket in {mid, high} (롤링 창: **동일 segment** 내 `volatility_std` tertile) |")
    lines.append(
        f"| MFE-B | mean_mfe >= 직전 최대 {MFE_B_K}개 창의 mean_mfe의 {int(MFE_B_P * 100)}%tile; 첫 창은 비교 불가(OFF) |"
    )
    lines.append(
        f"| fixed_180d VOL-A | 롤링(90d+60d) `volatility_std`로 학습한 전역 q1/q2로 bucket 부여 |"
    )
    lines.append(
        f"| fixed_180d MFE-B | rolling_90d의 시간순 `mean_mfe` 마지막 ≤{MFE_B_K}개로 임계 후 비교 |"
    )
    lines.append("")

    lines.append("## 후보별 요약 지표")
    lines.append("")
    lines.append("| 후보 | 구간 | total_return(선택) | ΔTR vs base | sth(선택) | Δsth vs base | ret std sel/base | ON ratio | pd count | pd damage | applied |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")

    for sm in summaries:
        c = sm["candidate"]
        f = sm["fixed180"]
        if f:
            lines.append(
                f"| {c} | fixed_180d | {f['total_return']:.6f} | {f['dret_vs_base']:.6f} | "
                f"{f['signal_exit_th_total_profit']:.6f} | {f['dsth_vs_base']:.6f} | — | "
                f"{float(f['on']):.0f} | {f['profit_damaged_count']:.4f} | {f['profit_damaged_total_damage']:.6f} | {f['applied_trade_count']:.1f} |"
            )
        for seg, key in (("rolling_90d", "roll90"), ("rolling_60d", "roll60")):
            s = sm[key]
            if not s:
                continue
            lines.append(
                f"| {c} | {seg} | {s['mean_total_return']:.6f} | {s['mean_dret_vs_base']:.6f} | "
                f"{s['mean_sth']:.6f} | {s['mean_dsth_vs_base']:.6f} | "
                f"{s['ret_std_sel']:.6f} / {s['ret_std_base']:.6f} | {s['on_ratio']:.4f} | "
                f"{s['mean_pd_count']:.4f} | {s['mean_pd_damage']:.6f} | {s['mean_applied']:.4f} |"
            )
    lines.append("")

    lines.append("## A. roll90 요약 (후보별)")
    lines.append("")
    lines.append("| 후보 | mean Δtotal_return | mean Δsth | std Δtotal_return (창별) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for sm in summaries:
        s = sm["roll90"]
        if not s:
            continue
        lines.append(
            f"| {sm['candidate']} | {s['mean_dret_vs_base']:.6f} | {s['mean_dsth_vs_base']:.6f} | {s['std_dret_vs_base']:.6f} |"
        )
    lines.append("")

    lines.append("## B. roll60 요약 (후보별)")
    lines.append("")
    lines.append("| 후보 | mean Δtotal_return | std Δtotal_return | profit_damaged 평균 | ret std sel / base |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for sm in summaries:
        s = sm["roll60"]
        if not s:
            continue
        lines.append(
            f"| {sm['candidate']} | {s['mean_dret_vs_base']:.6f} | {s['std_dret_vs_base']:.6f} | "
            f"{s['mean_pd_count']:.6f} | {s['ret_std_sel']:.6f} / {s['ret_std_base']:.6f} |"
        )
    lines.append("")

    lines.append("## C. fixed 180d 비교 (baseline = loss_aux OFF)")
    lines.append("")
    base180 = df[df["segment"] == "fixed_180d"].iloc[0]
    lines.append(
        f"- baseline: total_return={base180['ret_base']:.6f}, sth={base180['sth_base']:.6f}"
    )
    lines.append("")
    lines.append("| 후보 | ON | total_return | ΔTR | sth | Δsth | pd_count | damage | applied |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for sm in summaries:
        f = sm["fixed180"]
        if not f:
            continue
        lines.append(
            f"| {sm['candidate']} | {f['on']:.0f} | {f['total_return']:.6f} | {f['dret_vs_base']:.6f} | "
            f"{f['signal_exit_th_total_profit']:.6f} | {f['dsth_vs_base']:.6f} | "
            f"{f['profit_damaged_count']:.4f} | {f['profit_damaged_total_damage']:.6f} | {f['applied_trade_count']:.1f} |"
        )
    lines.append("")

    # 분석: ON vs OFF 특성 (roll90)
    lines.append("## 핵심 비교 분석 — 창 특성 (roll90)")
    lines.append("")
    for cname, _, cond in candidates:
        sim, _ = _simulate(df, cond, cname)
        on, off = _compare_on_off(sim, "rolling_90d")
        lines.append(f"### {cname}")
        lines.append("")
        if on.empty and off.empty:
            lines.append("(데이터 없음)")
            lines.append("")
            continue
        def _mean(col: str, d: pd.DataFrame) -> float:
            if d.empty or col not in d.columns:
                return float("nan")
            return float(pd.to_numeric(d[col], errors="coerce").mean())

        lines.append("**ON 창 평균 특성**")
        lines.append("")
        lines.append(
            f"- n={len(on)}, mean volatility_std={_mean('volatility_std', on):.6f}, "
            f"mean unique_round_trips={_mean('unique_round_trips', on):.2f}, mean_mfe={_mean('mean_mfe', on):.6f}, "
            f"vol_bucket: low/mid/high = "
            f"{(on['vol_bucket']=='low').sum() if len(on) else 0}/"
            f"{(on['vol_bucket']=='mid').sum() if len(on) else 0}/"
            f"{(on['vol_bucket']=='high').sum() if len(on) else 0}"
        )
        lines.append("")
        lines.append("**OFF 창 평균 특성**")
        lines.append("")
        lines.append(
            f"- n={len(off)}, mean volatility_std={_mean('volatility_std', off):.6f}, "
            f"mean unique_round_trips={_mean('unique_round_trips', off):.2f}, mean_mfe={_mean('mean_mfe', off):.6f}, "
            f"vol_bucket: low/mid/high = "
            f"{(off['vol_bucket']=='low').sum() if len(off) else 0}/"
            f"{(off['vol_bucket']=='mid').sum() if len(off) else 0}/"
            f"{(off['vol_bucket']=='high').sum() if len(off) else 0}"
        )
        lines.append("")
        for nl in _narrative_lines(cname, on, off):
            lines.append(nl)
        lines.append("- **망하는 구간(OFF)**: 위 표에서 OFF는 손실보조(B)가 적용되지 않은 창; Δ 성능은 baseline 대비 ‘미적용’ 효과.")
        lines.append("- **failure mode**: 아래 `failure mode 자동 요약` 참고.")
        lines.append("")

    lines.append("## failure mode 자동 요약 (후보별)")
    lines.append("")
    for sm in summaries:
        c = sm["candidate"]
        s60 = sm["roll60"]
        s90 = sm["roll90"]
        f180 = sm["fixed180"]
        bits: list[str] = []
        if s60 and s60["mean_pd_count"] > 5.0:
            bits.append("roll60에서 profit_damaged 평균 상승 우려")
        if s60 and s60["ret_std_sel"] > s60["ret_std_base"] * 1.05:
            bits.append("roll60 수익률 분산 증가")
        if s90 and s90["std_dret_vs_base"] > 0.015:
            bits.append("roll90 창별 ΔTR 분산이 큼(일관성 낮음)")
        if s90 and s90["on_ratio"] < 0.35:
            bits.append("roll90 ON 비율이 낮음(희귀 선택·일반화 주의)")
        if f180 and f180["dret_vs_base"] < 0:
            bits.append("fixed 180d 단일 구간에서 ΔTR 음수")
        if f180 and int(f180.get("on", 0)) == 0:
            bits.append("fixed 180d에서 activation OFF(고정 mean_mfe·임계 정합 확인 필요)")
        if not bits:
            bits.append("특이 실패 시그널 없음(지표 기준)")
        lines.append(f"- **{c}**: " + "; ".join(bits))
    lines.append("")

    lines.append("## 최종 판정 (자동)")
    lines.append("")
    lines.append("| 후보 | 판정 |")
    lines.append("| --- | --- |")
    for sm in summaries:
        lines.append(f"| {sm['candidate']} | **{sm['verdict']}** |")
    lines.append("")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")
    lines.append("")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[act-v2] wrote {out_md} rows_detail={len(out_detail)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
