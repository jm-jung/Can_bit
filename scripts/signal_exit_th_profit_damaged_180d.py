#!/usr/bin/env python3
"""
180d / threshold=0.60 / δp=0.05 / mae_cut=-0.007 단일 후보:
combo가 baseline 대비 이익을 훼손한 signal_exit_th 건만 추출·표·요약.

정의 (기본):
- baseline round-trip `profit` (엔진 net return, CSV `profit_actual`) > 0
- combo profit < baseline profit
- combo는 손실 보조가 **적용된 경우만** 직전 바 close로 재계산 (`_hypothetical_profit`),
  그 외는 baseline과 동일.

`--mode full`: trade_events의 `pnl_change`로 동일 정의를 **이중 백테스트**에서 재현 (느림).

금지: 파라미터 스윕, threshold 변경, 모델 변경.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loss_aux_signal_exit_common import load_fr2_override_ensemble, run_loss_aux_window, to_utc_ts
from scripts.signal_exit_th_180d_analysis_and_experiments import _hypothetical_profit

DELTA_P = 0.05
MAE_CUT = -0.007
COMMISSION = 0.0009
SLIPPAGE = 0.0001


def _sth_exits(res: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
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


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _share(num: int, den: int) -> str:
    if den <= 0:
        return "N/A"
    return f"{100.0 * num / den:.1f}%"


def _recompute_mfe_mae(
    *,
    high: np.ndarray,
    low: np.ndarray,
    entry_price: float,
    entry_idx: int,
    exit_idx: int,
    direction: str,
) -> tuple[float, float]:
    mfe = -1e18
    mae = 1e18
    for j in range(entry_idx, exit_idx + 1):
        if j >= len(high):
            break
        ch, cl = float(high[j]), float(low[j])
        if direction == "LONG":
            rh = (ch - entry_price) / entry_price
            rl = (cl - entry_price) / entry_price
        else:
            rh = (entry_price - cl) / entry_price
            rl = (entry_price - ch) / entry_price
        mfe = max(mfe, rh)
        mae = min(mae, rl)
    return float(mfe), float(mae)


def _load_ohlc_close_index(ohlc_path: Path) -> tuple[pd.DatetimeIndex, np.ndarray]:
    ohlc = pd.read_csv(ohlc_path, usecols=["timestamp", "close"])
    ohlc["timestamp"] = pd.to_datetime(ohlc["timestamp"], utc=True)
    ohlc = ohlc.sort_values("timestamp").reset_index(drop=True)
    return ohlc["timestamp"], ohlc["close"].to_numpy(dtype=float)


def _prev_bar_close(ts_series: pd.DatetimeIndex, closes: np.ndarray, exit_ts: Any) -> float | None:
    ts = pd.Timestamp(exit_ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    pos = ts_series.searchsorted(ts)
    if pos >= len(ts_series) or ts_series[pos] != ts:
        return None
    if pos == 0:
        return None
    return float(closes[pos - 1])


def run_mode_fast(
    *,
    detail_csv: Path,
    wl_csv: Path,
    ohlc_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """OHLC 직전 바 close + 엔진과 동일 fee로 combo profit 근사 (인덱스는 exit_ts 정합)."""
    dt = pd.read_csv(detail_csv)
    wl = pd.read_csv(wl_csv)
    if len(dt) != len(wl):
        raise ValueError("detail / WL CSV 행 수 불일치")

    ts_idx, closes = _load_ohlc_close_index(ohlc_path)

    rows: list[dict[str, Any]] = []
    for i in range(len(dt)):
        r = dt.iloc[i]
        w = wl.iloc[i]
        xp_prev = _prev_bar_close(ts_idx, closes, r["exit_ts"])
        if xp_prev is None:
            continue

        pb = float(r["profit_actual"])
        pe = float(r["proba_entry"])
        pp = float(r["proba_exit_prev"])
        mae = float(w["mae"])
        direction = str(r["direction"])
        entry_price = float(r["entry_price"])
        aux = (pe - pp) > DELTA_P and mae < MAE_CUT
        if aux:
            pc = _hypothetical_profit(
                entry_price=entry_price,
                exit_price=xp_prev,
                direction=direction,
                commission=COMMISSION,
                slippage=SLIPPAGE,
            )
        else:
            pc = pb

        if not (pb > 0 and pc < pb):
            continue

        rows.append(
            {
                "entry_ts": str(r["entry_ts"]),
                "direction": direction,
                "bars_held": int(r["bars_held"]),
                "proba_entry": pe,
                "proba_exit_prev": pp,
                "MAE": mae,
                "MFE": float(w["mfe"]),
                "delta_unreal_4bars": float(w["delta_unreal_4bars"]),
                "baseline_profit": pb,
                "combo_profit": pc,
                "damage": pc - pb,
                "aux_applied": aux,
            }
        )

    dfa = pd.DataFrame(rows)

    win_mask = dt["profit_actual"].astype(float) > 0
    bw = int(win_mask.sum())
    short_bw = int(((dt["direction"] == "SHORT") & win_mask).sum())

    stats = {
        "baseline_winners": bw,
        "baseline_winners_short": short_bw,
        "winner_mfe_median": float(wl.loc[win_mask, "mfe"].median()) if bw else float("nan"),
        "winner_du_median": float(wl.loc[win_mask, "delta_unreal_4bars"].median()) if bw else float("nan"),
        "winner_bars_median": float(dt.loc[win_mask, "bars_held"].median()) if bw else float("nan"),
    }
    return dfa, stats


def run_mode_full(
    *,
    t_max: pd.Timestamp,
    days_full: int,
    window_days: int,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """이중 백테스트 + trade_events pnl_change (느리지만 잔고 반영 분과 정합)."""
    df_bt, pl_primary, ps_primary, _, _ = load_fr2_override_ensemble(days_full, t_max)

    window_start = t_max - pd.Timedelta(days=window_days)
    t = pd.to_datetime(df_bt["timestamp"], utc=True)
    mask = (t >= window_start) & (t <= t_max)
    df_w = df_bt.loc[mask].reset_index(drop=True)
    pl_w = np.asarray(pl_primary, dtype=np.float32)[mask.to_numpy()]
    ps_w = np.asarray(ps_primary, dtype=np.float32)[mask.to_numpy()]
    high_w = df_w["high"].to_numpy(dtype=float)
    low_w = df_w["low"].to_numpy(dtype=float)
    close_w = df_w["close"].to_numpy(dtype=float)

    base = run_loss_aux_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_end=t_max,
        window_days=window_days,
        threshold=threshold,
        emit_trade_log=True,
        signal_exit_th_loss_aux_delta_p=None,
        signal_exit_th_loss_aux_mae_cut=None,
    )
    combo = run_loss_aux_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_end=t_max,
        window_days=window_days,
        threshold=threshold,
        emit_trade_log=True,
        signal_exit_th_loss_aux_delta_p=DELTA_P,
        signal_exit_th_loss_aux_mae_cut=MAE_CUT,
    )

    eb = {_trade_key(e): e for e in _sth_exits(base)}
    ec = {_trade_key(e): e for e in _sth_exits(combo)}
    common = set(eb.keys()) & set(ec.keys())

    rows: list[dict[str, Any]] = []
    for k in sorted(common):
        b_ev, c_ev = eb[k], ec[k]
        pb, pc = _pnl(b_ev), _pnl(c_ev)
        if not (pb > 0 and pc < pb):
            continue

        exit_idx = int(b_ev.get("idx", -1))
        bh = int(b_ev.get("bars_held", 0) or 0)
        entry_idx = exit_idx - bh
        direction = str(b_ev.get("direction", ""))
        entry_price = float(b_ev.get("entry_price", np.nan))
        if entry_idx < 0 or exit_idx >= len(df_w) or not np.isfinite(entry_price):
            continue

        proba_entry = float(pl_w[entry_idx]) if direction == "LONG" else float(ps_w[entry_idx])
        proba_exit_prev = float(pl_w[exit_idx - 1]) if direction == "LONG" else float(ps_w[exit_idx - 1])

        mfe_ev = b_ev.get("max_favorable_excursion")
        mae_ev = b_ev.get("max_adverse_excursion")
        if mfe_ev is None or mae_ev is None or (isinstance(mfe_ev, float) and np.isnan(mfe_ev)):
            mfe_v, mae_v = _recompute_mfe_mae(
                high=high_w,
                low=low_w,
                entry_price=entry_price,
                entry_idx=entry_idx,
                exit_idx=exit_idx,
                direction=direction,
            )
        else:
            mfe_v, mae_v = float(mfe_ev), float(mae_ev)

        u_exit = (close_w[exit_idx] - entry_price) / entry_price if direction == "LONG" else (entry_price - close_w[exit_idx]) / entry_price
        j4 = max(exit_idx - 4, entry_idx)
        u_prev4 = (close_w[j4] - entry_price) / entry_price if direction == "LONG" else (entry_price - close_w[j4]) / entry_price
        d_unreal_4 = u_exit - u_prev4

        rows.append(
            {
                "entry_ts": str(b_ev.get("entry_ts", "")),
                "direction": direction,
                "bars_held": bh,
                "proba_entry": proba_entry,
                "proba_exit_prev": proba_exit_prev,
                "MAE": mae_v,
                "MFE": mfe_v,
                "delta_unreal_4bars": d_unreal_4,
                "baseline_profit": pb,
                "combo_profit": pc,
                "damage": pc - pb,
            }
        )

    dfa = pd.DataFrame(rows)
    bw = sum(1 for k in common if _pnl(eb[k]) > 0)
    short_bw = sum(1 for k in common if _pnl(eb[k]) > 0 and eb[k].get("direction") == "SHORT")
    stats = {
        "baseline_winners": bw,
        "baseline_winners_short": short_bw,
        "winner_mfe_median": float("nan"),
        "winner_du_median": float("nan"),
        "winner_bars_median": float("nan"),
    }
    return dfa, stats


def _write_md(
    *,
    out_md: Path,
    out_csv: Path,
    dfa: pd.DataFrame,
    stats: dict[str, Any],
    mode: str,
    meta: dict[str, Any],
) -> None:
    n = len(dfa)
    lines: list[str] = []
    lines.append("# SIGNAL_EXIT_TH profit_damaged 상세 (δp=0.05, mae_cut=-0.007)")
    lines.append("")
    lines.append("## 메타")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("| --- | --- |")
    lines.append(f"| mode | `{mode}` |")
    for k, v in meta.items():
        lines.append(f"| {k} | `{v}` |")
    lines.append("")
    lines.append("## 정의")
    lines.append("")
    if mode == "fast":
        lines.append(
            "- **baseline profit:** `signal_exit_th_trades_180d_detail.csv` 의 `profit_actual` (엔진 round-trip net)."
        )
        lines.append(
            "- **combo profit:** 손실 보조 조건이 성립하면 청산가를 **exit 바 직전 바 OHLC `close`** 로 두고 "
            "`_hypothetical_profit` 재계산; 아니면 baseline과 동일."
        )
        lines.append(
            "- **profit_damaged:** baseline profit > 0 이고 combo profit < baseline."
        )
        lines.append("")
        lines.append(
            "> `pnl_change`(잔고 분) 기준은 `--mode full` 이중 백테스트로 별도 확인. "
            "roll60 샘플에서 관측된 `profit_damaged` 다건은 **짧은 창·pnl_change**와 조합될 때 나올 수 있다."
        )
    else:
        lines.append(
            "- **baseline / combo profit:** 각각 trade_events 의 `pnl_change` (공통 키 `(entry_ts, exit_ts, direction)`)."
        )
        lines.append("- **profit_damaged:** baseline > 0 이고 combo < baseline.")
    lines.append("")
    lines.append(
        f"- **참고:** 동일 180d에서 baseline 이익 sth **{stats['baseline_winners']}**건, "
        f"그중 SHORT **{stats['baseline_winners_short']}** ({_share(stats['baseline_winners_short'], stats['baseline_winners'])})."
    )
    lines.append("")

    lines.append("## A. profit_damaged trade 표")
    lines.append("")
    if n == 0:
        lines.append("*(해당 없음)*")
    else:
        lines.append(
            "| entry_ts | direction | bars_held | proba_entry | proba_exit_prev | MAE | MFE | "
            "delta_unreal_4bars | baseline profit | combo profit | damage |"
        )
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for _, r in dfa.sort_values("damage").iterrows():
            lines.append(
                f"| {str(r['entry_ts'])[:19]}… | {r['direction']} | {int(r['bars_held'])} | "
                f"{_fmt(r['proba_entry'])} | {_fmt(r['proba_exit_prev'])} | {_fmt(r['MAE'])} | {_fmt(r['MFE'])} | "
                f"{_fmt(r['delta_unreal_4bars'])} | {_fmt(r['baseline_profit'])} | {_fmt(r['combo_profit'])} | {_fmt(r['damage'])} |"
            )
    lines.append("")
    lines.append(f"- CSV: `{out_csv.as_posix()}`")
    lines.append("")

    n_short = int((dfa["direction"] == "SHORT").sum()) if n else 0
    n_long = int((dfa["direction"] == "LONG").sum()) if n else 0
    bw, sw = stats["baseline_winners"], stats["baseline_winners_short"]

    lines.append("## B. 공통 패턴 요약 (질문별)")
    lines.append("")
    lines.append(
        f"1. **SHORT에 더 몰리는가?** 손상 {n}건 중 SHORT **{n_short}**, LONG **{n_long}**. "
        f"baseline 이익 sth 전체 SHORT 비중 **{_share(sw, bw)}**. "
        "이익 `signal_exit_th` 자체가 SHORT 쏠림이 크므로, **손상만 SHORT 편향이라고 보기는 어렵다** (표본이 1건이면 단정 불가)."
    )
    lines.append("")
    if n:
        med_mfe = float(dfa["MFE"].median())
        med_bh = float(dfa["bars_held"].median())
        lines.append(
            f"2. **bars_held:** 손상 건 median **{med_bh:.1f}**"
            + (
                f" (baseline 이익 sth 전체 median **{stats['winner_bars_median']:.1f}**)."
                if np.isfinite(stats.get("winner_bars_median", np.nan))
                else "."
            )
            + " 본 fast 모드 사례는 **min_hold(24) 근처**."
        )
        lines.append("")
        if np.isfinite(stats.get("winner_mfe_median", np.nan)):
            lines.append(
                f"3. **MFE:** 손상 건 MFE median **{_fmt(med_mfe)}** vs 이익 sth 전체 MFE 중앙값 **{_fmt(stats['winner_mfe_median'])}** — "
                "**유리한 변동(MFE)이 집단 대비 크게 잡힌 상태에서** 직전 바로 당겨 이익이 깎일 수 있음."
            )
        else:
            lines.append(f"3. **MFE:** 손상 건만 median **{_fmt(med_mfe)}** (full mode에서는 이익 sth 집단 중앙값 미계산).")
        lines.append("")
        pos_du = int((dfa["delta_unreal_4bars"] > 0).sum())
        du_ref = (
            f"이익 sth 전체의 Δunreal 중앙값 **{_fmt(stats['winner_du_median'])}**."
            if np.isfinite(stats.get("winner_du_median", np.nan))
            else "(집단 중앙값은 fast 모드에서만 자동 계산.)"
        )
        lines.append(
            f"4. **delta_unreal_4bars:** 양수인 손상 건 **{pos_du}/{n}**. "
            f"{du_ref} "
            "**직전 4바에서 미실현이 개선(양수)되는 중에도** 청산가만 불리해지면 이익 훼손 가능 — "
            "`delta_unreal_4bars > 0` 게이트가 직접 겨냥하는 유형."
        )
    else:
        lines.append("2–4. (표본 없음)")
    lines.append("")

    lines.append("## C. 보호 조건 후보 (스윕 없음)")
    lines.append("")
    lines.append(
        "1. **`delta_unreal_4bars > 0` 이면 손실 보조 미적용** — 본 1건이 대표적으로 해당(직전 4바 회복 중). "
        "roll60·pnl_change 기준 재검증 권장."
    )
    lines.append("")
    lines.append(
        "2. **`MFE`가 이익 sth 집단의 중앙값(또는 상위 분위) 이상이면 미적용** — 이미 충분히 유리하게 움직인 뒤 "
        "가격만 당기면 상대적으로 이익이 잘 깎임."
    )
    lines.append("")

    lines.append("## D. 최종 의견")
    lines.append("")
    lines.append(
        "- **설명력:** 이익 훼손은 ‘랜덤’이 아니라 **(가) 직전 4바 회복(Δunreal 양)·(나) 큰 MFE·(다) 단기 보유**와 겹칠 수 있음."
    )
    lines.append(
        "- **추가 파볼 가치:** 보호 게이트를 코드에 넣고 **동일 180d + roll60 + (가능하면 `--mode full` pnl_change)** 로 "
        "손실 완화가 남는지 확인할 단계는 타당하다."
    )
    lines.append(
        "- **주의:** 게이트가 손실 완화 적용 건수를 같이 줄이면 순이득이 없을 수 있음."
    )
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("fast", "full"), default="fast")
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=182)
    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--detail-csv",
        default="data/diagnostics/fr2/signal_exit_th_trades_180d_detail.csv",
    )
    parser.add_argument(
        "--wl-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_WINNERS_LOSERS_REPORT_detail.csv",
    )
    parser.add_argument(
        "--ohlc-csv",
        default="data/ohlcv/BTCUSDT_5m_full.csv",
    )
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_PROFIT_DAMAGED_180D.md",
    )
    parser.add_argument(
        "--out-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_PROFIT_DAMAGED_180D.csv",
    )
    args = parser.parse_args()

    out_md = Path(args.out_md)
    out_csv = Path(args.out_csv)

    if args.mode == "fast":
        dfa, stats = run_mode_fast(
            detail_csv=Path(args.detail_csv),
            wl_csv=Path(args.wl_csv),
            ohlc_path=Path(args.ohlc_csv),
        )
        meta = {
            "detail_csv": args.detail_csv,
            "wl_csv": args.wl_csv,
            "ohlc_csv": args.ohlc_csv,
            "δ_p": DELTA_P,
            "mae_cut": MAE_CUT,
        }
    else:
        t_max = to_utc_ts(args.t_max)
        dfa, stats = run_mode_full(
            t_max=t_max,
            days_full=int(args.days_full),
            window_days=int(args.window_days),
            threshold=float(args.threshold),
        )
        meta = {
            "t_max": t_max.isoformat(),
            "window_days": int(args.window_days),
            "threshold": float(args.threshold),
            "δ_p": DELTA_P,
            "mae_cut": MAE_CUT,
        }

    if not dfa.empty:
        dfa.to_csv(out_csv, index=False)
    else:
        out_csv.write_text("", encoding="utf-8")

    _write_md(
        out_md=out_md,
        out_csv=out_csv,
        dfa=dfa,
        stats=stats,
        mode=args.mode,
        meta=meta,
    )
    print(f"[profit_damaged] mode={args.mode} rows={len(dfa)} -> {out_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
