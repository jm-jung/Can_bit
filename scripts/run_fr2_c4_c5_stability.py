#!/usr/bin/env python3
"""
FR2 Gated Strategy: C4/C5 stability verification

실험:
- 30d block stability (최근 90d, C0/C1/C4/C5)
- trade quality (720d/180d/90d, C1/C4/C5)
- gate behaviour (720d/180d/90d, C4/C5)
- forward-style test (train 360d, test 60d rolling, C1/C4/C5)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from scripts.run_fr2_oos_diagnosis import get_fr2_proba  # type: ignore
from scripts.run_fr2_regime_conditioning import add_regime_columns  # type: ignore
from scripts.run_fr2_regime_filtered_backtest import (  # type: ignore
    entry_mask_uptrend_and_high_vol,
    entry_mask_mid_or_high_vol,
    entry_mask_no_filter,
)
from scripts.run_fr2_gated_strategy_backtest import (  # type: ignore
    run_gated_backtest,
    PERSISTENCE_BARS,
    ROLLING_TRADES,
    COOLDOWN_BARS_GATE,
    CONSEC_LOSSES,
)


def _load_720d():
    triple, err = get_fr2_proba(720)
    if err:
        raise RuntimeError(f"get_fr2_proba failed: {err}")
    df_bt, pl, ps, future_ret = triple
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    close = df_bt["close"].values.astype(float)
    pl = np.asarray(pl)
    ps = np.asarray(ps)
    future_ret = np.asarray(future_ret)

    df_bt = add_regime_columns(720, df_bt)
    regime_uhv = entry_mask_uptrend_and_high_vol(df_bt)
    regime_midhigh = entry_mask_mid_or_high_vol(df_bt)
    regime_none = entry_mask_no_filter(df_bt)

    return df_bt, close, pl, ps, future_ret, regime_uhv, regime_midhigh, regime_none


def _candidate_config(cand_id: str, regime_uhv, regime_midhigh, regime_none):
    # 재사용: C0/C1/C4/C5만
    if cand_id == "0":
        return "no_filter", regime_none, "none", {}, False, False, False, False
    if cand_id == "1":
        return "uptrend_and_high_vol", regime_uhv, "none", {}, False, False, False, False
    if cand_id == "4":
        return (
            "uptrend_and_high_vol",
            regime_uhv,
            "min_persistence_bars",
            {"persistence": 6},
            False,
            False,
            True,
            False,
        )
    if cand_id == "5":
        return (
            "uptrend_and_high_vol",
            regime_uhv,
            "rolling_cost_gate+consecutive_loss_gate",
            {"recent_trades": 50, "consec_losses": 3, "cooldown_bars": 24},
            True,
            True,
            False,
            False,
        )
    raise ValueError(f"Unsupported candidate {cand_id}")


def _run_window(df_bt, close, pl, ps, regime_masks, window_days: int, cand_ids):
    t_max = df_bt["timestamp"].max()
    if window_days == 720:
        idx = np.ones(len(df_bt), dtype=bool)
    else:
        start = t_max - pd.Timedelta(days=window_days)
        idx = (df_bt["timestamp"] >= start).values
    if idx.sum() < 500:
        return {}

    close_w = close[idx]
    pl_w = pl[idx]
    ps_w = ps[idx]
    ts_w = df_bt["timestamp"].values[idx]
    regime_uhv_w = regime_masks["uhv"][idx]
    regime_midhigh_w = regime_masks["midhigh"][idx]
    regime_none_w = regime_masks["none"][idx]

    out = {}
    for cid in cand_ids:
        base_name, _, gate_name, gate_params, use_cost, use_loss, use_pers, use_spear = _candidate_config(
            cid, regime_uhv_w, regime_midhigh_w, regime_none_w
        )
        if base_name == "no_filter":
            regime_mask = regime_none_w
        elif base_name == "mid_or_high_vol":
            regime_mask = regime_midhigh_w
        else:
            regime_mask = regime_uhv_w

        res = run_gated_backtest(
            close_w,
            pl_w,
            ps_w,
            regime_mask,
            use_persistence=use_pers,
            persistence_bars=gate_params.get("persistence", PERSISTENCE_BARS),
            use_rolling_cost_gate=use_cost,
            rolling_trades=gate_params.get("recent_trades", ROLLING_TRADES),
            cost_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
            use_consecutive_loss_gate=use_loss,
            consec_losses=gate_params.get("consec_losses", CONSEC_LOSSES),
            loss_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
            use_rolling_spearman_gate=use_spear,
            spearman_trades=gate_params.get("spearman_trades", 50),
            spearman_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
        )
        res["timestamps"] = ts_w
        res["regime_mask"] = regime_mask
        out[cid] = res
    return out


def _max_consecutive_losses(pnls: list[float]) -> int:
    m = 0
    cur = 0
    for x in pnls:
        if x < 0:
            cur += 1
            m = max(m, cur)
        else:
            cur = 0
    return m


def _profit_factor(pnls: list[float]) -> float:
    if not pnls:
        return float("nan")
    gains = [x for x in pnls if x > 0]
    losses = [x for x in pnls if x < 0]
    if not losses:
        return float("nan")
    return sum(gains) / abs(sum(losses)) if losses else float("nan")


def experiment_blocks(df_bt, close, pl, ps, regime_masks):
    # 최근 90d, C0/C1/C4/C5
    cand_ids = ["0", "1", "4", "5"]
    res_map = _run_window(df_bt, close, pl, ps, regime_masks, 90, cand_ids)
    if not res_map:
        return
    t_max = df_bt["timestamp"].max()
    blocks = []
    for cid, res in res_map.items():
        ts = res["timestamps"]
        trades = res["trades"]
        eq = res["equity_curve"]
        if len(ts) == 0:
            continue
        b_edges = [
            (t_max - pd.Timedelta(days=30), t_max, "block1_0_30"),
            (t_max - pd.Timedelta(days=60), t_max - pd.Timedelta(days=30), "block2_30_60"),
            (t_max - pd.Timedelta(days=90), t_max - pd.Timedelta(days=60), "block3_60_90"),
        ]
        for start, end, bname in b_edges:
            bar_idx = (ts >= start) & (ts < end)
            if not bar_idx.any():
                trades_block = []
                eq_block = None
            else:
                idxs = np.where(bar_idx)[0]
                eq_block = eq[idxs[0] : idxs[-1] + 1]
                trades_block = [t for t in trades if "entry_bar" in t and bar_idx[t["entry_bar"]]]

            pnls = [t["pnl"] for t in trades_block]
            trades_n = len(pnls)
            if trades_n:
                cost_on = float(np.prod([1.0 + x for x in pnls]) - 1.0)
                avg_pnl = float(np.mean(pnls))
                median_pnl = float(np.median(pnls))
                win_rate = float(np.mean([1.0 if x > 0 else 0.0 for x in pnls]))
                max_consec = _max_consecutive_losses(pnls)
            else:
                cost_on = float("nan")
                avg_pnl = float("nan")
                median_pnl = float("nan")
                win_rate = float("nan")
                max_consec = 0

            if eq_block is not None and len(eq_block):
                base = eq_block[0] if eq_block[0] != 0 else 1.0
                norm_eq = eq_block / base
                running_max = np.maximum.accumulate(norm_eq)
                dd = (norm_eq - running_max) / np.where(running_max > 0, running_max, 1.0)
                mdd = float(np.min(dd))
            else:
                mdd = float("nan")

            blocks.append(
                {
                    "candidate": cid,
                    "block": bname,
                    "trades": trades_n,
                    "cost_on": cost_on,
                    "MDD": mdd,
                    "avg_pnl_per_trade": avg_pnl,
                    "median_pnl_per_trade": median_pnl,
                    "win_rate": win_rate,
                    "max_consecutive_losses": max_consec,
                }
            )

    pd.DataFrame(blocks).to_csv(OUT_DIR / "fr2_c4_c5_stability_blocks.csv", index=False)


def experiment_trade_quality(df_bt, close, pl, ps, regime_masks):
    cand_ids = ["1", "4", "5"]
    rows = []
    for window_days in [720, 180, 90]:
        res_map = _run_window(df_bt, close, pl, ps, regime_masks, window_days, cand_ids)
        if not res_map:
            continue
        for cid, res in res_map.items():
            pnls = [t["pnl"] for t in res["trades"]]
            trades_n = len(pnls)
            if trades_n:
                avg_pnl = float(np.mean(pnls))
                median_pnl = float(np.median(pnls))
                win_rate = float(res["win_rate"]) if res["win_rate"] == res["win_rate"] else float("nan")
                max_consec = _max_consecutive_losses(pnls)
                pf = _profit_factor(pnls)
            else:
                avg_pnl = float("nan")
                median_pnl = float("nan")
                win_rate = float("nan")
                max_consec = 0
                pf = float("nan")
            rows.append(
                {
                    "candidate": cid,
                    "window_days": window_days,
                    "trades": trades_n,
                    "avg_pnl_per_trade": avg_pnl,
                    "median_pnl_per_trade": median_pnl,
                    "win_rate": win_rate,
                    "max_consecutive_losses": max_consec,
                    "profit_factor": pf,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_c4_c5_trade_quality.csv", index=False)


def experiment_gate_behavior(df_bt, close, pl, ps, regime_masks):
    cand_ids = ["4", "5"]
    rows = []
    for window_days in [720, 180, 90]:
        res_map = _run_window(df_bt, close, pl, ps, regime_masks, window_days, cand_ids)
        if not res_map:
            continue
        for cid, res in res_map.items():
            durs = res.get("gate_off_durations", []) or []
            avg_dur = float(np.mean(durs)) if durs else 0.0
            max_dur = float(max(durs)) if durs else 0.0
            rows.append(
                {
                    "candidate": cid,
                    "window_days": window_days,
                    "active_ratio": res.get("active_ratio", float("nan")),
                    "gate_trigger_count": res.get("gate_trigger_count", 0),
                    "avg_gate_off_duration": avg_dur,
                    "max_gate_off_duration": max_dur,
                }
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_c4_c5_gate_behavior.csv", index=False)


def experiment_forward(df_bt, close, pl, ps, regime_masks):
    cand_ids = ["1", "4", "5"]
    t = df_bt["timestamp"].values
    t_min = t.min()
    t_max = t.max()
    rows = []
    k = 0
    while True:
        train_start = t_min + pd.Timedelta(days=60) * k
        train_end = train_start + pd.Timedelta(days=360)
        test_end = train_end + pd.Timedelta(days=60)
        if test_end > t_max:
            break
        test_idx = (t >= train_end) & (t < test_end)
        if test_idx.sum() < 200:
            k += 1
            continue
        close_w = close[test_idx]
        pl_w = pl[test_idx]
        ps_w = ps[test_idx]
        regime_uhv_w = regime_masks["uhv"][test_idx]
        regime_midhigh_w = regime_masks["midhigh"][test_idx]
        regime_none_w = regime_masks["none"][test_idx]

        for cid in cand_ids:
            base_name, _, gate_name, gate_params, use_cost, use_loss, use_pers, use_spear = _candidate_config(
                cid, regime_uhv_w, regime_midhigh_w, regime_none_w
            )
            if base_name == "no_filter":
                regime_mask = regime_none_w
            elif base_name == "mid_or_high_vol":
                regime_mask = regime_midhigh_w
            else:
                regime_mask = regime_uhv_w
            res = run_gated_backtest(
                close_w,
                pl_w,
                ps_w,
                regime_mask,
                use_persistence=use_pers,
                persistence_bars=gate_params.get("persistence", PERSISTENCE_BARS),
                use_rolling_cost_gate=use_cost,
                rolling_trades=gate_params.get("recent_trades", ROLLING_TRADES),
                cost_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
                use_consecutive_loss_gate=use_loss,
                consec_losses=gate_params.get("consec_losses", CONSEC_LOSSES),
                loss_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
                use_rolling_spearman_gate=use_spear,
                spearman_trades=gate_params.get("spearman_trades", 50),
                spearman_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
            )
            pnls = [t["pnl"] for t in res["trades"]]
            trades_n = len(pnls)
            avg_pnl = float(np.mean(pnls)) if trades_n else float("nan")
            rows.append(
                {
                    "window_id": k,
                    "candidate": cid,
                    "test_start": train_end,
                    "test_end": test_end,
                    "trades": trades_n,
                    "cost_on": res["total_return"],
                    "MDD": res["max_drawdown"],
                    "avg_pnl_per_trade": avg_pnl,
                }
            )
        k += 1

    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_c4_c5_forward_test.csv", index=False)


def write_report():
    blocks_p = OUT_DIR / "fr2_c4_c5_stability_blocks.csv"
    tq_p = OUT_DIR / "fr2_c4_c5_trade_quality.csv"
    gb_p = OUT_DIR / "fr2_c4_c5_gate_behavior.csv"
    fw_p = OUT_DIR / "fr2_c4_c5_forward_test.csv"

    lines = ["# FR2 C4/C5 Stability Report\n"]

    if blocks_p.exists():
        df_b = pd.read_csv(blocks_p)
        lines.append("## 30d block stability (recent 90d)\n")
        lines.append(df_b.to_markdown(index=False))
        lines.append("\n")

    if tq_p.exists():
        df_q = pd.read_csv(tq_p)
        lines.append("## Trade quality (720d / 180d / 90d)\n")
        lines.append(df_q.to_markdown(index=False))
        lines.append("\n")

    if gb_p.exists():
        df_g = pd.read_csv(gb_p)
        lines.append("## Gate behaviour (C4/C5)\n")
        lines.append(df_g.to_markdown(index=False))
        lines.append("\n")

    if fw_p.exists():
        df_f = pd.read_csv(fw_p)
        lines.append("## Forward-style test (train 360d, test 60d)\n")
        lines.append(df_f.to_markdown(index=False))
        lines.append("\n")

    lines.append("## Statistical cautions\n")
    lines.append("- C4 90d block에서 trades < 100 인 경우 sample_too_small 로 해석해야 함.\n")
    lines.append("- forward windows 수가 한정적이므로 개별 window의 양/음수는 과해석 금지.\n")

    (OUT_DIR / "FR2_C4_C5_STABILITY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[FR2-C4C5] Loading 720d...", flush=True)
    df_bt, close, pl, ps, future_ret, regime_uhv, regime_midhigh, regime_none = _load_720d()
    regime_masks = {"uhv": regime_uhv, "midhigh": regime_midhigh, "none": regime_none}

    print("[FR2-C4C5] Experiment 1: 30d blocks", flush=True)
    experiment_blocks(df_bt, close, pl, ps, regime_masks)

    print("[FR2-C4C5] Experiment 2: trade quality", flush=True)
    experiment_trade_quality(df_bt, close, pl, ps, regime_masks)

    print("[FR2-C4C5] Experiment 3: gate behaviour", flush=True)
    experiment_gate_behavior(df_bt, close, pl, ps, regime_masks)

    print("[FR2-C4C5] Experiment 4: forward-style test", flush=True)
    experiment_forward(df_bt, close, pl, ps, regime_masks)

    print("[FR2-C4C5] Writing report", flush=True)
    write_report()
    print("[FR2-C4C5] Done.", flush=True)


if __name__ == "__main__":
    main()

