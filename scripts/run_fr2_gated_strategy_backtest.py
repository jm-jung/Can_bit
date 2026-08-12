#!/usr/bin/env python3
"""
FR2 Gated Strategy: uptrend_and_high_vol + 운영 게이트(rolling_cost, consecutive_loss, persistence, rolling_spearman).

Bar-by-bar backtest로 게이트를 시계열 순서대로 적용(미래 데이터 누수 없음).
기존 run_fr2_regime_filtered_backtest의 regime filter 재사용.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMMISSION = 0.0009
SLIPPAGE = 0.0001
MIN_HOLD = 36
COOLDOWN = 12
THRESHOLD = 0.60

# Gate defaults
COOLDOWN_BARS_GATE = 24
ROLLING_TRADES = 50
CONSEC_LOSSES = 3
PERSISTENCE_BARS = 6

from scripts.run_fr2_oos_diagnosis import get_fr2_proba, _metrics
from scripts.run_fr2_regime_conditioning import add_regime_columns
from scripts.run_fr2_regime_filtered_backtest import (
    entry_mask_uptrend_and_high_vol,
    entry_mask_mid_or_high_vol,
    entry_mask_high_vol_only,
    entry_mask_no_filter,
)


def persistence_mask(regime_mask: np.ndarray, P: int) -> np.ndarray:
    """True where regime has been True for the last P bars (no lookahead)."""
    n = len(regime_mask)
    out = np.zeros(n, dtype=bool)
    for i in range(P - 1, n):
        if np.all(regime_mask[i - P + 1 : i + 1]):
            out[i] = True
    return out


def run_gated_backtest(
    close: np.ndarray,
    pl: np.ndarray,
    ps: np.ndarray,
    regime_mask: np.ndarray,
    *,
    use_persistence: bool = False,
    persistence_bars: int = 6,
    use_rolling_cost_gate: bool = False,
    rolling_trades: int = 50,
    cost_cooldown_bars: int = 24,
    use_consecutive_loss_gate: bool = False,
    consec_losses: int = 3,
    loss_cooldown_bars: int = 24,
    use_rolling_spearman_gate: bool = False,
    spearman_trades: int = 50,
    spearman_cutoff: float = 0.0,
    spearman_cooldown_bars: int = 24,
) -> dict:
    """
    Bar-by-bar backtest with optional operational gates. No lookahead.
    Entry: max(pl,ps) >= THRESHOLD, direction = argmax(long/short), regime + persistence + gates allow.
    """
    n = len(close)
    if n < MIN_HOLD + 10:
        return {"total_return": np.nan, "max_drawdown": np.nan, "total_trades": 0, "win_rate": np.nan, "trades": [], "active_bars": 0, "entries_allowed": 0}

    # Pre-compute persistence (no lookahead)
    if use_persistence and persistence_bars > 0:
        pers_mask = persistence_mask(regime_mask, persistence_bars)
    else:
        pers_mask = np.ones(n, dtype=bool)

    # State
    position = None  # {"side": 1 or -1, "entry_bar": int, "entry_price": float}
    cooldown_left = 0
    gate_off_until = 0  # bar index until which gates block
    gate_trigger_count = 0
    gate_block_start = None
    gate_block_durations: list[int] = []
    trades_pnl: list[float] = []
    trades_signal: list[float] = []  # pl-ps at entry
    consec_loss = 0
    balance = 1.0
    equity = [1.0]
    trades: list[dict] = []

    i = 0
    while i < n:
        # if a gate-off period ends at this bar, close its duration
        if gate_block_start is not None and i >= gate_off_until:
            gate_block_durations.append(i - gate_block_start)
            gate_block_start = None
        if position is not None:
            hold_bars = i - position["entry_bar"]
            if hold_bars >= MIN_HOLD:
                # Exit at bar i
                entry_price = position["entry_price"]
                exit_price = float(close[i])
                side = position["side"]
                if side == 1:  # long
                    gross = (exit_price - entry_price) / entry_price
                else:
                    gross = (entry_price - exit_price) / entry_price
                entry_cost = entry_price * (COMMISSION + SLIPPAGE)
                exit_cost = exit_price * (COMMISSION + SLIPPAGE)
                if side == 1:
                    net = (exit_price - exit_cost - (entry_price + entry_cost)) / (entry_price + entry_cost)
                else:
                    net = (entry_price - entry_cost - (exit_price + exit_cost)) / (entry_price - entry_cost)
                balance *= 1 + net
                equity.append(balance)
                trades.append({"pnl": net, "signal": position["signal_strength"], "entry_bar": position["entry_bar"]})
                trades_pnl.append(net)
                trades_signal.append(position["signal_strength"])

                # Update gates (no lookahead: only past trades)
                if net < 0:
                    consec_loss += 1
                    if use_consecutive_loss_gate and consec_loss >= consec_losses:
                        new_until = i + loss_cooldown_bars
                        if new_until > gate_off_until:
                            if i >= gate_off_until:
                                gate_block_start = i
                            gate_trigger_count += 1
                            gate_off_until = new_until
                else:
                    consec_loss = 0

                if use_rolling_cost_gate and len(trades_pnl) >= rolling_trades:
                    recent_cost = sum(trades_pnl[-rolling_trades:])
                    if recent_cost < 0:
                        new_until = i + cost_cooldown_bars
                        if new_until > gate_off_until:
                            if i >= gate_off_until:
                                gate_block_start = i
                            gate_trigger_count += 1
                            gate_off_until = new_until

                if use_rolling_spearman_gate and len(trades_pnl) >= spearman_trades:
                    sigs = trades_signal[-spearman_trades:]
                    pnls = trades_pnl[-spearman_trades:]
                    if np.std(sigs) > 1e-12 and np.std(pnls) > 1e-12:
                        sp = np.corrcoef(sigs, pnls)[0, 1]
                        if sp <= spearman_cutoff:
                            new_until = i + spearman_cooldown_bars
                            if new_until > gate_off_until:
                                if i >= gate_off_until:
                                    gate_block_start = i
                                gate_trigger_count += 1
                                gate_off_until = new_until

                position = None
                cooldown_left = COOLDOWN
            i += 1
            continue

        if cooldown_left > 0:
            cooldown_left -= 1
            equity.append(balance)
            i += 1
            continue

        # No position: check entry
        allow_regime = regime_mask[i] and pers_mask[i]
        allow_gate = i >= gate_off_until
        if not (allow_regime and allow_gate):
            equity.append(balance)
            i += 1
            continue

        p_flat = 1.0 - pl[i] - ps[i]
        p_flat = max(0.0, min(1.0, p_flat))
        mx = max(pl[i], ps[i], p_flat)
        if mx < THRESHOLD:
            equity.append(balance)
            i += 1
            continue

        if pl[i] >= ps[i] and pl[i] >= p_flat:
            side = 1
            signal_strength = pl[i] - ps[i]
        elif ps[i] >= p_flat:
            side = -1
            signal_strength = ps[i] - pl[i]
        else:
            equity.append(balance)
            i += 1
            continue

        position = {"side": side, "entry_bar": i, "entry_price": float(close[i]), "signal_strength": signal_strength}
        equity.append(balance)
        i += 1

    if position is not None:
        i = n - 1
        entry_price = position["entry_price"]
        exit_price = float(close[i])
        side = position["side"]
        if side == 1:
            gross = (exit_price - entry_price) / entry_price
        else:
            gross = (entry_price - exit_price) / entry_price
        entry_cost = entry_price * (COMMISSION + SLIPPAGE)
        exit_cost = exit_price * (COMMISSION + SLIPPAGE)
        if side == 1:
            net = (exit_price - exit_cost - (entry_price + entry_cost)) / (entry_price + entry_cost)
        else:
            net = (entry_price - entry_cost - (exit_price + exit_cost)) / (entry_price - entry_cost)
        balance *= 1 + net
        trades_pnl.append(net)
        trades_signal.append(position["signal_strength"])
        trades.append({"pnl": net, "signal": position["signal_strength"], "entry_bar": position["entry_bar"]})

    # close last gate-off block if still active
    if gate_block_start is not None and gate_off_until > gate_block_start:
        gate_block_durations.append(min(n, gate_off_until) - gate_block_start)

    total_return = balance - 1.0
    total_trades = len(trades)
    if total_trades > 0:
        wins = sum(1 for t in trades if t["pnl"] > 0)
        win_rate = wins / total_trades
    else:
        win_rate = np.nan

    eq = np.array(equity)
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / np.where(running_max > 0, running_max, 1.0)
    max_drawdown = float(np.min(dd)) if len(dd) else 0.0

    entries_allowed = int((regime_mask & pers_mask).sum()) if use_persistence else int(regime_mask.sum())
    active_ratio = entries_allowed / n if n else 0.0

    return {
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "total_trades": total_trades,
        "win_rate": win_rate,
        "trades": trades,
        "equity_curve": eq,
        "active_ratio": active_ratio,
        "entries_allowed": entries_allowed,
        "gate_trigger_count": gate_trigger_count,
        "gate_off_durations": gate_block_durations,
    }


def main():
    parser = argparse.ArgumentParser(description="FR2 gated strategy backtest")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate reports from existing CSV")
    parser.add_argument("--quick", action="store_true", help="90d only, candidates 0,1,2 (pipeline check)")
    args = parser.parse_args()

    quick = getattr(args, "quick", False)
    load_days = 90 if quick else 720
    windows_use = [90] if quick else [720, 180, 90]

    if getattr(args, "reports_only", False):
        write_reports()
        write_decision()
        print("[FR2-Gated] Reports only: done.", flush=True)
        return

    rows = []
    print(f"[FR2-Gated] Loading {load_days}d...", flush=True)
    triple, err = get_fr2_proba(load_days)
    if err:
        print("[FR2-Gated] Failed:", err, flush=True)
        pd.DataFrame(columns=["model_id", "base_filter", "gate_name", "gate_params", "window_days", "trades", "cost_on", "MDD", "win_rate", "sample_too_small", "entries_allowed", "active_ratio"]).to_csv(OUT_DIR / "fr2_gated_strategy_backtest.csv", index=False)
        write_reports()
        write_decision()
        return

    df_bt, pl, ps, future_ret = triple
    pl = np.asarray(pl)
    ps = np.asarray(ps)
    future_ret = np.asarray(future_ret)
    df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
    close = df_bt["close"].values.astype(float)
    t_max = df_bt["timestamp"].max()

    df_bt = add_regime_columns(load_days, df_bt)
    regime_uhv = entry_mask_uptrend_and_high_vol(df_bt)
    regime_midhigh = entry_mask_mid_or_high_vol(df_bt)
    regime_none = entry_mask_no_filter(df_bt)

    if quick:
        candidates = [
            ("0", "no_filter", regime_none, "none", {}, False, False, False, False),
            ("1", "uptrend_and_high_vol", regime_uhv, "none", {}, False, False, False, False),
            ("2", "uptrend_and_high_vol", regime_uhv, "rolling_cost_gate", {"recent_trades": 50, "cooldown_bars": 24}, True, False, False, False),
        ]
    else:
        candidates = [
            ("0", "no_filter", regime_none, "none", {}, False, False, False, False),
            ("1", "uptrend_and_high_vol", regime_uhv, "none", {}, False, False, False, False),
            ("2", "uptrend_and_high_vol", regime_uhv, "rolling_cost_gate", {"recent_trades": 50, "cooldown_bars": 24}, True, False, False, False),
            ("3", "uptrend_and_high_vol", regime_uhv, "consecutive_loss_gate", {"consec_losses": 3, "cooldown_bars": 24}, False, True, False, False),
            ("4", "uptrend_and_high_vol", regime_uhv, "min_persistence_bars", {"persistence": 6}, False, False, True, False),
            ("5", "uptrend_and_high_vol", regime_uhv, "rolling_cost_gate+consecutive_loss_gate", {"recent_trades": 50, "consec_losses": 3, "cooldown_bars": 24}, True, True, False, False),
            ("6", "uptrend_and_high_vol", regime_uhv, "rolling_spearman_gate", {"spearman_trades": 50, "cooldown_bars": 24}, False, False, False, True),
            ("7", "mid_or_high_vol", regime_midhigh, "rolling_cost_gate", {"recent_trades": 50, "cooldown_bars": 24}, True, False, False, False),
        ]

    for window_days in windows_use:
        if window_days == 720:
            close_w = close
            pl_w = pl
            ps_w = ps
            future_ret_w = future_ret
            regime_uhv_w = regime_uhv
            regime_midhigh_w = regime_midhigh
            regime_none_w = regime_none
        else:
            start = t_max - pd.Timedelta(days=window_days)
            idx = (df_bt["timestamp"] >= start).values
            if idx.sum() < 500:
                continue
            close_w = close[idx]
            pl_w = pl[idx]
            ps_w = ps[idx]
            future_ret_w = future_ret[idx]
            regime_uhv_w = regime_uhv[idx]
            regime_midhigh_w = regime_midhigh[idx]
            regime_none_w = regime_none[idx]

        for cand_id, base_name, _, gate_name, gate_params, use_cost, use_loss, use_pers, use_spearman in candidates:
            if base_name == "no_filter":
                regime_mask = regime_none_w
            elif base_name == "mid_or_high_vol":
                regime_mask = regime_midhigh_w
            else:
                regime_mask = regime_uhv_w

            res = run_gated_backtest(
                close_w, pl_w, ps_w, regime_mask,
                use_persistence=use_pers,
                persistence_bars=gate_params.get("persistence", PERSISTENCE_BARS),
                use_rolling_cost_gate=use_cost,
                rolling_trades=gate_params.get("recent_trades", ROLLING_TRADES),
                cost_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
                use_consecutive_loss_gate=use_loss,
                consec_losses=gate_params.get("consec_losses", CONSEC_LOSSES),
                loss_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
                use_rolling_spearman_gate=use_spearman,
                spearman_trades=gate_params.get("spearman_trades", 50),
                spearman_cooldown_bars=gate_params.get("cooldown_bars", COOLDOWN_BARS_GATE),
            )
            n_allowed = int(regime_mask.sum())
            fr_sub = future_ret_w[regime_mask]
            met = _metrics(pl_w[regime_mask], ps_w[regime_mask], fr_sub, threshold_signal=THRESHOLD) if n_allowed > 20 else {}
            row = {
                "model_id": "h15_micro_v1",
                "candidate": cand_id,
                "base_filter_name": base_name,
                "gate_name": gate_name,
                "gate_params": str(gate_params),
                "window_days": window_days,
                "trades": res["total_trades"],
                "entries_allowed": res["entries_allowed"],
                "signal_density": met.get("signal_density", np.nan),
                "direction_accuracy": met.get("direction_accuracy", np.nan),
                "spearman": met.get("spearman", np.nan),
                "mean_return_signal": met.get("mean_return_signal", np.nan),
                "cost_on": res["total_return"],
                "MDD": res["max_drawdown"],
                "win_rate": res["win_rate"],
                "sample_too_small": "yes" if res["total_trades"] < 100 else "no",
                "active_ratio": res["active_ratio"],
            }
            rows.append(row)
            print(f"  Cand{cand_id} {base_name} + {gate_name} {window_days}d -> trades={res['total_trades']} cost_on={res['total_return']:.4f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_gated_strategy_backtest.csv", index=False)
    print("[FR2-Gated] Wrote fr2_gated_strategy_backtest.csv", flush=True)
    write_reports()
    write_decision()
    print("[FR2-Gated] Done.", flush=True)


def write_reports():
    p = OUT_DIR / "fr2_gated_strategy_backtest.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = [
        "# FR2 Gated Strategy Backtest Report\n",
        "Base: uptrend_and_high_vol (or no_filter / mid_or_high_vol). Gates: rolling_cost, consecutive_loss, min_persistence, rolling_spearman. Bar-by-bar, no lookahead.\n",
        "| candidate | base_filter | gate_name | window_days | trades | cost_on | MDD | win_rate | active_ratio | sample_too_small |",
        "|-----------|-------------|-----------|-------------|--------|---------|-----|----------|--------------|------------------|",
    ]
    for _, r in df.iterrows():
        lines.append(f"| {r['candidate']} | {r['base_filter_name']} | {r['gate_name']} | {r['window_days']} | {r['trades']} | {float(r['cost_on']):.4f} | {float(r['MDD']):.4f} | {float(r['win_rate']) if pd.notna(r['win_rate']) else ''} | {float(r['active_ratio']) if pd.notna(r.get('active_ratio')) else ''} | {r.get('sample_too_small','')} |")
    (OUT_DIR / "FR2_GATED_STRATEGY_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-Gated] Wrote FR2_GATED_STRATEGY_REPORT.md", flush=True)


def write_decision():
    p = OUT_DIR / "fr2_gated_strategy_backtest.csv"
    if not p.exists():
        (OUT_DIR / "FR2_GATED_STRATEGY_DECISION.md").write_text("# FR2 Gated Strategy Decision\n\nNo data. Run script first.\n", encoding="utf-8")
        return
    df = pd.read_csv(p)
    base_1_90 = df[(df["candidate"] == "1") & (df["window_days"] == 90)]
    base_1_180 = df[(df["candidate"] == "1") & (df["window_days"] == 180)]
    best_90 = df[df["window_days"] == 90].sort_values("cost_on", ascending=False)
    best_90_row = best_90.iloc[0] if len(best_90) else None
    c1_90 = float(base_1_90["cost_on"].iloc[0]) if len(base_1_90) and pd.notna(base_1_90["cost_on"].iloc[0]) else None
    c1_180 = float(base_1_180["cost_on"].iloc[0]) if len(base_1_180) and pd.notna(base_1_180["cost_on"].iloc[0]) else None

    verdict = "CONDITIONAL_EDGE_REQUIRES_GATING"
    if best_90_row is not None and c1_90 is not None:
        best_c90 = float(best_90_row["cost_on"])
        if pd.isna(best_c90):
            verdict = "STILL_NOT_ROBUST"
        elif best_c90 > 0.006 and best_c90 > c1_90:
            best_c180 = df[(df["candidate"] == best_90_row["candidate"]) & (df["window_days"] == 180)]
            if len(best_c180) and pd.notna(best_c180["cost_on"].iloc[0]) and float(best_c180["cost_on"].iloc[0]) > (c1_180 or -1):
                verdict = "FILTERED_STRATEGY_CANDIDATE"
            else:
                verdict = "CONDITIONAL_EDGE_REQUIRES_GATING"
        elif best_c90 > c1_90:
            verdict = "CONDITIONAL_EDGE_REQUIRES_GATING"
        elif best_c90 <= -0.1:
            verdict = "STILL_NOT_ROBUST"

    sections = [
        "# FR2 Gated Strategy Decision\n",
        "## What was tested\n",
        "Candidates 0–7: no_filter, uptrend_and_high_vol (base), + rolling_cost_gate, + consecutive_loss_gate, + min_persistence_bars, + cost+consec, + rolling_spearman_gate, mid_or_high_vol + rolling_cost. Windows: 720d, 180d, 90d.\n",
        "## Base vs gated (90d)\n",
        f"- Candidate 1 (uptrend_and_high_vol only) 90d cost_on: {c1_90}\n",
        f"- Best 90d cost_on: {float(best_90_row['cost_on']) if best_90_row is not None else 'N/A'} (Candidate {best_90_row['candidate'] if best_90_row is not None else ''})\n",
        "## Key findings\n",
        "- 개선: gated 후보가 base(1) 대비 90d/180d cost_on 개선 시 CONDITIONAL_EDGE 또는 FILTERED_STRATEGY_CANDIDATE.\n",
        "- 미개선: 90d 여전히 음수·trades 급감만 있으면 STILL_NOT_ROBUST.\n",
        "\n## Final recommendation\n",
        f"**{verdict}**\n",
        "(FILTERED_STRATEGY_CANDIDATE | CONDITIONAL_EDGE_REQUIRES_GATING | STILL_NOT_ROBUST | RESEARCH_SIGNAL_ONLY)\n",
    ]
    (OUT_DIR / "FR2_GATED_STRATEGY_DECISION.md").write_text("".join(sections), encoding="utf-8")
    print("[FR2-Gated] Wrote FR2_GATED_STRATEGY_DECISION.md", flush=True)


if __name__ == "__main__":
    main()
