"""Shadow equity accounting — observation only, reuse forensic backtest identity."""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from canbit_equity.backtest import run_backtest, summarize_equity
from canbit_equity.regime.prospective.config import COST_BASE, COST_HIGH, COST_LOW, PROSP_ROOT
from canbit_equity.strategies import signal_buy_and_hold, signal_dual_trend


def _empty_shadow() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "effective_session",
            "previous_session",
            "previous_position",
            "target_position_at_open",
            "open_adj",
            "close_adj",
            "previous_close_adj",
            "overnight_return",
            "intraday_return",
            "gross_strategy_return",
            "turnover",
            "cost",
            "net_strategy_return",
            "equity",
            "drawdown",
            "signal_source_session",
            "provenance_tier",
            "strategy_id",
            "cost_bps",
        ]
    )


def update_shadow_from_predictions(
    live: pd.DataFrame,
    strict: pd.DataFrame,
    late: pd.DataFrame,
) -> Dict[str, Any]:
    preds = pd.concat([strict, late], ignore_index=True) if (len(strict) or len(late)) else pd.DataFrame()
    if len(preds) == 0:
        for name in ("candidate_daily_equity", "dual_daily_equity", "buy_hold_daily_equity"):
            path = PROSP_ROOT / f"shadow/{name}.parquet"
            if not path.exists():
                _empty_shadow().to_parquet(path, index=False)
        return {"status": "NO_PREDICTIONS", "rows": 0}

    preds = preds.copy()
    preds["signal_session"] = pd.to_datetime(preds["signal_session"]).dt.normalize()
    live = live.copy()
    live["session_date"] = pd.to_datetime(live["session_date"]).dt.normalize()

    start = preds["signal_session"].min()
    panel = live[live["session_date"] >= start].sort_values("session_date").reset_index(drop=True)
    if len(panel) == 0:
        return {"status": "NO_EFFECTIVE_SESSIONS", "rows": 0}

    pred_by_sig = preds.drop_duplicates("signal_session", keep="last").set_index("signal_session")
    cand_signal = pd.Series(0.0, index=panel.index)
    dual_signal = pd.Series(0.0, index=panel.index)
    for i, sess in enumerate(panel["session_date"]):
        if sess in pred_by_sig.index:
            cand_signal.iloc[i] = float(pred_by_sig.loc[sess, "target_position"])
            dual_val = pred_by_sig.loc[sess, "dual_target_position"]
            if dual_val is None or (isinstance(dual_val, float) and pd.isna(dual_val)):
                dual_signal.iloc[i] = float(signal_dual_trend(panel.iloc[[i]]).iloc[0])
            else:
                dual_signal.iloc[i] = float(dual_val)
        else:
            # no new signal → FLAT target for that close (position continuity handled by backtest shift)
            cand_signal.iloc[i] = 0.0
            dual_signal.iloc[i] = float(signal_dual_trend(panel.iloc[[i]]).iloc[0])

    results: Dict[str, Any] = {}
    for cost_name, cost in (("BASE", COST_BASE), ("LOW", COST_LOW), ("HIGH", COST_HIGH)):
        bt_c = run_backtest(panel, cand_signal, cost_bps_per_side=cost)
        bt_d = run_backtest(panel, dual_signal, cost_bps_per_side=cost)
        bt_b = run_backtest(panel, signal_buy_and_hold(panel), cost_bps_per_side=cost)
        results[cost_name] = {
            "candidate": summarize_equity(bt_c),
            "dual": summarize_equity(bt_d),
            "buy_hold": summarize_equity(bt_b),
        }
        if cost_name == "BASE":
            _write_strategy_shadow(panel, bt_c, pred_by_sig, "LOGISTIC_PRICE_PLUS_ALL_REGIME", cost, "candidate_daily_equity")
            _write_strategy_shadow(panel, bt_d, pred_by_sig, "DUAL_TREND_FILTER", cost, "dual_daily_equity")
            _write_strategy_shadow(panel, bt_b, pred_by_sig, "BUY_AND_HOLD", cost, "buy_hold_daily_equity")
            exec_rows = [
                {
                    "effective_session": panel.loc[i, "session_date"],
                    "strategy_id": "LOGISTIC_PRICE_PLUS_ALL_REGIME",
                    "net_return": float(bt_c["net_ret"].iloc[i]),
                    "position": float(bt_c["position"].iloc[i]),
                    "cost_bps": cost,
                }
                for i in range(len(panel))
            ]
            pd.DataFrame(exec_rows).to_parquet(PROSP_ROOT / "executions/shadow_executions.parquet", index=False)

    return {"status": "UPDATED", "rows": int(len(panel)), "metrics": results}


def _write_strategy_shadow(
    panel: pd.DataFrame,
    bt: Dict[str, Any],
    pred_by_sig: pd.DataFrame,
    strategy_id: str,
    cost: float,
    filename: str,
) -> None:
    eq = bt["equity"].astype(float)
    peak = eq.cummax()
    dd = eq / peak - 1.0
    pos = bt["position"].astype(float)
    prev_pos = pos.shift(1).fillna(0.0)
    open_ = panel["open_adj"].astype(float)
    close = panel["close_adj"].astype(float)
    prev_close = close.shift(1)
    overnight = open_ / prev_close - 1.0
    intraday = close / open_ - 1.0

    rows = []
    for i in range(len(panel)):
        sess = panel.loc[i, "session_date"]
        # position at open of sess comes from prior session signal
        prev_sess = panel.loc[i - 1, "session_date"] if i else None
        src = prev_sess if prev_sess is not None and prev_sess in pred_by_sig.index else None
        tier = pred_by_sig.loc[src, "provenance_tier"] if src is not None and "provenance_tier" in pred_by_sig.columns else None
        rows.append(
            {
                "effective_session": sess,
                "previous_session": prev_sess,
                "previous_position": float(prev_pos.iloc[i]),
                "target_position_at_open": float(pos.iloc[i]),
                "open_adj": float(open_.iloc[i]),
                "close_adj": float(close.iloc[i]),
                "previous_close_adj": float(prev_close.iloc[i]) if pd.notna(prev_close.iloc[i]) else None,
                "overnight_return": float(overnight.iloc[i]) if pd.notna(overnight.iloc[i]) else None,
                "intraday_return": float(intraday.iloc[i]),
                "gross_strategy_return": float(bt["gross_ret"].iloc[i]),
                "turnover": float(bt["turnover"].iloc[i]),
                "cost": float(bt["turnover"].iloc[i] * (cost / 10000.0)),
                "net_strategy_return": float(bt["net_ret"].iloc[i]),
                "equity": float(eq.iloc[i]),
                "drawdown": float(dd.iloc[i]),
                "signal_source_session": src,
                "provenance_tier": tier,
                "strategy_id": strategy_id,
                "cost_bps": cost,
            }
        )
    pd.DataFrame(rows).to_parquet(PROSP_ROOT / f"shadow/{filename}.parquet", index=False)
