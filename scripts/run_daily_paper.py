"""
Daily Paper Ops: 확정 파이프라인 + 가상 포지션 lifecycle (실제 주문 없음).

Usage:
    python -m scripts.run_daily_paper [--dry-run] [--no-discord]
    python -m scripts.run_daily_paper --reset-state
    python -m scripts.reset_paper_state

절대 live 주문 API를 호출하지 않음. mode=paper 고정.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ops.execution_router import ExecutionRouter
from src.ops.monitor import Monitor
from src.ops.risk_manager import RiskLimits, RiskManager
from src.ops.state_manager import StateManager, TradeRecord
from src.ops.trading_engine import TickContext, TradingEngine
from src.strategy_filters.regime_ablation import (
    compute_positive_regime_allow_mask,
    compute_regime_components,
)

from scripts.run_daily_shadow import (
    ALLOW_LAG_MINUTES_DEFAULT,
    FALLBACK_PROBA,
    PROBA_CACHE_DEFAULT_PATH,
    RAW_SIGNAL_ENTROPY_THRESHOLD,
    RECENT_WINDOW_HOURS,
    _entropy_nl,
    _load_and_align_proba,
    compute_recent_window_summary,
    diagnose_proba_source,
    load_ohlcv,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_paper")

MONITORING_DIR = Path("data/monitoring")
PAPER_STATE_FILE = Path("data/state/paper_trading_state.json")
PAPER_STATE_ARCHIVE_DIR = Path("data/state/archive")

PAPER_POSITION_SIZE = 0.05
PAPER_MAX_HOLDING_BARS = 12
PAPER_FEE_RATE = 0.0004
PAPER_SLIPPAGE_RATE = 0.0002
POSITIVE_FILTER_MODE = "allow_ema_below_or_strong_up"


def reset_paper_observation_state(
    state_path: Path = PAPER_STATE_FILE,
    archive_dir: Path = PAPER_STATE_ARCHIVE_DIR,
) -> Dict[str, Any]:
    """
    정식 Paper 관찰 시작 전 state 초기화.
    기존 파일은 archive로 백업 후 equity=1.0 등으로 재설정.
  """
    state_path = Path(state_path)
    archive_dir = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    backup_path: Optional[str] = None
    if state_path.exists():
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = str(archive_dir / f"paper_trading_state_{ts}.json")
        shutil.copy2(state_path, backup_path)

    today = date.today().isoformat()
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fresh: Dict[str, Any] = {
        "mode": "paper",
        "initial_equity": 1.0,
        "equity": 1.0,
        "daily_pnl": 0.0,
        "daily_return": 0.0,
        "daily_date": today,
        "peak_equity": 1.0,
        "cumulative_return": 0.0,
        "max_drawdown": 0.0,
        "total_trades": 0,
        "total_wins": 0,
        "consecutive_losses": 0,
        "last_trade_time": "",
        "activation_on_count": 0,
        "activation_off_count": 0,
        "kill_switch_active": False,
        "kill_switch_reason": "",
        "shadow_start_date": "",
        "paper_start_date": today,
        "trades": [],
        "trade_history": [],
        "open_position": None,
        "errors": [],
        "observation_reset_at": now_iso,
        "observation_reset_note": "formal paper observation day 1 baseline",
    }

    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fresh, indent=2, ensure_ascii=False), encoding="utf-8")
    shutil.move(str(tmp), str(state_path))

    return {
        "state_path": str(state_path),
        "backup_path": backup_path,
        "reset_at": now_iso,
        "paper_start_date": today,
    }


def _print_reset_summary(result: Dict[str, Any]) -> None:
    print("")
    print("=== Paper State Reset (정식 관찰 1일차 baseline) ===")
    if result.get("backup_path"):
        print(f"  백업: {result['backup_path']}")
    else:
        print("  백업: (기존 state 없음 — 신규 생성)")
    print(f"  state: {result['state_path']}")
    print(f"  reset_at: {result['reset_at']}")
    print(f"  paper_start_date: {result['paper_start_date']}")
    print("  equity=1.0 | cumulative_return=0 | daily_return=0")
    print("  trades=[] | open_position=None | kill_switch=false")
    print("  (Discord 미전송 — 다음 run_daily_paper부터 관찰 집계)")
    print("")


class PaperMonitor(Monitor):
    def _get_log_path(self) -> Path:
        return self._dir / f"paper_trading_log_{date.today().strftime('%Y%m%d')}.jsonl"


class PaperStateManager(StateManager):
    """equity *= (1 + net_return * position_size); shadow state와 분리."""

    def __init__(
        self,
        state_path: Path = PAPER_STATE_FILE,
        position_size: float = PAPER_POSITION_SIZE,
        fee_rate: float = PAPER_FEE_RATE,
        slippage_rate: float = PAPER_SLIPPAGE_RATE,
    ):
        super().__init__(state_path)
        self.position_size = float(position_size)
        self.fee_rate = float(fee_rate)
        self.slippage_rate = float(slippage_rate)
        st = self.state
        st.mode = "paper"
        if st.initial_equity <= 0:
            st.initial_equity = 1.0
        if st.equity <= 0:
            st.equity = st.initial_equity
        if st.peak_equity <= 0:
            st.peak_equity = st.equity
        self.reset_daily()
        self._daily_start_equity = float(st.equity)
        self.save()

    def net_return_from_raw(self, raw_return: float) -> float:
        return float(raw_return) - 2.0 * (self.fee_rate + self.slippage_rate)

    def record_trade(self, trade: TradeRecord) -> None:
        st = self.state
        net_ret = self.net_return_from_raw(float(trade.pnl))
        prev_eq = float(st.equity)
        st.equity = prev_eq * (1.0 + net_ret * self.position_size)
        st.daily_pnl = float(st.equity) - self._daily_start_equity
        st.trades.append(asdict(trade))
        st.total_trades += 1
        if net_ret > 0:
            st.total_wins += 1
            st.consecutive_losses = 0
        elif net_ret < 0:
            st.consecutive_losses += 1
        if st.equity > st.peak_equity:
            st.peak_equity = float(st.equity)
        st.last_trade_time = trade.exit_time or datetime.utcnow().isoformat()
        if len(st.trades) > 500:
            st.trades = st.trades[-500:]
        self.save()

    @property
    def cumulative_return(self) -> float:
        return float(self.state.equity) - float(self.state.initial_equity)

    @property
    def max_drawdown(self) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return (float(self.state.equity) - float(self.state.peak_equity)) / float(
            self.state.peak_equity
        )


def simulate_signals_paper(
    df: pd.DataFrame,
    cache_path: Path = PROBA_CACHE_DEFAULT_PATH,
    allow_lag_minutes: float = ALLOW_LAG_MINUTES_DEFAULT,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    _, _, is_high_vol, _, is_low_vol, trend_label, _ = compute_regime_components(df)
    positive_allow = compute_positive_regime_allow_mask(df, POSITIVE_FILTER_MODE)
    p_long_arr, p_short_arr, p_flat_arr, valid_mask, proba_meta = _load_and_align_proba(
        df, cache_path=cache_path, allow_lag_minutes=allow_lag_minutes
    )
    blocked_positive = 0
    fb_ticks = 0
    ticks: List[Dict[str, Any]] = []
    for i in range(200, len(df)):
        vb = "high" if is_high_vol[i] else ("low" if is_low_vol[i] else "mid")
        if valid_mask[i] and not (
            np.isnan(p_long_arr[i]) or np.isnan(p_short_arr[i]) or np.isnan(p_flat_arr[i])
        ):
            pl, ps, pf = float(p_long_arr[i]), float(p_short_arr[i]), float(p_flat_arr[i])
            used_fb = False
        else:
            pl, ps, pf = FALLBACK_PROBA
            used_fb = True
            fb_ticks += 1
        ent = _entropy_nl([pl, ps, pf])
        signal: Optional[str] = None
        if not used_fb:
            if pl > ps and ent <= RAW_SIGNAL_ENTROPY_THRESHOLD:
                signal = "LONG"
            elif ps > pl and ent <= RAW_SIGNAL_ENTROPY_THRESHOLD:
                signal = "SHORT"
        if signal and not bool(positive_allow[i]):
            signal = None
            blocked_positive += 1
        ts_iso = None
        if "timestamp" in df.columns and pd.notna(df["timestamp"].iloc[i]):
            ts_iso = pd.Timestamp(df["timestamp"].iloc[i]).isoformat()
        ticks.append({
            "price": float(df["close"].iloc[i]),
            "vol_bucket": vb,
            "trend_label": str(trend_label[i]),
            "p_long": pl, "p_short": ps, "p_flat": pf,
            "signal": signal, "exit_signal": False,
            "df_idx": int(i), "timestamp": ts_iso,
        })
    proba_meta["blocked_by_positive_filter"] = blocked_positive
    proba_meta["positive_filter_mode"] = POSITIVE_FILTER_MODE
    proba_meta["fallback_ticks_count"] = fb_ticks
    return ticks, proba_meta


def _opposite_signal(pos_side: str, tick_signal: Optional[str]) -> bool:
    if tick_signal is None:
        return False
    return (pos_side == "BUY" and tick_signal == "SHORT") or (
        pos_side == "SELL" and tick_signal == "LONG"
    )


def run_paper_batch(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    state_mgr = PaperStateManager()
    risk_mgr = RiskManager(state_mgr, RiskLimits())
    router = ExecutionRouter("paper", state_mgr)
    monitor = PaperMonitor(monitoring_dir=MONITORING_DIR, enable_discord=False)
    engine = TradingEngine(
        mode="paper", state_mgr=state_mgr, risk_mgr=risk_mgr,
        router=router, monitor=monitor,
    )
    session_start_equity = float(state_mgr.state.equity)
    session_trades: List[Dict[str, Any]] = []
    decisions, reasons, strategies, entropies = [], [], [], []
    holding_bars = 0
    pending_dir: Optional[str] = None
    pending_entry_ts: Optional[str] = None

    for t in ticks:
        exit_signal = False
        pre_exit = ""
        if state_mgr.state.kill_switch_active and router.has_position:
            exit_signal, pre_exit = True, "kill_switch_close"
        if router.has_position:
            holding_bars += 1
            pos = router.open_position
            assert pos is not None
            if _opposite_signal(pos["side"], t.get("signal")):
                exit_signal, pre_exit = True, pre_exit or "opposite_signal"
            elif holding_bars >= PAPER_MAX_HOLDING_BARS:
                exit_signal, pre_exit = True, pre_exit or "max_holding_bars"
        result = engine.process_tick(TickContext(
            price=t["price"], vol_bucket=t["vol_bucket"], trend_label=t["trend_label"],
            p_long=t["p_long"], p_short=t["p_short"], p_flat=t["p_flat"],
            signal=t["signal"], exit_signal=exit_signal,
        ))
        decisions.append(result["decision"])
        reasons.append(result.get("reason", ""))
        strategies.append(result.get("strategy", ""))
        ent = sum(-p * math.log(p) for p in (t["p_long"], t["p_short"], t["p_flat"]) if p > 1e-10)
        entropies.append(ent)
        if result["decision"] == "exit":
            holding_bars = 0
            raw_pnl = float(result.get("pnl", 0.0))
            session_trades.append({
                "direction": pending_dir,
                "net_return": state_mgr.net_return_from_raw(raw_pnl),
                "entry_timestamp": pending_entry_ts,
                "exit_reason": pre_exit or result.get("reason", ""),
            })
            pending_dir, pending_entry_ts = None, None
        elif result["decision"] == "enter":
            holding_bars = 0
            rsn = result.get("reason", "") or ""
            pending_dir = "LONG" if rsn.startswith("LONG") else "SHORT"
            pending_entry_ts = t.get("timestamp")

    engine.shutdown()
    st = state_mgr.state
    decision_dist = dict(Counter(decisions))
    strategy_dist = dict(Counter(s for s in strategies if s))
    total_ticks = len(ticks)
    act_on = sum(1 for t in ticks if t["vol_bucket"] in ("mid", "high"))
    ent_arr = np.array(entropies, dtype=float) if entropies else np.array([0.0])
    sig_long = sum(1 for d, r in zip(decisions, reasons) if d == "enter" and (r or "").startswith("LONG"))
    sig_short = sum(1 for d, r in zip(decisions, reasons) if d == "enter" and (r or "").startswith("SHORT"))

    recent_stats: Dict[str, Any] = {}
    try:
        recent_stats = compute_recent_window_summary(
            ticks, decisions, reasons, strategies, entropies, hours=RECENT_WINDOW_HOURS
        )
    except Exception as e:
        recent_stats = {"ok": False, "recent_24h_unavailable": True, "error": str(e), "total_ticks": 0}

    net_returns = [x["net_return"] for x in session_trades]
    daily_tc = len(session_trades)
    wins = [r for r in net_returns if r > 0]
    losses = [r for r in net_returns if r < 0]
    limits = risk_mgr.limits

    r24: List[float] = []
    if recent_stats.get("ok"):
        try:
            ref, cutoff = pd.Timestamp(recent_stats["reference_ts_max"]), pd.Timestamp(
                recent_stats["cutoff_ts"]
            )
            for tr in session_trades:
                ets = tr.get("entry_timestamp")
                if ets and cutoff <= pd.Timestamp(ets) <= ref:
                    r24.append(tr["net_return"])
        except Exception:
            pass
    r24_cnt = len(r24)

    unrealized = 0.0
    if router.has_position and ticks:
        pos, last_px = router.open_position, float(ticks[-1]["price"])
        if pos and pos["side"] == "BUY":
            unrealized = (last_px - pos["price"]) / pos["price"]
        elif pos:
            unrealized = (pos["price"] - last_px) / pos["price"]

    return {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "paper",
        "no_live_orders": True,
        "total_ticks": total_ticks,
        "recent_24h_ticks": int(recent_stats.get("total_ticks", 0) or 0),
        "activation_ratio": act_on / max(total_ticks, 1),
        "activation_on_count": act_on,
        "activation_off_count": total_ticks - act_on,
        "signal_count": decision_dist.get("enter", 0),
        "signal_long_count": sig_long,
        "signal_short_count": sig_short,
        "entropy_mean": round(float(np.mean(ent_arr)), 6),
        "entropy_std": round(float(np.std(ent_arr)), 6),
        "entropy_distribution": {
            "min": round(float(np.min(ent_arr)), 6),
            "p50": round(float(np.percentile(ent_arr, 50)), 6),
            "max": round(float(np.max(ent_arr)), 6),
        },
        "recent_24h_signal_count": int(recent_stats.get("signal_count", 0) or 0),
        "s1_count": strategy_dist.get("S1", 0),
        "s2_count": strategy_dist.get("S2", 0),
        "s1_ratio": round(strategy_dist.get("S1", 0) / max(sum(strategy_dist.values()), 1), 4),
        "s2_ratio": round(strategy_dist.get("S2", 0) / max(sum(strategy_dist.values()), 1), 4),
        "daily_trade_count": daily_tc,
        "cumulative_trade_count": st.total_trades,
        "open_position": router.has_position,
        "long_trade_count": sum(1 for x in session_trades if x.get("direction") == "LONG"),
        "short_trade_count": sum(1 for x in session_trades if x.get("direction") == "SHORT"),
        "win_rate": round(len(wins) / daily_tc, 6) if daily_tc else 0.0,
        "avg_profit": round(float(np.mean(net_returns)), 8) if net_returns else 0.0,
        "avg_win": round(float(np.mean(wins)), 8) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 8) if losses else 0.0,
        "payoff_ratio": round(abs(float(np.mean(wins) / np.mean(losses))), 6) if losses else 0.0,
        "initial_equity": float(st.initial_equity),
        "current_equity": round(float(st.equity), 10),
        "daily_return": round(float(st.equity) - session_start_equity, 10),
        "cumulative_return": round(state_mgr.cumulative_return, 10),
        "max_drawdown": round(state_mgr.max_drawdown, 8),
        "unrealized_pnl": round(unrealized, 8),
        "paper_position_size": PAPER_POSITION_SIZE,
        "paper_max_holding_bars": PAPER_MAX_HOLDING_BARS,
        "fee_rate": PAPER_FEE_RATE,
        "slippage_rate": PAPER_SLIPPAGE_RATE,
        "positive_filter_mode": POSITIVE_FILTER_MODE,
        "kill_switch_active": st.kill_switch_active,
        "kill_switch_reason": st.kill_switch_reason,
        "consecutive_losses": st.consecutive_losses,
        "daily_loss_limit_status": "BREACHED" if st.daily_pnl <= limits.max_daily_loss else "OK",
        "drawdown_limit_status": "BREACHED" if state_mgr.max_drawdown <= limits.max_drawdown else "OK",
        "recent_24h_trade_count": r24_cnt,
        "recent_24h_pnl_proxy": round(float(sum(r24)), 8),
        "recent_24h_win_rate": round(sum(1 for r in r24 if r > 0) / r24_cnt, 6) if r24_cnt else 0.0,
        "recent_24h_avg_profit": round(float(np.mean(r24)), 8) if r24 else 0.0,
        "recent_24h_return": round(float(sum(r24)), 8),
        "recent_window_24h": recent_stats,
        "decision_distribution": decision_dist,
    }


def generate_report_md(summary: Dict[str, Any]) -> str:
    return (
        f"# 【Paper Ops】Paper Daily Report\n\n"
        f"**실행**: {summary['run_timestamp']} | **mode**: paper | **실제 주문 없음**\n\n"
        f"- ticks: {summary['total_ticks']} | 24h ticks: {summary.get('recent_24h_ticks')}\n"
        f"- signals: {summary['signal_count']} (L{summary['signal_long_count']}/S{summary['signal_short_count']})\n"
        f"- daily/cum trades: {summary['daily_trade_count']}/{summary['cumulative_trade_count']}\n"
        f"- equity: {summary['current_equity']:.6f} | daily_ret: {summary['daily_return']:.6f} | "
        f"maxDD: {summary['max_drawdown']:.2%}\n"
        f"- kill_switch: {summary['kill_switch_active']} ({summary.get('kill_switch_reason', '')})\n"
        f"- fee/slip: {summary['fee_rate']}/{summary['slippage_rate']} | pos_size: {summary['paper_position_size']}\n"
    )


def send_paper_discord_report(summary: Dict[str, Any], dry_run: bool = False) -> bool:
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url and not dry_run:
        return False
    psd = summary.get("proba_source_diagnostics") or {}
    fb = summary.get("fallback_default_proba_used") or {}
    color = 0xE74C3C if summary.get("kill_switch_active") else (
        0xF1C40F if fb.get("is_fallback") or not psd.get("proba_loaded") else 0x3498DB
    )
    rw = summary.get("recent_window_24h") or {}
    r24_txt = (
        f"**24h trades**: {summary.get('recent_24h_trade_count')} Σ={summary.get('recent_24h_return', 0):.4f}"
        if rw.get("ok") else "**recent_24h_unavailable**"
    )
    fields = [
        {"name": "📋 실행", "value": f"**paper** · **실제 주문 없음**\n**ticks**: {summary['total_ticks']}", "inline": False},
        {"name": "📦 Proba", "value": f"loaded={psd.get('proba_loaded')} aligned={psd.get('alignment', {}).get('aligned_ratio')}", "inline": True},
        {"name": "📈 Signals", "value": f"count={summary['signal_count']} entropy μ={summary['entropy_mean']:.4f}", "inline": True},
        {"name": "📊 Trades", "value": f"daily={summary['daily_trade_count']} win={summary['win_rate']:.1%} open={summary['open_position']}", "inline": True},
        {"name": "💰 PnL", "value": f"eq={summary['current_equity']:.4f} daily={summary['daily_return']:.4f} maxDD={summary['max_drawdown']:.2%}", "inline": False},
        {"name": "🛡️ Risk", "value": f"KS={summary['kill_switch_active']} consec={summary['consecutive_losses']}", "inline": True},
        {"name": "🕐 24h", "value": r24_txt, "inline": False},
        {"name": "📁 Logs", "value": f"`{summary.get('report_json', '')}`", "inline": False},
    ]
    embed = {
        "title": "【Paper Ops】Production Paper 일일 요약",
        "description": (
            "확정 파이프라인을 **paper mode**로 가상 포지션 lifecycle까지 평가합니다. "
            "**실제 주문은 발생하지 않습니다.**"
        ),
        "color": color,
        "fields": fields,
        "footer": {"text": f"Can_bit Paper Ops | {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"},
    }
    payload = {"embeds": [embed]}
    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return True
    import requests
    requests.post(webhook_url, json=payload, timeout=10).raise_for_status()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Daily Paper Ops (no live orders). --reset-state: 관찰 baseline 초기화 only."
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--reset-state",
        action="store_true",
        help="기존 paper state 백업 후 초기화 (배치/Discord 실행 안 함)",
    )
    args = parser.parse_args()
    if args.live:
        logger.error("live forbidden")
        sys.exit(2)
    if args.reset_state:
        result = reset_paper_observation_state()
        _print_reset_summary(result)
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_json = MONITORING_DIR / f"paper_daily_report_{ts}.json"
    report_md = MONITORING_DIR / f"paper_daily_report_{ts}.md"
    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    try:
        df = load_ohlcv()
        if df is None:
            raise RuntimeError("OHLCV not found")
        ticks, proba_meta = simulate_signals_paper(df)
        summary = run_paper_batch(ticks)
        summary["status"] = "SUCCESS"
        summary["proba_source_diagnostics"] = diagnose_proba_source(df, proba_meta)
        summary["fallback_default_proba_used"] = {
            "is_fallback": bool(proba_meta.get("fallback_used")),
            "fallback_reason": proba_meta.get("fallback_reason"),
        }
        summary["report_json"] = str(report_json)
        summary["report_md"] = str(report_md)
        report_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        report_md.write_text(generate_report_md(summary), encoding="utf-8")
        if not args.no_discord:
            send_paper_discord_report(summary, dry_run=args.dry_run)
        print(f"OK JSON={report_json} state={PAPER_STATE_FILE}")
    except Exception:
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
