"""
H8 Candidate Ops Monitor (diagnostics only).

Parallel monitor: production_shadow_like vs h8_candidate (entropy<=0.96, trend==up).
No live orders, no paper/shadow state changes, no TradingEngine.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.diagnostics.validate_final_hypotheses import _simulate
from scripts.run_daily_shadow import (
    RECENT_WINDOW_HOURS,
    diagnose_proba_source,
    load_ohlcv,
    simulate_signals,
    _parse_tick_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("h8_candidate")

MONITORING_DIR = Path("data/monitoring")
ENTROPY_PRODUCTION = 1.0
ENTROPY_H8 = 0.96
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
POSITION_SIZE = 0.05

PRODUCTION_PARAMS: Dict[str, Any] = {}
H8_PARAMS: Dict[str, Any] = {"entropy_ub": 0.96, "trend_up_only": True}


def _entropy(t: Dict[str, Any]) -> float:
    e = 0.0
    for p in (float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])):
        if p > 1e-10:
            e -= p * math.log(p)
    return e


def _production_entry_ok(t: Dict[str, Any], ent: float) -> Tuple[bool, str]:
    if t.get("vol_bucket") not in ("mid", "high"):
        return False, "activation_off"
    if t.get("signal") is None:
        return False, "no_signal"
    strategy = "S2" if str(t.get("trend_label")) != "sideways" else "S1"
    if strategy == "S2" and ent > ENTROPY_PRODUCTION:
        return False, "entropy_filter_production"
    return True, "enter"


def _h8_entry_ok(t: Dict[str, Any], ent: float) -> Tuple[bool, str]:
    ok, reason = _production_entry_ok(t, ent)
    if not ok:
        return False, reason
    if ent > ENTROPY_H8:
        if str(t.get("trend_label")) != "up":
            return False, "both_entropy_and_trend"
        return False, "entropy_gt_0.96"
    if str(t.get("trend_label")) != "up":
        return False, "trend_not_up"
    return True, "enter"


def _slice_recent_ticks(ticks: List[Dict[str, Any]], hours: float = RECENT_WINDOW_HOURS) -> List[Dict[str, Any]]:
    ts_list = [(i, _parse_tick_timestamp(t)) for i, t in enumerate(ticks)]
    valid = [(i, ts) for i, ts in ts_list if ts is not None]
    if not valid:
        return []
    ts_max = max(ts for _, ts in valid)
    cutoff = ts_max - pd.Timedelta(hours=hours)
    return [ticks[i] for i, ts in valid if ts >= cutoff]


def _tick_pipeline_stats(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ticks:
        return {
            "ticks": 0,
            "activation_ratio": 0.0,
            "signal_count": 0,
            "long_count": 0,
            "short_count": 0,
            "entropy_mean": 0.0,
            "trend_up_count": 0,
            "production_enter_count": 0,
            "h8_enter_count": 0,
            "candidate_blocked_count": 0,
        }
    ents = [_entropy(t) for t in ticks]
    activation_on = sum(1 for t in ticks if t.get("vol_bucket") in ("mid", "high"))
    sigs = [t for t in ticks if t.get("signal") is not None]
    prod_ent = 0
    h8_ent = 0
    blocked = 0
    for t, ent in zip(ticks, ents):
        pok, _ = _production_entry_ok(t, ent)
        hok, _ = _h8_entry_ok(t, ent)
        if pok:
            prod_ent += 1
        if hok:
            h8_ent += 1
        if pok and not hok:
            blocked += 1
    return {
        "ticks": len(ticks),
        "activation_ratio": round(activation_on / len(ticks), 6),
        "signal_count": len(sigs),
        "long_count": sum(1 for t in sigs if t.get("signal") == "LONG"),
        "short_count": sum(1 for t in sigs if t.get("signal") == "SHORT"),
        "entropy_mean": round(float(np.mean(ents)), 6),
        "trend_up_count": sum(1 for t in ticks if str(t.get("trend_label")) == "up"),
        "production_enter_count": prod_ent,
        "h8_enter_count": h8_ent,
        "candidate_blocked_count": blocked,
    }


def _blocked_distribution(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """H8 vs production: why production-intent signals did not pass H8."""
    counts: Counter[str] = Counter()
    prod_only = 0
    for t in ticks:
        ent = _entropy(t)
        pok, _ = _production_entry_ok(t, ent)
        hok, hreason = _h8_entry_ok(t, ent)
        if not pok:
            if hreason in ("activation_off", "no_signal"):
                counts[hreason] += 1
            continue
        if pok and not hok:
            prod_only += 1
            if hreason in (
                "entropy_gt_0.96",
                "trend_not_up",
                "both_entropy_and_trend",
            ):
                counts[hreason] += 1
            else:
                counts[hreason] += 1
    total = sum(counts.values()) or 1
    return {
        "production_only_blocked_by_h8": prod_only,
        "blocked_by": dict(counts),
        "blocked_pct": {k: round(v / total, 4) for k, v in counts.items()},
    }


def _metrics_from_sim(m: Dict[str, Any], tdf: pd.DataFrame) -> Dict[str, Any]:
    vals = tdf["net_return"].tolist() if not tdf.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    avg_ret = float(np.mean(vals)) if vals else 0.0
    return {
        "trade_count": int(m["total_trades"]),
        "win_rate": float(m["win_rate"]),
        "avg_profit": avg_ret,
        "expectancy": float(m["expectancy"]),
        "profit_factor": float(pf),
        "net_return": float(m["net_return"]),
        "max_drawdown": float(m["max_drawdown"]),
        "risk_force_exit_count": int(m["risk_force_exit_count"]),
        "kill_switch_triggered": bool(m["kill_switch_triggered"]),
        "avg_hold_bars": float(tdf["hold_bars"].mean()) if not tdf.empty else 0.0,
        "trades_df_entries": tdf.to_dict(orient="records") if len(tdf) <= 500 else [],
    }


def _virtual_metrics(ticks: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
    m, tdf = _simulate(ticks, params, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    return _metrics_from_sim(m, tdf)


def _removal_proxy(
    prod_trades: pd.DataFrame,
    h8_trades: pd.DataFrame,
) -> Dict[str, Any]:
    if prod_trades.empty:
        return {
            "trade_reduction_pct": 0.0,
            "bad_trade_removed_proxy": 0,
            "good_trade_removed_proxy": 0,
            "removal_precision_proxy": 0.0,
        }
    prod_idx = set(prod_trades["entry_idx"].astype(int).tolist())
    h8_idx = set(h8_trades["entry_idx"].astype(int).tolist()) if not h8_trades.empty else set()
    removed_idx = prod_idx - h8_idx
    removed = prod_trades[prod_trades["entry_idx"].isin(removed_idx)]
    bad = int((removed["net_return"] < 0).sum()) if not removed.empty else 0
    good = int((removed["net_return"] > 0).sum()) if not removed.empty else 0
    total_removed = len(removed)
    prec = bad / total_removed if total_removed > 0 else 0.0
    red = (1.0 - len(h8_idx) / max(len(prod_idx), 1)) * 100.0
    return {
        "trade_reduction_pct": round(red, 2),
        "bad_trade_removed_proxy": bad,
        "good_trade_removed_proxy": good,
        "removal_precision_proxy": round(prec, 4),
    }


def _daily_verdict(prod: Dict[str, Any], h8: Dict[str, Any]) -> str:
    pt = int(prod.get("trade_count", 0))
    ht = int(h8.get("trade_count", 0))
    if pt + ht < 5:
        return "INSUFFICIENT_SAMPLE"
    if ht == 0:
        return "NO_TRADE"
    p_net = float(prod.get("net_return", 0))
    h_net = float(h8.get("net_return", 0))
    p_mdd = float(prod.get("max_drawdown", 0))
    h_mdd = float(h8.get("max_drawdown", 0))
    if h_net < p_net or h_mdd < p_mdd:
        return "H8_WORSE"
    if h_net > p_net and h_mdd >= p_mdd and ht >= 1:
        return "H8_BETTER"
    return "H8_WORSE"


def _compare_24h(prod: Dict[str, Any], h8: Dict[str, Any]) -> Dict[str, Any]:
    pt = int(prod.get("trade_count", 0))
    ht = int(h8.get("trade_count", 0))
    red = (1.0 - ht / max(pt, 1)) * 100.0 if pt > 0 else 0.0
    return {
        "production_signals": int(prod.get("production_enter_count", prod.get("signal_count", 0))),
        "h8_signals": int(h8.get("h8_enter_count", h8.get("signal_count", 0))),
        "production_virtual_pnl": float(prod.get("net_return", 0)),
        "h8_virtual_pnl": float(h8.get("net_return", 0)),
        "production_win_rate": float(prod.get("win_rate", 0)),
        "h8_win_rate": float(h8.get("win_rate", 0)),
        "trade_reduction_pct": round(red, 2),
    }


def build_report(
    ticks: List[Dict[str, Any]],
    proba_meta: Dict[str, Any],
    dry_run: bool = False,
) -> Dict[str, Any]:
    ts_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proba_diag = diagnose_proba_source(None, proba_meta)

    tick_stats_all = _tick_pipeline_stats(ticks)
    recent_ticks = _slice_recent_ticks(ticks)
    tick_stats_24h = _tick_pipeline_stats(recent_ticks)
    blocked = _blocked_distribution(ticks)

    prod_m, prod_tdf = _simulate(ticks, PRODUCTION_PARAMS, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    h8_m, h8_tdf = _simulate(ticks, H8_PARAMS, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    prod_metrics = _metrics_from_sim(prod_m, prod_tdf)
    h8_metrics = _metrics_from_sim(h8_m, h8_tdf)

    prod_24m_raw, prod_tdf_24 = _simulate(recent_ticks, PRODUCTION_PARAMS, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    h8_24m_raw, h8_tdf_24 = _simulate(recent_ticks, H8_PARAMS, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    prod_24m = _metrics_from_sim(prod_24m_raw, prod_tdf_24)
    prod_24m["production_enter_count"] = tick_stats_24h["production_enter_count"]
    prod_24m["signal_count"] = tick_stats_24h["signal_count"]
    h8_24m = _metrics_from_sim(h8_24m_raw, h8_tdf_24)
    h8_24m["h8_enter_count"] = tick_stats_24h["h8_enter_count"]
    h8_24m["signal_count"] = tick_stats_24h["signal_count"]

    removal = _removal_proxy(prod_tdf, h8_tdf)
    verdict = _daily_verdict(prod_24m, h8_24m)

    report: Dict[str, Any] = {
        "mode": "h8_candidate_monitor",
        "disclaimer": "진단 후보 모니터입니다. 실제 주문 없음. 운영 로직 미변경.",
        "run_timestamp": ts_run,
        "dry_run": dry_run,
        "h8_definition": {
            "entropy_max": ENTROPY_H8,
            "trend_state": "up",
            "base_pipeline": "activation → rule_C_trend → entropy<=1.00",
        },
        "candidates": {
            "production_shadow_like": {
                "label": "A",
                "params": PRODUCTION_PARAMS,
                "description": "activation → rule_C_trend → entropy<=1.00",
            },
            "h8_candidate": {
                "label": "B",
                "params": H8_PARAMS,
                "description": "production + entropy<=0.96 + trend==up",
            },
        },
        "proba_health": {
            "proba_loaded": proba_diag.get("proba_loaded"),
            "fallback_used": proba_diag.get("fallback", {}).get("fallback_used"),
            "aligned_ratio": proba_diag.get("alignment", {}).get("aligned_ratio"),
            "ohlcv_ts_max": proba_diag.get("alignment", {}).get("ohlcv_ts_max"),
            "proba_ts_max": proba_diag.get("alignment", {}).get("proba_ts_max"),
        },
        "tick_stats_full": tick_stats_all,
        "tick_stats_24h": tick_stats_24h,
        "blocked_analysis": blocked,
        "production_full": prod_metrics,
        "h8_full": h8_metrics,
        "production_24h": prod_24m,
        "h8_24h": h8_24m,
        "compare_24h": _compare_24h(
            {**prod_24m, **tick_stats_24h},
            {**h8_24m, **tick_stats_24h},
        ),
        "cumulative": {
            "production": prod_metrics,
            "h8": h8_metrics,
            **removal,
        },
        "risk_proxy_full": {
            "production": {
                "risk_force_exit": prod_metrics["risk_force_exit_count"],
                "kill_switch": prod_metrics["kill_switch_triggered"],
                "mdd": prod_metrics["max_drawdown"],
            },
            "h8": {
                "risk_force_exit": h8_metrics["risk_force_exit_count"],
                "kill_switch": h8_metrics["kill_switch_triggered"],
                "mdd": h8_metrics["max_drawdown"],
            },
        },
        "daily_verdict": verdict,
        "proba_source_diagnostics": proba_diag,
    }
    return report


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    c24 = report["compare_24h"]
    pf = report["production_full"]
    hf = report["h8_full"]
    blk = report["blocked_analysis"]
    lines = [
        "# H8 Candidate Daily Monitor",
        "",
        report["disclaimer"],
        "",
        f"- **Run**: {report['run_timestamp']}",
        f"- **Mode**: {report['mode']}",
        "",
        "## H8 Definition",
        f"- entropy <= {ENTROPY_H8}",
        "- trend_state == up",
        "",
        "## Proba Health",
        f"- proba_loaded: {report['proba_health']['proba_loaded']}",
        f"- fallback_used: {report['proba_health']['fallback_used']}",
        f"- aligned_ratio: {report['proba_health']['aligned_ratio']}",
        "",
        "## Production vs H8 — 24h",
        f"| Metric | Production | H8 |",
        f"|--------|------------|-----|",
        f"| virtual net_return | {c24['production_virtual_pnl']:.6f} | {c24['h8_virtual_pnl']:.6f} |",
        f"| win_rate | {c24['production_win_rate']:.2%} | {c24['h8_win_rate']:.2%} |",
        f"| trade_reduction | — | {c24['trade_reduction_pct']:.1f}% |",
        "",
        "## H8 Blocked (vs production intent)",
    ]
    for k, v in (blk.get("blocked_by") or {}).items():
        pct = blk.get("blocked_pct", {}).get(k, 0)
        lines.append(f"- {k}: {v} ({pct:.1%})")
    lines += [
        "",
        "## Cumulative",
        f"- production: trades={pf['trade_count']}, exp={pf['expectancy']:.6f}, net={pf['net_return']:.6f}",
        f"- h8: trades={hf['trade_count']}, exp={hf['expectancy']:.6f}, net={hf['net_return']:.6f}",
        f"- removal precision proxy: {report['cumulative'].get('removal_precision_proxy', 0):.2%}",
        "",
        f"## Daily Verdict: **{report['daily_verdict']}**",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def send_h8_discord(report: Dict[str, Any], json_path: str, md_path: str, dry_run: bool = False) -> bool:
    import os

    try:
        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    except Exception:
        pass

    webhook = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook and not dry_run:
        logger.warning("[Discord] DISCORD_WEBHOOK_URL missing")
        return False

    ph = report["proba_health"]
    c24 = report["compare_24h"]
    rp = report["risk_proxy_full"]
    blk = report["blocked_analysis"]
    blk_lines = "\n".join(
        f"- {k}: {v}" for k, v in list((blk.get("blocked_by") or {}).items())[:6]
    ) or "- none"

    fields = [
        {
            "name": "📋 [0] 실행 정보",
            "value": (
                f"**mode**: h8_candidate_monitor\n"
                f"**{report['disclaimer']}**\n"
                f"**timestamp**: {report['run_timestamp']}"
            ),
            "inline": False,
        },
        {
            "name": "📊 [1] Data / Proba Health",
            "value": (
                f"proba_loaded: {ph['proba_loaded']}\n"
                f"fallback: {ph['fallback_used']}\n"
                f"aligned_ratio: {ph.get('aligned_ratio')}\n"
                f"ohlcv_ts_max: {ph.get('ohlcv_ts_max')}\n"
                f"proba_ts_max: {ph.get('proba_ts_max')}"
            ),
            "inline": False,
        },
        {
            "name": "⚖️ [2] Production vs H8 — 24h",
            "value": (
                f"signals prod/H8: {c24.get('production_signals')} / {c24.get('h8_signals')}\n"
                f"virtual PnL prod/H8: {c24['production_virtual_pnl']:.6f} / {c24['h8_virtual_pnl']:.6f}\n"
                f"win prod/H8: {c24['production_win_rate']:.1%} / {c24['h8_win_rate']:.1%}\n"
                f"trade_reduction: {c24['trade_reduction_pct']:.1f}%"
            ),
            "inline": False,
        },
        {
            "name": "🔒 [3] H8 Filter",
            "value": (
                f"entropy <= {ENTROPY_H8}\n"
                f"trend == up\n"
                f"blocked (prod-only):\n{blk_lines}"
            ),
            "inline": False,
        },
        {
            "name": "🛡️ [4] Risk Proxy (full batch)",
            "value": (
                f"prod RFE/KS/MDD: {rp['production']['risk_force_exit']}/"
                f"{rp['production']['kill_switch']}/"
                f"{rp['production']['mdd']:.4f}\n"
                f"H8 RFE/KS/MDD: {rp['h8']['risk_force_exit']}/"
                f"{rp['h8']['kill_switch']}/"
                f"{rp['h8']['mdd']:.4f}"
            ),
            "inline": False,
        },
        {
            "name": "✅ [5] Final Daily Verdict",
            "value": f"**{report['daily_verdict']}**",
            "inline": True,
        },
        {
            "name": "📁 [6] Logs",
            "value": f"JSON: `{json_path}`\nMD: `{md_path}`",
            "inline": False,
        },
    ]

    verdict = report["daily_verdict"]
    color = 0x2ecc71 if verdict == "H8_BETTER" else 0xe74c3c if verdict == "H8_WORSE" else 0xf1c40f
    embed = {
        "title": "【H8 Candidate Ops】Entropy96 + TrendUp 후보 일일 요약",
        "description": report["disclaimer"],
        "color": color,
        "fields": fields,
        "footer": {"text": "Can_bit H8 Candidate Monitor (diagnostic only)"},
        "timestamp": datetime.utcnow().isoformat(),
    }
    payload = {"embeds": [embed]}

    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return True

    try:
        import requests

        resp = requests.post(webhook, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("[Discord] H8 candidate report sent")
        return True
    except Exception as e:
        logger.error("[Discord] send failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="H8 candidate parallel monitor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    df = load_ohlcv()
    if df is None or df.empty:
        logger.error("OHLCV load failed")
        sys.exit(1)

    ticks, proba_meta = simulate_signals(df)
    if not ticks:
        logger.error("No ticks from simulate_signals")
        sys.exit(1)

    report = build_report(ticks, proba_meta, dry_run=args.dry_run)

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = MONITORING_DIR / f"h8_candidate_daily_report_{ts_file}.json"
    md_path = MONITORING_DIR / f"h8_candidate_daily_report_{ts_file}.md"
    log_path = MONITORING_DIR / f"h8_candidate_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, md_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": report["run_timestamp"], "verdict": report["daily_verdict"], "report": report}, default=str) + "\n")

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info("Appended %s", log_path)

    if not args.no_discord:
        send_h8_discord(report, str(json_path), str(md_path), dry_run=args.dry_run)

    print("[H8 CANDIDATE MONITOR]")
    print(f"verdict: {report['daily_verdict']}")
    print(f"json: {json_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
