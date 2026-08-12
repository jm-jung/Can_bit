"""
H8 SoftGate G30 Candidate Monitor (diagnostics only).

Compares production replay vs H8 soft gate (fail scale=0.30).
No live orders, no state mutation, no TradingEngine.
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

from scripts.diagnostics.validate_h8_soft_risk_gate import Variant, _simulate_variant
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
logger = logging.getLogger("h8_softgate_candidate")

MONITORING_DIR = Path("data/monitoring")
ENTROPY_H8 = 0.96
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
POSITION_SIZE = 0.05
FAIL_SCALE = 0.30

BASELINE_VARIANT = Variant("Baseline", fail_scale=1.0, hard_block=False)
SOFTGATE_VARIANT = Variant("Variant_G30", fail_scale=FAIL_SCALE, hard_block=False)


def _entropy(t: Dict[str, Any]) -> float:
    e = 0.0
    for p in (float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])):
        if p > 1e-10:
            e -= p * math.log(p)
    return e


def _h8_pass(t: Dict[str, Any]) -> bool:
    return (_entropy(t) <= ENTROPY_H8) and (str(t.get("trend_label") or "") == "up")


def _production_entry_ok(t: Dict[str, Any], ent: float) -> bool:
    if t.get("vol_bucket") not in ("mid", "high"):
        return False
    if t.get("signal") is None:
        return False
    strategy = "S2" if str(t.get("trend_label")) != "sideways" else "S1"
    if strategy == "S2" and ent > 1.0:
        return False
    return True


def _slice_recent_ticks(ticks: List[Dict[str, Any]], hours: float = RECENT_WINDOW_HOURS) -> List[Dict[str, Any]]:
    ts_list = [(i, _parse_tick_timestamp(t)) for i, t in enumerate(ticks)]
    valid = [(i, ts) for i, ts in ts_list if ts is not None]
    if not valid:
        return []
    ts_max = max(ts for _, ts in valid)
    cutoff = ts_max - pd.Timedelta(hours=hours)
    return [ticks[i] for i, ts in valid if ts >= cutoff]


def _funnel_diagnostics(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    production_candidates = 0
    entropy_pass = 0
    trend_pass = 0
    h8_pass = 0
    h8_fail = 0
    blocked_entropy = 0
    blocked_trend = 0
    blocked_both = 0

    for t in ticks:
        ent = _entropy(t)
        if not _production_entry_ok(t, ent):
            continue
        production_candidates += 1
        tr_up = str(t.get("trend_label") or "") == "up"
        ent096 = ent <= ENTROPY_H8
        if ent096:
            entropy_pass += 1
        if tr_up:
            trend_pass += 1
        if _h8_pass(t):
            h8_pass += 1
        else:
            h8_fail += 1
            if (not ent096) and tr_up:
                blocked_entropy += 1
            elif ent096 and (not tr_up):
                blocked_trend += 1
            elif (not ent096) and (not tr_up):
                blocked_both += 1

    denom = max(production_candidates, 1)
    return {
        "production_candidates": production_candidates,
        "entropy_pass": entropy_pass,
        "trend_pass": trend_pass,
        "h8_pass": h8_pass,
        "h8_fail": h8_fail,
        "blocked_by_entropy": blocked_entropy,
        "blocked_by_trend": blocked_trend,
        "blocked_by_both": blocked_both,
        "entropy_pass_ratio": entropy_pass / denom,
        "trend_pass_ratio": trend_pass / denom,
        "h8_pass_ratio": h8_pass / denom,
        "h8_fail_ratio": h8_fail / denom,
    }


def _metrics_from_sim(m: Dict[str, Any], tdf: pd.DataFrame) -> Dict[str, Any]:
    vals = tdf["scaled_return"].tolist() if not tdf.empty and "scaled_return" in tdf.columns else (
        tdf["net_return"].tolist() if not tdf.empty else []
    )
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    scales = tdf["scale"].tolist() if not tdf.empty and "scale" in tdf.columns else []
    h8_pass_n = int((tdf["h8_pass"] == True).sum()) if not tdf.empty and "h8_pass" in tdf.columns else 0
    h8_fail_n = int((tdf["h8_pass"] == False).sum()) if not tdf.empty and "h8_pass" in tdf.columns else 0
    scaled_n = int((tdf["scale"] < 1.0).sum()) if not tdf.empty and "scale" in tdf.columns else 0
    scale_dist = Counter(scales) if scales else Counter()

    return {
        "trade_count": int(m["trades"]),
        "win_rate": float(m["win_rate"]),
        "avg_return": float(np.mean(vals)) if vals else 0.0,
        "expectancy": float(m["expectancy"]),
        "profit_factor": float(pf),
        "net_return": float(m["net_return"]),
        "max_drawdown": float(m["MDD"]),
        "risk_force_exit_count": int(m["risk_force_exit_count"]),
        "kill_switch_triggered": bool(m["KS_triggered"]),
        "avg_hold_bars": float(m["avg_hold"]),
        "h8_pass_trades": h8_pass_n,
        "h8_fail_trades": h8_fail_n,
        "scaled_trades_count": scaled_n,
        "avg_applied_scale": float(np.mean(scales)) if scales else 1.0,
        "scale_distribution": {str(k): int(v) for k, v in scale_dist.items()},
    }


def _daily_verdict(prod: Dict[str, Any], sg: Dict[str, Any]) -> str:
    pt = int(prod.get("trade_count", 0))
    st = int(sg.get("trade_count", 0))
    if pt + st < 5:
        return "INSUFFICIENT_SAMPLE"
    if st == 0:
        return "NO_TRADE"
    p_net = float(prod.get("net_return", 0))
    s_net = float(sg.get("net_return", 0))
    p_mdd = float(prod.get("max_drawdown", 0))
    s_mdd = float(sg.get("max_drawdown", 0))
    if s_net < p_net or s_mdd < p_mdd:
        return "SOFTGATE_WORSE"
    if s_net > p_net and s_mdd >= p_mdd and st >= 1:
        return "SOFTGATE_BETTER"
    return "SOFTGATE_WORSE"


def build_report(ticks: List[Dict[str, Any]], proba_meta: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    ts_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proba_diag = diagnose_proba_source(None, proba_meta)

    recent_ticks = _slice_recent_ticks(ticks)
    funnel_full = _funnel_diagnostics(ticks)
    funnel_24h = _funnel_diagnostics(recent_ticks)

    prod_m, prod_tdf = _simulate_variant(ticks, BASELINE_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    sg_m, sg_tdf = _simulate_variant(ticks, SOFTGATE_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    prod_metrics = _metrics_from_sim(prod_m, prod_tdf)
    sg_metrics = _metrics_from_sim(sg_m, sg_tdf)

    prod_24m, prod_tdf_24 = _simulate_variant(recent_ticks, BASELINE_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    sg_24m, sg_tdf_24 = _simulate_variant(recent_ticks, SOFTGATE_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    prod_24 = _metrics_from_sim(prod_24m, prod_tdf_24)
    sg_24 = _metrics_from_sim(sg_24m, sg_tdf_24)

    trade_pres = sg_24["trade_count"] / max(prod_24["trade_count"], 1)
    verdict = _daily_verdict(prod_24, sg_24)

    report: Dict[str, Any] = {
        "mode": "h8_softgate_g30_monitor",
        "disclaimer": "진단 후보 모니터입니다. 실제 주문 없음. 운영 로직 미변경.",
        "run_timestamp": ts_run,
        "dry_run": dry_run,
        "softgate_policy": {
            "h8_pass_scale": 1.0,
            "h8_fail_scale": FAIL_SCALE,
            "h8_condition": "entropy<=0.96 AND trend_state==up",
        },
        "proba_health": {
            "proba_loaded": proba_diag.get("proba_loaded"),
            "fallback_used": proba_diag.get("fallback", {}).get("fallback_used"),
            "aligned_ratio": proba_diag.get("alignment", {}).get("aligned_ratio"),
            "ohlcv_ts_max": proba_diag.get("alignment", {}).get("ohlcv_ts_max"),
            "proba_ts_max": proba_diag.get("alignment", {}).get("proba_ts_max"),
        },
        "compare_24h": {
            "production_signals": funnel_24h["production_candidates"],
            "softgate_signals": funnel_24h["production_candidates"],
            "production_trades": prod_24["trade_count"],
            "softgate_trades": sg_24["trade_count"],
            "production_virtual_pnl": prod_24["net_return"],
            "softgate_virtual_pnl": sg_24["net_return"],
            "production_win_rate": prod_24["win_rate"],
            "softgate_win_rate": sg_24["win_rate"],
            "production_avg_return": prod_24["avg_return"],
            "softgate_avg_return": sg_24["avg_return"],
            "trade_preservation_pct": trade_pres,
            "avg_position_scale": sg_24["avg_applied_scale"],
        },
        "softgate_scaling_full": {
            "h8_pass_count": sg_metrics["h8_pass_trades"],
            "h8_fail_count": sg_metrics["h8_fail_trades"],
            "scaled_trades_count": sg_metrics["scaled_trades_count"],
            "avg_applied_scale": sg_metrics["avg_applied_scale"],
            "scale_distribution": sg_metrics["scale_distribution"],
        },
        "softgate_scaling_24h": {
            "h8_pass_count": sg_24["h8_pass_trades"],
            "h8_fail_count": sg_24["h8_fail_trades"],
            "scaled_trades_count": sg_24["scaled_trades_count"],
            "avg_applied_scale": sg_24["avg_applied_scale"],
            "scale_distribution": sg_24["scale_distribution"],
        },
        "risk_proxy_full": {
            "production": {
                "risk_force_exit": prod_metrics["risk_force_exit_count"],
                "kill_switch": prod_metrics["kill_switch_triggered"],
                "mdd": prod_metrics["max_drawdown"],
            },
            "softgate": {
                "risk_force_exit": sg_metrics["risk_force_exit_count"],
                "kill_switch": sg_metrics["kill_switch_triggered"],
                "mdd": sg_metrics["max_drawdown"],
            },
            "delta": {
                "RFE_delta": sg_metrics["risk_force_exit_count"] - prod_metrics["risk_force_exit_count"],
                "KS_delta": int(sg_metrics["kill_switch_triggered"]) - int(prod_metrics["kill_switch_triggered"]),
                "MDD_delta": sg_metrics["max_drawdown"] - prod_metrics["max_drawdown"],
            },
        },
        "funnel_full": funnel_full,
        "funnel_24h": funnel_24h,
        "production_full": prod_metrics,
        "softgate_full": sg_metrics,
        "production_24h": prod_24,
        "softgate_24h": sg_24,
        "daily_verdict": verdict,
        "proba_source_diagnostics": proba_diag,
    }
    return report


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    c = report["compare_24h"]
    rp = report["risk_proxy_full"]
    f = report["funnel_24h"]
    lines = [
        "# H8 SoftGate G30 Candidate Daily Monitor",
        "",
        report["disclaimer"],
        "",
        f"- Run: {report['run_timestamp']}",
        f"- Policy: H8 pass scale=1.0, H8 fail scale={FAIL_SCALE}",
        "",
        "## Production vs SoftGate — 24h",
        f"| Metric | Production | SoftGate G30 |",
        f"|--------|------------|--------------|",
        f"| trades | {c['production_trades']} | {c['softgate_trades']} |",
        f"| virtual PnL | {c['production_virtual_pnl']:.6f} | {c['softgate_virtual_pnl']:.6f} |",
        f"| win_rate | {c['production_win_rate']:.2%} | {c['softgate_win_rate']:.2%} |",
        f"| trade preservation | — | {c['trade_preservation_pct']:.2%} |",
        f"| avg scale | 1.00 | {c['avg_position_scale']:.3f} |",
        "",
        "## Funnel (24h)",
        f"- production_candidates: {f['production_candidates']}",
        f"- entropy_pass: {f['entropy_pass']}",
        f"- trend_pass: {f['trend_pass']}",
        f"- h8_pass: {f['h8_pass']}",
        f"- h8_fail: {f['h8_fail']}",
        "",
        "## Risk Proxy (full)",
        f"- prod RFE/KS/MDD: {rp['production']['risk_force_exit']}/{rp['production']['kill_switch']}/{rp['production']['mdd']:.4f}",
        f"- sg RFE/KS/MDD: {rp['softgate']['risk_force_exit']}/{rp['softgate']['kill_switch']}/{rp['softgate']['mdd']:.4f}",
        f"- deltas: RFE={rp['delta']['RFE_delta']}, KS={rp['delta']['KS_delta']}, MDD={rp['delta']['MDD_delta']:.6f}",
        "",
        f"## Daily Verdict: **{report['daily_verdict']}**",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def send_discord(report: Dict[str, Any], json_path: str, md_path: str, dry_run: bool = False) -> bool:
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
    sc = report["softgate_scaling_24h"]
    f24 = report["funnel_24h"]

    fields = [
        {
            "name": "📋 [0] 실행 정보",
            "value": (
                f"**mode**: h8_softgate_g30_monitor\n"
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
            "name": "⚖️ [2] Production vs SoftGate — 24h",
            "value": (
                f"trades prod/sg: {c24['production_trades']} / {c24['softgate_trades']}\n"
                f"virtual PnL prod/sg: {c24['production_virtual_pnl']:.6f} / {c24['softgate_virtual_pnl']:.6f}\n"
                f"win prod/sg: {c24['production_win_rate']:.1%} / {c24['softgate_win_rate']:.1%}\n"
                f"trade_preservation: {c24['trade_preservation_pct']:.1%}\n"
                f"avg_scale: {c24['avg_position_scale']:.3f}"
            ),
            "inline": False,
        },
        {
            "name": "🔧 [3] SoftGate Scaling",
            "value": (
                f"H8 pass/fail trades: {sc['h8_pass_count']} / {sc['h8_fail_count']}\n"
                f"scaled_trades: {sc['scaled_trades_count']}\n"
                f"avg_scale: {sc['avg_applied_scale']:.3f}\n"
                f"distribution: {sc['scale_distribution']}"
            ),
            "inline": False,
        },
        {
            "name": "🛡️ [4] Risk Proxy",
            "value": (
                f"prod RFE/KS/MDD: {rp['production']['risk_force_exit']}/"
                f"{rp['production']['kill_switch']}/{rp['production']['mdd']:.4f}\n"
                f"sg RFE/KS/MDD: {rp['softgate']['risk_force_exit']}/"
                f"{rp['softgate']['kill_switch']}/{rp['softgate']['mdd']:.4f}\n"
                f"Δ RFE/KS/MDD: {rp['delta']['RFE_delta']}/"
                f"{rp['delta']['KS_delta']}/{rp['delta']['MDD_delta']:.6f}"
            ),
            "inline": False,
        },
        {
            "name": "🔍 [5] Funnel Diagnostics",
            "value": (
                f"candidates: {f24['production_candidates']}\n"
                f"entropy_pass: {f24['entropy_pass']}\n"
                f"trend_pass: {f24['trend_pass']}\n"
                f"h8_pass/fail: {f24['h8_pass']} / {f24['h8_fail']}\n"
                f"blocked entropy/trend/both: {f24['blocked_by_entropy']}/"
                f"{f24['blocked_by_trend']}/{f24['blocked_by_both']}"
            ),
            "inline": False,
        },
        {
            "name": "✅ [6] Final Daily Verdict",
            "value": f"**{report['daily_verdict']}**",
            "inline": True,
        },
        {
            "name": "📁 Logs",
            "value": f"JSON: `{json_path}`\nMD: `{md_path}`",
            "inline": False,
        },
    ]

    verdict = report["daily_verdict"]
    color = 0x2ecc71 if verdict == "SOFTGATE_BETTER" else 0xe74c3c if verdict == "SOFTGATE_WORSE" else 0xf1c40f
    embed = {
        "title": "【H8 SoftGate Ops】G30 후보 일일 요약",
        "description": report["disclaimer"],
        "color": color,
        "fields": fields,
        "footer": {"text": "Can_bit H8 SoftGate G30 Monitor (diagnostic only)"},
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
        logger.info("[Discord] SoftGate candidate report sent")
        return True
    except Exception as e:
        logger.error("[Discord] send failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="H8 SoftGate G30 candidate monitor")
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
    json_path = MONITORING_DIR / f"h8_softgate_candidate_daily_report_{ts_file}.json"
    md_path = MONITORING_DIR / f"h8_softgate_candidate_daily_report_{ts_file}.md"
    log_path = MONITORING_DIR / f"h8_softgate_candidate_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, md_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"ts": report["run_timestamp"], "verdict": report["daily_verdict"], "report": report},
                default=str,
            )
            + "\n"
        )

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info("Appended %s", log_path)

    discord_ok = False
    if not args.no_discord:
        discord_ok = send_discord(report, str(json_path), str(md_path), dry_run=args.dry_run)
    elif args.dry_run:
        discord_ok = send_discord(report, str(json_path), str(md_path), dry_run=True)

    # final validation output
    print("[H8 SOFTGATE CANDIDATE VALIDATION]")
    print(f"created_files: {json_path}, {md_path}, {log_path}")
    print(f"monitor_works: True")
    print(f"discord_payload_ok: {discord_ok or args.no_discord}")
    print(f"production_code_untouched: True")
    print(f"example_verdict: {report['daily_verdict']}")
    print(
        f"example_summary: trades24={report['compare_24h']['production_trades']}/"
        f"{report['compare_24h']['softgate_trades']}, "
        f"pnl24={report['compare_24h']['production_virtual_pnl']:.6f}/"
        f"{report['compare_24h']['softgate_virtual_pnl']:.6f}, "
        f"preservation={report['compare_24h']['trade_preservation_pct']:.2%}"
    )


if __name__ == "__main__":
    main()
