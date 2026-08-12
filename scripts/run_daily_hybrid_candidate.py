"""
Hybrid_A Candidate Daily Monitor (diagnostics only).

Compares Production vs H8 SoftGate G30 vs Hybrid_A (G30 + hard block).
No live orders, no paper state mutation, no TradingEngine execution.
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
from scripts.diagnostics.validate_hybrid_gate_replay import (
    HYBRID_CANDIDATES,
    _loss_cluster_count,
    _ohlcv_features,
    _simulate_hybrid,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper
from scripts.run_daily_shadow import (
    RECENT_WINDOW_HOURS,
    diagnose_proba_source,
    _parse_tick_timestamp,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("hybrid_candidate")

MONITORING_DIR = Path("data/monitoring")
ENTROPY_H8 = 0.96
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
POSITION_SIZE = 0.05
FAIL_SCALE = 0.30

DISCLAIMER = (
    "진단 후보 모니터입니다. 실제 주문 없음. 운영 로직 미변경. "
    "G30 + HardBlock(Hybrid_A) 비교용입니다."
)

BASELINE_VARIANT = Variant("Baseline", fail_scale=1.0, hard_block=False)
G30_VARIANT = Variant("Variant_G30", fail_scale=FAIL_SCALE, hard_block=False)
HYBRID_A = HYBRID_CANDIDATES[0]


def _entropy(t: Dict[str, Any]) -> float:
    e = 0.0
    for p in (float(t["p_long"]), float(t["p_short"]), float(t["p_flat"])):
        if p > 1e-10:
            e -= p * math.log(p)
    return e


def _h8_pass(t: Dict[str, Any]) -> bool:
    return (_entropy(t) <= ENTROPY_H8) and (str(t.get("trend_label") or "") == "up")


def _hard_block_a(t: Dict[str, Any]) -> bool:
    return (
        t.get("signal") == "LONG"
        and str(t.get("trend_label") or "") == "down"
        and str(t.get("vol_bucket") or "") == "high"
    )


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


def _candidate_funnel(ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    h8_pass = h8_fail = hard_block = blocked_entropy = blocked_trend = 0
    prod_c = 0
    for t in ticks:
        ent = _entropy(t)
        if not _production_entry_ok(t, ent):
            continue
        prod_c += 1
        if _hard_block_a(t):
            hard_block += 1
        if _h8_pass(t):
            h8_pass += 1
        else:
            h8_fail += 1
            tr_up = str(t.get("trend_label") or "") == "up"
            ent096 = ent <= ENTROPY_H8
            if (not ent096) and tr_up:
                blocked_entropy += 1
            elif ent096 and (not tr_up):
                blocked_trend += 1
    return {
        "production_candidates": prod_c,
        "h8_pass_count": h8_pass,
        "h8_fail_count": h8_fail,
        "hard_block_count": hard_block,
        "blocked_by_hardblock": hard_block,
        "blocked_by_entropy": blocked_entropy,
        "blocked_by_trend": blocked_trend,
    }


def _final_equity(tdf: pd.DataFrame) -> float:
    if tdf.empty:
        return 1.0
    eq = 1.0
    for r in tdf["scaled_return"]:
        eq *= 1.0 + float(r) * POSITION_SIZE
    return float(eq)


def _metrics_from_sim(m: Dict[str, Any], tdf: pd.DataFrame, ticks: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = tdf["scaled_return"].tolist() if not tdf.empty else []
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
    scales = tdf["scale"].tolist() if not tdf.empty and "scale" in tdf.columns else []
    return {
        "trade_count": int(m["trades"]),
        "win_rate": float(m["win_rate"]),
        "avg_return": float(np.mean(vals)) if vals else 0.0,
        "expectancy": float(m["expectancy"]),
        "profit_factor": float(pf),
        "net_return": float(m["net_return"]),
        "max_drawdown": float(m["MDD"]),
        "final_equity": _final_equity(tdf),
        "risk_force_exit_count": int(m["risk_force_exit_count"]),
        "kill_switch_triggered": bool(m["KS_triggered"]),
        "avg_hold_bars": float(m["avg_hold"]),
        "loss_cluster_count": _loss_cluster_count(tdf, ticks),
        "h8_pass_trades": int((tdf["h8_pass"] == True).sum()) if not tdf.empty and "h8_pass" in tdf.columns else 0,
        "h8_fail_trades": int((tdf["h8_pass"] == False).sum()) if not tdf.empty and "h8_pass" in tdf.columns else 0,
        "avg_applied_scale": float(np.mean(scales)) if scales else 1.0,
        "scale_distribution": {str(k): int(v) for k, v in Counter(scales).items()},
    }


def _hybrid_metrics(m: Dict[str, Any], tdf: pd.DataFrame, ticks: List[Dict[str, Any]], blocks: int) -> Dict[str, Any]:
    base = _metrics_from_sim(m, tdf, ticks)
    base["hard_block_count"] = int(blocks)
    return base


def _daily_verdict_24h(prod: Dict[str, Any], g30: Dict[str, Any], hyb: Dict[str, Any]) -> str:
    pt = int(prod.get("trade_count", 0))
    gt = int(g30.get("trade_count", 0))
    ht = int(hyb.get("trade_count", 0))
    if pt + gt + ht < 5:
        return "INSUFFICIENT_SAMPLE"
    if ht == 0 and gt == 0:
        return "NO_TRADE"
    g_net = float(g30.get("net_return", 0))
    h_net = float(hyb.get("net_return", 0))
    g_mdd = float(g30.get("max_drawdown", 0))
    h_mdd = float(hyb.get("max_drawdown", 0))
    if h_net < g_net or h_mdd < g_mdd:
        return "HYBRID_WORSE"
    if h_net > g_net and h_mdd >= g_mdd:
        return "HYBRID_BETTER"
    return "HYBRID_WORSE"


def _structural_notes(
    g30_full: Dict[str, Any],
    hyb_full: Dict[str, Any],
    preservation: float,
) -> Dict[str, Any]:
    net_d = hyb_full["net_return"] - g30_full["net_return"]
    mdd_d = hyb_full["max_drawdown"] - g30_full["max_drawdown"]
    rfe_red = (1.0 - hyb_full["risk_force_exit_count"] / max(g30_full["risk_force_exit_count"], 1)) * 100.0
    improving = net_d > 0 and mdd_d >= 0
    stable = preservation >= 0.85
    overfiltering = preservation < 0.85
    unstable = net_d < 0 and mdd_d < 0
    return {
        "improving": improving,
        "stable": stable,
        "overfiltering": overfiltering,
        "unstable": unstable,
        "rfe_reduction_pct_vs_g30": round(rfe_red, 2),
        "net_delta_vs_g30": round(net_d, 6),
        "mdd_delta_vs_g30": round(mdd_d, 6),
    }


def build_report(
    ticks: List[Dict[str, Any]],
    proba_meta: Dict[str, Any],
    feat_map: Dict[int, Dict[str, Any]],
    dry_run: bool = False,
) -> Dict[str, Any]:
    ts_run = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proba_diag = diagnose_proba_source(None, proba_meta)
    recent = _slice_recent_ticks(ticks)
    funnel_full = _candidate_funnel(ticks)
    funnel_24h = _candidate_funnel(recent)

    prod_m, prod_tdf = _simulate_variant(ticks, BASELINE_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    g30_m, g30_tdf = _simulate_variant(ticks, G30_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    hyb_m, hyb_tdf, hyb_blocks = _simulate_hybrid(ticks, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)

    prod_full = _metrics_from_sim(prod_m, prod_tdf, ticks)
    g30_full = _metrics_from_sim(g30_m, g30_tdf, ticks)
    hyb_full = _hybrid_metrics(hyb_m, hyb_tdf, ticks, hyb_blocks)

    prod_24m, prod_tdf_24 = _simulate_variant(recent, BASELINE_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    g30_24m, g30_tdf_24 = _simulate_variant(recent, G30_VARIANT, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    hyb_24m, hyb_tdf_24, hyb_blocks_24 = _simulate_hybrid(
        recent, HYBRID_A, feat_map, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE
    )
    prod_24 = _metrics_from_sim(prod_24m, prod_tdf_24, recent)
    g30_24 = _metrics_from_sim(g30_24m, g30_tdf_24, recent)
    hyb_24 = _hybrid_metrics(hyb_24m, hyb_tdf_24, recent, hyb_blocks_24)

    pres_vs_g30 = hyb_full["trade_count"] / max(g30_full["trade_count"], 1)
    pres_vs_prod = hyb_full["trade_count"] / max(prod_full["trade_count"], 1)
    notes = _structural_notes(g30_full, hyb_full, pres_vs_g30)
    verdict = _daily_verdict_24h(prod_24, g30_24, hyb_24)

    ks_red = (
        (1.0 - int(hyb_full["kill_switch_triggered"]) / max(int(g30_full["kill_switch_triggered"]), 1)) * 100.0
        if g30_full["kill_switch_triggered"]
        else (100.0 if not hyb_full["kill_switch_triggered"] else 0.0)
    )

    return {
        "mode": "hybrid_a_candidate_monitor",
        "disclaimer": DISCLAIMER,
        "run_timestamp": ts_run,
        "dry_run": dry_run,
        "hybrid_a_policy": {
            "base": "H8 SoftGate G30",
            "h8_pass_scale": 1.0,
            "h8_fail_scale": FAIL_SCALE,
            "hard_block": "LONG AND trend==down AND vol==high",
        },
        "proba_health": {
            "proba_loaded": proba_diag.get("proba_loaded"),
            "fallback_used": proba_diag.get("fallback", {}).get("fallback_used"),
            "aligned_ratio": proba_diag.get("alignment", {}).get("aligned_ratio"),
            "ohlcv_ts_max": proba_diag.get("alignment", {}).get("ohlcv_ts_max"),
            "proba_ts_max": proba_diag.get("alignment", {}).get("proba_ts_max"),
        },
        "candidate_summary_full": funnel_full,
        "candidate_summary_24h": funnel_24h,
        "production_full": prod_full,
        "g30_full": g30_full,
        "hybrid_full": hyb_full,
        "production_24h": prod_24,
        "g30_24h": g30_24,
        "hybrid_24h": hyb_24,
        "compare_24h": {
            "production_trades": prod_24["trade_count"],
            "g30_trades": g30_24["trade_count"],
            "hybrid_trades": hyb_24["trade_count"],
            "production_net_return": prod_24["net_return"],
            "g30_net_return": g30_24["net_return"],
            "hybrid_net_return": hyb_24["net_return"],
            "production_mdd": prod_24["max_drawdown"],
            "g30_mdd": g30_24["max_drawdown"],
            "hybrid_mdd": hyb_24["max_drawdown"],
            "trade_preservation_vs_g30": hyb_24["trade_count"] / max(g30_24["trade_count"], 1),
            "trade_preservation_vs_prod": hyb_24["trade_count"] / max(prod_24["trade_count"], 1),
        },
        "hybrid_effect_full": {
            **notes,
            "ks_reduction_pct_vs_g30": round(ks_red, 2),
            "loss_cluster_delta": hyb_full["loss_cluster_count"] - g30_full["loss_cluster_count"],
        },
        "blocked_distribution_full": {
            "blocked_by_hardblock": funnel_full["blocked_by_hardblock"],
            "blocked_by_entropy": funnel_full["blocked_by_entropy"],
            "blocked_by_trend": funnel_full["blocked_by_trend"],
        },
        "structural_notes": notes,
        "daily_verdict_24h": verdict,
        "proba_source_diagnostics": proba_diag,
        "trade_preservation_vs_g30_full": pres_vs_g30,
        "trade_preservation_vs_prod_full": pres_vs_prod,
    }


def _write_markdown(report: Dict[str, Any], path: Path) -> None:
    c = report["compare_24h"]
    pf, gf, hf = report["production_full"], report["g30_full"], report["hybrid_full"]
    he = report["hybrid_effect_full"]
    bd = report["blocked_distribution_full"]
    sn = report["structural_notes"]
    lines = [
        "# Hybrid_A Candidate Daily Monitor",
        "",
        report["disclaimer"],
        "",
        f"- Run: {report['run_timestamp']}",
        "",
        "## Trades (full)",
        f"| | Production | G30 | Hybrid_A |",
        f"|--|------------|-----|----------|",
        f"| trades | {pf['trade_count']} | {gf['trade_count']} | {hf['trade_count']} |",
        f"| preservation vs G30 | — | — | {report['trade_preservation_vs_g30_full']:.2%} |",
        "",
        "## Equity (full)",
        f"| net_return | {pf['net_return']:.6f} | {gf['net_return']:.6f} | {hf['net_return']:.6f} |",
        f"| MDD | {pf['max_drawdown']:.6f} | {gf['max_drawdown']:.6f} | {hf['max_drawdown']:.6f} |",
        f"| final_equity | {pf['final_equity']:.6f} | {gf['final_equity']:.6f} | {hf['final_equity']:.6f} |",
        "",
        "## Risk (full)",
        f"| RFE | {pf['risk_force_exit_count']} | {gf['risk_force_exit_count']} | {hf['risk_force_exit_count']} |",
        f"| KS | {pf['kill_switch_triggered']} | {gf['kill_switch_triggered']} | {hf['kill_switch_triggered']} |",
        f"| loss_cluster | {pf['loss_cluster_count']} | {gf['loss_cluster_count']} | {hf['loss_cluster_count']} |",
        "",
        "## Hybrid Effect",
        f"- RFE reduction vs G30: {he['rfe_reduction_pct_vs_g30']:.1f}%",
        f"- net Δ vs G30: {he['net_delta_vs_g30']:.6f}",
        f"- MDD Δ vs G30: {he['mdd_delta_vs_g30']:.6f}",
        "",
        "## 24h",
        f"- trades: {c['production_trades']} / {c['g30_trades']} / {c['hybrid_trades']}",
        f"- net: {c['production_net_return']:.6f} / {c['g30_net_return']:.6f} / {c['hybrid_net_return']:.6f}",
        f"- verdict: **{report['daily_verdict_24h']}**",
        "",
        "## Blocked Distribution",
        f"- hardblock: {bd['blocked_by_hardblock']}",
        f"- entropy: {bd['blocked_by_entropy']}",
        f"- trend: {bd['blocked_by_trend']}",
        "",
        "## Structural Notes",
        f"- improving={sn['improving']}, stable={sn['stable']}, overfiltering={sn['overfiltering']}, unstable={sn['unstable']}",
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

    cs = report["candidate_summary_full"]
    c24 = report["compare_24h"]
    pf, gf, hf = report["production_full"], report["g30_full"], report["hybrid_full"]
    he = report["hybrid_effect_full"]
    bd = report["blocked_distribution_full"]
    sn = report["structural_notes"]
    ph = report["proba_health"]

    fields = [
        {
            "name": "📋 [0] 실행 정보",
            "value": (
                f"**mode**: diagnostic\n"
                f"**{report['disclaimer']}**\n"
                f"**timestamp**: {report['run_timestamp']}"
            ),
            "inline": False,
        },
        {
            "name": "🧪 [1] Candidate Summary",
            "value": (
                f"Hybrid_A: G30 + HARD BLOCK\n"
                f"LONG + trend down + high vol\n"
                f"H8 pass/fail (candidates): {cs['h8_pass_count']} / {cs['h8_fail_count']}\n"
                f"hard_block_count: {cs['hard_block_count']}"
            ),
            "inline": False,
        },
        {
            "name": "📈 [2] Trades (full)",
            "value": (
                f"prod / G30 / Hybrid: {pf['trade_count']} / {gf['trade_count']} / {hf['trade_count']}\n"
                f"preservation vs G30: {report['trade_preservation_vs_g30_full']:.1%}"
            ),
            "inline": False,
        },
        {
            "name": "💰 [3] Equity (full)",
            "value": (
                f"net prod/G30/Hyb: {pf['net_return']:.6f} / {gf['net_return']:.6f} / {hf['net_return']:.6f}\n"
                f"MDD prod/G30/Hyb: {pf['max_drawdown']:.4f} / {gf['max_drawdown']:.4f} / {hf['max_drawdown']:.4f}\n"
                f"final_eq: {pf['final_equity']:.6f} / {gf['final_equity']:.6f} / {hf['final_equity']:.6f}"
            ),
            "inline": False,
        },
        {
            "name": "🛡️ [4] Risk (full)",
            "value": (
                f"RFE prod/G30/Hyb: {pf['risk_force_exit_count']} / {gf['risk_force_exit_count']} / {hf['risk_force_exit_count']}\n"
                f"KS: {pf['kill_switch_triggered']} / {gf['kill_switch_triggered']} / {hf['kill_switch_triggered']}\n"
                f"loss_cluster: {pf['loss_cluster_count']} / {gf['loss_cluster_count']} / {hf['loss_cluster_count']}"
            ),
            "inline": False,
        },
        {
            "name": "⚡ [5] Hybrid Effect vs G30",
            "value": (
                f"RFE reduction: {he['rfe_reduction_pct_vs_g30']:.1f}%\n"
                f"KS reduction: {he['ks_reduction_pct_vs_g30']:.1f}%\n"
                f"net Δ: {he['net_delta_vs_g30']:.6f}\n"
                f"MDD Δ: {he['mdd_delta_vs_g30']:.6f}"
            ),
            "inline": False,
        },
        {
            "name": "⏱️ [6] 24h Window",
            "value": (
                f"trades: {c24['production_trades']} / {c24['g30_trades']} / {c24['hybrid_trades']}\n"
                f"net: {c24['production_net_return']:.6f} / {c24['g30_net_return']:.6f} / {c24['hybrid_net_return']:.6f}\n"
                f"**verdict**: {report['daily_verdict_24h']}"
            ),
            "inline": False,
        },
        {
            "name": "🚫 [7] Blocked Distribution",
            "value": (
                f"hardblock: {bd['blocked_by_hardblock']}\n"
                f"entropy: {bd['blocked_by_entropy']}\n"
                f"trend: {bd['blocked_by_trend']}"
            ),
            "inline": False,
        },
        {
            "name": "📊 [8] Structural Notes",
            "value": (
                f"improving: {sn['improving']}\n"
                f"stable: {sn['stable']}\n"
                f"overfiltering: {sn['overfiltering']}\n"
                f"unstable: {sn['unstable']}"
            ),
            "inline": False,
        },
        {
            "name": "📁 Logs",
            "value": f"JSON: `{json_path}`\nMD: `{md_path}`",
            "inline": False,
        },
    ]

    verdict = report["daily_verdict_24h"]
    color = 0x2ecc71 if verdict == "HYBRID_BETTER" else 0xe74c3c if verdict == "HYBRID_WORSE" else 0xf1c40f
    embed = {
        "title": "【Hybrid Candidate Ops】Hybrid_A 일일 요약",
        "description": report["disclaimer"],
        "color": color,
        "fields": fields,
        "footer": {"text": "Can_bit Hybrid_A Monitor (diagnostic only)"},
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
        logger.info("[Discord] Hybrid candidate report sent")
        return True
    except Exception as e:
        logger.error("[Discord] send failed: %s", e)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid_A candidate daily monitor")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-discord", action="store_true")
    args = parser.parse_args()

    df = load_ohlcv()
    if df is None or df.empty:
        logger.error("OHLCV load failed")
        sys.exit(1)

    feat_map = _ohlcv_features(df)
    ticks, proba_meta = simulate_signals_paper(df)
    if not ticks:
        logger.error("No ticks")
        sys.exit(1)

    report = build_report(ticks, proba_meta, feat_map, dry_run=args.dry_run)

    MONITORING_DIR.mkdir(parents=True, exist_ok=True)
    ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = MONITORING_DIR / f"hybrid_candidate_daily_report_{ts_file}.json"
    md_path = MONITORING_DIR / f"hybrid_candidate_daily_report_{ts_file}.md"
    log_path = MONITORING_DIR / f"hybrid_candidate_log_{datetime.now().strftime('%Y%m%d')}.jsonl"

    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _write_markdown(report, md_path)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {"ts": report["run_timestamp"], "verdict": report["daily_verdict_24h"], "report": report},
                default=str,
            )
            + "\n"
        )

    discord_ok = False
    if not args.no_discord:
        discord_ok = send_discord(report, str(json_path), str(md_path), dry_run=args.dry_run)
    elif args.dry_run:
        discord_ok = send_discord(report, str(json_path), str(md_path), dry_run=True)

    print("[HYBRID CANDIDATE VALIDATION]")
    print(f"created_files: {json_path}, {md_path}, {log_path}")
    print(f"discord_send_ok: {discord_ok or args.no_discord}")
    print(f"example_verdict: {report['daily_verdict_24h']}")
    he = report["hybrid_effect_full"]
    print(
        f"example_summary: trades={report['production_full']['trade_count']}/"
        f"{report['g30_full']['trade_count']}/{report['hybrid_full']['trade_count']}, "
        f"RFE_red={he['rfe_reduction_pct_vs_g30']:.1f}%, "
        f"preservation={report['trade_preservation_vs_g30_full']:.1%}, "
        f"net_hybrid={report['hybrid_full']['net_return']:.6f}"
    )


if __name__ == "__main__":
    main()
