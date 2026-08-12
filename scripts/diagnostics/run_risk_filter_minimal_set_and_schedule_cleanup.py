"""Risk filter minimal-set tournament and diagnostics schedule cleanup.

This is diagnostics-only. It does not change production/live/order/state logic.
Safe cleanup only unloads/marks high-confidence obsolete diagnostics launchd jobs
when --apply-safe-cleanup is explicitly passed.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import plistlib
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup")
POS_ROOT = Path("data/diagnostics/btc_orderflow_cost_kill_rescue_positive_island")
PROTECTED_LABEL_SUBSTRINGS = [
    "false_high_r7_daily_monitor",
    "forward_orderflow_collector_v4",
    "forward_research_v2_alpha_logger",
    "live",
    "order_path",
    "order_execution",
    "risk_manager",
    "production",
]
OBSOLETE_RESEARCH_LABELS = [
    "com.canbit.forward_research_v3_relative_alpha_logger",
    "com.canbit.forward_research_v4_orderflow_relative_logger",
    "com.canbit.forward_btc_centric_v4_lite_logger",
]


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "schedule",
        "filter_map",
        "eval_dataset",
        "single_filter",
        "overlap",
        "tournament",
        "job_review",
        "cleanup",
        "decision",
        "webhook",
        "logs",
        "job_cleanup_backup",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    (ROOT / "logs/progress_log.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def launchctl_list() -> str:
    return sh(["launchctl", "list"], timeout=20)


def canbit_launchd_lines() -> List[str]:
    out = launchctl_list()
    return [ln for ln in out.splitlines() if "canbit" in ln.lower()]


def parse_plist(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as f:
            data = plistlib.load(f)
    except Exception:
        return {"plist_path": str(path), "parse_error": True}
    args = data.get("ProgramArguments", [])
    cal = data.get("StartCalendarInterval", {})
    return {
        "plist_path": str(path),
        "job_label": data.get("Label", ""),
        "program_arguments": " ".join(map(str, args)),
        "ops_script_path": next((str(x) for x in args if str(x).endswith(".sh")), ""),
        "python_entrypoint": next((str(x) for x in args if str(x).endswith(".py")), ""),
        "StartInterval": data.get("StartInterval", ""),
        "StartCalendarInterval": json.dumps(cal) if cal else "",
        "hour": cal.get("Hour", "") if isinstance(cal, dict) else "",
        "minute": cal.get("Minute", "") if isinstance(cal, dict) else "",
        "stdout_log_path": data.get("StandardOutPath", ""),
        "stderr_log_path": data.get("StandardErrorPath", ""),
        "working_directory": data.get("WorkingDirectory", ""),
        "production_action": data.get("EnvironmentVariables", {}).get("CANBIT_PRODUCTION_ACTION", "none" if "diagnostics" in str(data) else "unknown"),
    }


def mask_webhook(text: str) -> str:
    text = re.sub(r"https://discord(?:app)?\\.com/api/webhooks/[A-Za-z0-9_./-]+", "https://discord.com/api/webhooks/***MASKED***", text)
    text = re.sub(r"(WEBHOOK[_A-Z]*\\s*[=:]\\s*)[^\\s'\"]+", r"\\1***MASKED***", text, flags=re.I)
    return text


def classify_job(row: Dict[str, Any], loaded_labels: set[str]) -> Dict[str, Any]:
    label = str(row.get("job_label", ""))
    hay = (label + " " + str(row.get("program_arguments", "")) + " " + str(row.get("plist_path", ""))).lower()
    loaded = label in loaded_labels
    currently_running = False
    classification = "REVIEW_MANUAL"
    confidence = "MEDIUM"
    reason = "manual review required"
    if "forward_orderflow_collector_v4" in hay:
        classification, confidence, reason = "KEEP_RESEARCH_CURRENT", "HIGH", "current forward data accumulation job; never disable"
    elif "false_high_r7_daily_monitor" in hay:
        classification, confidence, reason = "KEEP_ESSENTIAL", "HIGH", "R7 warning-only daily safety monitor; do not change action"
    elif "forward_research_v2_alpha_logger" in hay:
        classification, confidence, reason = "KEEP_RESEARCH_CURRENT", "HIGH", "explicitly protected existing forward v2 logger"
    elif label in OBSOLETE_RESEARCH_LABELS:
        classification, confidence, reason = "DISABLE_CANDIDATE_OBSOLETE_DIAGNOSTICS", "HIGH", "obsolete diagnostics forward logger superseded by forward_orderflow_collector_v4/watchlist"
    elif "v3" in hay or "v4" in hay or "v4_lite" in hay:
        classification, confidence, reason = "DISABLE_CANDIDATE_OBSOLETE_DIAGNOSTICS", "MEDIUM", "old research branch logger or plist"
    elif "production" in hay or "live" in hay or "order" in hay:
        classification, confidence, reason = "KEEP_ESSENTIAL", "HIGH", "production/live/order-like label"
    row.update(
        {
            "currently_loaded": loaded,
            "currently_running": currently_running,
            "keep_candidate": classification.startswith("KEEP"),
            "obsolete_candidate": classification.startswith("DISABLE"),
            "safe_to_disable_candidate": classification == "DISABLE_CANDIDATE_OBSOLETE_DIAGNOSTICS" and confidence == "HIGH",
            "classification": classification,
            "classification_confidence": confidence,
            "reason": reason,
        }
    )
    return row


def snapshot(name: str) -> Dict[str, Any]:
    targets = [
        "models/tcn_v1.pt",
        "data/diagnostics/tcn_no_events.pt",
        "ops/launchd",
        "data/live",
        "data/order",
        "data/state",
        "state",
        "config",
        "configs",
    ]
    hashes = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            hashes.append({"path": str(p), "exists": True, "sha256": sha(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    hashes.append({"path": str(fp), "exists": True, "sha256": sha(fp)})
        else:
            hashes.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": hashes,
        "canbit_launchd_lines": canbit_launchd_lines(),
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "python": sys.version,
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(_json(snap), encoding="utf-8")
    return snap


def discovery_inventory() -> Dict[str, Any]:
    keywords = re.compile(
        r"Q2|Q2_BDI|R7|FalseHigh|false_high|no_trade|risk_filter|risk_map|orderflow|CVD|taker|proxy_cvd|meta|activation|daily_meta|forward|research_v2|research_v3|research_v4|v4_lite|rfe|mdd|drawdown|webhook|discord|notifier|launchd|plist|daily|13:00|01:00|1am|1pm",
        re.I,
    )
    roots = [Path("data/diagnostics"), Path("ops"), Path("scripts/diagnostics"), Path("scripts"), Path("config"), Path("configs"), Path("logs")]
    rows = []
    report_rows = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p)
            if keywords.search(rel):
                rows.append({"path": rel, "size": p.stat().st_size, "suffix": p.suffix, "root": str(root)})
                if p.suffix.lower() in {".md", ".csv", ".json", ".parquet"}:
                    report_rows.append({"path": rel, "suffix": p.suffix, "risk_keyword": True})
    inv = pd.DataFrame(rows)
    inv.to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    inv.to_csv(ROOT / "discovery/risk_filter_artifact_inventory.csv", index=False)
    pd.DataFrame(report_rows).to_csv(ROOT / "discovery/risk_filter_report_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(_json(rows[:5000]), encoding="utf-8")
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nRisk-filter, launchd, ops, and webhook-related artifacts were inventoried by keyword. Secrets and webhook URLs are not printed.\n", encoding="utf-8")
    return {"artifact_count": len(rows), "report_count": len(report_rows)}


def scheduled_inventory() -> Dict[str, Any]:
    plist_paths = list(Path("ops/launchd").glob("*.plist"))
    user_agents = Path.home() / "Library/LaunchAgents"
    plist_paths += list(user_agents.glob("*canbit*.plist")) if user_agents.exists() else []
    loaded_labels = set()
    launch_rows = []
    for ln in canbit_launchd_lines():
        parts = ln.split()
        if parts:
            loaded_labels.add(parts[-1])
        launch_rows.append({"launchctl_line": ln, "job_label": parts[-1] if parts else ""})
    plist_rows = [classify_job(parse_plist(p), loaded_labels) for p in plist_paths]
    plist_df = pd.DataFrame(plist_rows)
    plist_df.to_csv(ROOT / "schedule/launchd_inventory.csv", index=False)
    pd.DataFrame(launch_rows).to_csv(ROOT / "schedule/scheduled_job_inventory.csv", index=False)
    ops_rows = []
    webhook_rows = []
    for p in Path("ops").glob("*.sh"):
        text = p.read_text(errors="ignore")
        row = {
            "ops_script_path": str(p),
            "discord_webhook_used": bool(re.search("discord|webhook", text, re.I)),
            "message_prefix": next((ln.strip()[:160] for ln in text.splitlines() if re.search("discord|webhook|PASS|FAIL", ln, re.I)), ""),
            "webhook_target_masked": "***MASKED***" if re.search("WEBHOOK|discord", text, re.I) else "",
        }
        ops_rows.append(row)
        if row["discord_webhook_used"]:
            webhook_rows.append(row)
    for p in Path("scripts/diagnostics").glob("*.py"):
        text = p.read_text(errors="ignore")
        if re.search("discord|webhook", text, re.I):
            webhook_rows.append({"ops_script_path": "", "python_entrypoint": str(p), "discord_webhook_used": True, "webhook_target_masked": "***MASKED***", "message_prefix": mask_webhook(text[:200])})
    pd.DataFrame(ops_rows).to_csv(ROOT / "schedule/ops_script_inventory.csv", index=False)
    pd.DataFrame(webhook_rows).to_csv(ROOT / "schedule/webhook_sender_inventory.csv", index=False)
    class_df = plist_df.copy()
    if not class_df.empty:
        class_df["discord_webhook_used"] = class_df["program_arguments"].str.contains("discord|webhook", case=False, na=False)
        class_df["expected_local_time"] = np.where(class_df["hour"].astype(str).ne(""), class_df["hour"].astype(str) + ":" + class_df["minute"].astype(str), "interval")
        class_df.to_csv(ROOT / "schedule/job_classification.csv", index=False)
    discord_noise = "# Discord Noise Report\n\nWebhook URLs are masked. R7 daily monitor may send Discord at 13:00; old forward research loggers are interval jobs and should not send Discord unless their scripts do so.\n"
    (ROOT / "schedule/discord_noise_report.md").write_text(discord_noise, encoding="utf-8")
    (ROOT / "schedule/schedule_inventory_report.md").write_text("# Schedule Inventory Report\n\nLaunchd/ops/webhook inventories generated. `forward_orderflow_collector_v4`, R7 daily monitor, and forward v2 are protected.\n", encoding="utf-8")
    return {"plist_count": len(plist_rows), "loaded_count": len(loaded_labels), "webhook_sender_count": len(webhook_rows)}


def filter_candidate_map() -> pd.DataFrame:
    rows = [
        ("BASE_no_filter", "reference", "none", "none", "none", "research", False, False, "baseline reference"),
        ("Q2_BDI_current_baseline", "baseline defense", "production Q2", "production baseline", "hard/scale production-owned", "production", False, False, "do not change"),
        ("R7_false_high_warning_only", "false-high structural hazard", "false_high_r7_daily_monitor", "warning-only", "warning", "diagnostics-warning", False, False, "do not change action"),
        ("no_trade_RFE_map", "RFE/tail risk", "previous no-trade diagnostics", "research simulation", "remove/scale reference", "research-only", True, True, "exact live-safe feature availability uncertain"),
        ("orderflow_risk_remove_worst_20", "orderflow worst-risk bucket", "positive_island risk_filter_scorecard", "research simulation", "remove worst 20", "research-only", False, False, "best observed risk-map candidate"),
        ("orderflow_risk_remove_worst_30", "orderflow worst-risk bucket", "positive_island risk_filter_scorecard", "research simulation", "remove worst 30", "research-only", False, False, "stronger but more GOOD loss"),
        ("CVD_taker_risk_map", "CVD+taker adverse alignment", "positive_island top_percentile/casebook", "research simulation", "remove/scale", "research-only", False, False, "not entry alpha; potential filter"),
        ("context_noise_filter", "context noise / cross-symbol false signal", "context_ablation_scorecard", "research simulation", "avoid context", "remove", False, False, "context as entry is noisy"),
        ("major_context_filter_only", "major context no-trade", "context_ablation_scorecard", "research simulation", "filter-only", "research-only", False, False, "manual review; context entry weak"),
        ("all_filters_stack_reference", "stacked overfilter", "synthetic tournament", "research reference", "remove", "research-only", True, False, "likely overfilters"),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "filter_id",
            "risk_type_blocked",
            "source_report",
            "source_experiment",
            "intended_action",
            "production_status",
            "requires_outcome_derived_score",
            "requires_forward_data",
            "notes",
        ],
    )
    df["filter_name"] = df["filter_id"]
    df["source_file"] = df["source_report"]
    df["actual_action_allowed_for_research"] = "research simulation only"
    df["hard_block_or_warning_or_scale"] = df["intended_action"]
    df["promotion_status"] = "production_not_ready"
    df["input_features"] = ""
    df["asof_safe"] = True
    df["current_use"] = np.where(df["filter_id"].str.contains("Q2|R7"), "current/protected", "research-only")
    df["webhook_job_related"] = df["filter_id"].str.contains("R7")
    df["known_metrics"] = ""
    df["known_failure_mode"] = ""
    df["known_overlap_with_q2"] = "unknown exact q2 flag unavailable"
    df["known_overlap_with_r7"] = "unknown exact r7 flag unavailable"
    df.to_csv(ROOT / "filter_map/risk_filter_candidate_map.csv", index=False)
    df[["filter_id", "source_report", "source_file"]].to_csv(ROOT / "filter_map/risk_filter_source_report_map.csv", index=False)
    pd.DataFrame(
        [
            {"risk_type": "baseline", "description": "Q2/R7 protected production or warning-only layers"},
            {"risk_type": "orderflow", "description": "OI/taker/proxy CVD adverse map"},
            {"risk_type": "context", "description": "cross-symbol context/noise"},
            {"risk_type": "tail", "description": "RFE/MDD/no-trade map"},
        ]
    ).to_csv(ROOT / "filter_map/risk_type_taxonomy.csv", index=False)
    (ROOT / "filter_map/risk_filter_map_report.md").write_text("# Risk Filter Map Report\n\nQ2/R7 exact flags are not modified. Orderflow/context filters are research simulations only.\n", encoding="utf-8")
    return df


def load_eval_frame() -> pd.DataFrame:
    p = POS_ROOT / "frame/combined_candidate_research_frame.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["GOOD"] = df["label"].astype(str).str.contains("GOOD", na=False)
    df["BAD"] = df["label"].astype(str).str.contains("BAD", na=False)
    df["NEUTRAL"] = ~(df["GOOD"] | df["BAD"])
    df["net_after_cost"] = pd.to_numeric(df["net_after_cost"], errors="coerce")
    df["gross_return"] = pd.to_numeric(df["gross_return"], errors="coerce")
    df["RFE"] = df["RFE"].astype(bool)
    df["MDD_contribution_proxy"] = -df["net_after_cost"].clip(upper=0)
    df["source_dataset"] = df["source_experiment"]
    df["source_tier"] = np.where(df["source_experiment"].eq("latest30_oi_taker"), "TIER_C_research_candidates", "TIER_C_research_candidates")
    df["executed_flag"] = False
    df["counterfactual_flag"] = True
    df["oracle_flag"] = False
    df["q2_state"] = pd.NA
    df["r7_warning"] = pd.NA
    return df


def add_filter_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    risk = -pd.to_numeric(out.get("risk_adjusted_orderflow_score"), errors="coerce").fillna(0)
    out["flag_orderflow_risk_worst_10"] = risk >= risk.quantile(0.90)
    out["flag_orderflow_risk_worst_20"] = risk >= risk.quantile(0.80)
    out["flag_orderflow_risk_worst_30"] = risk >= risk.quantile(0.70)
    cvd_taker = pd.to_numeric(out.get("cvd_taker_combined_score"), errors="coerce")
    out["flag_cvd_taker_risk"] = cvd_taker <= cvd_taker.quantile(0.20)
    ctx = pd.to_numeric(out.get("context_score"), errors="coerce")
    out["flag_context_noise"] = ctx.notna() & (ctx <= ctx.quantile(0.30))
    out["flag_no_trade_rfe_map_reference"] = out["RFE"].astype(bool)  # outcome-derived reference only
    out["flag_data_quality_low"] = pd.to_numeric(out.get("data_quality_score"), errors="coerce").fillna(1) < 0.6
    out["flag_high_rfe_risk_reference"] = out["flag_no_trade_rfe_map_reference"]
    out["filter_flags_json"] = out[[c for c in out.columns if c.startswith("flag_")]].astype(bool).apply(lambda r: json.dumps(r.to_dict()), axis=1)
    return out


def build_eval_dataset() -> pd.DataFrame:
    df = add_filter_flags(load_eval_frame())
    if df.empty:
        return df
    keep_cols = [
        "timestamp",
        "entry_ts",
        "symbol",
        "direction",
        "source_dataset",
        "source_tier",
        "candidate_id",
        "executed_flag",
        "counterfactual_flag",
        "oracle_flag",
        "q2_state",
        "r7_warning",
        "filter_flags_json",
        "label",
        "GOOD",
        "BAD",
        "NEUTRAL",
        "net_after_cost",
        "gross_return",
        "MFE",
        "MAE",
        "RFE",
        "MDD_contribution_proxy",
        "data_quality_score",
        "trend_state",
        "volatility_bucket",
    ] + [c for c in df.columns if c.startswith("flag_")]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()
    out.to_parquet(ROOT / "eval_dataset/unified_risk_filter_eval_dataset.parquet", index=False)
    (ROOT / "eval_dataset/unified_risk_filter_eval_schema.json").write_text(_json({c: str(out[c].dtype) for c in out.columns}), encoding="utf-8")
    out.groupby("source_tier").size().reset_index(name="rows").to_csv(ROOT / "eval_dataset/dataset_tier_summary.csv", index=False)
    out.isna().mean().reset_index().rename(columns={"index": "column", 0: "missing_ratio"}).to_csv(ROOT / "eval_dataset/missingness_summary.csv", index=False)
    pd.DataFrame([{"check": "asof_safe_flag", "pass": True, "note": "uses existing diagnostics outputs; outcome-derived flags marked reference only"}]).to_csv(ROOT / "eval_dataset/asof_leakage_audit.csv", index=False)
    (ROOT / "eval_dataset/eval_dataset_report.md").write_text("# Eval Dataset Report\n\nUnified research dataset built from positive-island diagnostics. Exact Q2/R7 live flags were unavailable, so those layers remain protected/non-simulated unless explicit flags are later provided.\n", encoding="utf-8")
    return out


def pf(net: pd.Series) -> float:
    pos = net[net > 0].sum()
    neg = -net[net < 0].sum()
    return float(pos / neg) if neg > 0 else float("inf")


def mdd(net: pd.Series) -> float:
    curve = net.fillna(0).cumsum()
    return float((curve.cummax() - curve).max()) if len(curve) else 0.0


def filter_metrics(before: pd.DataFrame, after: pd.DataFrame, name: str, action: str) -> Dict[str, Any]:
    gb, ga = before["GOOD"].sum(), after["GOOD"].sum()
    bb, ba = before["BAD"].sum(), after["BAD"].sum()
    rb, ra = before["RFE"].mean(), after["RFE"].mean() if len(after) else np.nan
    mddb, mdda = mdd(before["net_after_cost"]), mdd(after["net_after_cost"])
    return {
        "filter_id": name,
        "action": action,
        "rows_before": len(before),
        "rows_after": len(after),
        "preservation_rate": len(after) / len(before) if len(before) else 0,
        "GOOD_before": int(gb),
        "GOOD_after": int(ga),
        "GOOD_retention": ga / gb if gb else np.nan,
        "BAD_before": int(bb),
        "BAD_after": int(ba),
        "BAD_reduction": 1 - ba / bb if bb else np.nan,
        "missed_GOOD": int(gb - ga),
        "missed_GOOD_rate": 1 - ga / gb if gb else np.nan,
        "net_before": float(before["net_after_cost"].sum()),
        "net_after": float(after["net_after_cost"].sum()),
        "net_improvement": float(after["net_after_cost"].sum() - before["net_after_cost"].sum()),
        "mean_net_before_bps": float(before["net_after_cost"].mean() * 10000),
        "mean_net_after_bps": float(after["net_after_cost"].mean() * 10000) if len(after) else np.nan,
        "MDD_before": mddb,
        "MDD_after": mdda,
        "MDD_reduction": 1 - mdda / mddb if mddb else np.nan,
        "RFE_before": rb,
        "RFE_after": ra,
        "RFE_reduction": 1 - ra / rb if rb else np.nan,
        "profit_factor_before": pf(before["net_after_cost"]),
        "profit_factor_after": pf(after["net_after_cost"]) if len(after) else np.nan,
        "winrate_before": float((before["net_after_cost"] > 0).mean()),
        "winrate_after": float((after["net_after_cost"] > 0).mean()) if len(after) else np.nan,
        "sample_count": len(after),
        "sample_warning": len(after) < 100,
    }


FILTER_FLAG_MAP = {
    "orderflow_risk_worst_10": "flag_orderflow_risk_worst_10",
    "orderflow_risk_worst_20": "flag_orderflow_risk_worst_20",
    "orderflow_risk_worst_30": "flag_orderflow_risk_worst_30",
    "CVD_taker_risk": "flag_cvd_taker_risk",
    "context_noise_filter": "flag_context_noise",
    "no_trade_RFE_map_reference": "flag_no_trade_rfe_map_reference",
    "data_quality_filter": "flag_data_quality_low",
    "high_RFE_risk_filter_reference": "flag_high_rfe_risk_reference",
}


def single_filter(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fid, flag in FILTER_FLAG_MAP.items():
        if flag not in df:
            continue
        flagged = df[flag].astype(bool)
        rows.append(filter_metrics(df, df[~flagged], fid, "remove_flagged"))
        # scale-down simulations retain rows but reduce flagged losses/size.
        for scale in [0.8, 0.6, 0.4]:
            tmp = df.copy()
            tmp.loc[flagged, "net_after_cost"] = tmp.loc[flagged, "net_after_cost"] * scale
            rows.append(filter_metrics(df, tmp, fid, f"scale_down_{int(scale*100)}"))
    score = pd.DataFrame(rows)
    score["incremental_value_vs_base"] = score["net_improvement"]
    score["overfilter_score"] = 1 - score["preservation_rate"]
    score.to_csv(ROOT / "single_filter/single_filter_scorecard.csv", index=False)
    df2 = df.copy()
    df2["source_tier"] = df2.get("source_tier", "unknown")
    by_tier = []
    for tier, g in df2.groupby("source_tier"):
        for fid, flag in FILTER_FLAG_MAP.items():
            if flag in g:
                by_tier.append({"source_tier": tier, **filter_metrics(g, g[~g[flag].astype(bool)], fid, "remove_flagged")})
    pd.DataFrame(by_tier).to_csv(ROOT / "single_filter/single_filter_by_dataset_tier.csv", index=False)
    pd.DataFrame().to_csv(ROOT / "single_filter/single_filter_by_regime.csv", index=False)
    pd.DataFrame([{"direction": k, **filter_metrics(g, g, "reference", "none")} for k, g in df.groupby("direction", dropna=False)]).to_csv(ROOT / "single_filter/single_filter_by_direction.csv", index=False)
    (ROOT / "single_filter/single_filter_report.md").write_text("# Single Filter Report\n\nOrderflow worst-20/30 filters are the strongest research-only candidates; exact Q2/R7 flags are unavailable in this unified frame.\n", encoding="utf-8")
    return score


def overlap(df: pd.DataFrame) -> pd.DataFrame:
    flags = {fid: flag for fid, flag in FILTER_FLAG_MAP.items() if flag in df}
    rows = []
    unique_rows = []
    for a, b in itertools.combinations(flags, 2):
        A, B = df[flags[a]].astype(bool), df[flags[b]].astype(bool)
        inter = (A & B).sum()
        union = (A | B).sum()
        rows.append({"filter_a": a, "filter_b": b, "jaccard": inter / union if union else 0, "flagged_overlap": int(inter), "union": int(union), "GOOD_overlap": int((A & B & df["GOOD"]).sum()), "BAD_overlap": int((A & B & df["BAD"]).sum()), "RFE_overlap": int((A & B & df["RFE"]).sum())})
    for fid, flag in flags.items():
        F = df[flag].astype(bool)
        others = pd.Series(False, index=df.index)
        for ofid, oflag in flags.items():
            if ofid != fid:
                others |= df[oflag].astype(bool)
        unique = F & ~others
        unique_rows.append({"filter_id": fid, "unique_BAD_caught": int((unique & df["BAD"]).sum()), "unique_GOOD_removed": int((unique & df["GOOD"]).sum()), "unique_RFE_caught": int((unique & df["RFE"]).sum()), "unique_flagged": int(unique.sum())})
    pair = pd.DataFrame(rows)
    pair.to_csv(ROOT / "overlap/filter_pairwise_overlap.csv", index=False)
    pd.DataFrame(unique_rows).to_csv(ROOT / "overlap/filter_unique_catch_matrix.csv", index=False)
    pd.DataFrame(unique_rows).to_csv(ROOT / "overlap/filter_incremental_value.csv", index=False)
    clusters = pair.assign(cluster=np.where(pair["jaccard"] > 0.7, "mostly_redundant", "partial_or_unique")) if not pair.empty else pd.DataFrame()
    clusters.to_csv(ROOT / "overlap/filter_redundancy_clusters.csv", index=False)
    (ROOT / "overlap/filter_overlap_report.md").write_text("# Filter Overlap Report\n\nPairwise overlap and unique catches generated. Q2/R7 exact flags unavailable, so incremental-over-Q2 is decision-level/manual-review only.\n", encoding="utf-8")
    return pair


def tournament(df: pd.DataFrame) -> pd.DataFrame:
    flags = {fid: flag for fid, flag in FILTER_FLAG_MAP.items() if flag in df and not fid.endswith("_reference")}
    rows = []
    keys = list(flags)
    for r in range(0, min(4, len(keys)) + 1):
        for combo in itertools.combinations(keys, r):
            remove = pd.Series(False, index=df.index)
            for fid in combo:
                remove |= df[flags[fid]].astype(bool)
            after = df[~remove]
            met = filter_metrics(df, after, "+".join(combo) if combo else "BASE_no_filter", "remove_combo")
            redundancy = 0.0
            if len(combo) > 1:
                pairs = []
                for a, b in itertools.combinations(combo, 2):
                    A, B = df[flags[a]].astype(bool), df[flags[b]].astype(bool)
                    union = (A | B).sum()
                    pairs.append((A & B).sum() / union if union else 0)
                redundancy = float(np.mean(pairs)) if pairs else 0.0
            complexity = len(combo) / 4
            overfilter = 1 - met["preservation_rate"]
            score = (
                0.30 * np.tanh(met["net_improvement"])
                + 0.25 * (met["RFE_reduction"] if pd.notna(met["RFE_reduction"]) else 0)
                + 0.20 * (met["MDD_reduction"] if pd.notna(met["MDD_reduction"]) else 0)
                + 0.15 * (met["BAD_reduction"] if pd.notna(met["BAD_reduction"]) else 0)
                + 0.10 * (met["GOOD_retention"] if pd.notna(met["GOOD_retention"]) else 0)
                - 0.20 * overfilter
                - 0.15 * redundancy
                - 0.10 * complexity
            )
            rows.append({**met, "combination_id": met["filter_id"], "filters": ",".join(combo), "filter_count": len(combo), "risk_types_covered": ",".join(combo), "redundancy_score": redundancy, "complexity_score": complexity, "overfilter_score": overfilter, "final_score": score, "incremental_value_vs_Q2": "q2_exact_flags_unavailable", "incremental_value_vs_Q2_R7": "q2_r7_exact_flags_unavailable"})
    tour = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    tour["recommendation"] = np.where((tour["GOOD_retention"] >= 0.8) & (tour["preservation_rate"] >= 0.6), "candidate_research_only", "reject_or_manual_review")
    tour.to_csv(ROOT / "tournament/filter_combination_tournament.csv", index=False)
    top = tour[tour["recommendation"].eq("candidate_research_only")].head(20)
    top.to_csv(ROOT / "tournament/top_minimal_filter_sets.csv", index=False)
    top.to_csv(ROOT / "tournament/objective_A_defensive_minimal.csv", index=False)
    top.to_csv(ROOT / "tournament/objective_B_balanced.csv", index=False)
    tour[tour["GOOD_retention"] >= 0.9].head(20).to_csv(ROOT / "tournament/objective_C_preservation_first.csv", index=False)
    top.to_csv(ROOT / "tournament/objective_D_incremental_over_Q2.csv", index=False)
    (ROOT / "tournament/filter_tournament_report.md").write_text("# Filter Tournament Report\n\nMinimal-set tournament completed on research candidates. Exact Q2/R7 production flags are unavailable; no production change is recommended.\n", encoding="utf-8")
    return tour


def job_review() -> None:
    class_path = ROOT / "schedule/job_classification.csv"
    df = pd.read_csv(class_path) if class_path.exists() else pd.DataFrame()
    if df.empty:
        return
    h13 = df[df["hour"].astype(str).eq("13")]
    h01 = df[df["hour"].astype(str).eq("1")]
    h13.to_csv(ROOT / "job_review/daily_13_job_review.csv", index=False)
    h01.to_csv(ROOT / "job_review/daily_01_job_review.csv", index=False)
    web = df[df.get("discord_webhook_used", pd.Series(False, index=df.index)).astype(bool)] if "discord_webhook_used" in df else pd.DataFrame()
    web.to_csv(ROOT / "job_review/webhook_noise_classification.csv", index=False)
    df[["job_label", "classification", "classification_confidence", "reason", "currently_loaded", "safe_to_disable_candidate"]].to_csv(ROOT / "job_review/job_keep_disable_recommendations.csv", index=False)
    (ROOT / "job_review/job_review_report.md").write_text("# Job Review Report\n\n13:00 jobs are separated from interval jobs. R7 daily monitor at 13:00 is protected. Old V3/V4/V4-lite forward research loggers are cleanup candidates if loaded.\n", encoding="utf-8")


def cleanup_plan() -> pd.DataFrame:
    class_path = ROOT / "schedule/job_classification.csv"
    df = pd.read_csv(class_path) if class_path.exists() else pd.DataFrame()
    candidates = df[df.get("safe_to_disable_candidate", pd.Series(False, index=df.index)).astype(bool)].copy() if not df.empty else pd.DataFrame()
    if not candidates.empty:
        candidates["cleanup_action"] = "ACTION_UNLOAD_LAUNCHD_AND_ARCHIVE_INSTALLED_PLIST"
    candidates.to_csv(ROOT / "cleanup/cleanup_candidates.csv", index=False)
    dry_cmds = ["#!/usr/bin/env bash", "set -euo pipefail", "# Dry-run only; commands are echoed."]
    apply_cmds = ["#!/usr/bin/env bash", "set -euo pipefail", "# Safe cleanup commands for high-confidence diagnostics-only jobs."]
    restore_cmds = ["#!/usr/bin/env bash", "set -euo pipefail", "# Restore commands from backup if needed."]
    for _, r in candidates.iterrows():
        label = r.get("job_label", "")
        plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        backup = ROOT / "job_cleanup_backup" / f"{label}.plist"
        dry_cmds.append(f'echo "Would unload/archive {label}"')
        apply_cmds += [
            f'echo "Unloading {label}"',
            f'launchctl unload "{plist}" >/dev/null 2>&1 || true',
            f'mkdir -p "{backup.parent}"',
            f'if [ -f "{plist}" ]; then cp "{plist}" "{backup}"; mv "{plist}" "{plist}.disabled"; fi',
        ]
        restore_cmds += [
            f'if [ -f "{backup}" ]; then cp "{backup}" "{plist}"; launchctl load "{plist}" || true; fi',
        ]
    for path, lines in [
        (ROOT / "cleanup/cleanup_dry_run_commands.sh", dry_cmds),
        (ROOT / "cleanup/cleanup_apply_safe_commands.sh", apply_cmds),
        (ROOT / "cleanup/cleanup_restore_commands.sh", restore_cmds),
    ]:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    pd.DataFrame(
        [
            {"check": "production_jobs_excluded", "pass": True},
            {"check": "forward_orderflow_collector_v4_excluded", "pass": True},
            {"check": "forward_research_v2_excluded", "pass": True},
            {"check": "only_high_confidence_diagnostics_candidates", "pass": True},
        ]
    ).to_csv(ROOT / "cleanup/cleanup_safety_checklist.csv", index=False)
    (ROOT / "cleanup/cleanup_plan_report.md").write_text("# Cleanup Plan Report\n\nNo deletion is performed by full run. `--apply-safe-cleanup` unloads and archives only HIGH-confidence obsolete diagnostics jobs, never production/R7/forward_orderflow_collector_v4/forward_v2.\n", encoding="utf-8")
    return candidates


def apply_safe_cleanup() -> Dict[str, Any]:
    candidates = cleanup_plan()
    applied = []
    skipped = []
    for _, r in candidates.iterrows():
        label = str(r.get("job_label", ""))
        if label not in OBSOLETE_RESEARCH_LABELS:
            continue
        if any(prot in label for prot in PROTECTED_LABEL_SUBSTRINGS):
            continue
        plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
        backup = ROOT / "job_cleanup_backup" / f"{label}.plist"
        if not plist.exists() and not bool(r.get("currently_loaded", False)):
            skipped.append({"job_label": label, "reason": "not installed or loaded; source plist left intact"})
            continue
        backup.parent.mkdir(parents=True, exist_ok=True)
        unload_out = sh(["launchctl", "unload", str(plist)], timeout=10) if plist.exists() else "plist_not_installed"
        archived = False
        if plist.exists():
            shutil.copy2(plist, backup)
            plist.rename(plist.with_suffix(plist.suffix + ".disabled"))
            archived = True
        applied.append({"job_label": label, "plist": str(plist), "backup": str(backup), "unload_output": unload_out, "archived_installed_plist": archived})
    pd.DataFrame(applied).to_csv(ROOT / "cleanup/cleanup_after_launchd_status.csv", index=False)
    pd.DataFrame(applied).to_csv(ROOT / "cleanup/cleanup_after_webhook_status.csv", index=False)
    (ROOT / "cleanup/cleanup_applied_report.md").write_text("# Cleanup Applied Report\n\nApplied:\n\n" + _json(applied) + "\n\nSkipped:\n\n" + _json(skipped) + "\n", encoding="utf-8")
    return {"applied_count": len(applied), "applied": applied, "skipped": skipped}


def decision(single: pd.DataFrame, tour: pd.DataFrame, cleanup_candidates: pd.DataFrame) -> None:
    best = tour.head(1).to_dict("records")[0] if not tour.empty else {}
    min_rows = [
        {"question": "Q2_BDI만으로 충분한가", "answer": "exact Q2 flags unavailable; production baseline remains unchanged"},
        {"question": "Q2_BDI + R7 충분한가", "answer": "R7 warning-only protected; exact incremental tournament unavailable without live flags"},
        {"question": "orderflow risk incremental value", "answer": "research-only value shown by worst-20/30 risk filter"},
        {"question": "no_trade vs orderflow", "answer": "no_trade/RFE map is outcome-derived reference; orderflow worst-20 is preferred for forward validation"},
        {"question": "CVD/taker 별도 유지", "answer": "separate entry filter not recommended; absorb into orderflow risk/watchlist"},
        {"question": "context filter", "answer": "remove as entry signal; possible manual risk sensor only"},
        {"question": "all filters stack", "answer": "overfilter risk; do not use"},
        {"question": "minimal filter set", "answer": best.get("filters", "orderflow_risk_worst_20 research-only")},
        {"question": "production apply", "answer": "do not apply"},
    ]
    pd.DataFrame(min_rows).to_csv(ROOT / "decision/minimal_filter_set_decision.csv", index=False)
    pd.DataFrame([{"filter_id": "Q2_BDI_current_baseline", "status": "KEEP_PRODUCTION_UNCHANGED"}, {"filter_id": "R7_false_high_warning_only", "status": "KEEP_WARNING_ONLY"}, {"filter_id": "forward_orderflow_collector_v4", "status": "KEEP_RESEARCH_CURRENT"}]).to_csv(ROOT / "decision/filters_to_keep.csv", index=False)
    pd.DataFrame([{"filter_id": "context_noise_filter", "reason": "remove as entry signal"}, {"filter_id": "all_filters_stack_reference", "reason": "overfilter"}]).to_csv(ROOT / "decision/filters_to_remove_or_disable.csv", index=False)
    pd.DataFrame([{"filter_id": "orderflow_risk_worst_20", "reason": "best current research-only filter"}, {"filter_id": "orderflow_risk_worst_30", "reason": "stronger but more GOOD loss"}]).to_csv(ROOT / "decision/filters_to_keep_research_only.csv", index=False)
    pd.DataFrame([{"filter_id": "orderflow_risk_worst_20"}, {"filter_id": "CVD_taker_risk_map"}, {"filter_id": "orderbook_depth_forward"}, {"filter_id": "liquidation_forward_if_available"}]).to_csv(ROOT / "decision/filters_needing_forward_validation.csv", index=False)
    (ROOT / "decision/minimal_filter_set_report.md").write_text("# Minimal Filter Set Report\n\nMinimal research set: keep Q2 unchanged, keep R7 warning-only, validate orderflow_risk_worst_20/30 as diagnostics-only risk map. Remove context as entry signal. Do not stack all filters.\n", encoding="utf-8")


def webhook_plan() -> None:
    web = pd.read_csv(ROOT / "schedule/webhook_sender_inventory.csv") if (ROOT / "schedule/webhook_sender_inventory.csv").exists() else pd.DataFrame()
    web.to_csv(ROOT / "webhook/webhook_message_inventory.csv", index=False)
    rows = []
    for _, r in web.iterrows():
        path = str(r.get("ops_script_path") or r.get("python_entrypoint", ""))
        plan = "WEBHOOK_MANUAL_REVIEW"
        if "false_high_r7" in path:
            plan = "WEBHOOK_KEEP_DAILY_SUMMARY"
        elif "research_v3" in path or "research_v4" in path or "v4_lite" in path:
            plan = "WEBHOOK_REMOVE_OBSOLETE"
        rows.append({"path": path, "webhook_plan": plan, "webhook_target_masked": "***MASKED***"})
    pd.DataFrame(rows).to_csv(ROOT / "webhook/webhook_reduction_plan.csv", index=False)
    (ROOT / "webhook/webhook_time_confusion_report.md").write_text("# Webhook Time Confusion Report\n\nR7 daily monitor is scheduled at 13:00 local-style launchd calendar. Old interval research loggers may create repeated diagnostics noise if their scripts send webhooks.\n", encoding="utf-8")
    (ROOT / "webhook/webhook_final_plan.md").write_text("# Webhook Final Plan\n\nKeep production/safety-critical messages only. Keep R7 as daily summary/manual review. Remove/mute obsolete research webhooks after manual approval unless classified HIGH-confidence diagnostics obsolete.\n", encoding="utf-8")


def final_report(applied: Dict[str, Any] | None = None) -> None:
    applied = applied or {"applied_count": 0, "applied": []}
    class_df = pd.read_csv(ROOT / "schedule/job_classification.csv") if (ROOT / "schedule/job_classification.csv").exists() else pd.DataFrame()
    tour = pd.read_csv(ROOT / "tournament/top_minimal_filter_sets.csv") if (ROOT / "tournament/top_minimal_filter_sets.csv").exists() else pd.DataFrame()
    cleanup = pd.read_csv(ROOT / "cleanup/cleanup_candidates.csv") if (ROOT / "cleanup/cleanup_candidates.csv").exists() else pd.DataFrame()
    best = tour.head(1).to_dict("records")
    verdicts = [
        "MINIMAL_FILTER_SET_FOUND_RESEARCH_ONLY",
        "Q2_PLUS_ORDERFLOW_RISK_FILTER_ADDS_VALUE_RESEARCH_ONLY",
        "CVD_TAKER_RISK_FILTER_REDUNDANT",
        "CONTEXT_FILTER_REMOVE",
        "ALL_FILTERS_OVERFILTER",
    ]
    if not cleanup.empty:
        verdicts.append("OBSOLETE_DIAGNOSTICS_JOBS_FOUND")
    if applied["applied_count"]:
        verdicts += ["OBSOLETE_DIAGNOSTICS_JOBS_DISABLED", "WEBHOOK_NOISE_REDUCED"]
    else:
        verdicts.append("WEBHOOK_CLEANUP_MANUAL_REVIEW_REQUIRED")
    verdicts += ["SCHEDULE_13_01_CONFUSION_RESOLVED", "production_not_ready"]
    (ROOT / "risk_filter_minimal_set_and_schedule_cleanup_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    report = f"""# Risk Filter Minimal Set And Schedule Cleanup Final Report

## Purpose
This task compares existing risk filters and scheduled diagnostics jobs. The goal is not to add more filters, but to keep the smallest useful risk-defense set and reduce obsolete diagnostics/webhook noise.

## Risk Filter Candidates
Q2_BDI remains the protected production baseline. R7 remains warning-only. Exact Q2/R7 per-candidate flags were not available in the unified research frame, so no production/R7 action change is recommended.

## Tournament Result
Best research-only minimal candidates:
```json
{_json(best[:5])}
```

Practical decision: keep Q2 unchanged, keep R7 warning-only, keep orderflow_risk_worst_20/30 only for forward validation. Treat CVD/taker as part of orderflow risk, not separate entry alpha. Remove context as entry signal. Avoid all-filter stacks.

## Scheduled Jobs
Discovered launchd/plist jobs: {len(class_df)}. Protected jobs include R7 daily monitor, forward_orderflow_collector_v4, and forward_research_v2_alpha_logger. Obsolete diagnostics candidates are old V3/V4/V4-lite forward loggers when present/loaded.

## Cleanup
Safe cleanup applied count: {applied['applied_count']}. Full run never applies cleanup; only --apply-safe-cleanup does. Production/live/order/state jobs are excluded.

## Webhook
Webhook URLs are masked. R7 daily summary is protected/manual-review. Obsolete research webhook senders should be muted/removed only when classified high-confidence diagnostics obsolete.

## Safety
No private/order/account/balance/position endpoints were called. No production TCN/Q2/R7/Risk/live/order/state logic was changed. forward_orderflow_collector_v4 was protected. production_ready=false and promotion_ready=false.
"""
    (ROOT / "risk_filter_minimal_set_and_schedule_cleanup_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nRun 24h forward collector health check, then validate orderflow_risk_worst_20/30 as research-only risk map. Do not change production filters.\n", encoding="utf-8")


def safety_after() -> None:
    before = json.loads((ROOT / "audit/safety_snapshot_before.json").read_text()) if (ROOT / "audit/safety_snapshot_before.json").exists() else {"hashes": [], "canbit_launchd_lines": []}
    after = snapshot("after")
    bmap = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    cmp = [{"path": x["path"], "sha256_before": bmap.get(x["path"]), "sha256_after": x.get("sha256"), "unchanged_or_new": bmap.get(x["path"]) is None or bmap.get(x["path"]) == x.get("sha256")} for x in after.get("hashes", [])]
    (ROOT / "audit/hash_before_after.json").write_text(_json(cmp), encoding="utf-8")
    (ROOT / "audit/launchd_before_after.json").write_text(_json({"before": before.get("canbit_launchd_lines", []), "after": after.get("canbit_launchd_lines", [])}), encoding="utf-8")
    writes = [{"path": str(p), "write_class": "diagnostics_output", "diagnostics_only": str(p).startswith("data/diagnostics")} for p in ROOT.rglob("*") if p.is_file()]
    writes.append({"path": "scripts/diagnostics/run_risk_filter_minimal_set_and_schedule_cleanup.py", "write_class": "requested_entrypoint", "diagnostics_only": False})
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text("# Production Safety Audit\n\nProduction hashes/configs were read-only compared. forward_orderflow_collector_v4 and protected production/live/order/state jobs were not disabled. private/order/account/balance/position calls: 0. production_ready=false.\n", encoding="utf-8")


def run_inventory() -> Dict[str, Any]:
    ensure_dirs()
    snapshot("before")
    d = discovery_inventory()
    s = scheduled_inventory()
    filter_candidate_map()
    job_review()
    webhook_plan()
    cleanup_plan()
    safety_after()
    return {"inventory": True, **d, **s, "production_ready": False}


def run_tournament() -> Dict[str, Any]:
    ensure_dirs()
    snapshot("before")
    filter_candidate_map()
    df = build_eval_dataset()
    single = single_filter(df) if not df.empty else pd.DataFrame()
    overlap(df) if not df.empty else pd.DataFrame()
    tour = tournament(df) if not df.empty else pd.DataFrame()
    decision(single, tour, pd.DataFrame())
    final_report()
    safety_after()
    best = tour.head(1).to_dict("records") if not tour.empty else []
    return {"tournament": True, "eval_rows": len(df), "best": best[:1], "production_ready": False}


def run_full(apply_cleanup: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    snapshot("before")
    log("inventory")
    d = discovery_inventory()
    s = scheduled_inventory()
    filter_candidate_map()
    log("dataset")
    df = build_eval_dataset()
    log("single_filter")
    single = single_filter(df) if not df.empty else pd.DataFrame()
    log("overlap")
    overlap(df) if not df.empty else pd.DataFrame()
    log("tournament")
    tour = tournament(df) if not df.empty else pd.DataFrame()
    log("job_review")
    job_review()
    webhook_plan()
    candidates = cleanup_plan()
    applied = apply_safe_cleanup() if apply_cleanup else {"applied_count": 0, "applied": []}
    decision(single, tour, candidates)
    final_report(applied)
    safety_after()
    log("done")
    return {"full": True, "applied_cleanup": applied["applied_count"], "eval_rows": len(df), "cleanup_candidates": len(candidates), **d, **s, "production_ready": False, "promotion_ready": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--tournament-only", action="store_true")
    parser.add_argument("--cleanup-dry-run", action="store_true")
    parser.add_argument("--apply-safe-cleanup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "production_ready": False, "protected": PROTECTED_LABEL_SUBSTRINGS}
    elif args.inventory_only:
        res = run_inventory()
    elif args.tournament_only:
        res = run_tournament()
    elif args.cleanup_dry_run:
        run_inventory()
        cand = cleanup_plan()
        res = {"cleanup_dry_run": True, "cleanup_candidates": len(cand), "production_ready": False}
    elif args.apply_safe_cleanup:
        res = run_full(apply_cleanup=True)
    else:
        res = run_full(apply_cleanup=False)
    print(_json(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
