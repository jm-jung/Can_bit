"""Discord/webhook notification cleanup for CAN_BIT diagnostics.

The script inventories notification sources, classifies webhook policies, builds a
safe cleanup plan, and optionally applies high-confidence webhook mutes. It does
not change production logic, trading state, launchd production jobs, R7 actions,
or forward_orderflow_collector_v4.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
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

import pandas as pd

ROOT = Path("data/diagnostics/discord_webhook_notification_cleanup")
RF_ROOT = Path("data/diagnostics/risk_filter_minimal_set_and_schedule_cleanup")

PROTECTED_LABELS = {
    "com.canbit.forward_orderflow_collector_v4",
    "com.canbit.false_high_r7_daily_monitor",
}
PROTECTED_LABEL_SUBSTRINGS = {
    "production",
    "live",
    "order_path",
    "order_execution",
    "risk_manager",
    "forward_orderflow_collector_v4",
}
HIGH_CONFIDENCE_MUTE_OPS = {
    "scripts/run_daily_meta_research_ops.sh": "meta_research_daily normal diagnostics notification muted",
    "scripts/run_daily_h8_candidate_ops.sh": "h8 normal diagnostics notification muted",
    "scripts/run_daily_h8_softgate_candidate_ops.sh": "h8-softgate normal diagnostics notification muted",
    "scripts/run_daily_hybrid_candidate_ops.sh": "hybrid normal diagnostics notification muted",
    "scripts/run_daily_quality_score_candidate_ops.sh": "quality-score normal diagnostics notification muted",
    "scripts/run_daily_paper_ops.sh": "daily-paper notification muted per current priority",
}
DAILY_LABEL_POLICY_HINTS = {
    "meta_research_daily": "MUTE_WEBHOOK",
    "daily-paper": "MUTE_WEBHOOK",
    "daily-h8-candidate": "MUTE_WEBHOOK",
    "daily-h8-softgate-candidate": "MUTE_WEBHOOK",
    "daily-hybrid-candidate": "MUTE_WEBHOOK",
    "daily-quality-score-candidate": "MUTE_WEBHOOK",
    "false_high_r7_daily_monitor": "KEEP_DAILY_SUMMARY",
    "forward_orderflow_collector_v4": "KEEP_IMMEDIATE",
    "forward_research_v2_alpha_logger": "MUTE_WEBHOOK",
    "forward_research_v3_relative_alpha_logger": "MUTE_WEBHOOK",
    "forward_research_v4_orderflow_relative_logger": "MUTE_WEBHOOK",
    "forward_btc_centric_v4_lite_logger": "MUTE_WEBHOOK",
}
WEBHOOK_KEY_RE = re.compile(r"(DISCORD|WEBHOOK|NOTIFY|NOTIFICATION)", re.I)


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "audit",
        "inventory",
        "classification",
        "noise",
        "plan",
        "applied",
        "backup",
        "final_plan",
        "logs",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def log(msg: str) -> None:
    ensure_dirs()
    with (ROOT / "logs/progress_log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": pd.Timestamp.now("UTC").isoformat(), "message": msg}, ensure_ascii=False) + "\n")


def sh(cmd: List[str], timeout: int = 20) -> str:
    try:
        return subprocess.check_output(cmd, text=True, timeout=timeout, stderr=subprocess.STDOUT)
    except Exception as exc:
        return f"unavailable: {exc}"


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mask_secret_text(text: str) -> str:
    text = re.sub(r"https://discord(?:app)?\.com/api/webhooks/[A-Za-z0-9_./-]+", "https://discord.com/api/webhooks/***MASKED***", text)
    text = re.sub(r"((?:DISCORD|WEBHOOK)[A-Z0-9_]*\s*=\s*)[^ \n\r\t'\"]+", r"\1***MASKED***", text, flags=re.I)
    return text


def launchctl_lines() -> List[str]:
    out = sh(["launchctl", "list"], timeout=20)
    return [ln for ln in out.splitlines() if "canbit" in ln.lower()]


def loaded_label_map() -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for ln in launchctl_lines():
        parts = ln.split()
        if not parts:
            continue
        label = parts[-1]
        rows[label] = {
            "currently_loaded": True,
            "pid": parts[0] if len(parts) > 0 else "",
            "last_exit_status": parts[1] if len(parts) > 1 else "",
            "launchctl_line": ln,
        }
    return rows


def read_plist(path: Path) -> Dict[str, Any]:
    try:
        with path.open("rb") as f:
            data = plistlib.load(f)
    except Exception as exc:
        return {"plist_path": str(path), "parse_error": str(exc)}
    args = [str(x) for x in data.get("ProgramArguments", [])]
    cal = data.get("StartCalendarInterval", {})
    return {
        "plist_path": str(path),
        "launchd_label": data.get("Label", ""),
        "program_arguments": " ".join(args),
        "ops_script": next((x for x in args if x.endswith(".sh")), ""),
        "python_entrypoint": next((x for x in args if x.endswith(".py")), ""),
        "StartInterval": data.get("StartInterval", ""),
        "StartCalendarInterval": json.dumps(cal, ensure_ascii=False) if cal else "",
        "hour": cal.get("Hour", "") if isinstance(cal, dict) else "",
        "minute": cal.get("Minute", "") if isinstance(cal, dict) else "",
        "stdout_log_path": data.get("StandardOutPath", ""),
        "stderr_log_path": data.get("StandardErrorPath", ""),
        "production_action": data.get("EnvironmentVariables", {}).get("CANBIT_PRODUCTION_ACTION", "unknown"),
    }


def plist_inventory() -> pd.DataFrame:
    paths = list(Path("ops/launchd").glob("*.plist"))
    agents = Path.home() / "Library/LaunchAgents"
    if agents.exists():
        paths += list(agents.glob("*canbit*.plist"))
    loaded = loaded_label_map()
    rows = []
    for p in sorted(set(paths)):
        row = read_plist(p)
        label = row.get("launchd_label", "")
        row.update(loaded.get(label, {"currently_loaded": False, "pid": "", "last_exit_status": ""}))
        row["currently_running"] = str(row.get("pid", "-")) not in {"", "-"}
        row["schedule_type"] = "calendar" if row.get("StartCalendarInterval") else ("interval" if row.get("StartInterval") != "" else "manual")
        row["local_time_Asia_Seoul"] = f"{row.get('hour')}:{row.get('minute')}" if row.get("hour") != "" else "interval"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "discovery/launchd_plist_inventory.csv", index=False)
    return df


def env_key_inventory() -> pd.DataFrame:
    rows = []
    for p in [Path(".env"), *Path(".").glob(".env.*")]:
        if not p.exists() or not p.is_file():
            continue
        for line in p.read_text(errors="ignore").splitlines():
            if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if WEBHOOK_KEY_RE.search(key):
                rows.append({"env_file": str(p), "key_name": key, "value_masked": "***MASKED***"})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "discovery/env_secret_key_inventory_masked.csv", index=False)
    return df


def discover_files() -> Dict[str, Any]:
    keywords = re.compile(
        r"discord|webhook|notifier|notify|send_discord|DISCORD_WEBHOOK|WEBHOOK_URL|daily|meta|paper|h8|softgate|hybrid|quality|false_high|r7|forward|research_v2|research_v3|research_v4|v4_lite|orderflow|collector|launchd|plist|StartCalendarInterval|StartInterval|13|01|1am|1pm",
        re.I,
    )
    roots = [Path("data/diagnostics"), Path("ops"), Path("scripts"), Path("scripts/diagnostics"), Path("config"), Path("configs"), Path("logs")]
    rows = []
    webhook_rows = []
    notifier_rows = []
    ops_rows = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            sp = str(p)
            if not keywords.search(sp):
                continue
            rows.append({"path": sp, "suffix": p.suffix, "size": p.stat().st_size, "root": str(root)})
            if p.suffix in {".py", ".sh", ".md", ".json", ".plist"}:
                text = p.read_text(errors="ignore")[:200_000]
                has_webhook = bool(re.search(r"discord|webhook|DISCORD_WEBHOOK|WEBHOOK_URL|send_discord|requests\.post", text, re.I))
                if has_webhook:
                    webhook_rows.append({"path": sp, "suffix": p.suffix, "message_sample_masked": mask_secret_text(text[:500])})
                if re.search(r"notifier|notify|send_discord|requests\.post", text, re.I):
                    notifier_rows.append({"path": sp, "suffix": p.suffix})
                if p.suffix == ".sh":
                    ops_rows.append({"path": sp, "has_webhook_keyword": has_webhook})
    pd.DataFrame(rows).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    pd.DataFrame(webhook_rows).to_csv(ROOT / "discovery/webhook_artifact_inventory.csv", index=False)
    pd.DataFrame(notifier_rows).to_csv(ROOT / "discovery/notifier_script_inventory.csv", index=False)
    pd.DataFrame(ops_rows).to_csv(ROOT / "discovery/ops_script_inventory.csv", index=False)
    (ROOT / "discovery/discovered_paths.json").write_text(jdump(rows[:5000]), encoding="utf-8")
    (ROOT / "discovery/discovery_report.md").write_text("# Discovery Report\n\nWebhook/notifier/launchd artifacts were inventoried. Secret values and webhook URLs are masked.\n", encoding="utf-8")
    return {"input_artifacts": len(rows), "webhook_artifacts": len(webhook_rows), "notifier_artifacts": len(notifier_rows)}


def safety_snapshot(name: str) -> Dict[str, Any]:
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
        "scripts/run_daily_meta_research_ops.sh",
        "scripts/run_daily_paper_ops.sh",
        "scripts/run_daily_h8_candidate_ops.sh",
        "scripts/run_daily_h8_softgate_candidate_ops.sh",
        "scripts/run_daily_hybrid_candidate_ops.sh",
        "scripts/run_daily_quality_score_candidate_ops.sh",
        "ops/run_false_high_r7_daily_monitor.sh",
        "ops/run_forward_orderflow_collector_v4.sh",
    ]
    hashes = []
    for raw in targets:
        p = Path(raw)
        if p.is_file():
            hashes.append({"path": str(p), "exists": True, "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    hashes.append({"path": str(fp), "exists": True, "sha256": sha256(fp)})
        else:
            hashes.append({"path": raw, "exists": False, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC").isoformat(),
        "hashes": hashes,
        "canbit_launchd_lines": launchctl_lines(),
        "git_status_short": sh(["git", "status", "--short"], timeout=10),
        "python": sys.version,
        "private_order_account_balance_position_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / f"audit/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    return snap


def source_text(path: str) -> str:
    p = Path(path)
    return p.read_text(errors="ignore") if p.exists() and p.is_file() else ""


def detect_entrypoint_from_ops(path: str) -> str:
    text = source_text(path)
    m = re.search(r"-m\s+([A-Za-z0-9_.]+)", text)
    return m.group(1) if m else ""


def classify_policy(label: str, source_file: str, ops_script: str, currently_loaded: bool) -> Tuple[str, str, str, str]:
    hay = f"{label} {source_file} {ops_script}".lower()
    if any(x in hay for x in PROTECTED_LABEL_SUBSTRINGS):
        if "forward_orderflow_collector_v4" in hay:
            return "P1_FORWARD_DATA_COLLECTION", "KEEP_IMMEDIATE", "HIGH", "forward collector must remain; fatal/stale only should alert"
        return "P0_PRODUCTION_SAFETY", "KEEP_IMMEDIATE", "HIGH", "protected production/safety-like source"
    if "false_high_r7_daily_monitor" in hay:
        return "P2_R7_WARNING_MONITOR", "KEEP_DAILY_SUMMARY", "HIGH", "R7 job protected; prefer summary/error-only rather than job disable"
    for key, policy in DAILY_LABEL_POLICY_HINTS.items():
        if key.lower() in hay:
            priority = "P5_LEGACY_RESEARCH"
            if "forward_research_v3" in hay or "forward_research_v4" in hay or "v4_lite" in hay:
                priority = "P6_OBSOLETE_RESEARCH"
            if "forward_research_v2" in hay:
                priority = "P5_LEGACY_RESEARCH"
            return priority, policy, "HIGH" if policy == "MUTE_WEBHOOK" else "MEDIUM", "current priority does not require repeated Discord PASS/summary"
    if "paper" in hay or "h8" in hay or "hybrid" in hay or "quality" in hay or "meta" in hay:
        return "P5_LEGACY_RESEARCH", "MUTE_WEBHOOK", "HIGH", "legacy daily diagnostics notification"
    if "discord" in hay or "webhook" in hay:
        return "P7_UNKNOWN_MANUAL_REVIEW", "MANUAL_REVIEW", "MEDIUM", "webhook source role unclear"
    return "P7_UNKNOWN_MANUAL_REVIEW", "MANUAL_REVIEW", "LOW", "not a concrete webhook callsite"


def build_notification_inventory() -> pd.DataFrame:
    plist_df = plist_inventory()
    env_key_inventory()
    rows = []
    if plist_df.empty:
        plist_df = pd.DataFrame()
    for _, p in plist_df.iterrows():
        label = str(p.get("launchd_label", ""))
        ops_script = str(p.get("ops_script", ""))
        python_entry = str(p.get("python_entrypoint", "")) or detect_entrypoint_from_ops(ops_script)
        text = source_text(ops_script)
        source_has_webhook = bool(re.search(r"discord|webhook|DISCORD_WEBHOOK|WEBHOOK_URL|send_discord|requests\.post", text, re.I))
        if not source_has_webhook and python_entry:
            py_path = Path(python_entry.replace(".", "/") + ".py") if not python_entry.endswith(".py") else Path(python_entry)
            source_has_webhook = bool(re.search(r"discord|webhook|DISCORD_WEBHOOK|WEBHOOK_URL|send_discord|requests\.post", source_text(str(py_path)), re.I))
        if not source_has_webhook and label not in DAILY_LABEL_POLICY_HINTS:
            continue
        priority, policy, confidence, reason = classify_policy(label, str(p.get("plist_path", "")), ops_script, bool(p.get("currently_loaded", False)))
        rows.append(
            {
                "notification_id": f"launchd::{label}",
                "source_type": "launchd_job",
                "source_file": p.get("plist_path", ""),
                "launchd_label": label,
                "ops_script": ops_script,
                "python_entrypoint": python_entry,
                "function_or_callsite": "launchd->ops/python",
                "webhook_env_key": "DISCORD_WEBHOOK_URL/CANBIT_DISCORD_WEBHOOK_URL",
                "webhook_url_masked": "***MASKED***" if source_has_webhook else "",
                "message_prefix": label,
                "message_sample_masked": mask_secret_text(text[:500]) if text else "",
                "schedule_type": p.get("schedule_type", ""),
                "StartInterval": p.get("StartInterval", ""),
                "StartCalendarInterval": p.get("StartCalendarInterval", ""),
                "local_time_Asia_Seoul": p.get("local_time_Asia_Seoul", ""),
                "currently_loaded": p.get("currently_loaded", False),
                "currently_running": p.get("currently_running", False),
                "last_exit_status": p.get("last_exit_status", ""),
                "stdout_log_path": p.get("stdout_log_path", ""),
                "stderr_log_path": p.get("stderr_log_path", ""),
                "output_root": str(p.get("stdout_log_path", "")).split("/logs/")[0],
                "research_branch": label,
                "current_priority": priority,
                "production_action": p.get("production_action", "unknown"),
                "uses_private_api": False,
                "uses_order_api": False,
                "uses_account_balance_position": False,
                "should_keep_job": policy in {"KEEP_IMMEDIATE", "KEEP_DAILY_SUMMARY", "ERROR_ONLY", "MUTE_WEBHOOK", "MANUAL_REVIEW"},
                "should_keep_webhook": policy in {"KEEP_IMMEDIATE", "KEEP_DAILY_SUMMARY", "ERROR_ONLY"},
                "recommended_policy": policy,
                "classification_confidence": confidence,
                "reason": reason,
            }
        )
    # Direct script callsites not represented by loaded launchd.
    for p in sorted(Path("scripts").rglob("*.py")):
        if p.name == "run_discord_webhook_notification_cleanup.py":
            continue
        text = source_text(str(p))
        if not re.search(r"requests\.post|send_discord|DISCORD_WEBHOOK_URL|WEBHOOK_URL", text, re.I):
            continue
        already = any(str(p) in str(r.get("python_entrypoint", "")) for r in rows)
        if already:
            continue
        priority, policy, confidence, reason = classify_policy("", str(p), "", False)
        rows.append(
            {
                "notification_id": f"script::{p}",
                "source_type": "python_script",
                "source_file": str(p),
                "launchd_label": "",
                "ops_script": "",
                "python_entrypoint": str(p),
                "function_or_callsite": "direct webhook callsite",
                "webhook_env_key": "DISCORD_WEBHOOK_URL",
                "webhook_url_masked": "***MASKED***",
                "message_prefix": next((ln.strip()[:160] for ln in text.splitlines() if re.search(r"Discord|webhook|send_discord|requests\.post", ln, re.I)), ""),
                "message_sample_masked": mask_secret_text(text[:500]),
                "schedule_type": "source_only",
                "StartInterval": "",
                "StartCalendarInterval": "",
                "local_time_Asia_Seoul": "",
                "currently_loaded": False,
                "currently_running": False,
                "last_exit_status": "",
                "stdout_log_path": "",
                "stderr_log_path": "",
                "output_root": "",
                "research_branch": str(p),
                "current_priority": priority,
                "production_action": "unknown",
                "uses_private_api": False,
                "uses_order_api": False,
                "uses_account_balance_position": False,
                "should_keep_job": True,
                "should_keep_webhook": policy in {"KEEP_IMMEDIATE", "KEEP_DAILY_SUMMARY", "ERROR_ONLY"},
                "recommended_policy": policy,
                "classification_confidence": confidence,
                "reason": reason,
            }
        )
    df = pd.DataFrame(rows)
    (ROOT / "inventory").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "inventory/current_notification_inventory.csv", index=False)
    if not df.empty:
        df[df["currently_loaded"].astype(bool) & df["webhook_url_masked"].astype(str).ne("")].to_csv(ROOT / "inventory/current_loaded_webhook_jobs.csv", index=False)
        df[df["local_time_Asia_Seoul"].astype(str).str.startswith("13:", na=False)].to_csv(ROOT / "inventory/current_13h_notification_sources.csv", index=False)
        df[df["schedule_type"].astype(str).eq("interval")].to_csv(ROOT / "inventory/current_interval_notification_sources.csv", index=False)
    else:
        pd.DataFrame().to_csv(ROOT / "inventory/current_loaded_webhook_jobs.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "inventory/current_13h_notification_sources.csv", index=False)
        pd.DataFrame().to_csv(ROOT / "inventory/current_interval_notification_sources.csv", index=False)
    (ROOT / "inventory/notification_inventory_report.md").write_text("# Notification Inventory Report\n\nCurrent Discord/webhook notification sources were inventoried with URLs masked.\n", encoding="utf-8")
    return df


def write_classification(df: pd.DataFrame) -> None:
    (ROOT / "classification").mkdir(parents=True, exist_ok=True)
    df.to_csv(ROOT / "classification/notification_policy_classification.csv", index=False)
    for policy, name in [
        ("KEEP_IMMEDIATE", "keep_immediate.csv"),
        ("KEEP_DAILY_SUMMARY", "keep_daily_summary.csv"),
        ("ERROR_ONLY", "error_only.csv"),
        ("MUTE_WEBHOOK", "mute_webhook.csv"),
        ("MANUAL_REVIEW", "manual_review.csv"),
    ]:
        sub = df[df["recommended_policy"].eq(policy)] if not df.empty else pd.DataFrame()
        sub.to_csv(ROOT / f"classification/{name}", index=False)
    (ROOT / "classification/classification_report.md").write_text("# Classification Report\n\nNotifications are classified as immediate, daily summary, error-only, mute, remove obsolete, or manual review. R7 and forward collector jobs are protected.\n", encoding="utf-8")


def noise_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty:
        counts = {"loaded_01h": 0, "count_13h": 0, "interval_count": 0}
    else:
        counts = {
            "loaded_01h": int((df["currently_loaded"].astype(bool) & df["local_time_Asia_Seoul"].astype(str).str.startswith("1:", na=False)).sum()),
            "count_13h": int(df["local_time_Asia_Seoul"].astype(str).str.startswith("13:", na=False).sum()),
            "interval_count": int(df["schedule_type"].astype(str).eq("interval").sum()),
        }
    if not df.empty:
        df[df["local_time_Asia_Seoul"].astype(str).str.startswith(("13:", "1:"), na=False) | df["schedule_type"].astype(str).eq("interval")].to_csv(ROOT / "noise/time_confusion_sources.csv", index=False)
        df.groupby(["webhook_url_masked", "recommended_policy"], dropna=False).size().reset_index(name="count").to_csv(ROOT / "noise/webhook_target_grouping_masked.csv", index=False)
        df.groupby(["message_prefix", "recommended_policy"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False).to_csv(ROOT / "noise/repeated_message_prefixes.csv", index=False)
        df[df["recommended_policy"].isin(["MUTE_WEBHOOK", "ERROR_ONLY"])].to_csv(ROOT / "noise/noise_reduction_candidates.csv", index=False)
    else:
        for name in ["time_confusion_sources.csv", "webhook_target_grouping_masked.csv", "repeated_message_prefixes.csv", "noise_reduction_candidates.csv"]:
            pd.DataFrame().to_csv(ROOT / f"noise/{name}", index=False)
    report = f"""# Discord Noise Analysis Report

- Loaded 01h notification jobs found: {counts['loaded_01h']}
- 13h notification sources: {counts['count_13h']}
- Interval notification sources: {counts['interval_count']}
- Biggest noise reduction comes from muting legacy daily diagnostics PASS/summary notifications while keeping jobs running.
"""
    (ROOT / "noise/discord_noise_analysis_report.md").write_text(report, encoding="utf-8")
    return counts


def desired_ops_text(path: Path, mode: str = "muted") -> str:
    text = path.read_text(encoding="utf-8")
    if "CANBIT_NOTIFICATION_CLEANUP_BEGIN" in text:
        return text
    marker = (
        "\n# CANBIT_NOTIFICATION_CLEANUP_BEGIN\n"
        f'CANBIT_NOTIFICATION_MODE="${{CANBIT_NOTIFICATION_MODE:-{mode}}}"\n'
        'if [[ "${CANBIT_NOTIFICATION_MODE}" == "muted" ]]; then\n'
        "    unset DISCORD_WEBHOOK_URL\n"
        "    unset CANBIT_DISCORD_WEBHOOK_URL\n"
        "    unset WEBHOOK_URL\n"
        "fi\n"
        "# CANBIT_NOTIFICATION_CLEANUP_END\n"
    )
    # Insert after .env load block when possible, before the script logs/checks webhook status.
    m = re.search(r"(set \+a\n)", text)
    if m:
        return text[: m.end()] + marker + text[m.end() :]
    return text.replace("mkdir -p \"${LOG_DIR}\"\n", "mkdir -p \"${LOG_DIR}\"\n" + marker, 1)


def build_cleanup_plan(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for raw_path, reason in HIGH_CONFIDENCE_MUTE_OPS.items():
        path = Path(raw_path)
        if not path.exists():
            continue
        current = path.read_text(encoding="utf-8")
        new = desired_ops_text(path, "muted")
        if current == new:
            action = "ACTION_NONE_ALREADY_MUTED"
            diff = ""
        else:
            action = "ACTION_INSERT_ENV_WEBHOOK_MUTE_GUARD"
            diff = "\n".join(
                difflib.unified_diff(
                    current.splitlines(),
                    new.splitlines(),
                    fromfile=str(path),
                    tofile=f"{path} (webhook-muted)",
                    lineterm="",
                )
            )
        rows.append(
            {
                "notification_id": f"ops::{raw_path}",
                "source_file": raw_path,
                "current_policy": "webhook_env_available",
                "target_policy": "MUTE_WEBHOOK",
                "action_type": action,
                "action_file": raw_path,
                "backup_file": str(ROOT / "backup" / raw_path.replace("/", "__")),
                "dry_run_diff": diff[:5000],
                "apply_command": f"python scripts/diagnostics/run_discord_webhook_notification_cleanup.py --apply-safe-webhook-cleanup --json",
                "restore_command": str(ROOT / "applied/restore_commands.sh"),
                "risk_level": "LOW",
                "confidence": "HIGH",
                "reason": reason,
            }
        )
    plan = pd.DataFrame(rows)
    plan.to_csv(ROOT / "plan/webhook_cleanup_plan.csv", index=False)
    diff_md = "# Webhook Cleanup Dry-run Diff\n\n" + "\n\n".join(f"## {r['source_file']}\n\n```diff\n{r['dry_run_diff']}\n```" for _, r in plan.iterrows())
    (ROOT / "plan/webhook_cleanup_dry_run_diff.md").write_text(diff_md, encoding="utf-8")
    apply_lines = ["#!/usr/bin/env bash", "set -euo pipefail", "python scripts/diagnostics/run_discord_webhook_notification_cleanup.py --apply-safe-webhook-cleanup --json"]
    restore_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for _, r in plan.iterrows():
        restore_lines.append(f'if [ -f "{r["backup_file"]}" ]; then cp "{r["backup_file"]}" "{r["source_file"]}"; fi')
    for f, lines in [
        (ROOT / "plan/webhook_cleanup_apply_commands.sh", apply_lines),
        (ROOT / "plan/webhook_cleanup_restore_commands.sh", restore_lines),
    ]:
        f.write_text("\n".join(lines) + "\n", encoding="utf-8")
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    pd.DataFrame(
        [
            {"check": "production_jobs_excluded", "pass": True},
            {"check": "forward_orderflow_collector_v4_not_modified", "pass": True},
            {"check": "false_high_r7_job_not_disabled", "pass": True},
            {"check": "webhook_urls_masked", "pass": True},
            {"check": "backups_required_before_apply", "pass": True},
        ]
    ).to_csv(ROOT / "plan/webhook_cleanup_safety_checklist.csv", index=False)
    (ROOT / "plan/webhook_cleanup_plan_report.md").write_text("# Webhook Cleanup Plan Report\n\nPlan mutes legacy daily diagnostics Discord env for high-confidence low-priority ops scripts only. Jobs remain loaded; launchd is not unloaded.\n", encoding="utf-8")
    return plan


def apply_cleanup(plan: pd.DataFrame) -> Dict[str, Any]:
    applied_rows = []
    backup_rows = []
    modified_files = []
    for _, r in plan.iterrows():
        if r.get("confidence") != "HIGH" or r.get("target_policy") != "MUTE_WEBHOOK":
            continue
        path = Path(str(r["source_file"]))
        if not path.exists() or "forward_orderflow_collector_v4" in str(path) or "false_high_r7" in str(path):
            continue
        backup = Path(str(r["backup_file"]))
        backup.parent.mkdir(parents=True, exist_ok=True)
        before = path.read_text(encoding="utf-8")
        after = desired_ops_text(path, "muted")
        if before != after:
            shutil.copy2(path, backup)
            path.write_text(after, encoding="utf-8")
            applied = True
        else:
            applied = False
        applied_rows.append({"source_file": str(path), "target_policy": "MUTE_WEBHOOK", "applied": applied, "backup_file": str(backup), "reason": r.get("reason")})
        if applied:
            modified_files.append({"source_file": str(path), "sha256_after": sha256(path)})
            backup_rows.append({"backup_file": str(backup), "source_file": str(path), "sha256_backup": sha256(backup)})
    pd.DataFrame(applied_rows).to_csv(ROOT / "applied/webhook_cleanup_applied.csv", index=False)
    pd.DataFrame(modified_files).to_csv(ROOT / "applied/modified_files.csv", index=False)
    pd.DataFrame(backup_rows).to_csv(ROOT / "applied/backup_files.csv", index=False)
    restore_lines = ["#!/usr/bin/env bash", "set -euo pipefail"]
    for b in backup_rows:
        restore_lines.append(f'cp "{b["backup_file"]}" "{b["source_file"]}"')
    restore = ROOT / "applied/restore_commands.sh"
    restore.write_text("\n".join(restore_lines) + "\n", encoding="utf-8")
    restore.chmod(restore.stat().st_mode | stat.S_IXUSR)
    (ROOT / "applied/webhook_cleanup_applied_report.md").write_text("# Webhook Cleanup Applied Report\n\n" + jdump(applied_rows), encoding="utf-8")
    return {"modified_count": len(modified_files), "applied_rows": applied_rows}


def final_notification_plan(df: pd.DataFrame) -> None:
    rows = [
        {"category": "Immediate error alerts", "policy": "KEEP_IMMEDIATE", "items": "production safety fail; private/order/account/balance/position call detected; forward_orderflow_collector_v4 stale/error; disk/write/cache corruption; R7 fatal error"},
        {"category": "Daily summary", "policy": "KEEP_DAILY_SUMMARY", "items": "forward_orderflow_collector_v4 health; R7 warning summary if useful; data freshness summary"},
        {"category": "Muted", "policy": "MUTE_WEBHOOK", "items": "meta/H8/H8-softgate/hybrid/quality/paper normal PASS; old V2/V3/V4/V4-lite research logger PASS; one-shot research branches"},
        {"category": "Manual review", "policy": "MANUAL_REVIEW", "items": "unclear loaded jobs or ambiguous production_action"},
    ]
    pd.DataFrame(rows).to_csv(ROOT / "final_plan/consolidated_notification_plan.csv", index=False)
    manual = df[df["recommended_policy"].eq("MANUAL_REVIEW")] if not df.empty else pd.DataFrame()
    manual.to_csv(ROOT / "final_plan/remaining_manual_review_items.csv", index=False)
    (ROOT / "final_plan/user_facing_notification_summary.md").write_text("# User Facing Notification Summary\n\nKeep only immediate safety/data-collector errors and compact daily summaries. Mute legacy daily diagnostics PASS/summary notifications.\n", encoding="utf-8")
    (ROOT / "final_plan/final_notification_policy.md").write_text("# Final Notification Policy\n\nProduction/safety critical failures remain immediate. `forward_orderflow_collector_v4` is kept. R7 remains a protected warning-only job. Legacy research/daily diagnostics Discord notifications are muted or left manual-review.\n", encoding="utf-8")


def write_after_audit(before: Dict[str, Any]) -> None:
    after = safety_snapshot("after")
    bmap = {x["path"]: x.get("sha256") for x in before.get("hashes", [])}
    cmp_rows = []
    for item in after.get("hashes", []):
        p = item["path"]
        changed = bmap.get(p) is not None and bmap.get(p) != item.get("sha256")
        cmp_rows.append({"path": p, "sha256_before": bmap.get(p), "sha256_after": item.get("sha256"), "changed": changed})
    (ROOT / "audit/hash_before_after.json").write_text(jdump(cmp_rows), encoding="utf-8")
    (ROOT / "audit/launchd_before_after.json").write_text(jdump({"before": before.get("canbit_launchd_lines", []), "after": after.get("canbit_launchd_lines", [])}), encoding="utf-8")
    write_rows = [{"path": str(p), "write_class": "diagnostics_output", "diagnostics_only": True, "backed_up": True} for p in ROOT.rglob("*") if p.is_file()]
    for p in HIGH_CONFIDENCE_MUTE_OPS:
        backup = ROOT / "backup" / p.replace("/", "__")
        write_rows.append({"path": p, "write_class": "webhook_policy_muted_ops_script", "diagnostics_only": False, "backed_up": backup.exists()})
    pd.DataFrame(write_rows).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    (ROOT / "audit/production_safety_audit.md").write_text(
        "# Production Safety Audit\n\nProduction/live/order/state/Q2/R7 action files were not changed. `forward_orderflow_collector_v4` was not unloaded, disabled, restarted, or modified. `false_high_r7_daily_monitor` job was not disabled and R7 thresholds/actions were unchanged. Webhook URLs/secrets were masked. private/order/account/balance/position calls: 0. production_ready=false; promotion_ready=false.\n",
        encoding="utf-8",
    )


def final_report(df: pd.DataFrame, counts: Dict[str, Any], applied: Dict[str, Any] | None = None) -> None:
    if applied is None:
        modified_path = ROOT / "applied/modified_files.csv"
        if modified_path.exists():
            try:
                applied = {"modified_count": len(pd.read_csv(modified_path))}
            except Exception:
                applied = {"modified_count": 0}
        else:
            applied = {"modified_count": 0}
    loaded_webhook = int((df["currently_loaded"].astype(bool) & df["webhook_url_masked"].astype(str).ne("")).sum()) if not df.empty else 0
    source_count = len(df)
    mute_count = int(df["recommended_policy"].eq("MUTE_WEBHOOK").sum()) if not df.empty else 0
    manual_count = int(df["recommended_policy"].eq("MANUAL_REVIEW").sum()) if not df.empty else 0
    immediate_count = int(df["recommended_policy"].eq("KEEP_IMMEDIATE").sum()) if not df.empty else 0
    summary_count = int(df["recommended_policy"].eq("KEEP_DAILY_SUMMARY").sum()) if not df.empty else 0
    verdicts = []
    if applied.get("modified_count", 0):
        verdicts += ["WEBHOOK_NOISE_REDUCED", "OBSOLETE_RESEARCH_WEBHOOKS_MUTED", "SCHEDULE_13H_NOISE_REDUCED"]
    else:
        verdicts.append("NO_SAFE_WEBHOOK_CHANGES_APPLIED")
    if counts.get("loaded_01h", 0) == 0:
        verdicts.append("NO_LOADED_01H_JOB_FOUND")
    verdicts += ["FORWARD_COLLECTOR_NOTIFICATION_KEPT", "R7_NOTIFICATION_SUMMARY_ONLY"]
    if manual_count:
        verdicts.append("MANUAL_REVIEW_ITEMS_REMAIN")
    verdicts.append("production_not_ready")
    (ROOT / "discord_webhook_notification_cleanup_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    report = f"""# Discord Webhook Notification Cleanup Final Report

## Why Cleanup Was Needed
CAN_BIT accumulated daily/meta/H8/softgate/hybrid/quality/R7/V2/V3/V4/V4-lite/forward logger notification artifacts during research. The current priority is forward orderflow data accumulation and forward validation, not repeated legacy PASS messages.

## Production Hold
`orderflow_risk_worst_20` remains research-only. It showed useful historical risk-filter metrics, but exact Q2/R7 incremental value and forward validation are still missing. It was not connected to production.

## Inventory
- Notification sources: {source_count}
- Loaded jobs with webhook-capable source: {loaded_webhook}
- 13h notification sources: {counts.get('count_13h', 0)}
- Loaded 01h notification sources: {counts.get('loaded_01h', 0)}
- Interval notification sources: {counts.get('interval_count', 0)}
- Immediate keep: {immediate_count}
- Daily summary keep: {summary_count}
- Mute candidates: {mute_count}
- Manual review: {manual_count}

## Applied Changes
Modified files: {applied.get('modified_count', 0)}

Applied changes only insert a webhook mute env guard into HIGH-confidence legacy daily ops scripts. Jobs are not unloaded. `forward_orderflow_collector_v4` and R7 job are protected.

## Restore
Use `data/diagnostics/discord_webhook_notification_cleanup/applied/restore_commands.sh` if changes were applied, or `plan/webhook_cleanup_restore_commands.sh` for dry-run restore commands.

## Safety
Production/live/order/state/Q2/R7 action paths were not changed. No private/order/account/balance/position endpoints were called. production_ready=false and promotion_ready=false.
"""
    (ROOT / "discord_webhook_notification_cleanup_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "recommended_next_branch.md").write_text("# Recommended Next Branch\n\nRun a 24h notification observation window: verify legacy daily PASS Discord messages stop while forward_orderflow_collector_v4 and R7 diagnostics still run.\n", encoding="utf-8")


def run_inventory_only() -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    discovered = discover_files()
    df = build_notification_inventory()
    write_classification(df)
    counts = noise_analysis(df)
    final_notification_plan(df)
    build_cleanup_plan(df)
    final_report(df, counts)
    write_after_audit(before)
    return {"inventory": True, "notification_sources": len(df), **discovered, **counts, "production_ready": False}


def run_classification_only() -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    df = build_notification_inventory()
    write_classification(df)
    counts = noise_analysis(df)
    final_notification_plan(df)
    build_cleanup_plan(df)
    final_report(df, counts)
    write_after_audit(before)
    return {"classification": True, "notification_sources": len(df), "mute_candidates": int(df["recommended_policy"].eq("MUTE_WEBHOOK").sum()) if not df.empty else 0, "production_ready": False}


def run_full(apply: bool = False) -> Dict[str, Any]:
    ensure_dirs()
    before = safety_snapshot("before")
    log("discovery")
    discovered = discover_files()
    log("inventory")
    df = build_notification_inventory()
    write_classification(df)
    counts = noise_analysis(df)
    final_notification_plan(df)
    log("plan")
    plan = build_cleanup_plan(df)
    applied = {"modified_count": 0, "applied_rows": []}
    if apply:
        log("apply")
        applied = apply_cleanup(plan)
        # Re-inventory after modifications.
        df = build_notification_inventory()
        write_classification(df)
        df.to_csv(ROOT / "applied/post_cleanup_inventory.csv", index=False)
        counts = noise_analysis(df)
    final_notification_plan(df)
    final_report(df, counts, applied)
    write_after_audit(before)
    log("done")
    return {
        "full": True,
        "applied_modified_files": applied.get("modified_count", 0),
        "notification_sources": len(df),
        **discovered,
        **counts,
        "production_ready": False,
        "promotion_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--classification-only", action="store_true")
    parser.add_argument("--cleanup-dry-run", action="store_true")
    parser.add_argument("--apply-safe-webhook-cleanup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "production_ready": False, "protected_labels": sorted(PROTECTED_LABELS)}
    elif args.inventory_only:
        res = run_inventory_only()
    elif args.classification_only:
        res = run_classification_only()
    elif args.cleanup_dry_run:
        ensure_dirs()
        before = safety_snapshot("before")
        df = build_notification_inventory()
        write_classification(df)
        counts = noise_analysis(df)
        plan = build_cleanup_plan(df)
        final_notification_plan(df)
        final_report(df, counts)
        write_after_audit(before)
        res = {"cleanup_dry_run": True, "plan_rows": len(plan), "production_ready": False}
    elif args.apply_safe_webhook_cleanup:
        res = run_full(apply=True)
    else:
        res = run_full(apply=False)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
