"""Daily observation-quality Discord notifier (diagnostics-only)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "data/diagnostics/microstructure_daily_discord_notifier"
LOGS = ROOT / "logs"
STATE = ROOT / "state"
OUTBOX = ROOT / "outbox"
OUTBOX_ARCHIVED = ROOT / "outbox/archived"
REPORTS = ROOT / "reports"
VALIDATION = ROOT / "validation"

PYTHON = REPO / ".venv/bin/python"
SUITE = REPO / "scripts/diagnostics/run_microstructure_observation_quality_suite.py"

DEFAULT_SECRET = Path.home() / ".config/can_bit/discord_observation_quality_webhook"
KST = ZoneInfo("Asia/Seoul")
LAUNCHD_LABEL = "com.canbit.microstructure-observation-quality-discord-daily"

COLOR_NORMAL = 0x2ECC71
COLOR_WARNING = 0xF1C40F
COLOR_CRITICAL = 0xE74C3C
COLOR_READY = 0x3498DB
COLOR_TEST = 0x95A5A6

READY_SET = {
    "READY_FOR_FORMAL_EVALUATION",
    "READY_FOR_FORMAL_EVALUATION_WITH_WARNINGS",
}
STABLE_OK = {"STABLE", "RECENTLY_RECOVERED"}

DAILY_UPDATE_TIMEOUT = 600
COMPACT_TIMEOUT = 120
HTTP_TIMEOUT = 15
MAX_RETRIES = 4
RETRY_SLEEPS = (0, 10, 30, 60)
MAX_RETRY_AFTER = 90

WEBHOOK_HOST_RE = re.compile(
    r"^https://(?:discord|discordapp)\.com/api/webhooks/\d+/[A-Za-z0-9_-]+/?$",
    re.I,
)


def ensure_dirs() -> None:
    for p in (LOGS, STATE, OUTBOX, OUTBOX_ARCHIVED, REPORTS, VALIDATION, ROOT / "preflight", ROOT / "backups", ROOT / "plist_templates"):
        p.mkdir(parents=True, exist_ok=True)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def now_kst() -> datetime:
    return now_utc().astimezone(KST)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def redact_text(text: str, secrets: Iterable[str] = ()) -> str:
    out = text or ""
    for s in secrets:
        if s:
            out = out.replace(s, "***REDACTED***")
    out = re.sub(
        r"https://(?:discord|discordapp)\.com/api/webhooks/\S+",
        "https://discord.com/api/webhooks/***REDACTED***",
        out,
        flags=re.I,
    )
    return out


def mask_webhook_url(url: str) -> str:
    if not url:
        return ""
    return "https://discord.com/api/webhooks/***REDACTED***"


def expected_complete_day_utc(when: Optional[datetime] = None) -> str:
    """At KST ~09:05, the UTC day that just completed is yesterday UTC (= today KST date - 1 day in UTC calendar sense).

    More precisely: evaluate the latest fully completed UTC calendar day,
    which at local morning after 00:05 UTC is yesterday's UTC date.
    """
    ts = when or now_utc()
    completed = (ts.astimezone(timezone.utc) - timedelta(days=1)).date()
    # If we're past 00:05 UTC, yesterday is complete; before that still previous.
    # Spec example: KST 2026-07-30 09:05 = UTC 2026-07-30 00:05 → evaluate 2026-07-29.
    return completed.isoformat()


def inspect_secret(path: Path = DEFAULT_SECRET) -> Dict[str, Any]:
    present = path.exists()
    mode = None
    perms_ok = False
    if present:
        mode = oct(path.stat().st_mode & 0o777)
        perms_ok = (path.stat().st_mode & 0o777) == 0o600
    parent = path.parent
    parent_mode = oct(parent.stat().st_mode & 0o777) if parent.exists() else None
    return {
        "secret_path": str(path),
        "present": present,
        "permissions": mode,
        "permissions_ok": perms_ok,
        "parent_exists": parent.exists(),
        "parent_permissions": parent_mode,
    }


def load_webhook_url(path: Path = DEFAULT_SECRET) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = inspect_secret(path)
    if not meta["present"]:
        meta["error"] = "WEBHOOK_SECRET_MISSING"
        return None, meta
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        meta["error"] = "WEBHOOK_SECRET_EMPTY"
        return None, meta
    if not WEBHOOK_HOST_RE.match(raw):
        meta["error"] = "WEBHOOK_SECRET_INVALID_FORMAT"
        return None, meta
    if not meta["permissions_ok"]:
        meta["warning"] = "WEBHOOK_SECRET_PERMISSIONS_NOT_600"
    meta["masked"] = mask_webhook_url(raw)
    return raw, meta


def run_suite(args: List[str], stdout_path: Path, stderr_path: Path, timeout: int) -> Dict[str, Any]:
    ensure_dirs()
    cmd = [str(PYTHON), str(SUITE), *args]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={k: v for k, v in os.environ.items() if "WEBHOOK" not in k.upper() and "DISCORD" not in k.upper()},
        )
        stdout_path.write_text(proc.stdout or "", encoding="utf-8")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8")
        parsed = None
        parse_error = None
        try:
            parsed = json.loads(proc.stdout or "")
        except Exception as exc:
            parse_error = type(exc).__name__
        return {
            "exit_code": proc.returncode,
            "parsed": parsed,
            "parse_error": parse_error,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "cmd": [str(PYTHON), str(SUITE), *args],
        }
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or ""), encoding="utf-8")
        stderr_path.write_text((exc.stderr.decode() if isinstance(exc.stderr, bytes) else (exc.stderr or "")) + "\nTIMEOUT\n", encoding="utf-8")
        return {"exit_code": 124, "parsed": None, "parse_error": "TimeoutExpired", "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), "cmd": cmd}


def _num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _int(v: Any, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _boolish(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "active", "ok"}
    return False


def evaluate_operational(compact: Dict[str, Any], *, daily_ok: bool, compact_ok: bool, target_day: str, expected_day: str) -> Dict[str, Any]:
    failures: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    def crit(cond: str, actual: Any, expected: Any) -> None:
        failures.append({"condition": cond, "actual": actual, "expected": expected})

    def warn(cond: str, actual: Any, expected: Any) -> None:
        warnings.append({"condition": cond, "actual": actual, "expected": expected})

    if not daily_ok:
        crit("daily_update_exit", "nonzero_or_failed", 0)
    if not compact_ok:
        crit("compact_status", "failed_or_unparsed", "ok")

    coverage = _num(compact.get("latest_complete_day_coverage_pct"))
    complete_day = compact.get("latest_complete_day_utc")
    health = str(compact.get("collector_health") or "")
    ws = str(compact.get("ws_state") or "")
    staleness = str(compact.get("staleness") or "")
    stability = str(compact.get("CURRENT_COLLECTION_STABILITY") or "")
    cov15 = _num(compact.get("CURRENT_15M_CORE_PERSIST_COVERAGE"))
    cov60 = _num(compact.get("CURRENT_60M_CORE_PERSIST_COVERAGE"))
    keep_active = _boolish(compact.get("KEEP_AWAKE_GUARD_ACTIVE"))
    keep_verdict = str(compact.get("KEEP_AWAKE_GUARD_VERDICT") or "")
    idle = _boolish(compact.get("IDLE_SLEEP_ASSERTION_ACTIVE"))
    system = _boolish(compact.get("SYSTEM_SLEEP_ASSERTION_ACTIVE"))
    wrapped = _boolish(compact.get("COLLECTOR_WRAPPED_BY_CAFFEINATE"))
    dup_guard = _int(compact.get("DUPLICATE_CAFFEINATE_GUARDS"))
    active_gens = _int(compact.get("ACTIVE_CONNECTION_GENERATIONS"), default=1)
    pending = _int(compact.get("pending_gaps"))
    failed = _int(compact.get("failed_gaps"))
    contamination = _int(compact.get("contamination"))
    power = str(compact.get("POWER_SOURCE") or "UNKNOWN")
    production = _boolish(compact.get("production_ready"))
    promotion = _boolish(compact.get("promotion_ready"))
    private_calls = _int(compact.get("private_endpoint_calls") or compact.get("private_calls") or 0)
    order_calls = _int(compact.get("order_endpoint_calls") or compact.get("order_calls") or 0)
    recon15 = _int(compact.get("CURRENT_15M_RECONNECT_CYCLES"))
    recon60 = _int(compact.get("CURRENT_60M_RECONNECT_CYCLES"))
    hard7 = _int(compact.get("hard_stale_incidents_7d"))

    if complete_day is None:
        crit("latest_complete_day_utc", None, expected_day)
    elif str(complete_day) != expected_day:
        # mismatch may be catch-up of latest available; treat as warning if coverage strong else critical lag
        if str(complete_day) != target_day:
            warn("latest_complete_day_mismatch", complete_day, expected_day)
        # if using catch-up target_day == complete_day, OK

    if coverage < 90:
        crit("latest_complete_day_coverage_pct", coverage, ">=90")
    elif coverage < 95:
        warn("latest_complete_day_coverage_pct", coverage, ">=95")

    if health != "HEALTHY":
        crit("collector_health", health, "HEALTHY")
    if ws != "HEALTHY":
        crit("ws_state", ws, "HEALTHY")
    if staleness != "STALENESS_OK":
        crit("staleness", staleness, "STALENESS_OK")
    if stability in {"FLAPPING", "PARTIAL_STREAM", "WRITER_STALLED", "HARD_STALE", "UNKNOWN", ""}:
        if stability not in STABLE_OK:
            crit("CURRENT_COLLECTION_STABILITY", stability, "STABLE|RECENTLY_RECOVERED")
    elif stability not in STABLE_OK:
        warn("CURRENT_COLLECTION_STABILITY", stability, "STABLE")

    if cov15 < 90:
        crit("CURRENT_15M_CORE_PERSIST_COVERAGE", cov15, ">=90")
    elif cov15 < 95:
        warn("CURRENT_15M_CORE_PERSIST_COVERAGE", cov15, ">=95")
    if cov60 < 90:
        crit("CURRENT_60M_CORE_PERSIST_COVERAGE", cov60, ">=90")
    elif cov60 < 95:
        warn("CURRENT_60M_CORE_PERSIST_COVERAGE", cov60, ">=95")

    if not keep_active or keep_verdict != "KEEP_AWAKE_GUARD_ACTIVE":
        crit("keep_awake_guard", keep_verdict or keep_active, "KEEP_AWAKE_GUARD_ACTIVE")
    if not idle:
        crit("IDLE_SLEEP_ASSERTION_ACTIVE", idle, True)
    if power == "AC" and not system:
        crit("SYSTEM_SLEEP_ASSERTION_ACTIVE", system, True)
    if not wrapped:
        crit("COLLECTOR_WRAPPED_BY_CAFFEINATE", wrapped, True)
    if dup_guard > 0:
        crit("DUPLICATE_CAFFEINATE_GUARDS", dup_guard, 0)
    if active_gens != 1:
        crit("ACTIVE_CONNECTION_GENERATIONS", active_gens, 1)
    if pending > 0:
        crit("pending_gaps", pending, 0)
    if failed > 0:
        crit("failed_gaps", failed, 0)
    if contamination > 0:
        crit("contamination", contamination, 0)
    if private_calls > 0:
        crit("private_calls", private_calls, 0)
    if order_calls > 0:
        crit("order_calls", order_calls, 0)
    if production:
        crit("production_ready", production, False)
    if promotion:
        crit("promotion_ready", promotion, False)

    # Historical 7d hard-stale totals may reflect pre-keepawake sleep gaps.
    # Only current 15m/60m reconnect activity elevates operational WARNING.
    if (recon15 > 0 or recon60 > 0) and not failures:
        warn("recent_recovery_activity", {"recon15": recon15, "recon60": recon60, "hard7d": hard7}, 0)

    if failures:
        severity = "CRITICAL"
    elif warnings:
        severity = "WARNING"
    else:
        severity = "NORMAL"

    readiness = str(compact.get("readiness") or "UNKNOWN")
    research_info_only = readiness in {"QUALITY_REVIEW_REQUIRED", "NOT_READY_FOR_EVALUATION", "SAMPLE_COUNT_MET_BUT_COMPOSITION_UNBALANCED"}
    # readiness alone should not force yellow operational embed
    operational_for_embed = severity
    if severity == "WARNING" and all(w["condition"].startswith("readiness") is False for w in warnings):
        pass
    if severity == "NORMAL" and research_info_only:
        operational_for_embed = "NORMAL"

    return {
        "operational_severity": severity,
        "operational_embed_severity": operational_for_embed,
        "failures": failures,
        "warnings": warnings,
        "readiness": readiness,
        "research_info_only": research_info_only,
        "metrics": {
            "complete_day": complete_day,
            "coverage": coverage,
            "health": health,
            "ws": ws,
            "staleness": staleness,
            "stability": stability,
            "cov15": cov15,
            "cov60": cov60,
            "keep_verdict": keep_verdict,
            "pending": pending,
            "failed": failed,
            "contamination": contamination,
            "power": power,
            "recon15": recon15,
            "recon60": recon60,
            "primary": compact.get("primary_markers"),
            "basis": compact.get("basis_markers"),
            "trusted_taker": compact.get("trusted_taker_markers"),
            "bottleneck": compact.get("readiness_bottleneck"),
            "recent_7d_coverage": compact.get("recent_7d_coverage_pct"),
            "recent_7d_pass": compact.get("recent_7d_pass_days"),
            "partial_coverage": compact.get("partial_day_strict_coverage_pct"),
            "partial_day": compact.get("current_partial_day_utc"),
        },
    }


def build_judgment_text(eval_result: Dict[str, Any]) -> str:
    m = eval_result["metrics"]
    sev = eval_result["operational_severity"]
    if sev == "NORMAL":
        return (
            f"전날 strict-live coverage는 {m['coverage']:.2f}%였고 collector, WS, writer, keep-awake guard가 모두 정상입니다. "
            "최근 15분과 60분 동안 reconnect나 hard stale이 없으며 추가 운영 조치는 필요하지 않습니다."
        )
    if sev == "WARNING":
        return (
            f"전날 coverage는 {m['coverage']:.2f}%이며 현재 collector/WS는 {m['health']}/{m['ws']}입니다. "
            "주의 조건이 감지되었으므로 오늘 수집 상태를 계속 관찰합니다."
        )
    fails = "; ".join(f"{f['condition']}={f['actual']} (expected {f['expected']})" for f in eval_result["failures"][:5])
    return (
        f"운영 비정상 조건이 감지되었습니다: {fails}. "
        "notifier는 collector를 자동 재시작하지 않았으므로 compact status와 launchd/keep-awake를 확인해야 합니다."
    )


def build_research_text(eval_result: Dict[str, Any]) -> str:
    m = eval_result["metrics"]
    readiness = eval_result["readiness"]
    if readiness in READY_SET:
        return (
            f"🎯 정식 평가 조건이 충족되었습니다 ({readiness}). "
            "다음 단계는 frozen observer 누적 성과 평가이며 production 승격이 아닙니다."
        )
    return (
        f"수집 운영과 별개로 연구 readiness는 {readiness}입니다. "
        f"Primary {m.get('primary')}/50, Basis {m.get('basis')}/10, Trusted taker {m.get('trusted_taker')}/10"
        + (f", bottleneck={m.get('bottleneck')}" if m.get("bottleneck") else "")
        + "."
    )


def build_copy_block(eval_result: Dict[str, Any], target_day: str) -> str:
    m = eval_result["metrics"]
    action = "KEEP_RUNNING" if eval_result["operational_severity"] != "CRITICAL" else "REVIEW_REQUIRED"
    return "\n".join(
        [
            "CAN_BIT DAILY:",
            f"DATE UTC: {target_day}",
            f"OPERATIONAL: {eval_result['operational_severity']}",
            f"COVERAGE: {m['coverage']:.2f}%",
            f"COLLECTOR/WS: {m['health']}/{m['ws']}",
            f"STABILITY: {m['stability']}",
            f"15M/60M: {m['cov15']:.0f}/{m['cov60']:.0f}",
            f"KEEP_AWAKE: {m['keep_verdict']}",
            f"GAPS: {m['pending']}/{m['failed']}",
            f"CONTAMINATION: {m['contamination']}",
            f"READINESS: {eval_result['readiness']}",
            f"PRIMARY: {m.get('primary')}/50",
            f"BASIS: {m.get('basis')}/10",
            f"TRUSTED_TAKER: {m.get('trusted_taker')}/10",
            f"ACTION: {action}",
        ]
    )


def severity_color(sev: str, readiness_ready: bool = False) -> int:
    if sev == "CRITICAL":
        return COLOR_CRITICAL
    if sev == "WARNING":
        return COLOR_WARNING
    if readiness_ready:
        return COLOR_READY
    return COLOR_NORMAL


def build_discord_payload(
    eval_result: Dict[str, Any],
    *,
    target_day: str,
    notification_type: str = "DAILY_SUMMARY",
    forced: bool = False,
    ready_transition: bool = False,
) -> Dict[str, Any]:
    m = eval_result["metrics"]
    sev = eval_result["operational_severity"]
    embed_sev = eval_result.get("operational_embed_severity") or sev
    # readiness-only: keep green
    if sev == "NORMAL":
        embed_sev = "NORMAL"
    title_map = {
        "NORMAL": f"✅ CAN_BIT 일일 수집 정상 — {target_day} UTC",
        "WARNING": f"⚠️ CAN_BIT 일일 수집 주의 — {target_day} UTC",
        "CRITICAL": f"🚨 CAN_BIT 수집 비정상 — 즉시 확인 필요 — {target_day} UTC",
    }
    title = title_map.get(embed_sev, title_map["WARNING"])
    if forced:
        title = "[FORCED RESEND] " + title
    if notification_type == "TEST":
        title = f"🧪 CAN_BIT OBSERVATION NOTIFIER TEST — {now_kst().strftime('%Y-%m-%d %H:%M KST')}"
        embed_sev = "NORMAL"

    description = build_judgment_text(eval_result) if notification_type != "TEST" else "테스트 메시지입니다. daily sent ledger에는 기록되지 않습니다."
    if ready_transition:
        description = "🎯 정식 누적 평가 조건 충족\n" + description

    fields = []
    if sev == "CRITICAL" and eval_result["failures"]:
        fail_lines = "\n".join(
            f"- {f['condition']}: {f['actual']} (기준 {f['expected']})" for f in eval_result["failures"][:8]
        )
        fields.append({"name": "실패 조건", "value": fail_lines[:1000], "inline": False})

    fields.extend(
        [
            {
                "name": "수집 품질",
                "value": (
                    f"완료일 coverage: {m['coverage']:.2f}%\n"
                    f"최근 7일 coverage: {_num(m.get('recent_7d_coverage')):.2f}%\n"
                    f"최근 7일 pass: {m.get('recent_7d_pass')}\n"
                    f"현재 partial: {_num(m.get('partial_coverage')):.2f}% ({m.get('partial_day')})"
                ),
                "inline": False,
            },
            {
                "name": "실시간 상태",
                "value": (
                    f"Collector / WS: {m['health']} / {m['ws']}\n"
                    f"Staleness: {m['staleness']}\n"
                    f"Stability: {m['stability']}\n"
                    f"15m / 60m persist: {m['cov15']:.0f}% / {m['cov60']:.0f}%\n"
                    f"Active generation: 1"
                ),
                "inline": False,
            },
            {
                "name": "Keep Awake",
                "value": (
                    f"Guard: {m['keep_verdict']}\n"
                    f"Power: {m['power']}\n"
                    f"Reconnect 15m/60m: {m['recon15']} / {m['recon60']}"
                ),
                "inline": False,
            },
            {
                "name": "장애·무결성",
                "value": (
                    f"Gap pending/failed: {m['pending']} / {m['failed']}\n"
                    f"Contamination: {m['contamination']}\n"
                    f"Private/Order: 0 / 0"
                ),
                "inline": False,
            },
            {
                "name": "연구 readiness",
                "value": build_research_text(eval_result)[:1000],
                "inline": False,
            },
            {
                "name": "오늘의 판단",
                "value": build_judgment_text(eval_result)[:1000],
                "inline": False,
            },
            {
                "name": "ChatGPT 복사용",
                "value": f"```\n{build_copy_block(eval_result, target_day)[:900]}\n```",
                "inline": False,
            },
        ]
    )
    if sev == "CRITICAL":
        fields.append(
            {
                "name": "권장 확인",
                "value": (
                    "```\n"
                    "./.venv/bin/python scripts/diagnostics/run_microstructure_observation_quality_suite.py --status --compact --json\n"
                    "pmset -g assertions\n"
                    "launchctl print gui/$(id -u)/com.canbit.microstructure-public-collector\n"
                    "```\n"
                    "자동 재시작 없음 · 알림만 전송"
                ),
                "inline": False,
            }
        )

    payload = {
        "embeds": [
            {
                "title": title[:250],
                "description": description[:2000],
                "color": severity_color(embed_sev, ready_transition),
                "fields": fields,
                "footer": {
                    "text": f"Generated at {now_kst().strftime('%Y-%m-%d %H:%M:%S KST')} · Observation-only · Production false · Promotion false"
                },
            }
        ]
    }
    return payload


def send_webhook(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    attempts = []
    last: Dict[str, Any] = {"success": False, "http_status": None, "error_category": "UNKNOWN"}
    for i in range(MAX_RETRIES):
        if i > 0:
            time.sleep(RETRY_SLEEPS[min(i, len(RETRY_SLEEPS) - 1)])
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "canbit-observation-notifier/1.0"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                status = getattr(resp, "status", None) or resp.getcode()
                _ = resp.read(256)
                attempts.append({"attempt": i + 1, "http_status": status})
                if status in (200, 204):
                    return {"success": True, "http_status": status, "attempts": attempts, "error_category": None}
                last = {"success": False, "http_status": status, "attempts": attempts, "error_category": f"HTTP_{status}"}
        except urllib.error.HTTPError as exc:
            status = exc.code
            retry_after = None
            try:
                retry_after = float(exc.headers.get("Retry-After") or 0)
            except Exception:
                retry_after = 0
            attempts.append({"attempt": i + 1, "http_status": status, "retry_after": retry_after})
            if status == 429:
                time.sleep(min(retry_after or 10, MAX_RETRY_AFTER))
                last = {"success": False, "http_status": status, "attempts": attempts, "error_category": "RATE_LIMIT"}
                continue
            if 500 <= status <= 599:
                last = {"success": False, "http_status": status, "attempts": attempts, "error_category": "SERVER_ERROR"}
                continue
            # permanent 4xx
            return {
                "success": False,
                "http_status": status,
                "attempts": attempts,
                "error_category": "CLIENT_ERROR",
                "permanent": True,
            }
        except Exception as exc:
            attempts.append({"attempt": i + 1, "error": type(exc).__name__})
            last = {
                "success": False,
                "http_status": None,
                "attempts": attempts,
                "error_category": type(exc).__name__,
            }
    last["attempts"] = attempts
    return last


def history_path() -> Path:
    return STATE / "discord_notification_history.jsonl"


def load_history() -> List[Dict[str, Any]]:
    path = history_path()
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def append_history(row: Dict[str, Any]) -> None:
    ensure_dirs()
    with history_path().open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def already_sent(target_day: str, notification_type: str, payload_hash: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for row in reversed(load_history()):
        if not row.get("success"):
            continue
        if row.get("target_day_utc") == target_day and row.get("notification_type") == notification_type:
            if payload_hash is None or row.get("payload_hash") == payload_hash:
                return row
    return None


def last_readiness() -> Optional[str]:
    for row in reversed(load_history()):
        if row.get("notification_type") == "DAILY_SUMMARY" and row.get("readiness"):
            return row.get("readiness")
    return None


def outbox_write(target_day: str, notification_type: str, payload: Dict[str, Any], send_result: Dict[str, Any], severity: str) -> Path:
    ensure_dirs()
    ph = sha256_obj(payload)
    path = OUTBOX / f"{target_day}_{notification_type}_{ph[:12]}.json"
    if path.exists():
        return path
    obj = {
        "target_day_utc": target_day,
        "notification_type": notification_type,
        "created_at_utc": now_utc().isoformat(),
        "severity": severity,
        "payload": payload,
        "payload_hash": ph,
        "attempt_count": len(send_result.get("attempts") or []),
        "last_http_status": send_result.get("http_status"),
        "last_error_category": send_result.get("error_category"),
        "next_retry_eligible_utc": (now_utc() + timedelta(minutes=30)).isoformat(),
    }
    dump_json(path, obj)
    return path


def retry_outbox(url: Optional[str]) -> Dict[str, Any]:
    ensure_dirs()
    results = []
    for path in sorted(OUTBOX.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # skip if already successfully sent
        if already_sent(obj.get("target_day_utc"), obj.get("notification_type"), obj.get("payload_hash")):
            dest = OUTBOX_ARCHIVED / path.name
            shutil.move(str(path), str(dest))
            results.append({"path": str(path), "action": "archived_already_sent"})
            continue
        if not url:
            results.append({"path": str(path), "action": "skipped_no_secret"})
            continue
        send_result = send_webhook(url, obj["payload"])
        if send_result.get("success"):
            append_history(
                {
                    "target_day_utc": obj.get("target_day_utc"),
                    "notification_type": "OUTBOX_RETRY",
                    "operational_severity": obj.get("severity"),
                    "readiness": None,
                    "payload_hash": obj.get("payload_hash"),
                    "sent_at_utc": now_utc().isoformat(),
                    "sent_at_kst": now_kst().isoformat(),
                    "http_status": send_result.get("http_status"),
                    "success": True,
                    "forced": False,
                }
            )
            dest = OUTBOX_ARCHIVED / path.name
            shutil.move(str(path), str(dest))
            results.append({"path": str(path), "action": "sent", "http_status": send_result.get("http_status")})
        else:
            obj["attempt_count"] = _int(obj.get("attempt_count")) + 1
            obj["last_http_status"] = send_result.get("http_status")
            obj["last_error_category"] = send_result.get("error_category")
            dump_json(path, obj)
            results.append({"path": str(path), "action": "retry_failed", "error": send_result.get("error_category")})
    return {"retried": results, "pending_outbox_count": len(list(OUTBOX.glob("*.json")))}


def launchd_status() -> Dict[str, Any]:
    uid = os.getuid()
    label = LAUNCHD_LABEL
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        out = proc.stdout or ""
        loaded = proc.returncode == 0 and "state =" in out
        return {
            "launchd_loaded": loaded,
            "label": label,
            "print_exit": proc.returncode,
            "state_line": next((ln.strip() for ln in out.splitlines() if "state =" in ln), None),
        }
    except Exception as exc:
        return {"launchd_loaded": False, "label": label, "error": type(exc).__name__}


def write_reports(result: Dict[str, Any], payload: Optional[Dict[str, Any]]) -> None:
    ensure_dirs()
    dump_json(REPORTS / "daily_discord_notification_latest.json", result)
    if payload is not None:
        dump_json(REPORTS / "discord_payload_latest.json", payload)
    md = [
        "# Daily Discord Notification Latest",
        "",
        f"- verdict: `{result.get('verdict')}`",
        f"- target_day_utc: `{result.get('target_day_utc')}`",
        f"- operational_severity: `{result.get('operational_severity')}`",
        f"- readiness: `{result.get('readiness')}`",
        f"- discord_success: `{result.get('discord_success')}`",
        f"- http_status: `{result.get('http_status')}`",
        f"- duplicate_prevented: `{result.get('duplicate_send_prevented')}`",
        "",
        "## Judgment",
        "",
        result.get("judgment_text") or "",
        "",
        "## Copy block",
        "",
        "```",
        result.get("copy_block") or "",
        "```",
    ]
    (REPORTS / "daily_discord_notification_latest.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def status_payload() -> Dict[str, Any]:
    ensure_dirs()
    secret = inspect_secret()
    hist = [r for r in load_history() if r.get("notification_type") == "DAILY_SUMMARY"]
    last = hist[-1] if hist else {}
    compact_path = REPO / "data/diagnostics/microstructure_observation_quality/reports/observation_quality_compact_status.json"
    compact = {}
    if compact_path.exists():
        try:
            compact = json.loads(compact_path.read_text(encoding="utf-8"))
        except Exception:
            compact = {}
    ld = launchd_status()
    notifier_status_path = STATE / "notifier_status.json"
    prev = {}
    if notifier_status_path.exists():
        try:
            prev = json.loads(notifier_status_path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    out = {
        "notifier_status": "ACTIVE" if ld.get("launchd_loaded") and secret.get("present") else (
            "WEBHOOK_SECRET_MISSING" if not secret.get("present") else "INSTALLED_NOT_LOADED" if not ld.get("launchd_loaded") else "READY"
        ),
        "launchd_loaded": bool(ld.get("launchd_loaded")),
        "schedule_kst": "09:05",
        "webhook_secret_present": bool(secret.get("present")),
        "webhook_secret_permissions_ok": bool(secret.get("permissions_ok")),
        "webhook_secret_path": secret.get("secret_path"),
        "last_run_at_kst": last.get("sent_at_kst") or prev.get("last_run_at_kst"),
        "last_target_day_utc": last.get("target_day_utc") or prev.get("last_target_day_utc"),
        "last_operational_severity": last.get("operational_severity") or prev.get("last_operational_severity"),
        "last_readiness": last.get("readiness") or prev.get("last_readiness"),
        "last_discord_send_success": last.get("success") if last else prev.get("last_discord_send_success"),
        "last_http_status": last.get("http_status") or prev.get("last_http_status"),
        "last_payload_hash": last.get("payload_hash") or prev.get("last_payload_hash"),
        "duplicate_send_prevented": bool(prev.get("duplicate_send_prevented")),
        "pending_outbox_count": len(list(OUTBOX.glob("*.json"))),
        "latest_complete_day_coverage_pct": compact.get("latest_complete_day_coverage_pct"),
        "collector_health": compact.get("collector_health"),
        "keep_awake_guard": compact.get("KEEP_AWAKE_GUARD_VERDICT"),
        "pending_gaps": compact.get("pending_gaps"),
        "failed_gaps": compact.get("failed_gaps"),
        "contamination": compact.get("contamination"),
        "production_ready": False,
        "promotion_ready": False,
    }
    dump_json(STATE / "notifier_status.json", out)
    return out


def run_notification(
    *,
    dry_run: bool = False,
    force: bool = False,
    send_test: bool = False,
    skip_daily_update: bool = False,
    compact_override: Optional[Dict[str, Any]] = None,
    secret_path: Path = DEFAULT_SECRET,
) -> Dict[str, Any]:
    ensure_dirs()
    url, secret_meta = load_webhook_url(secret_path)
    outbox_retry_info = {"retried": [], "pending_outbox_count": len(list(OUTBOX.glob("*.json")))}
    if not dry_run and not send_test:
        outbox_retry_info = retry_outbox(url)

    daily_result = {"exit_code": 0, "parsed": {}, "parse_error": None}
    if not skip_daily_update and not send_test:
        daily_result = run_suite(
            ["--daily-update", "--json"],
            LOGS / "daily_update_stdout_latest.log",
            LOGS / "daily_update_stderr_latest.log",
            DAILY_UPDATE_TIMEOUT,
        )
    daily_ok = daily_result.get("exit_code") == 0

    if compact_override is not None:
        compact = compact_override
        compact_ok = True
        compact_result = {"exit_code": 0, "parsed": compact, "parse_error": None}
    else:
        compact_result = run_suite(
            ["--status", "--compact", "--json"],
            LOGS / "compact_status_stdout_latest.log",
            LOGS / "compact_status_stderr_latest.log",
            COMPACT_TIMEOUT,
        )
        compact = compact_result.get("parsed") or {}
        compact_ok = compact_result.get("exit_code") == 0 and isinstance(compact, dict) and bool(compact)

    expected_day = expected_complete_day_utc()
    complete_day = str(compact.get("latest_complete_day_utc") or "")
    # catch-up: prefer latest complete day if not yet sent
    target_day = complete_day or expected_day
    if complete_day and complete_day != expected_day:
        # still evaluate that complete day (catch-up single day)
        target_day = complete_day

    if send_test:
        eval_result = {
            "operational_severity": "NORMAL",
            "operational_embed_severity": "NORMAL",
            "failures": [],
            "warnings": [],
            "readiness": compact.get("readiness") or "UNKNOWN",
            "research_info_only": True,
            "metrics": {
                "complete_day": target_day,
                "coverage": _num(compact.get("latest_complete_day_coverage_pct")),
                "health": compact.get("collector_health") or "UNKNOWN",
                "ws": compact.get("ws_state") or "UNKNOWN",
                "staleness": compact.get("staleness") or "UNKNOWN",
                "stability": compact.get("CURRENT_COLLECTION_STABILITY") or "UNKNOWN",
                "cov15": _num(compact.get("CURRENT_15M_CORE_PERSIST_COVERAGE")),
                "cov60": _num(compact.get("CURRENT_60M_CORE_PERSIST_COVERAGE")),
                "keep_verdict": compact.get("KEEP_AWAKE_GUARD_VERDICT") or "UNKNOWN",
                "pending": _int(compact.get("pending_gaps")),
                "failed": _int(compact.get("failed_gaps")),
                "contamination": _int(compact.get("contamination")),
                "power": compact.get("POWER_SOURCE") or "UNKNOWN",
                "recon15": _int(compact.get("CURRENT_15M_RECONNECT_CYCLES")),
                "recon60": _int(compact.get("CURRENT_60M_RECONNECT_CYCLES")),
                "primary": compact.get("primary_markers"),
                "basis": compact.get("basis_markers"),
                "trusted_taker": compact.get("trusted_taker_markers"),
                "bottleneck": compact.get("readiness_bottleneck"),
                "recent_7d_coverage": compact.get("recent_7d_coverage_pct"),
                "recent_7d_pass": compact.get("recent_7d_pass_days"),
                "partial_coverage": compact.get("partial_day_strict_coverage_pct"),
                "partial_day": compact.get("current_partial_day_utc"),
            },
        }
        notification_type = "TEST"
        ready_transition = False
    else:
        eval_result = evaluate_operational(
            compact,
            daily_ok=daily_ok,
            compact_ok=compact_ok,
            target_day=target_day,
            expected_day=expected_day,
        )
        prev_ready = last_readiness()
        ready_transition = (prev_ready not in READY_SET) and (eval_result["readiness"] in READY_SET)
        notification_type = "DAILY_SUMMARY"

    payload = build_discord_payload(
        eval_result,
        target_day=target_day or expected_day,
        notification_type=notification_type,
        forced=force,
        ready_transition=ready_transition,
    )
    payload_hash = sha256_obj(payload)

    duplicate = False
    prior = None
    if notification_type == "DAILY_SUMMARY" and not force:
        prior = already_sent(target_day, notification_type)
        if prior:
            duplicate = True

    send_result: Dict[str, Any] = {"success": False, "http_status": None, "error_category": None, "skipped": False}
    verdict = "UNKNOWN"
    if dry_run:
        dump_json(VALIDATION / "dry_run_payload.json", payload)
        send_result = {"success": False, "http_status": None, "error_category": None, "skipped": True, "dry_run": True}
        verdict = "DRY_RUN_OK"
    elif duplicate:
        send_result = {"success": True, "http_status": prior.get("http_status"), "skipped": True, "error_category": None}
        verdict = "ALREADY_SENT_NO_ACTION"
    elif not url:
        send_result = {"success": False, "http_status": None, "error_category": secret_meta.get("error") or "WEBHOOK_SECRET_MISSING", "skipped": True}
        if notification_type != "TEST":
            outbox_write(target_day, notification_type, payload, send_result, eval_result["operational_severity"])
        verdict = "WEBHOOK_SECRET_MISSING"
    else:
        send_result = send_webhook(url, payload)
        if send_result.get("success"):
            if notification_type != "TEST":
                append_history(
                    {
                        "target_day_utc": target_day,
                        "notification_type": notification_type,
                        "operational_severity": eval_result["operational_severity"],
                        "readiness": eval_result["readiness"],
                        "payload_hash": payload_hash,
                        "sent_at_utc": now_utc().isoformat(),
                        "sent_at_kst": now_kst().isoformat(),
                        "http_status": send_result.get("http_status"),
                        "success": True,
                        "forced": force,
                        "source_compact_hash": sha256_obj(compact) if compact else None,
                        "ready_transition": ready_transition,
                    }
                )
            verdict = "SENT_OK" if notification_type != "TEST" else "TEST_SENT_OK"
        else:
            if notification_type != "TEST":
                outbox_write(target_day, notification_type, payload, send_result, eval_result["operational_severity"])
            verdict = "SEND_FAILED"

    result = {
        "verdict": verdict,
        "generated_at_utc": now_utc().isoformat(),
        "generated_at_kst": now_kst().isoformat(),
        "target_day_utc": target_day,
        "expected_day_utc": expected_day,
        "notification_type": notification_type,
        "operational_severity": eval_result["operational_severity"],
        "readiness": eval_result["readiness"],
        "ready_transition": ready_transition,
        "failures": eval_result["failures"],
        "warnings": eval_result["warnings"],
        "judgment_text": build_judgment_text(eval_result),
        "research_text": build_research_text(eval_result),
        "copy_block": build_copy_block(eval_result, target_day or expected_day),
        "discord_success": bool(send_result.get("success")) and not send_result.get("dry_run"),
        "http_status": send_result.get("http_status"),
        "error_category": send_result.get("error_category"),
        "duplicate_send_prevented": duplicate,
        "forced": force,
        "dry_run": dry_run,
        "payload_hash": payload_hash,
        "secret": {k: v for k, v in secret_meta.items() if k != "masked" or True},
        "daily_update_exit_code": daily_result.get("exit_code"),
        "compact_exit_code": compact_result.get("exit_code"),
        "outbox_retry": outbox_retry_info,
        "pending_outbox_count": len(list(OUTBOX.glob("*.json"))),
        "metrics": eval_result["metrics"],
        "production_ready": False,
        "promotion_ready": False,
    }
    # scrub any accidental secret
    result["secret"] = {
        "secret_path": secret_meta.get("secret_path"),
        "present": secret_meta.get("present"),
        "permissions": secret_meta.get("permissions"),
        "permissions_ok": secret_meta.get("permissions_ok"),
        "error": secret_meta.get("error"),
        "warning": secret_meta.get("warning"),
        "masked": secret_meta.get("masked"),
    }
    write_reports(result, payload if dry_run or True else payload)
    dump_json(
        STATE / "notifier_status.json",
        {
            **status_payload(),
            "last_run_at_kst": result["generated_at_kst"],
            "last_target_day_utc": target_day,
            "last_operational_severity": result["operational_severity"],
            "last_readiness": result["readiness"],
            "last_discord_send_success": result["discord_success"] if not dry_run else None,
            "last_http_status": result["http_status"],
            "last_payload_hash": payload_hash,
            "duplicate_send_prevented": duplicate,
            "pending_outbox_count": result["pending_outbox_count"],
        },
    )
    return result
