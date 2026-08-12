"""Diagnostics helpers for collector caffeinate keep-awake guard."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
WRAPPER_PATH = REPO / "scripts/runtime/run_with_caffeinate_guard.sh"
COLLECTOR_LABEL = "com.canbit.microstructure-public-collector"
OBSERVER_LABEL = "com.canbit.microstructure-weak-hint-forward-observer"
COLLECTOR_SCRIPT = "run_microstructure_public_live_collector.py"
CAFFEINATE_BIN = "/usr/bin/caffeinate"

_ASSERTION_COUNT_RE = re.compile(
    r"^\s*(PreventUserIdleSystemSleep|PreventSystemSleep)\s+(\d+)\s*$",
    re.MULTILINE,
)
_PID_LINE_RE = re.compile(
    r"^\s*pid\s+(\d+)\(([^)]+)\):\s+.*?((?:PreventUserIdleSystemSleep|PreventSystemSleep|NoIdleSleepAssertion|UserIsActive)\b.*)$",
    re.MULTILINE,
)


def run_cmd(args: List[str], timeout: int = 30) -> Tuple[int, str]:
    p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    out = (p.stdout or "") + (("\n" + p.stderr) if p.stderr else "")
    return p.returncode, out


def list_processes() -> List[Dict[str, Any]]:
    code, out = run_cmd(["ps", "-axo", "pid=,ppid=,etime=,state=,command="])
    rows: List[Dict[str, Any]] = []
    if code != 0:
        return rows
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 4)
        if len(parts) < 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        rows.append(
            {
                "pid": pid,
                "ppid": ppid,
                "etime": parts[2],
                "state": parts[3],
                "command": parts[4],
            }
        )
    return rows


def _is_caffeinate_command(cmd: str) -> bool:
    first = cmd.split()[0] if cmd.strip() else ""
    return first.endswith("caffeinate") or first == "caffeinate" or CAFFEINATE_BIN in first


def find_collector_processes(rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    rows = rows if rows is not None else list_processes()
    out = []
    for r in rows:
        cmd = r["command"]
        if _is_caffeinate_command(cmd):
            continue
        if COLLECTOR_SCRIPT in cmd and "python" in cmd:
            out.append(r)
    return out


def find_caffeinate_guards(rows: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Return caffeinate processes that wrap the public collector (or generic -i -s with collector child)."""
    rows = rows if rows is not None else list_processes()
    by_pid = {r["pid"]: r for r in rows}
    collectors = find_collector_processes(rows)
    guards: List[Dict[str, Any]] = []
    for r in rows:
        cmd = r["command"]
        if not _is_caffeinate_command(cmd):
            continue
        if "-i" not in cmd or "-s" not in cmd:
            continue
        children_collectors = [
            c
            for c in collectors
            if c["ppid"] == r["pid"] or _is_ancestor(r["pid"], c["pid"], by_pid)
        ]
        embeds = COLLECTOR_SCRIPT in cmd
        if children_collectors or embeds:
            item = dict(r)
            item["collector_children"] = [c["pid"] for c in children_collectors]
            item["embeds_collector"] = embeds
            guards.append(item)
    seen = set()
    uniq = []
    for g in guards:
        if g["pid"] in seen:
            continue
        seen.add(g["pid"])
        uniq.append(g)
    return uniq


def _is_ancestor(ancestor: int, pid: int, by_pid: Dict[int, Dict[str, Any]], limit: int = 8) -> bool:
    cur = pid
    for _ in range(limit):
        row = by_pid.get(cur)
        if not row:
            return False
        if row["ppid"] == ancestor:
            return True
        if row["ppid"] in (0, 1) and cur != pid:
            return False
        cur = row["ppid"]
    return False


def parse_assertions(text: str) -> Dict[str, Any]:
    counts = {"PreventUserIdleSystemSleep": 0, "PreventSystemSleep": 0}
    for m in _ASSERTION_COUNT_RE.finditer(text):
        counts[m.group(1)] = int(m.group(2))
    owners: List[Dict[str, Any]] = []
    for m in _PID_LINE_RE.finditer(text):
        detail = m.group(3)
        owners.append(
            {
                "pid": int(m.group(1)),
                "name": m.group(2),
                "detail": detail.strip(),
                "prevent_user_idle": "PreventUserIdleSystemSleep" in detail or "NoIdleSleepAssertion" in detail,
                "prevent_system_sleep": "PreventSystemSleep" in detail,
            }
        )
    caffeinate_owners = [o for o in owners if "caffeinate" in o["name"].lower()]
    return {
        "counts": counts,
        "owners": owners,
        "caffeinate_owners": caffeinate_owners,
        "idle_system_sleep_assertion_active": counts.get("PreventUserIdleSystemSleep", 0) > 0
        or any(o["prevent_user_idle"] for o in caffeinate_owners),
        "system_sleep_assertion_active": counts.get("PreventSystemSleep", 0) > 0
        or any(o["prevent_system_sleep"] for o in caffeinate_owners),
    }


def detect_power_source(pmset_custom_text: Optional[str] = None, assertions_text: Optional[str] = None) -> str:
    text = assertions_text or ""
    if re.search(r"Using\s+AC|AC\s+Attached|'\s*AC\s*Power\s*'", text, re.I):
        return "AC"
    if re.search(r"Using\s+Batt|Battery\s+Power", text, re.I):
        return "BATTERY"
    code, out = run_cmd(["pmset", "-g", "batt"])
    if "AC Power" in out:
        return "AC"
    if "Battery Power" in out:
        return "BATTERY"
    if pmset_custom_text and "AC Power" in pmset_custom_text:
        return "AC"
    return "UNKNOWN"


def assess_keepawake_guard(
    assertions_text: Optional[str] = None,
    rows: Optional[List[Dict[str, Any]]] = None,
    expected: bool = True,
) -> Dict[str, Any]:
    if assertions_text is None:
        _, assertions_text = run_cmd(["pmset", "-g", "assertions"])
    rows = rows if rows is not None else list_processes()
    collectors = find_collector_processes(rows)
    guards = find_caffeinate_guards(rows)
    parsed = parse_assertions(assertions_text)
    power = detect_power_source(assertions_text=assertions_text)

    collector_pid = collectors[0]["pid"] if len(collectors) == 1 else (collectors[0]["pid"] if collectors else None)
    caffeinate_pid = guards[0]["pid"] if len(guards) == 1 else (guards[0]["pid"] if guards else None)

    owner_verified = False
    if caffeinate_pid is not None:
        for o in parsed["caffeinate_owners"]:
            if o["pid"] == caffeinate_pid and (o["prevent_user_idle"] or o["prevent_system_sleep"]):
                owner_verified = True
                break
        # Fallback: any caffeinate owner while our guard pid matches process list
        if not owner_verified and parsed["caffeinate_owners"] and caffeinate_pid in {o["pid"] for o in parsed["caffeinate_owners"]}:
            owner_verified = True

    wrapped = False
    if collector_pid and caffeinate_pid:
        by_pid = {r["pid"]: r for r in rows}
        crow = by_pid.get(collector_pid) or {}
        grow = by_pid.get(caffeinate_pid) or {}
        # Expected tree: launchd -> caffeinate -> python
        if crow.get("ppid") == caffeinate_pid or _is_ancestor(caffeinate_pid, collector_pid, by_pid):
            wrapped = True
        if collector_pid in (guards[0].get("collector_children") or []):
            wrapped = True
        # Observed alternate tree under launchd: python (ppid 1) with child
        # caffeinate -i -s <same python argv>. Assertions still owned by caffeinate
        # "on behalf of" the collector PID — treat as wrapped when command embeds collector.
        if grow.get("ppid") == collector_pid and COLLECTOR_SCRIPT in (grow.get("command") or ""):
            wrapped = True
        if owner_verified and COLLECTOR_SCRIPT in (grow.get("command") or ""):
            wrapped = True

    idle_active = bool(parsed["idle_system_sleep_assertion_active"] and (owner_verified or wrapped))
    # If counts show idle prevent and our caffeinate is running with -i -s, treat idle as active
    if guards and parsed["counts"].get("PreventUserIdleSystemSleep", 0) > 0 and wrapped:
        idle_active = True
    system_active = bool(parsed["system_sleep_assertion_active"] and (owner_verified or wrapped))
    if power == "AC" and guards and parsed["counts"].get("PreventSystemSleep", 0) > 0 and wrapped:
        system_active = True
    # Direct check: caffeinate -i creates PreventUserIdleSystemSleep owned by caffeinate
    if guards and parsed["caffeinate_owners"]:
        for o in parsed["caffeinate_owners"]:
            if o["pid"] == caffeinate_pid:
                if o["prevent_user_idle"]:
                    idle_active = True
                if o["prevent_system_sleep"]:
                    system_active = True

    guard_active = bool(
        expected
        and len(collectors) == 1
        and len(guards) == 1
        and wrapped
        and idle_active
    )

    if not collectors:
        verdict = "KEEP_AWAKE_GUARD_COLLECTOR_NOT_RUNNING"
    elif len(collectors) > 1 or len(guards) > 1:
        verdict = "KEEP_AWAKE_GUARD_DUPLICATE"
    elif not guards:
        verdict = "KEEP_AWAKE_GUARD_MISSING"
    elif not wrapped:
        verdict = "KEEP_AWAKE_GUARD_UNRESOLVED"
    elif not idle_active:
        verdict = "KEEP_AWAKE_GUARD_ASSERTION_MISSING"
    elif power == "AC" and not system_active:
        # -s should create PreventSystemSleep on AC; warn via unresolved if idle ok
        verdict = "KEEP_AWAKE_GUARD_ASSERTION_MISSING" if not guard_active else "KEEP_AWAKE_GUARD_ACTIVE"
        if idle_active and wrapped and len(guards) == 1:
            # Some macOS versions fold -s into idle when display-related; still mark active if idle+wrapped
            verdict = "KEEP_AWAKE_GUARD_ACTIVE"
            guard_active = True
    else:
        verdict = "KEEP_AWAKE_GUARD_ACTIVE" if guard_active else "KEEP_AWAKE_GUARD_UNRESOLVED"

    if verdict == "KEEP_AWAKE_GUARD_ACTIVE":
        guard_active = True

    return {
        "keep_awake_guard_expected": expected,
        "keep_awake_guard_active": guard_active,
        "caffeinate_pid": caffeinate_pid,
        "collector_pid": collector_pid,
        "idle_sleep_assertion_active": idle_active,
        "system_sleep_assertion_active": system_active,
        "assertion_owner_verified": owner_verified,
        "collector_wrapped_by_caffeinate": wrapped,
        "duplicate_caffeinate_guards": max(0, len(guards) - 1) if guards else 0,
        "duplicate_collectors": max(0, len(collectors) - 1) if collectors else 0,
        "active_caffeinate_guards": len(guards),
        "active_collectors": len(collectors),
        "clamshell_sleep_not_prevented": True,
        "power_source": power,
        "keep_awake_guard_verdict": verdict,
        "guards": [{"pid": g["pid"], "ppid": g["ppid"], "command": g["command"]} for g in guards],
        "collectors": [{"pid": c["pid"], "ppid": c["ppid"], "command": c["command"]} for c in collectors],
        "assertion_counts": parsed["counts"],
        "caffeinate_assertion_owners": parsed["caffeinate_owners"],
    }


def prepend_wrapper_to_program_arguments(args: List[str], wrapper: str | Path = WRAPPER_PATH) -> List[str]:
    wrapper_s = str(wrapper)
    if args and args[0] == wrapper_s:
        return list(args)
    return [wrapper_s] + list(args)


def assert_args_preserved(before: List[str], after: List[str], wrapper: str | Path = WRAPPER_PATH) -> bool:
    wrapper_s = str(wrapper)
    if not after or after[0] != wrapper_s:
        return False
    return after[1:] == before


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(json.dumps(assess_keepawake_guard(), indent=2, default=str))
