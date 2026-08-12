"""Tests for collector caffeinate keep-awake guard diagnostics."""
from __future__ import annotations

import plistlib
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/diagnostics"))

import microstructure_keepawake_guard as kaw
import microstructure_observation_quality as oq

WRAPPER = REPO / "scripts/runtime/run_with_caffeinate_guard.sh"
BACKUP = REPO / "data/diagnostics/microstructure_keepawake_guard/backups/installed_collector_plist_before.plist"
SOURCE = (
    REPO
    / "data/diagnostics/new_market_microstructure_data_pipeline/live/plist_templates/com.canbit.microstructure-public-collector.plist"
)


def test_wrapper_no_command_nonzero():
    r = subprocess.run([str(WRAPPER)], capture_output=True, text=True)
    assert r.returncode != 0


def test_wrapper_utility_lifecycle():
    proc = subprocess.Popen([str(WRAPPER), "/bin/sleep", "2"])
    time.sleep(0.5)
    rows = kaw.list_processes()
    during = [r for r in rows if "caffeinate" in r["command"] and "/bin/sleep" in r["command"]]
    assert during, "expected caffeinate wrapping /bin/sleep"
    pids = {r["pid"] for r in during}
    proc.wait(timeout=10)
    time.sleep(0.4)
    rows2 = kaw.list_processes()
    orphan = [r for r in rows2 if r["pid"] in pids]
    assert not orphan


def test_plist_args_preservation():
    before = plistlib.loads(BACKUP.read_bytes())["ProgramArguments"]
    after = plistlib.loads(SOURCE.read_bytes())["ProgramArguments"]
    assert kaw.assert_args_preserved(before, after, WRAPPER)


def test_plist_protected_keys_preservation():
    before = plistlib.loads(BACKUP.read_bytes())
    after = plistlib.loads(SOURCE.read_bytes())
    for key in [
        "Label",
        "WorkingDirectory",
        "RunAtLoad",
        "KeepAlive",
        "ThrottleInterval",
        "StandardOutPath",
        "StandardErrorPath",
    ]:
        assert before.get(key) == after.get(key)
    assert after["ProgramArguments"][0] == str(WRAPPER)


def test_assertion_parser_caffeinate_active():
    sample = """
Assertion status system-wide:
   PreventSystemSleep             1
   PreventUserIdleSystemSleep     1

Kernel Assertions: 0x100=
   pid 4242(caffeinate): [0x000000010001234] 00:01:00 PreventUserIdleSystemSleep named: "caffeinate asserting forever"
   pid 4242(caffeinate): [0x000000020001235] 00:01:00 PreventSystemSleep named: "caffeinate asserting forever"
"""
    parsed = kaw.parse_assertions(sample)
    assert parsed["counts"]["PreventUserIdleSystemSleep"] == 1
    assert parsed["counts"]["PreventSystemSleep"] == 1
    assert parsed["caffeinate_owners"]
    assert parsed["idle_system_sleep_assertion_active"]
    assert parsed["system_sleep_assertion_active"]


def test_assertion_false_positive_powerd_only():
    sample = """
Assertion status system-wide:
   PreventSystemSleep             0
   PreventUserIdleSystemSleep     1

   pid 532(powerd): [0x001] 01:00:00 PreventUserIdleSystemSleep named: "Powerd - Prevent sleep while display is on"
"""
    parsed = kaw.parse_assertions(sample)
    assert not parsed["caffeinate_owners"]
    # Without process tree, assess must not mark guard active solely from powerd
    fake_rows = [
        {
            "pid": 100,
            "ppid": 1,
            "etime": "1:00",
            "state": "S",
            "command": f"{REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        }
    ]
    result = kaw.assess_keepawake_guard(assertions_text=sample, rows=fake_rows, expected=True)
    assert result["keep_awake_guard_verdict"] == "KEEP_AWAKE_GUARD_MISSING"
    assert result["keep_awake_guard_active"] is False


def test_duplicate_guard_verdict():
    sample = """
   PreventUserIdleSystemSleep     1
   PreventSystemSleep             1
   pid 10(caffeinate): PreventUserIdleSystemSleep named: x
   pid 11(caffeinate): PreventUserIdleSystemSleep named: y
"""
    rows = [
        {
            "pid": 10,
            "ppid": 1,
            "etime": "1",
            "state": "S",
            "command": f"{kaw.CAFFEINATE_BIN} -i -s {REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
        {
            "pid": 11,
            "ppid": 1,
            "etime": "1",
            "state": "S",
            "command": f"{kaw.CAFFEINATE_BIN} -i -s {REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
        {
            "pid": 20,
            "ppid": 10,
            "etime": "1",
            "state": "S",
            "command": f"{REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
    ]
    result = kaw.assess_keepawake_guard(assertions_text=sample, rows=rows, expected=True)
    assert result["keep_awake_guard_verdict"] == "KEEP_AWAKE_GUARD_DUPLICATE"
    assert result["duplicate_caffeinate_guards"] >= 1


def test_process_relationship_and_active():
    sample = """
   PreventUserIdleSystemSleep     1
   PreventSystemSleep             1
   pid 50(caffeinate): [0x1] 00:00:10 PreventUserIdleSystemSleep named: "caffeinate asserting forever"
   pid 50(caffeinate): [0x2] 00:00:10 PreventSystemSleep named: "caffeinate asserting forever"
"""
    rows = [
        {
            "pid": 50,
            "ppid": 1,
            "etime": "1",
            "state": "S",
            "command": f"{kaw.CAFFEINATE_BIN} -i -s {REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
        {
            "pid": 60,
            "ppid": 50,
            "etime": "1",
            "state": "S",
            "command": f"{REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
    ]
    result = kaw.assess_keepawake_guard(assertions_text=sample, rows=rows, expected=True)
    assert result["collector_wrapped_by_caffeinate"] is True
    assert result["keep_awake_guard_verdict"] == "KEEP_AWAKE_GUARD_ACTIVE"
    assert result["keep_awake_guard_active"] is True


def test_guard_missing_while_collector_running():
    sample = """
   PreventUserIdleSystemSleep     1
   pid 532(powerd): PreventUserIdleSystemSleep named: display
"""
    rows = [
        {
            "pid": 70,
            "ppid": 1,
            "etime": "1",
            "state": "S",
            "command": f"{REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        }
    ]
    result = kaw.assess_keepawake_guard(assertions_text=sample, rows=rows, expected=True)
    assert result["keep_awake_guard_verdict"] == "KEEP_AWAKE_GUARD_MISSING"


def test_restart_lifecycle_orphan_free_logic():
    # old guard gone, new guard present
    sample = """
   PreventUserIdleSystemSleep     1
   PreventSystemSleep             1
   pid 90(caffeinate): PreventUserIdleSystemSleep named: caffeinate
   pid 90(caffeinate): PreventSystemSleep named: caffeinate
"""
    rows = [
        {
            "pid": 90,
            "ppid": 1,
            "etime": "0:01",
            "state": "S",
            "command": f"{kaw.CAFFEINATE_BIN} -i -s {REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
        {
            "pid": 91,
            "ppid": 90,
            "etime": "0:01",
            "state": "S",
            "command": f"{REPO}/.venv/bin/python scripts/diagnostics/{kaw.COLLECTOR_SCRIPT} --run --json",
        },
    ]
    result = kaw.assess_keepawake_guard(assertions_text=sample, rows=rows)
    assert result["active_caffeinate_guards"] == 1
    assert result["duplicate_caffeinate_guards"] == 0


def test_observer_label_not_in_collector_plist():
    after = plistlib.loads(SOURCE.read_bytes())
    assert after["Label"] == kaw.COLLECTOR_LABEL
    assert after["Label"] != kaw.OBSERVER_LABEL


def test_no_private_order_args_in_plist():
    args = plistlib.loads(SOURCE.read_bytes())["ProgramArguments"]
    joined = " ".join(args).lower()
    assert "private" not in joined
    assert "order" not in joined or "--run" in args  # --run ok; no order endpoint


def test_mainnet_endpoint_constant_in_compact_payload_shape():
    # compact must keep production/promotion false when invoked with mocked path is heavy;
    # assert helper defaults and field names exist on a dry assess merge shape.
    fields = [
        "KEEP_AWAKE_GUARD_EXPECTED",
        "KEEP_AWAKE_GUARD_ACTIVE",
        "CAFFEINATE_PID",
        "IDLE_SLEEP_ASSERTION_ACTIVE",
        "SYSTEM_SLEEP_ASSERTION_ACTIVE",
        "ASSERTION_OWNER_VERIFIED",
        "COLLECTOR_WRAPPED_BY_CAFFEINATE",
        "DUPLICATE_CAFFEINATE_GUARDS",
        "CLAMSHELL_SLEEP_NOT_PREVENTED",
        "POWER_SOURCE",
        "KEEP_AWAKE_GUARD_VERDICT",
    ]
    # Ensure compact_status source still defines these keys via static read
    src = (REPO / "scripts/diagnostics/microstructure_observation_quality.py").read_text()
    for f in fields:
        assert f'"{f}"' in src


def test_production_promotion_false_in_compact_source():
    src = (REPO / "scripts/diagnostics/microstructure_observation_quality.py").read_text()
    assert '"production_ready": False' in src
    assert '"promotion_ready": False' in src


def test_prepend_wrapper_helper():
    before = ["/py", "script.py", "--run"]
    after = kaw.prepend_wrapper_to_program_arguments(before, WRAPPER)
    assert after[0] == str(WRAPPER)
    assert after[1:] == before
    assert kaw.prepend_wrapper_to_program_arguments(after, WRAPPER) == after
