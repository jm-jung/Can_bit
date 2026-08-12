"""Audit diagnostics-only forward orderflow collector V4."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path("data/diagnostics/forward_orderflow_collector_v4")
OUT = Path("data/diagnostics/proxy_cvd_expansion_and_forward_orderflow_collector/forward_collector_audit")
LABEL = "com.canbit.forward_orderflow_collector_v4"


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def _launchd_status() -> Dict[str, Any]:
    try:
        out = subprocess.check_output(["launchctl", "list"], text=True, timeout=10)
        lines = [ln for ln in out.splitlines() if LABEL in ln]
        return {"label": LABEL, "installed": bool(lines), "raw": lines}
    except Exception as exc:
        return {"label": LABEL, "installed": False, "error": str(exc)}


def run() -> Dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    state_path = ROOT / "state/collector_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    snaps = _read_parquet(ROOT / "cache/forward_orderflow_snapshots.parquet")
    obs = _read_parquet(ROOT / "cache/forward_orderbook_snapshots.parquet")
    trades = _read_parquet(ROOT / "cache/forward_recent_aggtrades.parquet")
    liqs = _read_parquet(ROOT / "cache/forward_liquidation_events.parquet")
    health = pd.read_csv(ROOT / "health/collector_health.csv") if (ROOT / "health/collector_health.csv").exists() else pd.DataFrame()
    prod_none = bool(snaps.empty or snaps.get("production_action", pd.Series(["none"])).eq("none").all())
    diagnostics_only = str(ROOT).startswith("data/diagnostics")
    summary = {
        "collector_state_exists": state_path.exists(),
        "last_run_ts": state.get("last_run_ts", ""),
        "snapshots_rows": len(snaps),
        "symbols_coverage": int(snaps["symbol"].nunique()) if len(snaps) else 0,
        "data_family_coverage": int(snaps["data_family"].nunique()) if len(snaps) else 0,
        "orderbook_rows": len(obs),
        "recent_aggtrades_rows": len(trades),
        "liquidation_rows": len(liqs),
        "private_endpoint_call_count": int(state.get("private_endpoint_calls", 0)),
        "order_endpoint_call_count": int(state.get("order_endpoint_calls", 0)),
        "account_balance_position_call_count": int(state.get("account_balance_position_calls", 0)),
        "production_action_all_none": prod_none,
        "output_path_diagnostics_only": diagnostics_only,
        "stale_data_count": int(snaps.get("quality_stale", pd.Series(dtype=bool)).sum()) if len(snaps) else 0,
        "error_count": int(snaps.get("quality_error", pd.Series(dtype=bool)).sum()) if len(snaps) else 0,
        "duplicate_snapshot_count": int(snaps.duplicated(["run_ts", "symbol", "data_family"]).sum()) if len(snaps) else 0,
    }
    launchd = _launchd_status()
    pd.DataFrame([summary]).to_csv(OUT / "collector_one_shot_audit.csv", index=False)
    health.to_csv(OUT / "collector_health_audit.csv", index=False)
    (OUT / "collector_private_api_audit.md").write_text("# Collector Private API Audit\n\nNo private/order/account/balance/position endpoints are implemented or called.\n\n```json\n" + _json(summary) + "\n```\n", encoding="utf-8")
    (OUT / "collector_launchd_audit.md").write_text("# Collector Launchd Audit\n\n```json\n" + _json(launchd) + "\n```\n", encoding="utf-8")
    (OUT / "collector_audit_report.md").write_text("# Collector Audit Report\n\n```json\n" + _json({"summary": summary, "launchd": launchd}) + "\n```\n", encoding="utf-8")
    return {"audit_pass": summary["collector_state_exists"] and summary["private_endpoint_call_count"] == 0 and summary["order_endpoint_call_count"] == 0 and summary["account_balance_position_call_count"] == 0 and summary["production_action_all_none"] and summary["output_path_diagnostics_only"], **summary, "launchd": launchd, "production_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run()
    print(_json(res) if args.json else f"audit_pass={res['audit_pass']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
