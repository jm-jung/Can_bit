"""Diagnostics-only forward BTC-centric V4-lite logger skeleton."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path("data/diagnostics/public_v4_lite_btc_centric_retry/forward_design")
STATE = Path("data/diagnostics/forward_btc_centric_v4_lite_logger/state")

SCHEMA = {
    "btc_v4_lite_paper_trade_id": "string",
    "run_ts": "datetime",
    "candidate_ts": "datetime",
    "entry_ts": "datetime",
    "symbol": "BTCUSDT",
    "direction": "string",
    "generator_id": "string",
    "feature_group": "string",
    "btc_orderflow_score": "float",
    "context_risk_on_score": "float",
    "expected_edge_score": "float",
    "paper_entry_price": "float",
    "pending_or_resolved": "string",
    "production_action_none": "bool",
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in sections.items():
        lines += [f"## {k}", "", _json(v) if isinstance(v, (dict, list)) else str(v), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_design() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    _write_md(ROOT / "forward_btc_centric_v4_lite_logger_design.md", "Forward BTC-centric V4-lite Logger Design", {"target": "BTCUSDT", "context_symbols": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"], "production_action": "none", "install_default": False})
    (ROOT / "forward_btc_centric_v4_lite_trade_schema.json").write_text(_json(SCHEMA), encoding="utf-8")
    _write_md(ROOT / "forward_btc_centric_v4_lite_discord_message_example.md", "Discord Example", {"message": "[DIAGNOSTICS ONLY] BTC V4-lite production_action=none candidates=N"})
    pd.DataFrame([{"check": c, "required": True} for c in ["no_private_api", "no_order_endpoint", "production_action_none", "state_separate_from_live", "candidate_origin_independent"]]).to_csv(ROOT / "forward_btc_centric_v4_lite_quality_control_checklist.csv", index=False)


def run(dry_run: bool = False, once: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "production_action": "none", "private_api_calls": False, "order_endpoint_calls": False}
    _ensure_design()
    pending = STATE / "pending.parquet"
    resolved = STATE / "resolved.parquet"
    if not pending.exists():
        pd.DataFrame(columns=SCHEMA.keys()).to_parquet(pending, index=False)
    if not resolved.exists():
        pd.DataFrame(columns=SCHEMA.keys()).to_parquet(resolved, index=False)
    summary = {"run_ts": datetime.now(timezone.utc).isoformat(), "mode": "once" if once else "design", "pending": len(pd.read_parquet(pending)), "resolved": len(pd.read_parquet(resolved)), "production_action": "none", "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}
    (STATE / "last_run_summary.json").write_text(_json(summary), encoding="utf-8")
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--once", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args.dry_run, args.once)
    print(_json(res) if args.json else "forward_btc_v4_lite production_action=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
