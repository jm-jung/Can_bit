"""
Diagnostics-only forward Research V3 relative alpha logger skeleton.

Default behavior is design/dry-run/one-shot only. No launchd install here, no
private/order/account/position APIs, production_action_none always true.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd

ROOT = Path("data/diagnostics/research_v3_cross_symbol_relative_alpha/forward_design")
STATE = Path("data/diagnostics/forward_research_v3_relative_alpha_logger/state")


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in sections.items():
        lines += [f"## {k}", ""]
        if isinstance(v, (dict, list)):
            lines += ["```json", _json(v), "```"]
        else:
            lines.append(str(v))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


SCHEMA = {
    "research_v3_paper_trade_id": "string",
    "run_ts": "datetime64[ns, UTC]",
    "candidate_ts": "datetime64[ns, UTC]",
    "entry_ts": "datetime64[ns, UTC]",
    "symbol": "string",
    "cluster": "string",
    "direction": "string",
    "alpha_id": "string",
    "alpha_name": "string",
    "generator_id": "string",
    "market_regime": "string",
    "breadth_regime": "string",
    "relative_strength_rank": "float",
    "relative_strength_score": "float",
    "leader_symbol": "string",
    "leader_lag_horizon": "string",
    "leader_lagger_score": "float",
    "rotation_score": "float",
    "timeframe_stack": "string",
    "trigger_5m_context": "string",
    "expected_edge_score": "float",
    "bad_regime_score": "float",
    "q2_snapshot_if_available": "string",
    "r7_snapshot_if_available": "string",
    "tcn_snapshot_if_available": "string",
    "paper_entry_price": "float",
    "exit_policy_id": "string",
    "pending_or_resolved": "string",
    "resolution_ts": "datetime64[ns, UTC]",
    "paper_exit_price": "float",
    "net_after_cost": "float",
    "MFE": "float",
    "MAE": "float",
    "RFE": "bool",
    "entry_quality_label": "string",
    "exit_quality_label": "string",
    "censored_flag": "bool",
    "allowed_usage": "diagnostics_paper_only",
    "production_action_none": "bool",
}


def _ensure_design() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    STATE.mkdir(parents=True, exist_ok=True)
    _write_md(ROOT / "forward_research_v3_relative_alpha_logger_design.md", "Forward Research V3 Relative Alpha Logger Design", {
        "status": "design/dry-run/one-shot skeleton",
        "production_action": "none",
        "private_api_calls": False,
        "order_endpoint_calls": False,
        "install_policy": "not installed by default; requires explicit approval",
    })
    (ROOT / "forward_research_v3_trade_schema.json").write_text(_json(SCHEMA), encoding="utf-8")
    _write_md(ROOT / "forward_research_v3_discord_message_example.md", "Forward Research V3 Discord Example", {"message": "[DIAGNOSTICS ONLY] V3 relative alpha production_action=none candidates=N"})
    _write_md(ROOT / "forward_research_v3_milestone_plan.md", "Forward Research V3 Milestone Plan", {"milestones": [20, 50, 100, 200, 500]})
    pd.DataFrame([{"check": c, "required": True} for c in ["no_private_api", "no_order_endpoint", "relative_strength_lookahead_audit", "breadth_lookahead_audit", "leader_lagger_direction_audit", "production_action_none"]]).to_csv(ROOT / "forward_research_v3_quality_control_checklist.csv", index=False)


def run(dry_run: bool = False, once: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "production_action": "none", "private_api_calls": False, "order_endpoint_calls": False}
    _ensure_design()
    run_ts = datetime.now(timezone.utc).isoformat()
    pending = STATE / "forward_research_v3_pending_trades.parquet"
    resolved = STATE / "forward_research_v3_resolved_trades.parquet"
    if not pending.exists():
        pd.DataFrame(columns=SCHEMA.keys()).to_parquet(pending, index=False)
    if not resolved.exists():
        pd.DataFrame(columns=SCHEMA.keys()).to_parquet(resolved, index=False)
    summary = {
        "run_ts": run_ts,
        "mode": "once" if once else "design",
        "candidates": 0,
        "pending": len(pd.read_parquet(pending)),
        "resolved": len(pd.read_parquet(resolved)),
        "production_action": "none",
        "production_ready": False,
        "promotion_ready": False,
        "private_api_calls": False,
        "order_endpoint_calls": False,
        "note": "Skeleton one-shot creates/validates state schema; historical V3 script performs full backfill.",
    }
    (STATE / "last_forward_research_v3_run_summary.json").write_text(_json(summary), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run, once=args.once)
    print(_json(result) if args.json else f"forward_v3_relative_logger production_action=none candidates={result.get('candidates', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
