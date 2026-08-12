#!/usr/bin/env python3
"""QQQ regime research suite — gate then full external regime research on --full."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from canbit_equity.config import paths as qqq_paths
from canbit_equity.regime.config import EXPECTED_LABELED_HASH, V2_PATH
from canbit_equity.regime.orchestration import run_full_regime_research

DIAG = REPO / "data/diagnostics/equity_etf_qqq_regime"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def same_period_audit() -> dict:
    DIAG.joinpath("precheck").mkdir(parents=True, exist_ok=True)
    DIAG.joinpath("reports").mkdir(parents=True, exist_ok=True)
    labeled_path = qqq_paths()["features"] / "qqq_daily_features_labeled.parquet"
    labeled_hash = sha256_file(labeled_path)

    if V2_PATH.exists():
        art = json.loads(V2_PATH.read_text())
        table = art.get("candidate_table") or []
        starts = {r.get("common_start") for r in table}
        ends = {r.get("common_end") for r in table}
        rows = {r.get("common_rows") for r in table}
        folds = {r.get("fold_ids") for r in table}
        identical = (
            art.get("selection_version") == "SAME_PERIOD_V2"
            and len(starts) == 1
            and len(ends) == 1
            and len(rows) == 1
            and len(folds) == 1
            and art.get("cost_scenario") == "BASE_5bps"
            and art.get("old_holdout_used") is False
            and labeled_hash == EXPECTED_LABELED_HASH
            and art.get("fixed_input_hash") == EXPECTED_LABELED_HASH
        )
        out = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "SAME_PERIOD_COMPARISON_PASS" if identical else "SAME_PERIOD_COMPARISON_FAIL",
            "research_gate": "PASS" if identical else "QQQ_REGIME_RESEARCH_BLOCKED_BY_PERIOD_MISMATCH",
            "source_artifact": str(V2_PATH.relative_to(REPO)),
            "selection_version": art.get("selection_version"),
            "labeled_hash": labeled_hash,
            "labeled_hash_match": labeled_hash == EXPECTED_LABELED_HASH,
            "checks": {
                "identical_ranking_window": identical,
                "identical_fold_ids": len(folds) == 1,
                "identical_cost": art.get("cost_scenario") == "BASE_5bps",
                "old_holdout_used": bool(art.get("old_holdout_used")),
                "legacy_fallback_used": False,
            },
            "common_oos_start": art.get("common_oos_start"),
            "common_oos_end": art.get("common_oos_end"),
            "common_oos_rows": art.get("common_oos_rows"),
            "common_oos_index_hash": art.get("common_oos_index_hash"),
            "selected_candidate": art.get("selected_candidate"),
            "continue_regime_research": bool(identical),
            "current_action": (
                "PROCEED_TO_QQQ_REGIME_FEATURE_RESEARCH" if identical else "STOP_AND_FIX_SAME_PERIOD_COMPARISON"
            ),
            "production_ready": False,
            "promotion_ready": False,
        }
    else:
        out = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verdict": "SAME_PERIOD_COMPARISON_FAIL",
            "research_gate": "QQQ_REGIME_RESEARCH_BLOCKED_BY_PERIOD_MISMATCH",
            "reason": "SAME_PERIOD_V2 artifact missing",
            "legacy_fallback_used": False,
            "continue_regime_research": False,
            "current_action": "STOP_AND_FIX_SAME_PERIOD_COMPARISON",
            "production_ready": False,
            "promotion_ready": False,
        }

    (DIAG / "precheck/same_period_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    (DIAG / "precheck/same_period_comparison.md").write_text(
        f"# Same-Period Comparison\n\n**Verdict:** `{out['verdict']}`\ncontinue={out.get('continue_regime_research')}\n"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--same-period-audit", action="store_true")
    g.add_argument("--full", action="store_true")
    g.add_argument("--status", action="store_true")
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.status:
        path = DIAG / "reports/qqq_regime_compact_status.json"
        print(json.dumps(json.loads(path.read_text()) if path.exists() else {"status": "NOT_RUN"}, indent=2))
        return 0

    if args.same_period_audit:
        out = same_period_audit()
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("continue_regime_research") else 2

    if args.full:
        gate = same_period_audit()
        if not gate.get("continue_regime_research"):
            blocked = {
                "research_verdict": "QQQ_REGIME_RESEARCH_BLOCKED_BY_PERIOD_MISMATCH",
                "gate": gate,
                "external_data_downloaded": False,
                "external_research_run": False,
                "production_ready": False,
                "promotion_ready": False,
                "current_action": "STOP_AND_FIX_SAME_PERIOD_COMPARISON",
            }
            print(json.dumps(blocked, indent=2, default=str))
            return 2
        # Gate PASS → run full external regime research (NOT gate-only)
        result = run_full_regime_research()
        result = {
            **result,
            "gate": gate.get("verdict"),
            "research_gate": gate.get("research_gate"),
            "continue_regime_research": True,
            "corrected_selection_source": gate.get("source_artifact"),
        }
        print(json.dumps(result, indent=2, default=str))
        return 0 if result.get("external_research_run") else 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
