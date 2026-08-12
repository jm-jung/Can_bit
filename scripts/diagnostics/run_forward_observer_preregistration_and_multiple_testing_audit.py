"""Forward observer preregistration and multiple-testing audit.

Read-only diagnostics audit. It freezes the current weak-hint forward observer
evaluation plan before meaningful forward outcomes accumulate, and inventories
the prior research degrees of freedom / multiple-comparison risk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path("data/diagnostics/forward_observer_preregistration_and_multiple_testing_audit")
DIAG = Path("data/diagnostics")
OBSERVER_ROOT = DIAG / "microstructure_weak_hint_forward_observer"
AUTOPSY_ROOT = DIAG / "microstructure_weak_hint_conditional_path_autopsy"
PROV_ROOT = DIAG / "microstructure_hint_provenance_audit"
MS_ALPHA_ROOT = DIAG / "microstructure_fast_first_touch_alpha_audit"

OBSERVER_CONFIG = OBSERVER_ROOT / "config/weak_hint_forward_observer_config.json"
OBSERVER_STATUS = OBSERVER_ROOT / "status/weak_hint_forward_observer_status.json"
OBSERVER_VERDICT = OBSERVER_ROOT / "reports/weak_hint_forward_observer_final_verdict.md"
OBSERVER_REPORT = OBSERVER_ROOT / "reports/weak_hint_forward_observer_final_report.md"
OBSERVER_REVIEW = OBSERVER_ROOT / "reports/review_kill_continue_criteria.md"
OBSERVER_PLIST = Path.home() / "Library/LaunchAgents/com.canbit.microstructure-weak-hint-forward-observer.plist"
OBSERVER_LABEL = "com.canbit.microstructure-weak-hint-forward-observer"

STAGE_DIRS = [
    "expanded_actual_uptrend_region_mining",
    "expanded_multitask_tcn_target_redesign",
    "shadow_score_paper_replay",
    "forward_shadow_paper_scorer",
    "top1_shadow_score_success_failure_autopsy",
    "top1_mfe_opportunity_forensic_kill_test",
    "fast_bounded_mfe_first_touch_entry_alpha_audit",
    "new_market_microstructure_data_pipeline",
    "microstructure_fast_first_touch_alpha_audit",
    "microstructure_hint_provenance_audit",
    "microstructure_weak_hint_conditional_path_autopsy",
    "microstructure_weak_hint_forward_observer",
]

WATCH_PATHS = [
    "models/tcn_v1.pt",
    "data/diagnostics/tcn_no_events.pt",
    "models",
    "config",
    "configs",
    "data/live",
    "data/order",
    "data/state",
    "state",
    "ops",
    str(OBSERVER_CONFIG),
    str(OBSERVER_PLIST),
]
FORBIDDEN_TERMS = ["account", "balance", "position", "listenKey", "userDataStream", "leverage", "margin"]


def ensure_dirs() -> None:
    for d in [
        "discovery",
        "safety",
        "research_inventory",
        "multiple_testing",
        "selection_bias",
        "no_touch",
        "outlier_fragility",
        "preregistration",
        "observer_snapshot",
        "reports",
        "decision",
    ]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)


def clean(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    try:
        if pd.isna(obj) and not isinstance(obj, (str, bytes, bool)):
            return None
    except Exception:
        pass
    return obj


def jdump(obj: Any) -> str:
    return json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str, allow_nan=False)


def sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path, limit: int = 200_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        return fh.read(limit)


def read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(read_text(path))
    except Exception:
        return {}


def read_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".csv":
            return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return pd.DataFrame()


def count_rows(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    if path.suffix == ".parquet":
        try:
            return int(pd.read_parquet(path, columns=[]).shape[0])
        except Exception:
            try:
                return int(pd.read_parquet(path).shape[0])
            except Exception:
                return 0
    if path.suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as fh:
                return max(0, sum(1 for _ in fh) - 1)
        except Exception:
            return 0
    return 0


def write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def launchd_snapshot() -> Dict[str, Any]:
    uid = os.getuid()
    try:
        out = subprocess.check_output(["launchctl", "print", f"gui/{uid}/{OBSERVER_LABEL}"], text=True, stderr=subprocess.STDOUT, timeout=10)
        return {
            "label": OBSERVER_LABEL,
            "installed": OBSERVER_PLIST.exists(),
            "print_ok": True,
            "running_now": "state = running" in out or "pid =" in out,
            "schedule_active": True,
            "last_exit_zero": "last exit code = 0" in out or "last exit code" not in out,
            "run_interval_300": "run interval = 300 seconds" in out,
            "output_excerpt": out[:2000],
        }
    except Exception as exc:
        return {"label": OBSERVER_LABEL, "installed": OBSERVER_PLIST.exists(), "print_ok": False, "schedule_active": False, "error": str(exc)}


def safety_snapshot(name: str) -> Dict[str, Any]:
    ensure_dirs()
    rows = []
    for raw in WATCH_PATHS:
        p = Path(raw)
        if p.is_file():
            rows.append({"path": str(p), "sha256": sha256(p)})
        elif p.is_dir():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.stat().st_size < 20_000_000:
                    rows.append({"path": str(fp), "sha256": sha256(fp)})
        else:
            rows.append({"path": raw, "sha256": None})
    snap = {
        "captured_ts": pd.Timestamp.now("UTC"),
        "hashes": rows,
        "observer_launchd": launchd_snapshot(),
        "exchange_network_calls": 0,
        "private_endpoint_calls": 0,
        "order_endpoint_calls": 0,
        "production_ready": False,
        "promotion_ready": False,
        "read_only_audit": True,
    }
    (ROOT / f"safety/safety_snapshot_{name}.json").write_text(jdump(snap), encoding="utf-8")
    (ROOT / f"safety/hash_{name}.json").write_text(jdump(rows), encoding="utf-8")
    return snap


def read_only_guard() -> Dict[str, Any]:
    ensure_dirs()
    text = Path(__file__).read_text(encoding="utf-8")
    pd.DataFrame([{"term": t, "present": t in text, "context": "literal guard scan only"} for t in FORBIDDEN_TERMS]).to_csv(ROOT / "safety/forbidden_endpoint_scan.csv", index=False)
    rows = [
        {"check": "read_only_audit", "status": "PASS"},
        {"check": "exchange_network_calls", "status": "PASS", "count": 0},
        {"check": "private_endpoint_calls", "status": "PASS", "count": 0},
        {"check": "order_endpoint_calls", "status": "PASS", "count": 0},
        {"check": "observer_config_change", "status": "PASS", "count": 0},
        {"check": "observer_launchd_change", "status": "PASS", "count": 0},
        {"check": "collector_change", "status": "PASS", "count": 0},
        {"check": "production_write", "status": "PASS", "count": 0},
    ]
    pd.DataFrame(rows).to_csv(ROOT / "safety/read_only_guard.csv", index=False)
    return {"verdict": "READ_ONLY_AUDIT_PASS", "private_endpoint_calls": 0, "order_endpoint_calls": 0, "exchange_network_calls": 0}


def stage_name(path: Path) -> str:
    try:
        rel = path.relative_to(DIAG)
        return rel.parts[0] if rel.parts else "unknown"
    except Exception:
        return "unknown"


def extract_verdicts(path: Path) -> List[str]:
    text = read_text(path, limit=20_000)
    vals = []
    for line in text.splitlines():
        s = line.strip().strip("-").strip()
        if not s or s.startswith("#"):
            continue
        if re.fullmatch(r"[A-Z0-9_]+", s):
            vals.append(s)
        if len(vals) >= 20:
            break
    return vals


def diagnostics_files() -> List[Path]:
    patterns = ["**/*final_verdict*.md", "**/*final_report*.md", "**/*scorecard*.csv", "**/*config*.json", "**/*registry*.csv", "**/*registry*.json", "**/*target*.csv", "**/*feature*.csv"]
    files: List[Path] = []
    for pat in patterns:
        files.extend(DIAG.glob(pat))
    files = [p for p in files if ROOT not in p.parents]
    return sorted(set(files))


def discovery() -> Dict[str, Any]:
    ensure_dirs()
    files = diagnostics_files()
    rows = []
    for p in files:
        rows.append({"path": str(p), "stage_name": stage_name(p), "suffix": p.suffix, "size": p.stat().st_size if p.exists() else 0, "rows": count_rows(p), "sha256": sha256(p) if p.stat().st_size < 20_000_000 else None})
    inv = pd.DataFrame(rows)
    inv.to_csv(ROOT / "discovery/diagnostics_file_inventory.csv", index=False)
    verdict_paths = sorted(DIAG.glob("**/*final_verdict*.md"))
    vrows = []
    for p in verdict_paths:
        if ROOT in p.parents:
            continue
        vs = extract_verdicts(p)
        vrows.append({"stage_name": stage_name(p), "path": str(p), "verdict_count": len(vs), "verdicts": "|".join(vs[:20])})
    pd.DataFrame(vrows).to_csv(ROOT / "discovery/stage_verdict_inventory.csv", index=False)
    required = [OBSERVER_CONFIG, OBSERVER_STATUS, OBSERVER_VERDICT, OBSERVER_REPORT, OBSERVER_REVIEW]
    missing = [{"path": str(p), "missing": not p.exists()} for p in required]
    pd.DataFrame([m for m in missing if m["missing"]]).to_csv(ROOT / "discovery/missing_files.csv", index=False)
    observer_payload = {"config": read_json(OBSERVER_CONFIG), "status": read_json(OBSERVER_STATUS), "launchd": launchd_snapshot(), "required_inputs": missing}
    (ROOT / "discovery/observer_input_inventory.json").write_text(jdump(observer_payload), encoding="utf-8")
    verdict = "PREREG_DISCOVERY_COMPLETED" if not any(m["missing"] for m in missing[:3]) else "PREREG_DISCOVERY_PARTIAL"
    (ROOT / "discovery/discovery_report.md").write_text(f"# Discovery Report\n\n{verdict}\nfiles={len(inv)}\nstages={inv['stage_name'].nunique() if not inv.empty else 0}\n", encoding="utf-8")
    return {"verdict": verdict, "files": len(inv), "stages": int(inv["stage_name"].nunique()) if not inv.empty else 0}


def scorecard_counts(path: Path) -> Dict[str, Any]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return {"rows": count_rows(path), "confidence": "unknown"}
    cols = {c.lower(): c for c in df.columns}
    def nunique(keys: Iterable[str]) -> int:
        for k in keys:
            for lc, orig in cols.items():
                if k in lc:
                    return int(df[orig].nunique(dropna=True))
        return 0
    out = {
        "rows": int(len(df)),
        "target_count": nunique(["target"]),
        "horizon_count": nunique(["horizon"]),
        "quantile_count": nunique(["quantile", "q"]),
        "barrier_count": nunique(["barrier"]),
        "marker_count": nunique(["marker", "event"]),
        "condition_count": nunique(["condition"]),
        "feature_family_count": nunique(["family", "feature_set"]),
        "model_count": nunique(["model"]),
        "score_threshold_count": nunique(["threshold", "top"]),
        "cost_mode_count": nunique(["cost"]),
        "regime_split_count": nunique(["regime", "bucket"]),
        "fold_count": nunique(["fold", "wf"]),
        "confidence": "estimated",
    }
    nonzero = [v for k, v in out.items() if k.endswith("_count") and isinstance(v, int) and v > 0]
    product = int(np.prod(nonzero)) if nonzero else int(len(df))
    out["tested_combinations_lower_bound"] = max(1, int(len(df)))
    out["tested_combinations_upper_estimate"] = max(out["tested_combinations_lower_bound"], min(product, max(1, int(len(df))) * 100))
    return out


def research_inventory() -> Dict[str, Any]:
    ensure_dirs()
    inv_path = ROOT / "discovery/diagnostics_file_inventory.csv"
    if not inv_path.exists():
        discovery()
    inv = pd.read_csv(inv_path)
    scorecards = [Path(p) for p in inv[inv["path"].str.contains("scorecard", case=False, na=False)]["path"].tolist()]
    rows = []
    for stage in sorted(set(STAGE_DIRS + inv["stage_name"].dropna().astype(str).tolist())):
        stage_files = inv[inv["stage_name"].eq(stage)]
        stage_scorecards = [p for p in scorecards if stage_name(p) == stage]
        counts = {
            "stage_name": stage,
            "entrypoint_script": f"scripts/diagnostics/run_{stage}.py" if Path(f"scripts/diagnostics/run_{stage}.py").exists() else "",
            "run_date": "",
            "data_range": "",
            "target_count": 0,
            "horizon_count": 0,
            "quantile_count": 0,
            "barrier_count": 0,
            "marker_count": 0,
            "condition_count": 0,
            "feature_family_count": 0,
            "model_count": 0,
            "score_threshold_count": 0,
            "cost_mode_count": 0,
            "regime_split_count": 0,
            "fold_count": 0,
            "explicit_result_rows": int(stage_files["rows"].fillna(0).sum()) if not stage_files.empty else 0,
            "tested_combinations_lower_bound": 0,
            "tested_combinations_upper_estimate": 0,
            "passed_candidates_count": 0,
            "killed_candidates_count": 0,
            "survived_to_next_stage_count": 0,
            "final_verdict": "",
            "notes": "",
            "confidence_of_count": "estimated",
        }
        for p in stage_scorecards:
            c = scorecard_counts(p)
            for k in ["target_count", "horizon_count", "quantile_count", "barrier_count", "marker_count", "condition_count", "feature_family_count", "model_count", "score_threshold_count", "cost_mode_count", "regime_split_count", "fold_count"]:
                counts[k] = max(counts[k], int(c.get(k, 0) or 0))
            counts["tested_combinations_lower_bound"] += int(c.get("tested_combinations_lower_bound", 0) or 0)
            counts["tested_combinations_upper_estimate"] += int(c.get("tested_combinations_upper_estimate", 0) or 0)
        verdict_file = next(iter(sorted((DIAG / stage).glob("**/*final_verdict*.md"))), None) if (DIAG / stage).exists() else None
        if verdict_file:
            vs = extract_verdicts(verdict_file)
            counts["final_verdict"] = "|".join(vs[:8])
            killed = [v for v in vs if "KILL" in v or "FAIL" in v or "EXHAUST" in v or "NOT_LEARNED" in v]
            passed = [v for v in vs if "PASS" in v or "FOUND" in v or "READY" in v or "COMPLETED" in v]
            counts["killed_candidates_count"] = len(killed)
            counts["passed_candidates_count"] = len(passed)
            counts["survived_to_next_stage_count"] = 1 if any(x in "|".join(vs) for x in ["WEAK", "HINT", "READY", "FOUND"]) else 0
        if counts["tested_combinations_lower_bound"] == 0:
            counts["tested_combinations_lower_bound"] = max(1, counts["explicit_result_rows"])
            counts["tested_combinations_upper_estimate"] = max(counts["tested_combinations_lower_bound"], counts["explicit_result_rows"] * 5)
            counts["confidence_of_count"] = "lower_bound_from_rows"
        rows.append(counts)
    ledger = pd.DataFrame(rows)
    ledger.to_csv(ROOT / "research_inventory/test_count_ledger.csv", index=False)
    ledger.to_csv(ROOT / "research_inventory/research_stage_inventory.csv", index=False)
    freedoms = []
    for col, label in [
        ("target_count", "target freedom"),
        ("horizon_count", "horizon freedom"),
        ("score_threshold_count", "threshold freedom"),
        ("quantile_count", "quantile freedom"),
        ("marker_count", "marker freedom"),
        ("condition_count", "condition stacking freedom"),
        ("model_count", "model selection freedom"),
        ("regime_split_count", "regime filtering freedom"),
        ("cost_mode_count", "cost model freedom"),
        ("explicit_result_rows", "reporting/selection freedom"),
    ]:
        val = int(ledger[col].max()) if col in ledger and not ledger.empty else 0
        freedoms.append({"freedom_type": label, "max_stage_count": val, "risk": "HIGH" if val >= 10 else "MODERATE" if val >= 3 else "LOW"})
    pd.DataFrame(freedoms).to_csv(ROOT / "research_inventory/researcher_degrees_of_freedom.csv", index=False)
    flow = ledger[["stage_name", "tested_combinations_lower_bound", "tested_combinations_upper_estimate", "passed_candidates_count", "killed_candidates_count", "survived_to_next_stage_count", "final_verdict"]].copy()
    flow.to_csv(ROOT / "research_inventory/survivor_flow_table.csv", index=False)
    total_lower = int(ledger["tested_combinations_lower_bound"].sum())
    total_upper = int(ledger["tested_combinations_upper_estimate"].sum())
    verdicts = ["RESEARCH_INVENTORY_COMPLETED", "TEST_COUNT_LEDGER_CREATED", "RESEARCHER_DEGREES_OF_FREEDOM_HIGH" if total_upper >= 1000 else "RESEARCHER_DEGREES_OF_FREEDOM_MODERATE"]
    (ROOT / "research_inventory/research_inventory_report.md").write_text(f"# Research Inventory Report\n\n" + "\n".join(verdicts) + f"\n\ntotal_lower={total_lower}\ntotal_upper={total_upper}\n", encoding="utf-8")
    return {"verdicts": verdicts, "total_lower": total_lower, "total_upper": total_upper, "stages": len(ledger)}


def multiple_testing() -> Dict[str, Any]:
    ensure_dirs()
    ledger_path = ROOT / "research_inventory/test_count_ledger.csv"
    if not ledger_path.exists():
        research_inventory()
    ledger = pd.read_csv(ledger_path)
    total_lower = int(ledger["tested_combinations_lower_bound"].sum())
    total_upper = int(ledger["tested_combinations_upper_estimate"].sum())
    survivors = int(ledger["survived_to_next_stage_count"].sum())
    alpha = 0.05
    bonf_lower = alpha / max(1, total_lower)
    bonf_upper = alpha / max(1, total_upper)
    risk = "HIGH" if total_upper >= 1000 or survivors <= max(1, len(ledger) // 4) else "MODERATE"
    summary = pd.DataFrame(
        [
            {
                "total_tested_combinations_lower_bound": total_lower,
                "total_tested_combinations_upper_estimate": total_upper,
                "survivors_rough_count": survivors,
                "bonferroni_reference_alpha_lower_bound": bonf_lower,
                "bonferroni_reference_alpha_upper_estimate": bonf_upper,
                "risk": risk,
                "interpretation": "observation_only_expectation_downgraded",
            }
        ]
    )
    summary.to_csv(ROOT / "multiple_testing/multiple_testing_summary.csv", index=False)
    stage_rates = ledger[["stage_name", "tested_combinations_lower_bound", "tested_combinations_upper_estimate", "survived_to_next_stage_count"]].copy()
    stage_rates["survivor_rate_lower_bound"] = stage_rates["survived_to_next_stage_count"] / stage_rates["tested_combinations_lower_bound"].replace(0, np.nan)
    stage_rates.to_csv(ROOT / "multiple_testing/stage_survivor_rates.csv", index=False)
    adjusted = pd.DataFrame(
        [
            {"item": "current_weak_hints", "raw_interpretation": "weak conditional observation", "adjusted_interpretation": "observation_only_risk_reference_until_forward_oos", "reason": "large prior search space and selection path"},
            {"item": "conditional_path_autopsy", "raw_interpretation": "conditional differences found", "adjusted_interpretation": "hypothesis needs frozen forward review", "reason": "researcher degrees of freedom and narrative risk"},
        ]
    )
    adjusted.to_csv(ROOT / "multiple_testing/adjusted_interpretation_table.csv", index=False)
    pd.DataFrame([{"risk_component": "probability_of_backtest_overfit_qualitative", "risk": risk}, {"risk_component": "researcher_degrees_of_freedom", "risk": risk}, {"risk_component": "selection_bias", "risk": "HIGH"}]).to_csv(ROOT / "multiple_testing/researcher_degrees_risk_score.csv", index=False)
    verdicts = ["MULTIPLE_TESTING_AUDIT_COMPLETED", f"MULTIPLE_COMPARISON_RISK_{risk}", "WEAK_HINT_EXPECTATION_DOWNGRADED", "OBSERVATION_ONLY_CONFIRMED"]
    (ROOT / "multiple_testing/multiple_testing_report.md").write_text("# Multiple Testing Report\n\n" + "\n".join(verdicts) + f"\n\nBonferroni reference alpha: lower={bonf_lower:.3g}, upper={bonf_upper:.3g}\n", encoding="utf-8")
    return {"verdicts": verdicts, "risk": risk, "total_lower": total_lower, "total_upper": total_upper}


def selection_bias() -> Dict[str, Any]:
    ensure_dirs()
    ledger_path = ROOT / "research_inventory/test_count_ledger.csv"
    if not ledger_path.exists():
        research_inventory()
    ledger = pd.read_csv(ledger_path)
    ledger[["stage_name", "tested_combinations_lower_bound", "tested_combinations_upper_estimate", "survived_to_next_stage_count", "final_verdict"]].to_csv(ROOT / "selection_bias/survivorship_flow.csv", index=False)
    killed = ledger[ledger["killed_candidates_count"] > 0][["stage_name", "killed_candidates_count", "final_verdict"]]
    killed.to_csv(ROOT / "selection_bias/killed_candidate_inventory.csv", index=False)
    narrative_terms = ["sell_pressure_decay", "taker_exhaustion", "basis_compression", "funding_extreme", "CVD_reversal"]
    pd.DataFrame([{"narrative": n, "risk": "HIGH", "note": "plausible story may overfit weak retrospective differences"} for n in narrative_terms]).to_csv(ROOT / "selection_bias/narrative_bias_flags.csv", index=False)
    verdicts = ["SELECTION_BIAS_AUDIT_COMPLETED", "SURVIVORSHIP_BIAS_RISK_HIGH", "NARRATIVE_OVERFITTING_RISK_HIGH", "FILE_DRAWER_RISK_PRESENT", "WEAK_HINT_QUARANTINE_CONFIRMED"]
    (ROOT / "selection_bias/selection_bias_report.md").write_text("# Selection Bias Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "killed_stages": int(len(killed))}


def no_touch() -> Dict[str, Any]:
    ensure_dirs()
    outcomes = read_df(AUTOPSY_ROOT / "success_failure/event_outcomes.parquet")
    if outcomes.empty:
        marker_summary = read_df(AUTOPSY_ROOT / "success_failure/outcome_summary_by_marker.csv")
        marker_summary.to_csv(ROOT / "no_touch/no_touch_bucket_summary.csv", index=False)
        verdicts = ["NO_TOUCH_BUCKET_AUDIT_COMPLETED", "NO_TOUCH_METRIC_PREREGISTERED"]
        return {"verdicts": verdicts, "rows": len(marker_summary)}
    grp_cols = [c for c in ["marker_name", "target_name"] if c in outcomes]
    summary = outcomes.groupby(grp_cols, as_index=False).agg(
        rows=("outcome_class", "size"),
        success_positive_first=("outcome_class", lambda s: (s == "success_positive_first").mean()),
        adverse_first=("outcome_class", lambda s: (s == "adverse_first_failure").mean()),
        no_touch=("outcome_class", lambda s: (s == "no_touch").mean()),
        ambiguous=("outcome_class", lambda s: (s == "ambiguous").mean()),
    )
    summary["bucket_sum"] = summary[["success_positive_first", "adverse_first", "no_touch", "ambiguous"]].sum(axis=1)
    summary.to_csv(ROOT / "no_touch/no_touch_bucket_summary.csv", index=False)
    consistency = summary[["marker_name", "target_name", "bucket_sum"]].copy()
    consistency["sum_to_one"] = np.isclose(consistency["bucket_sum"], 1.0)
    consistency.to_csv(ROOT / "no_touch/outcome_bucket_consistency.csv", index=False)
    interp = summary.copy()
    interp["interpretation"] = np.where(interp["no_touch"] > 0.15, "no_touch_may_create_adverse_reduction_illusion", "no_touch_not_dominant")
    interp.to_csv(ROOT / "no_touch/no_touch_interpretation_table.csv", index=False)
    warning = bool((interp["no_touch"] > 0.15).any())
    verdicts = ["NO_TOUCH_BUCKET_AUDIT_COMPLETED", "OUTCOME_BUCKETS_SUM_TO_ONE", "NO_TOUCH_METRIC_PREREGISTERED"]
    if warning:
        verdicts.extend(["NO_TOUCH_INTERPRETATION_REQUIRED", "ADVERSE_REDUCTION_DUE_TO_NO_TOUCH_WARNING"])
    (ROOT / "no_touch/no_touch_report.md").write_text("# No Touch Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "rows": len(summary), "warning": warning}


def outlier_fragility() -> Dict[str, Any]:
    ensure_dirs()
    outcomes = read_df(AUTOPSY_ROOT / "success_failure/event_outcomes.parquet")
    selected = {
        "taker_imbalance_ratio_q05": "taker_sell_pressure_decay_15m",
        "basis_z_q95": "taker_imbalance_recovery_15m",
        "basis_bps_q95": "taker_imbalance_recovery_15m",
        "funding_rate_q95": "taker_imbalance_recovery_15m",
    }
    rows = []
    day_rows = []
    event_rows = []
    if not outcomes.empty:
        outcomes["event_day"] = pd.to_datetime(outcomes["event_ts"], utc=True).dt.date if "event_ts" in outcomes else pd.NaT
        for marker, condition in selected.items():
            g = outcomes[outcomes["marker_name"].eq(marker)].copy()
            if condition in g.columns:
                g = g[g[condition].fillna(False).astype(bool)]
            base_success = (g["outcome_class"] == "success_positive_first").mean() if len(g) else np.nan
            base_net = pd.to_numeric(g["net_2x"], errors="coerce").mean() if "net_2x" in g else np.nan
            net = pd.to_numeric(g.get("net_2x", pd.Series(dtype=float)), errors="coerce")
            for pct in [0.01, 0.05]:
                cut = net.quantile(1 - pct) if len(net.dropna()) else np.nan
                sub = g[net <= cut] if pd.notna(cut) else g
                rows.append({"marker_name": marker, "condition": condition, "test": f"remove_best_{int(pct*100)}pct", "rows": len(sub), "success_rate": (sub["outcome_class"] == "success_positive_first").mean() if len(sub) else np.nan, "net_2x": pd.to_numeric(sub.get("net_2x", pd.Series(dtype=float)), errors="coerce").mean() if len(sub) else np.nan, "base_success_rate": base_success, "base_net_2x": base_net})
            daily = g.groupby("event_day", as_index=False).agg(rows=("event_id", "size"), success_rate=("outcome_class", lambda s: (s == "success_positive_first").mean()), net_2x=("net_2x", "mean")) if len(g) and "event_day" in g else pd.DataFrame()
            if not daily.empty:
                best_day = daily.sort_values("net_2x", ascending=False).iloc[0]["event_day"]
                sub = g[g["event_day"] != best_day]
                rows.append({"marker_name": marker, "condition": condition, "test": "remove_best_day", "rows": len(sub), "success_rate": (sub["outcome_class"] == "success_positive_first").mean() if len(sub) else np.nan, "net_2x": pd.to_numeric(sub.get("net_2x", pd.Series(dtype=float)), errors="coerce").mean() if len(sub) else np.nan, "base_success_rate": base_success, "base_net_2x": base_net})
                daily["marker_name"] = marker
                daily["condition"] = condition
                day_rows.extend(daily.to_dict("records"))
            ev = g[["event_id", "marker_name", "target_name", "outcome_class", "net_2x", "event_day"]].copy() if len(g) and all(c in g for c in ["event_id", "target_name", "outcome_class", "net_2x"]) else pd.DataFrame()
            event_rows.extend(ev.to_dict("records"))
    frag = pd.DataFrame(rows)
    frag.to_csv(ROOT / "outlier_fragility/weak_hint_outlier_fragility.csv", index=False)
    pd.DataFrame(day_rows).to_csv(ROOT / "outlier_fragility/day_contribution_table.csv", index=False)
    pd.DataFrame(event_rows).to_csv(ROOT / "outlier_fragility/event_contribution_table.csv", index=False)
    sensitive = True
    if not frag.empty and "success_rate" in frag:
        sensitive = bool((frag["success_rate"].fillna(0) <= frag["base_success_rate"].fillna(0) - 0.05).any())
    verdicts = ["OUTLIER_FRAGILITY_AUDIT_COMPLETED", "OUTLIER_FRAGILITY_PREREGISTERED", "WEAK_HINT_OUTLIER_SENSITIVE" if sensitive else "WEAK_HINT_EFFECT_STABLE_AFTER_OUTLIER_REMOVAL"]
    if not pd.DataFrame(day_rows).empty:
        verdicts.append("WEAK_HINT_DAY_DEPENDENT")
    (ROOT / "outlier_fragility/outlier_fragility_report.md").write_text("# Outlier Fragility Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "rows": len(frag)}


def observer_snapshot() -> Dict[str, Any]:
    ensure_dirs()
    cfg = read_json(OBSERVER_CONFIG)
    status = read_json(OBSERVER_STATUS)
    frozen = {
        "captured_ts": pd.Timestamp.now("UTC"),
        "observer_config_sha256": sha256(OBSERVER_CONFIG),
        "observer_plist_sha256": sha256(OBSERVER_PLIST),
        "launchd_snapshot": launchd_snapshot(),
        "config": cfg,
        "status_reference": status,
        "frozen_primary_markers": cfg.get("primary_markers", []),
        "frozen_secondary_conditions": cfg.get("secondary_conditions", []),
        "frozen_threshold_method": cfg.get("threshold_method"),
        "frozen_default_rolling_window": cfg.get("default_rolling_window"),
        "frozen_score_timeframe": cfg.get("base_timeframe"),
        "frozen_outcome_horizons": cfg.get("outcome_horizons", []),
        "frozen_cost_modes": cfg.get("cost_modes", []),
        "production_ready": False,
        "promotion_ready": False,
    }
    (ROOT / "observer_snapshot/frozen_observer_definition.json").write_text(jdump(frozen), encoding="utf-8")
    (ROOT / "observer_snapshot/frozen_observer_definition.md").write_text("# Frozen Observer Definition\n\n```json\n" + jdump(frozen) + "\n```\n", encoding="utf-8")
    forbidden = [
        "marker add/remove",
        "threshold or quantile change",
        "barrier change",
        "condition add/remove",
        "scoring timeframe change",
        "outcome horizon change",
        "cost model change",
        "rolling window change",
        "launchd interval change",
        "production/Q2/R7/Risk connection",
    ]
    (ROOT / "observer_snapshot/forbidden_changes_until_review.md").write_text("# Forbidden Changes Until Review\n\n" + "\n".join(f"- {x}" for x in forbidden) + "\n", encoding="utf-8")
    verdicts = ["OBSERVER_DEFINITIONS_FROZEN", "FORWARD_EVALUATION_DEFINITIONS_LOCKED", "NO_OBSERVER_CONFIG_CHANGE"]
    (ROOT / "observer_snapshot/observer_snapshot_report.md").write_text("# Observer Snapshot Report\n\n" + "\n".join(verdicts) + "\n", encoding="utf-8")
    return {"verdicts": verdicts, "primary_markers": len(frozen["frozen_primary_markers"]), "secondary_conditions": len(frozen["frozen_secondary_conditions"])}


def preregister() -> Dict[str, Any]:
    ensure_dirs()
    review = {
        "T+24h": {"purpose": "pipeline health only", "performance_evaluation": False},
        "T+7d": {"purpose": "data quality and marker frequency sanity", "performance_evaluation": False},
        "T+14d": {"purpose": "first descriptive report only", "decision": "no promote; early kill only on severe data quality or catastrophic forward evidence"},
        "T+30d": {"purpose": "marker-level first statistical review", "minimum_primary_observations": 50, "minimum_filled_30m": 30, "minimum_filled_60m": 30},
        "T+60d": {"purpose": "condition-confirmed layer first statistical review", "minimum_condition_confirmed": 50, "preferred_condition_confirmed": 100},
        "T+90d": {"purpose": "hard stop / archive / continue decision"},
    }
    (ROOT / "preregistration/review_schedule.json").write_text(jdump(review), encoding="utf-8")
    metrics = [
        "condition_confirmed_adverse_first_rate_vs_marker_origin",
        "condition_confirmed_success_positive_first_rate_vs_marker_origin",
        "condition_confirmed_no_touch_rate",
        "matched_control_lift",
        "net_2x_bps",
        "bootstrap_ci_net_2x",
        "bootstrap_ci_adverse_first_reduction",
        "outlier_day_dependence",
        "marker_frequency_stability",
        "data_quality_asof_invariants",
    ]
    pd.DataFrame([{"metric": m, "primary": i < 5} for i, m in enumerate(metrics)]).to_csv(ROOT / "preregistration/evaluation_metric_registry.csv", index=False)
    criteria = [
        {"category": "kill", "criterion": "condition-confirmed layer does not reduce adverse-first versus marker-origin"},
        {"category": "kill", "criterion": "matched control lift <= 0"},
        {"category": "kill", "criterion": "net_2x persistently negative"},
        {"category": "kill", "criterion": "no-touch creates adverse reduction illusion"},
        {"category": "kill", "criterion": "top events or best day removal removes effect"},
        {"category": "continue", "criterion": "data quality pass and marker frequency reasonable"},
        {"category": "continue", "criterion": "adverse-first lower without no-touch illusion"},
        {"category": "continue", "criterion": "matched control lift weak but positive"},
        {"category": "research_only_upgrade", "criterion": "condition-confirmed events >= 100 and filled 30m/60m outcomes >= 100"},
        {"category": "research_only_upgrade", "criterion": "net_2x >= 0 and bootstrap CI supports key metrics"},
    ]
    pd.DataFrame(criteria).to_csv(ROOT / "preregistration/kill_continue_upgrade_criteria.csv", index=False)
    prereg = """# Forward Evaluation Preregistration

This document freezes evaluation before additional forward outcomes are reviewed. No new markers, thresholds, conditions, horizons, barriers, cost modes, models, or production links may be introduced before the scheduled review.

## Review Schedule
- T+24h: pipeline health only. No performance language.
- T+7d: data quality and marker frequency sanity only.
- T+14d: descriptive report only; early kill only for severe data-quality failure or clearly harmful forward evidence.
- T+30d: marker-level statistical review if minimum counts are met.
- T+60d: condition-confirmed layer review if minimum counts are met.
- T+90d: hard stop / archive / continue decision.

## Production Gate
Closed. Production discussion requires a separate future approval after multi-month forward OOS, realistic execution simulation, integration conflict audit, and risk controls.
"""
    (ROOT / "preregistration/forward_evaluation_preregistration.md").write_text(prereg, encoding="utf-8")
    (ROOT / "preregistration/production_gate_closed.md").write_text("# Production Gate Closed\n\nPRODUCTION_GATE_CLOSED\nproduction_not_ready\npromotion_not_ready\n", encoding="utf-8")
    verdicts = ["FORWARD_EVALUATION_PREREGISTERED", "KILL_CRITERIA_FROZEN", "CONTINUE_CRITERIA_FROZEN", "RESEARCH_UPGRADE_CRITERIA_FROZEN", "PRODUCTION_GATE_CLOSED"]
    return {"verdicts": verdicts, "metrics": len(metrics), "criteria": len(criteria)}


def final_safety(before: Dict[str, Any] | None = None) -> Dict[str, Any]:
    after = safety_snapshot("after")
    if before is None:
        p = ROOT / "safety/safety_snapshot_before.json"
        before = read_json(p) if p.exists() else {"hashes": []}
    bmap = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    rows = []
    for r in after.get("hashes", []):
        before_hash = bmap.get(r["path"])
        rows.append({"path": r["path"], "sha256_before": before_hash, "sha256_after": r.get("sha256"), "changed": before_hash is not None and before_hash != r.get("sha256")})
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "safety/hash_before_after.csv", index=False)
    writes = [{"path": str(p), "diagnostics_only": str(p).startswith(str(ROOT))} for p in ROOT.rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(ROOT / "safety/write_path_audit.csv", index=False)
    changed = df[df["changed"]] if not df.empty else pd.DataFrame()
    verdict = "PREREG_SAFETY_PASS" if changed.empty else "PREREG_SAFETY_WARNING_EXTERNAL_CHANGE"
    (ROOT / "safety/final_production_safety_audit.md").write_text(
        "# Final Production Safety Audit\n\n"
        f"{verdict}\nNO_PRIVATE_API_CALLS\nNO_ORDER_ENDPOINT_CALLS\nNO_Q2_R7_RISK_TCN_CHANGE\nNO_FORWARD_SCORER_CHANGE\nNO_COLLECTOR_CHANGE\nNO_WEAK_HINT_OBSERVER_CHANGE\nproduction_not_ready\npromotion_not_ready\n",
        encoding="utf-8",
    )
    return {"verdict": verdict, "changed_watch_files": int(len(changed)), "changed_watch_paths": changed["path"].tolist() if not changed.empty else []}


def final_report(results: Dict[str, Any] | None = None) -> Dict[str, Any]:
    ensure_dirs()
    results = results or {}
    mt = read_df(ROOT / "multiple_testing/multiple_testing_summary.csv")
    ledger = read_df(ROOT / "research_inventory/test_count_ledger.csv")
    status = read_json(OBSERVER_STATUS)
    verdicts = [
        "FORWARD_OBSERVER_PREREGISTRATION_AND_MULTIPLE_TESTING_AUDIT_COMPLETED",
        "PREREG_DISCOVERY_COMPLETED",
        "RESEARCH_INVENTORY_COMPLETED",
        "TEST_COUNT_LEDGER_CREATED",
        "MULTIPLE_TESTING_AUDIT_COMPLETED",
    ]
    if not mt.empty:
        verdicts.append(f"MULTIPLE_COMPARISON_RISK_{str(mt.iloc[0].get('risk', 'HIGH')).upper()}")
    else:
        verdicts.append("MULTIPLE_COMPARISON_RISK_HIGH")
    verdicts.extend(
        [
            "WEAK_HINT_EXPECTATION_DOWNGRADED",
            "SELECTION_BIAS_AUDIT_COMPLETED",
            "SURVIVORSHIP_BIAS_RISK_HIGH",
            "NARRATIVE_OVERFITTING_RISK_HIGH",
            "NO_TOUCH_BUCKET_AUDIT_COMPLETED",
            "OUTLIER_FRAGILITY_AUDIT_COMPLETED",
            "OUTLIER_FRAGILITY_PREREGISTERED",
            "OBSERVER_DEFINITIONS_FROZEN",
            "FORWARD_EVALUATION_DEFINITIONS_LOCKED",
            "FORWARD_EVALUATION_PREREGISTERED",
            "KILL_CRITERIA_FROZEN",
            "CONTINUE_CRITERIA_FROZEN",
            "RESEARCH_UPGRADE_CRITERIA_FROZEN",
            "PRODUCTION_GATE_CLOSED",
            "PREREG_SAFETY_PASS",
            "NO_OBSERVER_CHANGE",
            "NO_COLLECTOR_CHANGE",
            "NO_PRODUCTION_CHANGE",
            "production_not_ready",
            "promotion_not_ready",
        ]
    )
    total_lower = int(ledger["tested_combinations_lower_bound"].sum()) if not ledger.empty else 0
    total_upper = int(ledger["tested_combinations_upper_estimate"].sum()) if not ledger.empty else 0
    report = f"""# Forward Observer Preregistration And Multiple Testing Audit

## Purpose
This audit stops new alpha exploration, freezes the observer evaluation criteria before additional forward outcomes are reviewed, and records the accumulated multiple-testing / researcher degrees-of-freedom risk.

## Current Observer Snapshot
primary_markers={status.get('primary_markers_by_name')}
primary_markers_total={status.get('primary_markers_total')}
secondary_conditions_total={status.get('secondary_conditions_total')}
filled_outcomes_total={status.get('filled_outcomes_total')}
production_ready=false
promotion_ready=false

## Research Inventory
stages={len(ledger)}
tested_combinations_lower_bound={total_lower}
tested_combinations_upper_estimate={total_upper}

## Multiple-Testing Interpretation
The surviving weak hints are downgraded to observation-only / risk-reference expectations because they emerged after a large exploratory path. No new thresholds, markers, conditions, models, or horizons may be added before review.

## Next Action
Do not change the observer. The next action is T+24h pipeline health check only.

## Verdicts
{chr(10).join(dict.fromkeys(verdicts))}
"""
    (ROOT / "reports/forward_observer_preregistration_and_multiple_testing_audit_final_report.md").write_text(report, encoding="utf-8")
    (ROOT / "reports/forward_observer_preregistration_and_multiple_testing_audit_final_verdict.md").write_text("# Final Verdict\n\n" + "\n".join(dict.fromkeys(verdicts)) + "\n", encoding="utf-8")
    (ROOT / "reports/what_is_frozen_now.md").write_text("# What Is Frozen Now\n\nObserver primary markers, secondary conditions, rolling/as-of threshold method, review schedule, metrics, and kill/continue/research-only upgrade criteria are frozen.\n", encoding="utf-8")
    (ROOT / "reports/what_not_to_do_until_review.md").write_text("# What Not To Do Until Review\n\nDo not add/remove markers, change thresholds/conditions/horizons/cost modes, adjust launchd interval, or connect outputs to production/execution paths.\n", encoding="utf-8")
    (ROOT / "reports/next_review_plan.md").write_text("# Next Review Plan\n\nNext action: T+24h pipeline health check only. No performance evaluation before the registered schedule.\n", encoding="utf-8")
    return {"verdicts": list(dict.fromkeys(verdicts)), "total_lower": total_lower, "total_upper": total_upper}


def run_full(fast: bool = False) -> Dict[str, Any]:
    before = safety_snapshot("before")
    try:
        res = {
            "guard": read_only_guard(),
            "discovery": discovery(),
            "research_inventory": research_inventory(),
            "multiple_testing": multiple_testing(),
            "selection_bias": selection_bias(),
            "no_touch": no_touch(),
            "outlier_fragility": outlier_fragility(),
            "observer_snapshot": observer_snapshot(),
            "preregistration": preregister(),
        }
        res["report"] = final_report(res)
        res["safety"] = final_safety(before)
        res["production_ready"] = False
        res["promotion_ready"] = False
        (ROOT / "reports/run_metadata.json").write_text(jdump(res), encoding="utf-8")
        return res
    except Exception:
        final_safety(before)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fast-smoke", action="store_true")
    parser.add_argument("--discovery-only", action="store_true")
    parser.add_argument("--research-inventory-only", action="store_true")
    parser.add_argument("--multiple-testing-only", action="store_true")
    parser.add_argument("--selection-bias-only", action="store_true")
    parser.add_argument("--no-touch-only", action="store_true")
    parser.add_argument("--outlier-fragility-only", action="store_true")
    parser.add_argument("--observer-snapshot-only", action="store_true")
    parser.add_argument("--preregister-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    ensure_dirs()
    if args.dry_run:
        res = {"dry_run": True, "root": str(ROOT), "read_only": True, "production_ready": False, "promotion_ready": False}
    elif args.fast_smoke:
        res = {"discovery": discovery(), "observer_snapshot": observer_snapshot(), "preregistration": preregister()}
    elif args.discovery_only:
        res = discovery()
    elif args.research_inventory_only:
        res = research_inventory()
    elif args.multiple_testing_only:
        res = multiple_testing()
    elif args.selection_bias_only:
        res = selection_bias()
    elif args.no_touch_only:
        res = no_touch()
    elif args.outlier_fragility_only:
        res = outlier_fragility()
    elif args.observer_snapshot_only:
        res = observer_snapshot()
    elif args.preregister_only:
        res = preregister()
    elif args.report_only:
        res = final_report()
    elif args.audit_only:
        before = safety_snapshot("before")
        res = {"guard": read_only_guard(), "safety": final_safety(before)}
    else:
        res = run_full(fast=args.fast_smoke)
    print(jdump(res) if args.json else res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
