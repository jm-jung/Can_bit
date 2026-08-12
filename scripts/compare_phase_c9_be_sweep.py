#!/usr/bin/env python3
"""
Phase C9: Break-even stop sweep 비교.
365d pinned 기준: ADOPT_CANDIDATE (cost_on >= baseline+0.001, MDD <= baseline+0.005),
REJECT (cost_on <= baseline-0.003 or MDD >= baseline+0.005). BEST = cost_on 최대.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"


def load_json(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Load error {p}: {e}", file=sys.stderr)
        return None


def get_row(results: list, days: int) -> dict | None:
    for r in results:
        if r.get("days") == days:
            return r
    return None


def find_by_run_id(run_id: str) -> Path | None:
    candidates = sorted(
        DIAG.glob(f"{PREFIX}_*.json"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        if run_id in c.stem:
            d = load_json(c)
            if d and d.get("meta", {}).get("run_id") == run_id:
                return c
    for c in candidates:
        if run_id in c.stem:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-run-id", type=str, default="phase_c9_base")
    ap.add_argument("--other-run-ids", type=str, default="phase_c9_be002,phase_c9_be003,phase_c9_be004")
    ap.add_argument("--summary-md", type=str, default="phase_c9_be_sweep_summary.md")
    args = ap.parse_args()

    base_path = find_by_run_id(args.baseline_run_id)
    if not base_path or not base_path.exists():
        print(f"Baseline JSON not found (run_id={args.baseline_run_id}).", file=sys.stderr)
        return 1

    other_ids = [x.strip() for x in args.other_run_ids.split(",") if x.strip()]
    other_data = []
    for rid in other_ids:
        p = find_by_run_id(rid)
        if not p or not p.exists():
            print(f"Other run JSON not found (run_id={rid}).", file=sys.stderr)
            return 1
        other_data.append(load_json(p))

    d_base = load_json(base_path)
    if not d_base:
        return 1

    meta_base = d_base.get("meta", {})
    r_base = d_base.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    if not row_b365:
        print("Baseline has no 365d result.", file=sys.stderr)
        return 1

    cost_b365 = row_b365.get("cost_on_return")
    mdd_b365 = row_b365.get("max_drawdown")
    end_date = meta_base.get("end_date")

    print("=" * 72)
    print("Phase C9 Break-even sweep (pinned)" + (f" end_date={end_date}" if end_date else ""))
    print("=" * 72)
    print(f"baseline: {args.baseline_run_id}")
    for rid in other_ids:
        print(f"  other:  {rid}")
    print()

    w = 12
    cols = ["30d_base", "365d_base"] + [f"30d_{rid}" for rid in other_ids] + [f"365d_{rid}" for rid in other_ids]
    print(f"{'metric':<22} " + " ".join(f"{c:>{w}}" for c in cols))
    print("-" * (22 + (w + 1) * len(cols)))

    for key, fmt in [
        ("cost_on_return", ".4f"),
        ("cost_off_return", ".4f"),
        ("max_drawdown", ".4f"),
        ("trades", "d"),
        ("time_stop_exit_count", "d"),
        ("early_exit_count", "d"),
        ("break_even_exit_count", "d"),
    ]:
        vals = [row_b30.get(key) if row_b30 else None, row_b365.get(key)]
        for d in other_data:
            r30 = get_row(d.get("results", []), 30)
            vals.append(r30.get(key) if r30 else None)
        for d in other_data:
            r365 = get_row(d.get("results", []), 365)
            vals.append(r365.get(key) if r365 else None)
        if key in ("time_stop_exit_count", "early_exit_count", "break_even_exit_count"):
            parts = [f"{int(v)}" if v is not None else "N/A" for v in vals]
        else:
            parts = [f"{v:{fmt}}" if v is not None else "N/A" for v in vals]
        print(f"{key:<22} " + " ".join(f"{p:>{w}}" for p in parts))

    verdicts = {}
    for i, rid in enumerate(other_ids):
        r365 = get_row(other_data[i].get("results", []), 365)
        if not r365:
            verdicts[rid] = "REJECT"
            continue
        cost_o = r365.get("cost_on_return")
        mdd_o = r365.get("max_drawdown")
        if cost_o is None or cost_b365 is None:
            verdicts[rid] = "REJECT"
            continue
        if cost_o <= cost_b365 - 0.003 or (mdd_o is not None and mdd_b365 is not None and mdd_o > mdd_b365 + 0.005):
            verdicts[rid] = "REJECT"
        elif cost_o >= cost_b365 + 0.001 and (mdd_o is None or mdd_b365 is None or mdd_o <= mdd_b365 + 0.005):
            verdicts[rid] = "ADOPT_CANDIDATE"
        else:
            verdicts[rid] = "NO_IMPROVE"

    print()
    for rid in other_ids:
        print(f"판정 [{rid}]: {verdicts.get(rid, 'N/A')}")

    candidates = [(rid, get_row(other_data[i].get("results", []), 365).get("cost_on_return"))
                  for i, rid in enumerate(other_ids) if verdicts.get(rid) == "ADOPT_CANDIDATE"]
    if not candidates:
        best_run_id = args.baseline_run_id
    else:
        candidates.sort(key=lambda x: (-(x[1] if x[1] is not None else -1e9),))
        best_run_id = candidates[0][0]

    print()
    print(f"BEST run_id: {best_run_id}")

    summary_path = DIAG / args.summary_md
    lines = [
        "# Phase C9 Break-even sweep 요약",
        "",
        f"- **pinned end_date**: {end_date or 'N/A'}",
        f"- **baseline**: {args.baseline_run_id}",
        f"- **others**: {', '.join(other_ids)}",
        f"- **BEST**: {best_run_id}",
        "",
        "## 판정",
    ]
    for rid in other_ids:
        lines.append(f"- {rid}: {verdicts.get(rid, 'N/A')}")
    lines.extend(["", "## 365d cost_on / MDD (참고)", ""])
    lines.append(f"- baseline: cost_on={cost_b365}, MDD={mdd_b365}")
    for i, rid in enumerate(other_ids):
        r365 = get_row(other_data[i].get("results", []), 365)
        if r365:
            lines.append(f"- {rid}: cost_on={r365.get('cost_on_return')}, MDD={r365.get('max_drawdown')}")
    try:
        summary_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSummary: {summary_path}")
    except Exception as e:
        print(f"Warning: could not write summary: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
