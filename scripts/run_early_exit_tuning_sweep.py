#!/usr/bin/env python3
"""
Early Exit 파라미터 최소 스윕: Baseline 1회 + 5개 early_exit 조합 실행 후
compare 캡처 및 early_exit_tuning_summary.md 생성.

사용법:
  # 환경 변수 설정 후 실행 (세그폴트 방지)
  export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
  python -m scripts.run_early_exit_tuning_sweep

  # 이미 6개 JSON이 있으면 요약만 생성
  python -m scripts.run_early_exit_tuning_sweep --summary-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"

# 고정 run_id (파일명으로 어떤 run인지 식별)
BASELINE_RUN_ID = "tuning_baseline_early_exit_off"
EE_RUN_IDS = [
    ("tuning_ee_l12_p055_k8", "4-1 Base (lookback=12, p_floor=0.55, bad_k=8)"),
    ("tuning_ee_l12_p055_k10", "4-2 bad_k=10"),
    ("tuning_ee_l12_p055_k6", "4-3 bad_k=6"),
    ("tuning_ee_l12_p056_k8", "4-4 p_floor=0.56"),
    ("tuning_ee_l8_p055_k8", "4-5 lookback=8"),
]

VALIDATION_CMD_BASE = [
    sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation",
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-max-proba", "0.58", "--max-entropy", "1.35", "--min-hold", "48", "--cooldown", "24",
    "--regime-filter", "off",
    "--days-list", "30,365",
]


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


def run_baseline() -> bool:
    cmd = VALIDATION_CMD_BASE + ["--early-exit", "off", "--run-id", BASELINE_RUN_ID]
    print("[RUN] Baseline (early_exit OFF)", flush=True)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy())
    return r.returncode == 0


def run_early_exit(lookback: int, p_floor: float, bad_k: int, run_id: str) -> bool:
    cmd = VALIDATION_CMD_BASE + [
        "--early-exit", "on",
        "--early-exit-lookback", str(lookback),
        "--early-exit-p-floor", str(p_floor),
        "--early-exit-bad-k", str(bad_k),
        "--run-id", run_id,
    ]
    print(f"[RUN] Early exit lookback={lookback} p_floor={p_floor} bad_k={bad_k} run_id={run_id}", flush=True)
    r = subprocess.run(cmd, cwd=PROJECT_ROOT, env=os.environ.copy())
    return r.returncode == 0


def run_compare_and_capture() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "scripts.compare_baseline_vs_early_exit"],
        cwd=PROJECT_ROOT, capture_output=True, text=True, env=os.environ.copy(),
    )
    return (r.stdout or "") + (r.stderr or "")


def build_table_and_chosen(
    baseline_path: Path,
    ee_paths: list[Path],
    ee_labels: list[str],
) -> tuple[list[dict], str | None, str]:
    """Returns (rows for table, chosen_run_id or None, reason text)."""
    base = load_json(baseline_path)
    if not base:
        return [], None, "Baseline JSON load failed"
    r_base = base.get("results", [])
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)
    cost_b365 = row_b365.get("cost_on_return") if row_b365 else None
    mdd_b365 = row_b365.get("max_drawdown") if row_b365 else None
    trades_b365 = row_b365.get("trades") if row_b365 else None
    cost_b30 = row_b30.get("cost_on_return") if row_b30 else None

    rows = []
    candidates = []
    for path, label in zip(ee_paths, ee_labels):
        d = load_json(path)
        if not d:
            rows.append({"label": label, "file": path.name, "error": "load_failed"})
            continue
        r = d.get("results", [])
        row30 = get_row(r, 30)
        row365 = get_row(r, 365)
        co30 = row30.get("cost_on_return") if row30 else None
        co365 = row365.get("cost_on_return") if row365 else None
        mdd365 = row365.get("max_drawdown") if row365 else None
        tr365 = row365.get("trades") if row365 else None
        ee_count = row365.get("early_exit_count") if row365 else None
        ee_rate = row365.get("early_exit_rate") if row365 else None
        lb = row365.get("early_exit_lookback") if row365 else None
        pf = row365.get("early_exit_p_floor") if row365 else None
        bk = row365.get("early_exit_bad_k") if row365 else None
        rows.append({
            "label": label,
            "file": path.name,
            "cost_on_30": co30,
            "cost_on_365": co365,
            "cost_off_30": row30.get("cost_off_return") if row30 else None,
            "cost_off_365": row365.get("cost_off_return") if row365 else None,
            "mdd_30": row30.get("max_drawdown") if row30 else None,
            "mdd_365": mdd365,
            "trades_30": row30.get("trades") if row30 else None,
            "trades_365": tr365,
            "early_exit_count": ee_count,
            "early_exit_rate": ee_rate,
            "lookback": lb,
            "p_floor": pf,
            "bad_k": bk,
        })
        if co365 is None:
            continue
        # 탈락 조건
        if mdd_b365 is not None and mdd365 is not None and (mdd365 - mdd_b365) > 0.005:
            continue
        if trades_b365 and tr365 is not None and trades_b365 > 0:
            if (trades_b365 - tr365) / trades_b365 > 0.03:
                continue
        if cost_b30 is not None and co30 is not None and cost_b30 != 0:
            if (co30 - cost_b30) / abs(cost_b30) < -0.20:
                continue
        candidates.append((path.stem, co365, mdd365, tr365, label))

    if not candidates:
        chosen_id = None
        reason = "No candidate passed filters (MDD +0.005, trades -3%, 30d cost_on -20%)."
    else:
        # 1순위: 365d cost_on 최대
        best = max(candidates, key=lambda x: (x[1] if x[1] is not None else -1e9))
        chosen_id = best[0].replace(PREFIX + "_", "")
        reason = f"365d cost_on 최대={best[1]:.4f} (label={best[4]}, MDD={best[2]}, trades={best[3]})"

    return rows, chosen_id, reason


def write_summary(
    baseline_path: Path,
    ee_paths: list[Path],
    ee_labels: list[str],
    compare_log: str,
) -> None:
    base = load_json(baseline_path)
    r_base = base.get("results", []) if base else []
    row_b30 = get_row(r_base, 30)
    row_b365 = get_row(r_base, 365)

    rows, chosen_id, reason = build_table_and_chosen(baseline_path, ee_paths, ee_labels)
    out = DIAG / "early_exit_tuning_summary.md"
    lines = [
        "# Early Exit 파라미터 튜닝 요약",
        "",
        "## 1) Baseline JSON",
        f"- `{baseline_path.name}`",
        "",
        "## 2) 5개 Early Exit run JSON",
    ]
    for path, label in zip(ee_paths, ee_labels):
        lines.append(f"- **{label}**: `{path.name}`")
    lines.extend([
        "",
        "## 3) 한 눈에 보는 표 (30d / 365d + early_exit stats)",
        "",
        "| run | cost_on_30d | cost_on_365d | cost_off_30d | cost_off_365d | mdd_30d | mdd_365d | trades_30d | trades_365d | early_exit_count | early_exit_rate | lookback | p_floor | bad_k |",
        "|-----|-------------|---------------|--------------|---------------|---------|----------|------------|-------------|------------------|-----------------|----------|---------|-------|",
    ])
    # Baseline row
    if row_b30 is not None or row_b365 is not None:
        co30 = f"{row_b30['cost_on_return']:.4f}" if row_b30 and row_b30.get("cost_on_return") is not None else "-"
        co365 = f"{row_b365['cost_on_return']:.4f}" if row_b365 and row_b365.get("cost_on_return") is not None else "-"
        cf30 = f"{row_b30['cost_off_return']:.4f}" if row_b30 and row_b30.get("cost_off_return") is not None else "-"
        cf365 = f"{row_b365['cost_off_return']:.4f}" if row_b365 and row_b365.get("cost_off_return") is not None else "-"
        m30 = f"{row_b30['max_drawdown']:.4f}" if row_b30 and row_b30.get("max_drawdown") is not None else "-"
        m365 = f"{row_b365['max_drawdown']:.4f}" if row_b365 and row_b365.get("max_drawdown") is not None else "-"
        t30 = row_b30.get("trades") if row_b30 else "-"
        t365 = row_b365.get("trades") if row_b365 else "-"
        lines.append(f"| **Baseline (early_exit OFF)** | {co30} | {co365} | {cf30} | {cf365} | {m30} | {m365} | {t30} | {t365} | - | - | - | - | - |")
    for r in rows:
        if r.get("error"):
            lines.append(f"| {r['label']} | - | - | - | - | - | - | - | - | - | - | - | - | - |")
            continue
        co30 = f"{r['cost_on_30']:.4f}" if r.get("cost_on_30") is not None else "-"
        co365 = f"{r['cost_on_365']:.4f}" if r.get("cost_on_365") is not None else "-"
        cf30 = f"{r['cost_off_30']:.4f}" if r.get("cost_off_30") is not None else "-"
        cf365 = f"{r['cost_off_365']:.4f}" if r.get("cost_off_365") is not None else "-"
        m30 = f"{r['mdd_30']:.4f}" if r.get("mdd_30") is not None else "-"
        m365 = f"{r['mdd_365']:.4f}" if r.get("mdd_365") is not None else "-"
        t30 = r["trades_30"] if r.get("trades_30") is not None else "-"
        t365 = r["trades_365"] if r.get("trades_365") is not None else "-"
        eec = r.get("early_exit_count") if r.get("early_exit_count") is not None else "-"
        eer = f"{r['early_exit_rate']:.4f}" if r.get("early_exit_rate") is not None else "-"
        lb = r.get("lookback") if r.get("lookback") is not None else "-"
        pf = r.get("p_floor") if r.get("p_floor") is not None else "-"
        bk = r.get("bad_k") if r.get("bad_k") is not None else "-"
        lines.append(f"| {r['label']} | {co30} | {co365} | {cf30} | {cf365} | {m30} | {m365} | {t30} | {t365} | {eec} | {eer} | {lb} | {pf} | {bk} |")
    lines.extend([
        "",
        "## 4) 최종 CHOSEN 파라미터 및 이유",
        "",
        f"- **CHOSEN run_id**: `{chosen_id or 'None (탈락 후보 없음)'}`",
        f"- **선정 근거**: {reason}",
    ])
    def _is_chosen(r: dict) -> bool:
        if r.get("error") or r.get("cost_on_365") is None or not chosen_id:
            return False
        return chosen_id in r.get("file", "") or chosen_id in r.get("label", "")
    chosen_row = next((r for r in rows if _is_chosen(r)), None)
    if chosen_row and chosen_id:
        lines.extend([
        f"- **채택 파라미터**: lookback={chosen_row.get('lookback')}, p_floor={chosen_row.get('p_floor')}, bad_k={chosen_row.get('bad_k')}",
        "",
        ])
    else:
        lines.append("")
    lines.extend([
        "",
        "### 채택 기준 적용",
        "- 1순위: 365d cost_on_return 최대",
        "- 2순위: 365d max_drawdown baseline 대비 +0.005 초과 시 탈락",
        "- 3순위: trades 감소율 3% 초과 시 탈락",
        "- 4순위: 30d cost_on baseline 대비 20% 이상 악화 시 탈락",
        "",
        "## 5) compare 스크립트 출력 요약",
        "",
        "```",
        compare_log.strip()[:4000] if compare_log else "(없음)",
        "```",
        "",
        "## 6) 최종 선정 3줄 요약",
        "",
        f"- CHOSEN: `{chosen_id or 'None'}` — {reason}",
        "- 채택 기준: 365d cost_on 최대 우선, MDD +0.005 초과·trades -3% 초과·30d cost_on -20% 악화 시 탈락.",
        "- Early Exit 파라미터만 변경하여 entry/regime/scaling은 고정.",
        "",
    ])
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true", help="6개 JSON으로 요약만 생성 (실행 생략)")
    ap.add_argument("--skip-run", action="store_true", help="실행은 하되 validation run은 스킵 (기존 JSON으로 요약만)")
    args = ap.parse_args()

    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    baseline_path = DIAG / f"{PREFIX}_{BASELINE_RUN_ID}.json"
    ee_paths = [DIAG / f"{PREFIX}_{rid}.json" for rid, _ in EE_RUN_IDS]
    ee_labels = [label for _, label in EE_RUN_IDS]

    # summary-only이고 고정 run_id 파일이 없으면, 기존 JSON에서 meta로 5개 조합 찾기
    if args.summary_only:
        if not baseline_path.exists():
            candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            for c in candidates:
                d = load_json(c)
                if d and (d.get("meta", {}).get("early_exit") in ("off", None) or d.get("meta", {}).get("early_exit_enabled") in (False, None)):
                    baseline_path = c
                    break
        for i, (rid, label) in enumerate(EE_RUN_IDS):
            if not ee_paths[i].exists():
                # Param set for this slot
                params = [(12, 0.55, 8), (12, 0.55, 10), (12, 0.55, 6), (12, 0.56, 8), (8, 0.55, 8)][i]
                lb, pf, bk = params
                candidates = sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
                for c in candidates:
                    d = load_json(c)
                    if not d:
                        continue
                    m = d.get("meta", {})
                    if m.get("early_exit") != "on" and m.get("early_exit_enabled") is not True:
                        continue
                    if m.get("early_exit_lookback") == lb and m.get("early_exit_p_floor") == pf and m.get("early_exit_bad_k") == bk:
                        ee_paths[i] = c
                        break

    if not args.summary_only and not args.skip_run:
        if not run_baseline():
            print("Baseline run failed (exit 139 등 확인)", file=sys.stderr)
            return 1
        params = [
            (12, 0.55, 8),
            (12, 0.55, 10),
            (12, 0.55, 6),
            (12, 0.56, 8),
            (8, 0.55, 8),
        ]
        for (run_id, _), (lb, pf, bk) in zip(EE_RUN_IDS, params):
            if not run_early_exit(lb, pf, bk, run_id):
                print(f"Early exit run failed: {run_id}", file=sys.stderr)
            compare_out = run_compare_and_capture()
            with open(DIAG / "compare_log_early_exit_tuning.txt", "a", encoding="utf-8") as f:
                f.write(f"\n--- {run_id} ---\n")
                f.write(compare_out)

    compare_log = ""
    clog = DIAG / "compare_log_early_exit_tuning.txt"
    if clog.exists():
        compare_log = clog.read_text(encoding="utf-8")

    if not baseline_path.exists():
        print(f"Baseline not found: {baseline_path}. Run without --summary-only first.", file=sys.stderr)
        return 1
    missing = [p for p in ee_paths if not p.exists()]
    if missing:
        print(f"Some early_exit JSONs missing: {[p.name for p in missing]}. Proceeding with available.", file=sys.stderr)
    available = [(p, EE_RUN_IDS[i][1]) for i, p in enumerate(ee_paths) if p.exists()]
    ee_paths = [a[0] for a in available]
    ee_labels = [a[1] for a in available]

    write_summary(baseline_path, ee_paths, ee_labels, compare_log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
