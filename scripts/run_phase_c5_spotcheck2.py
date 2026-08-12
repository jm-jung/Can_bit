#!/usr/bin/env python3
"""
Phase C5 FLAG 결정 규칙: ts60 스팟 3개(spot + spot2 + spot3) + cost_on range/mean + PASS/FLAG 판정.
- spot2, spot3 2회 추가 실행
- cost_on_list = [ts60, spot, spot2, spot3] 365d cost_on → range, mean
- PASS: A) range<=0.005, B) mean>=ts72-0.001, C) mean_MDD<=ts72+0.005, D) mean_trades>=ts72*0.95
로그: data/diagnostics/phase_runs/phase_c5_spotcheck2_YYYYMMDD_HHMMSS.log
요약: data/diagnostics/phase_c5_spotcheck_summary.md 갱신
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"
PHASE_RUNS = DIAG / "phase_runs"
PREFIX = "tcn_candidate_validation_BTCUSDT_5m_h15_t0p004"

COMMON_ARGS = [
    "--id", "h15_t0p004", "--symbol", "BTCUSDT", "--timeframe", "5m",
    "--min-max-proba", "0.58", "--max-entropy", "1.35", "--min-hold", "48", "--cooldown", "24",
    "--regime-filter", "off", "--position-scaling", "off",
    "--early-exit", "on", "--early-exit-lookback", "12", "--early-exit-p-floor", "0.55", "--early-exit-bad-k", "8",
    "--days-list", "30,365",
]

TS60_RUN_IDS = ["phase_c5_ts60", "phase_c5_ts60_spot", "phase_c5_ts60_spot2", "phase_c5_ts60_spot3"]


def load_json(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_row(results: list, days: int) -> dict | None:
    for r in results:
        if r.get("days") == days:
            return r
    return None


def find_by_run_id(run_id: str) -> Path | None:
    for c in sorted(DIAG.glob(f"{PREFIX}_*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if run_id in c.stem:
            d = load_json(c)
            if d and d.get("meta", {}).get("run_id") == run_id:
                return c
    for c in DIAG.glob(f"{PREFIX}_*.json"):
        if run_id in c.stem:
            return c
    return None


def run_cmd(cmd: list[str], log_lines: list[str], cwd: Path) -> tuple[int, str]:
    log_lines.append("")
    log_lines.append("$ " + " ".join(cmd))
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=3600)
        out = (r.stdout or "") + (r.stderr or "")
        log_lines.append(out)
        if r.returncode != 0:
            log_lines.append(f"[exit code {r.returncode}]")
        return r.returncode, out
    except Exception as e:
        log_lines.append(str(e))
        return -1, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase C5 spotcheck2: spot2+spot3 실행, cost_on range/mean, PASS/FLAG")
    ap.add_argument("--skip-extra-spots", action="store_true", help="spot2/spot3 실행 생략, 기존 JSON으로 비교/판정만")
    args = ap.parse_args()

    PHASE_RUNS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PHASE_RUNS / f"phase_c5_spotcheck2_{ts}.log"
    summary_path = DIAG / "phase_c5_spotcheck_summary.md"
    log_lines: list[str] = []

    def plog(msg: str = ""):
        log_lines.append(msg)
        print(msg)

    plog(f"Phase C5 spotcheck2 (FLAG 결정 규칙) started at {ts}")
    plog(f"LOG: {log_path}")
    plog(f"SUMMARY: {summary_path}")

    # [2] spot2, spot3 추가 실행 (무조건, --skip-extra-spots면 생략)
    if not args.skip_extra_spots:
        plog()
        plog("[2] 추가 스팟체크 2회 실행 (phase_c5_ts60_spot2, phase_c5_ts60_spot3)")
        for rid in ["phase_c5_ts60_spot2", "phase_c5_ts60_spot3"]:
            plog(f"  실행: {rid}")
            code, _ = run_cmd(
                [sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation"]
                + COMMON_ARGS + ["--time-stop", "on", "--time-stop-bars", "60", "--run-id", rid],
                log_lines, PROJECT_ROOT,
            )
            if code != 0:
                plog(f"  {rid} exit={code}")
    else:
        plog()
        plog("[2] 추가 스팟체크 실행 생략 (--skip-extra-spots). 기존 spot2/spot3 JSON 사용.")

    # [3] 비교표 (존재하는 run만 전달)
    plog()
    existing_others = [rid for rid in TS60_RUN_IDS if find_by_run_id(rid) and find_by_run_id(rid).exists()]
    plog(f"[3] 비교표 (baseline=ts72 vs {existing_others})")
    if not existing_others:
        out_cmp = "(ts60/spot JSON 없음)"
        code_cmp = 1
    else:
        other_ids = ",".join(existing_others)
        code_cmp, out_cmp = run_cmd(
            [sys.executable, "-m", "scripts.compare_baseline_vs_time_stop_sweep",
             "--baseline-run-id", "phase_c5_base_ts72",
             "--other-run-ids", other_ids],
            log_lines, PROJECT_ROOT,
        )
    print(out_cmp)
    if code_cmp != 0:
        plog(f"[3] compare 실패 exit={code_cmp} (일부 run JSON 없을 수 있음)")

    # JSON 로드
    base_path = find_by_run_id("phase_c5_base_ts72")
    data_by_rid: dict[str, dict] = {}
    for rid in ["phase_c5_base_ts72"] + TS60_RUN_IDS:
        p = find_by_run_id(rid)
        if p:
            data_by_rid[rid] = load_json(p) or {}
        else:
            data_by_rid[rid] = {}

    def row365(rid: str) -> dict | None:
        return get_row(data_by_rid.get(rid, {}).get("results", []), 365)

    r72 = row365("phase_c5_base_ts72")
    cost_on_list: list[float] = []
    mdd_list: list[float] = []
    trades_list: list[int] = []
    for rid in TS60_RUN_IDS:
        r = row365(rid)
        if r:
            co = r.get("cost_on_return")
            if co is not None:
                cost_on_list.append(co)
            md = r.get("max_drawdown")
            if md is not None:
                mdd_list.append(md)
            tr = r.get("trades")
            if tr is not None:
                trades_list.append(tr)

    # [4] 판정 규칙
    plog()
    plog("[4] 자동 판정 (cost_on range/mean, PASS/FLAG)")
    failures: list[str] = []

    if not r72:
        failures.append("baseline(ts72) 365d row 없음")
    else:
        ts72_cost_on = r72.get("cost_on_return")
        ts72_mdd = r72.get("max_drawdown")
        ts72_trades = r72.get("trades")

        if len(cost_on_list) < 2:
            failures.append("ts60 측 run 2개 미만 (ts60+spot1~3 중 365d cost_on 확보 필요)")
        else:
            cost_on_range = max(cost_on_list) - min(cost_on_list)
            cost_on_mean = sum(cost_on_list) / len(cost_on_list)
            mean_mdd = sum(mdd_list) / len(mdd_list) if mdd_list else None
            mean_trades = sum(trades_list) / len(trades_list) if trades_list else None

            plog(f"  cost_on_list (365d): {cost_on_list}")
            plog(f"  range = max - min = {cost_on_range:.4f}")
            plog(f"  mean  = {cost_on_mean:.4f}")
            if mean_mdd is not None:
                plog(f"  mean_MDD = {mean_mdd:.4f}")
            if mean_trades is not None:
                plog(f"  mean_trades = {mean_trades:.1f}")
            plog(f"  ts72 cost_on={ts72_cost_on}, MDD={ts72_mdd}, trades={ts72_trades}")

            # A) range <= 0.005
            if cost_on_range > 0.005:
                failures.append(f"A) range({cost_on_range:.4f}) > 0.005 (스팟 변동 허용폭 초과)")
            # B) mean >= ts72_cost_on - 0.001
            if ts72_cost_on is not None and cost_on_mean < ts72_cost_on - 0.001:
                failures.append(f"B) mean({cost_on_mean:.4f}) < ts72({ts72_cost_on:.4f}) - 0.001")
            # C) mean_MDD <= ts72_MDD + 0.005
            if ts72_mdd is not None and mean_mdd is not None and mean_mdd > ts72_mdd + 0.005:
                failures.append(f"C) mean_MDD({mean_mdd:.4f}) > ts72_MDD({ts72_mdd:.4f}) + 0.005")
            # D) mean_trades >= ts72_trades * 0.95
            if ts72_trades is not None and mean_trades is not None and mean_trades < ts72_trades * 0.95:
                failures.append(f"D) mean_trades({mean_trades:.1f}) < ts72_trades({ts72_trades}) * 0.95")

    passed = len(failures) == 0
    plog()
    if passed:
        plog("판정: PASS (ts60 채택)")
    else:
        plog("판정: FLAG (보류)")
        for f in failures:
            plog(f"  - {f}")

    # [5] summary.md 갱신
    json_names = [f"  - {rid} → {(find_by_run_id(rid).name if find_by_run_id(rid) else 'N/A')}" for rid in ["phase_c5_base_ts72"] + TS60_RUN_IDS]
    cost_on_list_s = str(cost_on_list) if cost_on_list else "[]"
    range_s = f"{max(cost_on_list) - min(cost_on_list):.4f}" if len(cost_on_list) >= 2 else "N/A"
    mean_s = f"{sum(cost_on_list) / len(cost_on_list):.4f}" if cost_on_list else "N/A"

    summary_lines = [
        "# Phase C5 spotcheck 요약 (FLAG 결정 규칙)",
        "",
        "## 사용한 run_id / JSON",
        "- baseline: phase_c5_base_ts72",
    ]
    for rid in TS60_RUN_IDS:
        p = find_by_run_id(rid)
        summary_lines.append(f"- {rid}: {(p.name if p else 'N/A')}")
    summary_lines.extend([
        "",
        "## 30d / 365d 비교표 (cost_on, cost_off, MDD, trades)",
        "```",
        out_cmp.strip() if out_cmp else "(compare 출력 없음)",
        "```",
        "",
        "## ts60 측 cost_on (365d) — range / mean",
        f"- cost_on_list: {cost_on_list_s}",
        f"- range (max - min): {range_s}",
        f"- mean: {mean_s}",
        "",
        "## FINAL 판정",
        "**PASS (ts60 채택)**" if passed else "**FLAG (보류)**",
        "",
    ])
    if failures:
        summary_lines.append("실패 조건:")
        for f in failures:
            summary_lines.append(f"- {f}")
        summary_lines.append("")
    summary_lines.append(f"- 로그: `{log_path.name}`")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    plog(f"Summary written: {summary_path}")

    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    plog(f"Log written: {log_path}")

    if passed:
        plog("FINAL: PASS (ts60 채택)")
    else:
        plog("FINAL: FLAG (보류) — " + "; ".join(failures))

    return 0


if __name__ == "__main__":
    sys.exit(main())
