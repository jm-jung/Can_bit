#!/usr/bin/env python3
"""
Phase C5 채택 검증: ts60 스팟체크 1회 + ts72 vs ts60(및 ts60_spot) 비교 + PASS/FLAG 판정.
로그: data/diagnostics/phase_runs/phase_c5_spotcheck_YYYYMMDD_HHMMSS.log
요약: data/diagnostics/phase_c5_spotcheck_summary.md
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


def load_json(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
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
        r = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        out = (r.stdout or "") + (r.stderr or "")
        log_lines.append(out)
        if r.returncode != 0:
            log_lines.append(f"[exit code {r.returncode}]")
        return r.returncode, out
    except Exception as e:
        log_lines.append(str(e))
        return -1, str(e)


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase C5 spotcheck: ts60_spot + compare + PASS/FLAG")
    ap.add_argument("--skip-spot-run", action="store_true", help="스팟체크 실행(2-4) 생략, 기존 JSON으로 비교/판정만")
    args = ap.parse_args()

    PHASE_RUNS.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = PHASE_RUNS / f"phase_c5_spotcheck_{ts}.log"
    summary_path = DIAG / "phase_c5_spotcheck_summary.md"
    log_lines: list[str] = []
    log_lines.append(f"Phase C5 spotcheck started at {ts}")
    log_lines.append(f"LOG: {log_path}")
    log_lines.append(f"SUMMARY: {summary_path}")

    def plog(msg: str = ""):
        log_lines.append(msg)
        print(msg)

    # (2-1) 기존 C5 파일 존재 확인
    base_path = find_by_run_id("phase_c5_base_ts72")
    ts60_path = find_by_run_id("phase_c5_ts60")
    plog()
    plog("[2-1] 기존 C5 JSON 확인")
    plog(f"  phase_c5_base_ts72: {base_path.name if base_path else 'NOT FOUND'}")
    plog(f"  phase_c5_ts60:     {ts60_path.name if ts60_path else 'NOT FOUND'}")

    # (2-2)(2-3) 없으면 재실행
    if not base_path or not base_path.exists():
        plog("[2-2] baseline(ts72) 실행 중...")
        code, _ = run_cmd(
            [sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation"]
            + COMMON_ARGS + ["--time-stop", "on", "--time-stop-bars", "72", "--run-id", "phase_c5_base_ts72"],
            log_lines, PROJECT_ROOT,
        )
        if code != 0:
            plog(f"[2-2] baseline 실행 실패 exit={code}")
        base_path = find_by_run_id("phase_c5_base_ts72")
    if not ts60_path or not ts60_path.exists():
        plog("[2-3] best(ts60) 실행 중...")
        code, _ = run_cmd(
            [sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation"]
            + COMMON_ARGS + ["--time-stop", "on", "--time-stop-bars", "60", "--run-id", "phase_c5_ts60"],
            log_lines, PROJECT_ROOT,
        )
        if code != 0:
            plog(f"[2-3] ts60 실행 실패 exit={code}")
        ts60_path = find_by_run_id("phase_c5_ts60")

    # (2-4) 스팟체크 ts60_spot 무조건 실행 (--skip-spot-run이면 생략)
    if not args.skip_spot_run:
        plog()
        plog("[2-4] 스팟체크 phase_c5_ts60_spot 실행 (무조건 1회)")
        code_spot, out_spot = run_cmd(
            [sys.executable, "-X", "faulthandler", "-m", "scripts.run_tcn_candidate_validation"]
            + COMMON_ARGS + ["--time-stop", "on", "--time-stop-bars", "60", "--run-id", "phase_c5_ts60_spot"],
            log_lines, PROJECT_ROOT,
        )
        if code_spot != 0:
            plog(f"[2-4] ts60_spot 실행 실패 exit={code_spot}. 계속 진행하여 비교/판정만 수행.")
    else:
        plog()
        plog("[2-4] 스팟체크 실행 생략 (--skip-spot-run). 기존 phase_c5_ts60_spot JSON 있으면 비교에 포함.")

    # (2-5) 비교표 (spot 있으면 ts60+spot, 없으면 ts60만)
    plog()
    other_ids = "phase_c5_ts60,phase_c5_ts60_spot" if (find_by_run_id("phase_c5_ts60_spot") and find_by_run_id("phase_c5_ts60_spot").exists()) else "phase_c5_ts60"
    plog(f"[2-5] 비교표 (baseline=ts72 vs {other_ids})")
    code_cmp, out_cmp = run_cmd(
        [sys.executable, "-m", "scripts.compare_baseline_vs_time_stop_sweep",
         "--baseline-run-id", "phase_c5_base_ts72",
         "--other-run-ids", other_ids],
        log_lines, PROJECT_ROOT,
    )
    print(out_cmp)
    if code_cmp != 0:
        plog(f"[2-5] compare 실패 exit={code_cmp}")

    # JSON 로드 (판정용)
    base_path = find_by_run_id("phase_c5_base_ts72")
    ts60_path = find_by_run_id("phase_c5_ts60")
    spot_path = find_by_run_id("phase_c5_ts60_spot")
    d_base = load_json(base_path) if base_path else None
    d_ts60 = load_json(ts60_path) if ts60_path else None
    d_spot = load_json(spot_path) if spot_path else None

    def row365(d: dict | None) -> dict | None:
        return get_row(d.get("results", []), 365) if d else None

    r72 = row365(d_base)
    r60 = row365(d_ts60)
    r_spot = row365(d_spot)

    # [3] PASS/FLAG 판정
    plog()
    plog("[3] 자동 판정 (PASS/FLAG)")
    failures: list[str] = []

    # A) ts60 채택 유지 조건
    if r72 and r60:
        co72 = r72.get("cost_on_return")
        co60 = r60.get("cost_on_return")
        if co72 is not None and co60 is not None and co60 < co72 - 0.0005:
            failures.append("A) 365d cost_on(ts60) < cost_on(ts72) - 0.0005")
        mdd72 = r72.get("max_drawdown")
        mdd60 = r60.get("max_drawdown")
        if mdd72 is not None and mdd60 is not None and mdd60 > mdd72 + 0.005:
            failures.append("A) 365d MDD(ts60) > MDD(ts72) + 0.005")
        tr72 = r72.get("trades")
        tr60 = r60.get("trades")
        if tr72 is not None and tr60 is not None and tr60 < tr72 * 0.95:
            failures.append("A) 365d trades(ts60) < trades(ts72) * 0.95")
        coff72 = r72.get("cost_off_return")
        coff60 = r60.get("cost_off_return")
        if coff72 is not None and coff60 is not None and coff60 > coff72 + 0.005:
            failures.append("A) 365d cost_off(ts60) > cost_off(ts72) + 0.005")
    else:
        failures.append("A) ts72 또는 ts60 365d row 없음")

    # B) 스팟체크 일치성
    if r60 and r_spot:
        co60 = r60.get("cost_on_return")
        co_spot = r_spot.get("cost_on_return")
        if co60 is not None and co_spot is not None and abs(co_spot - co60) > 0.003:
            failures.append("B) |cost_on(ts60_spot) - cost_on(ts60)| > 0.003")
        mdd60 = r60.get("max_drawdown")
        mdd_spot = r_spot.get("max_drawdown")
        if mdd60 is not None and mdd_spot is not None and abs(mdd_spot - mdd60) > 0.005:
            failures.append("B) |MDD(ts60_spot) - MDD(ts60)| > 0.005")
        tc60 = r60.get("time_stop_exit_count")
        tc_spot = r_spot.get("time_stop_exit_count")
        if tc60 is not None and tc_spot is not None and abs(tc_spot - tc60) > 15:
            failures.append("B) |time_stop_exit_count(spot) - time_stop_exit_count(ts60)| > 15")
    else:
        if not d_spot or not r_spot:
            failures.append("B) ts60_spot JSON 또는 365d row 없음 (스팟 실행 실패 가능)")

    passed = len(failures) == 0
    if passed:
        plog("판정: PASS (ts60 채택 유지)")
    else:
        plog("판정: FLAG")
        for f in failures:
            plog(f"  - {f}")

    # [4] summary.md 작성
    summary_lines = [
        "# Phase C5 spotcheck 요약",
        "",
        "## 사용한 run_id / JSON",
        f"- baseline: phase_c5_base_ts72  → {base_path.name if base_path else 'N/A'}",
        f"- best:    phase_c5_ts60       → {ts60_path.name if ts60_path else 'N/A'}",
        f"- spot:    phase_c5_ts60_spot  → {spot_path.name if spot_path else 'N/A'}",
        "",
        "## 30d / 365d 비교표 (cost_on, cost_off, MDD, trades)",
        "```",
    ]
    summary_lines.append(out_cmp if out_cmp.strip() else "(compare 출력 없음)")
    summary_lines.append("```")
    summary_lines.append("")
    summary_lines.append("## Time-stop stats (365d)")
    if r72:
        summary_lines.append(f"- phase_c5_base_ts72: time_stop_bars=72, time_stop_exit_count={r72.get('time_stop_exit_count')}, pct={r72.get('pct_time_stop_exits')}")
    if r60:
        summary_lines.append(f"- phase_c5_ts60:      time_stop_bars=60, time_stop_exit_count={r60.get('time_stop_exit_count')}, pct={r60.get('pct_time_stop_exits')}")
    if r_spot:
        summary_lines.append(f"- phase_c5_ts60_spot: time_stop_bars=60, time_stop_exit_count={r_spot.get('time_stop_exit_count')}, pct={r_spot.get('pct_time_stop_exits')}")
    summary_lines.extend([
        "",
        "## 최종 판정",
        "**PASS (ts60 유지)**" if passed else "**FLAG**",
        "",
    ])
    if failures:
        summary_lines.append("사유:")
        for f in failures:
            summary_lines.append(f"- {f}")
        summary_lines.append("")
    summary_lines.append(f"- 로그: `{log_path.name}`")
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    plog(f"Summary written: {summary_path}")

    # 로그 파일 저장
    log_path.write_text("\n".join(log_lines), encoding="utf-8")
    plog(f"Log written: {log_path}")

    # FINAL 한 줄
    if passed:
        plog("FINAL: PASS (ts60 유지)")
    else:
        plog("FINAL: FLAG (사유: " + "; ".join(failures) + ")")

    return 0


if __name__ == "__main__":
    sys.exit(main())
