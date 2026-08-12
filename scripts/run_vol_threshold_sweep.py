#!/usr/bin/env python3
"""
vol_compress 레짐 필터의 vol_threshold만 스윕 (vol_window=48 고정).
days_list=30,365 로만 실행하여 시간 절약.

사용법 (프로젝트 루트에서 .venv 활성화 후):

  # 1) 4개 threshold 한 번에 실행 후 요약
  python -m scripts.run_vol_threshold_sweep

  # 2) 이미 돌린 결과만 vol_threshold별로 수집해 요약 (run_id 무관)
  python -m scripts.run_vol_threshold_sweep --summary-from-dir

  # 3) 수동 4회 실행 후 요약 (run_id로 생성된 파일만 사용)
  python -m scripts.run_vol_threshold_sweep --summary-only

수동 4회 실행 예 (각각 독립 실행):
  python -m scripts.run_tcn_candidate_validation --id h15_t0p004 --symbol BTCUSDT --timeframe 5m \\
    --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24 \\
    --regime-filter vol_compress --vol-window 48 --vol-threshold 0.0010 --days-list 30,365
  (0.0015 / 0.0020 / 0.0025 로 동일 옵션 반복 후 --summary-from-dir 실행)

산출물:
  - data/diagnostics/tcn_candidate_validation_*_<run_id>.json (및 .md)
  - 터미널에 30d/365d 요약 표 + 추천 threshold 출력
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIAG = PROJECT_ROOT / "data" / "diagnostics"

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
MODEL_ID = "h15_t0p004"
MIN_MAX_PROBA = 0.58
MAX_ENTROPY = 1.35
MIN_HOLD = 48
COOLDOWN = 24
VOL_WINDOW = 48
DAYS_LIST = "30,365"

THRESHOLDS = [0.0010, 0.0015, 0.0020, 0.0025]


def run_id_for_threshold(t: float) -> str:
    # 0.0010 -> vol_t0p0010, 0.0025 -> vol_t0p0025
    s = f"{t:.4f}".replace("0.", "t0p").replace(".", "p")
    return f"vol_{s}"


def run_sweep() -> list[Path]:
    """4회 validation 실행, 생성된 JSON 경로 목록 반환."""
    json_paths = []
    for th in THRESHOLDS:
        run_id = run_id_for_threshold(th)
        cmd = [
            sys.executable, "-m", "scripts.run_tcn_candidate_validation",
            "--id", MODEL_ID,
            "--symbol", SYMBOL,
            "--timeframe", TIMEFRAME,
            "--min-max-proba", str(MIN_MAX_PROBA),
            "--max-entropy", str(MAX_ENTROPY),
            "--min-hold", str(MIN_HOLD),
            "--cooldown", str(COOLDOWN),
            "--regime-filter", "vol_compress",
            "--vol-window", str(VOL_WINDOW),
            "--vol-threshold", str(th),
            "--days-list", DAYS_LIST,
            "--run-id", run_id,
        ]
        print(f"[SWEEP] threshold={th} (run_id={run_id}) ...", flush=True)
        ret = subprocess.run(cmd, cwd=PROJECT_ROOT)
        if ret.returncode != 0:
            print(f"[SWEEP] WARN: threshold={th} exited with {ret.returncode}", flush=True)
        p = DIAG / f"tcn_candidate_validation_{SYMBOL}_{TIMEFRAME}_{MODEL_ID}_{run_id}.json"
        if p.exists():
            json_paths.append(p)
        else:
            print(f"[SWEEP] MISSING: {p}", flush=True)
    return json_paths


def load_result(p: Path) -> dict | None:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Load error {p}: {e}", flush=True)
        return None


def row_by_days(results: list, days: int):
    for r in results:
        if r.get("days") == days:
            return r
    return None


def summarize_and_recommend(json_paths: list[Path]) -> None:
    """4개 JSON에서 30d/365d만 추출해 표 출력 + 추천 threshold."""
    rows = []
    for p in sorted(json_paths):
        data = load_result(p)
        if not data:
            continue
        meta = data.get("meta", {})
        th = meta.get("vol_threshold")
        if th is None:
            run_id = meta.get("run_id", p.stem)
            for r in data.get("results", []):
                th = r.get("vol_threshold")
                if th is not None:
                    break
        res = data.get("results", [])
        r30 = row_by_days(res, 30)
        r365 = row_by_days(res, 365)
        summary = data.get("summary", {})
        overblock_nogo = summary.get("overblock_nogo", False)
        overblock_warn = summary.get("overblock_warning", False)
        rows.append({
            "path": p.name,
            "threshold": th,
            "r30": r30,
            "r365": r365,
            "overblock_nogo": overblock_nogo,
            "overblock_warn": overblock_warn,
        })

    if not rows:
        print("No results to summarize.", flush=True)
        return

    # 표: 30d / 365d
    print("", flush=True)
    print("========== 30d ==========", flush=True)
    print(f"{'threshold':<12} {'trades':>8} {'cost_on':>10} {'cost_off':>10} {'MDD':>8} {'blocked_ratio':>12} {'pct_compress':>12}", flush=True)
    print("-" * 80, flush=True)
    for x in rows:
        r = x["r30"]
        th = x["threshold"]
        if r is None:
            th_s = f"{th:.4f}" if isinstance(th, (int, float)) else str(th)
            print(f"{th_s:<12} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>12} {'N/A':>12}", flush=True)
            continue
        trades = r.get("trades")
        cost_on = r.get("cost_on_return")
        cost_off = r.get("cost_off_return")
        mdd = r.get("max_drawdown")
        br = r.get("blocked_ratio")
        pc = r.get("pct_compress")
        br_s = f"{br:.2%}" if br is not None else "N/A"
        pc_s = f"{pc:.2%}" if pc is not None else "N/A"
        cost_on_s = f"{cost_on:.4f}" if cost_on is not None else "N/A"
        cost_off_s = f"{cost_off:.4f}" if cost_off is not None else "N/A"
        mdd_s = f"{mdd:.4f}" if mdd is not None else "N/A"
        th_s = f"{th:.4f}" if isinstance(th, (int, float)) else str(th)
        print(f"{th_s:<12} {trades or 'N/A':>8} {cost_on_s:>10} {cost_off_s:>10} {mdd_s:>8} {br_s:>12} {pc_s:>12}", flush=True)

    print("", flush=True)
    print("========== 365d ==========", flush=True)
    print(f"{'threshold':<12} {'trades':>8} {'cost_on':>10} {'cost_off':>10} {'MDD':>8} {'blocked_ratio':>12} {'pct_compress':>12} {'overblock':>10}", flush=True)
    print("-" * 95, flush=True)
    for x in rows:
        r = x["r365"]
        th = x["threshold"]
        ok = "NO-GO" if x["overblock_nogo"] else ("WARN" if x["overblock_warn"] else "")
        if r is None:
            th_s = f"{th:.4f}" if isinstance(th, (int, float)) else str(th)
            print(f"{th_s:<12} {'N/A':>8} {'N/A':>10} {'N/A':>10} {'N/A':>8} {'N/A':>12} {'N/A':>12} {ok:>10}", flush=True)
            continue
        trades = r.get("trades")
        cost_on = r.get("cost_on_return")
        cost_off = r.get("cost_off_return")
        mdd = r.get("max_drawdown")
        br = r.get("blocked_ratio")
        pc = r.get("pct_compress")
        br_s = f"{br:.2%}" if br is not None else "N/A"
        pc_s = f"{pc:.2%}" if pc is not None else "N/A"
        cost_on_s = f"{cost_on:.4f}" if cost_on is not None else "N/A"
        cost_off_s = f"{cost_off:.4f}" if cost_off is not None else "N/A"
        mdd_s = f"{mdd:.4f}" if mdd is not None else "N/A"
        th_s = f"{th:.4f}" if isinstance(th, (int, float)) else str(th)
        print(f"{th_s:<12} {trades or 'N/A':>8} {cost_on_s:>10} {cost_off_s:>10} {mdd_s:>8} {br_s:>12} {pc_s:>12} {ok:>10}", flush=True)

    # 추천: 365d cost_on 최대 + overblock_nogo 아님 + MDD 허용
    valid = [x for x in rows if x["r365"] is not None and not x["overblock_nogo"]]
    if not valid:
        print("", flush=True)
        print("추천: 없음 (모두 overblock_nogo 또는 365d 결과 없음). threshold 상향 중단 또는 동적 threshold 검토.", flush=True)
        return
    best = max(valid, key=lambda x: (x["r365"].get("cost_on_return") or -1e9, -(x["r365"].get("max_drawdown") or 1)))
    th_best = best["threshold"]
    c_on = best["r365"].get("cost_on_return")
    mdd_best = best["r365"].get("max_drawdown")
    trades_best = best["r365"].get("trades")
    print("", flush=True)
    print(f"추천 threshold: {th_best} (근거: 365d cost_on={c_on:.4f}, MDD={mdd_best:.4f}, trades={trades_best}, overblock_nogo=False)", flush=True)
    if c_on is not None and c_on < 0:
        print("  -> 365d cost_on 아직 음수. 후속: vol_window(24/48/96) 비교 또는 동적 quantile threshold 검토.", flush=True)
    elif mdd_best is not None and mdd_best > 0.08:
        print("  -> 365d MDD > 8%. 필요 시 cooldown/min_hold 재조정.", flush=True)


def collect_by_run_id(json_dir: Path) -> list[Path]:
    prefix = f"tcn_candidate_validation_{SYMBOL}_{TIMEFRAME}_{MODEL_ID}_"
    json_paths = []
    for th in THRESHOLDS:
        run_id = run_id_for_threshold(th)
        p = json_dir / f"{prefix}{run_id}.json"
        if p.exists():
            json_paths.append(p)
    return json_paths


def collect_by_vol_threshold(json_dir: Path) -> list[Path]:
    """디렉터리에서 vol_compress 결과를 vol_threshold별로 1개씩 수집 (최신 파일)."""
    prefix = f"tcn_candidate_validation_{SYMBOL}_{TIMEFRAME}_{MODEL_ID}_"
    by_th: dict[float, Path] = {}
    for p in json_dir.glob(f"{prefix}*.json"):
        data = load_result(p)
        if not data:
            continue
        meta = data.get("meta", {})
        if meta.get("regime_rule") != "vol_compress":
            continue
        th = meta.get("vol_threshold")
        if th is None:
            for r in data.get("results", []):
                th = r.get("vol_threshold")
                if th is not None:
                    break
        if th is None:
            continue
        th = round(th, 4)
        prev = by_th.get(th)
        if prev is None or p.stat().st_mtime > prev.stat().st_mtime:
            by_th[th] = p
    # THRESHOLDS 순서로 반환 (없으면 해당 threshold는 스킵)
    out = []
    for t in THRESHOLDS:
        if t in by_th:
            out.append(by_th[t])
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="vol_compress vol_threshold sweep (30,365d only)")
    ap.add_argument("--summary-only", action="store_true", help="기존 JSON만 수집해 요약 (스윕 실행 안 함)")
    ap.add_argument("--summary-from-dir", action="store_true", help="디렉터리에서 vol_compress 결과를 vol_threshold별로 수집 후 요약")
    ap.add_argument("--json-dir", type=Path, default=DIAG, help="JSON 디렉터리 (summary 시)")
    args = ap.parse_args()

    if args.summary_only:
        json_paths = collect_by_run_id(args.json_dir)
        if not json_paths:
            print("No JSONs with run_id vol_t0p0010/15/20/25. Try --summary-from-dir or run full sweep.", flush=True)
        else:
            print(f"Found {len(json_paths)}/4 by run_id.", flush=True)
        summarize_and_recommend(json_paths)
        return 0

    if args.summary_from_dir:
        json_paths = collect_by_vol_threshold(args.json_dir)
        print(f"Found {len(json_paths)} vol_compress result(s) by vol_threshold.", flush=True)
        summarize_and_recommend(json_paths)
        return 0

    json_paths = run_sweep()
    summarize_and_recommend(json_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
