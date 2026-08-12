#!/usr/bin/env python3
"""
TCN top2 전용 과매매 억제 필터 스윕: sweep-ids(h30_t0p004, h15_t0p004 등)에 대해
min_max_proba / max_entropy / min_hold / cooldown 그리드를 돌려 cost_on_return 개선 조합을 찾고,
실전 후보 1개 + 차선 1개를 추천.

사용법:
  source .venv/bin/activate
  python -m scripts.run_tcn_filter_sweep_top2 --days 7 --symbol BTCUSDT --timeframe 5m --sweep-ids "h30_t0p004,h15_t0p004" --quick
  python -m scripts.run_tcn_filter_sweep_top2 --days 7 --symbol BTCUSDT --timeframe 5m --sweep-ids "h30_t0p004,h15_t0p004"
  python -m scripts.run_tcn_filter_sweep_top2 --days 30 --symbol BTCUSDT --timeframe 5m --sweep-ids "h15_t0p004" --ultra  # 초보수 16조합
  python -m scripts.run_tcn_filter_sweep_top2 --days 30 --symbol BTCUSDT --timeframe 5m --sweep-ids "h15_t0p004" --ultra-mini  # B 중심 8조합
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DIAG = PROJECT_ROOT / "data" / "diagnostics"
MODELS_DIR = DIAG / "models"

# Grid: full (270) unchanged; quick = aggressive 36 (h15 중심, trades/MDD 억제)
QUICK_GRID_MIN_MAX_PROBA = [0.57, 0.60, 0.63]
QUICK_GRID_MAX_ENTROPY = [1.40, 1.35, 1.30]
QUICK_GRID_MIN_HOLD = [36, 48]
QUICK_GRID_COOLDOWN = [12, 24]
QUICK_GRID = {
    "min_max_proba": QUICK_GRID_MIN_MAX_PROBA,
    "max_entropy": QUICK_GRID_MAX_ENTROPY,
    "min_hold": QUICK_GRID_MIN_HOLD,
    "cooldown": QUICK_GRID_COOLDOWN,
}
FULL_GRID = {
    "min_max_proba": [0.45, 0.47, 0.50, 0.52, 0.55, 0.57],
    "max_entropy": [1.55, 1.50, 1.45, 1.40, 1.35],
    "min_hold": [12, 24, 36],
    "cooldown": [6, 12, 24],
}

# Ultra: 초보수 16조합 (장기 cost_on 안정화, trades 최소화)
ULTRA_GRID = {
    "min_max_proba": [0.60, 0.63],
    "max_entropy": [1.30, 1.25],
    "min_hold": [48, 60],
    "cooldown": [24, 36],
}

# Ultra-mini: B 중심 미세 압축 8조합 (h15_t0p004 베이스 0.57/1.35/48/24 근처)
ULTRA_MINI_GRID_MIN_MAX_PROBA = [0.57, 0.58, 0.59, 0.60]
ULTRA_MINI_GRID_MAX_ENTROPY = [1.35, 1.33]
ULTRA_MINI_GRID_MIN_HOLD = [48]
ULTRA_MINI_GRID_COOLDOWN = [24]
ULTRA_MINI_GRID = {
    "min_max_proba": ULTRA_MINI_GRID_MIN_MAX_PROBA,
    "max_entropy": ULTRA_MINI_GRID_MAX_ENTROPY,
    "min_hold": ULTRA_MINI_GRID_MIN_HOLD,
    "cooldown": ULTRA_MINI_GRID_COOLDOWN,
}


def _grid_combos(quick: bool, ultra: bool, ultra_mini: bool, max_combos: int | None):
    # 우선순위: ultra_mini > ultra > quick > full
    if ultra_mini:
        g = ULTRA_MINI_GRID
    elif ultra:
        g = ULTRA_GRID
    elif quick:
        g = QUICK_GRID
    else:
        g = FULL_GRID
    combos = []
    for mmp in g["min_max_proba"]:
        for me in g["max_entropy"]:
            for mh in g["min_hold"]:
                for cd in g["cooldown"]:
                    combos.append({"min_max_proba": mmp, "max_entropy": me, "min_hold": mh, "cooldown": cd})
                    if max_combos and not ultra and not ultra_mini and len(combos) >= max_combos:
                        return combos
    return combos


def main() -> int:
    parser = argparse.ArgumentParser(description="TCN top2 filter sweep: min_max_proba / max_entropy / min_hold / cooldown")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--sweep-ids", type=str, required=True, help="Comma-separated model ids, e.g. h30_t0p004,h15_t0p004")
    parser.add_argument("--out-prefix", type=str, default="tcn_filter_sweep_top2")
    parser.add_argument("--run-id", type=str, default=None, help="Optional run id; if not set and unique-out, auto YYYYMMDD_HHMMSS")
    parser.add_argument("--no-unique-out", action="store_true", help="Use legacy YYYYMMDD-only filename (no symbol/timeframe/days/mode)")
    parser.add_argument("--quick", action="store_true", help="Smaller grid (~36 combos per id)")
    parser.add_argument("--ultra", action="store_true", help="Ultra conservative grid (16 combos: proba 0.60/0.63, entropy 1.30/1.25, hold 48/60, cd 24/36)")
    parser.add_argument("--ultra-mini", action="store_true", dest="ultra_mini", help="Ultra-mini 8 combos (B 중심: proba 0.57~0.60, entropy 1.35/1.33, hold 48, cd 24)")
    parser.add_argument("--max-combos", type=int, default=None, help="Cap total combos per id (e.g. 90)")
    args = parser.parse_args()

    if sum([args.quick, args.ultra, args.ultra_mini]) > 1:
        print("ERROR: --quick, --ultra, --ultra-mini are mutually exclusive", file=sys.stderr)
        return 1

    sweep_ids = [x.strip() for x in args.sweep_ids.split(",") if x.strip()]
    if not sweep_ids:
        print("ERROR: --sweep-ids required", file=sys.stderr)
        return 1

    from scripts.run_tcn_label_sweep_v2 import (
        get_7d_ohlcv_and_proba,
        run_backtest_7d,
        signal_quality_7d,
    )

    commission = 0.0009
    slippage = 0.0001
    combos = _grid_combos(args.quick, args.ultra, args.ultra_mini, args.max_combos)
    n_combos = len(combos)
    mode = "ultra_mini" if args.ultra_mini else ("ultra" if args.ultra else ("quick" if args.quick else "full"))
    print(f"[SWEEP] days={args.days} symbol={args.symbol} timeframe={args.timeframe} sweep-ids={args.sweep_ids} quick={args.quick} ultra={args.ultra} ultra_mini={args.ultra_mini} combos_per_id={n_combos} mode={mode}", flush=True)

    per_id_results: dict[str, list[dict]] = {}
    per_id_inspect: dict[str, dict] = {}
    per_id_error: dict[str, str] = {}

    for sid in sweep_ids:
        model_path = MODELS_DIR / f"tcn_{sid}.pt"
        if not model_path.exists():
            per_id_error[sid] = f"model not found: {model_path}"
            print(f"[SWEEP] id={sid} SKIP: model not found", flush=True)
            continue
        print(f"[SWEEP] id={sid} get_7d_ohlcv_and_proba ...", flush=True)
        triple, eval_err = get_7d_ohlcv_and_proba(
            model_path, args.symbol, args.timeframe, args.days, log_prefix=f"{sid} ",
        )
        if triple is None:
            per_id_error[sid] = eval_err or "get_7d returned None"
            print(f"[SWEEP] id={sid} SKIP: {per_id_error[sid]}", flush=True)
            continue
        df_bt, pl, ps = triple
        if len(df_bt) != len(pl):
            min_len = min(len(df_bt), len(pl))
            print(f"[SWEEP][FALLBACK] id={sid} mismatch len(df_bt)={len(df_bt)} len(pl)={len(pl)} -> tail-align min_len={min_len}", flush=True)
            import numpy as np
            df_bt = df_bt.tail(min_len).copy()
            pl = np.asarray(pl[-min_len:], dtype=pl.dtype)
            ps = np.asarray(ps[-min_len:], dtype=ps.dtype)
        per_id_inspect[sid] = signal_quality_7d(pl, ps)
        results = []
        for i, co in enumerate(combos):
            print(f"[SWEEP] id={sid} combo {i+1}/{n_combos} min_max_proba={co['min_max_proba']} max_entropy={co['max_entropy']} min_hold={co['min_hold']} cooldown={co['cooldown']}", flush=True)
            row = {
                "id": sid,
                "min_max_proba": co["min_max_proba"],
                "max_entropy": co["max_entropy"],
                "min_hold": co["min_hold"],
                "cooldown": co["cooldown"],
                "trades": 0,
                "cost_on_return": 0.0,
                "cost_off_return": 0.0,
                "cost_on_win_rate": 0.0,
                "cost_off_win_rate": 0.0,
                "max_drawdown": 0.0,
                "entries_attempted": None,
                "entries_executed": None,
                "filter_skip_stats": {},
                "backtest_error_on": None,
                "backtest_error_off": None,
            }
            try:
                res_on, err_on = run_backtest_7d(
                    args.symbol, args.timeframe, df_bt, pl, ps,
                    commission, slippage,
                    min_max_proba=co["min_max_proba"],
                    max_entropy=co["max_entropy"],
                    min_hold=co["min_hold"],
                    cooldown=co["cooldown"],
                )
                res_off, err_off = run_backtest_7d(
                    args.symbol, args.timeframe, df_bt, pl, ps,
                    0.0, 0.0,
                    min_max_proba=co["min_max_proba"],
                    max_entropy=co["max_entropy"],
                    min_hold=co["min_hold"],
                    cooldown=co["cooldown"],
                )
                if err_on:
                    print(f"  [BT] cost_on error: {err_on[:200]}", flush=True)
                if err_off:
                    print(f"  [BT] cost_off error: {err_off[:200]}", flush=True)
                res_on = res_on or {}
                res_off = res_off or {}
                cts = res_on.get("cap_trigger_stats") or {}
                row.update({
                    "trades": int(res_on.get("total_trades", 0)),
                    "cost_on_return": float(res_on.get("total_return", 0.0)),
                    "cost_off_return": float(res_off.get("total_return", 0.0)),
                    "cost_on_win_rate": float(res_on.get("win_rate", 0.0)),
                    "cost_off_win_rate": float(res_off.get("win_rate", 0.0)),
                    "max_drawdown": float(res_on.get("max_drawdown", 0.0)),
                    "entries_attempted": res_on.get("entries_attempted"),
                    "entries_executed": cts.get("entries_executed"),
                    "filter_skip_stats": res_on.get("filter_skip_stats") or {},
                    "backtest_error_on": err_on[:500] if err_on else None,
                    "backtest_error_off": err_off[:500] if err_off else None,
                })
            except Exception as e:
                import traceback
                err_msg = f"{e}\n{traceback.format_exc()}"[:500]
                row["backtest_error_on"] = err_msg
                row["backtest_error_off"] = err_msg
                print(f"  [BT] combo exception: {e}", flush=True)
            results.append(row)
            if i == 0 or (i + 1) % 20 == 0 or row["cost_on_return"] >= 0:
                print(f"  [BT] cost_on return={row['cost_on_return']:.4f} cost_off={row['cost_off_return']:.4f} trades={row['trades']}", flush=True)
        # Sort: (1) cost_on_return desc, (2) trades asc, (3) mdd asc, (4) cost_off_return desc
        def sort_key(r):
            return (
                -float(r["cost_on_return"]),
                int(r["trades"]),
                float(r["max_drawdown"]),
                -float(r["cost_off_return"]),
            )
        results_sorted = sorted(results, key=sort_key)
        per_id_results[sid] = results_sorted
        best = results_sorted[0] if results_sorted else None
        if best:
            print(f"[SWEEP][TOP] id={sid} best cost_on_return={best['cost_on_return']:.4f} trades={best['trades']} mmp={best['min_max_proba']} me={best['max_entropy']}", flush=True)

    # Global best 1 + runner-up 1 (across all ids)
    all_rows = []
    for sid, rows in per_id_results.items():
        for r in rows:
            all_rows.append(r)
    def global_sort_key(r):
        return (
            -float(r["cost_on_return"]),
            int(r["trades"]),
            float(r["max_drawdown"]),
            -float(r["cost_off_return"]),
        )
    all_sorted = sorted(all_rows, key=global_sort_key)
    best_one = all_sorted[0] if all_sorted else None
    runner_up = all_sorted[1] if len(all_sorted) > 1 else None

    cost_on_ge_zero = [r for r in all_rows if r["cost_on_return"] is not None and r["cost_on_return"] >= 0]
    cost_on_achieved = len(cost_on_ge_zero) > 0

    if args.no_unique_out:
        date_str = datetime.now().strftime("%Y%m%d")
        out_md = DIAG / f"{args.out_prefix}_{date_str}.md"
        out_json = DIAG / f"{args.out_prefix}_{date_str}.json"
        run_id = None
    else:
        run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = DIAG / f"{args.out_prefix}_{args.symbol}_{args.timeframe}_{args.days}d_{mode}_{run_id}.json"
        out_md = DIAG / f"{args.out_prefix}_{args.symbol}_{args.timeframe}_{args.days}d_{mode}_{run_id}.md"

    # Build MD
    lines = [
        "# TCN top2 필터 스윕 결과",
        f"생성: {datetime.now().isoformat()}",
        "",
        "## 1) 실행 파라미터",
        f"- days={args.days}, symbol={args.symbol}, timeframe={args.timeframe}",
        f"- sweep-ids={args.sweep_ids}",
        f"- mode={mode}, quick={args.quick}, ultra={args.ultra}, ultra_mini={args.ultra_mini}, combos_per_id={n_combos}",
        f"- grid (quick={args.quick}): min_max_proba, max_entropy, min_hold, cooldown",
        "",
    ]
    if per_id_error:
        lines.append("## 에러 (스킵된 ID)")
        for sid, err in per_id_error.items():
            lines.append(f"- **{sid}**: {err}")
        lines.append("")

    lines.append("## 2) cost_on_return >= 0 달성 여부")
    if cost_on_achieved:
        lines.append(f"**달성: {len(cost_on_ge_zero)}개 조합** (cost_on_return >= 0)")
        for r in sorted(cost_on_ge_zero, key=lambda x: -x["cost_on_return"])[:10]:
            lines.append(f"- {r['id']} mmp={r['min_max_proba']} me={r['max_entropy']} mh={r['min_hold']} cd={r['cooldown']} → cost_on={r['cost_on_return']:.4f} trades={r['trades']}")
    else:
        lines.append("**미달성**: cost_on_return >= 0인 조합 없음. 최대한 0에 근접한 조합을 추천.")
    lines.append("")

    lines.append("## 3) ID별 상위 10개 조합")
    for sid in sweep_ids:
        if sid in per_id_error:
            continue
        rows = per_id_results.get(sid, [])[:10]
        insp = per_id_inspect.get(sid) or {}
        lines.append(f"### {sid}")
        lines.append(f"inspect_7d: avg_max_proba={insp.get('avg_max_proba')}, entropy_mean={insp.get('entropy_mean')}, max_proba_lt_055={insp.get('max_proba_lt_055')}")
        lines.append("| min_max_proba | max_entropy | min_hold | cooldown | trades | cost_on_return | cost_off_return | cost_on_wr | cost_off_wr | mdd | entries_attempted | entries_executed |")
        lines.append("|---------------|-------------|----------|----------|--------|----------------|-----------------|------------|-------------|-----|-------------------|------------------|")
        for r in rows:
            lines.append(
                "| {:.2f} | {:.2f} | {} | {} | {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {} | {} |".format(
                    r["min_max_proba"], r["max_entropy"], r["min_hold"], r["cooldown"],
                    r["trades"], r["cost_on_return"], r["cost_off_return"],
                    r["cost_on_win_rate"], r["cost_off_win_rate"], r["max_drawdown"],
                    r.get("entries_attempted") if r.get("entries_attempted") is not None else "-",
                    r.get("entries_executed") if r.get("entries_executed") is not None else "-",
                )
            )
        lines.append("")
        lines.append("filter_skip_stats (상위 3개):")
        for i, r in enumerate(rows[:3], 1):
            lines.append(f"  - #{i}: {r.get('filter_skip_stats', {})}")
        lines.append("")

    lines.append("## 4) 실전 후보 1개 + 차선 1개 추천")
    if best_one:
        lines.append("### Best 1")
        lines.append(f"- **id**: {best_one['id']}")
        lines.append(f"- **필터**: min_max_proba={best_one['min_max_proba']}, max_entropy={best_one['max_entropy']}, min_hold={best_one['min_hold']}, cooldown={best_one['cooldown']}")
        lines.append(f"- cost_on_return={best_one['cost_on_return']:.4f}, cost_off_return={best_one['cost_off_return']:.4f}, trades={best_one['trades']}, mdd={best_one['max_drawdown']:.4f}")
    if runner_up:
        lines.append("### Runner-up 1")
        lines.append(f"- **id**: {runner_up['id']}")
        lines.append(f"- **필터**: min_max_proba={runner_up['min_max_proba']}, max_entropy={runner_up['max_entropy']}, min_hold={runner_up['min_hold']}, cooldown={runner_up['cooldown']}")
        lines.append(f"- cost_on_return={runner_up['cost_on_return']:.4f}, cost_off_return={runner_up['cost_off_return']:.4f}, trades={runner_up['trades']}, mdd={runner_up['max_drawdown']:.4f}")
    lines.append("")

    lines.append("## 5) 해석")
    lines.append("- cost_off는 괜찮은데 cost_on만 무너지는지: 필터로 진입 수를 줄여 수수료 부담을 낮춤.")
    lines.append("- 어떤 필터가 trade 수를 줄였는지: filter_skip_stats의 by_min_max_proba, by_max_entropy 등으로 확인.")
    lines.append("- 다음 단계: commission/slippage 그리드, long-only, regime 필터 등으로 추가 개선 가능.")
    lines.append("")
    lines.append(f"저장: {out_md} | {out_json}")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Integrity check (산출물 검증)
    if args.ultra_mini and n_combos != 8:
        raise RuntimeError(f"Ultra-mini sweep integrity failed: combos_per_id={n_combos} (expected 8)")
    if args.ultra and n_combos != 16:
        raise RuntimeError(f"Ultra sweep integrity failed: combos_per_id={n_combos} (expected 16)")
    if not args.quick and not args.ultra and not args.ultra_mini and n_combos != 270:
        raise RuntimeError(f"Full sweep integrity failed: combos_per_id={n_combos} (expected 270)")
    for sid in sweep_ids:
        if sid in per_id_error:
            continue
        n_rows = len(per_id_results.get(sid, []))
        if n_rows < 10:
            print(f"[WARN] id={sid} has only {n_rows} results (target >=10); recording available.", flush=True)
    top10_per_id = {sid: per_id_results.get(sid, [])[:10] for sid in sweep_ids}

    payload = {
        "meta": {
            "generated_at": datetime.now().isoformat(),
            "days": args.days,
            "symbol": args.symbol,
            "timeframe": args.timeframe,
            "sweep_ids": sweep_ids,
            "quick": args.quick,
            "ultra": args.ultra,
            "ultra_mini": args.ultra_mini,
            "combos_per_id": n_combos,
            "mode": mode,
            "run_id": run_id,
            "out_json_path": str(out_json),
            "out_md_path": str(out_md),
        },
        "per_id_best": {sid: (per_id_results[sid][0] if per_id_results.get(sid) else None) for sid in sweep_ids},
        "top10_per_id": top10_per_id,
        "per_id_inspect_7d": per_id_inspect,
        "per_id_error": per_id_error,
        "verdict": {
            "cost_on_return_ge_zero_achieved": cost_on_achieved,
            "num_combos_ge_zero": len(cost_on_ge_zero),
            "best_one": best_one,
            "runner_up": runner_up,
        },
    }
    def _json_default(obj):
        import numpy as np
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(type(obj))

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_json_default)

    print("")
    print("=" * 60)
    print("TCN top2 필터 스윕 완료")
    print("=" * 60)
    print(f"저장: {out_md}, {out_json}")
    print(f"cost_on_return >= 0: {cost_on_achieved} ({len(cost_on_ge_zero)}개)")
    if best_one:
        print(f"Best: {best_one['id']} mmp={best_one['min_max_proba']} me={best_one['max_entropy']} cost_on={best_one['cost_on_return']:.4f}")

    # 결과 요약: verdict.best_one (없으면 top10 1등) + 저장 경로
    summary_row = best_one
    if summary_row is None and sweep_ids:
        for sid in sweep_ids:
            top10 = per_id_results.get(sid, [])
            if top10:
                summary_row = top10[0]
                break
    if summary_row:
        print("")
        print("[결과 요약] verdict.best_one (또는 top10 1등)")
        print(f"  id={summary_row['id']} min_max_proba={summary_row['min_max_proba']} max_entropy={summary_row['max_entropy']} min_hold={summary_row['min_hold']} cooldown={summary_row['cooldown']} trades={summary_row['trades']}")
        print(f"  cost_on_return={summary_row['cost_on_return']} cost_off_return={summary_row['cost_off_return']} max_drawdown={summary_row['max_drawdown']}")
        print(f"  win_rate_on(cost_on_win_rate)={summary_row.get('cost_on_win_rate')} win_rate_off(cost_off_win_rate)={summary_row.get('cost_off_win_rate')}")
        print(f"  out_json_path={out_json}")
        print(f"  out_md_path={out_md}")

    # 30d ultra_mini: 실전 후보 선정 규칙 요약 (사람이 1개 고를 수 있도록)
    if args.ultra_mini and sweep_ids:
        cand_id = "h15_t0p004" if "h15_t0p004" in sweep_ids else sweep_ids[0]
        rows = per_id_results.get(cand_id, [])
        print("")
        print("[ultra_mini] best_one / runner_up 한 줄 요약")
        if best_one:
            b = best_one
            print(f"  best_one:   id={b['id']} mmp={b['min_max_proba']} me={b['max_entropy']} mh={b['min_hold']} cd={b['cooldown']} trades={b['trades']} cost_on={b['cost_on_return']:.4f} mdd={b['max_drawdown']:.4f} wr_on={b.get('cost_on_win_rate', 0):.2f}")
        if runner_up:
            r = runner_up
            print(f"  runner_up:  id={r['id']} mmp={r['min_max_proba']} me={r['max_entropy']} mh={r['min_hold']} cd={r['cooldown']} trades={r['trades']} cost_on={r['cost_on_return']:.4f} mdd={r['max_drawdown']:.4f} wr_on={r.get('cost_on_win_rate', 0):.2f}")
        # 규칙: cost_on>=0.005, trades 40~60, MDD<=0.012, win_rate_on>=0.45
        def _satisfies(r):
            co = float(r.get("cost_on_return") or 0)
            t = int(r.get("trades") or 0)
            mdd = float(r.get("max_drawdown") or 1)
            wr = float(r.get("cost_on_win_rate") or 0)
            return co >= 0.005, 40 <= t <= 60, mdd <= 0.012, wr >= 0.45
        satisfied = [r for r in rows if all(_satisfies(r))]
        n_ok = len(satisfied)
        # 가장 조건에 근접한 1개: 충족 개수 최대 → cost_on 내림 → |trades-50| 최소
        def _score(r):
            ok = _satisfies(r)
            n = sum(ok)
            co = float(r.get("cost_on_return") or 0)
            t = int(r.get("trades") or 0)
            return (n, co, -abs(t - 50))
        closest = max(rows, key=lambda r: _score(r)) if rows else None
        print("")
        print("[ultra_mini] 실전 후보 선정 규칙 요약 (cost_on>=0.005, trades 40~60, MDD<=0.012, win_rate_on>=0.45)")
        print(f"  1) 조건 충족 조합 개수: {n_ok} / {len(rows)}")
        if closest:
            c_ok = _satisfies(closest)
            print(f"  2) 가장 근접한 1개: mmp={closest['min_max_proba']} me={closest['max_entropy']} mh={closest['min_hold']} cd={closest['cooldown']} | trades={closest['trades']} cost_on={closest['cost_on_return']:.4f} mdd={closest['max_drawdown']:.4f} wr_on={closest.get('cost_on_win_rate', 0):.2f}")
            if n_ok == 0:
                reasons = []
                if float(closest.get("cost_on_return") or 0) < 0.005:
                    reasons.append("cost_on 미달")
                t = int(closest.get("trades") or 0)
                if t < 40:
                    reasons.append("trades 과소")
                elif t > 60:
                    reasons.append("trades 과대")
                if float(closest.get("max_drawdown") or 1) > 0.012:
                    reasons.append("MDD 초과")
                if float(closest.get("cost_on_win_rate") or 0) < 0.45:
                    reasons.append("win_rate 미달")
                print(f"  3) 조건 미충족 사유: {' / '.join(reasons)}")
            else:
                print(f"  3) 조건 충족 조합 있음 → 위 2) 또는 JSON top10_per_id에서 실전 후보 1개 선택 가능.")
        else:
            print("  2) 조합 없음.")
            print("  3) best_one 기준 요약: cost_on / trades / MDD 는 위 [결과 요약] 참고.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
