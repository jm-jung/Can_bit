#!/usr/bin/env python3
"""
Phase D14-Final: threshold rule 마지막 확인 + 재학습 분기 계획.

목적:
- max_proba / entropy 기반 threshold 전략 최종 확인
- baseline subset 재확인
- 결과가 음수면 RULE_BRANCH_EXHAUSTED, 재학습 추천
- 재학습 후보 및 label distribution, 학습 커맨드 정리
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

COST_DEFAULT = 0.001
BASELINE_MIN_MAX_PROBA = 0.575
BASELINE_MAX_ENTROPY = 1.30
D14_ALIGNMENT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "d14_alignment"
OUT_DIR_DEFAULT = PROJECT_ROOT / "data" / "diagnostics" / "d14_final"

# argmax_class: 0=FLAT, 1=LONG, 2=SHORT. FLAT 행은 argmax side-adjusted 집계 시 제외.
FLAT_CLASS = 0


def _load_raw(days: int, alignment_dir: Path) -> pd.DataFrame | None:
    p = alignment_dir / f"d14_alignment_raw_{days}d.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)


def _non_flat(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["argmax_class"] != FLAT_CLASS].copy()


def _agg_directional(sub: pd.DataFrame, cost: float) -> dict:
    """집계: directional rows만 (argmax in {LONG,SHORT}). FLAT 제외."""
    directional = _non_flat(sub)
    n_dir = len(directional)
    if n_dir == 0:
        return {
            "mean_argmax_side_adjusted": np.nan,
            "median_argmax_side_adjusted": np.nan,
            "std_argmax_side_adjusted": np.nan,
            "mean_cost_adj_argmax": np.nan,
            "hit_rate_sign_correct": np.nan,
            "long_count": 0,
            "short_count": 0,
        }
    adj = directional["argmax_side_adjusted_return"]
    cost_adj = directional["cost_adjusted_argmax_side_return"]
    correct = directional["correct_direction_binary"]
    return {
        "mean_argmax_side_adjusted": adj.mean(),
        "median_argmax_side_adjusted": adj.median(),
        "std_argmax_side_adjusted": adj.std(),
        "mean_cost_adj_argmax": cost_adj.mean(),
        "hit_rate_sign_correct": correct.mean(),
        "long_count": int((directional["argmax_class"] == 1).sum()),
        "short_count": int((directional["argmax_class"] == 2).sum()),
    }


def _maxproba_top_analysis(df: pd.DataFrame, cost: float, top_pcts: list[float]) -> pd.DataFrame:
    rows = []
    for pct in top_pcts:
        q = 1.0 - pct / 100.0
        th = df["max_proba"].quantile(q)
        sub = df[df["max_proba"] >= th]
        agg = _agg_directional(sub, cost)
        rows.append({
            "pct": pct,
            "threshold_min": th,
            "count": len(sub),
            "count_directional": len(_non_flat(sub)),
            "mean_argmax_side_adjusted": agg["mean_argmax_side_adjusted"],
            "median_argmax_side_adjusted": agg["median_argmax_side_adjusted"],
            "std_argmax_side_adjusted": agg["std_argmax_side_adjusted"],
            "mean_cost_adj_argmax": agg["mean_cost_adj_argmax"],
            "hit_rate_sign_correct": agg["hit_rate_sign_correct"],
            "long_count": agg["long_count"],
            "short_count": agg["short_count"],
            "avg_max_proba": sub["max_proba"].mean(),
            "avg_entropy": sub["entropy"].mean(),
            "avg_abs_future_return": sub["abs_future_return"].mean(),
        })
    return pd.DataFrame(rows)


def _entropy_bottom_analysis(df: pd.DataFrame, cost: float, bottom_pcts: list[float]) -> pd.DataFrame:
    rows = []
    for pct in bottom_pcts:
        q = pct / 100.0
        th = df["entropy"].quantile(q)
        sub = df[df["entropy"] <= th]
        agg = _agg_directional(sub, cost)
        rows.append({
            "pct": pct,
            "threshold_max": th,
            "count": len(sub),
            "count_directional": len(_non_flat(sub)),
            "mean_argmax_side_adjusted": agg["mean_argmax_side_adjusted"],
            "median_argmax_side_adjusted": agg["median_argmax_side_adjusted"],
            "std_argmax_side_adjusted": agg["std_argmax_side_adjusted"],
            "mean_cost_adj_argmax": agg["mean_cost_adj_argmax"],
            "hit_rate_sign_correct": agg["hit_rate_sign_correct"],
            "long_count": agg["long_count"],
            "short_count": agg["short_count"],
            "avg_max_proba": sub["max_proba"].mean(),
            "avg_entropy": sub["entropy"].mean(),
            "avg_abs_future_return": sub["abs_future_return"].mean(),
        })
    return pd.DataFrame(rows)


def _intersection_analysis(
    df: pd.DataFrame, cost: float,
    maxproba_pcts: list[float], entropy_pcts: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """max_proba top AND entropy bottom 교집합. heatmap용 pivot + row table."""
    rows = []
    for mp in maxproba_pcts:
        for ep in entropy_pcts:
            q_mp = 1.0 - mp / 100.0
            q_ep = ep / 100.0
            th_mp = df["max_proba"].quantile(q_mp)
            th_ep = df["entropy"].quantile(q_ep)
            sub = df[(df["max_proba"] >= th_mp) & (df["entropy"] <= th_ep)]
            agg = _agg_directional(sub, cost)
            rows.append({
                "maxproba_top_pct": mp,
                "entropy_bottom_pct": ep,
                "count": len(sub),
                "mean_argmax_side_adjusted": agg["mean_argmax_side_adjusted"],
                "mean_cost_adj_argmax": agg["mean_cost_adj_argmax"],
                "median": agg["median_argmax_side_adjusted"],
                "std": agg["std_argmax_side_adjusted"],
                "hit_rate_sign_correct": agg["hit_rate_sign_correct"],
                "avg_max_proba": sub["max_proba"].mean() if len(sub) else np.nan,
                "avg_entropy": sub["entropy"].mean() if len(sub) else np.nan,
            })
    table = pd.DataFrame(rows)
    pivot_cost = table.pivot_table(
        values="mean_cost_adj_argmax", index="entropy_bottom_pct", columns="maxproba_top_pct"
    )
    return table, pivot_cost


def _baseline_subset(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["max_proba"] >= BASELINE_MIN_MAX_PROBA) & (df["entropy"] <= BASELINE_MAX_ENTROPY)
    ].copy()


def _threshold_rules_analysis(
    df: pd.DataFrame, cost: float,
    maxproba_thresholds: list[float],
    entropy_quantiles: list[float],
) -> pd.DataFrame:
    """Rule 1: max_proba >= t. Rule 2: entropy <= q. Rule 3: both."""
    directional = _non_flat(df)
    if len(directional) == 0:
        return pd.DataFrame()
    rows = []
    # Rule 1
    for t in maxproba_thresholds:
        sub = directional[directional["max_proba"] >= t]
        if len(sub) < 10:
            continue
        adj = sub["argmax_side_adjusted_return"]
        cadj = sub["cost_adjusted_argmax_side_return"]
        rows.append({
            "rule": "max_proba_ge",
            "param": t,
            "count": len(sub),
            "mean_argmax_side_adjusted": adj.mean(),
            "mean_cost_adj_argmax": cadj.mean(),
            "cumulative_proxy_sum": cadj.sum(),
            "hit_rate": sub["correct_direction_binary"].mean(),
        })
    # Rule 2: entropy <= q (q10, q5, q2, q1)
    for q in entropy_quantiles:
        th = directional["entropy"].quantile(q)
        sub = directional[directional["entropy"] <= th]
        if len(sub) < 10:
            continue
        rows.append({
            "rule": "entropy_le_q",
            "param": f"q{int(round(q*100))}",
            "param_value": float(th),
            "count": len(sub),
            "mean_argmax_side_adjusted": sub["argmax_side_adjusted_return"].mean(),
            "mean_cost_adj_argmax": sub["cost_adjusted_argmax_side_return"].mean(),
            "cumulative_proxy_sum": sub["cost_adjusted_argmax_side_return"].sum(),
            "hit_rate": sub["correct_direction_binary"].mean(),
        })
    # Rule 3: max_proba >= x AND entropy <= y (use same thresholds as intersection)
    for mp in [0.5, 1.0, 2.0, 5.0]:
        for ep in [0.5, 1.0, 2.0, 5.0]:
            th_mp = directional["max_proba"].quantile(1.0 - mp / 100.0)
            th_ep = directional["entropy"].quantile(ep / 100.0)
            sub = directional[(directional["max_proba"] >= th_mp) & (directional["entropy"] <= th_ep)]
            if len(sub) < 10:
                continue
            rows.append({
                "rule": "maxproba_top_and_entropy_bottom",
                "param": f"mp{mp}_ep{ep}",
                "count": len(sub),
                "mean_argmax_side_adjusted": sub["argmax_side_adjusted_return"].mean(),
                "mean_cost_adj_argmax": sub["cost_adjusted_argmax_side_return"].mean(),
                "cumulative_proxy_sum": sub["cost_adjusted_argmax_side_return"].sum(),
                "hit_rate": sub["correct_direction_binary"].mean(),
            })
    return pd.DataFrame(rows)


def _retrain_candidates_and_label_dist(
    symbol: str, timeframe: str, days: int, alignment_dir: Path
) -> pd.DataFrame:
    """재학습 후보 (horizon, threshold) + label distribution 추정. 720d raw에 future_return_15bar만 있으므로 h15만 정확하고 나머지는 동일 데이터로 근사 또는 별도 로드 필요."""
    raw_720 = _load_raw(720, alignment_dir)
    if raw_720 is None or len(raw_720) == 0:
        return _retrain_candidates_static()
    # 현재 데이터는 horizon=15 기준. 다른 horizon은 OHLCV 재로드로 계산 가능하나 여기선 후보 목록 + h15 기준 비율만.
    candidates = _retrain_candidates_static()
    # h15_t0p004에 대해 raw의 future_return_15bar로 class 비율 계산
    fr = raw_720["future_return_15bar"].values
    pos_thr = 0.004
    neg_thr = -0.004
    long_r = float((fr > pos_thr).mean())
    short_r = float((fr < neg_thr).mean())
    flat_r = float(1.0 - long_r - short_r)
    idx = candidates["id"] == "h15_t0p004"
    candidates = candidates.copy()
    candidates.loc[idx, "long_ratio"] = long_r
    candidates.loc[idx, "flat_ratio"] = flat_r
    candidates.loc[idx, "short_ratio"] = short_r
    return candidates


def _retrain_candidates_static() -> pd.DataFrame:
    """재학습 후보 테이블 (label ratio는 추정 또는 비어 있음)."""
    rows = [
        {"id": "h10_t0p003", "horizon": 10, "pos_threshold": 0.003, "neg_threshold": -0.003, "group": "G1"},
        {"id": "h15_t0p004", "horizon": 15, "pos_threshold": 0.004, "neg_threshold": -0.004, "group": "G1"},
        {"id": "h20_t0p005", "horizon": 20, "pos_threshold": 0.005, "neg_threshold": -0.005, "group": "G1"},
        {"id": "h30_t0p006", "horizon": 30, "pos_threshold": 0.006, "neg_threshold": -0.006, "group": "G1"},
        {"id": "h15_t0p005", "horizon": 15, "pos_threshold": 0.005, "neg_threshold": -0.005, "group": "G2"},
        {"id": "h15_t0p006", "horizon": 15, "pos_threshold": 0.006, "neg_threshold": -0.006, "group": "G2"},
        {"id": "h10_t0p004", "horizon": 10, "pos_threshold": 0.004, "neg_threshold": -0.004, "group": "G3"},
        {"id": "h20_t0p004", "horizon": 20, "pos_threshold": 0.004, "neg_threshold": -0.004, "group": "G3"},
        {"id": "h30_t0p004", "horizon": 30, "pos_threshold": 0.004, "neg_threshold": -0.004, "group": "G3"},
    ]
    df = pd.DataFrame(rows)
    for c in ["long_ratio", "flat_ratio", "short_ratio"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def _training_command(sweep_id: str, horizon: int, pos_threshold: float, neg_threshold: float) -> str:
    out_pt = f"data/diagnostics/models/tcn_{sweep_id}.pt"
    return (
        f"python -m src.dl.train.train_tcn --symbol BTCUSDT --timeframe 5m "
        f"--epochs 50 --horizon-bars {horizon} --pos-threshold {pos_threshold} "
        f"--neg-threshold {neg_threshold} --out-model {out_pt} --seed 42"
    )


# 실전성 있는 양수로 인정할 최소 mean cost-adjusted (0.02% = 2bp 이상; 그 미만은 0에 수렴으로 간주)
MIN_MEAN_COST_ADJ_POSITIVE = 0.0002


def _judge_rule_branch(
    maxproba_720: pd.DataFrame,
    entropy_720: pd.DataFrame,
    intersection_720: pd.DataFrame,
    threshold_rules_720: pd.DataFrame,
    cost: float,
) -> tuple[str, str]:
    """RULE_BRANCH_STILL_POSSIBLE vs RULE_BRANCH_EXHAUSTED_RETRAIN_RECOMMENDED + 근거."""
    reasons = []
    any_meaningful_positive = False
    min_count_meaningful = 50
    # max_proba 극소수: 의미 있는 양수만 인정 (MIN_MEAN_COST_ADJ_POSITIVE 초과)
    if not maxproba_720.empty:
        for _, r in maxproba_720.iterrows():
            cadj = r.get("mean_cost_adj_argmax", np.nan)
            if pd.isna(cadj):
                continue
            if r["count"] >= min_count_meaningful and cadj > MIN_MEAN_COST_ADJ_POSITIVE:
                any_meaningful_positive = True
                reasons.append(f"max_proba top {r['pct']}% mean_cost_adj_argmax={cadj:.4f} (n={r['count']})")
    # entropy 극소수
    if not entropy_720.empty:
        for _, r in entropy_720.iterrows():
            cadj = r.get("mean_cost_adj_argmax", np.nan)
            if pd.isna(cadj):
                continue
            if r["count"] >= min_count_meaningful and cadj > MIN_MEAN_COST_ADJ_POSITIVE:
                any_meaningful_positive = True
                reasons.append(f"entropy bottom {r['pct']}% mean_cost_adj_argmax={cadj:.4f} (n={r['count']})")
    # intersection: 의미 있는 양수 셀
    if not intersection_720.empty:
        inter = intersection_720[intersection_720["count"] >= min_count_meaningful]
        pos_inter = inter[inter["mean_cost_adj_argmax"] > MIN_MEAN_COST_ADJ_POSITIVE]
        if len(pos_inter) > 0:
            any_meaningful_positive = True
            reasons.append(f"intersection: {len(pos_inter)} cells mean_cost_adj > {MIN_MEAN_COST_ADJ_POSITIVE} (n>={min_count_meaningful})")
    if any_meaningful_positive and reasons:
        return "RULE_BRANCH_STILL_POSSIBLE", "; ".join(reasons[:5])
    # 기본: 모두 음수 또는 0에 수렴하는 미미한 양수만 있음 → 재학습 전환
    return "RULE_BRANCH_EXHAUSTED_RETRAIN_RECOMMENDED", (
        "max_proba/entropy 극소수 구간 및 intersection에서 "
        f"실전성 있는 cost-adjusted 양수(>{MIN_MEAN_COST_ADJ_POSITIVE}) 없음. "
        "0에 수렴하는 구간만 있거나 표본 부족."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase D14-Final: threshold final check + retrain plan",
    )
    parser.add_argument("--model-id", type=str, default="h15_t0p004")
    parser.add_argument("--days", type=int, nargs="+", default=[365, 720])
    parser.add_argument("--cost", type=float, default=COST_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--alignment-dir", type=str, default=None)
    parser.add_argument("--include-baseline-subset", action="store_true")
    parser.add_argument(
        "--top-percentiles",
        type=str,
        default="0.1,0.25,0.5,1,2,5",
        help="max_proba top / entropy bottom percentiles",
    )
    parser.add_argument(
        "--maxproba-thresholds",
        type=str,
        default="0.60,0.65,0.70,0.75,0.80",
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR_DEFAULT
    alignment_dir = Path(args.alignment_dir) if args.alignment_dir else D14_ALIGNMENT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    top_pcts = [float(x) for x in args.top_percentiles.split(",")]
    maxproba_thresholds = [float(x) for x in args.maxproba_thresholds.split(",")]
    entropy_quantiles = [0.10, 0.05, 0.02, 0.01]
    maxproba_intersect = [0.5, 1.0, 2.0, 5.0]
    entropy_intersect = [0.5, 1.0, 2.0, 5.0]

    results_365 = {}
    results_720 = {}
    for days in args.days:
        df = _load_raw(days, alignment_dir)
        if df is None or len(df) < 100:
            print(f"[D14-Final] Skip {days}d: no raw data", file=sys.stderr)
            continue
        print(f"[D14-Final] {days}d rows={len(df)}", flush=True)
        # A-1 max_proba top
        t1 = _maxproba_top_analysis(df, args.cost, top_pcts)
        t1.to_csv(out_dir / f"d14_final_maxproba_top_{days}d.csv", index=False)
        if days == 365:
            results_365["maxproba_top"] = t1
        else:
            results_720["maxproba_top"] = t1
        # A-2 entropy bottom
        t2 = _entropy_bottom_analysis(df, args.cost, top_pcts)
        t2.to_csv(out_dir / f"d14_final_entropy_bottom_{days}d.csv", index=False)
        if days == 365:
            results_365["entropy_bottom"] = t2
        else:
            results_720["entropy_bottom"] = t2
        # A-3 intersection
        tab, pivot = _intersection_analysis(df, args.cost, maxproba_intersect, entropy_intersect)
        tab.to_csv(out_dir / f"d14_final_intersection_{days}d.csv", index=False)
        pivot.to_csv(out_dir / f"d14_final_intersection_heatmap_{days}d.csv")
        if days == 365:
            results_365["intersection"] = tab
        else:
            results_720["intersection"] = tab
        # A-5 threshold rules (720만 상세 저장, 365도 저장)
        rules = _threshold_rules_analysis(df, args.cost, maxproba_thresholds, entropy_quantiles)
        rules.to_csv(out_dir / f"d14_final_threshold_rules_{days}d.csv", index=False)
        if days == 720:
            results_720["threshold_rules"] = rules
        # A-4 baseline subset
        if args.include_baseline_subset:
            base = _baseline_subset(df)
            if len(base) >= 50:
                b1 = _maxproba_top_analysis(base, args.cost, top_pcts)
                b2 = _entropy_bottom_analysis(base, args.cost, top_pcts)
                b1.to_csv(out_dir / f"d14_final_baseline_subset_maxproba_{days}d.csv", index=False)
                b2.to_csv(out_dir / f"d14_final_baseline_subset_entropy_{days}d.csv", index=False)
                tab_b, _ = _intersection_analysis(base, args.cost, maxproba_intersect, entropy_intersect)
                tab_b.to_csv(out_dir / f"d14_final_baseline_subset_intersection_{days}d.csv", index=False)
                if days == 365:
                    results_365["baseline_maxproba"] = b1
                    results_365["baseline_entropy"] = b2
                else:
                    results_720["baseline_maxproba"] = b1
                    results_720["baseline_entropy"] = b2

    # Retrain candidates + label dist
    retrain_df = _retrain_candidates_and_label_dist("BTCUSDT", "5m", 720, alignment_dir)
    retrain_df["train_cmd"] = retrain_df.apply(
        lambda r: _training_command(
            r["id"], int(r["horizon"]), float(r["pos_threshold"]), float(r["neg_threshold"])
        ),
        axis=1,
    )
    retrain_df.to_csv(out_dir / "d14_final_retrain_candidates.csv", index=False)

    # Judgment
    maxproba_720 = results_720.get("maxproba_top", pd.DataFrame())
    entropy_720 = results_720.get("entropy_bottom", pd.DataFrame())
    inter_720 = results_720.get("intersection", pd.DataFrame())
    rules_720 = results_720.get("threshold_rules", pd.DataFrame())
    verdict, reason = _judge_rule_branch(
        maxproba_720, entropy_720, inter_720, rules_720, args.cost
    )

    # Summary md
    summary_md = out_dir / "d14_final_summary.md"
    summary_json = out_dir / "d14_final_summary.json"
    _write_summary(
        out_dir=out_dir,
        model_id=args.model_id,
        days_list=args.days,
        cost=args.cost,
        results_365=results_365,
        results_720=results_720,
        retrain_df=retrain_df,
        verdict=verdict,
        reason=reason,
        summary_md_path=summary_md,
        summary_json_path=summary_json,
    )
    print(f"[D14-Final] Wrote {summary_md} and {summary_json}", flush=True)
    print(f"[D14-Final] Verdict: {verdict}", flush=True)
    print("[D14-Final] Done.", flush=True)
    return 0


def _write_summary(
    out_dir: Path,
    model_id: str,
    days_list: list[int],
    cost: float,
    results_365: dict,
    results_720: dict,
    retrain_df: pd.DataFrame,
    verdict: str,
    reason: str,
    summary_md_path: Path,
    summary_json_path: Path,
) -> None:
    lines = [
        "# Phase D14-Final: Threshold 최종 확인 및 재학습 분기",
        "",
        "## 1. 사용 모델",
        f"- tcn_{model_id}.pt",
        "",
        "## 2. 분석 구간",
        f"- {', '.join(str(d) + 'd' for d in days_list)}",
        "",
        "## 3. 비용 및 FLAT 처리",
        f"- cost = {cost} (0.1%). cost-adjusted proxy는 엄밀한 백테스트 PnL이 아님.",
        "- argmax == FLAT 행은 **argmax side-adjusted / cost-adjusted 집계 시 제외** (LONG/SHORT만 사용).",
        "",
        "## 4. Threshold final check 결과",
    ]
    if results_720.get("maxproba_top") is not None and not results_720["maxproba_top"].empty:
        t = results_720["maxproba_top"]
        lines.append("### 4.1 max_proba 상위 극소수 (720d)")
        for _, r in t.iterrows():
            cadj = r.get("mean_cost_adj_argmax", np.nan)
            lines.append(f"- top {r['pct']}%: count={int(r['count'])}, mean_cost_adj_argmax={cadj:.4f}")
    if results_720.get("entropy_bottom") is not None and not results_720["entropy_bottom"].empty:
        t = results_720["entropy_bottom"]
        lines.append("")
        lines.append("### 4.2 entropy 하위 극소수 (720d)")
        for _, r in t.iterrows():
            cadj = r.get("mean_cost_adj_argmax", np.nan)
            lines.append(f"- bottom {r['pct']}%: count={int(r['count'])}, mean_cost_adj_argmax={cadj:.4f}")
    if results_720.get("intersection") is not None and not results_720["intersection"].empty:
        inter = results_720["intersection"]
        pos = inter[inter["mean_cost_adj_argmax"] > MIN_MEAN_COST_ADJ_POSITIVE]
        lines.append("")
        lines.append("### 4.3 max_proba high AND entropy low 교집합 (720d)")
        lines.append(f"- 실전성 있는 양수 셀 수(mean_cost_adj > {MIN_MEAN_COST_ADJ_POSITIVE}): {len(pos)}")
    if results_720.get("baseline_maxproba") is not None and not results_720["baseline_maxproba"].empty:
        lines.append("")
        lines.append("### 4.4 baseline subset (max_proba>=0.575, entropy<=1.30) (720d)")
        lines.append("- d14_final_baseline_subset_maxproba_720d.csv, d14_final_baseline_subset_entropy_720d.csv, d14_final_baseline_subset_intersection_720d.csv 참고.")
    lines.extend([
        "",
        "## 5. 최종 판정",
        f"**{verdict}**",
        "",
        "## 6. 근거",
        reason,
        "",
        "## 7. 재학습 추천 후보 Top 3",
        "1) **h20_t0p005** — cost 대비 label margin 확대 + 100분 horizon으로 노이즈 완화 기대",
        "2) **h30_t0p006** — 극단적 move 검출, 단 label sparsity 리스크 있음",
        "3) **h15_t0p005** — 동일 horizon, threshold만 상향해 더 큰 move만 학습",
        "",
        "## 8. 바로 실행 가능한 다음 step",
        "재학습 sweep: 위 후보에 대해 `scripts/run_tcn_label_sweep_v2` 또는 `train_tcn` 직접 실행.",
        "",
    ])
    # Training commands
    lines.append("## 9. 학습 커맨드 예시")
    for _, r in retrain_df.head(5).iterrows():
        lines.append(f"- `{r['train_cmd']}`")
    summary_md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "model_id": model_id,
        "days": days_list,
        "cost": cost,
        "verdict": verdict,
        "reason": reason,
        "retrain_candidates": retrain_df.drop(columns=["train_cmd"], errors="ignore").to_dict(orient="records"),
    }
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)


if __name__ == "__main__":
    sys.exit(main())
