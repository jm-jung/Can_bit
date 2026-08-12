"""Nested expanding walk-forward + ranking for regime candidates."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from canbit_equity.backtest import run_backtest, summarize_equity
from canbit_equity.features import logistic_feature_columns
from canbit_equity.metrics import compare_to_buyhold
from canbit_equity.model import fit_predict_proba, select_threshold
from canbit_equity.regime.config import (
    CANDIDATE_FEATURE_GROUPS,
    COST_BASE,
    COST_HIGH,
    COST_LOW,
    LR_THRESHOLDS,
    MIN_TRAIN,
    SELECTION_DEV_END_MAX,
    STEP_SESSIONS,
    TEST_SESSIONS,
    VAL_SESSIONS,
    paths,
)
from canbit_equity.strategies import signal_buy_and_hold, signal_dual_trend


def _sha_list(xs: List[int]) -> str:
    return hashlib.sha256(",".join(str(i) for i in xs).encode()).hexdigest()


def score_components(sharpe: float, mdd_imp: Optional[float], cagr_pres: Optional[float]) -> float:
    return float(sharpe or 0) + 0.01 * float(mdd_imp or 0) + 0.001 * float(cagr_pres or 0)


def build_common_period(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Restrict to sessions where all regime features valid and date <= 2024-06-26."""
    from canbit_equity.regime.config import ALL_REGIME_FEATURES

    d = df.copy()
    d["session_date"] = pd.to_datetime(d["session_date"]).dt.normalize()
    end_max = pd.Timestamp(SELECTION_DEV_END_MAX)
    d = d[d["session_date"] <= end_max].copy()
    # require price features + regime features + label
    price_cols = logistic_feature_columns()
    need = price_cols + ALL_REGIME_FEATURES + ["label_long_20d", "open_adj", "close_adj", "sma_200", "sma_50"]
    for c in need:
        if c not in d.columns:
            raise KeyError(c)
    mask = d[need].notna().all(axis=1)
    common = d.loc[mask].reset_index(drop=True)
    # need enough for min train + val + test
    meta = {
        "common_start": str(common["session_date"].iloc[0].date()) if len(common) else None,
        "common_end": str(common["session_date"].iloc[-1].date()) if len(common) else None,
        "common_rows": int(len(common)),
        "selection_dev_end_max": SELECTION_DEV_END_MAX,
        "old_holdout_rows_included": int((common["session_date"] >= pd.Timestamp("2024-06-27")).sum()),
        "verdict": "COMMON_PERIOD_PASS" if len(common) >= MIN_TRAIN + VAL_SESSIONS + TEST_SESSIONS else "COMMON_PERIOD_FAIL",
    }
    return common, meta


def slice_outer_folds(n: int) -> List[Dict[str, int]]:
    folds = []
    train_end = MIN_TRAIN
    while True:
        val_start = train_end
        val_end = val_start + VAL_SESSIONS
        test_start = val_end
        test_end = test_start + TEST_SESSIONS
        if test_end > n:
            break
        folds.append(
            {
                "fold_id": len(folds),
                "train_start": 0,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "oos_start": test_start,
                "oos_end": test_end,
            }
        )
        train_end += STEP_SESSIONS
        if train_end >= n:
            break
    return folds


def _feature_cols(extra: List[str]) -> List[str]:
    return logistic_feature_columns() + list(extra)


def _aggregate_stream(net_ret: pd.Series, bh_ret: pd.Series, position: pd.Series) -> Dict[str, Any]:
    rets = net_ret.astype(float).reset_index(drop=True)
    bh = bh_ret.astype(float).reset_index(drop=True)
    eq = (1.0 + rets.fillna(0)).cumprod()
    beq = (1.0 + bh.fillna(0)).cumprod()
    n = len(rets)
    years = max(n / 252.0, 1e-9)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1)
    bcagr = float((beq.iloc[-1] / beq.iloc[0]) ** (1 / years) - 1)
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    bsharpe = float(bh.mean() / bh.std() * np.sqrt(252)) if bh.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    bmdd = float((beq / beq.cummax() - 1).min())
    cagr_pres = (cagr / bcagr * 100.0) if bcagr else None
    mdd_imp = ((abs(bmdd) - abs(mdd)) / abs(bmdd) * 100.0) if bmdd < 0 else None
    turnover = position.diff().abs().fillna(float(position.iloc[0] if len(position) else 0))
    return {
        "total_return": float(eq.iloc[-1] / eq.iloc[0] - 1),
        "CAGR": cagr,
        "Sharpe": sharpe,
        "max_drawdown": mdd,
        "benchmark_CAGR": bcagr,
        "benchmark_Sharpe": bsharpe,
        "benchmark_MDD": bmdd,
        "cagr_preservation_pct": cagr_pres,
        "mdd_improvement_pct": mdd_imp,
        "long_exposure": float(position.mean()) if len(position) else 0.0,
        "trade_count": int((turnover > 0).sum()),
        "sessions": int(n),
        "score": score_components(sharpe, mdd_imp, cagr_pres),
    }


def run_nested_walkforward(common: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    p = paths()
    d = common.reset_index(drop=True)
    folds = slice_outer_folds(len(d))
    if not folds:
        return {"verdict": "QQQ_REGIME_COMMON_PERIOD_FAIL", "error": "no_folds", "n": len(d)}

    # verify OOS indices identical plan
    oos_indices = []
    for f in folds:
        oos_indices.extend(list(range(f["oos_start"], f["oos_end"])))
    index_hash = _sha_list(oos_indices)
    assert int((pd.to_datetime(d.iloc[oos_indices]["session_date"]) >= pd.Timestamp("2024-06-27")).sum()) == 0

    candidates = list(CANDIDATE_FEATURE_GROUPS.keys()) + ["DUAL_TREND_FILTER", "BUY_AND_HOLD"]
    fold_metrics: List[Dict[str, Any]] = []
    thr_rows: List[Dict[str, Any]] = []
    streams: Dict[str, List[pd.DataFrame]] = {c: [] for c in candidates}
    pred_rows: List[pd.DataFrame] = []

    for f in folds:
        train = d.iloc[f["train_start"] : f["train_end"]]
        val = d.iloc[f["val_start"] : f["val_end"]]
        oos = d.iloc[f["oos_start"] : f["oos_end"]].reset_index(drop=True)
        dates = pd.to_datetime(oos["session_date"])
        gidx = list(range(f["oos_start"], f["oos_end"]))

        # rules on OOS only
        for name, sig_fn in (("BUY_AND_HOLD", signal_buy_and_hold), ("DUAL_TREND_FILTER", signal_dual_trend)):
            sig = sig_fn(oos)
            bt = run_backtest(oos, sig, COST_BASE)
            bh = run_backtest(oos, signal_buy_and_hold(oos), COST_BASE)
            comp = compare_to_buyhold(oos, sig, COST_BASE)
            s = comp["strategy"]
            fold_metrics.append(
                {
                    "fold_id": f["fold_id"],
                    "candidate": name,
                    "total_return": s["total_return"],
                    "Sharpe": s["Sharpe"],
                    "CAGR": s["CAGR"],
                    "MDD": s["max_drawdown"],
                    "exposure": s["long_exposure"],
                    "trades": s["trade_count"],
                    "positive_fold": s["total_return"] > 0,
                    "cagr_preservation_pct": comp["cagr_preservation_pct"],
                    "mdd_improvement_pct": comp["mdd_improvement_pct"],
                    "threshold": None,
                }
            )
            part = pd.DataFrame(
                {
                    "fold_id": f["fold_id"],
                    "global_index": gidx,
                    "session_date": dates.values,
                    "candidate": name,
                    "signal": np.asarray(sig, float),
                    "position": bt["position"].values,
                    "net_ret": bt["net_ret"].values,
                    "benchmark_net_ret": bh["net_ret"].values,
                }
            )
            streams[name].append(part)
            pred_rows.append(part)

        # logistic variants
        for cand, extra in CANDIDATE_FEATURE_GROUPS.items():
            cols = _feature_cols(extra)
            # ensure columns exist
            pipe, p_va = fit_predict_proba(train, val, feature_cols=cols, seed=seed)
            thr_info = select_threshold(val, p_va, LR_THRESHOLDS, COST_BASE, 0.25, 5)
            thr = float(thr_info["selected_threshold"])
            thr_rows.append({"fold_id": f["fold_id"], "candidate": cand, "selected_threshold": thr})
            _, p_oos = fit_predict_proba(train, oos, feature_cols=cols, seed=seed)
            sig = pd.Series((p_oos >= thr).astype(float))
            bt = run_backtest(oos, sig, COST_BASE)
            bh = run_backtest(oos, signal_buy_and_hold(oos), COST_BASE)
            comp = compare_to_buyhold(oos, sig, COST_BASE)
            s = comp["strategy"]
            fold_metrics.append(
                {
                    "fold_id": f["fold_id"],
                    "candidate": cand,
                    "total_return": s["total_return"],
                    "Sharpe": s["Sharpe"],
                    "CAGR": s["CAGR"],
                    "MDD": s["max_drawdown"],
                    "exposure": s["long_exposure"],
                    "trades": s["trade_count"],
                    "positive_fold": s["total_return"] > 0,
                    "cagr_preservation_pct": comp["cagr_preservation_pct"],
                    "mdd_improvement_pct": comp["mdd_improvement_pct"],
                    "threshold": thr,
                }
            )
            part = pd.DataFrame(
                {
                    "fold_id": f["fold_id"],
                    "global_index": gidx,
                    "session_date": dates.values,
                    "candidate": cand,
                    "signal": np.asarray(sig, float),
                    "position": bt["position"].values,
                    "net_ret": bt["net_ret"].values,
                    "benchmark_net_ret": bh["net_ret"].values,
                    "threshold": thr,
                }
            )
            streams[cand].append(part)
            pred_rows.append(part)

    fold_df = pd.DataFrame(fold_metrics)
    thr_df = pd.DataFrame(thr_rows)
    pred_df = pd.concat(pred_rows, ignore_index=True)

    # verify identical indices across candidates
    base = streams["DUAL_TREND_FILTER"]
    base_cat = pd.concat(base, ignore_index=True)
    for name, parts in streams.items():
        cat = pd.concat(parts, ignore_index=True)
        assert list(cat["global_index"]) == list(base_cat["global_index"]), name

    ranking_rows = []
    cost_rows = []
    for name, parts in streams.items():
        cat = pd.concat(parts, ignore_index=True)
        agg = _aggregate_stream(cat["net_ret"], cat["benchmark_net_ret"], cat["position"])
        fm = fold_df[fold_df["candidate"] == name]
        pos_ratio = float(fm["positive_fold"].mean()) if len(fm) else 0.0
        # HIGH cost aggregate: scale is approximate via replaying signals with high cost on each fold
        high_rets = []
        low_rets = []
        for f in folds:
            oos = d.iloc[f["oos_start"] : f["oos_end"]].reset_index(drop=True)
            sub = cat[cat["fold_id"] == f["fold_id"]]
            sig = pd.Series(sub["signal"].values)
            high_rets.append(run_backtest(oos, sig, COST_HIGH)["net_ret"])
            low_rets.append(run_backtest(oos, sig, COST_LOW)["net_ret"])
        high_stream = pd.concat(high_rets, ignore_index=True)
        low_stream = pd.concat(low_rets, ignore_index=True)
        high_total = float((1 + high_stream.fillna(0)).cumprod().iloc[-1] - 1)
        low_total = float((1 + low_stream.fillna(0)).cumprod().iloc[-1] - 1)

        # top fold removal: drop best return fold, recompute mean return
        fold_rets = fm["total_return"].values
        if len(fold_rets) > 1:
            mask = np.ones(len(fold_rets), dtype=bool)
            mask[int(np.argmax(fold_rets))] = False
            top_removed_mean = float(np.mean(fold_rets[mask]))
            top_removed_pos = float(np.mean(fold_rets[mask] > 0))
        else:
            top_removed_mean = float(fold_rets[0]) if len(fold_rets) else 0.0
            top_removed_pos = float(fold_rets[0] > 0) if len(fold_rets) else 0.0

        thr_dist = None
        if name.startswith("LOGISTIC"):
            thr_dist = fm["threshold"].value_counts().to_dict()
            thr_dist = {str(k): int(v) for k, v in thr_dist.items()}

        eligible = (
            float(fm["total_return"].mean()) > 0
            and pos_ratio >= 0.60
            and 0.25 <= agg["long_exposure"] <= 0.90
            and np.isfinite(agg["score"])
            and agg["trade_count"] > 0
            and agg["total_return"] > 0
            and high_total > 0
        )
        reasons = []
        if not (float(fm["total_return"].mean()) > 0):
            reasons.append("mean_fold_return<=0")
        if pos_ratio < 0.60:
            reasons.append("positive_folds<60%")
        if not (0.25 <= agg["long_exposure"] <= 0.90):
            reasons.append("exposure_out_of_range")
        if not (agg["total_return"] > 0):
            reasons.append("base_return<=0")
        if not (high_total > 0):
            reasons.append("high_cost_return<=0")

        ranking_rows.append(
            {
                "candidate": name,
                "feature_group": name.replace("LOGISTIC_", "") if name.startswith("LOGISTIC") else name,
                **{k: agg[k] for k in agg},
                "positive_folds": int(fm["positive_fold"].sum()),
                "fold_count": int(len(fm)),
                "positive_fold_ratio": pos_ratio,
                "mean_fold_return": float(fm["total_return"].mean()),
                "worst_fold": float(fm["total_return"].min()),
                "best_fold": float(fm["total_return"].max()),
                "top_fold_removal_mean_return": top_removed_mean,
                "top_fold_removal_positive_ratio": top_removed_pos,
                "low_cost_total_return": low_total,
                "high_cost_total_return": high_total,
                "threshold_distribution": json.dumps(thr_dist) if thr_dist else None,
                "eligible": eligible,
                "ineligible_reason": ";".join(reasons) if reasons else None,
                "common_index_hash": index_hash,
            }
        )
        cost_rows.append({"candidate": name, "LOW": low_total, "BASE": agg["total_return"], "HIGH": high_total})

    rank_df = pd.DataFrame(ranking_rows)
    # selection among eligible only
    elig = rank_df[rank_df["eligible"]].copy()
    if len(elig):
        elig = elig.sort_values(
            by=["score", "Sharpe", "cagr_preservation_pct", "mdd_improvement_pct", "candidate"],
            ascending=[False, False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)
        elig["rank"] = np.arange(1, len(elig) + 1)
        selected = elig.iloc[0].to_dict()
    else:
        selected = None
        elig = elig

    # full table ranks for reporting
    rank_df = rank_df.sort_values(
        by=["eligible", "score", "Sharpe", "candidate"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rank_df["rank"] = np.arange(1, len(rank_df) + 1)

    # persist
    fold_df.to_csv(p.root / "models" / "qqq_regime_outer_fold_metrics.csv", index=False)
    thr_df.to_csv(p.root / "models" / "qqq_regime_threshold_selection.csv", index=False)
    pred_df.to_parquet(p.root / "models" / "qqq_regime_outer_fold_predictions.parquet", index=False)
    rank_df.to_csv(p.diag / "selection" / "candidate_ranking.csv", index=False)
    (p.root / "models" / "qqq_regime_candidate_ranking.json").write_text(
        json.dumps(rank_df.to_dict(orient="records"), indent=2, default=str) + "\n"
    )
    pd.DataFrame(cost_rows).to_csv(p.diag / "selection" / "cost_sensitivity.csv", index=False)

    return {
        "folds": folds,
        "n_folds": len(folds),
        "index_hash": index_hash,
        "fold_metrics": fold_df,
        "ranking": rank_df,
        "eligible_ranking": elig,
        "selected": selected,
        "streams": {k: pd.concat(v, ignore_index=True) for k, v in streams.items()},
        "oos_start": str(pd.Timestamp(d.iloc[folds[0]["oos_start"]]["session_date"]).date()),
        "oos_end": str(pd.Timestamp(d.iloc[folds[-1]["oos_end"] - 1]["session_date"]).date()),
        "oos_rows": len(oos_indices),
    }
