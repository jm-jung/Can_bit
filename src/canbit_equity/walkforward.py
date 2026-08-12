"""Walk-forward development + single final holdout evaluation."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .backtest import run_backtest, summarize_equity
from .config import QQQConfig, paths
from .features import logistic_feature_columns
from .metrics import compare_to_buyhold
from .model import fit_predict_proba, select_threshold
from .strategies import RULE_SIGNAL_FNS, signal_trend_vol


def _slice_folds(n: int, cfg: QQQConfig) -> Tuple[List[Dict[str, int]], Dict[str, int]]:
    holdout = cfg.final_holdout_sessions
    if n <= holdout + cfg.minimum_training_sessions + cfg.validation_sessions:
        # shrink
        holdout = min(holdout, max(126, n // 5))
    holdout_start = n - holdout
    dev_end = holdout_start
    folds = []
    train_end = cfg.minimum_training_sessions
    while True:
        val_start = train_end
        val_end = val_start + cfg.validation_sessions
        test_start = val_end
        test_end = test_start + cfg.test_sessions
        if test_end > dev_end:
            break
        folds.append(
            {
                "train_start": 0,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "test_start": test_start,
                "test_end": test_end,
            }
        )
        train_end += cfg.step_sessions
        if train_end >= dev_end:
            break
    holdout_idx = {"start": holdout_start, "end": n}
    return folds, holdout_idx


def evaluate_rules(df: pd.DataFrame, cost_bps: float, cfg: QQQConfig) -> Dict[str, Any]:
    results = {}
    for name, fn in RULE_SIGNAL_FNS.items():
        sig = fn(df)
        comp = compare_to_buyhold(df, sig, cost_bps)
        results[name] = {
            "strategy": comp["strategy"],
            "buy_hold": comp["buy_hold"],
            "cagr_preservation_pct": comp["cagr_preservation_pct"],
            "mdd_improvement_pct": comp["mdd_improvement_pct"],
        }
    # trend vol with train-less expanding cap on full df for rules-only simple pass:
    # for fair WF, vol cap computed later per fold; here use expanding 95% past-only
    vol = df["realized_vol_20"]
    cap = vol.expanding(min_periods=60).quantile(0.95)
    sig = ((df["close_adj"] > df["sma_200"]) & (df["sma_50"] > df["sma_200"]) & (df["ret_60d"] > 0) & (vol <= cap)).astype(float)
    comp = compare_to_buyhold(df, sig, cost_bps)
    results["TREND_VOLATILITY_FILTER"] = {
        "strategy": comp["strategy"],
        "buy_hold": comp["buy_hold"],
        "cagr_preservation_pct": comp["cagr_preservation_pct"],
        "mdd_improvement_pct": comp["mdd_improvement_pct"],
    }
    return results


def run_walkforward(df: pd.DataFrame, cfg: QQQConfig = QQQConfig(), cost_bps: float = 5.0) -> Dict[str, Any]:
    d = df.reset_index(drop=True)
    folds, holdout_idx = _slice_folds(len(d), cfg)
    fold_rows = []
    rule_scores: Dict[str, List[float]] = {k: [] for k in list(RULE_SIGNAL_FNS) + ["TREND_VOLATILITY_FILTER", "LOGISTIC_REGRESSION"]}

    for i, f in enumerate(folds):
        train = d.iloc[f["train_start"] : f["train_end"]]
        val = d.iloc[f["val_start"] : f["val_end"]]
        test = d.iloc[f["test_start"] : f["test_end"]]
        vol_cap = float(train["realized_vol_20"].quantile(0.95))

        fold_result = {"fold": i, **f, "vol_cap": vol_cap, "strategies": {}}
        for name, fn in RULE_SIGNAL_FNS.items():
            sig = fn(test)
            s = summarize_equity(run_backtest(test, sig, cost_bps))
            fold_result["strategies"][name] = s
            rule_scores[name].append(s["total_return"])

        sig_tv = signal_trend_vol(test, vol_cap)
        s_tv = summarize_equity(run_backtest(test, sig_tv, cost_bps))
        fold_result["strategies"]["TREND_VOLATILITY_FILTER"] = s_tv
        rule_scores["TREND_VOLATILITY_FILTER"].append(s_tv["total_return"])

        # logistic: fit train, threshold on val, apply test
        try:
            pipe, proba_val = fit_predict_proba(train, val, seed=cfg.random_seed)
            thr_info = select_threshold(
                val,
                proba_val,
                thresholds=cfg.lr_thresholds,
                cost_bps=cost_bps,
                min_long_exposure=cfg.min_long_exposure,
                min_trades=cfg.min_trades_per_validation_year,
            )
            thr = thr_info["selected_threshold"]
            _, proba_test = fit_predict_proba(train, test, seed=cfg.random_seed)
            sig_lr = pd.Series((proba_test >= thr).astype(float), index=test.index)
            s_lr = summarize_equity(run_backtest(test, sig_lr, cost_bps))
            fold_result["strategies"]["LOGISTIC_REGRESSION"] = {**s_lr, "threshold": thr}
            rule_scores["LOGISTIC_REGRESSION"].append(s_lr["total_return"])
        except Exception as exc:
            fold_result["strategies"]["LOGISTIC_REGRESSION"] = {"error": type(exc).__name__}
            rule_scores["LOGISTIC_REGRESSION"].append(np.nan)

        fold_rows.append(fold_result)

    # pick candidate on development aggregate (mean sharpe proxy via mean total return and avg metrics)
    agg = {}
    for name, rets in rule_scores.items():
        arr = np.array([x for x in rets if x is not None and np.isfinite(x)], dtype=float)
        agg[name] = {
            "fold_count": int(len(arr)),
            "mean_test_total_return": float(np.nanmean(arr)) if len(arr) else None,
            "positive_fold_ratio": float(np.mean(arr > 0)) if len(arr) else None,
        }

    # recompute development period metrics for ranking on last available contiguous? Use mean_test_total_return
    # Prefer dual trend / sma200 / logistic by economic score on full development (pre-holdout)
    dev = d.iloc[: holdout_idx["start"]]
    ranking = []
    for name in ["SMA_200_FILTER", "DUAL_TREND_FILTER", "TREND_VOLATILITY_FILTER", "LOGISTIC_REGRESSION", "BUY_AND_HOLD"]:
        if name == "LOGISTIC_REGRESSION":
            # fit on first 70% of dev, threshold on next 15%, score remaining 15% of dev for selection only
            n = len(dev)
            tr = dev.iloc[: int(n * 0.7)]
            va = dev.iloc[int(n * 0.7) : int(n * 0.85)]
            te = dev.iloc[int(n * 0.85) :]
            if len(tr) < 200 or len(va) < 50 or len(te) < 50:
                continue
            pipe, p_va = fit_predict_proba(tr, va, seed=cfg.random_seed)
            thr_info = select_threshold(va, p_va, cfg.lr_thresholds, cost_bps, cfg.min_long_exposure, cfg.min_trades_per_validation_year)
            thr = thr_info["selected_threshold"]
            _, p_te = fit_predict_proba(tr, te, seed=cfg.random_seed)
            sig = pd.Series((p_te >= thr).astype(float), index=te.index)
            comp = compare_to_buyhold(te, sig, cost_bps)
            ranking.append({"name": name, "threshold": thr, **comp["strategy"], "cagr_preservation_pct": comp["cagr_preservation_pct"], "mdd_improvement_pct": comp["mdd_improvement_pct"]})
        elif name == "TREND_VOLATILITY_FILTER":
            vol_cap = float(dev["realized_vol_20"].iloc[: int(len(dev) * 0.7)].quantile(0.95))
            sig = signal_trend_vol(dev, vol_cap)
            comp = compare_to_buyhold(dev, sig, cost_bps)
            ranking.append({"name": name, "threshold": None, "vol_cap": vol_cap, **comp["strategy"], "cagr_preservation_pct": comp["cagr_preservation_pct"], "mdd_improvement_pct": comp["mdd_improvement_pct"]})
        else:
            sig = RULE_SIGNAL_FNS[name](dev)
            comp = compare_to_buyhold(dev, sig, cost_bps)
            ranking.append({"name": name, "threshold": None, **comp["strategy"], "cagr_preservation_pct": comp["cagr_preservation_pct"], "mdd_improvement_pct": comp["mdd_improvement_pct"]})

    def rank_key(r):
        sharpe_edge = (r.get("Sharpe") or 0) - 0  # absolute
        mdd_imp = r.get("mdd_improvement_pct") or 0
        cagr_pres = r.get("cagr_preservation_pct") or 0
        return (sharpe_edge + 0.01 * mdd_imp + 0.001 * cagr_pres, r.get("Sharpe") or -999)

    ranking_sorted = sorted(ranking, key=rank_key, reverse=True)
    selected = ranking_sorted[0] if ranking_sorted else {"name": "SMA_200_FILTER", "threshold": None}

    out = {
        "folds": fold_rows,
        "fold_aggregate": agg,
        "development_ranking": ranking_sorted,
        "selected_candidate": selected,
        "holdout_index": holdout_idx,
        "n_rows": len(d),
        "cost_bps_per_side": cost_bps,
        "production_ready": False,
        "promotion_ready": False,
    }
    p = paths(cfg)
    (p["reports"] / "qqq_walkforward_report.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    (p["diag"].joinpath("walkforward") / "qqq_walkforward_report.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    return out


def run_final_holdout(df: pd.DataFrame, wf: Dict[str, Any], cfg: QQQConfig = QQQConfig(), cost_bps: float = 5.0) -> Dict[str, Any]:
    p = paths(cfg)
    lock_path = p["holdout_lock"]
    d = df.reset_index(drop=True)
    h = wf["holdout_index"]
    selected = wf["selected_candidate"]
    holdout = d.iloc[h["start"] : h["end"]].reset_index(drop=True)
    train_dev = d.iloc[: h["start"]]

    config_hash = str(hash(json.dumps(cfg.to_dict(), sort_keys=True)))
    feature_hash = str(hash(tuple(logistic_feature_columns())))
    lock_payload = {
        "candidate_id": selected.get("name"),
        "feature_set_hash": feature_hash,
        "config_hash": config_hash,
        "holdout_start": str(pd.Timestamp(holdout["session_date"].iloc[0]).date()) if len(holdout) else None,
        "holdout_end": str(pd.Timestamp(holdout["session_date"].iloc[-1]).date()) if len(holdout) else None,
        "threshold": selected.get("threshold"),
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    if lock_path.exists():
        prev = json.loads(lock_path.read_text())
        # allow identical re-run; block different candidate re-search
        if prev.get("candidate_id") != lock_payload["candidate_id"] or prev.get("threshold") != lock_payload["threshold"]:
            # If already evaluated a different candidate, do not switch — reuse previous lock result path if present
            if Path(prev.get("result_path") or "").exists():
                return json.loads(Path(prev["result_path"]).read_text())

    name = selected.get("name")
    if name == "LOGISTIC_REGRESSION":
        # fit on all development, use selected threshold
        thr = float(selected.get("threshold") or 0.5)
        # use last 15% of dev as implicit val already done; fit on full dev
        pipe, proba = fit_predict_proba(train_dev, holdout, seed=cfg.random_seed)
        sig = pd.Series((proba >= thr).astype(float), index=holdout.index)
    elif name == "TREND_VOLATILITY_FILTER":
        vol_cap = float(selected.get("vol_cap") or train_dev["realized_vol_20"].quantile(0.95))
        sig = signal_trend_vol(holdout, vol_cap)
    else:
        sig = RULE_SIGNAL_FNS[name](holdout)

    costs = {
        "LOW": cfg.cost_bps_per_side_low,
        "BASE": cfg.cost_bps_per_side_base,
        "HIGH": cfg.cost_bps_per_side_high,
    }
    cost_results = {}
    for label, bps in costs.items():
        cost_results[label] = compare_to_buyhold(holdout, sig, bps)

    base = cost_results["BASE"]
    # top fold removal sensitivity from wf
    fold_rets = []
    for fr in wf.get("folds", []):
        s = fr.get("strategies", {}).get(name) or {}
        if "total_return" in s:
            fold_rets.append(s["total_return"])
    arr = np.array(fold_rets, dtype=float)
    if len(arr) >= 2:
        without_top = float(np.mean(np.delete(arr, np.nanargmax(arr))))
        top_fold_removal = {"mean_with_all": float(np.nanmean(arr)), "mean_without_top": without_top}
    else:
        top_fold_removal = {"mean_with_all": float(np.nanmean(arr)) if len(arr) else None, "mean_without_top": None}

    # recent windows on full labeled history for reference
    def window_metrics(years: int):
        n = int(years * 252)
        if len(d) < n:
            return None
        w = d.iloc[-n:]
        if name == "LOGISTIC_REGRESSION":
            return None
        if name == "TREND_VOLATILITY_FILTER":
            sig_w = signal_trend_vol(w, float(train_dev["realized_vol_20"].quantile(0.95)))
        else:
            sig_w = RULE_SIGNAL_FNS[name](w)
        return compare_to_buyhold(w, sig_w, cfg.cost_bps_per_side_base)["strategy"]

    result = {
        "selected_candidate": name,
        "selected_threshold": selected.get("threshold"),
        "holdout_start": lock_payload["holdout_start"],
        "holdout_end": lock_payload["holdout_end"],
        "holdout_sessions": int(len(holdout)),
        "base": {
            "strategy": base["strategy"],
            "buy_hold": base["buy_hold"],
            "cagr_preservation_pct": base["cagr_preservation_pct"],
            "mdd_improvement_pct": base["mdd_improvement_pct"],
        },
        "cost_sensitivity": {
            k: {
                "strategy": v["strategy"],
                "buy_hold": v["buy_hold"],
                "cagr_preservation_pct": v["cagr_preservation_pct"],
                "mdd_improvement_pct": v["mdd_improvement_pct"],
            }
            for k, v in cost_results.items()
        },
        "top_fold_removal": top_fold_removal,
        "recent_5y": window_metrics(5),
        "recent_10y": window_metrics(10),
        "positive_fold_ratio": (wf.get("fold_aggregate") or {}).get(name, {}).get("positive_fold_ratio"),
        "production_ready": False,
        "promotion_ready": False,
    }
    result_path = p["reports"] / "qqq_final_holdout_report.json"
    result_path.write_text(json.dumps(result, indent=2, default=str) + "\n")
    lock_payload["result_path"] = str(result_path)
    lock_path.write_text(json.dumps(lock_payload, indent=2) + "\n")

    md = [
        "# QQQ Final Holdout Report",
        "",
        f"- candidate: `{name}`",
        f"- threshold: `{selected.get('threshold')}`",
        f"- holdout: {lock_payload['holdout_start']} → {lock_payload['holdout_end']}",
        f"- strategy CAGR: {base['strategy']['CAGR']:.4f}",
        f"- buyhold CAGR: {base['buy_hold']['CAGR']:.4f}",
        f"- strategy Sharpe: {base['strategy']['Sharpe']:.4f}",
        f"- buyhold Sharpe: {base['buy_hold']['Sharpe']:.4f}",
        f"- strategy MDD: {base['strategy']['max_drawdown']:.4f}",
        f"- buyhold MDD: {base['buy_hold']['max_drawdown']:.4f}",
        "",
        "Holdout evaluated once after candidate selection. production_ready=false, promotion_ready=false.",
    ]
    (p["reports"] / "qqq_final_holdout_report.md").write_text("\n".join(md) + "\n")
    return result


def research_verdict(wf: Dict[str, Any], holdout: Dict[str, Any], data_quality: str, lookahead: str) -> str:
    if data_quality == "QQQ_DATA_PIPELINE_FAIL" or lookahead == "LOOKAHEAD_AUDIT_FAIL":
        return "QQQ_BASELINE_RESEARCH_REJECT"
    base = holdout["base"]
    s = base["strategy"]
    b = base["buy_hold"]
    if s["total_return"] <= 0:
        return "QQQ_BASELINE_RESEARCH_REJECT"
    sharpe_edge = s["Sharpe"] - b["Sharpe"]
    mdd_imp = base.get("mdd_improvement_pct") or 0
    cagr_pres = base.get("cagr_preservation_pct") or 0
    high = holdout["cost_sensitivity"]["HIGH"]["strategy"]["total_return"]
    pos_fold = holdout.get("positive_fold_ratio") or 0
    exposure = s.get("long_exposure") or 0
    if (
        (sharpe_edge >= 0.15 or mdd_imp >= 25)
        and (mdd_imp < 25 or cagr_pres >= 70)
        and high > 0
        and pos_fold >= 0.5
        and 0.05 < exposure < 0.98
        and s.get("trade_count", 0) >= 3
    ):
        # top fold removal sanity
        tfr = holdout.get("top_fold_removal") or {}
        if tfr.get("mean_without_top") is not None and tfr["mean_without_top"] < -0.05 and (tfr.get("mean_with_all") or 0) > 0:
            return "QQQ_BASELINE_RESEARCH_INCONCLUSIVE"
        return "QQQ_BASELINE_RESEARCH_PROMISING"
    if s["Sharpe"] < b["Sharpe"] and abs(s["max_drawdown"]) >= abs(b["max_drawdown"]) and high <= 0:
        return "QQQ_BASELINE_RESEARCH_REJECT"
    return "QQQ_BASELINE_RESEARCH_INCONCLUSIVE"
