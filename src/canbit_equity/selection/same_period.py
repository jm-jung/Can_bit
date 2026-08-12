"""Corrected same-period candidate selection (v2) — no legacy overwrite."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from canbit_equity.backtest import run_backtest, summarize_equity
from canbit_equity.config import QQQConfig, paths
from canbit_equity.features import logistic_feature_columns
from canbit_equity.metrics import compare_to_buyhold
from canbit_equity.model import fit_predict_proba, select_threshold
from canbit_equity.strategies import (
    RULE_SIGNAL_FNS,
    signal_buy_and_hold,
    signal_dual_trend,
    signal_sma_200,
    signal_trend_vol,
)
from canbit_equity.walkforward import _slice_folds

REPO = Path(__file__).resolve().parents[3]
DIAG = REPO / "data/diagnostics/equity_etf_qqq_same_period_selection"
EXPECTED_LABELED = "5070477672a866749a2e2b7674a2029c1fb5b461e349bd3dc52d9e4e337d0bf5"
EXPECTED_CONFIG = "8be96beae42f273e79ca348ce9d4aa62101f6356fab041bb8ba41ba1b624a2b3"
EXPECTED_FEATURE = "c9594a79e468045f3879d9866b31adee3ba34bb0a04169d1cabec22ef916a9f8"
COST_BPS = 5.0
CANDIDATES = [
    "BUY_AND_HOLD",
    "SMA_200_FILTER",
    "DUAL_TREND_FILTER",
    "TREND_VOLATILITY_FILTER",
    "LOGISTIC_REGRESSION",
]


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, pd.DataFrame):
        if path.suffix == ".parquet":
            obj.to_parquet(path, index=False)
        else:
            obj.to_csv(path, index=False)
        return
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def ensure_diag_dirs() -> None:
    for sub in (
        "preflight",
        "inventory",
        "common_period",
        "folds",
        "candidates",
        "reconciliation",
        "tests",
        "reports",
        "backups",
    ):
        (DIAG / sub).mkdir(parents=True, exist_ok=True)


def score_from_components(sharpe: float, mdd_imp: Optional[float], cagr_pres: Optional[float]) -> float:
    return float(sharpe or 0.0) + 0.01 * float(mdd_imp or 0.0) + 0.001 * float(cagr_pres or 0.0)


def verify_fixed_input(cfg: QQQConfig = QQQConfig()) -> Dict[str, Any]:
    p = paths(cfg)
    labeled_path = p["features"] / "qqq_daily_features_labeled.parquet"
    labeled_hash = sha256_file(labeled_path)
    config_hash = sha256_file(REPO / "src/canbit_equity/config.py")
    feature_hash = sha256_bytes(",".join(logistic_feature_columns()).encode())
    labeled = pd.read_parquet(labeled_path).reset_index(drop=True)
    folds, holdout_idx = _slice_folds(len(labeled), cfg)
    feat_rows = len(pd.read_parquet(p["features_file"])) if p["features_file"].exists() else None
    ok = (
        labeled_hash == EXPECTED_LABELED
        and config_hash == EXPECTED_CONFIG
        and feature_hash == EXPECTED_FEATURE
        and len(labeled) == 6618
        and feat_rows is not None
        and feat_rows < 100  # stale intermediate must NOT be selected as input
    )
    if feat_rows is not None and feat_rows >= 1000 and labeled_hash != EXPECTED_LABELED:
        verdict = "FIXED_INPUT_HASH_MISMATCH"
    elif feat_rows == 68 and labeled_hash == EXPECTED_LABELED:
        verdict = "FIXED_INPUT_PASS"  # stale intermediate present but unused
    elif not ok:
        verdict = "FIXED_INPUT_HASH_MISMATCH"
    else:
        verdict = "FIXED_INPUT_PASS"
    out = {
        "verdict": verdict,
        "labeled_hash": labeled_hash,
        "config_hash": config_hash,
        "feature_hash": feature_hash,
        "labeled_rows": int(len(labeled)),
        "label_valid_end": str(pd.Timestamp(labeled["session_date"].iloc[-1]).date()),
        "development_end": str(pd.Timestamp(labeled["session_date"].iloc[holdout_idx["start"] - 1]).date()),
        "holdout_start": str(pd.Timestamp(labeled["session_date"].iloc[holdout_idx["start"]]).date()),
        "holdout_index": holdout_idx,
        "n_folds": int(len(folds)),
        "stale_features_parquet_rows": feat_rows,
        "stale_intermediate_used": False,
        "research_input": str(labeled_path.relative_to(REPO)),
        "ok": ok and verdict == "FIXED_INPUT_PASS",
    }
    dump(DIAG / "inventory/fixed_input_verification.json", out)
    return out


def write_legacy_inventory(cfg: QQQConfig = QQQConfig()) -> Dict[str, Any]:
    p = paths(cfg)
    wf = json.loads((p["reports"] / "qqq_walkforward_report.json").read_text())
    ranking = wf.get("development_ranking") or []
    rows = []
    for r in ranking:
        name = r["name"]
        if name == "LOGISTIC_REGRESSION":
            period = "dev_last_15pct"
            note = "score from compare_to_buyhold on last 15% of development after 70/15 train/val"
        else:
            period = "full_dev"
            note = "score from compare_to_buyhold on entire development before holdout"
        rows.append(
            {
                "candidate": name,
                "legacy_period_label": period,
                "legacy_Sharpe": r.get("Sharpe"),
                "legacy_mdd_improvement_pct": r.get("mdd_improvement_pct"),
                "legacy_cagr_preservation_pct": r.get("cagr_preservation_pct"),
                "legacy_score": score_from_components(
                    r.get("Sharpe"), r.get("mdd_improvement_pct"), r.get("cagr_preservation_pct")
                ),
                "note": note,
                "source": "walkforward.run_walkforward development_ranking",
            }
        )
    path_info = {
        "signal_generation": "canbit_equity.strategies",
        "logistic_fit": "canbit_equity.model.fit_predict_proba",
        "threshold_selection": "canbit_equity.model.select_threshold",
        "returns": "canbit_equity.backtest.run_backtest",
        "metrics": "canbit_equity.metrics.compare_to_buyhold",
        "ranking": "canbit_equity.walkforward.run_walkforward rank_key",
        "lock": "canbit_equity.walkforward.run_final_holdout",
        "unfairness": {
            "cause": "Rule candidates scored on full_dev; Logistic scored on last 15% of development only.",
            "same_score_function": True,
            "same_cost": True,
            "same_row_index": False,
            "fold_aggregates_used_for_ranking": False,
        },
        "selected_legacy": (wf.get("selected_candidate") or {}).get("name"),
    }
    dump(DIAG / "inventory/legacy_selection_code_path.json", path_info)
    dump(DIAG / "inventory/legacy_candidate_score_inputs.csv", pd.DataFrame(rows))
    (DIAG / "inventory/legacy_selection_code_path.md").write_text(
        "# Legacy Selection Code Path\n\n"
        "Ranking in `walkforward.run_walkforward` used **unequal score input windows**:\n"
        "- Rules (incl. DUAL): full development\n"
        "- Logistic: last 15% of development\n"
        "- Cost identical (BASE 5bps); fold OOS aggregates were reporting-only.\n"
    )
    return {"path": path_info, "rows": rows, "wf_selected": path_info["selected_legacy"]}


def build_common_oos_folds(
    labeled: pd.DataFrame, cfg: QQQConfig = QQQConfig()
) -> Tuple[List[Dict[str, Any]], pd.DataFrame, Dict[str, Any]]:
    d = labeled.reset_index(drop=True)
    folds, holdout_idx = _slice_folds(len(d), cfg)
    holdout_start = holdout_idx["start"]
    fold_rows = []
    index_parts: List[np.ndarray] = []
    for i, f in enumerate(folds):
        # OOS = existing walk-forward test window (LR outer OOS convention)
        oos_start, oos_end = int(f["test_start"]), int(f["test_end"])
        assert oos_end <= holdout_start, "OOS must not enter old holdout"
        idx = np.arange(oos_start, oos_end)
        dates = pd.to_datetime(d.iloc[oos_start:oos_end]["session_date"])
        fold_rows.append(
            {
                "fold_id": i,
                "train_start": int(f["train_start"]),
                "train_end": int(f["train_end"]),
                "threshold_validation_start": int(f["val_start"]),
                "threshold_validation_end": int(f["val_end"]),
                "OOS_start": oos_start,
                "OOS_end": oos_end,
                "OOS_rows": int(oos_end - oos_start),
                "OOS_date_start": str(dates.iloc[0].date()),
                "OOS_date_end": str(dates.iloc[-1].date()),
                "cost_scenario": "BASE_5bps",
            }
        )
        index_parts.append(idx)

    all_idx = np.concatenate(index_parts)
    # overlap audit
    overlaps = []
    for a in fold_rows:
        for b in fold_rows:
            if a["fold_id"] >= b["fold_id"]:
                continue
            sa, ea = a["OOS_start"], a["OOS_end"]
            sb, eb = b["OOS_start"], b["OOS_end"]
            if sa < eb and sb < ea:
                overlaps.append((a["fold_id"], b["fold_id"]))
    assert len(overlaps) == 0, f"unexpected OOS overlap: {overlaps}"

    # first-OOS-wins policy (deterministic; no duplicates here)
    unique_idx = all_idx  # already unique
    row_df = pd.DataFrame(
        {
            "global_index": unique_idx,
            "session_date": pd.to_datetime(d.iloc[unique_idx]["session_date"]).values,
            "fold_id": np.concatenate([[fr["fold_id"]] * fr["OOS_rows"] for fr in fold_rows]),
        }
    )
    # ensure no holdout rows
    assert int((row_df["global_index"] >= holdout_start).sum()) == 0

    index_hash = sha256_bytes(
        (",".join(str(int(x)) for x in unique_idx.tolist()) + "|" + ",".join(str(pd.Timestamp(x).date()) for x in row_df["session_date"])).encode()
    )
    meta = {
        "verdict": "COMMON_OOS_FOLDS_PASS",
        "n_folds": len(fold_rows),
        "common_oos_start": fold_rows[0]["OOS_date_start"],
        "common_oos_end": fold_rows[-1]["OOS_date_end"],
        "common_oos_rows": int(len(unique_idx)),
        "common_oos_index_hash": index_hash,
        "holdout_rows_in_common": 0,
        "overlap_pairs": overlaps,
        "overlap_policy": "first_fold_OOS_wins_chronological; tests are non-overlapping expanding windows",
        "fold_ids": [fr["fold_id"] for fr in fold_rows],
        "cost_scenario": "BASE_5bps",
    }
    return fold_rows, row_df, meta


def _bt_on_slice(df_slice: pd.DataFrame, signal: pd.Series, cost: float = COST_BPS) -> Dict[str, Any]:
    return compare_to_buyhold(df_slice.reset_index(drop=True), signal.reset_index(drop=True), cost)


def replay_all_candidates(
    labeled: pd.DataFrame,
    fold_defs: List[Dict[str, Any]],
    row_df: pd.DataFrame,
    cfg: QQQConfig = QQQConfig(),
) -> Dict[str, Any]:
    d = labeled.reset_index(drop=True)
    fold_metric_rows: List[Dict[str, Any]] = []
    thr_rows: List[Dict[str, Any]] = []
    stream_parts: Dict[str, List[pd.DataFrame]] = {c: [] for c in CANDIDATES}
    signal_hashes: Dict[str, str] = {}

    for fr in fold_defs:
        i = fr["fold_id"]
        train = d.iloc[fr["train_start"] : fr["train_end"]]
        val = d.iloc[fr["threshold_validation_start"] : fr["threshold_validation_end"]]
        test = d.iloc[fr["OOS_start"] : fr["OOS_end"]].copy()
        test_reset = test.reset_index(drop=True)
        dates = pd.to_datetime(test_reset["session_date"])

        # shared benchmark on this fold OOS
        bh_sig = signal_buy_and_hold(test_reset)
        bh_comp = _bt_on_slice(test_reset, bh_sig)

        # rules — evaluate only on OOS rows; signals computed from OOS frame columns (already past-only features)
        vol_cap = float(train["realized_vol_20"].quantile(0.95))
        rule_sigs = {
            "BUY_AND_HOLD": signal_buy_and_hold(test_reset),
            "SMA_200_FILTER": signal_sma_200(test_reset),
            "DUAL_TREND_FILTER": signal_dual_trend(test_reset),
            "TREND_VOLATILITY_FILTER": signal_trend_vol(test_reset, vol_cap),
        }
        for name, sig in rule_sigs.items():
            comp = _bt_on_slice(test_reset, sig)
            s = comp["strategy"]
            fold_metric_rows.append(
                {
                    "fold_id": i,
                    "candidate": name,
                    "total_return": s["total_return"],
                    "CAGR": s["CAGR"],
                    "Sharpe": s["Sharpe"],
                    "Sortino": s["Sortino"],
                    "MDD": s["max_drawdown"],
                    "Calmar": s["Calmar"],
                    "exposure": s["long_exposure"],
                    "turnover": s["turnover"],
                    "trades": s["trade_count"],
                    "positive_fold": bool(s["total_return"] > 0),
                    "benchmark_return": bh_comp["strategy"]["total_return"],
                    "benchmark_CAGR": bh_comp["strategy"]["CAGR"],
                    "benchmark_Sharpe": bh_comp["strategy"]["Sharpe"],
                    "benchmark_MDD": bh_comp["strategy"]["max_drawdown"],
                    "cagr_preservation_pct": comp["cagr_preservation_pct"],
                    "mdd_improvement_pct": comp["mdd_improvement_pct"],
                    "threshold": None,
                    "vol_cap": vol_cap if name == "TREND_VOLATILITY_FILTER" else None,
                }
            )
            bt = run_backtest(test_reset, sig, COST_BPS)
            part = pd.DataFrame(
                {
                    "fold_id": i,
                    "session_date": dates.values,
                    "global_index": np.arange(fr["OOS_start"], fr["OOS_end"]),
                    "candidate": name,
                    "signal": np.asarray(sig, dtype=float),
                    "position": bt["position"].values,
                    "net_ret": bt["net_ret"].values,
                    "benchmark_net_ret": run_backtest(test_reset, bh_sig, COST_BPS)["net_ret"].values,
                }
            )
            stream_parts[name].append(part)
            # accumulate signal hash pieces
            signal_hashes[name] = sha256_bytes(
                (signal_hashes.get(name, "") + "|" + ",".join(f"{x:.0f}" for x in np.asarray(sig))).encode()
            )

        # Logistic
        pipe, p_va = fit_predict_proba(train, val, seed=cfg.random_seed)
        thr_info = select_threshold(
            val,
            p_va,
            thresholds=cfg.lr_thresholds,
            cost_bps=COST_BPS,
            min_long_exposure=cfg.min_long_exposure,
            min_trades=cfg.min_trades_per_validation_year,
        )
        thr = float(thr_info["selected_threshold"])
        thr_rows.append(
            {
                "fold_id": i,
                "selected_threshold": thr,
                "constraint_relaxed": bool(thr_info.get("selected", {}).get("constraint_relaxed")),
                "val_start": fr["threshold_validation_start"],
                "val_end": fr["threshold_validation_end"],
                "OOS_start": fr["OOS_start"],
                "OOS_end": fr["OOS_end"],
            }
        )
        _, p_te = fit_predict_proba(train, test_reset, seed=cfg.random_seed)
        sig_lr = pd.Series((p_te >= thr).astype(float))
        comp_lr = _bt_on_slice(test_reset, sig_lr)
        s = comp_lr["strategy"]
        fold_metric_rows.append(
            {
                "fold_id": i,
                "candidate": "LOGISTIC_REGRESSION",
                "total_return": s["total_return"],
                "CAGR": s["CAGR"],
                "Sharpe": s["Sharpe"],
                "Sortino": s["Sortino"],
                "MDD": s["max_drawdown"],
                "Calmar": s["Calmar"],
                "exposure": s["long_exposure"],
                "turnover": s["turnover"],
                "trades": s["trade_count"],
                "positive_fold": bool(s["total_return"] > 0),
                "benchmark_return": bh_comp["strategy"]["total_return"],
                "benchmark_CAGR": bh_comp["strategy"]["CAGR"],
                "benchmark_Sharpe": bh_comp["strategy"]["Sharpe"],
                "benchmark_MDD": bh_comp["strategy"]["max_drawdown"],
                "cagr_preservation_pct": comp_lr["cagr_preservation_pct"],
                "mdd_improvement_pct": comp_lr["mdd_improvement_pct"],
                "threshold": thr,
                "vol_cap": None,
            }
        )
        bt = run_backtest(test_reset, sig_lr, COST_BPS)
        stream_parts["LOGISTIC_REGRESSION"].append(
            pd.DataFrame(
                {
                    "fold_id": i,
                    "session_date": dates.values,
                    "global_index": np.arange(fr["OOS_start"], fr["OOS_end"]),
                    "candidate": "LOGISTIC_REGRESSION",
                    "signal": np.asarray(sig_lr, dtype=float),
                    "position": bt["position"].values,
                    "net_ret": bt["net_ret"].values,
                    "benchmark_net_ret": run_backtest(test_reset, bh_sig, COST_BPS)["net_ret"].values,
                    "threshold": thr,
                }
            )
        )

    fold_df = pd.DataFrame(fold_metric_rows)
    thr_df = pd.DataFrame(thr_rows)
    streams = {k: pd.concat(v, ignore_index=True) for k, v in stream_parts.items()}

    # verify identical OOS index across candidates
    base_idx = streams["DUAL_TREND_FILTER"][["global_index", "session_date"]].copy()
    for name, st in streams.items():
        assert list(st["global_index"]) == list(base_idx["global_index"]), f"index mismatch {name}"
        assert len(st) == len(row_df)

    dump(DIAG / "candidates/rule_fold_metrics.csv", fold_df[fold_df["candidate"] != "LOGISTIC_REGRESSION"])
    dump(DIAG / "candidates/logistic_fold_metrics.csv", fold_df[fold_df["candidate"] == "LOGISTIC_REGRESSION"])
    dump(DIAG / "candidates/logistic_threshold_selection.csv", thr_df)
    dump(DIAG / "candidates/all_fold_metrics.csv", fold_df)
    dump(DIAG / "candidates/logistic_common_oos_stream.parquet", streams["LOGISTIC_REGRESSION"])
    dump(DIAG / "candidates/rule_common_oos_streams.parquet", pd.concat([streams[c] for c in CANDIDATES if c != "LOGISTIC_REGRESSION"], ignore_index=True))
    dump(DIAG / "candidates/rule_signal_hashes.json", signal_hashes)
    return {"fold_metrics": fold_df, "thresholds": thr_df, "streams": streams}


def aggregate_common_stream(stream: pd.DataFrame) -> Dict[str, Any]:
    """Build one chronological equity from concatenated non-overlapping OOS fold returns."""
    rets = stream["net_ret"].astype(float).reset_index(drop=True)
    bh = stream["benchmark_net_ret"].astype(float).reset_index(drop=True)
    eq = (1.0 + rets.fillna(0.0)).cumprod()
    beq = (1.0 + bh.fillna(0.0)).cumprod()
    n = len(rets)
    years = n / 252.0
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    cagr = float((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    btotal = float(beq.iloc[-1] / beq.iloc[0] - 1.0)
    bcagr = float((beq.iloc[-1] / beq.iloc[0]) ** (1 / years) - 1) if years > 0 else 0.0
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    bsharpe = float(bh.mean() / bh.std() * np.sqrt(252)) if bh.std() > 0 else 0.0
    dd = eq / eq.cummax() - 1.0
    bdd = beq / beq.cummax() - 1.0
    mdd = float(dd.min())
    bmdd = float(bdd.min())
    cagr_pres = (cagr / bcagr * 100.0) if bcagr not in (0, None) and np.isfinite(bcagr) else None
    mdd_imp = (abs(bmdd) - abs(mdd)) / abs(bmdd) * 100.0 if bmdd < 0 else None
    pos = stream["position"].astype(float)
    turnover = pos.diff().abs().fillna(pos.iloc[0])
    return {
        "common_rows": int(n),
        "total_return": total,
        "CAGR": cagr,
        "Sharpe": sharpe,
        "max_drawdown": mdd,
        "long_exposure": float(pos.mean()),
        "trade_count": int((turnover > 0).sum()),
        "benchmark_total_return": btotal,
        "benchmark_CAGR": bcagr,
        "benchmark_Sharpe": bsharpe,
        "benchmark_MDD": bmdd,
        "cagr_preservation_pct": cagr_pres,
        "mdd_improvement_pct": mdd_imp,
        "score": score_from_components(sharpe, mdd_imp, cagr_pres),
    }


def rank_candidates(
    streams: Dict[str, pd.DataFrame],
    fold_metrics: pd.DataFrame,
    fold_meta: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for name in CANDIDATES:
        agg = aggregate_common_stream(streams[name])
        fm = fold_metrics[fold_metrics["candidate"] == name]
        pos_folds = int(fm["positive_fold"].sum())
        thr_dist = None
        if name == "LOGISTIC_REGRESSION":
            thr_dist = fm["threshold"].value_counts().to_dict()
            thr_dist = {str(k): int(v) for k, v in thr_dist.items()}
        # Legacy eligibility: effectively none on ranking table — keep all finite-score candidates
        eligible = bool(np.isfinite(agg["score"]))
        rows.append(
            {
                "candidate": name,
                "common_start": fold_meta["common_oos_start"],
                "common_end": fold_meta["common_oos_end"],
                "common_rows": fold_meta["common_oos_rows"],
                "fold_ids": ",".join(str(x) for x in fold_meta["fold_ids"]),
                "Sharpe": agg["Sharpe"],
                "benchmark_Sharpe": agg["benchmark_Sharpe"],
                "MDD": agg["max_drawdown"],
                "benchmark_MDD": agg["benchmark_MDD"],
                "MDD_improvement_percent": agg["mdd_improvement_pct"],
                "CAGR": agg["CAGR"],
                "benchmark_CAGR": agg["benchmark_CAGR"],
                "CAGR_preservation_percent": agg["cagr_preservation_pct"],
                "exposure": agg["long_exposure"],
                "trades": agg["trade_count"],
                "total_return": agg["total_return"],
                "positive_folds": pos_folds,
                "fold_count": int(len(fm)),
                "logistic_threshold_distribution": json.dumps(thr_dist) if thr_dist else None,
                "eligible": eligible,
                "ineligible_reason": None if eligible else "non_finite_score",
                "score": agg["score"],
            }
        )
    df = pd.DataFrame(rows)
    # sort: score desc, Sharpe desc, then candidate_id asc (deterministic stability tie-break)
    df = df.sort_values(
        by=["score", "Sharpe", "candidate"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    selected = df.iloc[0].to_dict()
    return df, selected


def run_corrected_selection(cfg: QQQConfig = QQQConfig()) -> Dict[str, Any]:
    ensure_diag_dirs()
    fixed = verify_fixed_input(cfg)
    if not fixed["ok"]:
        return {
            "verdict": "SAME_PERIOD_SELECTION_CORRECTION_FAILED",
            "reason": "FIXED_INPUT_FAIL",
            "fixed": fixed,
            "current_action": "STOP_DUE_TO_INPUT_HASH_MISMATCH",
            "production_ready": False,
            "promotion_ready": False,
        }

    legacy = write_legacy_inventory(cfg)
    p = paths(cfg)
    labeled = pd.read_parquet(p["features"] / "qqq_daily_features_labeled.parquet").reset_index(drop=True)

    fold_defs, row_df, fold_meta = build_common_oos_folds(labeled, cfg)
    dump(DIAG / "folds/common_oos_folds.csv", pd.DataFrame(fold_defs))
    dump(DIAG / "folds/common_oos_row_index.parquet", row_df)
    dump(DIAG / "folds/common_oos_index_hash.json", fold_meta)
    (DIAG / "folds/fold_definition.md").write_text(
        f"# Common OOS Folds\n\n`{fold_meta['verdict']}`\n\n"
        f"- folds={fold_meta['n_folds']}\n"
        f"- common={fold_meta['common_oos_start']} → {fold_meta['common_oos_end']} ({fold_meta['common_oos_rows']} rows)\n"
        f"- index_hash={fold_meta['common_oos_index_hash']}\n"
        f"- holdout rows in common: 0\n"
        f"- overlap: none; expanding test windows abut without overlap\n"
    )
    dump(DIAG / "folds/fold_overlap_audit.json", {"overlaps": fold_meta["overlap_pairs"], "verdict": fold_meta["verdict"]})

    replay = replay_all_candidates(labeled, fold_defs, row_df, cfg)
    ranking_df, selected = rank_candidates(replay["streams"], replay["fold_metrics"], fold_meta)

    # common benchmark metrics from BH stream
    bh_agg = aggregate_common_stream(replay["streams"]["BUY_AND_HOLD"])
    dump(DIAG / "common_period/common_benchmark_metrics.json", bh_agg)
    dump(DIAG / "common_period/common_benchmark_stream.parquet", replay["streams"]["BUY_AND_HOLD"])
    dump(DIAG / "common_period/corrected_candidate_score_components.csv", ranking_df)
    dump(DIAG / "common_period/corrected_candidate_ranking.csv", ranking_df)
    dump(DIAG / "common_period/corrected_candidate_ranking.json", ranking_df.to_dict(orient="records"))

    previous = legacy["wf_selected"]
    changed = selected["candidate"] != previous
    if abs(float(ranking_df.iloc[0]["score"]) - float(ranking_df.iloc[1]["score"])) < 1e-12 and ranking_df.iloc[0]["Sharpe"] == ranking_df.iloc[1]["Sharpe"]:
        verdict = "SAME_PERIOD_SELECTION_CORRECTED_TIE"
        action = "STOP_DUE_TO_SAME_PERIOD_SELECTION_TIE"
    elif changed:
        verdict = "SAME_PERIOD_SELECTION_CORRECTED_CANDIDATE_CHANGED"
        action = "PRESERVE_RESULTS_AND_CREATE_NEW_PROSPECTIVE_SELECTION"
    else:
        verdict = "SAME_PERIOD_SELECTION_CORRECTED_DUAL_REMAINS"
        action = "PROCEED_TO_QQQ_REGIME_FEATURE_RESEARCH"

    # reconciliation
    legacy_map = {r["candidate"]: r for r in legacy["rows"]}
    recon_rows = []
    for _, row in ranking_df.iterrows():
        name = row["candidate"]
        leg = legacy_map.get(name, {})
        recon_rows.append(
            {
                "candidate": name,
                "legacy_period": leg.get("legacy_period_label"),
                "legacy_rows": None,
                "legacy_score": leg.get("legacy_score"),
                "corrected_period": f"{row['common_start']}→{row['common_end']}",
                "corrected_rows": row["common_rows"],
                "corrected_score": row["score"],
                "legacy_rank": None,
                "corrected_rank": int(row["rank"]),
                "selected_before": name == previous,
                "selected_after": name == selected["candidate"],
            }
        )
    # fill legacy ranks
    leg_sorted = sorted(legacy["rows"], key=lambda r: r["legacy_score"], reverse=True)
    for i, r in enumerate(leg_sorted, 1):
        for rr in recon_rows:
            if rr["candidate"] == r["candidate"]:
                rr["legacy_rank"] = i
                if r["candidate"] == "DUAL_TREND_FILTER":
                    rr["legacy_rows"] = 6114
                    rr["legacy_period"] = "2000-03-08→2024-06-26"
                elif r["candidate"] == "LOGISTIC_REGRESSION":
                    rr["legacy_rows"] = 918
                    rr["legacy_period"] = "2020-10-30→2024-06-26"
                else:
                    rr["legacy_rows"] = 6114
                    rr["legacy_period"] = "2000-03-08→2024-06-26"
    recon_df = pd.DataFrame(recon_rows).sort_values("corrected_rank")
    dump(DIAG / "reconciliation/legacy_vs_corrected_ranking.csv", recon_df)
    recon_json = {
        "previous_selected": previous,
        "corrected_selected": selected["candidate"],
        "candidate_changed": changed,
        "verdict": verdict,
        "note": "Do not interpret legacy vs corrected score magnitudes as performance improvement; periods differ by design for legacy.",
        "old_holdout_status": "SEEN_HISTORICAL_REFERENCE",
        "confirmatory_value": "NONE_FOR_CORRECTED_SELECTION",
        "used_for_selection": False,
        "used_for_score": False,
        "used_for_verdict": False,
    }
    dump(DIAG / "reconciliation/legacy_vs_corrected_selection.json", recon_json)
    (DIAG / "reconciliation/legacy_vs_corrected_selection.md").write_text(
        f"# Legacy vs Corrected\n\n`{verdict}`\n\n"
        f"- previous: `{previous}`\n- corrected: `{selected['candidate']}`\n"
        f"- changed: {changed}\n"
        f"- common OOS: {fold_meta['common_oos_start']} → {fold_meta['common_oos_end']} ({fold_meta['common_oos_rows']} rows)\n"
    )

    artifact = {
        "selection_version": "SAME_PERIOD_V2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_input_hash": fixed["labeled_hash"],
        "config_hash": fixed["config_hash"],
        "feature_hash": fixed["feature_hash"],
        "common_oos_start": fold_meta["common_oos_start"],
        "common_oos_end": fold_meta["common_oos_end"],
        "common_oos_rows": fold_meta["common_oos_rows"],
        "common_oos_index_hash": fold_meta["common_oos_index_hash"],
        "fold_ids": fold_meta["fold_ids"],
        "cost_scenario": "BASE_5bps",
        "score_formula": "Sharpe + 0.01*MDD_improvement_percent + 0.001*CAGR_preservation_percent",
        "selection_sort": ["score desc", "Sharpe desc", "candidate_id asc"],
        "candidate_table": ranking_df.to_dict(orient="records"),
        "selected_candidate": selected["candidate"],
        "selected_score": float(selected["score"]),
        "selected_metrics": {k: selected[k] for k in selected if k != "logistic_threshold_distribution"},
        "previous_selected_candidate": previous,
        "candidate_changed": changed,
        "old_holdout_used": False,
        "old_holdout_seen": True,
        "old_holdout_status": "SEEN_HISTORICAL_REFERENCE",
        "confirmatory_value": "NONE_FOR_CORRECTED_SELECTION",
        "network_used": False,
        "external_data_downloaded": False,
        "production_ready": False,
        "promotion_ready": False,
        "verdict": verdict,
        "current_action": action,
    }
    art_path = p["models"] / "qqq_corrected_same_period_selection_v2.json"
    # never overwrite legacy locks
    assert art_path.name != "final_holdout_evaluation_lock.json"
    dump(art_path, artifact)

    lr_row = ranking_df[ranking_df["candidate"] == "LOGISTIC_REGRESSION"].iloc[0]
    dual_row = ranking_df[ranking_df["candidate"] == "DUAL_TREND_FILTER"].iloc[0]

    compact = {
        "generated_at_utc": artifact["created_at_utc"],
        "verdict": verdict,
        "selection_version": "SAME_PERIOD_V2",
        "fixed_input_hash": fixed["labeled_hash"],
        "config_hash": fixed["config_hash"],
        "feature_hash": fixed["feature_hash"],
        "network_used": False,
        "external_data_downloaded": False,
        "common_oos_start": fold_meta["common_oos_start"],
        "common_oos_end": fold_meta["common_oos_end"],
        "common_oos_rows": fold_meta["common_oos_rows"],
        "common_oos_index_hash": fold_meta["common_oos_index_hash"],
        "common_folds": fold_meta["n_folds"],
        "common_cost": "BASE_5bps",
        "previous_selected_candidate": previous,
        "corrected_selected_candidate": selected["candidate"],
        "candidate_changed": changed,
        "dual_score": float(dual_row["score"]),
        "logistic_score": float(lr_row["score"]),
        "logistic_positive_folds": int(lr_row["positive_folds"]),
        "old_holdout_used_for_selection": False,
        "legacy_artifacts_overwritten": False,
        "corrected_artifact": str(art_path.relative_to(REPO)),
        "original_qqq_verdict": "QQQ_BASELINE_RESEARCH_INCONCLUSIVE",
        "corrected_baseline_status": (
            "DUAL_REMAINS_INCONCLUSIVE_HISTORICAL_REFERENCE"
            if not changed
            else "CANDIDATE_CHANGED_UNCONFIRMED_NO_HOLDOUT"
        ),
        "prospective_holdout_required": bool(changed),
        "regime_same_period_gate": None,
        "production_ready": False,
        "promotion_ready": False,
        "current_action": action,
    }

    report = {
        **compact,
        "ranking": ranking_df.to_dict(orient="records"),
        "reconciliation": recon_json,
        "fold_meta": fold_meta,
        "legacy_unfairness": legacy["path"]["unfairness"],
    }
    dump(DIAG / "reports/qqq_same_period_selection_final_report.json", report)
    dump(DIAG / "reports/qqq_same_period_selection_compact.json", compact)
    (DIAG / "reports/qqq_same_period_selection_final_report.md").write_text(
        f"""# QQQ Same-Period Selection Correction

## Verdict
`{verdict}`

## Purpose
Remove unfair full_dev vs last-15% ranking; re-rank all baseline candidates on identical walk-forward OOS rows.

## Fixed input
- labeled hash: `{fixed['labeled_hash']}`
- common OOS: {fold_meta['common_oos_start']} → {fold_meta['common_oos_end']} ({fold_meta['common_oos_rows']} rows, {fold_meta['n_folds']} folds)
- cost: BASE 5 bps/side
- old holdout used: **false**

## Corrected ranking
```
{ranking_df[['rank','candidate','score','Sharpe','MDD_improvement_percent','CAGR_preservation_percent','exposure','positive_folds']].to_string(index=False)}
```

## Selection
- previous: `{previous}`
- corrected: `{selected['candidate']}`
- changed: {changed}

## Holdout policy
old_holdout_status=SEEN_HISTORICAL_REFERENCE · confirmatory_value=NONE_FOR_CORRECTED_SELECTION

## CURRENT ACTION
`{action}`

production_ready=false · promotion_ready=false
"""
    )

    return {
        "verdict": verdict,
        "compact": compact,
        "artifact": artifact,
        "ranking": ranking_df,
        "fold_meta": fold_meta,
        "current_action": action,
        "production_ready": False,
        "promotion_ready": False,
    }
