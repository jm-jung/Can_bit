"""Logistic Regression baseline (train-only scaler/imputer)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import logistic_feature_columns


def make_pipeline(seed: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    random_state=seed,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def fit_predict_proba(
    train: pd.DataFrame,
    apply: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    seed: int = 42,
) -> Tuple[Pipeline, np.ndarray]:
    cols = feature_cols or logistic_feature_columns()
    pipe = make_pipeline(seed)
    y = train[f"label_long_20d"].astype(int).values
    Xtr = train[cols]
    pipe.fit(Xtr, y)
    proba = pipe.predict_proba(apply[cols])[:, 1]
    return pipe, proba


def select_threshold(
    validation: pd.DataFrame,
    proba: np.ndarray,
    thresholds: Tuple[float, ...] = (0.45, 0.50, 0.55, 0.60),
    cost_bps: float = 5.0,
    min_long_exposure: float = 0.25,
    min_trades: int = 5,
) -> Dict[str, Any]:
    from .backtest import run_backtest, summarize_equity

    best = None
    rows = []
    for thr in thresholds:
        sig = pd.Series((proba >= thr).astype(float), index=validation.index)
        bt = run_backtest(validation, sig, cost_bps_per_side=cost_bps)
        s = summarize_equity(bt)
        years = max(s["sessions"] / 252.0, 1e-9)
        ok = s["long_exposure"] >= min_long_exposure and s["trade_count"] >= max(min_trades, int(5 * years * 0.5))
        row = {"threshold": thr, **s, "eligible": ok}
        rows.append(row)
        if not ok:
            continue
        score = (s["Sharpe"], -abs(s["max_drawdown"]), s["CAGR"])
        if best is None or score > best["score"]:
            best = {"threshold": thr, "score": score, "metrics": s}
    if best is None:
        # fallback: highest sharpe among all even if constraints fail — mark ineligible
        rows_sorted = sorted(rows, key=lambda r: (r["Sharpe"], -abs(r["max_drawdown"])), reverse=True)
        best = {"threshold": rows_sorted[0]["threshold"], "score": None, "metrics": rows_sorted[0], "constraint_relaxed": True}
    return {"selected_threshold": best["threshold"], "candidates": rows, "selected": best}
