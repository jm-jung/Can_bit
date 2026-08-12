"""Fixed regime features (past-only)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from canbit_equity.features import logistic_feature_columns
from canbit_equity.regime.alignment import build_aligned_panel
from canbit_equity.regime.config import ALL_REGIME_FEATURES, REGIME_FEATURES, paths


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _rel_ret(a: pd.Series, b: pd.Series, n: int = 20) -> pd.Series:
    ratio = a / b.replace(0, np.nan)
    return ratio / ratio.shift(n) - 1.0


def build_regime_features(
    qqq_labeled: pd.DataFrame,
    *,
    persist: bool = True,
    output_feat_path: Path | None = None,
    output_manifest_path: Path | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = paths()
    qqq = qqq_labeled.copy()
    qqq["session_date"] = pd.to_datetime(qqq["session_date"]).dt.normalize()
    panel, align_audit = build_aligned_panel(qqq)
    panel["session_date"] = pd.to_datetime(panel["session_date"]).dt.normalize()

    f = panel.copy()
    f["vix_level"] = f["vix"]
    f["vix_change_5d"] = f["vix"] / f["vix"].shift(5) - 1.0
    sma20 = f["vix"].rolling(20, min_periods=20).mean()
    f["vix_sma20_ratio"] = f["vix"] / sma20 - 1.0
    f["dgs10_change_20d"] = f["dgs10"] - f["dgs10"].shift(20)
    f["dgs2_change_20d"] = f["dgs2"] - f["dgs2"].shift(20)
    f["t10y2y_level"] = f["t10y2y"]
    f["rsp_spy_relative_return_20d"] = _rel_ret(f["rsp"], f["spy"], 20)
    # qqq close from labeled join
    qclose = qqq.set_index("session_date")["close_adj"]
    f = f.set_index("session_date")
    f["qqq_close"] = qclose.reindex(f.index)
    f["qew_qqq_relative_return_20d"] = _rel_ret(f["qew"], f["qqq_close"], 20)
    f["smh_qqq_relative_return_20d"] = _rel_ret(f["smh"], f["qqq_close"], 20)
    f["hyg_ief_relative_return_20d"] = _rel_ret(f["hyg"], f["ief"], 20)
    f = f.reset_index()

    feat_cols = ALL_REGIME_FEATURES
    keep = ["session_date"] + feat_cols
    out = f[keep].copy()
    # merge onto qqq labeled rows only (left join qqq)
    merged = qqq.merge(out, on="session_date", how="left")

    meta_rows = []
    for g, cols in REGIME_FEATURES.items():
        for c in cols:
            s = merged[c]
            first = s.first_valid_index()
            last = s.last_valid_index()
            meta_rows.append(
                {
                    "feature_name": c,
                    "group": g,
                    "lookback": 20 if "20" in c or c.endswith("_20d") or "sma20" in c else (5 if "5d" in c else 0),
                    "availability_lag": 1 if c.startswith("dgs") or c.startswith("t10") else 0,
                    "first_valid_session": str(pd.Timestamp(merged.loc[first, "session_date"]).date()) if first is not None else None,
                    "last_valid_session": str(pd.Timestamp(merged.loc[last, "session_date"]).date()) if last is not None else None,
                    "missing_rows": int(s.isna().sum()),
                    "feature_hash": _sha(c + "|" + g),
                }
            )
    feature_manifest = {
        "features": meta_rows,
        "all_regime_features": ALL_REGIME_FEATURES,
        "price_only_features": logistic_feature_columns(),
        "alignment_verdict": align_audit.get("verdict"),
        "feature_set_hash": _sha("|".join(ALL_REGIME_FEATURES + logistic_feature_columns())),
    }
    feat_path = Path(output_feat_path) if output_feat_path else (p.root / "features" / "qqq_regime_features.parquet")
    # store only regime columns + session + label helpers from qqq
    save_cols = ["session_date", "open_adj", "close_adj", "label_long_20d"] + logistic_feature_columns() + ALL_REGIME_FEATURES
    # include rule helper cols
    for extra in ("sma_50", "sma_200", "ret_20d", "ret_60d", "realized_vol_20"):
        if extra in merged.columns and extra not in save_cols:
            save_cols.append(extra)
    # de-dupe columns while preserving order
    save_cols = list(dict.fromkeys(save_cols))
    if persist:
        feat_path.parent.mkdir(parents=True, exist_ok=True)
        write_df = merged.loc[:, ~merged.columns.duplicated()]
        write_df = write_df[[c for c in save_cols if c in write_df.columns]]
        write_df.to_parquet(feat_path, index=False)
        man_path = Path(output_manifest_path) if output_manifest_path else (p.root / "manifests" / "qqq_regime_feature_manifest.json")
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(feature_manifest, indent=2, default=str) + "\n")
        feature_manifest["path"] = str(feat_path)
    feature_manifest["align_audit"] = align_audit
    return merged, feature_manifest
