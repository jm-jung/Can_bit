"""Daily QQQ + external regime source update and live feature build (no model retrain)."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from canbit_equity.config import QQQConfig
from canbit_equity.data_provider import download_qqq_daily
from canbit_equity.features import build_features, logistic_feature_columns
from canbit_equity.regime.config import ALL_REGIME_FEATURES
from canbit_equity.regime.features import build_regime_features
from canbit_equity.regime.prospective.config import DIAG_ROOT, PROSP_ROOT, ensure_dirs
from canbit_equity.regime.prospective.file_lock import atomic_write_json
from canbit_equity.regime.prospective.research_lock import audit_external_data
from canbit_equity.regime.providers import update_all_external


LIVE_FEAT = PROSP_ROOT / "cache/live_regime_features.parquet"
PREV_EXT_HASH = PROSP_ROOT / "cache/previous_external_manifest_hash.txt"
REVISION_LOG = PROSP_ROOT / "cache/source_revisions.jsonl"
HIST_FEAT = PROSP_ROOT.parents[0] / "features/qqq_regime_features.parquet"


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_labels_keep_tail(features: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    """Same label timing as research, but keep unlabeled incomplete-horizon tail."""
    d = features.copy().reset_index(drop=True)
    cost = 2.0 * (cost_bps / 10000.0)
    open_ = d["open_adj"].astype(float)
    close = d["close_adj"].astype(float)
    for h in (5, 10, 20):
        entry = open_.shift(-1)
        exit_ = close.shift(-h)
        gross = exit_ / entry - 1.0
        net = gross - cost
        d[f"forward_return_{h}d_gross"] = gross
        d[f"forward_return_{h}d_net"] = net
        d[f"label_long_{h}d"] = np.where(net.notna(), (net > 0).astype(float), np.nan)
    return d


def restore_historical_regime_features_from_labeled() -> str:
    """Rebuild immutable historical research feature table from frozen labeled input."""
    from canbit_equity.config import paths as qqq_paths

    labeled = pd.read_parquet(qqq_paths()["features"] / "qqq_daily_features_labeled.parquet")
    build_regime_features(labeled, persist=True)
    return _sha_file(HIST_FEAT)


def update_sources_and_features() -> Dict[str, Any]:
    ensure_dirs()
    hist_hash_before = _sha_file(HIST_FEAT) if HIST_FEAT.exists() else None

    qqq_man: Dict[str, Any] = {}
    qqq_err: Optional[str] = None
    try:
        _, qqq_man = download_qqq_daily(QQQConfig())
    except Exception as exc:
        qqq_err = f"{type(exc).__name__}:{exc}"

    ext = update_all_external()
    dq = audit_external_data(ext)

    revision_detected = False
    prev_hash = PREV_EXT_HASH.read_text().strip() if PREV_EXT_HASH.exists() else None
    cur_hash = ext.get("manifest_sha256")
    if prev_hash and cur_hash and prev_hash != cur_hash:
        revision_detected = True
        with REVISION_LOG.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "detected_at_utc": datetime.now(timezone.utc).isoformat(),
                        "previous_manifest_hash": prev_hash,
                        "new_manifest_hash": cur_hash,
                        "note": "external_manifest_changed; existing predictions not rewritten",
                    }
                )
                + "\n"
            )
    if cur_hash:
        PREV_EXT_HASH.write_text(cur_hash + "\n")

    from canbit_equity.config import paths as qqq_paths

    qp = qqq_paths()
    norm = pd.read_parquet(qp["normalized_file"])
    # persist=False: never overwrite stale qqq_daily_features.parquet intermediate
    feat, _ = build_features(norm, QQQConfig(), persist=False)
    feat = _add_labels_keep_tail(feat, cost_bps=5.0)

    # Write live features ONLY to prospective cache — never overwrite historical research artifact.
    merged, feat_man = build_regime_features(
        feat,
        persist=True,
        output_feat_path=LIVE_FEAT,
        output_manifest_path=PROSP_ROOT / "manifests/live_feature_manifest.json",
    )
    wanted = list(
        dict.fromkeys(
            ["session_date", "open_adj", "close_adj", "sma_50", "sma_200", "ret_20d", "ret_60d", "realized_vol_20"]
            + logistic_feature_columns()
            + ALL_REGIME_FEATURES
            + [f"label_long_{h}d" for h in (5, 10, 20)]
            + [f"forward_return_{h}d_net" for h in (5, 10, 20)]
            + [f"forward_return_{h}d_gross" for h in (5, 10, 20)]
        )
    )
    live_cols = [c for c in wanted if c in merged.columns]
    live = merged.loc[:, ~merged.columns.duplicated()].copy()
    live = live[[c for c in live_cols if c in live.columns]]
    live["session_date"] = pd.to_datetime(live["session_date"]).dt.normalize()
    LIVE_FEAT.parent.mkdir(parents=True, exist_ok=True)
    live.to_parquet(LIVE_FEAT, index=False)

    hist_hash_after = _sha_file(HIST_FEAT) if HIST_FEAT.exists() else None
    # Daily live build must never mutate historical research feature artifact.
    hist_preserved = hist_hash_before == hist_hash_after

    align = feat_man.get("align_audit") or {}
    out = {
        "qqq_update": qqq_man or {"error": qqq_err},
        "qqq_error": qqq_err,
        "external_manifest_hash": cur_hash,
        "external_data_ok": bool(ext.get("external_data_downloaded")),
        "data_quality_verdict": dq.get("verdict"),
        "alignment_verdict": align.get("verdict"),
        "future_join_count": int(align.get("future_join_count") or 0),
        "macro_same_day_unlagged_count": int(align.get("macro_same_day_unlagged_count") or 0),
        "source_revision_detected": revision_detected,
        "live_feature_path": str(LIVE_FEAT),
        "live_feature_rows": int(len(live)),
        "live_feature_end": str(live["session_date"].max().date()) if len(live) else None,
        "historical_feature_hash_before": hist_hash_before,
        "historical_feature_hash_after": hist_hash_after,
        "historical_feature_hash_preserved": hist_preserved,
        "qew_provider_mapping": "canonical_id=QEW; yfinance_ticker=QQEW",
        "production_ready": False,
        "promotion_ready": False,
    }
    atomic_write_json(PROSP_ROOT / "manifests/prospective_data_manifest.json", out)
    atomic_write_json(DIAG_ROOT / "data_quality/data_quality.json", out)
    return out


def load_live_features() -> pd.DataFrame:
    if LIVE_FEAT.exists():
        df = pd.read_parquet(LIVE_FEAT)
    else:
        df = pd.read_parquet(HIST_FEAT)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.normalize()
    return df
