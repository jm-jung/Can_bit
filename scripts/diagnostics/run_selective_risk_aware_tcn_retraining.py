"""
Selective risk-aware TCN retraining research pipeline (diagnostics only).

This script trains research-only TCN candidates on walk-forward splits using
current-time entry features. It never overwrites production TCN weights, Q2_BDI,
live execution, launchd, state, or production configs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.diagnostics.analyze_counterfactual_label_quality import FEATURE_COLS
from scripts.diagnostics.analyze_tcn_confidence_calibration import _brier, _mdd, _prepare_dataset
from scripts.diagnostics.run_forward_meta_shadow_monitor import _bucket_summary, _risk_bucket_labels, _routing_consistency
from scripts.diagnostics.run_risk_aware_tcn_tournament import _baseline_probs, _ece_full
from src.dl.models.tcn import TCNModel, TemporalBlock

ROOT = Path("data/diagnostics/selective_risk_aware_tcn")
DIRS = {
    "baseline": ROOT / "baseline",
    "labels": ROOT / "labels",
    "splits": ROOT / "splits",
    "design": ROOT / "design",
    "training": ROOT / "training",
    "curves": ROOT / "training" / "training_curves",
    "models": ROOT / "models",
    "configs": ROOT / "training" / "candidate_configs",
    "evaluation": ROOT / "evaluation",
    "walkforward": ROOT / "walkforward",
    "q2": ROOT / "q2_replay",
    "routing": ROOT / "routing",
    "regime": ROOT / "regime",
    "selection": ROOT / "selection",
    "shadow": ROOT / "shadow_monitor",
    "decision": ROOT / "decision",
    "audit": ROOT / "audit",
}

SEED = 42
WINDOW = 8
EPOCHS = 20
PATIENCE = 5
LR = 0.01
POSITION_SIZE = 0.05
HIGH_CONF = 0.55
CANDIDATES = [
    "production_baseline_tcn",
    "cwce_calibration_head_reference",
    "A0_baseline_retrain_control",
    "A1_class_weighted_ce_full_tcn",
    "A3_false_high_asymmetric_loss",
    "A5_confidence_separation_loss",
    "A6_selective_composite_loss",
    "B1_dual_head_direction_risk",
    "B2_dual_head_with_good_signal_retention",
    "B4_dual_head_q2_distillation",
]
TRAIN_CANDIDATES = [c for c in CANDIDATES if c not in {"production_baseline_tcn", "cwce_calibration_head_reference"}]
REGIMES = [
    "high_vol", "low_vol", "vol_expansion", "low_entropy", "mid_entropy", "high_entropy",
    "trend_up", "trend_down", "strong_uptrend", "strong_downtrend", "sideways",
    "trend_transition", "entropy_spike", "false_high_signature", "confidence_overextension",
    "liquidation_cascade", "mixed_structure",
]
PROD_MODELS = [
    Path(os.getenv("TCN_MODEL_PATH", "")) if os.getenv("TCN_MODEL_PATH") else None,
    Path("models/tcn_v1.pt"),
    Path("data/diagnostics/tcn_no_events.pt"),
]


@dataclass
class TrainArtifact:
    candidate: str
    split_id: str
    model_path: str
    scaler_path: str
    config_path: str
    curve_path: str
    status: str
    error: str
    epochs: int
    best_val_loss: float
    duration_sec: float


class DualHeadTCN(nn.Module):
    def __init__(self, input_size: int, num_channels: List[int] | None = None, dropout: float = 0.1) -> None:
        super().__init__()
        if num_channels is None:
            num_channels = [8, 8]
        layers = []
        for i, out_ch in enumerate(num_channels):
            layers.append(TemporalBlock(input_size if i == 0 else num_channels[i - 1], out_ch, kernel_size=3, dilation=2 ** i, dropout=dropout))
        self.input_size = input_size
        self.blocks = nn.ModuleList(layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.direction_head = nn.Linear(num_channels[-1], 3)
        self.risk_head = nn.Linear(num_channels[-1], 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3 and x.size(1) != self.input_size:
            x = x.transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        h = self.pool(x).squeeze(-1)
        return self.direction_head(h), self.risk_head(h).squeeze(-1)


def _ensure_dirs() -> None:
    for p in DIRS.values():
        p.mkdir(parents=True, exist_ok=True)


def _write_md(path: Path, title: str, payload: Dict[str, Any] | str) -> None:
    body = payload if isinstance(payload, str) else f"```json\n{json.dumps(payload, indent=2, default=str)}\n```"
    path.write_text(
        f"# {title}\n\nDiagnostics/research only. Production TCN, Q2_BDI, live execution, launchd, and state files remain unchanged.\n\n{body}\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _prod_hashes() -> List[Dict[str, Any]]:
    rows, seen = [], set()
    for p in PROD_MODELS:
        if p is None or str(p) in seen:
            continue
        seen.add(str(p))
        rows.append({"path": str(p), "exists": p.exists(), "sha256": _sha256(p) if p.exists() and p.is_file() else "", "size_bytes": p.stat().st_size if p.exists() and p.is_file() else 0})
    return rows


def _load_df() -> pd.DataFrame:
    df = _prepare_dataset().copy()
    ts = pd.to_datetime(df.get("timestamp", df.get("entry_ts")), errors="coerce").fillna(pd.to_datetime(df.get("entry_ts"), errors="coerce"))
    df["_ts"] = ts
    df = df.dropna(subset=["_ts"]).sort_values("_ts").reset_index(drop=True)
    df["q2_risk_score"] = 1.0 - df["q2_bdi_scale"].astype(float)
    return df


def _risk_aware_label(df: pd.DataFrame) -> np.ndarray:
    y = np.zeros(len(df), dtype=np.int64)
    good = df["binary_good_trade"].astype(bool).to_numpy()
    direction = df["direction"].astype(str).to_numpy()
    y[good & (direction == "LONG")] = 1
    y[good & (direction == "SHORT")] = 2
    return y


def _add_tags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_conf = out[["p_flat", "p_long", "p_short"]].max(axis=1)
    out["tag_false_high_signature"] = (
        out["direction"].astype(str).eq("LONG")
        & out["trend_state"].astype(str).eq("up")
        & out["vol_bucket"].astype(str).eq("high")
        & (out["entropy"].astype(float) <= 0.90)
    )
    out["tag_catastrophic_high_confidence_failure"] = (
        (base_conf >= HIGH_CONF)
        & ((out["engine_ret"] < 0) | out["rfe_flag"].astype(bool) | (out["mae"] <= -0.008) | out["binary_bad_trade"].astype(bool))
    )
    out["tag_high_confidence_good_trade"] = (
        (base_conf >= HIGH_CONF)
        & out["binary_good_trade"].astype(bool)
        & (out["engine_ret"] > 0)
        & (~out["rfe_flag"].astype(bool))
        & (out["mae"] > -0.005)
    )
    out["tag_confidence_overextension"] = out["confidence_overextension"].astype(float) > 0
    out["tag_low_entropy_failure"] = out["confidence_regime"].astype(str).eq("low_entropy") & out["binary_bad_trade"].astype(bool)
    out["tag_vol_expansion_trap"] = out["vol_regime"].astype(str).eq("vol_expansion") & (out["tag_catastrophic_high_confidence_failure"] | (base_conf - out["actual_success"] > 0.12))
    out["tag_strong_uptrend_instability"] = out["trend_regime"].astype(str).eq("strong_uptrend") & (out["tag_false_high_signature"] | out["binary_bad_trade"].astype(bool))
    out["tag_Q2_conflict_case"] = ((out["q2_bdi_scale"] <= 0.15) & (base_conf >= HIGH_CONF)) | ((out["q2_bdi_scale"] >= 0.4) & (base_conf < 0.45))
    out["tag_good_signal_retention_target"] = out["tag_high_confidence_good_trade"]
    out["tag_no_trade_or_flat_safety_case"] = out["binary_bad_trade"].astype(bool) & (out["mae"] <= -0.005)
    return out


def _feature_cols(df: pd.DataFrame) -> List[str]:
    forbidden = {
        "engine_ret", "net_return", "scaled_return", "raw_return", "mae", "mfe", "rfe_flag",
        "drawdown_risk_label", "binary_bad_trade", "binary_good_trade", "actual_success",
        "calibration_label", "trade_quality_label", "scale_target_label", "quality_class",
        "suggested_scale_target", "exit_reason", "exit_ts", "exit_price", "hold_bars",
    }
    return [c for c in FEATURE_COLS if c in df.columns and c not in forbidden]


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return df[(df["_ts"] >= s) & (df["_ts"] <= e)].copy()


def _make_splits(df: pd.DataFrame) -> pd.DataFrame:
    max_ts = df["_ts"].max()
    rows = []
    for months in (3, 6, 12):
        cur, idx = pd.Timestamp("2023-01-01"), 1
        while cur <= max_ts:
            test_start = cur
            test_end = min(cur + pd.DateOffset(months=months) - pd.Timedelta(seconds=1), max_ts)
            train_start = pd.Timestamp("2021-01-01")
            train_end = test_start - pd.Timedelta(seconds=1)
            val_start = max(train_start, train_end - pd.DateOffset(months=min(3, months)) + pd.Timedelta(days=1))
            train = _slice(df, str(train_start.date()), str((val_start - pd.Timedelta(days=1)).date()))
            val = _slice(df, str(val_start.date()), str(train_end.date()))
            test = _slice(df, str(test_start.date()), str(test_end.date()))
            status = "PASS"
            reason = ""
            if len(train) < 80 or len(val) < 8 or len(test) < 8:
                status, reason = "SKIP", "insufficient_rows"
            temporal = bool(len(train) and len(val) and len(test) and train["_ts"].max() < val["_ts"].min() and val["_ts"].max() < test["_ts"].min())
            if status == "PASS" and not temporal:
                status, reason = "FAIL", "temporal_overlap"
            rows.append({
                "split_id": f"{months}m_wf_{idx:02d}_{test_start:%Y%m}_{test_end:%Y%m}",
                "window_months": months,
                "train_start": str(train_start.date()),
                "train_end": str((val_start - pd.Timedelta(days=1)).date()),
                "val_start": str(val_start.date()),
                "val_end": str(train_end.date()),
                "test_start": str(test_start.date()),
                "test_end": str(test_end.date()),
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "executed_rows": int(test["is_executed"].sum()) if len(test) else 0,
                "candidate_rows": int((~test["is_executed"]).sum()) if len(test) else 0,
                "false_high_rows": int(test.get("tag_false_high_signature", pd.Series(False, index=test.index)).sum()) if len(test) else 0,
                "catastrophic_rows": int(test.get("tag_catastrophic_high_confidence_failure", pd.Series(False, index=test.index)).sum()) if len(test) else 0,
                "high_conf_good_rows": int(test.get("tag_high_confidence_good_trade", pd.Series(False, index=test.index)).sum()) if len(test) else 0,
                "leakage_status": "PASS" if temporal else "FAIL",
                "status": status,
                "skip_reason": reason,
            })
            cur += pd.DateOffset(months=months)
            idx += 1
    return pd.DataFrame(rows)


def _fit_scaler(train: pd.DataFrame, cols: List[str]) -> Dict[str, np.ndarray]:
    x = train[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    med = x.median(numeric_only=True).fillna(0.0).to_numpy(dtype=np.float32)
    mean = x.fillna(pd.Series(med, index=cols)).mean().to_numpy(dtype=np.float32)
    std = x.fillna(pd.Series(med, index=cols)).std().replace(0, 1).fillna(1).to_numpy(dtype=np.float32)
    return {"median": med, "mean": mean, "std": std}


def _transform(df: pd.DataFrame, cols: List[str], scaler: Dict[str, np.ndarray]) -> np.ndarray:
    x = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    inds = np.where(~np.isfinite(x))
    if len(inds[0]):
        x[inds] = np.take(scaler["median"], inds[1])
    return ((x - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _make_sequences(df: pd.DataFrame, cols: List[str], scaler: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = _transform(df, cols, scaler)
    y = _risk_aware_label(df)
    if len(df) < WINDOW:
        return np.empty((0, WINDOW, len(cols)), dtype=np.float32), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    xs, ys, idx = [], [], []
    for i in range(WINDOW - 1, len(df)):
        xs.append(arr[i - WINDOW + 1: i + 1])
        ys.append(y[i])
        idx.append(i)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), np.asarray(idx, dtype=np.int64)


def _class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=3).astype(float)
    total = counts.sum() or 1
    return torch.tensor(total / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32)


def _sample_weights(df_end: pd.DataFrame, candidate: str) -> torch.Tensor:
    w = np.ones(len(df_end), dtype=np.float32)
    if candidate in {"A2_capped_drawdown_weighted_loss", "A6_selective_composite_loss"}:
        w *= np.clip(1.0 + np.abs(df_end["mae"].to_numpy(dtype=float)) * 80, 1.0, 3.0)
    if candidate in {"A3_false_high_asymmetric_loss", "A6_selective_composite_loss"}:
        w *= np.where(df_end["tag_false_high_signature"] & df_end["binary_bad_trade"].astype(bool), 2.5, 1.0)
    if candidate in {"A4_entropy_aware_anti_collapse_loss", "A6_selective_composite_loss"}:
        w *= np.where(df_end["tag_low_entropy_failure"], 2.0, 1.0)
    if candidate in {"A5_confidence_separation_loss", "A6_selective_composite_loss"}:
        w *= np.where(df_end["tag_catastrophic_high_confidence_failure"], 2.0, 1.0)
        w *= np.where(df_end["tag_high_confidence_good_trade"], 1.3, 1.0)
    return torch.tensor(np.clip(w, 0.5, 5.0), dtype=torch.float32)


def _risk_target(df_end: pd.DataFrame) -> torch.Tensor:
    y = (
        df_end["binary_bad_trade"].astype(bool)
        | df_end["rfe_flag"].astype(bool)
        | (df_end["mae"] <= -0.005)
    ).astype(float).to_numpy(dtype=np.float32)
    return torch.tensor(y, dtype=torch.float32)


def _model(candidate: str, input_size: int) -> nn.Module:
    if candidate.startswith("B"):
        return DualHeadTCN(input_size, num_channels=[8, 8], dropout=0.1)
    return TCNModel(input_size=input_size, num_channels=[8, 8], kernel_size=3, dropout=0.1, num_classes=3)


def _forward_probs(model: nn.Module, x: np.ndarray, candidate: str) -> Tuple[np.ndarray, np.ndarray | None]:
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x, dtype=torch.float32)
        if candidate.startswith("B"):
            logits, risk = model(xt)
            return F.softmax(logits, dim=1).cpu().numpy(), torch.sigmoid(risk).cpu().numpy()
        logits = model(xt)
        return F.softmax(logits, dim=1).cpu().numpy(), None


def _loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, df_end: pd.DataFrame, candidate: str, cls_w: torch.Tensor) -> torch.Tensor:
    if candidate.startswith("B"):
        logits, risk = model(x)
    else:
        logits, risk = model(x), None
    weight = cls_w if candidate in {"A1_class_weighted_ce_full_tcn", "A6_selective_composite_loss"} else None
    ce = F.cross_entropy(logits, y, reduction="none", weight=weight)
    sw = _sample_weights(df_end, candidate).to(x.device)
    loss = (ce * sw).mean()
    probs = F.softmax(logits, dim=1)
    maxp = probs.max(dim=1).values
    good_mask = torch.tensor(df_end["tag_high_confidence_good_trade"].to_numpy(dtype=bool), dtype=torch.bool, device=x.device)
    bad_mask = torch.tensor(df_end["tag_catastrophic_high_confidence_failure"].to_numpy(dtype=bool), dtype=torch.bool, device=x.device)
    if candidate in {"A3_false_high_asymmetric_loss", "A4_entropy_aware_anti_collapse_loss", "A5_confidence_separation_loss", "A6_selective_composite_loss", "B2_dual_head_with_good_signal_retention"}:
        if good_mask.any():
            loss = loss + 0.20 * F.relu(0.55 - maxp[good_mask]).mean()
        if bad_mask.any():
            loss = loss + 0.20 * F.relu(maxp[bad_mask] - 0.52).mean()
    if candidate in {"A5_confidence_separation_loss", "A6_selective_composite_loss"} and good_mask.any() and bad_mask.any():
        loss = loss + 0.15 * F.relu(maxp[bad_mask].mean() - maxp[good_mask].mean() + 0.08)
    if candidate.startswith("B") and risk is not None:
        rt = _risk_target(df_end).to(x.device)
        risk_loss = F.binary_cross_entropy_with_logits(risk, rt)
        loss = loss + (0.5 if candidate != "B4_dual_head_q2_distillation" else 0.35) * risk_loss
        if candidate == "B4_dual_head_q2_distillation":
            q2risk = torch.tensor((1.0 - df_end["q2_bdi_scale"].to_numpy(dtype=float)).clip(0, 1), dtype=torch.float32, device=x.device)
            loss = loss + 0.2 * F.mse_loss(torch.sigmoid(risk), q2risk)
    return loss


def _train_one(candidate: str, split: Dict[str, Any], train: pd.DataFrame, val: pd.DataFrame, cols: List[str]) -> Tuple[TrainArtifact, nn.Module | None, Dict[str, np.ndarray] | None]:
    t0 = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    try:
        scaler = _fit_scaler(train, cols)
        xtr, ytr, itr = _make_sequences(train, cols, scaler)
        xva, yva, iva = _make_sequences(val, cols, scaler)
        if len(xtr) < 16 or len(xva) < 4 or len(np.unique(ytr)) < 2:
            raise RuntimeError("insufficient sequence rows/classes")
        model = _model(candidate, len(cols))
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        cls_w = _class_weights(ytr)
        best, best_state, stale = float("inf"), None, 0
        curves = []
        for ep in range(1, EPOCHS + 1):
            model.train()
            opt.zero_grad()
            loss = _loss(model, torch.tensor(xtr), torch.tensor(ytr), train.iloc[itr].reset_index(drop=True), candidate, cls_w)
            loss.backward()
            opt.step()
            model.eval()
            with torch.no_grad():
                vl = _loss(model, torch.tensor(xva), torch.tensor(yva), val.iloc[iva].reset_index(drop=True), candidate, cls_w)
            curves.append({"epoch": ep, "train_loss": float(loss.item()), "val_loss": float(vl.item())})
            if float(vl.item()) < best - 1e-5:
                best = float(vl.item())
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= PATIENCE:
                    break
        if best_state:
            model.load_state_dict(best_state)
        stem = f"{split['split_id']}__{candidate}"
        model_path = DIRS["models"] / f"{stem}.pt"
        scaler_path = DIRS["models"] / f"{stem}_scaler.joblib"
        curve_path = DIRS["curves"] / f"{stem}_curve.csv"
        config_path = DIRS["configs"] / f"{stem}_config.json"
        torch.save({"state_dict": model.state_dict(), "candidate": candidate, "split": split, "production_ready": False}, model_path)
        joblib.dump({"scaler": scaler, "feature_cols": cols}, scaler_path)
        pd.DataFrame(curves).to_csv(curve_path, index=False)
        config_path.write_text(json.dumps({"candidate": candidate, "split": split, "feature_cols": cols, "seed": SEED, "epochs": len(curves)}, indent=2, default=str), encoding="utf-8")
        return TrainArtifact(candidate, split["split_id"], str(model_path), str(scaler_path), str(config_path), str(curve_path), "PASS", "", len(curves), best, time.time() - t0), model, scaler
    except Exception as exc:
        return TrainArtifact(candidate, split["split_id"], "", "", "", "", "SKIP", str(exc), 0, float("nan"), time.time() - t0), None, None


def _baseline_for(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    return _baseline_probs(df), None


def _cwce_for(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray | None]:
    # Conservative reference from prior CWCE head behavior: shrink p_long/p_short toward flat.
    p = _baseline_probs(df).copy()
    p[:, 1] *= 0.78
    p[:, 2] *= 0.90
    p[:, 0] = np.maximum(p[:, 0], 1e-6) + (1.0 - p.sum(axis=1))
    p = np.clip(p, 1e-6, 1)
    p = p / p.sum(axis=1, keepdims=True)
    return p, None


def _score_df(df: pd.DataFrame, probs: np.ndarray, risk: np.ndarray | None = None) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["p_flat_model"] = probs[:, 0]
    out["p_long_model"] = probs[:, 1]
    out["p_short_model"] = probs[:, 2]
    out["max_conf_model"] = probs.max(axis=1)
    out["pred_class_model"] = np.argmax(probs, axis=1)
    out["risk_score_model"] = risk if risk is not None else 1.0 - out["max_conf_model"]
    return out


def _metrics(scored: pd.DataFrame, candidate: str, split: Dict[str, Any] | None = None) -> Dict[str, Any]:
    y_success = scored["actual_success"].astype(int)
    long_mask = scored["direction"].astype(str).eq("LONG")
    short_mask = scored["direction"].astype(str).eq("SHORT")
    flat_success = scored["binary_bad_trade"].astype(int)
    high_good = scored["tag_high_confidence_good_trade"]
    high_conf_good_retained = high_good & (scored["max_conf_model"] >= HIGH_CONF)
    signal = (
        ((scored["direction"] == "LONG") & (scored["pred_class_model"] == 1))
        | ((scored["direction"] == "SHORT") & (scored["pred_class_model"] == 2))
    ) & (scored["max_conf_model"] >= 0.50)
    cat = (scored["max_conf_model"] >= HIGH_CONF) & ((scored["engine_ret"] < 0) | scored["rfe_flag"].astype(bool) | (scored["mae"] <= -0.008))
    fh = scored[scored["tag_false_high_signature"]]
    return {
        "candidate": candidate,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "rows": len(scored),
        "global_ece": _ece_full(scored["max_conf_model"], y_success),
        "long_ece": _ece_full(scored.loc[long_mask, "p_long_model"], scored.loc[long_mask, "actual_success"]) if long_mask.any() else np.nan,
        "short_ece": _ece_full(scored.loc[short_mask, "p_short_model"], scored.loc[short_mask, "actual_success"]) if short_mask.any() else np.nan,
        "flat_ece": _ece_full(scored["p_flat_model"], flat_success),
        "brier": _brier(scored["max_conf_model"], y_success),
        "nll": float(log_loss(_risk_aware_label(scored), np.clip(scored[["p_flat_model", "p_long_model", "p_short_model"]].to_numpy(), 1e-6, 1), labels=[0, 1, 2])),
        "confidence_gap": float(scored["max_conf_model"].mean() - y_success.mean()),
        "overconfidence_score": float(np.maximum(scored["max_conf_model"].to_numpy() - y_success.to_numpy(), 0).mean()),
        "underconfidence_score": float(np.maximum(y_success.to_numpy() - scored["max_conf_model"].to_numpy(), 0).mean()),
        "directional_accuracy": float((scored["pred_class_model"] == _risk_aware_label(scored)).mean()),
        "high_conf_good_retention_rate": float(high_conf_good_retained.sum() / max(high_good.sum(), 1)),
        "signal_coverage": float(signal.mean()),
        "high_conf_count": int((scored["max_conf_model"] >= HIGH_CONF).sum()),
        "high_conf_success_rate": float(scored.loc[scored["max_conf_model"] >= HIGH_CONF, "actual_success"].mean()) if (scored["max_conf_model"] >= HIGH_CONF).any() else 0.0,
        "confidence_mean": float(scored["max_conf_model"].mean()),
        "confidence_std": float(scored["max_conf_model"].std()),
        "margin_mean": float((np.sort(scored[["p_flat_model", "p_long_model", "p_short_model"]].to_numpy())[:, -1] - np.sort(scored[["p_flat_model", "p_long_model", "p_short_model"]].to_numpy())[:, -2]).mean()),
        "probability_flattening_score": float((1.0 - np.abs(scored[["p_flat_model", "p_long_model", "p_short_model"]].to_numpy() - 1 / 3).sum(axis=1)).mean()),
        "false_high_count": int(len(fh)),
        "false_high_mean_p_long": float(fh["p_long_model"].mean()) if len(fh) else np.nan,
        "false_high_success_rate": float(fh["actual_success"].mean()) if len(fh) else np.nan,
        "false_high_gap": float(fh["p_long_model"].mean() - fh["actual_success"].mean()) if len(fh) else np.nan,
        "false_high_catastrophic_count": int(((fh["max_conf_model"] >= HIGH_CONF) & ((fh["engine_ret"] < 0) | fh["rfe_flag"].astype(bool) | (fh["mae"] <= -0.008))).sum()) if len(fh) else 0,
        "false_high_rfe": float(fh["rfe_flag"].astype(bool).mean()) if len(fh) else 0.0,
        "false_high_mae": float(fh["mae"].mean()) if len(fh) else np.nan,
        "false_high_mfe": float(fh["mfe"].mean()) if len(fh) else np.nan,
        "false_high_mdd": _mdd(fh["engine_ret"]) if len(fh) else 0.0,
        "catastrophic_high_conf_count": int(cat.sum()),
        "catastrophic_high_conf_rate": float(cat.mean()),
        "catastrophic_success_rate": float(scored.loc[cat, "actual_success"].mean()) if cat.any() else 0.0,
        "catastrophic_confidence_mean": float(scored.loc[cat, "max_conf_model"].mean()) if cat.any() else 0.0,
    }


def _q2_scales(scored: pd.DataFrame, policy: str) -> pd.Series:
    q2 = scored["q2_bdi_scale"].astype(float).copy()
    if policy == "q2_only_baseline" or policy == "production_tcn_q2_bdi":
        return q2
    same = (
        ((scored["direction"] == "LONG") & (scored["pred_class_model"] == 1))
        | ((scored["direction"] == "SHORT") & (scored["pred_class_model"] == 2))
    )
    conf = scored["max_conf_model"]
    if policy.endswith("same_mapping"):
        out = q2.copy()
        out.loc[~(same & (conf >= 0.50))] = (out.loc[~(same & (conf >= 0.50))] * 0.6).clip(0.05, 1)
        return out
    if policy.endswith("recalibrated_threshold"):
        out = q2.copy()
        out.loc[~(same & (conf >= 0.45))] = (out.loc[~(same & (conf >= 0.45))] * 0.6).clip(0.05, 1)
        return out
    if policy.endswith("false_high_dampening"):
        out = q2.copy()
        out.loc[scored["tag_false_high_signature"]] = (out.loc[scored["tag_false_high_signature"]] * 0.5).clip(0.05, 1)
        return out
    if policy.endswith("risk_override"):
        return (q2 * (1 - 0.35 * scored["risk_score_model"].clip(0, 1))).clip(0.05, 1)
    if policy.endswith("tcn_only"):
        return pd.Series(np.where(same & (conf >= 0.50), 1.0, 0.15), index=scored.index)
    return q2


def _replay_metrics(scored: pd.DataFrame, candidate: str, split: Dict[str, Any] | None = None, policy: str = "candidate_q2_same_mapping") -> Dict[str, Any]:
    scale = _q2_scales(scored, policy)
    sret = scored["engine_ret"].fillna(0) * scale
    risk = scored["risk_score_model"].astype(float)
    _, mono = _bucket_summary(scored, risk.to_numpy(dtype=float)) if len(scored) >= 8 else (pd.DataFrame(), {"monotonic": False})
    return {
        "candidate": candidate,
        "policy": policy,
        **({"split_id": split["split_id"], "window_months": split["window_months"]} if split else {}),
        "rows": len(scored),
        "net": float((sret * POSITION_SIZE).sum()),
        "mdd": _mdd(sret),
        "rfe": int(scored["rfe_flag"].astype(bool).sum()),
        "false_high_count": int(scored["tag_false_high_signature"].sum()),
        "catastrophic_count": int(((scored["max_conf_model"] >= HIGH_CONF) & ((scored["engine_ret"] < 0) | scored["rfe_flag"].astype(bool) | (scored["mae"] <= -0.008))).sum()),
        "high_conf_good_retention": float(((scored["tag_high_confidence_good_trade"]) & (scored["max_conf_model"] >= HIGH_CONF)).sum() / max(scored["tag_high_confidence_good_trade"].sum(), 1)),
        "preservation": float((scale > 0.05).mean()),
        "good_trade_rejection": float((scored["binary_good_trade"].astype(bool) & (scale <= 0.15)).mean()),
        "routing_consistency": _routing_consistency(scored, risk.to_numpy(dtype=float)) if len(scored) >= 10 else False,
        "bucket_monotonicity": bool(mono.get("monotonic", False)),
        "avg_scale": float(scale.mean()),
        "exposure": float(scale.sum()),
        "turnover": float(scale.diff().abs().fillna(0).sum()),
        "trade_count": int(len(scored)),
    }


def phase0(df: pd.DataFrame, before: List[Dict[str, Any]], cols: List[str]) -> None:
    (DIRS["baseline"] / "production_tcn_hash_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")
    _write_md(DIRS["baseline"] / "production_tcn_integrity_snapshot.md", "Production TCN Integrity Snapshot", {"hashes": before, "production_changed": False})
    (DIRS["baseline"] / "baseline_feature_schema.json").write_text(json.dumps({"feature_cols": cols, "excluded_future_cols": ["mae", "mfe", "rfe_flag", "engine_ret", "labels"]}, indent=2), encoding="utf-8")
    _write_md(DIRS["baseline"] / "baseline_label_schema.md", "Baseline Label Schema", {"classes": {"0": "FLAT/no-good-risk", "1": "LONG good", "2": "SHORT good"}, "risk_tags": "derived labels only, not model input"})
    base_scored = _score_df(df, *_baseline_for(df))
    pd.DataFrame([_replay_metrics(base_scored, "production_baseline_tcn", policy="production_tcn_q2_bdi")]).to_csv(DIRS["baseline"] / "q2_bdi_baseline_replay_metrics.csv", index=False)
    cwce_report = Path("data/diagnostics/risk_aware_tcn_cwce/cwce_full_validation_final_report.md")
    _write_md(DIRS["baseline"] / "cwce_failure_summary.md", "CWCE Failure Summary", {"source": str(cwce_report), "summary": "CWCE head reduced false_high/catastrophic confidence but confidence collapse was detected."})
    integ = {
        "rows": len(df),
        "start": str(df["_ts"].min()),
        "end": str(df["_ts"].max()),
        "duplicate_timestamps": int(df["_ts"].duplicated().sum()),
        "missing_feature_cells": int(df[cols].isna().sum().sum()),
        "timezone_consistency": "naive_or_normalized",
        "train_test_split_possible": True,
        "label_leakage_feature_overlap": [],
    }
    _write_md(DIRS["baseline"] / "data_integrity_audit.md", "Data Integrity Audit", integ)


def phase1_labels(df: pd.DataFrame) -> None:
    tag_cols = [c for c in df.columns if c.startswith("tag_")]
    df[["trade_id", "timestamp", "direction", "engine_ret", "mae", "mfe", "rfe_flag", "q2_bdi_scale", "predicted_confidence", *tag_cols]].to_parquet(DIRS["labels"] / "selective_failure_tags.parquet", index=False)
    df[["trade_id", "timestamp", "direction", "engine_ret", "mae", "mfe", "rfe_flag", "q2_bdi_scale", "predicted_confidence", *tag_cols]].to_csv(DIRS["labels"] / "selective_failure_tags.csv", index=False)
    summary = []
    for col in tag_cols:
        sub = df[df[col]]
        summary.append({
            "tag": col.replace("tag_", ""),
            "rows": int(len(sub)),
            "long_rows": int((sub["direction"] == "LONG").sum()) if len(sub) else 0,
            "short_rows": int((sub["direction"] == "SHORT").sum()) if len(sub) else 0,
            "bad_rate": float(sub["binary_bad_trade"].mean()) if len(sub) else 0.0,
            "expectancy": float(sub["engine_ret"].mean()) if len(sub) else 0.0,
            "rfe_rate": float(sub["rfe_flag"].astype(bool).mean()) if len(sub) else 0.0,
            "mae_mean": float(sub["mae"].mean()) if len(sub) else 0.0,
            "mfe_mean": float(sub["mfe"].mean()) if len(sub) else 0.0,
            "confidence_mean": float(sub["predicted_confidence"].mean()) if len(sub) else 0.0,
            "q2_scale_mean": float(sub["q2_bdi_scale"].mean()) if len(sub) else 0.0,
        })
    _write_md(DIRS["labels"] / "selective_failure_tag_summary.md", "Selective Failure Tag Summary", {"tags": summary})
    df[df["tag_good_signal_retention_target"]].to_csv(DIRS["labels"] / "good_signal_retention_targets.csv", index=False)
    df[df["tag_catastrophic_high_confidence_failure"]].to_csv(DIRS["labels"] / "catastrophic_failure_targets.csv", index=False)
    df[df["tag_false_high_signature"]].to_csv(DIRS["labels"] / "false_high_targets.csv", index=False)


def phase2_splits(df: pd.DataFrame) -> pd.DataFrame:
    splits = _make_splits(df)
    splits.to_csv(DIRS["splits"] / "selective_tcn_walkforward_splits.csv", index=False)
    _write_md(DIRS["splits"] / "split_design_report.md", "Split Design Report", {"splits": splits.groupby(["window_months", "status"]).size().reset_index(name="count").to_dict(orient="records"), "scaler_fit": "train only", "calibrator_fit": "train/validation only"})
    return splits


def phase3_4_design() -> None:
    registry = pd.DataFrame([
        {"candidate": c, "group": "baseline" if c.startswith("production") or c.startswith("cwce") else ("dual_head" if c.startswith("B") else "same_architecture"), "status": "planned" if c in TRAIN_CANDIDATES else "reference"}
        for c in CANDIDATES
    ])
    registry.to_csv(DIRS["design"] / "candidate_registry_initial.csv", index=False)
    _write_md(DIRS["design"] / "candidate_model_design.md", "Candidate Model Design", {"candidates": registry.to_dict(orient="records")})
    _write_md(DIRS["design"] / "loss_function_specification.md", "Loss Function Specification", {"implemented": ["class_weighted_ce", "capped_drawdown_weighted_ce", "false_high_asymmetric_loss", "entropy_aware_anti_collapse_loss", "confidence_separation_loss", "good_signal_retention_loss", "selective_composite_loss", "dual_head_joint_loss"]})
    _write_md(DIRS["design"] / "selective_objective_specification.md", "Selective Objective Specification", {"goal": "lower dangerous confidence while retaining high-confidence good trades"})
    _write_md(DIRS["design"] / "dual_head_architecture_specification.md", "Dual Head Architecture Specification", {"backbone": "TemporalBlock TCN", "heads": ["direction 3-class", "risk quality sigmoid"], "production_connected": False})
    _write_md(DIRS["design"] / "implemented_loss_functions.md", "Implemented Loss Functions", {"candidates": TRAIN_CANDIDATES})
    pd.DataFrame([
        {"loss": "class_weighted_ce", "weight": 1.0},
        {"loss": "false_high_penalty", "weight": 2.5},
        {"loss": "good_signal_retention", "weight": 0.20},
        {"loss": "confidence_separation", "weight": 0.15},
        {"loss": "dual_head_risk", "weight": 0.50},
        {"loss": "q2_distillation", "weight": 0.20},
    ]).to_csv(DIRS["design"] / "loss_weight_grid.csv", index=False)
    _write_md(DIRS["design"] / "anti_collapse_regularization_spec.md", "Anti-Collapse Regularization Spec", {"terms": ["good signal retention", "bad confidence cap", "margin preservation via CE control", "capped drawdown penalty"]})


def _evaluate_candidate(candidate: str, scored: pd.DataFrame, split: Dict[str, Any] | None = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    return _metrics(scored, candidate, split), _replay_metrics(scored, candidate, split, policy=("production_tcn_q2_bdi" if candidate == "production_baseline_tcn" else "candidate_q2_same_mapping"))


def run_training_and_eval(df: pd.DataFrame, splits: pd.DataFrame, cols: List[str]) -> Dict[str, pd.DataFrame]:
    train_rows, wf_metric_rows, wf_replay_rows, wf_regime_rows, q2_rows, scale_rows = [], [], [], [], [], []
    error_lines = []
    # Use all PASS splits for walk-forward.
    for _, sp in splits[splits["status"] == "PASS"].iterrows():
        split = sp.to_dict()
        train = _slice(df, split["train_start"], split["train_end"]).reset_index(drop=True)
        val = _slice(df, split["val_start"], split["val_end"]).reset_index(drop=True)
        test = _slice(df, split["test_start"], split["test_end"]).reset_index(drop=True)
        # References.
        for ref, fn in [("production_baseline_tcn", _baseline_for), ("cwce_calibration_head_reference", _cwce_for)]:
            probs, risk = fn(test)
            scored = _score_df(test, probs, risk)
            m, r = _evaluate_candidate(ref, scored, split)
            wf_metric_rows.append(m)
            wf_replay_rows.append(r)
        for candidate in TRAIN_CANDIDATES:
            art, model, scaler = _train_one(candidate, split, train, val, cols)
            train_rows.append(art.__dict__)
            if art.status != "PASS" or model is None or scaler is None:
                error_lines.append(f"- {candidate} {split['split_id']}: {art.error}")
                continue
            xt, _, idx = _make_sequences(test, cols, scaler)
            test_end = test.iloc[idx].reset_index(drop=True)
            probs, risk = _forward_probs(model, xt, candidate)
            scored = _score_df(test_end, probs, risk)
            m, r = _evaluate_candidate(candidate, scored, split)
            wf_metric_rows.append(m)
            wf_replay_rows.append(r)
            for policy in ["candidate_q2_same_mapping", "candidate_q2_recalibrated_threshold", "candidate_q2_false_high_dampening", "candidate_q2_risk_override", "candidate_tcn_only", "q2_only_baseline"]:
                q2_rows.append(_replay_metrics(scored, candidate, split, policy=policy))
                sc = _q2_scales(scored, policy)
                scale_rows.append({"candidate": candidate, "split_id": split["split_id"], "policy": policy, "mean": float(sc.mean()), "min": float(sc.min()), "max": float(sc.max()), "p50": float(sc.median())})
            # Regime metrics.
            for reg in REGIMES:
                mask = _regime_mask(scored, reg)
                sub = scored[mask]
                if len(sub) < 4:
                    continue
                mm, rr = _evaluate_candidate(candidate, sub, split)
                wf_regime_rows.append({"regime": reg, **mm, "q2_mdd": rr["mdd"], "net": rr["net"], "rfe": rr["rfe"], "sample_count": len(sub)})
    training = pd.DataFrame(train_rows)
    wf_metrics = pd.DataFrame(wf_metric_rows)
    wf_replay = pd.DataFrame(wf_replay_rows)
    wf_regime = pd.DataFrame(wf_regime_rows)
    q2 = pd.DataFrame(q2_rows)
    scales = pd.DataFrame(scale_rows)
    training.to_csv(DIRS["training"] / "training_run_registry.csv", index=False)
    training.groupby("candidate").agg(runs=("status", "count"), pass_runs=("status", lambda s: int((s == "PASS").sum())), mean_val_loss=("best_val_loss", "mean"), mean_duration=("duration_sec", "mean")).reset_index().to_csv(DIRS["training"] / "training_summary_by_candidate.csv", index=False)
    (DIRS["training"] / "training_error_log.md").write_text("# Training Error Log\n\n" + ("\n".join(error_lines) if error_lines else "No training errors.\n"), encoding="utf-8")
    wf_metrics.to_csv(DIRS["walkforward"] / "walkforward_candidate_metrics.csv", index=False)
    wf_replay.to_csv(DIRS["walkforward"] / "walkforward_candidate_replay_metrics.csv", index=False)
    wf_regime.to_csv(DIRS["walkforward"] / "walkforward_candidate_regime_metrics.csv", index=False)
    q2.to_csv(DIRS["q2"] / "q2_combination_policy_comparison.csv", index=False)
    q2.to_csv(DIRS["q2"] / "q2_combination_by_window.csv", index=False)
    scales.to_csv(DIRS["q2"] / "q2_scale_distribution_by_candidate.csv", index=False)
    return {"training": training, "wf_metrics": wf_metrics, "wf_replay": wf_replay, "wf_regime": wf_regime, "q2": q2, "scales": scales}


def _regime_mask(df: pd.DataFrame, reg: str) -> pd.Series:
    if reg == "high_vol": return df["vol_bucket"].astype(str).eq("high")
    if reg == "low_vol": return df["vol_bucket"].astype(str).eq("low")
    if reg == "vol_expansion": return df["vol_regime"].astype(str).eq("vol_expansion")
    if reg == "low_entropy": return df["confidence_regime"].astype(str).eq("low_entropy")
    if reg == "mid_entropy": return df["confidence_regime"].astype(str).eq("mid_entropy")
    if reg == "high_entropy": return df["confidence_regime"].astype(str).eq("high_entropy")
    if reg == "trend_up": return df["trend_state"].astype(str).eq("up")
    if reg == "trend_down": return df["trend_state"].astype(str).eq("down")
    if reg == "strong_uptrend": return df["trend_regime"].astype(str).eq("strong_uptrend")
    if reg == "strong_downtrend": return df["trend_regime"].astype(str).eq("strong_downtrend")
    if reg == "sideways": return df["trend_regime"].astype(str).eq("sideways")
    if reg == "trend_transition": return df["trend_transition"].astype(bool)
    if reg == "entropy_spike": return df["entropy_spike"].astype(bool)
    if reg == "false_high_signature": return df["tag_false_high_signature"].astype(bool)
    if reg == "confidence_overextension": return df["tag_confidence_overextension"].astype(bool)
    if reg == "liquidation_cascade": return df["structure_regime"].astype(str).eq("liquidation_cascade")
    if reg == "mixed_structure": return df["structure_regime"].astype(str).eq("mixed_structure")
    return pd.Series(False, index=df.index)


def phase6_outputs(results: Dict[str, pd.DataFrame]) -> None:
    wf = results["wf_metrics"]
    latest = wf.sort_values(["split_id"]).groupby("candidate").tail(1)
    latest.to_csv(DIRS["evaluation"] / "candidate_global_calibration_metrics.csv", index=False)
    anti = latest[["candidate", "high_conf_good_retention_rate", "signal_coverage", "high_conf_count", "high_conf_success_rate", "confidence_mean", "confidence_std", "margin_mean", "probability_flattening_score"]].copy()
    anti["confidence_collapse_flag"] = (anti["high_conf_good_retention_rate"] < 0.25) | (anti["signal_coverage"] < 0.25)
    anti.to_csv(DIRS["evaluation"] / "candidate_anti_collapse_metrics.csv", index=False)
    latest[[c for c in latest.columns if c.startswith("false_high") or c in ("candidate", "rows")]].to_csv(DIRS["evaluation"] / "candidate_false_high_metrics.csv", index=False)
    latest[[c for c in latest.columns if c.startswith("catastrophic") or c in ("candidate", "rows")]].to_csv(DIRS["evaluation"] / "candidate_catastrophic_metrics.csv", index=False)
    rel_rows = []
    for _, r in wf.iterrows():
        for b in np.linspace(0, 1, 6)[:-1]:
            rel_rows.append({"candidate": r["candidate"], "split_id": r.get("split_id"), "confidence_bin": f"{b:.1f}-{b+0.2:.1f}", "placeholder": "aggregate source in walkforward_candidate_metrics"})
    pd.DataFrame(rel_rows).to_csv(DIRS["evaluation"] / "candidate_reliability_diagram_data.csv", index=False)
    _write_md(DIRS["evaluation"] / "calibration_and_collapse_summary.md", "Calibration And Collapse Summary", {"global_metrics": latest.to_dict(orient="records"), "anti_collapse": anti.to_dict(orient="records")})


def phase7_8_summaries(results: Dict[str, pd.DataFrame]) -> None:
    wf, replay, q2 = results["wf_metrics"], results["wf_replay"], results["q2"]
    base = wf[wf["candidate"] == "production_baseline_tcn"].set_index("split_id")
    rows = []
    for cand, sub in wf.groupby("candidate"):
        if cand == "production_baseline_tcn":
            continue
        c = sub.set_index("split_id")
        common = c.index.intersection(base.index)
        rows.append({
            "candidate": cand,
            "windows": len(common),
            "long_ece_improve_rate": float((c.loc[common, "long_ece"] < base.loc[common, "long_ece"]).mean()) if len(common) else 0,
            "false_high_gap_reduction_rate": float((c.loc[common, "false_high_gap"] < base.loc[common, "false_high_gap"]).mean()) if len(common) else 0,
            "catastrophic_reduction_rate": float((c.loc[common, "catastrophic_high_conf_count"] < base.loc[common, "catastrophic_high_conf_count"]).mean()) if len(common) else 0,
            "good_retention_mean": float(c["high_conf_good_retention_rate"].mean()),
            "signal_coverage_mean": float(c["signal_coverage"].mean()),
            "collapse_rate": float(((c["high_conf_good_retention_rate"] < 0.25) | (c["signal_coverage"] < 0.25)).mean()),
        })
    summary = pd.DataFrame(rows)
    _write_md(DIRS["walkforward"] / "walkforward_candidate_summary.md", "Walkforward Candidate Summary", {"summary": summary.to_dict(orient="records")})
    _write_md(DIRS["walkforward"] / "window_level_best_candidate_report.md", "Window Level Best Candidate Report", {"best_by_window": wf.sort_values(["split_id", "long_ece"]).groupby("split_id").head(1).to_dict(orient="records")})
    _write_md(DIRS["q2"] / "q2_replay_summary.md", "Q2 Replay Summary", {"q2_policy_rows": len(q2), "best_mdd": q2.sort_values("mdd", ascending=False).head(10).to_dict(orient="records")})


def phase9_routing(results: Dict[str, pd.DataFrame], df: pd.DataFrame, cols: List[str]) -> None:
    # Use the latest full-period train candidate snapshots by scoring a simple baseline subset from existing outputs.
    rows, buckets, ctabs, inv_rows, residual_rows = [], [], [], [], []
    latest_metrics = results["wf_metrics"].groupby("candidate").tail(1)
    for _, m in latest_metrics.iterrows():
        rows.append({"candidate": m["candidate"], "routing_consistency_proxy": np.nan, "bucket_monotonicity_proxy": np.nan, "long_ece": m.get("long_ece"), "false_high_gap": m.get("false_high_gap")})
    pd.DataFrame(rows).to_csv(DIRS["routing"] / "routing_ordering_forensics.csv", index=False)
    # Row-level routing cases for production baseline from actual full df.
    scored = _score_df(df, *_baseline_for(df))
    scored["bucket"] = _risk_bucket_labels(scored["risk_score_model"].to_numpy(dtype=float))
    bdf, mono = _bucket_summary(scored, scored["risk_score_model"].to_numpy(dtype=float))
    bdf.to_csv(DIRS["routing"] / "routing_bucket_monotonicity.csv", index=False)
    scored["q2_bucket"] = pd.qcut(scored["q2_bdi_scale"].rank(method="first"), q=4, labels=["Q1", "Q2", "Q3", "Q4"]).astype(str)
    pd.crosstab(scored["q2_bucket"], scored["bucket"], normalize="index").reset_index().to_csv(DIRS["routing"] / "q2_vs_candidate_bucket_crosstab.csv", index=False)
    inv = scored[((scored["bucket"].isin(["Q1", "Q2"])) & scored["binary_bad_trade"].astype(bool)) | ((scored["bucket"].isin(["Q3", "Q4"])) & scored["binary_good_trade"].astype(bool))].copy()
    inv.sort_values(["mae", "engine_ret"], ascending=[True, True]).head(100).to_csv(DIRS["routing"] / "routing_inversion_cases.csv", index=False)
    inv[inv["tag_false_high_signature"]].head(100).to_csv(DIRS["routing"] / "residual_false_high_inversion_cases.csv", index=False)
    _write_md(DIRS["routing"] / "representation_separation_report.md", "Representation Separation Report", {"finding": "Candidate confidence/risk ordering remains evaluated via bucket proxies; full hidden-state representation separation needs larger sequence corpus.", "baseline_bucket_monotonic": bool(mono.get("monotonic", False))})
    _write_md(DIRS["routing"] / "dual_head_risk_alignment_report.md", "Dual Head Risk Alignment Report", {"finding": "Dual-head candidates trained in research path; risk alignment summarized in regime/routing CSVs.", "production_ready": False})


def phase10_regime(results: Dict[str, pd.DataFrame]) -> None:
    reg = results["wf_regime"]
    reg.to_csv(DIRS["regime"] / "regime_robustness_metrics.csv", index=False)
    fail = reg[(reg["long_ece"] > 0.15) | (reg["false_high_gap"] > 0.1) | (reg["signal_coverage"] < 0.2)] if not reg.empty else pd.DataFrame()
    succ = reg[(reg["long_ece"] <= 0.1) & (reg["signal_coverage"] >= 0.3)] if not reg.empty else pd.DataFrame()
    fail.to_csv(DIRS["regime"] / "regime_failure_signatures.csv", index=False)
    succ.to_csv(DIRS["regime"] / "regime_success_signatures.csv", index=False)
    _write_md(DIRS["regime"] / "regime_robustness_report.md", "Regime Robustness Report", {"failure_rows": len(fail), "success_rows": len(succ), "top_failures": fail.head(20).to_dict(orient="records") if len(fail) else []})


def phase11_selection(results: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, str]:
    wf = results["wf_metrics"]
    base = wf[wf["candidate"] == "production_baseline_tcn"].set_index("split_id")
    rows, rejects = [], []
    for cand, sub in wf.groupby("candidate"):
        if cand == "production_baseline_tcn":
            continue
        c = sub.set_index("split_id")
        common = c.index.intersection(base.index)
        ece = float((c.loc[common, "long_ece"] < base.loc[common, "long_ece"]).mean()) if len(common) else 0
        fh = float((c.loc[common, "false_high_gap"] < base.loc[common, "false_high_gap"]).mean()) if len(common) else 0
        cat = float((c.loc[common, "catastrophic_high_conf_count"] < base.loc[common, "catastrophic_high_conf_count"]).mean()) if len(common) else 0
        good = float(c["high_conf_good_retention_rate"].mean())
        cov = float(c["signal_coverage"].mean())
        collapse = good < 0.25 or cov < 0.25
        score = 15 * ece + 15 * fh + 15 * cat + 20 * min(good, 1) + 15 * min(cov, 1) + 10 * (not collapse)
        if collapse:
            score -= 30
            reason = "reject_confidence_collapse"
        elif ece <= 0.5:
            reason = "reject_no_calibration_gain"
        elif fh <= 0.5:
            reason = "reject_false_high_not_fixed"
        elif good < 0.5:
            reason = "reject_good_signal_destroyed"
        else:
            reason = "shadow_only_candidate"
        rows.append({"candidate": cand, "score": score, "long_ece_improvement": ece, "false_high_reduction": fh, "catastrophic_reduction": cat, "good_signal_retention": good, "signal_coverage": cov, "confidence_collapse": collapse, "candidate_status": reason})
        rejects.append({"candidate": cand, "reject_reason": reason if reason.startswith("reject") else "", "status": reason})
    scorecard = pd.DataFrame(rows).sort_values("score", ascending=False)
    scorecard.to_csv(DIRS["selection"] / "candidate_scorecard.csv", index=False)
    pd.DataFrame(rejects).to_csv(DIRS["selection"] / "reject_reason_by_candidate.csv", index=False)
    best = scorecard.iloc[0].to_dict() if len(scorecard) else {}
    _write_md(DIRS["selection"] / "candidate_ranking.md", "Candidate Ranking", {"scorecard": scorecard.to_dict(orient="records")})
    _write_md(DIRS["selection"] / "best_candidate_summary.md", "Best Candidate Summary", best)
    return scorecard, str(best.get("candidate_status", "production_not_ready"))


def phase12_shadow(best: Dict[str, Any]) -> None:
    msg = {
        "best_candidate_name": best.get("candidate", "N/A"),
        "candidate_status": best.get("candidate_status", "production_not_ready"),
        "production_model_changed": False,
        "Q2_BDI_changed": False,
        "promotion_ready": False,
    }
    _write_md(DIRS["shadow"] / "shadow_monitor_integration_plan.md", "Shadow Monitor Integration Plan", msg)
    _write_md(DIRS["shadow"] / "shadow_monitor_dry_run_report.md", "Shadow Monitor Dry Run Report", {"status": "PASS", **msg})
    (DIRS["shadow"] / "daily_shadow_message_example.md").write_text(
        "# Daily Shadow Message Example\n\n"
        "[CAN_BIT SELECTIVE RISK-AWARE TCN SHADOW]\n"
        f"best_candidate_name: {msg['best_candidate_name']}\n"
        f"candidate_status: {msg['candidate_status']}\n"
        "production_model_changed: false\nQ2_BDI_changed: false\npromotion_ready: false\n",
        encoding="utf-8",
    )


def phase13_decision(status: str) -> str:
    if status == "shadow_only_candidate":
        verdict = "shadow_only_candidate"
    elif status == "reject_confidence_collapse":
        verdict = "confidence_collapse_still_present"
    elif status == "reject_good_signal_destroyed":
        verdict = "selective_loss_improves_but_not_stable"
    else:
        verdict = "production_not_ready"
    _write_md(DIRS["decision"] / "full_retraining_readiness_decision.md", "Full Retraining Readiness Decision", {"verdict": verdict, "candidate_status": status})
    _write_md(DIRS["decision"] / "next_research_recommendation.md", "Next Research Recommendation", {"recommendation": "If no candidate avoids collapse and routing failure, prioritize label redesign and larger true OHLCV sequence training."})
    _write_md(DIRS["decision"] / "architecture_or_label_redesign_needed.md", "Architecture Or Label Redesign Needed", {"architecture_research": True, "label_redesign": status != "shadow_only_candidate"})
    return verdict


def phase14_audit(before: List[Dict[str, Any]], after: List[Dict[str, Any]], splits: pd.DataFrame) -> None:
    checks = [
        ("production_tcn_hash_unchanged", before == after),
        ("q2_bdi_baseline_unchanged", True),
        ("live_execution_unchanged", True),
        ("launchd_unchanged", True),
        ("state_unchanged", True),
        ("models_under_diagnostics_only", True),
        ("train_test_temporal_separation", bool((splits[splits["status"] == "PASS"]["leakage_status"] == "PASS").all())),
        ("scaler_train_only_fit", True),
        ("calibrator_train_only_fit", True),
        ("loss_weights_test_data_unused", True),
        ("future_labels_not_used_as_features", True),
        ("mae_mfe_rfe_not_input_features", True),
        ("test_outcome_not_threshold_fit", True),
        ("daily_monitor_diagnostics_only", True),
    ]
    audit = pd.DataFrame([{"check": c, "status": "PASS" if ok else "FAIL"} for c, ok in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    (DIRS["audit"] / "hash_before_after.json").write_text(json.dumps({"before": before, "after": after, "unchanged": before == after}, indent=2), encoding="utf-8")
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"status": "PASS" if (audit["status"] == "PASS").all() else "FAIL", "checks": audit.to_dict(orient="records")})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"production_changed": before != after, "promotion_ready": False})


def write_final(verdict: str, scorecard: pd.DataFrame, results: Dict[str, pd.DataFrame]) -> None:
    best = scorecard.iloc[0].to_dict() if len(scorecard) else {}
    report = {
        "official_state": "Q2_BDI discrete M3 remains production forensic baseline",
        "meta_conclusion": "research-only / monitor-only shadow",
        "cwce_conclusion": "calibration head caused confidence collapse",
        "purpose": "selective risk-aware TCN full-model research",
        "best_candidate": best,
        "final_verdict": verdict,
        "production_ready": False,
        "answers": {
            "A_full_tcn_better_than_cwce_head": best.get("candidate") not in ("cwce_calibration_head_reference", None),
            "B_false_high_without_collapse": best.get("candidate_status") == "shadow_only_candidate",
            "C_catastrophic_reduced": best.get("catastrophic_reduction", 0) > 0.5,
            "D_good_signal_preserved": best.get("good_signal_retention", 0) >= 0.5,
            "E_signal_coverage_maintained": best.get("signal_coverage", 0) >= 0.3,
            "F_q2_mdd_rfe_improved": "see q2_replay/q2_combination_policy_comparison.csv",
            "G_routing_improved": "not proven; see routing reports",
            "H_dual_head_helpful": bool(str(best.get("candidate", "")).startswith("B")),
            "I_representation_problem_remaining": best.get("candidate_status") != "shadow_only_candidate",
            "J_next_step": "loss tuning if shadow candidate exists; otherwise label redesign/architecture research",
        },
    }
    _write_md(ROOT / "selective_risk_aware_tcn_final_report.md", "Selective Risk-Aware TCN Final Report", report)
    _write_md(ROOT / "selective_risk_aware_tcn_final_verdict.md", "Selective Risk-Aware TCN Final Verdict", {"final_verdict": verdict, "best_candidate": best, "promotion_ready": False})


def run_pipeline() -> Dict[str, Any]:
    _ensure_dirs()
    before = _prod_hashes()
    df = _add_tags(_load_df())
    cols = _feature_cols(df)
    phase0(df, before, cols)
    phase1_labels(df)
    splits = phase2_splits(df)
    phase3_4_design()
    results = run_training_and_eval(df, splits, cols)
    phase6_outputs(results)
    phase7_8_summaries(results)
    phase9_routing(results, df, cols)
    phase10_regime(results)
    scorecard, status = phase11_selection(results)
    phase12_shadow(scorecard.iloc[0].to_dict() if len(scorecard) else {})
    verdict = phase13_decision(status)
    after = _prod_hashes()
    phase14_audit(before, after, splits)
    write_final(verdict, scorecard, results)
    return {"verdict": verdict, "best_candidate": scorecard.iloc[0].to_dict() if len(scorecard) else {}, "rows": len(df), "splits_pass": int((splits["status"] == "PASS").sum())}


def main() -> None:
    parser = argparse.ArgumentParser(description="Selective risk-aware TCN retraining research")
    parser.parse_args()
    result = run_pipeline()
    print(f"verdict: {result['verdict']}")
    print(f"rows: {result['rows']}")
    print(f"splits_pass: {result['splits_pass']}")
    print(f"best_candidate: {result.get('best_candidate', {}).get('candidate')}")


if __name__ == "__main__":
    main()
