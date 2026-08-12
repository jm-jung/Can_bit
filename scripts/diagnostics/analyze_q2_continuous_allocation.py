"""
Q2 continuous risk-allocation optimization (diagnostics only).

Shifts from binary filtering to continuous capital allocation on Q2_PD/Q2_BDI baseline.
Counterfactual replay on production entries — no hard reject, no production changes.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.analyze_meta_v2_recalibration import (
    SCORE_Q2_BDI,
    SCORE_Q2_PD,
    _metrics_row,
    _mdd,
)
from scripts.diagnostics.build_meta_label_dataset import load_v2_dataset
from scripts.diagnostics.validate_h8_soft_risk_gate import _entropy
from scripts.diagnostics.validate_q2_penalty_tournament import apply_penalties, _ohlcv_features as _q2_feat
from scripts.diagnostics.validate_quality_score_replay import (
    POSITION_SIZE,
    _risk_routing,
    map_m3,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper

OUT_DIR = Path("data/diagnostics/q2_continuous")
PENALTY_D = frozenset({"D"})
PENALTY_BDI = frozenset({"B", "D", "I"})
MIN_SCALE, MAX_SCALE = 0.05, 1.0
Q2_BDI_MDD_REF = -0.001454
FALSE_HIGH_ENT = 0.90

ScaleFn = Callable[[float], float]
ModifierFn = Callable[[float, pd.Series, Dict[str, Any], float], float]


def _clamp_scale(s: float) -> float:
    return float(max(MIN_SCALE, min(MAX_SCALE, s)))


def _economic_utility(net: float, mdd: float) -> float:
    return float(net / abs(mdd)) if mdd != 0 else float(net)


def _is_false_high_sig(t: Dict[str, Any], direction: str) -> bool:
    return (
        direction == "LONG"
        and str(t.get("trend_label")) == "up"
        and str(t.get("vol_bucket")) == "high"
        and _entropy(t) <= FALSE_HIGH_ENT
    )


# --- Phase 1: scale mapping variants ---

def scale_discrete_m3(score: float) -> float:
    return float(map_m3(score) or MIN_SCALE)


def scale_continuous_linear(score: float) -> float:
    return _clamp_scale(0.15 + 0.85 * score)


def scale_sigmoid(score: float, k: float = 8.0, mid: float = 0.55) -> float:
    raw = 1.0 / (1.0 + math.exp(-k * (score - mid)))
    return _clamp_scale(0.15 + 0.85 * raw)


def scale_temperature(score: float, temp: float = 1.4) -> float:
    adj = score ** (1.0 / temp) if score > 0 else 0.0
    return _clamp_scale(0.15 + 0.85 * adj)


def scale_vol_aware_base(score: float) -> float:
    return scale_continuous_linear(score)


def scale_vol_compress(score: float, vol_high: bool) -> float:
    base = scale_continuous_linear(score)
    if vol_high:
        return _clamp_scale(base * 0.82)
    return base


# --- Phase 2: risk budget modifiers ---

def mod_none(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    return scale


def mod_entropy(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    ent = float(row.get("entropy", _entropy(t)))
    factor = 1.0 - 0.12 * max(0.0, ent - 0.90)
    return _clamp_scale(scale * factor)


def mod_vol(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    factor = 0.88 if str(row.get("vol_bucket", t.get("vol_bucket"))) == "high" else 1.0
    return _clamp_scale(scale * factor)


def mod_q2_penalty(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    pen = float(row.get("q2_bdi_penalty_score", 0) or 0)
    return _clamp_scale(scale * (1.0 - 0.5 * min(pen, 0.6)))


def mod_false_high_sig(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    if int(row.get("false_high_signature_flag", 0)) or _is_false_high_sig(t, str(row["direction"])):
        return _clamp_scale(scale * 0.65)
    return scale


def mod_confidence_overext(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    co = float(row.get("confidence_overextension", 0) or 0)
    return _clamp_scale(scale * (1.0 - min(co * 8.0, 0.25)))


def mod_dynamic_budget(scale: float, row: pd.Series, t: Dict[str, Any], score: float) -> float:
    ent = float(row.get("entropy", _entropy(t)))
    vol_h = int(row.get("vol_bucket_high", 0)) or (str(row.get("vol_bucket")) == "high")
    pen = float(row.get("q2_bdi_penalty_score", 0) or 0)
    fh = int(row.get("false_high_signature_flag", 0))
    risk = 0.0
    risk += 0.15 * max(0.0, ent - 0.88)
    risk += 0.12 * vol_h
    risk += 0.20 * min(pen, 0.5)
    risk += 0.18 * fh
    return _clamp_scale(scale * (1.0 - min(risk, 0.45)))


MODIFIERS: Dict[str, ModifierFn] = {
    "none": mod_none,
    "entropy": mod_entropy,
    "vol": mod_vol,
    "q2_penalty": mod_q2_penalty,
    "false_high_sig": mod_false_high_sig,
    "confidence_overext": mod_confidence_overext,
    "dynamic_budget": mod_dynamic_budget,
}

SCALE_MAPS: Dict[str, ScaleFn] = {
    "A_discrete_m3": scale_discrete_m3,
    "B_continuous_linear": scale_continuous_linear,
    "C_sigmoid": scale_sigmoid,
    "D_temperature": scale_temperature,
    "E_vol_compression": lambda s: s,  # applied with vol in simulate
}


def _simulate_counterfactual(
    dataset: pd.DataFrame,
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    q2_fn,
    scale_fn: ScaleFn,
    modifier: ModifierFn,
    *,
    vol_aware: bool = False,
    hard_reject_fh: bool = False,
    soft_fh_factor: Optional[float] = None,
) -> pd.DataFrame:
    idx_to_tick = {int(t.get("df_idx", i)): (i, t) for i, t in enumerate(ticks)}
    rows: List[Dict[str, Any]] = []

    for _, row in dataset.sort_values("df_idx").iterrows():
        df_idx = int(row["df_idx"])
        if df_idx not in idx_to_tick:
            continue
        i, t = idx_to_tick[df_idx]
        direction = str(row["direction"])
        score = float(q2_fn(t, feat_map))

        if hard_reject_fh and _is_false_high_sig(t, direction):
            continue

        if vol_aware:
            vol_h = str(t.get("vol_bucket")) == "high"
            scale = scale_vol_compress(score, vol_h)
        else:
            scale = scale_fn(score)

        scale = modifier(scale, row, t, score)

        if soft_fh_factor is not None and _is_false_high_sig(t, direction):
            scale = _clamp_scale(scale * soft_fh_factor)

        net = float(row["net_return"])
        rows.append({
            "entry_idx": i,
            "df_idx": df_idx,
            "direction": direction,
            "quality_score": score,
            "scale": scale,
            "net_return": net,
            "scaled_return": net * scale,
            "exit_reason": row.get("exit_reason", ""),
            "vol_bucket": str(row.get("vol_bucket", "")),
            "false_high_sig": int(_is_false_high_sig(t, direction)),
        })

    return pd.DataFrame(rows)


def _metrics_extended(
    name: str,
    tdf: pd.DataFrame,
    ticks: List[Dict[str, Any]],
    dataset: pd.DataFrame,
    prod_trades: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    row = _metrics_row(name, tdf, ticks, dataset, prod_trades, extra)
    row["economic_utility"] = _economic_utility(row["net_return"], row["MDD"])
    row["scale_utilization"] = float(tdf["scale"].mean()) if not tdf.empty else 0.0
    row["scale_std"] = float(tdf["scale"].std()) if len(tdf) > 1 else 0.0
    if not tdf.empty and "quality_score" in tdf.columns:
        bins = pd.cut(tdf["quality_score"], bins=5, labels=False)
        cal_err = 0.0
        for b in range(5):
            sub = tdf[bins == b]
            if len(sub) < 2:
                continue
            cal_err += abs(sub["quality_score"].mean() - (sub["scaled_return"] > 0).mean())
        row["calibration_error"] = float(cal_err / 5.0)
    return row


def _build_folds(dataset: pd.DataFrame) -> List[Dict[str, Any]]:
    df = dataset.sort_values("df_idx").reset_index(drop=True)
    n = len(df)
    folds: List[Dict[str, Any]] = []
    fid = 0
    for scheme, min_train, test_size, step in [("expanding", 45, 12, 10), ("rolling", 55, 12, 10)]:
        if scheme == "expanding":
            ts = min_train
            while ts + test_size <= n:
                folds.append({
                    "fold_id": f"{scheme}_{fid}",
                    "window_type": scheme,
                    "test": df.iloc[ts : ts + test_size].copy(),
                    "test_start": int(df.iloc[ts]["df_idx"]),
                    "test_end": int(df.iloc[ts + test_size - 1]["df_idx"]),
                })
                fid += 1
                ts += step
        else:
            start = 0
            while start + min_train + test_size <= n:
                te = start + min_train
                folds.append({
                    "fold_id": f"{scheme}_{fid}",
                    "window_type": scheme,
                    "test": df.iloc[te : te + test_size].copy(),
                    "test_start": int(df.iloc[te]["df_idx"]),
                    "test_end": int(df.iloc[te + test_size - 1]["df_idx"]),
                })
                fid += 1
                start += step
    return folds


def _oos_fold_rows(
    folds: List[Dict[str, Any]],
    dataset: pd.DataFrame,
    ticks: List[Dict[str, Any]],
    feat_map: Dict[int, Dict[str, Any]],
    q2_fn,
    candidate: str,
    scale_fn: ScaleFn,
    modifier: ModifierFn,
    **kw,
) -> List[Dict[str, Any]]:
    rows = []
    baseline_mdds = []
    cand_mdds = []
    for fold in folds:
        test = fold["test"]
        tc = len(test)
        base_tdf = _simulate_counterfactual(test, ticks, feat_map, q2_fn, scale_discrete_m3, mod_none)
        cand_tdf = _simulate_counterfactual(test, ticks, feat_map, q2_fn, scale_fn, modifier, **kw)
        base_m = _metrics_extended(f"Q2_BDI_discrete", base_tdf, ticks, dataset, tc, {"fold_id": fold["fold_id"]})
        cand_m = _metrics_extended(candidate, cand_tdf, ticks, dataset, tc, {"fold_id": fold["fold_id"]})
        cand_m["mdd_delta_vs_baseline"] = cand_m["MDD"] - base_m["MDD"]
        cand_m["mdd_improved"] = cand_m["MDD"] >= base_m["MDD"]
        cand_m["preservation_vs_full"] = len(cand_tdf) / max(len(dataset), 1)
        rows.append(cand_m)
        baseline_mdds.append(base_m["MDD"])
        cand_mdds.append(cand_m["MDD"])
    return rows


def _scale_stability(dataset: pd.DataFrame, ticks: List[Dict[str, Any]], feat_map: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    records = []
    for q2_name, q2_fn in [("Q2_PD", SCORE_Q2_PD), ("Q2_BDI", SCORE_Q2_BDI)]:
        tdf = _simulate_counterfactual(dataset, ticks, feat_map, q2_fn, scale_discrete_m3, mod_none)
        if tdf.empty:
            continue
        for _, r in tdf.iterrows():
            records.append({
                "q2_base": q2_name,
                "scale": r["scale"],
                "quality_score": r["quality_score"],
                "direction": r["direction"],
                "vol_bucket": r["vol_bucket"],
                "false_high_sig": r["false_high_sig"],
                "scaled_return": r["scaled_return"],
            })
    df = pd.DataFrame(records)
    if df.empty:
        return {"error": "no trades"}

    analysis: Dict[str, Any] = {"q2_bases": {}}
    for base in df["q2_base"].unique():
        sub = df[df["q2_base"] == base]
        hv = sub[sub["vol_bucket"] == "high"]
        analysis["q2_bases"][base] = {
            "scale_mean": float(sub["scale"].mean()),
            "scale_std": float(sub["scale"].std()),
            "scale_p25": float(sub["scale"].quantile(0.25)),
            "scale_p50": float(sub["scale"].quantile(0.50)),
            "scale_p75": float(sub["scale"].quantile(0.75)),
            "high_vol_scale_mean": float(hv["scale"].mean()) if len(hv) else 0,
            "mid_vol_scale_mean": float(sub[sub["vol_bucket"] == "mid"]["scale"].mean()) if (sub["vol_bucket"] == "mid").any() else 0,
            "long_scale_mean": float(sub[sub["direction"] == "LONG"]["scale"].mean()),
            "short_scale_mean": float(sub[sub["direction"] == "SHORT"]["scale"].mean()),
            "false_high_scale_mean": float(sub[sub["false_high_sig"] == 1]["scale"].mean()) if (sub["false_high_sig"] == 1).any() else 0,
            "non_false_high_scale_mean": float(sub[sub["false_high_sig"] == 0]["scale"].mean()),
        }
    return analysis


def _verdict(
    tournament: pd.DataFrame,
    oos_df: pd.DataFrame,
    dampening: pd.DataFrame,
    n: int,
) -> str:
    bdi_row = tournament[tournament["candidate"] == "Q2_BDI_A_discrete_m3"]
    bdi_mdd = float(bdi_row.iloc[0]["MDD"]) if not bdi_row.empty else Q2_BDI_MDD_REF

    best = tournament[tournament["preservation"] >= 0.90].sort_values("MDD", ascending=False)
    if best.empty:
        return "preservation_instability_detected"

    top = best.iloc[0]
    beats_bdi = top["MDD"] > bdi_mdd
    bdi_fh = int(bdi_row.iloc[0]["false_high"]) if not bdi_row.empty else 3
    if beats_bdi and top["false_high"] <= bdi_fh + 1:
        beats_bdi = True
    else:
        beats_bdi = False

    oos_improved = float(oos_df["mdd_improved"].mean()) if not oos_df.empty and "mdd_improved" in oos_df.columns else 0
    damp_soft = dampening[dampening["mode"] == "B_soft_reduction"] if not dampening.empty else pd.DataFrame()
    fh_effective = False
    if not damp_soft.empty:
        base_fh = int(dampening[dampening["mode"] == "baseline"]["false_high"].iloc[0]) if "baseline" in dampening["mode"].values else 3
        fh_effective = int(damp_soft.iloc[0]["false_high"]) < base_fh and damp_soft.iloc[0]["MDD"] >= bdi_mdd

    if n < 300 and not beats_bdi and oos_improved < 0.5:
        return "Q2_baseline_still_best"
    if fh_effective:
        return "false_high_dampening_effective"
    if beats_bdi and oos_improved > 0.5:
        return "continuous_allocation_viable"
    if top["candidate"].str.contains("dynamic_budget").any() if isinstance(top["candidate"], str) else False:
        return "risk_budget_feature_effective"
    if oos_improved < 0.35:
        return "Q2_baseline_still_best"
    return "needs_better_calibration"


def run_continuous_allocation() -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset, _ = load_v2_dataset()
    ohlcv = load_ohlcv()
    feat_map = _q2_feat(ohlcv)
    ticks, _ = simulate_signals_paper(ohlcv)
    n = len(dataset)
    folds = _build_folds(dataset)

    # Phase 1: scale tournament (Q2_BDI + Q2_PD)
    tournament_rows: List[Dict[str, Any]] = []
    for q2_name, q2_fn in [("Q2_PD", SCORE_Q2_PD), ("Q2_BDI", SCORE_Q2_BDI)]:
        for map_name, scale_fn in [
            ("A_discrete_m3", scale_discrete_m3),
            ("B_continuous_linear", scale_continuous_linear),
            ("C_sigmoid", scale_sigmoid),
            ("D_temperature", scale_temperature),
        ]:
            tdf = _simulate_counterfactual(dataset, ticks, feat_map, q2_fn, scale_fn, mod_none)
            tournament_rows.append(_metrics_extended(
                f"{q2_name}_{map_name}", tdf, ticks, dataset, n, {"q2_base": q2_name, "scale_map": map_name},
            ))
        tdf = _simulate_counterfactual(dataset, ticks, feat_map, q2_fn, scale_continuous_linear, mod_none, vol_aware=True)
        tournament_rows.append(_metrics_extended(
            f"{q2_name}_E_vol_compression", tdf, ticks, dataset, n, {"q2_base": q2_name, "scale_map": "E_vol_compression"},
        ))

    # Phase 2: risk budget on Q2_BDI continuous linear
    for mod_name, mod_fn in MODIFIERS.items():
        if mod_name == "none":
            continue
        tdf = _simulate_counterfactual(dataset, ticks, feat_map, SCORE_Q2_BDI, scale_continuous_linear, mod_fn)
        tournament_rows.append(_metrics_extended(
            f"Q2_BDI_continuous_{mod_name}", tdf, ticks, dataset, n,
            {"q2_base": "Q2_BDI", "modifier": mod_name},
        ))

    tournament_df = pd.DataFrame(tournament_rows)
    tournament_df.to_csv(OUT_DIR / "continuous_scale_tournament.csv", index=False)

    # Phase 3: false-high dampening
    damp_rows = []
    for mode, kwargs in [
        ("baseline", {}),
        ("A_hard_reject", {"hard_reject_fh": True}),
        ("B_soft_reduction", {"soft_fh_factor": 0.55}),
        ("C_dynamic_budget", {}),
    ]:
        if mode == "C_dynamic_budget":
            tdf = _simulate_counterfactual(dataset, ticks, feat_map, SCORE_Q2_BDI, scale_continuous_linear, mod_dynamic_budget)
        else:
            tdf = _simulate_counterfactual(dataset, ticks, feat_map, SCORE_Q2_BDI, scale_discrete_m3, mod_none, **kwargs)
        damp_rows.append(_metrics_extended(
            f"FH_damp_{mode}", tdf, ticks, dataset, n, {"mode": mode},
        ))
    damp_df = pd.DataFrame(damp_rows)
    fh_sub = dataset.copy()
    fh_sub["fh_sig"] = fh_sub.apply(
        lambda r: int(r.get("false_high_signature_flag", 0)), axis=1,
    )
    fh_trades = fh_sub[fh_sub["fh_sig"] == 1]
    damp_analysis = {
        "false_high_trade_count": int(len(fh_trades)),
        "modes": damp_df[["mode", "trades", "preservation", "MDD", "false_high", "RFE", "routing_valid", "economic_utility"]].to_dict(orient="records") if "mode" in damp_df.columns else damp_df.to_dict(orient="records"),
        "good_trade_preservation_soft": float(damp_df.loc[damp_df["mode"] == "B_soft_reduction", "preservation"].iloc[0]) if "mode" in damp_df.columns and (damp_df["mode"] == "B_soft_reduction").any() else 1.0,
    }

    # Phase 5: OOS on top continuous candidates vs discrete baseline
    oos_candidates = [
        ("Q2_BDI_B_continuous_linear", scale_continuous_linear, mod_none, {}),
        ("Q2_BDI_C_sigmoid", scale_sigmoid, mod_none, {}),
        ("Q2_BDI_dynamic_budget", scale_continuous_linear, mod_dynamic_budget, {}),
        ("Q2_BDI_E_vol_compression", scale_continuous_linear, mod_none, {"vol_aware": True}),
    ]
    oos_rows: List[Dict[str, Any]] = []
    for cand, sfn, mfn, kw in oos_candidates:
        oos_rows.extend(_oos_fold_rows(folds, dataset, ticks, feat_map, SCORE_Q2_BDI, cand, sfn, mfn, **kw))
    oos_df = pd.DataFrame(oos_rows)

    # Phase 6: scale stability
    stability = _scale_stability(dataset, ticks, feat_map)

    verdict = _verdict(tournament_df, oos_df, damp_df, n)

    # Risk budget feature analysis
    risk_rows = []
    bdi_tdf = _simulate_counterfactual(dataset, ticks, feat_map, SCORE_Q2_BDI, scale_discrete_m3, mod_none)
    scale_by_idx = bdi_tdf.set_index("df_idx")["scale"].to_dict() if not bdi_tdf.empty else {}
    for feat in ["entropy", "entropy_delta", "trend_strength", "vol_bucket_high", "realized_vol",
                 "q2_bdi_penalty_score", "false_high_signature_flag", "confidence_overextension"]:
        if feat not in dataset.columns:
            continue
        x = dataset[feat].fillna(0).astype(float)
        scales = dataset["df_idx"].map(scale_by_idx).fillna(0).astype(float)
        risk_rows.append({
            "feature": feat,
            "mean": float(x.mean()),
            "std": float(x.std()),
            "corr_bad_trade": float(np.corrcoef(x, dataset["binary_bad_trade"])[0, 1]) if len(x) > 2 else 0,
            "corr_scale_discrete": float(np.corrcoef(x, scales)[0, 1]) if len(x) > 2 and scales.std() > 0 else 0,
        })
    risk_feat_df = pd.DataFrame(risk_rows)

    # Write deliverables
    (OUT_DIR / "q2_continuous_scale_research.md").write_text(
        f"# Q2 Continuous Scale Research\n\n"
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')} | **Rows:** {n}\n\n"
        f"## Phase 1 — Scale mapping tournament\n{tournament_df.to_string(index=False)}\n\n"
        f"## Key question\nHard threshold 없이 continuous sizing만으로 MDD/RFE 개선 가능한가?\n"
        f"→ See tournament MDD vs Q2_BDI_A_discrete_m3\n",
        encoding="utf-8",
    )

    (OUT_DIR / "risk_budget_feature_analysis.md").write_text(
        f"# Risk Budget Features\n\n"
        f"Features as risk budget modifiers (not reject signals).\n\n"
        f"{risk_feat_df.to_string(index=False) if not risk_feat_df.empty else 'N/A'}\n\n"
        f"## Modifiers tested\n{list(MODIFIERS.keys())}\n",
        encoding="utf-8",
    )

    (OUT_DIR / "false_high_dampening_analysis.md").write_text(
        f"# False-High Dampening\n\n```json\n{json.dumps(damp_analysis, indent=2, default=str)}\n```\n\n"
        f"{damp_df.to_string(index=False)}\n",
        encoding="utf-8",
    )

    pres_oos = oos_df.groupby(oos_df.get("candidate", pd.Series())).agg({
        "preservation": "mean", "MDD": "mean", "mdd_improved": "mean",
        "false_high": "mean", "routing_valid": "mean",
    }).reset_index() if not oos_df.empty else pd.DataFrame()

    (OUT_DIR / "preservation_first_oos_report.md").write_text(
        f"# Preservation-First OOS\n\n"
        f"Folds: {len(folds)} | Min preservation target: 90%\n\n"
        f"## Per-candidate OOS aggregate\n{pres_oos.to_string(index=False) if not pres_oos.empty else 'N/A'}\n\n"
        f"## Fold detail rows: {len(oos_df)}\n",
        encoding="utf-8",
    )

    (OUT_DIR / "q2_scale_stability_report.md").write_text(
        f"# Q2 Scale Stability\n\n```json\n{json.dumps(stability, indent=2, default=str)}\n```\n",
        encoding="utf-8",
    )

    (OUT_DIR / "continuous_allocation_final_verdict.md").write_text(
        f"# Final Verdict\n\n**{verdict}**\n\n"
        f"- n={n} (<300: conservative)\n"
        f"- Q2_BDI discrete MDD ref: {Q2_BDI_MDD_REF}\n"
        f"- Best preservation≥90% candidate MDD: {float(tournament_df[tournament_df['preservation']>=0.9]['MDD'].max()) if len(tournament_df[tournament_df['preservation']>=0.9]) else 'N/A'}\n"
        f"- OOS mdd_improved mean: {float(oos_df['mdd_improved'].mean()) if not oos_df.empty else 0:.1%}\n"
        f"- Meta/hard-reject/hybrid: prior reject — Q2 continuous path tested independently\n\n"
        f"## Recommendation\n"
        f"Q2_BDI discrete M3 remains primary baseline unless rows≥300 and OOS confirms continuous beat.\n",
        encoding="utf-8",
    )

    return {
        "verdict": verdict,
        "tournament": tournament_df.to_dict(orient="records"),
        "oos_rows": len(oos_df),
        "output_dir": str(OUT_DIR),
    }


def main() -> None:
    r = run_continuous_allocation()
    print(f"verdict: {r['verdict']}")
    print(f"output: {r['output_dir']}")


if __name__ == "__main__":
    main()
