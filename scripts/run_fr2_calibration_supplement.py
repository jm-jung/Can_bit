#!/usr/bin/env python3
"""
Platt / Isotonic calibrated score의 eval 구간 min, max, quantile 계산.
0 trades 원인 확인용. 결과를 FR2_CALIBRATION_REPORT.md 끝에 추가.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
REPORT_PATH = OUT_DIR / "FR2_CALIBRATION_REPORT.md"

from scripts.run_fr2_calibration import (  # type: ignore
    BASELINE_PT,
    FR2_PT,
    DAYS,
    WINDOW_SIZE,
    HORIZON,
    FIT_RATIO,
    STRATEGIES,
    _align_by_timestamp,
    _score_series,
    get_ohlcv_and_proba,
    add_regime_columns,
    build_fr2_c4_mask,
    EnsembleInputs,
    build_ensemble_proba,
)

def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    print("[FR2-CAL-SUP] Loading data...", flush=True)
    triple_base, _ = get_ohlcv_and_proba(DAYS, BASELINE_PT, "base", False, temperature=1.0)
    triple_fr2, _ = get_ohlcv_and_proba(DAYS, FR2_PT, "microstructure_v1", True, temperature=1.0)
    if triple_base is None or triple_fr2 is None:
        raise RuntimeError("get_ohlcv_and_proba failed")
    df_b, pl_b, ps_b, _ = triple_base
    df_f, pl_f, ps_f, _ = triple_fr2
    df_bt, pl_b, ps_b, pl_f, ps_f = _align_by_timestamp(df_b, pl_b, ps_b, df_f, pl_f, ps_f)
    n_total = len(df_bt)
    valid_len = n_total - WINDOW_SIZE - HORIZON
    fit_end = int(n_total * FIT_RATIO)
    fit_end_v = min(fit_end, valid_len)
    eval_start = fit_end

    close = df_bt["close"].to_numpy(dtype=float)
    base = close[WINDOW_SIZE : WINDOW_SIZE + valid_len]
    fut = close[WINDOW_SIZE + HORIZON : WINDOW_SIZE + valid_len + HORIZON]
    fit_future_ret = (fut / base - 1.0)[:valid_len]

    df_bt_reg = add_regime_columns(DAYS, df_bt)
    base_c4 = (
        (df_bt_reg["trend_regime"] == "uptrend")
        & (df_bt_reg["vol_regime"] == "high_vol")
    ).to_numpy()
    c4_mask = build_fr2_c4_mask(base_c4, persistence_bars=6)
    ensemble_inputs = EnsembleInputs(
        pl_base=pl_b, ps_base=ps_b, pl_fr2=pl_f, ps_fr2=ps_f, c4_active=c4_mask
    )

    def platt_transform(score: np.ndarray, lr: LogisticRegression) -> np.ndarray:
        s = np.clip(score, 1e-6, 1.0 - 1e-6)
        logit = np.log(s / (1 - s)).reshape(-1, 1)
        return lr.predict_proba(logit)[:, 1]

    lines: list[str] = []
    for mode, name in STRATEGIES:
        pl_full, ps_full = build_ensemble_proba(ensemble_inputs, mode=mode)  # type: ignore[arg-type]
        score_fit = _score_series(pl_full[:fit_end_v], ps_full[:fit_end_v])
        outcome = (fit_future_ret[:fit_end_v] > 0).astype(float) if fit_end_v > 0 else np.zeros(fit_end_v)
        lr = LogisticRegression(C=1e10, max_iter=500)
        logit_fit = np.log(np.clip(score_fit, 1e-6, 1 - 1e-6) / (1 - np.clip(score_fit, 1e-6, 1 - 1e-6))).reshape(-1, 1)
        lr.fit(logit_fit, outcome)
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(score_fit, outcome)

        pl_eval = pl_full[eval_start:]
        ps_eval = ps_full[eval_start:]
        score_eval = _score_series(pl_eval, ps_eval)
        platt_cal = platt_transform(score_eval, lr)
        iso_cal = np.clip(iso.predict(score_eval), 0.0, 1.0)

        for label, arr in [("platt_scaled", platt_cal), ("isotonic_scaled", iso_cal)]:
            q = np.nanquantile(arr, [0.1, 0.5, 0.9])
            lines.append(f"| {name} | {label} | {float(np.min(arr)):.4f} | {float(np.max(arr)):.4f} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} |")

    section = [
        "",
        "### Calibrated score (eval) min / max / quantiles",
        "",
        "| strategy | calibration_method | min | max | q0.1 | q0.5 | q0.9 |",
        "|----------|-------------------|-----|-----|------|------|------|",
    ] + lines + [""]

    report_path = REPORT_PATH
    if report_path.exists():
        orig = report_path.read_text(encoding="utf-8")
        report_path.write_text(orig.rstrip() + "\n" + "\n".join(section) + "\n", encoding="utf-8")
        print("[FR2-CAL-SUP] Appended section 8 to FR2_CALIBRATION_REPORT.md", flush=True)
    else:
        (OUT_DIR / "FR2_CALIBRATION_SUPPLEMENT.md").write_text(
            "# FR2 Calibration Supplement\n\n" + "\n".join(section) + "\n", encoding="utf-8"
        )
        print("[FR2-CAL-SUP] Wrote FR2_CALIBRATION_SUPPLEMENT.md (report not found)", flush=True)


if __name__ == "__main__":
    main()
