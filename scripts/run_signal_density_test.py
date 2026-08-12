#!/usr/bin/env python3
"""
Signal Density Test for TCN h15_t0p004.

목표:
- 현재 TCN 모델의 directional signal이 실제로 존재하는지, 아니면 통계적 노이즈인지 진단.
- 새로운 모델/전략/룰 없이, 기존 D14 alignment raw + OHLC만 사용.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

SYMBOL = "BTCUSDT"
TIMEFRAME = "5m"
END_DATE = "2026-03-03"
DAYS = 720

RAW_PATH = PROJECT_ROOT / "data" / "diagnostics" / "d14_alignment" / "d14_alignment_raw_720d.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "signal_density"


def _load_ohlcv_range_5m(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame | None:
    csv_path = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
    if symbol.upper() != "BTCUSDT" or not csv_path.exists():
        return None
    chunks = []
    for chunk in pd.read_csv(csv_path, parse_dates=["timestamp"], chunksize=60_000):
        if chunk["timestamp"].min() > end_ts:
            break
        if chunk["timestamp"].max() < start_ts:
            continue
        chunk = chunk[(chunk["timestamp"] >= start_ts) & (chunk["timestamp"] < end_ts)]
        if len(chunk) > 0:
            chunks.append(chunk)
    if not chunks:
        return None
    return pd.concat(chunks, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def load_base_dataset() -> pd.DataFrame:
    """
    Load D14 alignment raw (720d) for h15_t0p004 and ensure required columns.
    """
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"D14 raw parquet not found: {RAW_PATH}")
    df = pd.read_parquet(RAW_PATH)
    # Ensure required columns
    required = [
        "timestamp",
        "future_return_15bar",
        "p_flat",
        "p_long",
        "p_short",
        "max_proba",
        "entropy",
        "argmax_class",
        "long_edge",
        "short_edge",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in raw dataset: {missing}")
    # Direction sign and cost-adjusted proxy (assume cost=0.001)
    df["direction"] = np.where(df["argmax_class"] == 1, 1, np.where(df["argmax_class"] == 2, -1, 0))
    df["cost_adj_return"] = np.where(
        df["direction"] == 1,
        df["future_return_15bar"] - 0.001,
        np.where(df["direction"] == 2, -df["future_return_15bar"] - 0.001, 0.0),
    )
    return df.reset_index(drop=True)


def run_confidence_accuracy(df: pd.DataFrame, out_dir: Path) -> None:
    """
    [3] Directional Accuracy by Confidence (max_proba buckets).
    Buckets:
      0.33–0.40, 0.40–0.45, 0.45–0.50, 0.50–0.55, 0.55–0.60, 0.60–0.65, 0.65+
    """
    bins = [0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.01]
    labels = ["0.33-0.40", "0.40-0.45", "0.45-0.50", "0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65+"]
    df = df.copy()
    df["conf_bucket"] = pd.cut(df["max_proba"], bins=bins, labels=labels, right=False, include_lowest=True)

    rows = []
    y = df["future_return_15bar"].values
    for label in labels:
        sub = df[df["conf_bucket"] == label]
        if len(sub) < 20:
            continue
        direction = sub["direction"].values
        sign_y = np.sign(sub["future_return_15bar"].values)
        mask_nonflat = direction != 0
        if mask_nonflat.any():
            acc = (np.sign(sign_y[mask_nonflat]) == direction[mask_nonflat]).mean()
        else:
            acc = np.nan
        rows.append(
            {
                "conf_bucket": label,
                "count": len(sub),
                "direction_accuracy": acc,
                "mean_future_return": sub["future_return_15bar"].mean(),
                "median_future_return": sub["future_return_15bar"].median(),
                "mean_cost_adj_return": sub["cost_adj_return"].mean(),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "confidence_accuracy.csv", index=False)
    print(f"[SignalDensity] Wrote confidence_accuracy.csv (rows={len(out)})", flush=True)


def run_direction_accuracy_baseline(df: pd.DataFrame, out_dir: Path) -> None:
    """
    [4] Directional Accuracy vs Random.
    - accuracy_model
    - accuracy_random
    - accuracy_long_only
    - accuracy_short_only
    """
    y = df["future_return_15bar"].values
    sign_y = np.sign(y)
    n = len(df)

    # Model argmax direction
    direction_model = df["direction"].values
    mask_nonflat = direction_model != 0
    if mask_nonflat.any():
        acc_model = (np.sign(sign_y[mask_nonflat]) == direction_model[mask_nonflat]).mean()
    else:
        acc_model = np.nan

    rng = np.random.default_rng(42)
    # Random directions in {-1, 0, 1}
    direction_random = rng.integers(-1, 2, size=n)
    mask_nr = direction_random != 0
    acc_random = (np.sign(sign_y[mask_nr]) == direction_random[mask_nr]).mean() if mask_nr.any() else np.nan

    # Always LONG
    direction_long = np.ones(n)
    acc_long = (np.sign(sign_y) == direction_long).mean()

    # Always SHORT
    direction_short = -np.ones(n)
    acc_short = (np.sign(sign_y) == direction_short).mean()

    out = pd.DataFrame(
        [
            {"model": "TCN", "accuracy": acc_model},
            {"model": "Random", "accuracy": acc_random},
            {"model": "Always_LONG", "accuracy": acc_long},
            {"model": "Always_SHORT", "accuracy": acc_short},
        ]
    )
    out.to_csv(out_dir / "direction_accuracy_baseline.csv", index=False)
    print(f"[SignalDensity] Wrote direction_accuracy_baseline.csv", flush=True)


def run_signal_density(df: pd.DataFrame, out_dir: Path) -> None:
    """
    [5] Signal Density:
    usable signal = max_proba >= 0.60 OR entropy <= p10
    """
    total_rows = len(df)
    ent_p10 = df["entropy"].quantile(0.10)
    signal_mask = (df["max_proba"] >= 0.60) | (df["entropy"] <= ent_p10)
    signal_count = int(signal_mask.sum())
    signal_density = signal_count / total_rows if total_rows > 0 else 0.0
    mean_return_signal = df.loc[signal_mask, "future_return_15bar"].mean() if signal_count > 0 else np.nan
    mean_return_all = df["future_return_15bar"].mean()
    out = pd.DataFrame(
        [
            {
                "total_rows": total_rows,
                "signal_count": signal_count,
                "signal_density": signal_density,
                "mean_return_signal": mean_return_signal,
                "mean_return_all": mean_return_all,
            }
        ]
    )
    out.to_csv(out_dir / "signal_density_summary.csv", index=False)
    print(f"[SignalDensity] Wrote signal_density_summary.csv", flush=True)


def run_regime_signal_density(df: pd.DataFrame, out_dir: Path) -> None:
    """
    [6] Regime-specific Density for vol_q2 subset.
    - vol = rolling_std_20 of close from OHLCV
    - vol tertiles -> vol_q0/1/2
    """
    # Load OHLCV for vol calculation
    start_ts = pd.Timestamp(END_DATE).tz_localize("UTC") - pd.Timedelta(days=DAYS)
    end_ts = pd.Timestamp(END_DATE).tz_localize("UTC") + pd.Timedelta(days=1)
    start_naive = start_ts.tz_localize(None)
    end_naive = end_ts.tz_localize(None)
    ohlcv = _load_ohlcv_range_5m(SYMBOL, start_naive, end_naive)
    if ohlcv is None or ohlcv.empty:
        print("[SignalDensity] No OHLCV for regime density; writing empty file.", flush=True)
        pd.DataFrame(
            columns=[
                "regime",
                "total_rows",
                "signal_count",
                "signal_density",
                "direction_accuracy",
                "mean_return_all",
                "mean_return_signal",
            ]
        ).to_csv(out_dir / "regime_signal_density.csv", index=False)
        return
    ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
    ohlcv = ohlcv.sort_values("timestamp")
    ohlcv["rolling_std_20"] = ohlcv["close"].rolling(window=20, min_periods=1).std()
    vol_df = ohlcv[["timestamp", "rolling_std_20"]].copy()

    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
    merged = pd.merge_asof(
        df_ts.sort_values("timestamp"),
        vol_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    # Vol tertiles
    try:
        merged["vol_q"] = pd.qcut(merged["rolling_std_20"], q=3, labels=[0, 1, 2], duplicates="drop")
    except Exception:
        merged["vol_q"] = 0

    ent_p10 = merged["entropy"].quantile(0.10)
    signal_mask_all = (merged["max_proba"] >= 0.60) | (merged["entropy"] <= ent_p10)

    rows = []
    sub = merged[merged["vol_q"] == 2]  # vol_q2
    if not sub.empty:
        total_rows = len(sub)
        signal_mask = signal_mask_all.loc[sub.index]
        signal_count = int(signal_mask.sum())
        signal_density = signal_count / total_rows if total_rows > 0 else 0.0
        direction = sub["direction"].values
        sign_y = np.sign(sub["future_return_15bar"].values)
        mask_nonflat = direction != 0
        if mask_nonflat.any():
            acc = (np.sign(sign_y[mask_nonflat]) == direction[mask_nonflat]).mean()
        else:
            acc = np.nan
        mean_return_all = sub["future_return_15bar"].mean()
        mean_return_signal = (
            sub.loc[signal_mask, "future_return_15bar"].mean() if signal_count > 0 else np.nan
        )
        rows.append(
            {
                "regime": "vol_q2",
                "total_rows": total_rows,
                "signal_count": signal_count,
                "signal_density": signal_density,
                "direction_accuracy": acc,
                "mean_return_all": mean_return_all,
                "mean_return_signal": mean_return_signal,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "regime_signal_density.csv", index=False)
    print(f"[SignalDensity] Wrote regime_signal_density.csv", flush=True)


def write_summary(out_dir: Path) -> None:
    """
    [7] signal_density_summary.md
    - 전체 directional accuracy
    - confidence bucket accuracy
    - signal density
    - vol_q2 subset signal density
    - 판정: EDGE_PRESENT / NO_SIGNAL
    """
    md_path = out_dir / "signal_density_summary.md"

    acc_path = out_dir / "direction_accuracy_baseline.csv"
    conf_path = out_dir / "confidence_accuracy.csv"
    dens_path = out_dir / "signal_density_summary.csv"
    reg_path = out_dir / "regime_signal_density.csv"

    acc_df = pd.read_csv(acc_path) if acc_path.exists() else pd.DataFrame()
    conf_df = pd.read_csv(conf_path) if conf_path.exists() else pd.DataFrame()
    dens_df = pd.read_csv(dens_path) if dens_path.exists() else pd.DataFrame()
    reg_df = pd.read_csv(reg_path) if reg_path.exists() else pd.DataFrame()

    acc_model = acc_random = None
    if not acc_df.empty:
        row_m = acc_df[acc_df["model"] == "TCN"]
        row_r = acc_df[acc_df["model"] == "Random"]
        if len(row_m):
            acc_model = float(row_m["accuracy"].iloc[0])
        if len(row_r):
            acc_random = float(row_r["accuracy"].iloc[0])

    density = None
    mean_ret_sig = None
    mean_ret_all = None
    if not dens_df.empty:
        density = float(dens_df["signal_density"].iloc[0])
        mean_ret_sig = float(dens_df["mean_return_signal"].iloc[0])
        mean_ret_all = float(dens_df["mean_return_all"].iloc[0])

    reg_row = reg_df[reg_df["regime"] == "vol_q2"].iloc[0] if not reg_df.empty else None

    # 판정 로직 (간단하지만 보수적으로)
    edge_present = False
    if acc_model is not None and acc_random is not None and acc_model > acc_random + 0.01:
        edge_present = True
    if density is not None and density > 0 and mean_ret_sig is not None and mean_ret_sig > 0:
        edge_present = True
    verdict = "EDGE_PRESENT" if edge_present else "NO_SIGNAL"

    lines = [
        "# Signal Density Test: 요약",
        "",
        "## 1. 전체 directional accuracy",
        "",
    ]
    if acc_model is not None and acc_random is not None:
        lines.append(f"- TCN: accuracy={acc_model:.4f}")
        lines.append(f"- Random: accuracy={acc_random:.4f}")
    else:
        lines.append("- (accuracy 데이터 없음)")
    lines.extend(["", "## 2. Confidence bucket accuracy", ""])
    if not conf_df.empty:
        for _, r in conf_df.iterrows():
            lines.append(
                f"- {r['conf_bucket']}: count={int(r['count'])}, "
                f"direction_accuracy={r['direction_accuracy']:.4f} "
                f"(mean_return={r['mean_future_return']:.6f})"
            )
    else:
        lines.append("- (confidence_accuracy.csv 없음)")
    lines.extend(["", "## 3. Signal density", ""])
    if density is not None:
        lines.append(
            f"- total_rows={int(dens_df['total_rows'].iloc[0])}, signal_count={int(dens_df['signal_count'].iloc[0])}, "
            f"signal_density={density:.4f}"
        )
        lines.append(
            f"- mean_return_signal={mean_ret_sig:.6f}, mean_return_all={mean_ret_all:.6f}"
        )
    else:
        lines.append("- (signal_density_summary.csv 없음)")
    lines.extend(["", "## 4. vol_q2 subset signal density", ""])
    if reg_row is not None:
        lines.append(
            f"- vol_q2: total_rows={int(reg_row['total_rows'])}, "
            f"signal_density={reg_row['signal_density']:.4f}, "
            f"direction_accuracy={reg_row['direction_accuracy']:.4f}, "
            f"mean_return_all={reg_row['mean_return_all']:.6f}, "
            f"mean_return_signal={reg_row['mean_return_signal']:.6f}"
        )
    else:
        lines.append("- (regime_signal_density.csv 없음 또는 vol_q2 행 없음)")
    lines.extend(
        [
            "",
            "## 5. 모델 signal이 실제 edge인지 여부",
            "",
            f"- 최종 판정: **{verdict}**",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SignalDensity] Wrote {md_path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Signal Density Test for TCN h15_t0p004")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_base_dataset()
    print(f"[SignalDensity] Loaded raw dataset rows={len(df)}", flush=True)

    run_confidence_accuracy(df, out_dir)
    run_direction_accuracy_baseline(df, out_dir)
    run_signal_density(df, out_dir)
    run_regime_signal_density(df, out_dir)
    write_summary(out_dir)
    print("[SignalDensity] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

