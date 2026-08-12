#!/usr/bin/env python3
"""
FR2 Regime-Filtered Backtest: regime filter를 실제 진입 규칙으로 적용했을 때 성과 검증.

필터: no_filter, high_vol_only, mid_or_high_vol, uptrend_and_high_vol, downtrend_and_high_vol,
      any_trend_high_vol, uptrend_only, downtrend_only
구간: 720d, 180d, 90d
진입 gating: filter False인 bar에서는 pl/ps를 0으로 두어 flat(진입 안 함) 처리.

사용:
  .venv/bin/python scripts/run_fr2_regime_filtered_backtest.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.60

from scripts.run_fr2_oos_diagnosis import get_fr2_proba, run_bt, _metrics
from scripts.run_fr2_regime_conditioning import add_regime_columns


def entry_mask_no_filter(df_bt: pd.DataFrame) -> np.ndarray:
    return np.ones(len(df_bt), dtype=bool)


def entry_mask_high_vol_only(df_bt: pd.DataFrame) -> np.ndarray:
    return (df_bt["vol_regime"] == "high_vol").fillna(False).values


def entry_mask_mid_or_high_vol(df_bt: pd.DataFrame) -> np.ndarray:
    v = df_bt["vol_regime"].fillna("")
    return ((v == "mid_vol") | (v == "high_vol")).values


def entry_mask_uptrend_and_high_vol(df_bt: pd.DataFrame) -> np.ndarray:
    return (
        (df_bt["trend_regime"] == "uptrend").fillna(False).values
        & (df_bt["vol_regime"] == "high_vol").fillna(False).values
    )


def entry_mask_downtrend_and_high_vol(df_bt: pd.DataFrame) -> np.ndarray:
    return (
        (df_bt["trend_regime"] == "downtrend").fillna(False).values
        & (df_bt["vol_regime"] == "high_vol").fillna(False).values
    )


def entry_mask_any_trend_high_vol(df_bt: pd.DataFrame) -> np.ndarray:
    c = df_bt["cross_regime"].fillna("")
    return ((c == "uptrend_highvol") | (c == "downtrend_highvol")).values


def entry_mask_uptrend_only(df_bt: pd.DataFrame) -> np.ndarray:
    return (df_bt["trend_regime"] == "uptrend").fillna(False).values


def entry_mask_downtrend_only(df_bt: pd.DataFrame) -> np.ndarray:
    return (df_bt["trend_regime"] == "downtrend").fillna(False).values


FILTERS = [
    ("no_filter", entry_mask_no_filter),
    ("high_vol_only", entry_mask_high_vol_only),
    ("mid_or_high_vol", entry_mask_mid_or_high_vol),
    ("uptrend_and_high_vol", entry_mask_uptrend_and_high_vol),
    ("downtrend_and_high_vol", entry_mask_downtrend_and_high_vol),
    ("any_trend_high_vol", entry_mask_any_trend_high_vol),
    ("uptrend_only", entry_mask_uptrend_only),
    ("downtrend_only", entry_mask_downtrend_only),
]


def apply_regime_gate(pl: np.ndarray, ps: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Where mask is False, set pl/ps to 0 so p_flat=1 (no entry)."""
    pl_g = np.where(mask, pl, 0.0).astype(np.float32)
    ps_g = np.where(mask, ps, 0.0).astype(np.float32)
    return pl_g, ps_g


def run_filtered_backtest(
    df_bt: pd.DataFrame,
    pl: np.ndarray,
    ps: np.ndarray,
    future_ret: np.ndarray,
    filter_name: str,
    filter_fn,
    window_days: int,
) -> dict:
    mask = filter_fn(df_bt)
    n_allowed = int(mask.sum())
    if n_allowed < 20:
        return {
            "model_id": "h15_micro_v1",
            "filter_name": filter_name,
            "window_days": window_days,
            "trades": np.nan,
            "signal_density": np.nan,
            "direction_accuracy": np.nan,
            "spearman": np.nan,
            "mean_return_signal": np.nan,
            "cost_on": np.nan,
            "MDD": np.nan,
            "sample_too_small": "yes",
            "entries_allowed": n_allowed,
        }
    pl_g, ps_g = apply_regime_gate(pl, ps, mask)
    res = run_bt(THRESHOLD, df_bt, pl_g, ps_g)
    pl_sub = pl[mask]
    ps_sub = ps[mask]
    fr_sub = future_ret[mask]
    met = _metrics(pl_sub, ps_sub, fr_sub, threshold_signal=THRESHOLD)
    trades = res.get("total_trades") if res else np.nan
    row = {
        "model_id": "h15_micro_v1",
        "filter_name": filter_name,
        "window_days": window_days,
        "trades": trades,
        "signal_density": met["signal_density"],
        "direction_accuracy": met["direction_accuracy"],
        "spearman": met["spearman"],
        "mean_return_signal": met["mean_return_signal"],
        "cost_on": res.get("total_return") if res else np.nan,
        "MDD": res.get("max_drawdown") if res else np.nan,
        "sample_too_small": "yes" if (trades is not None and int(trades) < 100) else "no",
        "entries_allowed": n_allowed,
    }
    if res and "win_rate" in res:
        row["win_rate"] = res["win_rate"]
    return row


def main():
    parser = argparse.ArgumentParser(description="FR2 regime-filtered backtest")
    parser.add_argument("--reports-only", action="store_true", help="Regenerate reports from existing CSV")
    parser.add_argument("--quick", action="store_true", help="90d only, no_filter + high_vol_only (pipeline check)")
    args = parser.parse_args()
    if getattr(args, "reports_only", False):
        write_reports()
        write_decision()
        print("[FR2-Filter] Reports only: done.", flush=True)
        return

    quick = getattr(args, "quick", False)
    if quick:
        filters_use = [(n, f) for n, f in FILTERS if n in ("no_filter", "high_vol_only")]
        windows_use = [90]
        load_days = 90
    else:
        filters_use = FILTERS
        windows_use = [720, 180, 90]
        load_days = 720

    rows = []
    print(f"[FR2-Filter] Loading {load_days}d...", flush=True)
    triple, err = get_fr2_proba(load_days)
    if err:
        print(f"[FR2-Filter] {load_days}d failed: {err}", flush=True)
        for days in windows_use:
            for name, _ in filters_use:
                rows.append({"model_id": "h15_micro_v1", "filter_name": name, "window_days": days, "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan, "sample_too_small": "yes", "entries_allowed": 0})
    else:
        df_bt, pl, ps, future_ret = triple
        pl = np.asarray(pl)
        ps = np.asarray(ps)
        future_ret = np.asarray(future_ret)
        df_bt["timestamp"] = pd.to_datetime(df_bt["timestamp"])
        t_max = df_bt["timestamp"].max()
        for days in windows_use:
            if (quick and days == 90) or (not quick and days == load_days):
                df_w = df_bt.copy()
                pl_w = pl
                ps_w = ps
                fr_w = future_ret
            elif not quick:
                start = t_max - pd.Timedelta(days=days)
                idx = (df_bt["timestamp"] >= start).values
                if idx.sum() < 100:
                    for name, _ in filters_use:
                        rows.append({"model_id": "h15_micro_v1", "filter_name": name, "window_days": days, "trades": np.nan, "signal_density": np.nan, "direction_accuracy": np.nan, "spearman": np.nan, "mean_return_signal": np.nan, "cost_on": np.nan, "MDD": np.nan, "sample_too_small": "yes", "entries_allowed": 0})
                    continue
                df_w = df_bt.loc[idx].copy().reset_index(drop=True)
                pl_w = pl[idx]
                ps_w = ps[idx]
                fr_w = future_ret[idx]
            else:
                continue
            df_w = add_regime_columns(days, df_w)
            for name, fn in filters_use:
                row = run_filtered_backtest(df_w, pl_w, ps_w, fr_w, name, fn, days)
                rows.append(row)
                print(f"  {name} {days}d -> trades={row.get('trades')} cost_on={row.get('cost_on')}", flush=True)

    pd.DataFrame(rows).to_csv(OUT_DIR / "fr2_regime_filtered_backtest.csv", index=False)
    print("[FR2-Filter] Wrote fr2_regime_filtered_backtest.csv", flush=True)
    write_reports()
    write_decision()
    print("[FR2-Filter] Done.", flush=True)


def write_reports():
    p = OUT_DIR / "fr2_regime_filtered_backtest.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    lines = [
        "# FR2 Regime-Filtered Backtest Report\n",
        "Entry gating: filter=False인 bar에서는 진입 불가(flat). Threshold=0.60.\n",
        "| model_id | filter_name | window_days | trades | signal_density | direction_accuracy | spearman | mean_return_signal | cost_on | MDD | sample_too_small | entries_allowed |",
        "|----------|-------------|-------------|--------|----------------|--------------------|----------|--------------------|---------|-----|------------------|-----------------|",
    ]
    for _, r in df.iterrows():
        tr = r.get("trades", "")
        sd = float(r["signal_density"]) if pd.notna(r.get("signal_density")) else ""
        acc = float(r["direction_accuracy"]) if pd.notna(r.get("direction_accuracy")) else ""
        sp = float(r["spearman"]) if pd.notna(r.get("spearman")) else ""
        mrs = float(r["mean_return_signal"]) if pd.notna(r.get("mean_return_signal")) else ""
        co = float(r["cost_on"]) if pd.notna(r.get("cost_on")) else ""
        mdd = float(r["MDD"]) if pd.notna(r.get("MDD")) else ""
        lines.append(f"| {r['model_id']} | {r['filter_name']} | {r['window_days']} | {tr} | {sd} | {acc} | {sp} | {mrs} | {co} | {mdd} | {r.get('sample_too_small','')} | {r.get('entries_allowed','')} |")
    (OUT_DIR / "FR2_REGIME_FILTERED_BACKTEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("[FR2-Filter] Wrote FR2_REGIME_FILTERED_BACKTEST_REPORT.md", flush=True)


def write_decision():
    p = OUT_DIR / "fr2_regime_filtered_backtest.csv"
    if not p.exists():
        (OUT_DIR / "FR2_FILTER_DECISION.md").write_text("# FR2 Filter Decision\n\nNo data. Run: .venv/bin/python scripts/run_fr2_regime_filtered_backtest.py\n", encoding="utf-8")
        return
    df = pd.read_csv(p)
    no_filter_90 = df[(df["filter_name"] == "no_filter") & (df["window_days"] == 90)]
    hv_90 = df[(df["filter_name"] == "high_vol_only") & (df["window_days"] == 90)]
    no_filter_180 = df[(df["filter_name"] == "no_filter") & (df["window_days"] == 180)]
    hv_180 = df[(df["filter_name"] == "high_vol_only") & (df["window_days"] == 180)]
    no_filter_720 = df[(df["filter_name"] == "no_filter") & (df["window_days"] == 720)]
    hv_720 = df[(df["filter_name"] == "high_vol_only") & (df["window_days"] == 720)]

    def _float0(ser):
        if ser is None or len(ser) == 0: return None
        v = ser.iloc[0]
        return float(v) if pd.notna(v) else None
    c90_nf = _float0(no_filter_90["cost_on"]) if len(no_filter_90) else None
    c90_hv = _float0(hv_90["cost_on"]) if len(hv_90) else None
    c180_nf = _float0(no_filter_180["cost_on"]) if len(no_filter_180) else None
    c180_hv = _float0(hv_180["cost_on"]) if len(hv_180) else None
    c720_nf = _float0(no_filter_720["cost_on"]) if len(no_filter_720) else None
    c720_hv = _float0(hv_720["cost_on"]) if len(hv_720) else None

    verdict = "STILL_NOT_ROBUST"
    if c90_hv is not None and c90_nf is not None:
        if c90_hv > 0.006 and c180_hv is not None and c180_hv > c180_nf and c720_hv is not None and c720_hv > 0.5:
            verdict = "FILTERED_STRATEGY_CANDIDATE"
        elif c90_hv > c90_nf and c180_hv is not None and c180_hv > c180_nf:
            verdict = "CONDITIONAL_EDGE_REQUIRES_GATING"
        elif c90_hv > c90_nf:
            verdict = "CONDITIONAL_EDGE_REQUIRES_GATING"
    if c90_hv is not None and c90_hv <= 0 and c720_hv is not None and c720_hv < 0.5:
        verdict = "RESEARCH_SIGNAL_ONLY"

    sections = [
        "# FR2 Filter Decision\n",
        "## 1. high_vol_only 필터가 recent 90d 손실을 줄이는가?\n",
        f"- no_filter 90d cost_on: {c90_nf}\n",
        f"- high_vol_only 90d cost_on: {c90_hv}\n",
        "→ 개선 여부: high_vol_only 90d cost_on이 no_filter보다 크면 줄어든 것.\n",
        "## 2. high_vol_only 필터가 180d에서도 개선을 만드는가?\n",
        f"- no_filter 180d: {c180_nf}, high_vol_only 180d: {c180_hv}\n",
        "## 3. high_vol 필터가 720d 성과를 지나치게 훼손하지 않는가?\n",
        f"- no_filter 720d: {c720_nf}, high_vol_only 720d: {c720_hv}\n",
        "## 4. uptrend/downtrend를 더 얹으면 성과가 더 안정화되는가?\n",
        "→ uptrend_and_high_vol, downtrend_and_high_vol, any_trend_high_vol 행 비교.\n",
        "## 5. trades 감소를 감안해도 실전 후보로 볼 수 있는가?\n",
        "→ sample_too_small=no 이고 trades>=100 근처인 필터만 실전 고려.\n",
        "\n## 최종 판정\n",
        f"**{verdict}**\n",
        "(FILTERED_STRATEGY_CANDIDATE | CONDITIONAL_EDGE_REQUIRES_GATING | STILL_NOT_ROBUST | RESEARCH_SIGNAL_ONLY)\n",
    ]
    (OUT_DIR / "FR2_FILTER_DECISION.md").write_text("".join(sections), encoding="utf-8")
    print("[FR2-Filter] Wrote FR2_FILTER_DECISION.md", flush=True)


if __name__ == "__main__":
    main()
