"""
Survivor trade expectancy leak forensics (diagnostics only).

Compares baseline vs entropy<=0.96 survivor trades; finds what still drags expectancy.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnostics.validate_entry_quality_filters import (
    FilterSpec,
    _entropy,
    _margin,
    _simulate,
)
from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper


OUT_DIR = Path("data/diagnostics/survivor_forensics")
FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
POSITION_SIZE = 0.05
SURVIVOR_SPEC = FilterSpec("entropy_le_0.96", "entropy_sweep", {"entropy_ub": 0.96})
BASELINE_SPEC = FilterSpec("baseline_current", "baseline", {})


def _expectancy(vals: List[float]) -> float:
    if not vals:
        return 0.0
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals)
    avg_w = mean(wins) if wins else 0.0
    avg_l = mean(losses) if losses else 0.0
    return (wr * avg_w) - ((1.0 - wr) * abs(avg_l))


def _profit_factor(vals: List[float]) -> float:
    wins = sum(x for x in vals if x > 0)
    losses = abs(sum(x for x in vals if x < 0))
    return wins / losses if losses > 0 else 0.0


def _segment_stats(df: pd.DataFrame, label: str) -> Dict[str, Any]:
    vals = df["net_return"].tolist() if not df.empty else []
    return {
        "segment": label,
        "trade_count": len(df),
        "win_rate": float((df["net_return"] > 0).mean()) if len(df) else 0.0,
        "expectancy": _expectancy(vals),
        "profit_factor": _profit_factor(vals),
        "avg_return": float(np.mean(vals)) if vals else 0.0,
        "median_return": float(np.median(vals)) if vals else 0.0,
        "avg_hold": float(df["hold_bars"].mean()) if len(df) else 0.0,
        "avg_entropy": float(df["entropy"].mean()) if len(df) and "entropy" in df.columns else np.nan,
        "avg_margin": float(df["margin"].mean()) if len(df) and "margin" in df.columns else np.nan,
    }


def _ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    px = pd.to_numeric(df["close"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    o = pd.to_numeric(df["open"], errors="coerce")
    ema = px.ewm(span=20, adjust=False).mean()
    rr24 = px.pct_change(24)
    return pd.DataFrame(
        {
            "df_idx": np.arange(len(df)),
            "price_vs_ema": (px > ema).astype(int),
            "recent_return_24": rr24,
        }
    )


def _enrich_trades(trades: pd.DataFrame, ticks: List[Dict[str, Any]], ohlcv_feat: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    rows: List[Dict[str, Any]] = []
    feat_by_idx = ohlcv_feat.set_index("df_idx").to_dict(orient="index")
    for _, tr in trades.iterrows():
        i = int(tr["entry_idx"])
        t = ticks[i]
        ent = _entropy(t)
        mar = _margin(t)
        df_idx = int(t.get("df_idx", i))
        extra = feat_by_idx.get(df_idx, {})
        rr24 = extra.get("recent_return_24", np.nan)
        if pd.isna(rr24):
            rr_bucket = "neutral"
        elif float(rr24) < -0.001:
            rr_bucket = "negative"
        elif float(rr24) > 0.001:
            rr_bucket = "positive"
        else:
            rr_bucket = "neutral"
        pve = extra.get("price_vs_ema", np.nan)
        ema_state = "above_ema" if pd.notna(pve) and int(pve) == 1 else "below_ema"
        rows.append(
            {
                **tr.to_dict(),
                "entropy": ent,
                "margin": mar,
                "trend_state": str(t.get("trend_label") or ""),
                "vol_state": str(t.get("vol_bucket") or ""),
                "ema_state": ema_state,
                "recent_return_24_bucket": rr_bucket,
                "recent_return_24": float(rr24) if pd.notna(rr24) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _entropy_bucket(e: float) -> str:
    if e < 0.80:
        return "<0.80"
    if e < 0.85:
        return "0.80~0.85"
    if e < 0.90:
        return "0.85~0.90"
    if e < 0.95:
        return "0.90~0.95"
    if e <= 0.96:
        return "0.95~0.96"
    return ">0.96"


def _margin_bucket(m: float) -> str:
    if m < 0.03:
        return "0.00~0.03"
    if m < 0.05:
        return "0.03~0.05"
    if m < 0.07:
        return "0.05~0.07"
    if m < 0.10:
        return "0.07~0.10"
    return "0.10+"


def _bucket_expectancy(trades: pd.DataFrame, col: str, bucket_fn: Callable[[Any], str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["_bucket"] = tmp[col].apply(bucket_fn)
    rows = []
    for b, g in tmp.groupby("_bucket", sort=False):
        vals = g["net_return"].tolist()
        rows.append(
            {
                "feature": col,
                "bucket": b,
                "trade_count": len(g),
                "expectancy": _expectancy(vals),
                "profit_factor": _profit_factor(vals),
                "win_rate": float((g["net_return"] > 0).mean()),
                "avg_return": float(np.mean(vals)) if vals else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _signature_key(row: pd.Series) -> str:
    return (
        f"{row['direction']}|entropy {_entropy_bucket(float(row['entropy']))}|"
        f"margin {_margin_bucket(float(row['margin']))}|trend {row['trend_state']}|vol {row['vol_state']}"
    )


def _group_signatures(trades: pd.DataFrame, top_n: int, ascending: bool) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["signature"] = tmp.apply(_signature_key, axis=1)
    total_loss = float(trades.loc[trades["net_return"] < 0, "net_return"].sum())
    rows = []
    for sig, g in tmp.groupby("signature"):
        vals = g["net_return"].tolist()
        loss_sum = float(g.loc[g["net_return"] < 0, "net_return"].sum())
        rows.append(
            {
                "signature": sig,
                "trade_count": len(g),
                "expectancy": _expectancy(vals),
                "profit_factor": _profit_factor(vals),
                "win_rate": float((g["net_return"] > 0).mean()),
                "loss_contribution": (loss_sum / total_loss) if total_loss < 0 else 0.0,
                "net_sum": float(np.sum(vals)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["expectancy", "trade_count"], ascending=[ascending, False]).head(top_n)
    return out.reset_index(drop=True)


def _residual_trim_analysis(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    base_exp = _expectancy(trades["net_return"].tolist())
    sorted_df = trades.sort_values("net_return").reset_index(drop=True)
    n = len(sorted_df)
    rows = []
    for pct in (0.20, 0.30, 0.40):
        k = max(1, int(n * pct))
        remain = sorted_df.iloc[k:]
        vals = remain["net_return"].tolist()
        rows.append(
            {
                "trim_pct_worst": pct,
                "trades_removed": k,
                "trades_remaining": len(remain),
                "baseline_expectancy": base_exp,
                "trimmed_expectancy": _expectancy(vals),
                "expectancy_lift": _expectancy(vals) - base_exp,
                "remaining_pf": _profit_factor(vals),
            }
        )
    return pd.DataFrame(rows)


def _candidate_rules() -> List[Tuple[str, Callable[[pd.Series], bool]]]:
    return [
        ("margin_ge_0.08", lambda r: float(r["margin"]) >= 0.08),
        ("margin_ge_0.07", lambda r: float(r["margin"]) >= 0.07),
        ("margin_ge_0.06", lambda r: float(r["margin"]) >= 0.06),
        ("trend_not_sideways", lambda r: str(r["trend_state"]) != "sideways"),
        ("vol_not_high", lambda r: str(r["vol_state"]) != "high"),
        ("LONG_margin_ge_0.07", lambda r: r["direction"] == "LONG" and float(r["margin"]) >= 0.07),
        ("LONG_entropy_le_0.92", lambda r: r["direction"] == "LONG" and float(r["entropy"]) <= 0.92),
        ("LONG_trend_up", lambda r: r["direction"] == "LONG" and str(r["trend_state"]) == "up"),
        ("entropy_le_0.90", lambda r: float(r["entropy"]) <= 0.90),
        ("entropy_0.85_0.90", lambda r: 0.85 <= float(r["entropy"]) < 0.90),
        ("LONG_vol_mid", lambda r: r["direction"] == "LONG" and str(r["vol_state"]) == "mid"),
        ("LONG_not_high_vol_margin_ge_0.05", lambda r: r["direction"] == "LONG" and str(r["vol_state"]) != "high" and float(r["margin"]) >= 0.05),
    ]


def _score_candidates(survivor: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    base_vals = survivor["net_return"].tolist()
    base_exp = _expectancy(base_vals)
    base_net = float(np.sum(base_vals))
    base_pf = _profit_factor(base_vals)
    base_ks = int((survivor["exit_reason"] == "kill_switch_close").sum()) if "exit_reason" in survivor.columns else 0
    base_rfe = int((survivor["exit_reason"] == "risk_force_exit").sum()) if "exit_reason" in survivor.columns else 0
    n_surv = max(len(survivor), 1)

    rows = []
    for name, fn in _candidate_rules():
        mask = survivor.apply(fn, axis=1)
        kept = survivor[mask]
        removed = survivor[~mask]
        kvals = kept["net_return"].tolist()
        if not kvals:
            continue
        exp_k = _expectancy(kvals)
        pf_k = _profit_factor(kvals)
        net_k = float(np.sum(kvals))
        rows.append(
            {
                "candidate": name,
                "trades_kept": len(kept),
                "trade_reduction_ratio": 1.0 - len(kept) / n_surv,
                "survivor_expectancy_before": base_exp,
                "survivor_expectancy_after": exp_k,
                "expected_exp_delta": exp_k - base_exp,
                "expected_net_delta": net_k - base_net,
                "expected_pf_delta": pf_k - base_pf,
                "expected_ks_delta": int((kept["exit_reason"] == "kill_switch_close").sum()) - base_ks if "exit_reason" in kept.columns else 0,
                "expected_rfe_delta": int((kept["exit_reason"] == "risk_force_exit").sum()) - base_rfe if "exit_reason" in kept.columns else 0,
                "bad_removed": int((removed["net_return"] < 0).sum()),
                "good_removed": int((removed["net_return"] > 0).sum()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["survivor_expectancy_after", "expected_exp_delta", "trade_reduction_ratio"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _largest_leak(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "n/a"
    slices: List[Tuple[str, float, int]] = []
    for col, fn in [
        ("direction", lambda r: str(r["direction"])),
        ("entropy", _entropy_bucket),
        ("margin", _margin_bucket),
        ("trend_state", lambda r: str(r["trend_state"])),
        ("vol_state", lambda r: str(r["vol_state"])),
        ("ema_state", lambda r: str(r["ema_state"])),
        ("recent_return_24_bucket", lambda r: str(r["recent_return_24_bucket"])),
        ("exit_reason", lambda r: str(r.get("exit_reason", ""))),
    ]:
        tmp = trades.copy()
        if col in ("entropy", "margin"):
            tmp["_b"] = tmp[col].astype(float).apply(fn)
        else:
            tmp["_b"] = tmp[col].astype(str)
        for b, g in tmp.groupby("_b"):
            if len(g) < 3:
                continue
            exp = _expectancy(g["net_return"].tolist())
            loss_mass = float(g.loc[g["net_return"] < 0, "net_return"].sum())
            slices.append((f"{col}={b}", exp, len(g), loss_mass))
    if not slices:
        return "insufficient sample"
    # worst expectancy with meaningful count
    slices.sort(key=lambda x: (x[1], x[3]))
    worst = slices[0]
    return f"{worst[0]} (n={worst[2]}, exp={worst[1]:.6f})"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"survivor_expectancy_leak_{ts}.csv"
    md_path = OUT_DIR / f"survivor_expectancy_leak_{ts}.md"

    df = load_ohlcv()
    ticks, _ = simulate_signals_paper(df)
    ohlcv_feat = _ohlcv_features(df)

    _, base_trades, _, _ = _simulate(ticks, BASELINE_SPEC, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)
    _, surv_trades, blocked, _ = _simulate(ticks, SURVIVOR_SPEC, POSITION_SIZE, FEE_RATE, SLIPPAGE_RATE)

    base_enriched = _enrich_trades(base_trades, ticks, ohlcv_feat)
    surv_enriched = _enrich_trades(surv_trades, ticks, ohlcv_feat)

    tables: List[pd.DataFrame] = []

    # Phase 1–2: overview + LONG/SHORT
    overview = pd.DataFrame(
        [
            _segment_stats(base_enriched, "baseline_all"),
            _segment_stats(surv_enriched, "survivor_entropy_le_0.96"),
            _segment_stats(base_enriched[base_enriched["direction"] == "LONG"], "baseline_LONG"),
            _segment_stats(base_enriched[base_enriched["direction"] == "SHORT"], "baseline_SHORT"),
            _segment_stats(surv_enriched[surv_enriched["direction"] == "LONG"], "survivor_LONG"),
            _segment_stats(surv_enriched[surv_enriched["direction"] == "SHORT"], "survivor_SHORT"),
        ]
    )
    overview["section"] = "direction_split"
    tables.append(overview)

    # Phase 3: bucket analysis (survivor focus)
    bucket_parts = [
        _bucket_expectancy(surv_enriched, "entropy", lambda x: _entropy_bucket(float(x))),
        _bucket_expectancy(surv_enriched, "margin", lambda x: _margin_bucket(float(x))),
        _bucket_expectancy(surv_enriched, "trend_state", str),
        _bucket_expectancy(surv_enriched, "vol_state", str),
        _bucket_expectancy(surv_enriched, "ema_state", str),
        _bucket_expectancy(surv_enriched, "recent_return_24_bucket", str),
    ]
    buckets = pd.concat(bucket_parts, ignore_index=True)
    buckets["section"] = "feature_buckets_survivor"
    tables.append(buckets)

    # Phase 4–5: top loser/winner groups
    top_losers = _group_signatures(surv_enriched, 20, ascending=True)
    top_losers["section"] = "top_loser_groups"
    top_winners = _group_signatures(surv_enriched, 20, ascending=False)
    top_winners["section"] = "top_winner_groups"
    tables.extend([top_losers, top_winners])

    # Phase 6: residual trim
    residual = _residual_trim_analysis(surv_enriched)
    residual["section"] = "residual_loss_trim"
    tables.append(residual)

    # Phase 7–8: candidates
    candidates = _score_candidates(surv_enriched, base_enriched)
    candidates["section"] = "next_filter_candidates"
    tables.append(candidates)

    # Trades baseline took but survivor filter excluded (by entry_idx)
    if not base_enriched.empty and not surv_enriched.empty:
        surv_idx = set(surv_enriched["entry_idx"].astype(int).tolist())
        removed = base_enriched[~base_enriched["entry_idx"].isin(surv_idx)]
        removed_stats = pd.DataFrame([_segment_stats(removed, "removed_by_entropy_filter")])
        removed_stats["section"] = "removed_by_entropy"
        tables.append(removed_stats)

    combined = pd.concat(tables, ignore_index=True, sort=False)
    combined.to_csv(csv_path, index=False)

    largest_leak = _largest_leak(surv_enriched)
    top_loser_sig = top_losers.iloc[0]["signature"] if len(top_losers) else "n/a"
    top_winner_sig = top_winners.iloc[0]["signature"] if len(top_winners) else "n/a"

    trim20 = residual[residual["trim_pct_worst"] == 0.20].iloc[0] if len(residual) else None
    concentration = "dispersed"
    if trim20 is not None and float(trim20["expectancy_lift"]) > abs(float(trim20["baseline_expectancy"])) * 0.5:
        concentration = "concentrated (worst 20% trim lifts expectancy materially)"
    elif trim20 is not None and float(trim20["expectancy_lift"]) > 0:
        concentration = "partially_concentrated"

    cand_a = candidates.iloc[0]["candidate"] if len(candidates) > 0 else "n/a"
    cand_b = candidates.iloc[1]["candidate"] if len(candidates) > 1 else "n/a"
    cand_c = candidates.iloc[2]["candidate"] if len(candidates) > 2 else "n/a"
    cand_d = candidates.iloc[3]["candidate"] if len(candidates) > 3 else "n/a"
    recommended = cand_a

    # Final verdict
    surv_exp = _expectancy(surv_enriched["net_return"].tolist())
    if surv_exp >= 0:
        verdict = "survivor_edge_positive"
    elif len(candidates) and float(candidates.iloc[0]["survivor_expectancy_after"]) >= 0:
        verdict = "secondary_filter_may_flip_positive"
    elif concentration.startswith("concentrated"):
        verdict = "targeted_residual_filter_recommended"
    else:
        verdict = "lifecycle_or_signal_refinement_needed"

    md_lines = [
        "# Survivor Expectancy Leak Forensics",
        "",
        "## Summary",
        f"- baseline trades: {len(base_enriched)}, expectancy: {_expectancy(base_enriched['net_return'].tolist()):.6f}",
        f"- survivor (entropy<=0.96) trades: {len(surv_enriched)}, expectancy: {surv_exp:.6f}",
        f"- blocked entries: {len(blocked)}",
        "",
        "## 1. largest_expectancy_leak",
        f"- {largest_leak}",
        "",
        "## 2. top_loser_signature",
        f"- {top_loser_sig}",
        "",
        "## 3. top_winner_signature",
        f"- {top_winner_sig}",
        "",
        "## 4. survivor_loss_concentration",
        f"- {concentration}",
        "",
        "## 5–8. filter candidates",
        f"- candidate_A: {cand_a}",
        f"- candidate_B: {cand_b}",
        f"- candidate_C: {cand_c}",
        f"- candidate_D: {cand_d}",
        f"- recommended: {recommended}",
        "",
        "## FINAL VERDICT",
        f"- {verdict}",
        "",
        f"Generated: {ts}",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print("[SURVIVOR EXPECTANCY LEAK FORENSICS]")
    print("")
    print(f"1. largest_expectancy_leak:\n- {largest_leak}")
    print(f"\n2. top_loser_signature:\n- {top_loser_sig}")
    print(f"\n3. top_winner_signature:\n- {top_winner_sig}")
    print(f"\n4. survivor_loss_concentration:\n- {concentration}")
    print(f"\n5. candidate_A:\n- {cand_a}")
    print(f"\n6. candidate_B:\n- {cand_b}")
    print(f"\n7. candidate_C:\n- {cand_c}")
    print(f"\n8. candidate_D:\n- {cand_d}")
    print(f"\n9. recommended:\n- {recommended}")
    print(f"\n10. FINAL VERDICT:\n- {verdict}")
    print(f"\ncreated:\n- {csv_path}\n- {md_path}")


if __name__ == "__main__":
    main()
