"""
Can_bit Loss Cluster Forensics Report (diagnostics only).

Outputs:
  data/diagnostics/loss_forensics/loss_cluster_forensics_<timestamp>.csv
  data/diagnostics/loss_forensics/loss_cluster_forensics_<timestamp>.md
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "data" / "state" / "paper_trading_state.json"
MON_DIR = REPO_ROOT / "data" / "monitoring"
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "loss_forensics"
LIFECYCLE_MASTER = REPO_ROOT / "data" / "diagnostics" / "lifecycle" / "LIFECYCLE_MASTER_REPORT.md"

FEE_RATE = 0.0004
SLIPPAGE_RATE = 0.0002
ROUND_TRIP_COST = 2.0 * (FEE_RATE + SLIPPAGE_RATE)
KS_CONSEC_THRESHOLD = 5
POSITION_SIZE = 0.05


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _load_state() -> Dict[str, Any]:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def _iter_log_paths() -> List[Path]:
    return sorted(MON_DIR.glob("paper_trading_log_*.jsonl"))


def _load_events() -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    idx = 0
    for p in _iter_log_paths():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") != "tick":
                    continue
                price = _safe_float(ev.get("price"), default=np.nan)
                if np.isnan(price):
                    continue
                ev["_idx"] = idx
                ev["_price"] = price
                ev["_ts"] = str(ev.get("ts") or "")
                events.append(ev)
                idx += 1
    return events


def _direction_from_reason(reason: str) -> str:
    return "SHORT" if str(reason).startswith("SHORT") else "LONG"


def _strategy_from_reason(reason: str, fallback: str = "") -> str:
    if "_" in str(reason):
        return str(reason).rsplit("_", 1)[-1]
    return fallback or "UNKNOWN"


def _ret(direction: str, entry: float, px: float) -> float:
    if direction == "LONG":
        return (px - entry) / entry
    return (entry - px) / entry


def _reconstruct_trades(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    open_pos: Optional[Dict[str, Any]] = None

    for ev in events:
        decision = str(ev.get("decision") or "")
        reason = str(ev.get("reason") or "")
        price = float(ev["_price"])

        if open_pos is not None:
            open_pos["hold_bars"] += 1

        if decision == "enter" and open_pos is None:
            direction = _direction_from_reason(reason)
            p_long = _safe_float(ev.get("p_long"), np.nan)
            p_short = _safe_float(ev.get("p_short"), np.nan)
            p_flat = _safe_float(ev.get("p_flat"), np.nan)
            probs = [x for x in [p_long, p_short, p_flat] if not np.isnan(x)]
            max_proba = max(probs) if probs else np.nan
            margin = abs(p_long - p_short) if not (np.isnan(p_long) or np.isnan(p_short)) else np.nan
            open_pos = {
                "entry_idx": int(ev["_idx"]),
                "entry_time": ev["_ts"],
                "entry_price": price,
                "direction": direction,
                "strategy": _strategy_from_reason(reason, str(ev.get("strategy") or "")),
                "entropy": _safe_float(ev.get("entropy"), np.nan),
                "activation": bool(ev.get("activation")) if ev.get("activation") is not None else np.nan,
                "vol_bucket": str(ev.get("vol_bucket") or ""),
                "trend_state": str(ev.get("trend_state") or ""),
                "p_long": p_long,
                "p_short": p_short,
                "p_flat": p_flat,
                "margin": margin,
                "max_proba": max_proba,
                "hold_bars": 0,
            }
            continue

        if decision == "exit" and open_pos is not None:
            raw = _safe_float(ev.get("pnl"), np.nan)
            if np.isnan(raw):
                raw = _ret(open_pos["direction"], open_pos["entry_price"], price)
            net = raw - ROUND_TRIP_COST
            rows.append(
                {
                    **open_pos,
                    "exit_idx": int(ev["_idx"]),
                    "exit_time": ev["_ts"],
                    "exit_price": price,
                    "exit_reason": reason,
                    "raw_return": float(raw),
                    "net_return": float(net),
                    "is_loss": bool(net < 0),
                    "is_win": bool(net > 0),
                }
            )
            open_pos = None
    return pd.DataFrame(rows)


def _align_to_state_window(trades_df: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    st_trades = state.get("trades") or state.get("trade_history") or []
    if not isinstance(st_trades, list):
        return trades_df
    n = len(st_trades)
    if n <= 0 or len(trades_df) <= n:
        return trades_df.tail(max(n, 1)).reset_index(drop=True) if n > 0 else trades_df
    target = np.array([_safe_float(x.get("pnl"), 0.0) for x in st_trades], dtype=float)
    arr = trades_df["raw_return"].to_numpy(dtype=float)
    best_start = len(trades_df) - n
    best_err = float("inf")
    for i in range(0, len(trades_df) - n + 1):
        err = float(np.mean(np.abs(arr[i : i + n] - target)))
        if err < best_err:
            best_err = err
            best_start = i
    return trades_df.iloc[best_start : best_start + n].reset_index(drop=True)


def _add_context_features(trades_df: pd.DataFrame, events: List[Dict[str, Any]]) -> pd.DataFrame:
    px = pd.Series([float(e["_price"]) for e in events], dtype=float)
    tick_ret = px.pct_change()
    roll_vol = tick_ret.rolling(12, min_periods=5).std()
    out = trades_df.copy()
    out["entry_volatility"] = out["entry_idx"].apply(
        lambda i: float(roll_vol.iloc[int(i)]) if 0 <= int(i) < len(roll_vol) else np.nan
    )

    # future return if held 3 more bars after original exit
    idx_to_price = {int(e["_idx"]): float(e["_price"]) for e in events}
    fut_vals = []
    recov_vals = []
    for _, r in out.iterrows():
        fut_idx = int(r["exit_idx"]) + 3
        if fut_idx in idx_to_price:
            fut_raw = _ret(str(r["direction"]), float(r["entry_price"]), float(idx_to_price[fut_idx]))
            fut_net = fut_raw - ROUND_TRIP_COST
            fut_vals.append(float(fut_net))
            recov_vals.append(bool(fut_net > 0))
        else:
            fut_vals.append(np.nan)
            recov_vals.append(np.nan)
    out["future_return_if_held_3"] = fut_vals
    out["recovery_if_held_3"] = recov_vals
    return out


def _equity_and_dd(trades_df: pd.DataFrame) -> pd.DataFrame:
    out = trades_df.copy()
    eq = 1.0
    peak = 1.0
    eq_curve = []
    dd_curve = []
    for r in out["net_return"].tolist():
        eq *= 1.0 + float(r) * POSITION_SIZE
        peak = max(peak, eq)
        dd = (eq - peak) / peak if peak > 0 else 0.0
        eq_curve.append(eq)
        dd_curve.append(dd)
    out["equity"] = eq_curve
    out["drawdown"] = dd_curve
    out["drawdown_before"] = pd.Series(dd_curve).shift(1).fillna(0.0).tolist()
    return out


def _detect_loss_clusters(trades_df: pd.DataFrame) -> pd.DataFrame:
    clusters: List[Dict[str, Any]] = []
    i = 0
    cid = 1
    while i < len(trades_df):
        if not bool(trades_df.loc[i, "is_loss"]):
            i += 1
            continue
        j = i
        while j < len(trades_df) and bool(trades_df.loc[j, "is_loss"]):
            j += 1
        seg = trades_df.iloc[i:j]
        dirs = seg["direction"].value_counts().to_dict()
        sratio = seg["strategy"].value_counts(normalize=True).to_dict()
        clusters.append(
            {
                "cluster_id": cid,
                "cluster_start_trade_idx": int(i),
                "cluster_end_trade_idx": int(j - 1),
                "cluster_length": int(len(seg)),
                "cumulative_loss": float(seg["net_return"].sum()),
                "avg_loss": float(seg["net_return"].mean()),
                "directions": str(dirs),
                "avg_entropy": float(seg["entropy"].astype(float).mean()),
                "avg_margin": float(seg["margin"].astype(float).mean()),
                "avg_max_proba": float(seg["max_proba"].astype(float).mean()),
                "avg_hold_bars": float(seg["hold_bars"].mean()),
                "avg_volatility": float(seg["entry_volatility"].mean()),
                "avg_recent_return": float(trades_df["net_return"].iloc[max(0, i - 5) : i].mean()) if i > 0 else 0.0,
                "avg_drawdown_before_cluster": float(seg["drawdown_before"].mean()),
                "activation_ratio": float(pd.to_numeric(seg["activation"], errors="coerce").mean()),
                "s1_ratio": float(sratio.get("S1", 0.0)),
                "s2_ratio": float(sratio.get("S2", 0.0)),
            }
        )
        cid += 1
        i = j
    cdf = pd.DataFrame(clusters)
    if cdf.empty:
        return cdf
    return cdf.sort_values(["cumulative_loss", "cluster_length"]).head(20).reset_index(drop=True)


def _winner_loser_regime(trades_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "entropy",
        "margin",
        "p_long",
        "p_short",
        "p_flat",
        "hold_bars",
        "entry_volatility",
        "max_proba",
        "raw_return",
        "net_return",
    ]
    rows = []
    wins = trades_df[trades_df["is_win"]]
    losses = trades_df[trades_df["is_loss"]]
    for c in cols:
        w = float(pd.to_numeric(wins[c], errors="coerce").mean()) if len(wins) else np.nan
        l = float(pd.to_numeric(losses[c], errors="coerce").mean()) if len(losses) else np.nan
        rows.append({"metric": c, "winners_mean": w, "losers_mean": l, "delta_w_minus_l": w - l})

    # categorical comparison
    for c in ["vol_bucket", "trend_state", "strategy", "direction", "activation", "exit_reason"]:
        w_mode = wins[c].mode().iloc[0] if len(wins) and not wins[c].mode().empty else ""
        l_mode = losses[c].mode().iloc[0] if len(losses) and not losses[c].mode().empty else ""
        rows.append({"metric": f"{c}_mode", "winners_mean": w_mode, "losers_mean": l_mode, "delta_w_minus_l": ""})
    return pd.DataFrame(rows)


def _entropy_zone(trades_df: pd.DataFrame, ks_seq_idx: List[int]) -> pd.DataFrame:
    zones = [(0.80, 0.90), (0.90, 0.95), (0.95, 1.00), (1.00, 1.05), (1.05, 1.10)]
    rows = []
    for a, b in zones:
        seg = trades_df[(trades_df["entropy"] >= a) & (trades_df["entropy"] < b)]
        if seg.empty:
            rows.append(
                {
                    "entropy_zone": f"{a:.2f}~{b:.2f}",
                    "trades": 0,
                    "wr": 0.0,
                    "expectancy": 0.0,
                    "pf": 0.0,
                    "avg_hold": 0.0,
                    "ks_involvement": 0,
                    "force_exit_ratio": 0.0,
                }
            )
            continue
        rets = seg["net_return"].tolist()
        wins = [x for x in rets if x > 0]
        losses = [x for x in rets if x < 0]
        wr = len(wins) / len(rets)
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        exp = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss))
        pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
        ks_inv = int(sum(1 for x in seg.index.tolist() if int(x) in ks_seq_idx))
        rows.append(
            {
                "entropy_zone": f"{a:.2f}~{b:.2f}",
                "trades": int(len(seg)),
                "wr": float(wr),
                "expectancy": float(exp),
                "pf": float(pf),
                "avg_hold": float(seg["hold_bars"].mean()),
                "ks_involvement": ks_inv,
                "force_exit_ratio": float((seg["exit_reason"] == "risk_force_exit").mean()),
            }
        )
    return pd.DataFrame(rows)


def _hold_forensics(trades_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in sorted(trades_df["hold_bars"].astype(int).unique().tolist()):
        key = "12+" if h >= 12 else str(h)
        seg = trades_df[trades_df["hold_bars"].astype(int) == h] if h < 12 else trades_df[trades_df["hold_bars"].astype(int) >= 12]
        if seg.empty:
            continue
        rets = seg["net_return"].tolist()
        wins = [x for x in rets if x > 0]
        losses = [x for x in rets if x < 0]
        wr = len(wins) / len(rets)
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        exp = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss))
        rows.append(
            {
                "hold_bucket": key,
                "trades": int(len(seg)),
                "wr": float(wr),
                "expectancy": float(exp),
                "force_exit_ratio": float((seg["exit_reason"] == "risk_force_exit").mean()),
                "exit_signal_ratio": float((seg["exit_reason"] == "exit_signal").mean()),
                "avg_future_return_if_held": float(pd.to_numeric(seg["future_return_if_held_3"], errors="coerce").mean()),
                "recovery_probability": float(pd.to_numeric(seg["recovery_if_held_3"], errors="coerce").mean()),
            }
        )
        if h >= 12:
            break
    return pd.DataFrame(rows)


def _directional(trades_df: pd.DataFrame, ks_seq_idx: List[int], cluster_members: List[int]) -> pd.DataFrame:
    rows = []
    for d in ["LONG", "SHORT"]:
        seg = trades_df[trades_df["direction"] == d]
        if seg.empty:
            continue
        rets = seg["net_return"].tolist()
        wins = [x for x in rets if x > 0]
        losses = [x for x in rets if x < 0]
        wr = len(wins) / len(rets)
        avg_win = mean(wins) if wins else 0.0
        avg_loss = mean(losses) if losses else 0.0
        exp = (wr * avg_win) - ((1.0 - wr) * abs(avg_loss))
        pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
        rows.append(
            {
                "direction": d,
                "trades": int(len(seg)),
                "expectancy": float(exp),
                "pf": float(pf),
                "ks_involvement": int(sum(1 for x in seg.index.tolist() if int(x) in ks_seq_idx)),
                "loss_cluster_involvement": int(sum(1 for x in seg.index.tolist() if int(x) in cluster_members)),
                "avg_entropy": float(seg["entropy"].mean()),
                "avg_margin": float(seg["margin"].mean()),
                "avg_hold": float(seg["hold_bars"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _ks_root_cause(trades_df: pd.DataFrame) -> pd.DataFrame:
    consec = 0
    trigger_idx = -1
    for i, r in enumerate(trades_df["net_return"].tolist()):
        if r < 0:
            consec += 1
            if consec >= KS_CONSEC_THRESHOLD:
                trigger_idx = i
                break
        else:
            consec = 0
    if trigger_idx < 0:
        return pd.DataFrame()
    seq = trades_df.iloc[max(0, trigger_idx - 19) : trigger_idx + 1].copy()
    seq["wl"] = seq["is_win"].apply(lambda x: "W" if x else "L")
    seq["cum_drawdown_progression"] = seq["drawdown"]
    return seq[
        [
            "entry_time",
            "exit_time",
            "wl",
            "net_return",
            "entropy",
            "direction",
            "hold_bars",
            "exit_reason",
            "trend_state",
            "vol_bucket",
            "cum_drawdown_progression",
        ]
    ].reset_index(drop=True)


def _feature_corr(trades_df: pd.DataFrame) -> pd.DataFrame:
    target = trades_df["is_loss"].astype(int)
    rows = []
    for c in ["entropy", "margin", "hold_bars", "entry_volatility", "p_long", "p_short", "p_flat", "max_proba"]:
        x = pd.to_numeric(trades_df[c], errors="coerce")
        corr = x.corr(target) if x.notna().sum() > 3 else np.nan
        rows.append({"feature": c, "corr_with_loss": float(corr) if pd.notna(corr) else np.nan})
    act = pd.to_numeric(trades_df["activation"], errors="coerce")
    rows.append({"feature": "activation", "corr_with_loss": float(act.corr(target)) if act.notna().sum() > 3 else np.nan})
    s2 = (trades_df["strategy"] == "S2").astype(int)
    rows.append({"feature": "is_S2", "corr_with_loss": float(s2.corr(target)) if len(s2) > 3 else np.nan})
    return pd.DataFrame(rows).sort_values("corr_with_loss", ascending=False)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"loss_cluster_forensics_{ts}.csv"
    md_path = OUT_DIR / f"loss_cluster_forensics_{ts}.md"

    state = _load_state()
    events = _load_events()
    trades = _reconstruct_trades(events)
    trades = _align_to_state_window(trades, state)
    trades = _add_context_features(trades, events)
    trades = _equity_and_dd(trades)

    clusters = _detect_loss_clusters(trades)
    wl_cmp = _winner_loser_regime(trades)
    ks_seq = _ks_root_cause(trades)
    ks_seq_idx = []
    if not ks_seq.empty:
        # map ks seq to trade indices in aligned dataframe tail
        ks_seq_idx = list(range(len(trades) - len(ks_seq), len(trades)))
    ent_zone = _entropy_zone(trades, ks_seq_idx)
    hold_df = _hold_forensics(trades)
    cluster_members = []
    for _, r in clusters.iterrows():
        cluster_members.extend(list(range(int(r["cluster_start_trade_idx"]), int(r["cluster_end_trade_idx"]) + 1)))
    directional = _directional(trades, ks_seq_idx, cluster_members)
    corr_df = _feature_corr(trades)

    # single csv with section tag
    tables = []
    for name, df in [
        ("loss_clusters_top20", clusters),
        ("winner_loser_regime", wl_cmp),
        ("entropy_zone", ent_zone),
        ("hold_duration", hold_df),
        ("directional", directional),
        ("ks_root_cause_last20", ks_seq),
        ("feature_corr", corr_df),
    ]:
        if df is None or df.empty:
            continue
        t = df.copy()
        t.insert(0, "section", name)
        tables.append(t)
    if tables:
        pd.concat(tables, ignore_index=True, sort=False).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([{"section": "empty", "note": "no data"}]).to_csv(csv_path, index=False)

    # conclusions
    worst_cluster = clusters.iloc[0] if not clusters.empty else None
    riskiest_entropy = ent_zone.sort_values("expectancy").iloc[0] if not ent_zone.empty else None
    safest_entropy = ent_zone.sort_values("expectancy", ascending=False).iloc[0] if not ent_zone.empty else None
    riskiest_hold = hold_df.sort_values("expectancy").iloc[0] if not hold_df.empty else None
    safest_hold = hold_df.sort_values("expectancy", ascending=False).iloc[0] if not hold_df.empty else None
    worst_dir = directional.sort_values("expectancy").iloc[0] if not directional.empty else None
    best_dir = directional.sort_values("expectancy", ascending=False).iloc[0] if not directional.empty else None

    md: List[str] = []
    md.append("# Can_bit Loss Cluster Forensics Report")
    md.append("")
    md.append("## 1. Executive Summary")
    md.append(f"- trades_analyzed: {len(trades)}")
    md.append(f"- total_loss_clusters: {len(clusters)}")
    if worst_cluster is not None:
        md.append(
            f"- worst_cluster: length={int(worst_cluster['cluster_length'])}, cumulative_loss={worst_cluster['cumulative_loss']:.6f}, "
            f"avg_hold={worst_cluster['avg_hold_bars']:.2f}, avg_entropy={worst_cluster['avg_entropy']:.4f}"
        )
    md.append("")
    md.append("## 2. Worst Loss Clusters")
    if clusters.empty:
        md.append("- no loss clusters detected")
    else:
        for _, r in clusters.head(10).iterrows():
            md.append(
                f"- cluster#{int(r['cluster_id'])}: len={int(r['cluster_length'])}, cum_loss={r['cumulative_loss']:.6f}, "
                f"dir={r['directions']}, entropy={r['avg_entropy']:.4f}, hold={r['avg_hold_bars']:.2f}, vol={r['avg_volatility']:.6f}"
            )
    md.append("")
    md.append("## 3. KS Root Cause")
    if ks_seq.empty:
        md.append("- KS trigger sequence not found")
    else:
        md.append(f"- KS recent sequence length={len(ks_seq)} (latest 20 trades window)")
        md.append(
            f"- KS sequence avg entropy={pd.to_numeric(ks_seq['entropy'], errors='coerce').mean():.4f}, "
            f"risk_force_exit_ratio={(ks_seq['exit_reason'] == 'risk_force_exit').mean():.2%}"
        )
    md.append("")
    md.append("## 4. Entropy Zone Analysis")
    if riskiest_entropy is not None and safest_entropy is not None:
        md.append(
            f"- worst_zone: {riskiest_entropy['entropy_zone']} (expectancy={riskiest_entropy['expectancy']:.6f}, "
            f"force_exit_ratio={riskiest_entropy['force_exit_ratio']:.2%})"
        )
        md.append(
            f"- best_zone: {safest_entropy['entropy_zone']} (expectancy={safest_entropy['expectancy']:.6f}, "
            f"force_exit_ratio={safest_entropy['force_exit_ratio']:.2%})"
        )
    md.append("")
    md.append("## 5. Hold Duration Analysis")
    if riskiest_hold is not None and safest_hold is not None:
        md.append(
            f"- worst_hold: {riskiest_hold['hold_bucket']} (expectancy={riskiest_hold['expectancy']:.6f}, "
            f"force_exit_ratio={riskiest_hold['force_exit_ratio']:.2%}, recovery_prob={riskiest_hold['recovery_probability']:.2%})"
        )
        md.append(
            f"- best_hold: {safest_hold['hold_bucket']} (expectancy={safest_hold['expectancy']:.6f}, "
            f"force_exit_ratio={safest_hold['force_exit_ratio']:.2%}, recovery_prob={safest_hold['recovery_probability']:.2%})"
        )
    md.append("")
    md.append("## 6. LONG vs SHORT")
    if worst_dir is not None and best_dir is not None:
        md.append(
            f"- worst_direction: {worst_dir['direction']} (expectancy={worst_dir['expectancy']:.6f}, pf={worst_dir['pf']:.4f})"
        )
        md.append(
            f"- best_direction: {best_dir['direction']} (expectancy={best_dir['expectancy']:.6f}, pf={best_dir['pf']:.4f})"
        )
    md.append("")
    md.append("## 7. Regime Analysis")
    mode_row = wl_cmp[wl_cmp["metric"] == "trend_state_mode"]
    vol_row = wl_cmp[wl_cmp["metric"] == "vol_bucket_mode"]
    if not mode_row.empty:
        md.append(
            f"- trend_mode winners={mode_row.iloc[0]['winners_mean']} vs losers={mode_row.iloc[0]['losers_mean']}"
        )
    if not vol_row.empty:
        md.append(
            f"- vol_mode winners={vol_row.iloc[0]['winners_mean']} vs losers={vol_row.iloc[0]['losers_mean']}"
        )
    md.append("")
    md.append("## 8. Feature Correlations")
    if not corr_df.empty:
        for _, r in corr_df.head(5).iterrows():
            md.append(f"- {r['feature']}: corr_with_loss={r['corr_with_loss']:.4f}")
    md.append("")
    md.append("## 9. Strongest Failure Conditions")
    if worst_cluster is not None:
        md.append(
            f"- long loss streak (len={int(worst_cluster['cluster_length'])}) + "
            f"high force-exit context + entropy around {worst_cluster['avg_entropy']:.4f} + hold {worst_cluster['avg_hold_bars']:.2f}"
        )
    if worst_dir is not None:
        md.append(f"- directional weakness: {worst_dir['direction']} side")
    if riskiest_entropy is not None:
        md.append(f"- entropy risk zone: {riskiest_entropy['entropy_zone']}")
    if riskiest_hold is not None:
        md.append(f"- hold risk bucket: {riskiest_hold['hold_bucket']}")
    md.append("")
    md.append("## 10. Strongest Survival Conditions")
    if safest_entropy is not None:
        md.append(f"- entropy survival zone: {safest_entropy['entropy_zone']}")
    if safest_hold is not None:
        md.append(f"- hold survival bucket: {safest_hold['hold_bucket']}")
    if best_dir is not None:
        md.append(f"- stronger direction: {best_dir['direction']}")
    md.append("")
    md.append("## 11. Recommended Next Step")
    md.append("- 우선순위: force_exit tuning -> directional filter -> hold extension -> entropy zone refinement -> volatility filter")
    md.append("- 운영 로직 변경 전, 동일 조건 rolling forward replay로 실패 패턴 재현성 재확인")
    if LIFECYCLE_MASTER.exists():
        md.append(f"- 참고: `{LIFECYCLE_MASTER}`")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[LOSS CLUSTER FORENSICS REPORT]")
    print(f"csv: {csv_path}")
    print(f"md: {md_path}")


if __name__ == "__main__":
    main()
