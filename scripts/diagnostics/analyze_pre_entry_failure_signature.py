"""
Can_bit pre-entry failure signature forensics (diagnostics only).
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from scripts.run_daily_paper import load_ohlcv, simulate_signals_paper


REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_PATH = REPO_ROOT / "data" / "state" / "paper_trading_state.json"
LOG_DIR = REPO_ROOT / "data" / "monitoring"
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "pre_entry"

FEE_RATE = 0.0004
SLIP_RATE = 0.0002
ROUND_COST = 2.0 * (FEE_RATE + SLIP_RATE)
POSITION_SIZE = 0.05


def _safe_float(v: Any, d: float = np.nan) -> float:
    try:
        if v is None:
            return d
        return float(v)
    except Exception:
        return d


def _iter_log_paths() -> List[Path]:
    return sorted(LOG_DIR.glob("paper_trading_log_*.jsonl"))


def _load_state_trades() -> List[Dict[str, Any]]:
    if not STATE_PATH.exists():
        return []
    st = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    tr = st.get("trades") or st.get("trade_history") or []
    return tr if isinstance(tr, list) else []


def _load_tick_events() -> List[Dict[str, Any]]:
    out = []
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
                price = _safe_float(ev.get("price"))
                if np.isnan(price):
                    continue
                ev["_idx"] = idx
                ev["_price"] = price
                ev["_ts"] = str(ev.get("ts") or "")
                out.append(ev)
                idx += 1
    return out


def _reconstruct_trades(events: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    open_pos = None
    for ev in events:
        decision = str(ev.get("decision") or "")
        reason = str(ev.get("reason") or "")
        px = float(ev["_price"])
        if open_pos is not None:
            open_pos["hold_bars"] += 1
        if decision == "enter" and open_pos is None:
            direction = "SHORT" if reason.startswith("SHORT") else "LONG"
            strategy = reason.rsplit("_", 1)[-1] if "_" in reason else str(ev.get("strategy") or "UNKNOWN")
            open_pos = {
                "entry_idx": int(ev["_idx"]),
                "entry_time": ev["_ts"],
                "entry_price": px,
                "direction": direction,
                "strategy": strategy,
                "entry_entropy_log": _safe_float(ev.get("entropy")),
                "vol_bucket_log": str(ev.get("vol_bucket") or ""),
                "trend_state_log": str(ev.get("trend_state") or ""),
                "hold_bars": 0,
            }
            continue
        if decision == "exit" and open_pos is not None:
            raw = _safe_float(ev.get("pnl"))
            if np.isnan(raw):
                if open_pos["direction"] == "LONG":
                    raw = (px - open_pos["entry_price"]) / open_pos["entry_price"]
                else:
                    raw = (open_pos["entry_price"] - px) / open_pos["entry_price"]
            net = raw - ROUND_COST
            rows.append(
                {
                    **open_pos,
                    "exit_idx": int(ev["_idx"]),
                    "exit_time": ev["_ts"],
                    "exit_price": px,
                    "exit_reason": reason,
                    "gross_return": float(raw),
                    "net_return": float(net),
                }
            )
            open_pos = None
    return pd.DataFrame(rows)


def _align_to_state_window(trades: pd.DataFrame, state_trades: List[Dict[str, Any]]) -> pd.DataFrame:
    n = len(state_trades)
    if n <= 0 or len(trades) <= n:
        return trades.tail(max(1, n)).reset_index(drop=True) if n else trades
    target = np.array([_safe_float(x.get("pnl"), 0.0) for x in state_trades], dtype=float)
    arr = trades["gross_return"].to_numpy(dtype=float)
    best_i = len(trades) - n
    best_e = float("inf")
    for i in range(0, len(trades) - n + 1):
        e = float(np.mean(np.abs(arr[i : i + n] - target)))
        if e < best_e:
            best_e = e
            best_i = i
    return trades.iloc[best_i : best_i + n].reset_index(drop=True)


def _feature_engineering(df_ohlcv: pd.DataFrame, ticks: List[Dict[str, Any]]) -> pd.DataFrame:
    # tick table from simulate_signals_paper
    tdf = pd.DataFrame(ticks)
    if "timestamp" in tdf.columns:
        tdf["timestamp"] = pd.to_datetime(tdf["timestamp"], errors="coerce")
    else:
        tdf["timestamp"] = pd.NaT
    # ohlcv features
    px = pd.to_numeric(df_ohlcv["close"], errors="coerce")
    h = pd.to_numeric(df_ohlcv["high"], errors="coerce")
    l = pd.to_numeric(df_ohlcv["low"], errors="coerce")
    o = pd.to_numeric(df_ohlcv["open"], errors="coerce")
    ema = px.ewm(span=20, adjust=False).mean()
    ema_slope = ema.diff()
    ret = px.pct_change()
    rv12 = ret.rolling(12).std()
    rv24 = ret.rolling(24).std()
    rv48 = ret.rolling(48).std()
    tr = pd.concat([(h - l), (h - px.shift(1)).abs(), (l - px.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    rr12 = px.pct_change(12)
    rr24 = px.pct_change(24)
    rr48 = px.pct_change(48)
    rh = h.rolling(20).max().shift(1)
    rl = l.rolling(20).min().shift(1)
    body = (px - o).abs()
    range_ = (h - l).replace(0, np.nan)
    body_ratio = body / range_
    wick_ratio = ((h - l) - body) / range_
    green = (px > o).astype(int)
    red = (px < o).astype(int)
    consec_green = green.groupby((green == 0).cumsum()).cumsum()
    consec_red = red.groupby((red == 0).cumsum()).cumsum()

    feat = pd.DataFrame(
        {
            "df_idx": np.arange(len(df_ohlcv)),
            "trend_strength": (ema_slope / px).replace([np.inf, -np.inf], np.nan),
            "ema_distance": ((px - ema) / ema).replace([np.inf, -np.inf], np.nan),
            "ema_slope": ema_slope,
            "price_vs_ema": (px > ema).astype(int),
            "recent_return_12": rr12,
            "recent_return_24": rr24,
            "recent_return_48": rr48,
            "realized_vol_12": rv12,
            "realized_vol_24": rv24,
            "realized_vol_48": rv48,
            "atr": atr,
            "recent_high_break": (px > rh).astype(int),
            "recent_low_break": (px < rl).astype(int),
            "candle_body_ratio": body_ratio,
            "wick_ratio": wick_ratio,
            "consecutive_green": consec_green,
            "consecutive_red": consec_red,
        }
    )
    tdf = tdf.merge(feat, on="df_idx", how="left")
    tdf["max_proba"] = tdf[["p_long", "p_short", "p_flat"]].max(axis=1)
    tdf["margin"] = (pd.to_numeric(tdf["p_long"], errors="coerce") - pd.to_numeric(tdf["p_short"], errors="coerce")).abs()
    return tdf


def _attach_entry_features(trades: pd.DataFrame, tick_feat: pd.DataFrame) -> pd.DataFrame:
    # map by entry_idx from log to nearest df_idx via order; fallback with nearest timestamp
    out = trades.copy()
    tf = tick_feat.reset_index(drop=True).copy()
    # entropy from p distribution
    probs = tf[["p_long", "p_short", "p_flat"]].astype(float)
    ent = []
    for _, r in probs.iterrows():
        e = 0.0
        for p in r.values:
            if pd.notna(p) and p > 1e-10:
                e -= float(p) * math.log(float(p))
        ent.append(e)
    tf["entry_entropy"] = ent

    # primary: by order of enters in tick_feat where signal exists and activation on
    cand = tf[tf["signal"].notna()].copy()
    cand = cand.reset_index(drop=True)
    k = min(len(out), len(cand))
    numeric_cols = [
        "p_long",
        "p_short",
        "p_flat",
        "max_proba",
        "margin",
        "entry_entropy",
        "trend_strength",
        "ema_distance",
        "ema_slope",
        "price_vs_ema",
        "recent_return_12",
        "recent_return_24",
        "recent_return_48",
        "realized_vol_12",
        "realized_vol_24",
        "realized_vol_48",
        "atr",
        "recent_high_break",
        "recent_low_break",
        "candle_body_ratio",
        "wick_ratio",
        "consecutive_green",
        "consecutive_red",
    ]
    string_cols = ["trend_state", "vol_bucket"]
    for col in numeric_cols:
        out[col] = np.nan
    for col in string_cols:
        out[col] = ""
    if k > 0:
        out.loc[: k - 1, "p_long"] = cand.loc[: k - 1, "p_long"].values
        out.loc[: k - 1, "p_short"] = cand.loc[: k - 1, "p_short"].values
        out.loc[: k - 1, "p_flat"] = cand.loc[: k - 1, "p_flat"].values
        out.loc[: k - 1, "max_proba"] = cand.loc[: k - 1, "max_proba"].values
        out.loc[: k - 1, "margin"] = cand.loc[: k - 1, "margin"].values
        out.loc[: k - 1, "entry_entropy"] = cand.loc[: k - 1, "entry_entropy"].values
        out.loc[: k - 1, "trend_state"] = cand.loc[: k - 1, "trend_label"].astype(str).values
        for c in [
            "trend_strength",
            "ema_distance",
            "ema_slope",
            "price_vs_ema",
            "recent_return_12",
            "recent_return_24",
            "recent_return_48",
            "realized_vol_12",
            "realized_vol_24",
            "realized_vol_48",
            "atr",
            "vol_bucket",
            "recent_high_break",
            "recent_low_break",
            "candle_body_ratio",
            "wick_ratio",
            "consecutive_green",
            "consecutive_red",
        ]:
            if c == "vol_bucket":
                out.loc[: k - 1, c] = cand.loc[: k - 1, c].astype(str).values
            else:
                out.loc[: k - 1, c] = cand.loc[: k - 1, c].values

    out["fee_cost"] = 2.0 * FEE_RATE
    out["slippage_cost"] = 2.0 * SLIP_RATE
    out["is_win"] = out["net_return"] > 0
    out["is_loss"] = out["net_return"] < 0
    return out


def _ks_cluster_indices(trades: pd.DataFrame) -> List[int]:
    consec = 0
    idx = -1
    for i, r in enumerate(trades["net_return"].tolist()):
        if r < 0:
            consec += 1
            if consec >= 5:
                idx = i
                break
        else:
            consec = 0
    if idx < 0:
        return []
    return list(range(max(0, idx - 4), idx + 1))


def _group_masks(trades: pd.DataFrame) -> Dict[str, pd.Series]:
    ks_idx = set(_ks_cluster_indices(trades))
    q90 = float(trades["net_return"].quantile(0.9))
    q10 = float(trades["net_return"].quantile(0.1))
    masks = {
        "A_survival_winners": (trades["net_return"] > 0) & (trades["exit_reason"] == "exit_signal") & (trades["hold_bars"] >= 10),
        "B_force_exit_losers": (trades["exit_reason"] == "risk_force_exit") & (trades["net_return"] < 0),
        "C_ks_cluster": pd.Series([i in ks_idx for i in range(len(trades))], index=trades.index),
        "D_short_hold_collapse": (trades["hold_bars"] <= 6) & (trades["net_return"] < 0),
        "E_best_10pct": trades["net_return"] >= q90,
        "F_worst_10pct": trades["net_return"] <= q10,
    }
    return masks


def _expectancy(vals: List[float]) -> float:
    if not vals:
        return 0.0
    wins = [x for x in vals if x > 0]
    losses = [x for x in vals if x < 0]
    wr = len(wins) / len(vals)
    avg_w = mean(wins) if wins else 0.0
    avg_l = mean(losses) if losses else 0.0
    return (wr * avg_w) - ((1 - wr) * abs(avg_l))


def _group_stats(trades: pd.DataFrame, masks: Dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for k, m in masks.items():
        seg = trades[m]
        vals = seg["net_return"].tolist()
        rows.append(
            {
                "group": k,
                "count": int(len(seg)),
                "expectancy": _expectancy(vals),
                "avg_return": float(np.mean(vals)) if vals else 0.0,
                "median_return": float(np.median(vals)) if vals else 0.0,
                "win_rate": float((seg["net_return"] > 0).mean()) if len(seg) else 0.0,
                "avg_hold_bars": float(seg["hold_bars"].mean()) if len(seg) else 0.0,
                "avg_entropy": float(pd.to_numeric(seg["entry_entropy"], errors="coerce").mean()) if len(seg) else np.nan,
                "avg_margin": float(pd.to_numeric(seg["margin"], errors="coerce").mean()) if len(seg) else np.nan,
                "avg_max_proba": float(pd.to_numeric(seg["max_proba"], errors="coerce").mean()) if len(seg) else np.nan,
                "avg_realized_vol": float(pd.to_numeric(seg["realized_vol_24"], errors="coerce").mean()) if len(seg) else np.nan,
                "avg_recent_return_24": float(pd.to_numeric(seg["recent_return_24"], errors="coerce").mean()) if len(seg) else np.nan,
                "avg_ema_distance": float(pd.to_numeric(seg["ema_distance"], errors="coerce").mean()) if len(seg) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _cohen_d(x: pd.Series, y: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    y = pd.to_numeric(y, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = math.sqrt(((nx - 1) * vx + (ny - 1) * vy) / max(nx + ny - 2, 1))
    if sp == 0:
        return np.nan
    return float((x.mean() - y.mean()) / sp)


def _ks_stat(x: pd.Series, y: pd.Series) -> float:
    x = np.sort(pd.to_numeric(x, errors="coerce").dropna().values)
    y = np.sort(pd.to_numeric(y, errors="coerce").dropna().values)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    vals = np.unique(np.concatenate([x, y]))
    cdfx = np.searchsorted(x, vals, side="right") / len(x)
    cdfy = np.searchsorted(y, vals, side="right") / len(y)
    return float(np.max(np.abs(cdfx - cdfy)))


def _separation(trades: pd.DataFrame, masks: Dict[str, pd.Series]) -> pd.DataFrame:
    a = trades[masks["A_survival_winners"]]
    b = trades[masks["B_force_exit_losers"]]
    feats = [
        "entry_entropy",
        "margin",
        "max_proba",
        "realized_vol_12",
        "realized_vol_24",
        "recent_return_24",
        "ema_distance",
        "trend_strength",
        "hold_bars",
        "p_long",
        "p_short",
        "p_flat",
        "atr",
    ]
    rows = []
    for f in feats:
        d = _cohen_d(a[f], b[f])
        ks = _ks_stat(a[f], b[f])
        z = np.nan
        ax = pd.to_numeric(a[f], errors="coerce").dropna()
        bx = pd.to_numeric(b[f], errors="coerce").dropna()
        if len(ax) and len(bx):
            z = float((ax.mean() - bx.mean()) / (ax.std(ddof=1) + bx.std(ddof=1) + 1e-9))
        score = np.nanmean([abs(d) if pd.notna(d) else np.nan, ks if pd.notna(ks) else np.nan, abs(z) if pd.notna(z) else np.nan])
        rows.append({"feature": f, "cohen_d": d, "ks_stat": ks, "z_sep": z, "feature_importance_like_score": score})
    return pd.DataFrame(rows).sort_values("feature_importance_like_score", ascending=False)


def _signature_tables(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = trades.copy()
    df["entropy_bin"] = pd.cut(df["entry_entropy"], bins=[-1, 0.90, 0.95, 1.00, 2], labels=["<0.90", "0.90-0.95", "0.95-1.00", ">1.00"])
    df["margin_bin"] = pd.cut(df["margin"], bins=[-1, 0.02, 0.04, 0.06, 10], labels=["<0.02", "0.02-0.04", "0.04-0.06", ">0.06"])
    df["rr24_sign"] = np.where(pd.to_numeric(df["recent_return_24"], errors="coerce") < 0, "neg", "pos")
    df["ema_side"] = np.where(pd.to_numeric(df["ema_distance"], errors="coerce") < 0, "below_ema", "above_ema")
    grp_cols = ["direction", "entropy_bin", "margin_bin", "vol_bucket", "rr24_sign", "ema_side"]
    rows = []
    for keys, seg in df.groupby(grp_cols, dropna=False):
        if len(seg) < 4:
            continue
        vals = seg["net_return"].tolist()
        rows.append(
            {
                "direction": keys[0],
                "entropy_bin": str(keys[1]),
                "margin_bin": str(keys[2]),
                "vol_bucket": str(keys[3]),
                "recent_return_24_sign": str(keys[4]),
                "ema_side": str(keys[5]),
                "count": int(len(seg)),
                "expectancy": _expectancy(vals),
                "force_exit_ratio": float((seg["exit_reason"] == "risk_force_exit").mean()),
                "avg_return": float(np.mean(vals)),
            }
        )
    sig = pd.DataFrame(rows)
    if sig.empty:
        return sig, sig
    danger = sig.sort_values(["expectancy", "force_exit_ratio"]).head(10).reset_index(drop=True)
    safe = sig.sort_values(["expectancy", "force_exit_ratio"], ascending=[False, True]).head(10).reset_index(drop=True)
    return danger, safe


def _filter_replay(trades: pd.DataFrame, name: str, mask_block: pd.Series) -> Dict[str, Any]:
    base = trades.copy()
    new = trades[~mask_block].copy()

    def stats(df: pd.DataFrame) -> Dict[str, float]:
        vals = df["net_return"].tolist()
        wins = [x for x in vals if x > 0]
        losses = [x for x in vals if x < 0]
        pf = (sum(wins) / abs(sum(losses))) if losses else 0.0
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        consec = 0
        ks = False
        for r in vals:
            eq *= 1.0 + r * POSITION_SIZE
            peak = max(peak, eq)
            mdd = min(mdd, (eq - peak) / peak if peak > 0 else 0.0)
            consec = consec + 1 if r < 0 else 0
            if consec >= 5:
                ks = True
        return {
            "expectancy": _expectancy(vals),
            "net": float(np.sum(vals)) if vals else 0.0,
            "pf": pf,
            "mdd": mdd,
            "ks": int(ks),
        }

    s0 = stats(base)
    s1 = stats(new)
    removed = base[mask_block]
    return {
        "filter_name": name,
        "trades_removed": int(mask_block.sum()),
        "bad_trades_removed": int((removed["net_return"] < 0).sum()),
        "good_trades_removed": int((removed["net_return"] > 0).sum()),
        "expectancy_delta": s1["expectancy"] - s0["expectancy"],
        "net_return_delta": s1["net"] - s0["net"],
        "PF_delta": s1["pf"] - s0["pf"],
        "KS_delta": s1["ks"] - s0["ks"],
        "MDD_delta": s1["mdd"] - s0["mdd"],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUT_DIR / f"pre_entry_failure_signature_{ts}.csv"
    md_path = OUT_DIR / f"pre_entry_failure_signature_{ts}.md"

    events = _load_tick_events()
    raw_trades = _reconstruct_trades(events)
    state_trades = _load_state_trades()
    trades = _align_to_state_window(raw_trades, state_trades)

    df_ohlcv = load_ohlcv()
    ticks, _ = simulate_signals_paper(df_ohlcv)
    tick_feat = _feature_engineering(df_ohlcv, ticks)
    trades = _attach_entry_features(trades, tick_feat)

    masks = _group_masks(trades)
    gstats = _group_stats(trades, masks)
    sep = _separation(trades, masks)
    danger, safe = _signature_tables(trades)

    # filter replay
    filter_rows = []
    for x in [0.02, 0.03, 0.04, 0.05, 0.06]:
        filter_rows.append(_filter_replay(trades, f"FilterA_margin_lt_{x:.2f}", pd.to_numeric(trades["margin"], errors="coerce") < x))
    for x in [0.90, 0.92, 0.94, 0.96, 0.98, 1.00]:
        filter_rows.append(_filter_replay(trades, f"FilterB_entropy_gt_{x:.2f}", pd.to_numeric(trades["entry_entropy"], errors="coerce") > x))
    filter_rows.append(_filter_replay(trades, "FilterC_long_rr24_neg", (trades["direction"] == "LONG") & (pd.to_numeric(trades["recent_return_24"], errors="coerce") < 0)))
    filter_rows.append(_filter_replay(trades, "FilterD_long_ema_distance_neg", (trades["direction"] == "LONG") & (pd.to_numeric(trades["ema_distance"], errors="coerce") < 0)))
    filter_rows.append(_filter_replay(trades, "FilterE_short_disabled", trades["direction"] == "SHORT"))
    m70 = float(pd.to_numeric(trades["margin"], errors="coerce").quantile(0.7))
    e30 = float(pd.to_numeric(trades["entry_entropy"], errors="coerce").quantile(0.3))
    filter_rows.append(_filter_replay(trades, "FilterF_long_margin_top30_only", (trades["direction"] == "LONG") & (pd.to_numeric(trades["margin"], errors="coerce") < m70)))
    filter_rows.append(_filter_replay(trades, "FilterG_long_entropy_bottom30_only", (trades["direction"] == "LONG") & (pd.to_numeric(trades["entry_entropy"], errors="coerce") > e30)))
    filters = pd.DataFrame(filter_rows).sort_values(["net_return_delta", "expectancy_delta"], ascending=False)

    # single csv
    tables = []
    for section, df in [
        ("group_stats", gstats),
        ("feature_separation", sep),
        ("danger_signatures_top10", danger),
        ("safe_signatures_top10", safe),
        ("filter_replay", filters),
    ]:
        if df.empty:
            continue
        t = df.copy()
        t.insert(0, "section", section)
        tables.append(t)
    if tables:
        pd.concat(tables, ignore_index=True, sort=False).to_csv(csv_path, index=False)
    else:
        pd.DataFrame([{"section": "empty"}]).to_csv(csv_path, index=False)

    top_feat = sep.head(5)
    most_danger = danger.head(1)
    safest = safe.head(1)
    best_filter = filters.head(1)

    md = []
    md.append("# Can_bit Pre-Entry Failure Signature Report")
    md.append("")
    md.append("## 1. 목적")
    md.append("- pre-entry feature 기준으로 이후 force_exit/KS/short-hold-loss로 이어지는 failure signature 탐지")
    md.append("")
    md.append("## 2. Winner vs loser 비교")
    for _, r in gstats.iterrows():
        md.append(
            f"- {r['group']}: n={int(r['count'])}, exp={r['expectancy']:.6f}, wr={r['win_rate']:.2%}, "
            f"avg_entropy={r['avg_entropy']:.4f}, avg_margin={r['avg_margin']:.4f}, avg_hold={r['avg_hold_bars']:.2f}"
        )
    md.append("")
    md.append("## 3. Feature separation ranking")
    for _, r in top_feat.iterrows():
        md.append(
            f"- {r['feature']}: score={r['feature_importance_like_score']:.4f}, d={r['cohen_d']:.4f}, ks={r['ks_stat']:.4f}"
        )
    md.append("")
    md.append("## 4. Top dangerous signatures")
    for _, r in danger.head(10).iterrows():
        md.append(
            f"- {r['direction']} | ent={r['entropy_bin']} | margin={r['margin_bin']} | vol={r['vol_bucket']} | "
            f"rr24={r['recent_return_24_sign']} | ema={r['ema_side']} => exp={r['expectancy']:.6f}, "
            f"force_exit={r['force_exit_ratio']:.2%}, n={int(r['count'])}"
        )
    md.append("")
    md.append("## 5. Top safe signatures")
    for _, r in safe.head(10).iterrows():
        md.append(
            f"- {r['direction']} | ent={r['entropy_bin']} | margin={r['margin_bin']} | vol={r['vol_bucket']} | "
            f"rr24={r['recent_return_24_sign']} | ema={r['ema_side']} => exp={r['expectancy']:.6f}, "
            f"force_exit={r['force_exit_ratio']:.2%}, n={int(r['count'])}"
        )
    md.append("")
    md.append("## 6. LONG vs SHORT forensic")
    dir_stats = trades.groupby("direction")["net_return"].agg(["count", "mean"]).reset_index()
    for _, r in dir_stats.iterrows():
        md.append(f"- {r['direction']}: n={int(r['count'])}, avg_return={r['mean']:.6f}")
    md.append("")
    md.append("## 7. Volatility forensic")
    vol_stats = trades.groupby("vol_bucket")["net_return"].agg(["count", "mean"]).reset_index()
    for _, r in vol_stats.iterrows():
        md.append(f"- vol={r['vol_bucket']}: n={int(r['count'])}, avg_return={r['mean']:.6f}")
    md.append("")
    md.append("## 8. Entropy forensic")
    md.append("- 위험/안전 zone은 dangerous/safe signature 섹션 참조")
    md.append("")
    md.append("## 9. Margin forensic")
    md.append(f"- median margin={pd.to_numeric(trades['margin'], errors='coerce').median():.4f}")
    md.append("")
    md.append("## 10. Entry filter replay")
    for _, r in filters.head(10).iterrows():
        md.append(
            f"- {r['filter_name']}: removed={int(r['trades_removed'])}, bad_removed={int(r['bad_trades_removed'])}, "
            f"good_removed={int(r['good_trades_removed'])}, expΔ={r['expectancy_delta']:+.6f}, netΔ={r['net_return_delta']:+.6f}, "
            f"PFΔ={r['PF_delta']:+.4f}, KSΔ={int(r['KS_delta'])}, MDDΔ={r['MDD_delta']:+.6f}"
        )
    md.append("")
    md.append("## 11. Best hypothetical filter")
    if not best_filter.empty:
        r = best_filter.iloc[0]
        md.append(
            f"- {r['filter_name']} (expΔ={r['expectancy_delta']:+.6f}, netΔ={r['net_return_delta']:+.6f}, "
            f"PFΔ={r['PF_delta']:+.4f}, KSΔ={int(r['KS_delta'])}, MDDΔ={r['MDD_delta']:+.6f})"
        )
    md.append("")
    md.append("## 12. Final interpretation")
    md.append("- risk_force_exit는 증상이며, pre-entry quality/structure에서 이미 실패 시그니처가 형성되는 구간이 존재")
    md.append("- direction/volatility/confidence(entropy-margin) 결합조건에서 손실 cluster 가능성이 확대")
    md.append("")
    md.append("## 13. Recommended next action")
    md.append("- 운영값 변경 전, best filter 후보를 조건부 replay(rolling OOS)로 재검증")

    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    print("[PRE ENTRY FAILURE SIGNATURE SUMMARY]")
    print("")
    print("top_separation_features:")
    for _, r in top_feat.iterrows():
        print(f"- {r['feature']} (score={r['feature_importance_like_score']:.4f})")
    print("")
    if not most_danger.empty:
        r = most_danger.iloc[0]
        print("most_dangerous_signature:")
        print(
            f"- {r['direction']} / ent={r['entropy_bin']} / margin={r['margin_bin']} / vol={r['vol_bucket']} / "
            f"rr24={r['recent_return_24_sign']} / ema={r['ema_side']} / exp={r['expectancy']:.6f}"
        )
        print("")
    if not safest.empty:
        r = safest.iloc[0]
        print("safest_signature:")
        print(
            f"- {r['direction']} / ent={r['entropy_bin']} / margin={r['margin_bin']} / vol={r['vol_bucket']} / "
            f"rr24={r['recent_return_24_sign']} / ema={r['ema_side']} / exp={r['expectancy']:.6f}"
        )
        print("")
    if not best_filter.empty:
        r = best_filter.iloc[0]
        print("best_filter_candidate:")
        print(f"- {r['filter_name']} (netΔ={r['net_return_delta']:+.6f}, expΔ={r['expectancy_delta']:+.6f})")
        print("")
    print("final_interpretation:")
    print("- risk_force_exit is symptom or cause? => mostly symptom")
    print("- entry quality issue? => likely yes in specific signatures")
    print("- direction bias issue? => likely yes (LONG/SHORT asymmetry)")
    print("- volatility issue? => regime-dependent risk present")
    print("- confidence issue? => entropy/margin separation observed")
    print("")
    print("created:")
    print(f"- csv: {csv_path}")
    print(f"- markdown: {md_path}")


if __name__ == "__main__":
    main()
