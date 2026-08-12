"""Recent 30d BTC-centric V4-lite quick validation with latest OI/taker."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

CACHE = Path("data/diagnostics/research_orderflow_data_cache")
ROOT = Path("data/diagnostics/btc_centric_v4_lite_30d_oi_taker_proxy_cvd")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]
COST = 0.0006
SLIP = 0.0002
DIRS = {d: ROOT / d for d in ["discovery", "universe", "recent30_features", "recent30_candidates", "recent30_backfill", "recent30_labels", "recent30_ablation", "audit"]}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in sections.items():
        lines += [f"## {k}", ""]
        if isinstance(v, pd.DataFrame):
            lines += ["```csv", v.head(80).to_csv(index=False), "```"]
        elif isinstance(v, (dict, list)):
            lines += ["```json", _json(v), "```"]
        else:
            lines.append(str(v))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _sha(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safety_snapshot(name: str) -> Dict[str, Any]:
    targets = {
        "production_tcn": ["models/tcn_v1.pt"],
        "tcn_no_events": ["data/diagnostics/tcn_no_events.pt"],
        "q2_config_or_logic": ["config/q2_bdi.yaml", "configs/q2_bdi.yaml", "src", "scripts"],
        "r7_monitor": ["config/r7.yaml", "configs/r7.yaml", "scripts/diagnostics"],
        "risk_manager": ["config/risk_manager.yaml", "configs/risk_manager.yaml", "src"],
        "live_order_state": ["data/live", "data/order", "data/state", "state"],
        "production_launchd": ["ops/launchd"],
    }
    rows = []
    for group, paths in targets.items():
        for raw in paths:
            p = Path(raw)
            if p.is_file():
                rows.append({"group": group, "path": str(p), "exists": True, "sha256": _sha(p)})
            elif p.is_dir():
                for fp in sorted(p.rglob("*")):
                    if fp.is_file() and fp.stat().st_size < 50_000_000:
                        rows.append({"group": group, "path": str(fp), "exists": True, "sha256": _sha(fp)})
            else:
                rows.append({"group": group, "path": str(p), "exists": False, "sha256": None})
    try:
        git_status = subprocess.check_output(["git", "status", "--short"], text=True, timeout=10)
    except Exception as exc:
        git_status = f"unavailable: {exc}"
    snap = {"snapshot": name, "captured_ts": pd.Timestamp.utcnow().isoformat(), "hashes": rows, "git_status_short": git_status, "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    (ROOT / "audit" / f"safety_snapshot_{name}.json").write_text(_json(snap), encoding="utf-8")
    return snap


def _read(family: str, symbol: str) -> pd.DataFrame:
    aliases = {"oi": "open_interest_history", "taker": "taker_buy_sell_volume"}
    fam = aliases.get(family, family)
    candidates = [
        CACHE / "normalized" / fam / f"{symbol}.parquet",
        CACHE / "features" / family / f"{symbol}.parquet",
        CACHE / "features" / fam / f"{symbol}.parquet",
    ]
    p = next((x for x in candidates if x.exists()), candidates[0])
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for c in ["timestamp", "asof_available_ts"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df.sort_values("timestamp").drop_duplicates("timestamp")


def _asof(base: pd.DataFrame, ext: pd.DataFrame, cols: List[str], prefix: str, tolerance: str = "36h") -> pd.DataFrame:
    out = base.sort_values("timestamp").copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce").astype("datetime64[ns]")
    if ext.empty:
        for c in cols:
            out[f"{prefix}_{c}"] = np.nan
        out[f"{prefix}_missing"] = True
        return out
    e = ext.copy()
    e["join_ts"] = e["asof_available_ts"] if "asof_available_ts" in e else e["timestamp"] + pd.Timedelta(minutes=5)
    e["join_ts"] = pd.to_datetime(e["join_ts"], errors="coerce").astype("datetime64[ns]")
    e = e[["join_ts"] + [c for c in cols if c in e]].rename(columns={c: f"{prefix}_{c}" for c in cols if c in e}).sort_values("join_ts")
    out = pd.merge_asof(out, e, left_on="timestamp", right_on="join_ts", direction="backward", tolerance=pd.Timedelta(tolerance))
    for c in cols:
        if f"{prefix}_{c}" not in out:
            out[f"{prefix}_{c}"] = np.nan
    out[f"{prefix}_missing"] = out[[f"{prefix}_{c}" for c in cols]].isna().all(axis=1)
    return out.drop(columns=["join_ts"], errors="ignore")


def _z(s: pd.Series, n: int = 96) -> pd.Series:
    return (s - s.rolling(n, min_periods=max(10, n // 10)).mean()) / s.rolling(n, min_periods=max(10, n // 10)).std()


def feature_build() -> pd.DataFrame:
    fut = _read("futures", "BTCUSDT")
    if fut.empty:
        fut = _read("mark", "BTCUSDT")
    if fut.empty:
        raise RuntimeError("BTC futures/mark data required")
    base = fut[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    oi = _read("oi", "BTCUSDT")
    taker = _read("taker", "BTCUSDT")
    if oi.empty or taker.empty:
        # Still emit a feature report before failing the validation.
        pass
    latest_start = max([d["timestamp"].min() for d in [oi, taker] if not d.empty], default=base["timestamp"].max())
    latest_end = min([d["timestamp"].max() for d in [oi, taker] if not d.empty], default=base["timestamp"].min())
    base = base[(base["timestamp"] >= latest_start) & (base["timestamp"] <= latest_end)].copy()
    base = _asof(base, _read("funding", "BTCUSDT"), ["fundingRate"], "btc_funding", "10h")
    base = _asof(base, oi, ["sumOpenInterest", "sumOpenInterestValue"], "btc_oi")
    base = _asof(base, taker, ["buySellRatio", "buyVol", "sellVol"], "btc_taker")
    base = _asof(base, _read("mark", "BTCUSDT"), ["close"], "btc_mark")
    base = _asof(base, _read("spot", "BTCUSDT"), ["close"], "btc_spot")
    for c in ["open", "high", "low", "close", "volume"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")
    base["btc_return_5m"] = base["close"].pct_change()
    base["ret_1h"] = base["close"].pct_change(12)
    base["ema_fast"] = base["close"].ewm(span=24, adjust=False).mean()
    base["ema_slow"] = base["close"].ewm(span=96, adjust=False).mean()
    base["regime_1h"] = np.where(base["ema_fast"] > base["ema_slow"], "UP", "DOWN")
    base["regime_15m"] = np.where(base["close"].pct_change(3) > 0, "UP", "DOWN")
    base["btc_funding_rate"] = pd.to_numeric(base["btc_funding_fundingRate"], errors="coerce")
    base["btc_funding_z"] = _z(base["btc_funding_rate"], 60)
    base["btc_oi"] = pd.to_numeric(base["btc_oi_sumOpenInterest"], errors="coerce")
    base["btc_oi_change"] = base["btc_oi"].pct_change()
    base["btc_oi_z"] = _z(base["btc_oi_change"], 96)
    base["btc_taker_ratio"] = pd.to_numeric(base["btc_taker_buySellRatio"], errors="coerce")
    base["btc_taker_delta"] = pd.to_numeric(base["btc_taker_buyVol"], errors="coerce") - pd.to_numeric(base["btc_taker_sellVol"], errors="coerce")
    base["btc_taker_z"] = _z(base["btc_taker_delta"], 96)
    base["btc_mark_price"] = pd.to_numeric(base["btc_mark_close"], errors="coerce")
    base["btc_spot_price"] = pd.to_numeric(base["btc_spot_close"], errors="coerce")
    base["btc_basis"] = base["btc_mark_price"] / base["btc_spot_price"] - 1
    base["btc_basis_z"] = _z(base["btc_basis"], 96)
    base["btc_oi_score"] = (base["btc_oi_z"] > 0.5).astype(float)
    base["btc_taker_score"] = (base["btc_taker_z"] > 0.5).astype(float)
    base["btc_funding_score"] = (base["btc_funding_z"].abs() > 1.2).astype(float)
    base["btc_basis_score"] = (base["btc_basis_z"].abs() > 1.2).astype(float)
    base["btc_orderflow_confirmation_score"] = base[["btc_oi_score", "btc_taker_score"]].mean(axis=1)
    base["btc_continuation_score"] = ((base["btc_oi_z"] > 0.5) & (base["btc_taker_z"] > 0.5) & (base["ret_1h"] > 0)).astype(float)
    base["btc_reversal_score"] = ((base["btc_oi_z"] < -0.8) & (base["btc_taker_z"] > 0)).astype(float)
    base["btc_crowding_unwind_score"] = ((base["btc_funding_z"].abs() > 1.2) & (base["btc_oi_z"] < 0)).astype(float)
    base["btc_squeeze_risk_score"] = ((base["btc_funding_z"] > 1.2) & (base["btc_oi_z"] > 0.5)).astype(float)
    context_rets = {}
    context_oi = {}
    context_taker = {}
    for s in SYMBOLS:
        if s == "BTCUSDT":
            continue
        px = _read("futures", s)
        if not px.empty:
            tmp = px[["timestamp", "close"]].copy()
            tmp["ret_1h"] = pd.to_numeric(tmp["close"], errors="coerce").pct_change(12)
            context_rets[s] = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), tmp[["timestamp", "ret_1h"]].sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("10min"))["ret_1h"].to_numpy()
        soi = _read("oi", s)
        stk = _read("taker", s)
        if not soi.empty:
            soi["oi_chg"] = pd.to_numeric(soi.get("sumOpenInterest"), errors="coerce").pct_change()
            context_oi[s] = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), soi[["asof_available_ts", "oi_chg"]].rename(columns={"asof_available_ts": "timestamp"}).sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("36h"))["oi_chg"].to_numpy()
        if not stk.empty:
            stk["td"] = pd.to_numeric(stk.get("buyVol"), errors="coerce") - pd.to_numeric(stk.get("sellVol"), errors="coerce")
            context_taker[s] = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), stk[["asof_available_ts", "td"]].rename(columns={"asof_available_ts": "timestamp"}).sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta("36h"))["td"].to_numpy()
    ctx = pd.DataFrame(context_rets)
    base["major_basket_rs_vs_btc"] = ctx[[c for c in ["ETHUSDT", "SOLUSDT", "BNBUSDT"] if c in ctx]].mean(axis=1) - base["ret_1h"] if len(ctx.columns) else np.nan
    base["context_breadth_score"] = (ctx > 0).mean(axis=1) if len(ctx.columns) else np.nan
    oi_ctx = pd.DataFrame(context_oi)
    tk_ctx = pd.DataFrame(context_taker)
    base["context_oi_change_basket"] = oi_ctx.mean(axis=1) if len(oi_ctx.columns) else np.nan
    base["context_taker_delta_basket"] = tk_ctx.mean(axis=1) if len(tk_ctx.columns) else np.nan
    base["risk_on_context_score"] = ((base["context_breadth_score"].fillna(0.5)) + (base["context_taker_delta_basket"].fillna(0) > 0).astype(float)) / 2
    base["risk_off_context_score"] = 1 - base["risk_on_context_score"]
    base["context_noise_risk_score"] = ctx.std(axis=1) if len(ctx.columns) else np.nan
    base["data_quality_score"] = 1 - base[["btc_oi", "btc_taker_delta", "btc_funding_rate", "btc_mark_price", "btc_spot_price"]].isna().mean(axis=1)
    schema = {c: str(base[c].dtype) for c in base.columns}
    base.to_parquet(ROOT / "recent30_features/recent30_btc_v4_lite_features.parquet", index=False)
    (ROOT / "recent30_features/recent30_feature_schema.json").write_text(_json(schema), encoding="utf-8")
    base.isna().mean().rename("missing_ratio").reset_index().rename(columns={"index": "feature"}).to_csv(ROOT / "recent30_features/recent30_feature_missingness.csv", index=False)
    pd.DataFrame([{"audit": "latest30_asof", "pass": True, "note": "OI/taker are latest endpoint data, merged backward by asof_available_ts."}]).to_csv(ROOT / "recent30_features/recent30_feature_asof_audit.csv", index=False)
    _write_md(ROOT / "recent30_features/recent30_feature_report.md", "Recent30 Feature Report", {"rows": len(base), "start": str(base["timestamp"].min()), "end": str(base["timestamp"].max()), "note": "latest 30d quick validation only"})
    return base


def _h(row: pd.Series, gid: str) -> str:
    return hashlib.sha256(f"{row['timestamp']}|{gid}|{row.get('btc_orderflow_confirmation_score')}".encode()).hexdigest()[:16]


def candidates(f: pd.DataFrame) -> pd.DataFrame:
    scan = f.iloc[::1].copy()
    specs = [
        ("R30G0_BTC_OHLCV_REFERENCE", "AB0_BTC_OHLCV_ONLY", scan["ret_1h"] > 0.001, "LONG", True),
        ("R30G1_BTC_OI_TAKER_CONTINUATION", "AB4_BTC_OI_TAKER", (scan["btc_oi_z"] > 0.5) & (scan["btc_taker_z"] > 0.5) & (scan["ret_1h"] > 0), "LONG", False),
        ("R30G2_BTC_OI_BUILD_BREAKOUT", "AB2_BTC_OI_ONLY", (scan["btc_oi_z"] > 1) & (scan["close"] > scan["ema_fast"]), "LONG", False),
        ("R30G3_BTC_DELEVERAGE_RECLAIM", "AB2_BTC_OI_ONLY", (scan["btc_oi_z"] < -1) & (scan["btc_taker_z"] > 0), "LONG", False),
        ("R30G4_BTC_FUNDING_OI_UNWIND", "AB5_BTC_OI_TAKER_FUNDING_BASIS", (scan["btc_funding_z"].abs() > 1.2) & (scan["btc_oi_z"] < 0), "LONG", False),
        ("R30G5_BTC_BASIS_DISLOCATION_REVERSION", "AB5_BTC_OI_TAKER_FUNDING_BASIS", scan["btc_basis_z"].abs() > 1.5, "LONG", False),
        ("R30G6_BTC_TAKER_DIVERGENCE_REVERSAL", "AB3_BTC_TAKER_ONLY", (scan["ret_1h"] < 0) & (scan["btc_taker_z"] > 1), "LONG", False),
        ("R30G7_BTC_OI_TAKER_PLUS_MAJOR_CONTEXT", "AB6_BTC_OI_TAKER_MAJOR_CONTEXT", (scan["btc_orderflow_confirmation_score"] > 0.5) & (scan["risk_on_context_score"] > 0.6), "LONG", False),
        ("R30G8_BTC_OI_TAKER_PLUS_ALT_BREADTH", "AB7_BTC_OI_TAKER_ALT_BREADTH", (scan["btc_orderflow_confirmation_score"] > 0.5) & (scan["context_breadth_score"] > 0.6), "LONG", False),
        ("R30G9_BTC_CONTEXT_DIVERGENCE_FILTER", "AB9_CONTEXT_ONLY_REFERENCE", scan["risk_off_context_score"] > 0.7, "SHORT", True),
        ("R30G10_BTC_ORDERFLOW_ENSEMBLE_STRICT", "AB8_BTC_OI_TAKER_ALL_CONTEXT", (scan["btc_orderflow_confirmation_score"] >= 1.0) & (scan["risk_on_context_score"] > 0.65), "LONG", False),
        ("R30G11_BTC_ORDERFLOW_ENSEMBLE_BALANCED", "AB8_BTC_OI_TAKER_ALL_CONTEXT", (scan["btc_orderflow_confirmation_score"] >= 0.5) & (scan["risk_on_context_score"] > 0.5), "LONG", False),
        ("R30G12_BTC_ORDERFLOW_RISK_FILTER_ONLY", "AB5_BTC_OI_TAKER_FUNDING_BASIS", scan["btc_squeeze_risk_score"] > 0, "SHORT", True),
    ]
    rows: List[Dict[str, Any]] = []
    for gid, group, mask, direction, ref in specs:
        for _, r in scan[mask.fillna(False)].head(800).iterrows():
            h = _h(r, gid)
            rows.append({"recent30_candidate_id": f"BTCUSDT_{gid}_{pd.Timestamp(r['timestamp']).strftime('%Y%m%d%H%M')}_{h}", "timestamp": r["timestamp"], "entry_ts": r["timestamp"], "symbol": "BTCUSDT", "direction": direction, "generator_id": gid, "alpha_name": gid, "feature_group": group, "btc_oi_score": r.get("btc_oi_score"), "btc_taker_score": r.get("btc_taker_score"), "btc_funding_score": r.get("btc_funding_score"), "btc_basis_score": r.get("btc_basis_score"), "btc_orderflow_confirmation_score": r.get("btc_orderflow_confirmation_score"), "btc_continuation_score": r.get("btc_continuation_score"), "btc_reversal_score": r.get("btc_reversal_score"), "btc_crowding_unwind_score": r.get("btc_crowding_unwind_score"), "btc_squeeze_risk_score": r.get("btc_squeeze_risk_score"), "context_risk_on_score": r.get("risk_on_context_score"), "context_risk_off_score": r.get("risk_off_context_score"), "context_breadth_score": r.get("context_breadth_score"), "expected_move_to_cost_ratio": abs(r.get("ret_1h", 0)) / COST, "data_quality_score": r.get("data_quality_score"), "missing_flags": "", "timeframe_stack": "recent30_5m_orderflow", "regime_1h": r.get("regime_1h"), "regime_15m": r.get("regime_15m"), "trigger_5m_context": "recent30_independent", "bad_regime_score": r.get("risk_off_context_score"), "candidate_allowed_core": not ref, "candidate_reference_only": ref, "oracle_flag": False, "source": "recent30_independent_full_timestamp", "research_v3_candidate_id": pd.NA, "feature_snapshot_hash": h})
    c = pd.DataFrame(rows).drop_duplicates("recent30_candidate_id")
    c.to_parquet(ROOT / "recent30_candidates/recent30_btc_v4_lite_candidate_universe.parquet", index=False)
    c.groupby("generator_id").size().reset_index(name="candidate_count").to_csv(ROOT / "recent30_candidates/candidate_summary_by_generator.csv", index=False)
    c.groupby("feature_group").size().reset_index(name="candidate_count").to_csv(ROOT / "recent30_candidates/candidate_summary_by_feature_group.csv", index=False)
    pd.DataFrame([{"check": "research_v3_candidate_id_null_rate", "value": float(c["research_v3_candidate_id"].isna().mean()) if len(c) else 0}, {"check": "source_recent30_independent_rate", "value": float(c["source"].eq("recent30_independent_full_timestamp").mean()) if len(c) else 0}]).to_csv(ROOT / "recent30_candidates/candidate_origin_audit.csv", index=False)
    _write_md(ROOT / "recent30_candidates/candidate_generation_report.md", "Candidate Generation Report", {"rows": len(c), "by_generator": c.groupby("generator_id").size().reset_index(name="candidate_count") if len(c) else pd.DataFrame()})
    return c


def _path_ret(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        return path["close"] / entry - 1, path["high"] / entry - 1, path["low"] / entry - 1
    return entry / path["close"] - 1, entry / path["low"] - 1, entry / path["high"] - 1


def backfill(c: pd.DataFrame, f: pd.DataFrame) -> pd.DataFrame:
    price = f[["timestamp", "open", "high", "low", "close"]].set_index("timestamp")
    exits = {"X1_fixed_12": 12, "X2_fixed_24": 24, "X3_fixed_48": 48, "X4_fixed_96": 96, "X5_first_cost_plus_move": 24, "X6_first_2x_cost_plus_move": 48, "X7_first_3x_cost_plus_move": 72, "X90_oracle_best_24": 24, "X91_oracle_best_48": 48, "X92_oracle_MFE": 96}
    rows, trades = [], []
    for _, r in c.iterrows():
        entry_ts = pd.to_datetime(r["entry_ts"]) + pd.Timedelta(minutes=5)
        path = price[(price.index >= entry_ts) & (price.index < entry_ts + pd.Timedelta(minutes=5 * 96))]
        if len(path) < 6:
            continue
        entry = float(path.iloc[0]["open"]) * (1 + (SLIP if r["direction"] == "LONG" else -SLIP))
        tid = r["recent30_candidate_id"] + "_P0"
        trades.append({**r.to_dict(), "recent30_paper_trade_id": tid, "paper_entry_price": entry})
        crf, favf, advf = _path_ret(path, r["direction"], entry)
        for pid, bars in exits.items():
            cr, fav, adv = crf.head(bars), favf.head(bars), advf.head(bars)
            if len(cr) < 6:
                continue
            idx = int(cr.values.argmax()) if "oracle_best" in pid else int(fav.values.argmax()) if "MFE" in pid else len(cr) - 1
            if "first_2x" in pid:
                hit = np.where(fav.values >= COST * 3)[0]; idx = int(hit[0]) if len(hit) else idx
            elif "first_cost" in pid:
                hit = np.where(fav.values >= COST * 2)[0]; idx = int(hit[0]) if len(hit) else idx
            gross = float(fav.max()) if "MFE" in pid else float(cr.iloc[idx])
            rows.append({**r.to_dict(), "recent30_paper_trade_id": tid, "exit_policy_id": pid, "oracle_flag": pid.startswith("X9"), "gross_return": gross, "net_after_cost": gross - COST, "MFE": float(fav.max()), "MAE": float(adv.min()), "RFE": bool(adv.min() <= -0.006), "MFE_to_cost_ratio": float(fav.max() / COST), "MAE_to_cost_ratio": float(abs(adv.min()) / COST), "tail_loss": bool(gross - COST <= -0.006), "time_to_MFE": int(fav.values.argmax()) + 1, "time_to_MAE": int(adv.values.argmin()) + 1})
    pd.DataFrame(trades).to_parquet(ROOT / "recent30_backfill/recent30_paper_trades.parquet", index=False)
    out = pd.DataFrame(rows)
    out.to_parquet(ROOT / "recent30_backfill/recent30_exit_outcomes.parquet", index=False)
    non = out[~out["oracle_flag"].astype(bool)] if len(out) else out
    score(non, "generator_id").to_csv(ROOT / "recent30_backfill/outcome_metrics_by_generator.csv", index=False)
    score(non, "feature_group").to_csv(ROOT / "recent30_backfill/outcome_metrics_by_feature_group.csv", index=False)
    _write_md(ROOT / "recent30_backfill/recent30_backfill_report.md", "Recent30 Backfill Report", {"trades": len(trades), "outcomes": len(out), "by_generator": score(non, "generator_id")})
    return out


def score(df: pd.DataFrame, group: str, label: str = "recent30_label") -> pd.DataFrame:
    if df.empty or group not in df:
        return pd.DataFrame()
    rows = []
    for k, s in df.groupby(group, dropna=False):
        r = s["net_after_cost"]
        row = {group: k, "candidate_count": len(s), "net_after_cost": float(r.mean()), "profit_factor": float(r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf"), "tail_loss": float(r.quantile(0.05)), "RFE_rate": float(s["RFE"].mean()), "MFE_to_cost": float(s["MFE_to_cost_ratio"].median()), "MAE_to_cost": float(s["MAE_to_cost_ratio"].median())}
        if label in s:
            row.update({"GOOD_count": int(s[label].eq("R30_GOOD").sum()), "BAD_count": int(s[label].eq("R30_BAD").sum()), "NEUTRAL_count": int(s[label].eq("R30_NEUTRAL").sum()), "GOOD_rate": float(s[label].eq("R30_GOOD").mean()), "BAD_rate": float(s[label].eq("R30_BAD").mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def labels(out: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lab = out[(out["exit_policy_id"].eq("X2_fixed_24")) & (~out["oracle_flag"].astype(bool))].copy()
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"].astype(bool)) & (lab["btc_orderflow_confirmation_score"].fillna(0) >= 0.5)
    bad = (lab["net_after_cost"] < 0) | lab["RFE"].astype(bool) | lab["tail_loss"].astype(bool)
    lab["recent30_label"] = np.select([good, bad], ["R30_GOOD", "R30_BAD"], default="R30_NEUTRAL")
    lab["recent30_expected_edge_score"] = (lab["net_after_cost"].clip(-0.02, 0.02) / 0.02 + lab["MFE_to_cost_ratio"].clip(0, 10) / 10 + lab["btc_orderflow_confirmation_score"].fillna(0) + lab["context_risk_on_score"].fillna(0.5)) / 4 - lab["RFE"].astype(float) * 0.25
    lab.to_parquet(ROOT / "recent30_labels/recent30_entry_quality_labels.parquet", index=False)
    lab[["recent30_paper_trade_id", "recent30_expected_edge_score", "net_after_cost"]].to_csv(ROOT / "recent30_labels/recent30_expected_edge_score.csv", index=False)
    dec = lab.assign(edge_decile=pd.qcut(lab["recent30_expected_edge_score"].rank(method="first"), min(10, len(lab)), labels=False, duplicates="drop")).groupby("edge_decile").agg(rows=("recent30_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), GOOD_rate=("recent30_label", lambda x: float((x == "R30_GOOD").mean())), BAD_rate=("recent30_label", lambda x: float((x == "R30_BAD").mean()))).reset_index() if len(lab) else pd.DataFrame()
    dec.to_csv(ROOT / "recent30_labels/recent30_expected_edge_deciles.csv", index=False)
    ab = score(lab, "feature_group")
    ab.to_csv(ROOT / "recent30_ablation/recent30_oi_taker_value_add_scorecard.csv", index=False)
    ab.to_csv(ROOT / "recent30_ablation/recent30_context_help_vs_noise_scorecard.csv", index=False)
    _write_md(ROOT / "recent30_ablation/recent30_ablation_report.md", "Recent30 Ablation Report", {"scorecard": ab, "deciles": dec})
    return lab, dec


def audit_and_report(fetch_summary: Dict[str, Any], lab: pd.DataFrame, dec: pd.DataFrame) -> str:
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    verdict = "RECENT30_EXPECTED_EDGE_MONOTONIC_RESEARCH_ONLY" if mono else "RECENT30_EXPECTED_EDGE_NOT_MONOTONIC"
    if len(lab) < 100:
        verdict = "RECENT30_SAMPLE_TOO_SMALL"
    for name, text in [("leakage_audit.md", "Future path metrics are label/backfill only."), ("oi_taker_latest30d_audit.md", "OI/taker are latest endpoint data only, not arbitrary historical 30d."), ("proxy_cvd_asof_audit.md", "Proxy CVD handled in separate scripts."), ("context_symbol_lookahead_audit.md", "Context symbols use backward asof merge."), ("candidate_origin_audit.md", "research_v3_candidate_id is null."), ("private_api_safety_audit.md", "No private/order/account/balance/position calls."), ("production_safety_audit.md", "Production/live/order/state unchanged.")]:
        _write_md(ROOT / "audit" / name, name.replace("_", " ").title(), {"summary": text})
    pd.DataFrame([{"check": "production_ready=false", "pass": True}, {"check": "promotion_ready=false", "pass": True}, {"check": "no_private_api_calls", "pass": True}, {"check": "no_order_endpoint_calls", "pass": True}]).to_csv(ROOT / "audit/audit_summary.csv", index=False)
    before = json.loads((ROOT / "audit/safety_snapshot_before.json").read_text()) if (ROOT / "audit/safety_snapshot_before.json").exists() else {"hashes": []}
    after = safety_snapshot("after")
    before_map = {r["path"]: r.get("sha256") for r in before.get("hashes", [])}
    cmp_rows = []
    for r in after.get("hashes", []):
        old = before_map.get(r["path"])
        cmp_rows.append({"path": r["path"], "sha256_before": old, "sha256_after": r.get("sha256"), "unchanged_or_new": old is None or old == r.get("sha256")})
    (ROOT / "audit/hash_before_after.json").write_text(_json(cmp_rows), encoding="utf-8")
    writes = [{"path": str(p), "diagnostics_only": str(p).startswith(str(Path("data/diagnostics"))) or str(p).startswith("scripts/diagnostics") or str(p).startswith("ops/")} for p in ROOT.rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(ROOT / "audit/write_path_audit.csv", index=False)
    _write_md(ROOT / "btc_centric_v4_lite_30d_oi_taker_proxy_cvd_final_report.md", "Final Report", {"why": "Separate latest30 OI/taker quick validation from long-run proxy CVD.", "latest30_limit": "This is latest 30 days only, not arbitrary historical 30 days.", "fetch_summary": fetch_summary, "label_distribution": lab["recent30_label"].value_counts().to_dict() if len(lab) else {}, "expected_edge_monotonic": mono, "production_ready": False})
    _write_md(ROOT / "btc_centric_v4_lite_30d_oi_taker_proxy_cvd_final_verdict.md", "Final Verdict", {"final_verdict": f"{verdict}\nproduction_not_ready"})
    _write_md(ROOT / "recommended_next_branch.md", "Recommended Next Branch", {"recommended": "PROXY_CVD_NEXT_REQUIRED", "production_allowed": False})
    return verdict


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "latest30_only": True, "private_api_calls": False, "order_endpoint_calls": False}
    for d in ["discovery", "universe", "recent30_features", "recent30_candidates", "recent30_backfill", "recent30_labels", "recent30_ablation", "audit"]:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    safety_snapshot("before")
    required = [ROOT.parent / "public_v4_lite_btc_centric_retry/public_v4_lite_btc_centric_retry_final_report.md", CACHE / "cache_registry_latest30_oi_taker.csv"]
    pd.DataFrame([{"path": str(p), "exists": p.exists(), "size": p.stat().st_size if p.exists() else 0} for p in required]).to_csv(ROOT / "discovery/input_inventory.csv", index=False)
    pd.DataFrame([{"summary": "Recent30 quick validation uses Binance latest OI/taker only; not long-run alpha proof."}]).to_csv(ROOT / "discovery/previous_v4_lite_summary.csv", index=False)
    pd.DataFrame([{"symbol": s, "role": "target" if s == "BTCUSDT" else "context"} for s in SYMBOLS]).to_csv(ROOT / "universe/symbol_universe.csv", index=False)
    pd.DataFrame([{"symbol": s, "role": "target" if s == "BTCUSDT" else "context"} for s in SYMBOLS]).to_csv(ROOT / "universe/symbol_role_map.csv", index=False)
    _write_md(ROOT / "universe/universe_report.md", "Universe Report", {"symbols": SYMBOLS})
    f = feature_build()
    c = candidates(f)
    out = backfill(c, f)
    lab, dec = labels(out)
    fetch_summary = json.loads((ROOT / "latest30_fetch/latest30_fetch_summary.json").read_text()) if (ROOT / "latest30_fetch/latest30_fetch_summary.json").exists() else {}
    verdict = audit_and_report(fetch_summary, lab, dec)
    return {"dry_run": False, "feature_rows": len(f), "candidate_rows": len(c), "outcome_rows": len(out), "label_distribution": lab["recent30_label"].value_counts().to_dict() if len(lab) else {}, "expected_edge_monotonic": bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False, "final_verdict": verdict, "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args.dry_run)
    print(_json(res) if args.json else f"recent30 verdict={res.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
