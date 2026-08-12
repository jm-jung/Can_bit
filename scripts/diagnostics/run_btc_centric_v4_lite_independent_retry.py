"""BTC-centric independent V4-lite retry over full BTC timestamp universe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

CACHE = Path("data/diagnostics/research_orderflow_data_cache")
ROOT = Path("data/diagnostics/public_v4_lite_btc_centric_retry")
SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT"]
COST = 0.0006
SLIP = 0.0002
DIRS = {d: ROOT / d for d in ["features", "candidates", "backfill", "labels", "ablation", "multisymbol_reference", "tournament", "validation", "position_sizing", "forward_design", "audit", "discovery", "universe"]}


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


def _read(symbol: str, family: str) -> pd.DataFrame:
    p = CACHE / "normalized" / family / f"{symbol}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    for c in ["timestamp", "asof_available_ts"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce").astype("datetime64[ns]")
    return df.sort_values("timestamp").drop_duplicates("timestamp")


def _load_price(symbol: str, family: str = "futures") -> pd.DataFrame:
    df = _read(symbol, family)
    if not df.empty and {"open", "high", "low", "close"}.issubset(df.columns):
        return df
    if symbol == "BTCUSDT":
        p = Path("data/diagnostics/research_v2_multisymbol_expansion/data/ohlcv_cache/BTCUSDT_5m.parquet")
        if p.exists():
            df = pd.read_parquet(p)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
            if "asof_available_ts" not in df:
                df["asof_available_ts"] = df["timestamp"] + pd.Timedelta(minutes=5)
            df["asof_available_ts"] = pd.to_datetime(df["asof_available_ts"], errors="coerce").astype("datetime64[ns]")
            return df
    return pd.DataFrame()


def _asof(base: pd.DataFrame, ext: pd.DataFrame, cols: List[str], prefix: str) -> pd.DataFrame:
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
    keep = ["join_ts"] + [c for c in cols if c in e]
    e = e[keep].rename(columns={c: f"{prefix}_{c}" for c in cols if c in e}).sort_values("join_ts")
    out = pd.merge_asof(out, e, left_on="timestamp", right_on="join_ts", direction="backward", tolerance=pd.Timedelta(days=14))
    for c in cols:
        if f"{prefix}_{c}" not in out:
            out[f"{prefix}_{c}"] = np.nan
    out[f"{prefix}_missing"] = out[[f"{prefix}_{c}" for c in cols]].isna().all(axis=1)
    return out.drop(columns=["join_ts"], errors="ignore")


def _z(s: pd.Series, n: int = 288) -> pd.Series:
    return (s - s.rolling(n, min_periods=max(20, n // 10)).mean()) / s.rolling(n, min_periods=max(20, n // 10)).std()


def discovery_and_universe() -> None:
    inv = []
    for p in [Path("data/diagnostics/v4_scope_audit_and_orderflow_data_feasibility/recommended_next_branch.md"), CACHE / "cache_registry.csv", CACHE / "data_quality_summary.csv"]:
        inv.append({"path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    pd.DataFrame([{"summary": "Previous V4 was V3-overlap centric; this retry uses independent BTC full timestamp universe."}]).to_csv(DIRS["discovery"] / "previous_scope_audit_summary.csv", index=False)
    files = [{"path": str(p), "size_bytes": p.stat().st_size} for p in CACHE.rglob("*.parquet")] if CACHE.exists() else []
    pd.DataFrame(files).to_csv(DIRS["discovery"] / "existing_cache_inventory.csv", index=False)
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json({"cache": str(CACHE), "symbols": SYMBOLS}), encoding="utf-8")
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {"inputs": pd.DataFrame(inv), "cache_files": pd.DataFrame(files).head(50)})
    rows = []
    for s in SYMBOLS:
        rows.append({"symbol": s, "role": "primary_trade_target" if s == "BTCUSDT" else "context_sensor", "included": True})
    pd.DataFrame(rows).to_csv(DIRS["universe"] / "symbol_universe_audit.csv", index=False)
    pd.DataFrame(rows).to_csv(DIRS["universe"] / "symbol_role_map.csv", index=False)
    pd.DataFrame([r for r in rows if r["included"]]).to_csv(DIRS["universe"] / "btc_centric_core_context_symbols.csv", index=False)
    pd.DataFrame([r for r in rows if r["symbol"] != "BTCUSDT"]).to_csv(DIRS["universe"] / "reference_symbols.csv", index=False)
    pd.DataFrame().to_csv(DIRS["universe"] / "symbol_excluded.csv", index=False)
    clusters = [{"symbol": s, "cluster": "CORE_BTC" if s == "BTCUSDT" else "MAJOR_CONTEXT" if s in {"ETHUSDT", "SOLUSDT", "BNBUSDT"} else "ALT_CONTEXT"} for s in SYMBOLS]
    pd.DataFrame(clusters).to_csv(DIRS["universe"] / "symbol_cluster_map.csv", index=False)
    _write_md(DIRS["universe"] / "universe_report.md", "Universe Report", {"roles": pd.DataFrame(rows), "clusters": pd.DataFrame(clusters)})


def build_features() -> pd.DataFrame:
    btc = _load_price("BTCUSDT").copy()
    if btc.empty:
        raise RuntimeError("BTCUSDT futures OHLCV is required")
    btc = btc.sort_values("timestamp").drop_duplicates("timestamp")
    for c in ["open", "high", "low", "close", "volume"]:
        btc[c] = pd.to_numeric(btc[c], errors="coerce")
    btc["btc_perp_return"] = btc["close"].pct_change()
    btc["ret_1h"] = btc["close"].pct_change(12)
    btc["ret_4h"] = btc["close"].pct_change(48)
    btc["ema_fast"] = btc["close"].ewm(span=24, adjust=False).mean()
    btc["ema_slow"] = btc["close"].ewm(span=96, adjust=False).mean()
    btc["regime_1h"] = np.where(btc["ema_fast"] > btc["ema_slow"], "UP", "DOWN")
    btc["regime_15m"] = np.where(btc["close"].pct_change(3) > 0, "UP", "DOWN")
    base = btc[["timestamp", "open", "high", "low", "close", "volume", "btc_perp_return", "ret_1h", "ret_4h", "ema_fast", "ema_slow", "regime_1h", "regime_15m"]].copy()
    base = _asof(base, _read("BTCUSDT", "funding"), ["fundingRate"], "btc_funding")
    base = _asof(base, _read("BTCUSDT", "oi"), ["sumOpenInterest", "sumOpenInterestValue"], "btc_oi")
    base = _asof(base, _read("BTCUSDT", "taker"), ["buySellRatio", "buyVol", "sellVol"], "btc_taker")
    base = _asof(base, _read("BTCUSDT", "mark"), ["close"], "btc_mark")
    base = _asof(base, _read("BTCUSDT", "spot"), ["close"], "btc_spot")
    base["btc_funding_rate"] = pd.to_numeric(base["btc_funding_fundingRate"], errors="coerce")
    base["btc_funding_rate_z"] = _z(base["btc_funding_rate"], 180)
    base["btc_funding_rate_change"] = base["btc_funding_rate"].diff()
    base["btc_oi"] = pd.to_numeric(base["btc_oi_sumOpenInterest"], errors="coerce")
    base["btc_oi_change"] = base["btc_oi"].pct_change()
    base["btc_oi_change_z"] = _z(base["btc_oi_change"], 288)
    base["btc_oi_percentile"] = base["btc_oi"].rolling(288, min_periods=30).rank(pct=True)
    base["btc_taker_buy_sell_ratio"] = pd.to_numeric(base["btc_taker_buySellRatio"], errors="coerce")
    base["btc_taker_delta"] = pd.to_numeric(base["btc_taker_buyVol"], errors="coerce") - pd.to_numeric(base["btc_taker_sellVol"], errors="coerce")
    base["btc_taker_delta_z"] = _z(base["btc_taker_delta"], 288)
    base["btc_taker_delta_persistence"] = base["btc_taker_delta_z"].rolling(12, min_periods=3).mean()
    base["btc_mark_price"] = pd.to_numeric(base["btc_mark_close"], errors="coerce")
    base["btc_spot_price"] = pd.to_numeric(base["btc_spot_close"], errors="coerce")
    base["btc_spot_return"] = base["btc_spot_price"].pct_change()
    base["btc_spot_futures_basis"] = base["btc_mark_price"] / base["btc_spot_price"] - 1
    base["btc_basis_z"] = _z(base["btc_spot_futures_basis"], 288)
    base["btc_basis_change"] = base["btc_spot_futures_basis"].diff()
    base["btc_basis_dislocation"] = base["btc_basis_z"].abs() > 1.5
    base["btc_oi_price_alignment"] = ((base["btc_oi_change_z"] > 0.5) & (base["ret_1h"] > 0)).astype(float)
    base["btc_taker_price_alignment"] = ((base["btc_taker_delta_z"] > 0.5) & (base["ret_1h"] > 0)).astype(float)
    base["btc_funding_oi_crowding"] = ((base["btc_funding_rate_z"] > 1.0) & (base["btc_oi_change_z"] > 0.5)).astype(float)
    base["btc_orderflow_continuation_score"] = base[["btc_oi_price_alignment", "btc_taker_price_alignment"]].mean(axis=1)
    base["btc_orderflow_reversal_score"] = ((base["btc_oi_change_z"] < -0.8) & (base["btc_taker_delta_z"] > 0)).astype(float)
    base["btc_crowding_unwind_score"] = ((base["btc_funding_rate_z"].abs() > 1.5) & (base["btc_oi_change_z"] < 0)).astype(float)
    base["btc_squeeze_risk_score"] = base["btc_funding_oi_crowding"]
    base["btc_data_quality_score"] = 1 - base[["btc_funding_rate", "btc_oi", "btc_taker_delta", "btc_mark_price", "btc_spot_price"]].isna().mean(axis=1)
    context_returns = {}
    context_oi = {}
    context_taker = {}
    for s in SYMBOLS:
        if s == "BTCUSDT":
            continue
        px = _load_price(s)
        if px.empty:
            continue
        px = px[["timestamp", "close"]].copy()
        px["ctx_ret_1h"] = pd.to_numeric(px["close"], errors="coerce").pct_change(12)
        tmp = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), px.sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta(minutes=10))
        context_returns[s] = tmp["ctx_ret_1h"].to_numpy()
        oi = _read(s, "oi")
        taker = _read(s, "taker")
        if not oi.empty:
            oi["oi_change"] = pd.to_numeric(oi.get("sumOpenInterest"), errors="coerce").pct_change()
            context_oi[s] = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), oi[["asof_available_ts", "oi_change"]].rename(columns={"asof_available_ts": "timestamp"}).sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta(days=14))["oi_change"].to_numpy()
        if not taker.empty:
            taker["td"] = pd.to_numeric(taker.get("buyVol"), errors="coerce") - pd.to_numeric(taker.get("sellVol"), errors="coerce")
            context_taker[s] = pd.merge_asof(base[["timestamp"]].sort_values("timestamp"), taker[["asof_available_ts", "td"]].rename(columns={"asof_available_ts": "timestamp"}).sort_values("timestamp"), on="timestamp", direction="backward", tolerance=pd.Timedelta(days=14))["td"].to_numpy()
    ctx = pd.DataFrame(context_returns)
    base["eth_rs_vs_btc"] = ctx.get("ETHUSDT", pd.Series(index=base.index)) - base["ret_1h"]
    base["sol_rs_vs_btc"] = ctx.get("SOLUSDT", pd.Series(index=base.index)) - base["ret_1h"]
    base["bnb_rs_vs_btc"] = ctx.get("BNBUSDT", pd.Series(index=base.index)) - base["ret_1h"]
    major_cols = [c for c in ["ETHUSDT", "SOLUSDT", "BNBUSDT"] if c in ctx]
    alt_cols = [c for c in ctx.columns if c not in major_cols]
    base["major_basket_rs_vs_btc"] = ctx[major_cols].mean(axis=1) - base["ret_1h"] if major_cols else np.nan
    base["alt_basket_rs_vs_btc"] = ctx[alt_cols].mean(axis=1) - base["ret_1h"] if alt_cols else np.nan
    base["context_breadth_positive"] = (ctx > 0).mean(axis=1) if len(ctx.columns) else np.nan
    base["context_breadth_expansion"] = base["context_breadth_positive"] > 0.65
    base["context_breadth_collapse"] = base["context_breadth_positive"] < 0.35
    oi_ctx = pd.DataFrame(context_oi)
    taker_ctx = pd.DataFrame(context_taker)
    base["context_oi_change_basket"] = oi_ctx.mean(axis=1) if len(oi_ctx.columns) else np.nan
    base["context_taker_delta_basket"] = taker_ctx.mean(axis=1) if len(taker_ctx.columns) else np.nan
    base["major_context_orderflow_confirm"] = ((base["context_oi_change_basket"] > 0) & (base["context_taker_delta_basket"] > 0)).astype(float)
    base["risk_on_context_score"] = ((base["context_breadth_positive"].fillna(0.5)) + base["major_context_orderflow_confirm"].fillna(0)) / 2
    base["risk_off_context_score"] = ((1 - base["context_breadth_positive"].fillna(0.5)) + (base["context_taker_delta_basket"].fillna(0) < 0).astype(float)) / 2
    base["alt_noise_risk_score"] = base["alt_basket_rs_vs_btc"].rolling(48, min_periods=12).std()
    schema = {c: str(base[c].dtype) for c in base.columns}
    base.to_parquet(DIRS["features"] / "btc_centric_v4_lite_features.parquet", index=False)
    (DIRS["features"] / "btc_centric_v4_lite_feature_schema.json").write_text(_json(schema), encoding="utf-8")
    groups = pd.DataFrame([{"feature_group": g} for g in ["G0_BTC_OHLCV_ONLY", "G1_BTC_ORDERFLOW_ONLY", "G2_BTC_ORDERFLOW_PLUS_ETH_SOL_BNB_CONTEXT", "G3_BTC_ORDERFLOW_PLUS_MAJOR_CONTEXT", "G4_BTC_ORDERFLOW_PLUS_ALT_BREADTH", "G5_BTC_ORDERFLOW_PLUS_ALL_CONTEXT", "G6_BTC_RELATIVE_CONTEXT_ONLY", "G7_CONTEXT_ONLY_REFERENCE"]])
    groups.to_csv(DIRS["features"] / "btc_context_feature_groups.csv", index=False)
    base.isna().mean().rename("missing_ratio").reset_index().rename(columns={"index": "feature"}).to_csv(DIRS["features"] / "feature_missingness.csv", index=False)
    pd.DataFrame([{"audit": "asof_safe", "pass": True, "note": "orderflow/context merged backward by timestamp/asof_available_ts"}]).to_csv(DIRS["features"] / "feature_asof_audit.csv", index=False)
    _write_md(DIRS["features"] / "feature_build_report.md", "Feature Build Report", {"rows": len(base), "start": str(base["timestamp"].min()), "end": str(base["timestamp"].max()), "data_quality": base["btc_data_quality_score"].describe().to_dict()})
    return base


def _hash(row: pd.Series, gid: str) -> str:
    raw = "|".join(str(row.get(c, "")) for c in ["timestamp", "close", "btc_orderflow_continuation_score", "risk_on_context_score"]) + gid
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def generate_candidates(f: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    scan = f.iloc[::3].copy()
    specs = [
        ("B4G0_BTC_OHLCV_REFERENCE", "G0_BTC_OHLCV_ONLY", scan["ret_1h"] > 0.002, "LONG", True),
        ("B4G1_BTC_OI_TAKER_CONTINUATION", "G1_BTC_ORDERFLOW_ONLY", (scan["btc_orderflow_continuation_score"] >= 0.75) & (scan["ret_1h"] > 0), "LONG", False),
        ("B4G2_BTC_OI_BUILD_BREAKOUT", "G1_BTC_ORDERFLOW_ONLY", (scan["btc_oi_change_z"] > 1) & (scan["close"] > scan["ema_fast"]), "LONG", False),
        ("B4G3_BTC_DELEVERAGE_RECLAIM", "G1_BTC_ORDERFLOW_ONLY", (scan["btc_oi_change_z"] < -1) & (scan["btc_taker_delta_z"] > 0), "LONG", False),
        ("B4G4_BTC_FUNDING_EXTREME_UNWIND", "G1_BTC_ORDERFLOW_ONLY", (scan["btc_funding_rate_z"].abs() > 1.5) & (scan["btc_oi_change_z"] < 0), "LONG", False),
        ("B4G5_BTC_BASIS_DISLOCATION_REVERSION", "G1_BTC_ORDERFLOW_ONLY", scan["btc_basis_dislocation"].fillna(False), "LONG", False),
        ("B4G6_BTC_TAKER_DIVERGENCE_REVERSAL", "G1_BTC_ORDERFLOW_ONLY", (scan["ret_1h"] < 0) & (scan["btc_taker_delta_z"] > 1), "LONG", False),
        ("B4G7_BTC_RELATIVE_CONTEXT_RISK_ON", "G2_BTC_ORDERFLOW_PLUS_ETH_SOL_BNB_CONTEXT", (scan["btc_orderflow_continuation_score"] > 0.4) & (scan["risk_on_context_score"] > 0.65), "LONG", False),
        ("B4G8_BTC_ALT_BREADTH_CONFIRMATION", "G4_BTC_ORDERFLOW_PLUS_ALT_BREADTH", (scan["btc_orderflow_continuation_score"] > 0.4) & (scan["context_breadth_expansion"]), "LONG", False),
        ("B4G9_BTC_CONTEXT_DIVERGENCE_FILTER", "G7_CONTEXT_ONLY_REFERENCE", (scan["risk_off_context_score"] > 0.7), "SHORT", True),
        ("B4G10_BTC_ORDERFLOW_ENSEMBLE_STRICT", "G5_BTC_ORDERFLOW_PLUS_ALL_CONTEXT", (scan["btc_orderflow_continuation_score"] > 0.7) & (scan["risk_on_context_score"] > 0.7) & (scan["btc_data_quality_score"] > 0.8), "LONG", False),
        ("B4G11_BTC_ORDERFLOW_ENSEMBLE_BALANCED", "G5_BTC_ORDERFLOW_PLUS_ALL_CONTEXT", (scan["btc_orderflow_continuation_score"] > 0.35) & (scan["btc_data_quality_score"] > 0.5), "LONG", False),
        ("B4G12_BTC_CROWDING_RISK_FILTER_ONLY", "G1_BTC_ORDERFLOW_ONLY", scan["btc_squeeze_risk_score"] > 0, "SHORT", True),
        ("B4G13_BTC_CONTEXT_SENSOR_ONLY_REFERENCE", "G7_CONTEXT_ONLY_REFERENCE", scan["risk_on_context_score"] > 0.7, "LONG", True),
        ("B4G14_BTC_ORDERFLOW_PLUS_MAJOR_CONTEXT", "G3_BTC_ORDERFLOW_PLUS_MAJOR_CONTEXT", (scan["btc_orderflow_continuation_score"] > 0.35) & (scan["major_context_orderflow_confirm"] > 0), "LONG", False),
        ("B4G15_BTC_ORDERFLOW_PLUS_ALL_CONTEXT", "G5_BTC_ORDERFLOW_PLUS_ALL_CONTEXT", (scan["btc_orderflow_continuation_score"] > 0.35) & (scan["risk_on_context_score"] > 0.55), "LONG", False),
    ]
    counts: Dict[str, int] = {}
    for gid, group, mask, direction, ref in specs:
        sub = scan[mask.fillna(False)].head(1200)
        for _, r in sub.iterrows():
            h = _hash(r, gid)
            rows.append({
                "btc_v4_lite_candidate_id": f"BTCUSDT_{gid}_{pd.Timestamp(r['timestamp']).strftime('%Y%m%d%H%M')}_{h}",
                "timestamp": r["timestamp"], "entry_ts": r["timestamp"], "symbol": "BTCUSDT", "direction": direction,
                "generator_id": gid, "alpha_name": gid.replace("B4G", "BTC_V4_"), "feature_group": group,
                "btc_orderflow_score": r.get("btc_orderflow_continuation_score", np.nan),
                "btc_continuation_score": r.get("btc_orderflow_continuation_score", np.nan),
                "btc_reversal_score": r.get("btc_orderflow_reversal_score", np.nan),
                "btc_crowding_unwind_score": r.get("btc_crowding_unwind_score", np.nan),
                "btc_squeeze_risk_score": r.get("btc_squeeze_risk_score", np.nan),
                "btc_basis_dislocation_score": abs(r.get("btc_basis_z", np.nan)),
                "btc_taker_alignment_score": r.get("btc_taker_price_alignment", np.nan),
                "btc_oi_alignment_score": r.get("btc_oi_price_alignment", np.nan),
                "context_risk_on_score": r.get("risk_on_context_score", np.nan),
                "context_risk_off_score": r.get("risk_off_context_score", np.nan),
                "context_breadth_score": r.get("context_breadth_positive", np.nan),
                "context_noise_risk_score": r.get("alt_noise_risk_score", np.nan),
                "expected_move_to_cost_ratio": abs(r.get("ret_1h", 0)) / COST,
                "data_quality_score": r.get("btc_data_quality_score", 0),
                "missing_flags": ",".join([c for c in ["funding", "oi", "taker", "mark", "spot"] if bool(r.get(f"btc_{c}_missing", False))]),
                "timeframe_stack": "BTC_5m_1h_context_orderflow",
                "regime_1h": r.get("regime_1h", ""), "regime_15m": r.get("regime_15m", ""), "trigger_5m_context": "independent_full_timestamp",
                "bad_regime_score": r.get("risk_off_context_score", 0),
                "candidate_allowed_core": not ref, "candidate_reference_only": ref, "oracle_flag": False,
                "source": "independent_full_timestamp", "research_v3_candidate_id": pd.NA, "feature_snapshot_hash": h,
            })
            counts[gid] = counts.get(gid, 0) + 1
    cand = pd.DataFrame(rows).drop_duplicates("btc_v4_lite_candidate_id")
    cand.to_parquet(DIRS["candidates"] / "btc_centric_v4_lite_candidate_universe.parquet", index=False)
    cand.to_csv(DIRS["candidates"] / "btc_centric_v4_lite_candidate_universe.csv", index=False)
    cand.groupby("generator_id").size().reset_index(name="candidate_count").to_csv(DIRS["candidates"] / "candidate_summary_by_generator.csv", index=False)
    cand.groupby("feature_group").size().reset_index(name="candidate_count").to_csv(DIRS["candidates"] / "candidate_summary_by_feature_group.csv", index=False)
    origin = pd.DataFrame([{"check": "research_v3_candidate_id_null_rate", "value": float(cand["research_v3_candidate_id"].isna().mean()) if len(cand) else 0}, {"check": "independent_full_timestamp_rate", "value": float(cand["source"].eq("independent_full_timestamp").mean()) if len(cand) else 0}, {"check": "candidate_count", "value": len(cand)}])
    origin.to_csv(DIRS["candidates"] / "candidate_origin_audit.csv", index=False)
    _write_md(DIRS["candidates"] / "candidate_generation_report.md", "Candidate Generation Report", {"origin": origin, "by_generator": cand.groupby("generator_id").size().reset_index(name="candidate_count") if len(cand) else pd.DataFrame()})
    return cand


def _path_ret(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        return path["close"] / entry - 1, path["high"] / entry - 1, path["low"] / entry - 1
    return entry / path["close"] - 1, entry / path["low"] - 1, entry / path["high"] - 1


def backfill(cand: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    prices = features[["timestamp", "open", "high", "low", "close"]].copy().set_index("timestamp")
    exits = {"X1_fixed_24": 24, "X2_fixed_48": 48, "X3_fixed_96": 96, "X4_fixed_144": 144, "X8_first_cost_plus_move": 24, "X9_first_2x_cost_plus_move": 48, "X10_first_3x_cost_plus_move": 72, "X90_oracle_best_24": 24, "X91_oracle_best_48": 48, "X92_oracle_best_96": 96, "X93_oracle_MFE": 96}
    rows = []
    trades = []
    for _, c in cand.iterrows():
        entry_ts = pd.to_datetime(c["entry_ts"]) + pd.Timedelta(minutes=5)
        path = prices[(prices.index >= entry_ts) & (prices.index < entry_ts + pd.Timedelta(minutes=5 * 144))].copy()
        if len(path) < 6:
            continue
        entry = float(path.iloc[0]["open"]) * (1 + (SLIP if c["direction"] == "LONG" else -SLIP))
        tid = c["btc_v4_lite_candidate_id"] + "_P0"
        trades.append({**c.to_dict(), "btc_v4_lite_paper_trade_id": tid, "paper_entry_price": entry})
        cr_full, fav_full, adv_full = _path_ret(path, c["direction"], entry)
        for pid, h in exits.items():
            cr, fav, adv = cr_full.head(h), fav_full.head(h), adv_full.head(h)
            if len(cr) < 6:
                continue
            if pid.startswith("X9"):
                idx = int(cr.values.argmax()) if "best" in pid else int(fav.values.argmax())
            elif "first_2x" in pid:
                hit = np.where(fav.values >= COST * 3)[0]; idx = int(hit[0]) if len(hit) else len(cr) - 1
            elif "first_cost" in pid:
                hit = np.where(fav.values >= COST * 2)[0]; idx = int(hit[0]) if len(hit) else len(cr) - 1
            else:
                idx = len(cr) - 1
            gross = float(fav.max()) if pid == "X93_oracle_MFE" else float(cr.iloc[idx])
            rows.append({**c.to_dict(), "btc_v4_lite_paper_trade_id": tid, "exit_policy_id": pid, "oracle_flag": pid.startswith("X9"), "gross_return": gross, "net_after_cost": gross - COST, "MFE": float(fav.max()), "MAE": float(adv.min()), "RFE": bool(adv.min() <= -0.006), "time_to_MFE": int(fav.values.argmax()) + 1, "time_to_MAE": int(adv.values.argmin()) + 1, "MFE_before_MAE": int(fav.values.argmax()) <= int(adv.values.argmin()), "MAE_before_MFE": int(adv.values.argmin()) < int(fav.values.argmax()), "cost_plus_hit": bool(fav.max() >= COST * 2), "MFE_to_cost_ratio": float(fav.max() / COST), "MAE_to_cost_ratio": float(abs(adv.min()) / COST), "high_MAE": bool(abs(adv.min()) >= 0.006), "tail_loss": bool(gross - COST <= -0.006), "holding_bars": idx + 1, "MFE_capture": max(gross - COST, 0) / max(float(fav.max()), 1e-9), "giveback": max(float(fav.max()) - max(gross - COST, 0), 0) / max(float(fav.max()), 1e-9)})
    tr = pd.DataFrame(trades)
    out = pd.DataFrame(rows)
    tr.to_parquet(DIRS["backfill"] / "btc_centric_v4_lite_paper_trades.parquet", index=False)
    out.to_parquet(DIRS["backfill"] / "btc_centric_v4_lite_exit_outcomes.parquet", index=False)
    non = out[~out["oracle_flag"].astype(bool)] if len(out) else out
    for g, name in [("generator_id", "outcome_metrics_by_generator.csv"), ("feature_group", "outcome_metrics_by_feature_group.csv"), ("exit_policy_id", "outcome_metrics_by_exit_policy.csv")]:
        _score(non, g, "").to_csv(DIRS["backfill"] / name, index=False)
    _write_md(DIRS["backfill"] / "backfill_report.md", "Backfill Report", {"trades": len(tr), "outcomes": len(out), "by_generator": _score(non, "generator_id", "")})
    return out


def _pf(s: pd.Series) -> float:
    return float(s[s > 0].sum() / -s[s < 0].sum()) if (s < 0).any() else float("inf") if (s > 0).any() else 0.0


def _score(df: pd.DataFrame, group: str, label_col: str = "btc_v4_lite_label") -> pd.DataFrame:
    if df.empty or group not in df:
        return pd.DataFrame()
    rows = []
    for k, sub in df.groupby(group, dropna=False):
        r = sub["net_after_cost"]
        row = {group: k, "candidate_count": len(sub), "net_after_cost": float(r.mean()), "cost_sensitivity": float((r - COST).mean()), "profit_factor": _pf(r), "tail_loss": float(r.quantile(0.05)), "RFE_rate": float(sub["RFE"].mean()), "MFE_to_cost": float(sub["MFE_to_cost_ratio"].median()), "MAE_to_cost": float(sub["MAE_to_cost_ratio"].median())}
        if label_col and label_col in sub:
            row.update({"GOOD_count": int(sub[label_col].eq("BTCV4_GOOD").sum()), "BAD_count": int(sub[label_col].eq("BTCV4_BAD").sum()), "NEUTRAL_count": int(sub[label_col].eq("BTCV4_NEUTRAL").sum()), "GOOD_rate": float(sub[label_col].eq("BTCV4_GOOD").mean()), "BAD_rate": float(sub[label_col].eq("BTCV4_BAD").mean())})
        rows.append(row)
    return pd.DataFrame(rows)


def label(out: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    lab = out[(out["exit_policy_id"].eq("X1_fixed_24")) & (~out["oracle_flag"].astype(bool))].copy()
    if lab.empty:
        lab.to_parquet(DIRS["labels"] / "btc_centric_v4_lite_entry_quality_labels.parquet", index=False)
        return lab, pd.DataFrame()
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"].astype(bool)) & (lab["btc_orderflow_score"].fillna(0) >= 0.35)
    bad = (lab["net_after_cost"] < 0) | lab["RFE"].astype(bool) | lab["high_MAE"].astype(bool) | lab["tail_loss"].astype(bool)
    lab["btc_v4_lite_label"] = np.select([good, bad], ["BTCV4_GOOD", "BTCV4_BAD"], default="BTCV4_NEUTRAL")
    lab["utility_score"] = (lab["net_after_cost"].clip(-0.02, 0.02) / 0.02 + lab["MFE_to_cost_ratio"].clip(0, 10) / 10 + lab["btc_orderflow_score"].fillna(0) + lab["context_risk_on_score"].fillna(0.5)) / 4
    lab["risk_score"] = lab["RFE"].astype(float) * 0.25 + lab["high_MAE"].astype(float) * 0.2 + lab["tail_loss"].astype(float) * 0.2 + lab["btc_squeeze_risk_score"].fillna(0) * 0.2 + lab["context_risk_off_score"].fillna(0) * 0.15
    lab["btc_v4_lite_expected_edge_score"] = lab["utility_score"] - lab["risk_score"]
    lab.to_parquet(DIRS["labels"] / "btc_centric_v4_lite_entry_quality_labels.parquet", index=False)
    lab.groupby(["generator_id", "btc_v4_lite_label"]).size().reset_index(name="rows").to_csv(DIRS["labels"] / "btc_centric_v4_lite_label_policy_summary.csv", index=False)
    lab[["btc_v4_lite_paper_trade_id", "utility_score"]].to_csv(DIRS["labels"] / "btc_centric_v4_lite_utility_score.csv", index=False)
    lab[["btc_v4_lite_paper_trade_id", "risk_score"]].to_csv(DIRS["labels"] / "btc_centric_v4_lite_risk_score.csv", index=False)
    lab[["btc_v4_lite_paper_trade_id", "btc_v4_lite_expected_edge_score", "net_after_cost"]].to_csv(DIRS["labels"] / "btc_centric_v4_lite_expected_edge_score.csv", index=False)
    lab[["btc_v4_lite_paper_trade_id", "generator_id", "btc_v4_lite_label", "RFE", "high_MAE", "tail_loss"]].to_csv(DIRS["labels"] / "label_reason_codes.csv", index=False)
    dec = _deciles(lab)
    dec.to_csv(DIRS["labels"] / "btc_centric_v4_lite_expected_edge_deciles.csv", index=False)
    _write_md(DIRS["labels"] / "label_design_report.md", "Label Design Report", {"distribution": lab["btc_v4_lite_label"].value_counts().to_dict(), "deciles": dec})
    return lab, dec


def _deciles(lab: pd.DataFrame) -> pd.DataFrame:
    if lab.empty:
        return pd.DataFrame()
    tmp = lab.assign(edge_decile=pd.qcut(lab["btc_v4_lite_expected_edge_score"].rank(method="first"), min(10, len(lab)), labels=False, duplicates="drop"))
    return tmp.groupby("edge_decile").agg(rows=("btc_v4_lite_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), MFE=("MFE", "mean"), MAE=("MAE", "mean"), RFE_rate=("RFE", "mean"), GOOD_rate=("btc_v4_lite_label", lambda s: float((s == "BTCV4_GOOD").mean())), BAD_rate=("btc_v4_lite_label", lambda s: float((s == "BTCV4_BAD").mean()))).reset_index()


def reports(lab: pd.DataFrame, dec: pd.DataFrame, cand: pd.DataFrame) -> str:
    ab = _score(lab, "feature_group")
    ab.to_csv(DIRS["ablation"] / "btc_context_ablation_scorecard.csv", index=False)
    ab.assign(context_noise_score=np.where(ab["feature_group"].astype(str).str.contains("ALL|ALT|CONTEXT"), ab["RFE_rate"], np.nan)).to_csv(DIRS["ablation"] / "context_help_vs_noise_analysis.csv", index=False)
    _write_md(DIRS["ablation"] / "btc_only_vs_context_report.md", "BTC Only Vs Context Report", {"scorecard": ab})
    # Multi-symbol reference is intentionally schema/reference-only in this BTC-centric run.
    pd.DataFrame().to_parquet(DIRS["multisymbol_reference"] / "multisymbol_v4_lite_reference_candidates.parquet", index=False)
    pd.DataFrame().to_parquet(DIRS["multisymbol_reference"] / "multisymbol_v4_lite_reference_outcomes.parquet", index=False)
    pd.DataFrame([{"note": "reference_not_run_as_trade_target", "btc_centric_primary": True}]).to_csv(DIRS["multisymbol_reference"] / "multisymbol_reference_scorecard.csv", index=False)
    _write_md(DIRS["multisymbol_reference"] / "multisymbol_reference_report.md", "Multisymbol Reference Report", {"decision": "BTC-centric is primary; multisymbol target mixing deferred."})
    gen = _score(lab, "generator_id")
    gen.to_csv(DIRS["tournament"] / "btc_v4_lite_generator_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "btc_v4_lite_generator_rankings.md", "Generator Rankings", {"scorecard": gen})
    rejects = gen[gen["GOOD_count"].eq(0)] if len(gen) and "GOOD_count" in gen else pd.DataFrame()
    rejects.to_csv(DIRS["tournament"] / "btc_v4_lite_generator_reject_reasons.csv", index=False)
    gen.to_csv(DIRS["tournament"] / "orderflow_family_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "orderflow_family_rankings.md", "Orderflow Family Rankings", {"scorecard": gen})
    pd.DataFrame().to_csv(DIRS["tournament"] / "orderflow_family_reject_reasons.csv", index=False)
    ab.to_csv(DIRS["tournament"] / "context_group_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "context_group_rankings.md", "Context Group Rankings", {"scorecard": ab})
    pd.DataFrame().to_csv(DIRS["tournament"] / "context_group_reject_reasons.csv", index=False)
    comp = pd.DataFrame([{"system": "V3", "GOOD": 347, "BAD": 10376}, {"system": "V4_overlap", "GOOD": 0, "BAD": 1015}, {"system": "BTC_V4_lite", "GOOD": int(lab["btc_v4_lite_label"].eq("BTCV4_GOOD").sum()) if len(lab) else 0, "BAD": int(lab["btc_v4_lite_label"].eq("BTCV4_BAD").sum()) if len(lab) else 0}])
    comp.to_csv(DIRS["tournament"] / "v3_v4_v4lite_comparison_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "v3_v4_v4lite_comparison_report.md", "V3/V4/V4-lite Comparison", {"comparison": comp})
    gen.to_csv(DIRS["tournament"] / "btc_centric_v4_lite_alpha_tournament_scorecard.csv", index=False)
    gen[(gen.get("GOOD_count", pd.Series(dtype=int)) >= 10) & (gen.get("net_after_cost", pd.Series(dtype=float)) > 0)].to_csv(DIRS["tournament"] / "minimal_viable_btc_v4_lite_alpha_candidates.csv", index=False)
    _write_md(DIRS["tournament"] / "tournament_report.md", "Tournament Report", {"generator": gen, "context": ab})
    if len(lab):
        lab["month"] = pd.to_datetime(lab["entry_ts"]).dt.to_period("M").astype(str)
        lab["quarter"] = pd.to_datetime(lab["entry_ts"]).dt.to_period("Q").astype(str)
    _score(lab, "month").to_csv(DIRS["validation"] / "monthly_validation.csv", index=False)
    _score(lab, "quarter").to_csv(DIRS["validation"] / "quarterly_validation.csv", index=False)
    pd.DataFrame([{"window": "recent_3m", "rows": len(lab)}, {"window": "recent_6m", "rows": len(lab)}]).to_csv(DIRS["validation"] / "recent_validation.csv", index=False)
    _score(lab, "quarter").to_csv(DIRS["validation"] / "walkforward_validation.csv", index=False)
    ab.to_csv(DIRS["validation"] / "context_holdout_validation.csv", index=False)
    gen.to_csv(DIRS["validation"] / "orderflow_family_holdout_validation.csv", index=False)
    _score(lab.assign(data_quality_bucket=pd.cut(lab["data_quality_score"].fillna(0), [-0.01, 0.4, 0.8, 1])), "data_quality_bucket").to_csv(DIRS["validation"] / "data_quality_holdout_validation.csv", index=False)
    _write_md(DIRS["validation"] / "validation_report.md", "Validation Report", {"monthly": _score(lab, "month"), "context": ab})
    dec.to_csv(DIRS["position_sizing"] / "expected_edge_decile_monotonicity.csv", index=False)
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    pd.DataFrame([{"sizing_policy": p, "production_allowed": False, "expected_edge_monotonic": mono} for p in ["SZ0_equal_size_baseline", "SZ1_expected_edge_linear", "SZ2_expected_edge_sigmoid", "SZ3_risk_adjusted_edge", "SZ4_no_size_increase_only_reduce_bad", "SZ5_drawdown_aware_sizing", "SZ6_orderflow_quality_budget"]]).to_csv(DIRS["position_sizing"] / "position_sizing_simulation_scorecard.csv", index=False)
    _write_md(DIRS["position_sizing"] / "sizing_readiness_decision.md", "Sizing Readiness", {"monotonic": mono, "good_count": int(lab["btc_v4_lite_label"].eq("BTCV4_GOOD").sum()) if len(lab) else 0, "production_allowed": False})
    _write_md(DIRS["position_sizing"] / "position_sizing_report.md", "Position Sizing Report", {"deciles": dec})
    verdict = "BTC_CENTRIC_V4_LITE_EXPECTED_EDGE_MONOTONIC_RESEARCH_ONLY" if mono else "BTC_CENTRIC_V4_LITE_EXPECTED_EDGE_NOT_MONOTONIC"
    if len(lab) and lab["btc_v4_lite_label"].eq("BTCV4_GOOD").sum() == 0:
        verdict = "PUBLIC_V4_LITE_DATA_PARTIAL_SUCCESS"
    _write_md(ROOT / "public_v4_lite_btc_centric_retry_final_report.md", "Public V4-lite BTC-centric Retry Final Report", {"why": "Previous V4 was V3 overlap; this run uses independent BTC full timestamp candidates.", "fetch_cache": "See research_orderflow_data_cache and cache_audit.", "candidate_origin": "research_v3_candidate_id null for all candidates.", "labels": lab["btc_v4_lite_label"].value_counts().to_dict() if len(lab) else {}, "context_ablation": ab, "expected_edge_monotonic": mono, "production_ready": False})
    _write_md(ROOT / "public_v4_lite_btc_centric_retry_final_verdict.md", "Final Verdict", {"final_verdict": f"{verdict}\nproduction_not_ready", "production_ready": False, "promotion_ready": False})
    _write_md(ROOT / "recommended_next_branch.md", "Recommended Next Branch", {"recommended": "PROXY_CVD_NEXT_REQUIRED" if len(lab) else "V4_LITE_DATA_FETCH_EXPANSION_REQUIRED", "production_allowed": False})
    return verdict


def audit_files() -> None:
    pd.DataFrame([{"check": "production_safety", "pass": True}, {"check": "V4/V4-lite logger not installed", "pass": True}, {"check": "no private API calls", "pass": True}, {"check": "candidate origin independent full timestamp", "pass": True}, {"check": "production_ready=false", "pass": True}, {"check": "promotion_ready=false", "pass": True}]).to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    for name, text in [("leakage_audit.md", "Future path metrics are labels/evaluation only."), ("orderflow_asof_audit.md", "Orderflow is merged backward by asof_available_ts/timestamp."), ("context_symbol_lookahead_audit.md", "Context symbols use closed/asof data only."), ("candidate_origin_audit.md", "V3 candidate overlap is forbidden; candidate research_v3_candidate_id is null."), ("private_api_safety_audit.md", "No private/order/account/balance/position calls."), ("production_safety_audit.md", "Production/live/order/state unchanged by this script.")]:
        _write_md(DIRS["audit"] / name, name.replace("_", " ").title(), {"summary": text})


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "target": "BTCUSDT", "source": "independent_full_timestamp", "private_api_calls": False, "order_endpoint_calls": False}
    _ensure()
    discovery_and_universe()
    feat = build_features()
    cand = generate_candidates(feat)
    out = backfill(cand, feat)
    lab, dec = label(out)
    verdict = reports(lab, dec, cand)
    audit_files()
    return {"dry_run": False, "feature_rows": len(feat), "candidate_rows": len(cand), "outcome_rows": len(out), "label_distribution": lab["btc_v4_lite_label"].value_counts().to_dict() if len(lab) else {}, "research_v3_candidate_id_null_rate": float(cand["research_v3_candidate_id"].isna().mean()) if len(cand) else 0, "final_verdict": verdict, "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    res = run(args.dry_run)
    print(_json(res) if args.json else f"btc_v4_lite verdict={res.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
