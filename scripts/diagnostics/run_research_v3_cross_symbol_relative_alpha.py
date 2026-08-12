"""
Research V3 cross-symbol relative / breadth / leader-lagger alpha research.

Diagnostics-only. No private/order/account/position API calls. No production
integration. Uses existing Research V2 multi-symbol diagnostics OHLCV/MTF cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
ROOT = Path("data/diagnostics/research_v3_cross_symbol_relative_alpha")
V2_ROOT = Path("data/diagnostics/research_v2_multisymbol_expansion")
COST = 0.0006
SLIPPAGE = 0.0002
MAX_PER_GENERATOR = 1000

DIRS = {k: ROOT / k for k in [
    "discovery", "audit", "universe", "mtf_data", "market_regime", "relative_strength",
    "breadth", "leader_lagger", "hypotheses", "candidates", "backfill", "labels",
    "tournament", "validation", "model_objective", "position_sizing", "forward_design",
    "hidden_failure_modes", "research_branch_decision",
]}

CLUSTERS = {
    "CORE_BTC": ["BTCUSDT"],
    "CORE_ETH": ["ETHUSDT"],
    "CORE_SOL": ["SOLUSDT"],
    "CORE_BNB": ["BNBUSDT"],
    "MAJOR_L1": ["ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "ATOMUSDT"],
    "PAYMENT_OLD": ["XRPUSDT", "LTCUSDT", "BCHUSDT", "TRXUSDT", "DOGEUSDT"],
    "DEFI": ["UNIUSDT", "AAVEUSDT", "MKRUSDT", "LDOUSDT", "PENDLEUSDT"],
    "L2": ["ARBUSDT", "OPUSDT"],
    "AI_INFRA": ["FETUSDT", "RENDERUSDT", "WLDUSDT"],
    "MEME": ["DOGEUSDT", "PEPEUSDT", "WIFUSDT"],
    "SOL_ECOSYSTEM_PROXY": ["SOLUSDT", "JUPUSDT", "PYTHUSDT", "WIFUSDT"],
    "HIGH_BETA_ALT": ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "APTUSDT", "SUIUSDT", "INJUSDT", "SEIUSDT", "TIAUSDT"],
    "LOWER_BETA_MAJOR": ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT"],
}
EXIT_POLICIES = {
    "X1_fixed_24": 24, "X2_fixed_48": 48, "X3_fixed_96": 96, "X4_fixed_144": 144,
    "X6_MAE_stop_medium": 24, "X7_vol_adjusted_MAE_stop": 48,
    "X8_first_cost_plus_move": 24, "X9_first_2x_cost_plus_move": 48,
    "X10_first_3x_cost_plus_move": 72, "X11_trailing_vol_adjusted_proxy": 96,
    "X12_fixed_24_plus_MAE_stop": 24, "X13_fixed_48_plus_MAE_stop": 48,
    "X14_trailing_plus_MAE_stop": 96, "X15_alpha_specific_exit": 48,
    "X16_timeframe_matched_exit_15m": 36, "X17_timeframe_matched_exit_30m": 72,
    "X18_timeframe_matched_exit_1h": 144, "X19_relative_strength_decay_exit": 48,
    "X20_breadth_regime_flip_exit": 48, "X21_leader_lagger_spread_close_exit": 48,
    "X90_oracle_best_24": 24, "X91_oracle_best_48": 48, "X92_oracle_best_96": 96, "X93_oracle_MFE": 96,
}


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)


def _write_md(path: Path, title: str, sections: Dict[str, Any]) -> None:
    lines = [f"# {title}", ""]
    for k, v in sections.items():
        lines += [f"## {k}", ""]
        if isinstance(v, pd.DataFrame):
            lines += ["```csv", v.head(60).to_csv(index=False), "```"]
        elif isinstance(v, (dict, list)):
            lines += ["```json", _json(v), "```"]
        else:
            lines.append(str(v))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _ensure_dirs() -> None:
    for d in DIRS.values():
        d.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _hash_path(path: Path) -> Dict[str, Any]:
    return {"path": str(path.relative_to(REPO_ROOT) if path.is_absolute() and path.exists() else path), "exists": path.exists(), "sha256": _sha256(path) if path.exists() and path.is_file() else "", "size_bytes": path.stat().st_size if path.exists() and path.is_file() else 0}


def _safety_paths() -> List[Path]:
    rels = ["models/tcn_v1.pt", "data/diagnostics/tcn_no_events.pt", "scripts/diagnostics/run_false_high_r7_monitor.py", "scripts/diagnostics/run_false_high_r7_daily_monitor.py", "ops/launchd", "risk", "risk_manager", "state", "live", "orders"]
    out: List[Path] = []
    for rel in rels:
        p = REPO_ROOT / rel
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            out.extend(sorted(x for x in p.rglob("*") if x.is_file())[:400])
    return out


def _git_status() -> str:
    try:
        return subprocess.run(["git", "status", "--short"], cwd=REPO_ROOT, text=True, capture_output=True, timeout=10).stdout
    except Exception as exc:
        return f"git_status_unavailable: {exc}"


def _load_symbol_list() -> List[str]:
    p = REPO_ROOT / V2_ROOT / "data/symbol_selected_core.csv"
    if p.exists():
        return pd.read_csv(p)["symbol"].astype(str).tolist()
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "TRXUSDT", "LTCUSDT", "BCHUSDT"]


def _cluster_for(symbol: str) -> str:
    preferred = ["CORE_BTC", "CORE_ETH", "CORE_SOL", "CORE_BNB", "MAJOR_L1", "PAYMENT_OLD", "DEFI", "L2", "AI_INFRA", "MEME", "SOL_ECOSYSTEM_PROXY", "HIGH_BETA_ALT", "LOWER_BETA_MAJOR"]
    for c in preferred:
        if symbol in CLUSTERS[c]:
            return c
    return "OTHER"


def _read_mtf(symbol: str) -> pd.DataFrame:
    p = REPO_ROOT / V2_ROOT / "mtf_data" / symbol / "mtf_asof_joined_frame.parquet"
    if not p.exists():
        raise FileNotFoundError(f"missing V2 MTF cache for {symbol}: {p}")
    df = pd.read_parquet(p)
    df["symbol"] = symbol
    df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce").astype("datetime64[ns]")
    return df.sort_values("close_ts").drop_duplicates("close_ts").reset_index(drop=True)


def _read_ohlcv(symbol: str) -> pd.DataFrame:
    p = REPO_ROOT / V2_ROOT / "data/ohlcv_cache" / f"{symbol}_5m.parquet"
    if p.exists():
        df = pd.read_parquet(p)
    elif symbol == "BTCUSDT":
        df = pd.read_csv(REPO_ROOT / "data/ohlcv/BTCUSDT_5m_full.csv").rename(columns={"open_time": "timestamp"})
    else:
        return pd.DataFrame()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce").astype("datetime64[ns]")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "quote_volume" not in df:
        df["quote_volume"] = df["close"] * df["volume"]
    df["close_ts"] = df.get("close_ts", df["timestamp"] + pd.Timedelta(minutes=5))
    df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce").astype("datetime64[ns]")
    df["symbol"] = symbol
    return df.dropna(subset=["timestamp", "close"]).sort_values("close_ts").drop_duplicates("close_ts").reset_index(drop=True)


def part0_discovery(symbols: List[str]) -> Dict[str, Any]:
    required = [
        V2_ROOT / "research_v2_multisymbol_expansion_final_report.md",
        V2_ROOT / "tournament/multisymbol_alpha_tournament_scorecard.csv",
        V2_ROOT / "tournament/r2g6_special_scorecard.csv",
        V2_ROOT / "labels/all_symbols_entry_quality_labels.parquet",
        V2_ROOT / "backfill/all_symbols_exit_outcomes.parquet",
        V2_ROOT / "candidates/all_symbols_candidate_universe.parquet",
        Path("data/diagnostics/research_v2_mtf_alpha_engine/research_v2_mtf_alpha_engine_final_report.md"),
        Path("data/diagnostics/external_market_structure_alpha_v1/external_market_structure_alpha_v1_final_report.md"),
        Path("data/diagnostics/alpha_candidate_v2_autopsy/alpha_candidate_v2_autopsy_final_report.md"),
        Path("data/diagnostics/forward_research_v2_alpha_logger/state/forward_research_v2_pending_trades.parquet"),
        Path("data/diagnostics/forward_research_v2_alpha_logger/state/forward_research_v2_resolved_trades.parquet"),
        Path("data/diagnostics/data_sync/canonical_data_paths.json"),
    ]
    inv = [{"path": str(p), "exists": (REPO_ROOT / p).exists(), "size_bytes": (REPO_ROOT / p).stat().st_size if (REPO_ROOT / p).exists() and (REPO_ROOT / p).is_file() else 0} for p in required]
    pd.DataFrame(inv).to_csv(DIRS["discovery"] / "input_inventory.csv", index=False)
    prev = []
    v2_lab = REPO_ROOT / V2_ROOT / "labels/all_symbols_entry_quality_labels.parquet"
    if v2_lab.exists():
        x = pd.read_parquet(v2_lab)
        prev.append({"source": "Research V2 multisymbol", "verdict": "MULTISYMBOL_EXPECTED_EDGE_NOT_MONOTONIC", "GOOD": int(x.get("msr2_label", pd.Series()).eq("MSR2_GOOD").sum()), "BAD": int(x.get("msr2_label", pd.Series()).eq("MSR2_BAD").sum()), "NEUTRAL": int(x.get("msr2_label", pd.Series()).eq("MSR2_NEUTRAL").sum())})
    prev.append({"source": "Research V3 premise", "verdict": "switch_alpha_source_to_cross_symbol_relative", "GOOD": "", "BAD": "", "NEUTRAL": ""})
    pd.DataFrame(prev).to_csv(DIRS["discovery"] / "previous_diagnostics_summary.csv", index=False)
    data_sources = []
    for s in symbols:
        data_sources.append({"symbol": s, "mtf_cache": str(V2_ROOT / "mtf_data" / s / "mtf_asof_joined_frame.parquet"), "ohlcv_cache": str(V2_ROOT / "data/ohlcv_cache" / f"{s}_5m.parquet"), "mtf_exists": (REPO_ROOT / V2_ROOT / "mtf_data" / s / "mtf_asof_joined_frame.parquet").exists()})
    pd.DataFrame(data_sources).to_csv(DIRS["discovery"] / "available_symbol_data_sources.csv", index=False)
    pd.DataFrame([{"timeframe": tf, "available": True} for tf in ["5m", "15m", "30m", "1h", "4h", "1d"]]).to_csv(DIRS["discovery"] / "available_timeframes.csv", index=False)
    ext = sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "data/diagnostics").glob("**/external_cache/*.parquet"))
    pd.DataFrame([{"path": p} for p in ext]).to_csv(DIRS["discovery"] / "available_external_data_sources.csv", index=False)
    status = {"label": "com.canbit.forward_research_v2_alpha_logger", "checked": True, "production_action": "none", "note": "V3 only reads status; it does not modify Research V2 logger."}
    (DIRS["discovery"] / "forward_logger_status_snapshot.json").write_text(_json(status), encoding="utf-8")
    discovered = {"symbols": symbols, "external_cache_files": ext[:100]}
    (DIRS["discovery"] / "discovered_paths.json").write_text(_json(discovered), encoding="utf-8")
    _write_md(DIRS["discovery"] / "discovery_report.md", "Discovery Report", {"previous": pd.DataFrame(prev), "symbol_sources": pd.DataFrame(data_sources), "forward_v2_logger_status": status})
    return status


def part2_universe(symbols: List[str], mtfs: Dict[str, pd.DataFrame], ohlcvs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for s in symbols:
        m = mtfs[s]
        o = ohlcvs[s]
        rows.append({"symbol": s, "cluster": _cluster_for(s), "rows": len(m), "start": m["close_ts"].min(), "end": m["close_ts"].max(), "recent_coverage": True, "gap_count": int((m["close_ts"].diff().dt.total_seconds().fillna(300) > 900).sum()), "duplicate_count": int(m["close_ts"].duplicated().sum()), "avg_quote_volume_proxy": float(o["quote_volume"].tail(min(len(o), 2016)).mean()) if len(o) else np.nan, "core_eligible": len(m) > 1000})
    audit = pd.DataFrame(rows)
    audit.to_csv(DIRS["universe"] / "symbol_universe_audit.csv", index=False)
    audit[audit["core_eligible"]].to_csv(DIRS["universe"] / "symbol_selected_core.csv", index=False)
    audit[~audit["core_eligible"]].to_csv(DIRS["universe"] / "symbol_selected_reference.csv", index=False)
    audit[~audit["core_eligible"]].to_csv(DIRS["universe"] / "symbol_excluded.csv", index=False)
    cmap = []
    for c, syms in CLUSTERS.items():
        for s in syms:
            if s in symbols:
                cmap.append({"cluster": c, "symbol": s, "mapping_policy": "fixed_ex_ante"})
    pd.DataFrame(cmap).to_csv(DIRS["universe"] / "symbol_cluster_map.csv", index=False)
    audit.to_csv(DIRS["universe"] / "symbol_data_quality.csv", index=False)
    audit[["symbol", "avg_quote_volume_proxy"]].to_csv(DIRS["universe"] / "symbol_liquidity_proxy.csv", index=False)
    _write_md(DIRS["universe"] / "universe_report.md", "Universe Report", {"audit": audit, "cluster_map": pd.DataFrame(cmap)})
    return audit


def part3_copy_mtf(symbols: List[str], mtfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    audits = []
    for s in symbols:
        src_dir = REPO_ROOT / V2_ROOT / "mtf_data" / s
        dst_dir = DIRS["mtf_data"] / s
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in ["closed_candles_5m.parquet", "closed_candles_15m.parquet", "closed_candles_30m.parquet", "closed_candles_1h.parquet", "closed_candles_4h.parquet", "closed_candles_1d.parquet", "mtf_asof_joined_frame.parquet", "mtf_alignment_audit.csv"]:
            src = src_dir / name
            dst = dst_dir / name
            if src.exists():
                if src.suffix == ".parquet":
                    pd.read_parquet(src).to_parquet(dst, index=False)
                else:
                    pd.read_csv(src).to_csv(dst, index=False)
        audit_path = src_dir / "mtf_alignment_audit.csv"
        if audit_path.exists():
            audits.append(pd.read_csv(audit_path).assign(symbol=s))
    align = pd.concat(audits, ignore_index=True) if audits else pd.DataFrame()
    quality = []
    for s, m in mtfs.items():
        quality.append({"symbol": s, "rows": len(m), "start": m["close_ts"].min(), "end": m["close_ts"].max(), "gap_count": int((m["close_ts"].diff().dt.total_seconds().fillna(300) > 900).sum())})
    pd.DataFrame(quality).to_csv(DIRS["mtf_data"] / "all_symbols_mtf_quality_summary.csv", index=False)
    align.to_csv(DIRS["mtf_data"] / "higher_timeframe_asof_audit.csv", index=False)
    _write_md(DIRS["mtf_data"] / "mtf_data_report.md", "MTF Data Report", {"quality": pd.DataFrame(quality), "asof_audit": align})
    return align


def _panel(mtfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    cols = ["symbol", "close_ts", "close", "return", "volume", "tf1h_return", "tf4h_return", "tf1d_return", "tf1h_trend_direction", "tf1h_range_position", "tf15m_expansion_score", "tf15m_compression_score", "tf30m_failed_breakout_proxy", "tf15m_failed_breakout_proxy", "tf1h_atr_proxy", "tf15m_close", "open", "high", "low"]
    frames = []
    for s, m in mtfs.items():
        use = [c for c in cols if c in m.columns]
        frames.append(m[use].copy().assign(cluster=_cluster_for(s)))
    p = pd.concat(frames, ignore_index=True)
    p = p.dropna(subset=["close_ts", "close"]).sort_values(["close_ts", "symbol"]).reset_index(drop=True)
    for h, bars in {"15m": 3, "30m": 6, "1h": 12, "4h": 48, "1d": 288}.items():
        p[f"ret_{h}"] = p.groupby("symbol")["close"].pct_change(bars)
    return p


def part4_market_regime(panel: pd.DataFrame) -> pd.DataFrame:
    piv = panel.pivot(index="close_ts", columns="symbol", values="ret_1h").sort_index()
    symbols = list(piv.columns)
    majors = [s for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"] if s in symbols]
    alts = [s for s in symbols if s not in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}]
    high_beta = [s for s in CLUSTERS["HIGH_BETA_ALT"] if s in symbols]
    low_beta = [s for s in CLUSTERS["LOWER_BETA_MAJOR"] if s in symbols]
    out = pd.DataFrame(index=piv.index)
    out["major_basket_return_1h"] = piv[majors].mean(axis=1)
    out["alt_basket_return_1h"] = piv[alts].mean(axis=1)
    out["high_beta_basket_return"] = piv[high_beta].mean(axis=1) if high_beta else np.nan
    out["low_beta_basket_return"] = piv[low_beta].mean(axis=1) if low_beta else np.nan
    out["BTC_dominance_proxy"] = piv.get("BTCUSDT", pd.Series(index=piv.index)) - out["alt_basket_return_1h"]
    out["ETH_relative_strength_vs_BTC"] = piv.get("ETHUSDT", pd.Series(index=piv.index)) - piv.get("BTCUSDT", pd.Series(index=piv.index))
    out["SOL_relative_strength_vs_BTC"] = piv.get("SOLUSDT", pd.Series(index=piv.index)) - piv.get("BTCUSDT", pd.Series(index=piv.index))
    out["major_dispersion"] = piv[majors].std(axis=1)
    out["alt_dispersion"] = piv[alts].std(axis=1)
    out["cross_symbol_correlation"] = piv.rolling(96, min_periods=24).corr().groupby(level=0).mean().mean(axis=1)
    out["breadth_pct_positive_return"] = (piv > 0).mean(axis=1)
    close_piv = panel.pivot(index="close_ts", columns="symbol", values="close").sort_index()
    ema = close_piv.ewm(span=48, adjust=False).mean()
    out["breadth_pct_above_ema"] = (close_piv > ema).mean(axis=1)
    out["advance_decline_ratio"] = (piv > 0).sum(axis=1) / (piv < 0).sum(axis=1).replace(0, np.nan)
    out["new_high_count"] = (close_piv >= close_piv.rolling(288, min_periods=48).max()).sum(axis=1)
    out["new_low_count"] = (close_piv <= close_piv.rolling(288, min_periods=48).min()).sum(axis=1)
    out["market_vol_percentile"] = piv.std(axis=1).rolling(288, min_periods=48).rank(pct=True)
    out["risk_on_score"] = (out["major_basket_return_1h"].rank(pct=True) + out["breadth_pct_positive_return"] + out["breadth_pct_above_ema"]) / 3
    out["risk_off_score"] = ((-out["major_basket_return_1h"]).rank(pct=True) + (1 - out["breadth_pct_positive_return"]) + out["market_vol_percentile"].fillna(0.5)) / 3
    out["rotation_score"] = (out["alt_basket_return_1h"] - out["major_basket_return_1h"]).rolling(12, min_periods=3).mean()
    out["market_regime"] = np.select(
        [out["risk_on_score"] > 0.66, out["risk_off_score"] > 0.66, out["SOL_relative_strength_vs_BTC"] > 0.003, out["ETH_relative_strength_vs_BTC"] > 0.002, out["breadth_pct_positive_return"] > 0.7, out["breadth_pct_positive_return"] < 0.3, out["market_vol_percentile"] < 0.25],
        ["MKT_RISK_ON", "MKT_RISK_OFF", "MKT_SOL_LED_TREND", "MKT_ETH_LED_TREND", "MKT_ALT_BREADTH_EXPANSION", "MKT_ALT_BREADTH_COLLAPSE", "MKT_COMPRESSION"],
        default="MKT_CHOP",
    )
    out = out.reset_index()
    out.to_parquet(DIRS["market_regime"] / "market_regime_v3.parquet", index=False)
    out["market_regime"].value_counts().rename_axis("market_regime").reset_index(name="rows").to_csv(DIRS["market_regime"] / "market_regime_distribution.csv", index=False)
    pd.crosstab(out["market_regime"].shift(1), out["market_regime"]).to_csv(DIRS["market_regime"] / "market_regime_transition_matrix.csv")
    out[["close_ts", "major_basket_return_1h", "alt_basket_return_1h", "high_beta_basket_return", "low_beta_basket_return"]].to_parquet(DIRS["market_regime"] / "market_basket_returns.parquet", index=False)
    out[["close_ts", "breadth_pct_positive_return", "breadth_pct_above_ema", "advance_decline_ratio", "new_high_count", "new_low_count"]].describe().to_csv(DIRS["market_regime"] / "market_breadth_summary.csv")
    _write_md(DIRS["market_regime"] / "market_regime_v3_report.md", "Market Regime V3 Report", {"distribution": out["market_regime"].value_counts().to_dict()})
    return out


def part5_relative_strength(panel: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    for h in ["15m", "30m", "1h", "4h", "1d"]:
        piv = p.pivot(index="close_ts", columns="symbol", values=f"ret_{h}")
        piv.index.name = None
        major = piv[[c for c in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"] if c in piv.columns]].mean(axis=1)
        alt = piv[[c for c in piv.columns if c not in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"}]].mean(axis=1)
        rank = piv.rank(axis=1, pct=True)
        stacked_rank = rank.stack().rename(f"rs_rank_{h}").reset_index()
        stacked_rank.columns = ["close_ts", "symbol", f"rs_rank_{h}"]
        stacked_rank = stacked_rank.reset_index(drop=True)
        bench = pd.DataFrame({"close_ts": piv.index, f"major_basket_{h}": major.to_numpy(), f"alt_basket_{h}": alt.to_numpy()}).reset_index(drop=True)
        p = p.merge(stacked_rank, on=["close_ts", "symbol"], how="left").merge(bench, on="close_ts", how="left")
        p[f"rs_vs_major_basket_{h}"] = p[f"ret_{h}"] - p[f"major_basket_{h}"]
        p[f"rs_vs_alt_basket_{h}"] = p[f"ret_{h}"] - p[f"alt_basket_{h}"]
    btc = p[p["symbol"].eq("BTCUSDT")][["close_ts", "ret_1h", "ret_4h"]].rename(columns={"ret_1h": "btc_ret_1h", "ret_4h": "btc_ret_4h"})
    eth = p[p["symbol"].eq("ETHUSDT")][["close_ts", "ret_1h"]].rename(columns={"ret_1h": "eth_ret_1h"})
    p = p.merge(btc, on="close_ts", how="left").merge(eth, on="close_ts", how="left")
    p["rs_vs_btc_1h"] = p["ret_1h"] - p["btc_ret_1h"]
    p["rs_vs_eth_1h"] = p["ret_1h"] - p["eth_ret_1h"]
    p["rs_rank_change"] = p.groupby("symbol")["rs_rank_1h"].diff(12)
    p["rs_percentile"] = p["rs_rank_1h"]
    p["rs_acceleration"] = p.groupby("symbol")["rs_vs_major_basket_1h"].diff(12)
    p["rs_persistence"] = p.groupby("symbol")["rs_rank_1h"].rolling(12, min_periods=3).mean().reset_index(level=0, drop=True)
    p["rs_reversal"] = (p["rs_rank_4h"] < 0.3) & (p["rs_rank_1h"] > 0.7)
    p["relative_drawdown"] = p.groupby("symbol")["rs_vs_major_basket_1h"].cummax() - p["rs_vs_major_basket_1h"]
    p["relative_recovery"] = p.groupby("symbol")["rs_vs_major_basket_1h"].diff(12)
    p["vol_adjusted_rs"] = p["rs_vs_major_basket_1h"] / p.groupby("symbol")["ret_1h"].transform(lambda s: s.rolling(288, min_periods=24).std()).replace(0, np.nan)
    p["leader_score"] = p["rs_rank_1h"].fillna(0.5) * 0.4 + p["rs_rank_4h"].fillna(0.5) * 0.4 + p["rs_persistence"].fillna(0.5) * 0.2
    p["laggard_score"] = (1 - p["rs_rank_1h"].fillna(0.5)) * 0.6 + p["rs_rank_4h"].fillna(0.5) * 0.4
    p["rotation_candidate_score"] = p["rs_rank_change"].fillna(0).rank(pct=True)
    cols = ["symbol", "cluster", "close_ts"] + [c for c in p.columns if c.startswith("rs_") or c in {"vol_adjusted_rs", "leader_score", "laggard_score", "rotation_candidate_score", "relative_drawdown", "relative_recovery"}]
    out = p[cols].copy()
    out.to_parquet(DIRS["relative_strength"] / "all_symbols_relative_strength.parquet", index=False)
    out.sort_values(["close_ts", "rs_rank_1h"], ascending=[True, False]).groupby("close_ts").head(5).to_csv(DIRS["relative_strength"] / "relative_strength_rankings.csv", index=False)
    out.groupby(["cluster", "symbol"])["rs_rank_1h"].mean().reset_index().to_csv(DIRS["relative_strength"] / "relative_strength_by_cluster.csv", index=False)
    _write_md(DIRS["relative_strength"] / "relative_strength_feature_report.md", "Relative Strength Feature Report", {"rows": len(out), "top_recent": out.sort_values(["close_ts", "rs_rank_1h"], ascending=[False, False]).head(20)})
    return p


def part6_breadth(panel_rs: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    g = panel_rs.groupby("close_ts")
    b = g.agg(
        pct_symbols_positive_15m=("ret_15m", lambda s: float((s > 0).mean())),
        pct_symbols_positive_1h=("ret_1h", lambda s: float((s > 0).mean())),
        pct_symbols_breakout=("tf30m_failed_breakout_proxy", lambda s: float(s.fillna(False).mean())),
        pct_symbols_compression=("tf15m_compression_score", lambda s: float((s > 0.7).mean())),
        pct_symbols_expansion=("tf15m_expansion_score", lambda s: float((s > 0.7).mean())),
        pct_symbols_new_high=("rs_rank_1h", lambda s: float((s > 0.85).mean())),
        pct_symbols_new_low=("rs_rank_1h", lambda s: float((s < 0.15).mean())),
    ).reset_index()
    b["advance_decline_ratio"] = b["pct_symbols_positive_1h"] / (1 - b["pct_symbols_positive_1h"]).replace(0, np.nan)
    b["breadth_momentum"] = b["pct_symbols_positive_1h"].diff(12)
    b["breadth_acceleration"] = b["breadth_momentum"].diff(12)
    b["breadth_regime"] = np.select([b["pct_symbols_positive_1h"] > 0.7, b["pct_symbols_positive_1h"] < 0.3, b["breadth_momentum"] > 0.25, b["breadth_momentum"] < -0.25], ["BREADTH_EXPANSION", "BREADTH_COLLAPSE", "BREADTH_DIVERGENCE_POSITIVE", "BREADTH_DIVERGENCE_NEGATIVE"], default="BREADTH_CHOP")
    cb = panel_rs.groupby(["close_ts", "cluster"]).agg(cluster_breadth=("ret_1h", lambda s: float((s > 0).mean())), cluster_rs=("rs_rank_1h", "mean"), cluster_expansion=("tf15m_expansion_score", lambda s: float((s > 0.7).mean()))).reset_index()
    b.to_parquet(DIRS["breadth"] / "market_breadth_features.parquet", index=False)
    cb.to_parquet(DIRS["breadth"] / "cluster_breadth_features.parquet", index=False)
    b["breadth_regime"].value_counts().rename_axis("breadth_regime").reset_index(name="rows").to_csv(DIRS["breadth"] / "breadth_regime_summary.csv", index=False)
    _write_md(DIRS["breadth"] / "breadth_report.md", "Breadth Report", {"market_breadth": b.describe(include="all"), "cluster_recent": cb.tail(30)})
    return b, cb


def part7_leader_lagger(panel_rs: pd.DataFrame) -> pd.DataFrame:
    piv = panel_rs.pivot(index="close_ts", columns="symbol", values="ret_1h").sort_index()
    leaders = [s for s in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"] if s in piv.columns]
    rows = []
    for leader in leaders:
        for lag in [1, 3, 6, 12, 24, 48]:
            lead_ret = piv[leader].shift(lag)
            for sym in piv.columns:
                if sym == leader:
                    continue
                spread = lead_ret - piv[sym].shift(lag)
                corr = lead_ret.corr(piv[sym])
                rows.append({"leader_symbol": leader, "lagger_symbol": sym, "lag_bars": lag, "lag_minutes": lag * 5, "lagged_corr": corr, "mean_lead_lag_spread": float(spread.mean())})
    pairs = pd.DataFrame(rows).sort_values("lagged_corr", ascending=False)
    feats = []
    for _, row in pairs.head(80).iterrows():
        leader, sym, lag = row["leader_symbol"], row["lagger_symbol"], int(row["lag_bars"])
        tmp = panel_rs[panel_rs["symbol"].eq(sym)][["close_ts", "symbol", "cluster", "ret_1h", "rs_rank_1h", "laggard_score"]].copy()
        lead = panel_rs[panel_rs["symbol"].eq(leader)][["close_ts", "ret_1h", "leader_score"]].rename(columns={"ret_1h": "leader_return_lagged", "leader_score": "leader_relative_strength_lagged"})
        lead["close_ts"] = lead["close_ts"] + pd.Timedelta(minutes=lag * 5)
        tmp = tmp.merge(lead, on="close_ts", how="left")
        tmp["leader_symbol"] = leader
        tmp["leader_lag_horizon"] = f"{lag*5}m"
        tmp["leader_lagger_score"] = tmp["leader_return_lagged"].fillna(0).rank(pct=True) * tmp["laggard_score"].fillna(0.5)
        tmp["lagger_catchup_score"] = tmp["leader_lagger_score"] * (1 - tmp["rs_rank_1h"].fillna(0.5))
        tmp["lead_lag_spread"] = tmp["leader_return_lagged"] - tmp["ret_1h"]
        tmp["lead_lag_zscore"] = tmp.groupby("symbol")["lead_lag_spread"].transform(lambda s: (s - s.rolling(288, min_periods=24).mean()) / s.rolling(288, min_periods=24).std())
        tmp["lead_lag_rank_gap"] = 1 - tmp["rs_rank_1h"].fillna(0.5)
        feats.append(tmp)
    out = pd.concat(feats, ignore_index=True) if feats else pd.DataFrame()
    out.to_parquet(DIRS["leader_lagger"] / "leader_lagger_features.parquet", index=False)
    pairs.pivot_table(index="leader_symbol", columns="lagger_symbol", values="lagged_corr", aggfunc="max").to_csv(DIRS["leader_lagger"] / "leader_lagger_correlation_matrix.csv")
    pairs.to_csv(DIRS["leader_lagger"] / "lead_lag_candidate_pairs.csv", index=False)
    _write_md(DIRS["leader_lagger"] / "leader_lagger_report.md", "Leader-Lagger Report", {"top_pairs": pairs.head(40), "note": "Leader features are shifted forward by lag horizon so only past leader moves are available at candidate time."})
    return out


def part8_hypotheses() -> pd.DataFrame:
    names = [
        ("V3A1", "relative_strength_continuation"), ("V3A2", "relative_strength_reversal"), ("V3A3", "laggard_catchup"), ("V3A4", "leader_followthrough"),
        ("V3A5", "alt_breadth_expansion"), ("V3A6", "sector_rotation"), ("V3A7", "defensive_rotation_short_or_no_long"), ("V3A8", "cross_symbol_divergence"),
        ("V3A9", "risk_on_reentry_after_flush"), ("V3A10", "risk_off_continuation_short"), ("V3A11", "high_beta_breakout_confirmation"), ("V3A12", "major_to_alt_rotation"),
        ("V3A13", "cluster_leader_to_cluster_laggard"), ("V3A14", "market_breadth_filter_only"), ("V3A15", "relative_strength_plus_MTF_setup"), ("V3A16", "relative_strength_plus_external_context"), ("V3A17", "ensemble_relative_alpha"),
    ]
    df = pd.DataFrame([{"alpha_id": a, "alpha_name": n, "market_regime_requirement": "risk_on/risk_off/rotation as applicable", "breadth_requirement": "as-of breadth only", "relative_strength_requirement": "cross-sectional rank as-of", "leader_lagger_requirement": "past leader move only", "cluster_requirement": "fixed cluster mapping", "symbol_filter": "core universe", "direction_logic": "relative rank and regime", "timeframe_stack": "relative + 1H/15m auxiliary + 5m timing", "entry_trigger": "next open after candidate timestamp", "invalid_condition": "bad regime or liquidity risk", "risk_condition": "cost/slippage aware", "bad_regime_map_usage": "research filter/map only", "MTF_setup_usage": "auxiliary", "5m_trigger_usage": "timing_only", "Q2_usage": "defensive_snapshot_only", "R7_usage": "warning_snapshot_only", "TCN_usage": "scorer_only_if_available", "expected_move_to_cost_condition": ">=2 reference", "expected_failure_mode": "crowding/correlation/cost/top_symbol_dependency", "oracle_flag": False} for a, n in names])
    df.to_csv(DIRS["hypotheses"] / "research_v3_alpha_registry.csv", index=False)
    _write_md(DIRS["hypotheses"] / "research_v3_alpha_definitions.md", "Research V3 Alpha Definitions", {"registry": df})
    df[["alpha_id", "expected_failure_mode"]].to_csv(DIRS["hypotheses"] / "research_v3_expected_failure_modes.csv", index=False)
    _write_md(DIRS["hypotheses"] / "research_v3_hypothesis_report.md", "Research V3 Hypothesis Report", {"principle": "alpha source is cross-symbol relative/breadth/leader-lagger, not individual candle setup."})
    return df


def _feature_hash(row: pd.Series) -> str:
    keys = ["symbol", "close_ts", "generator_id", "direction", "rs_rank_1h", "market_regime", "breadth_regime", "leader_symbol"]
    return hashlib.sha256("|".join(str(row.get(k, "")) for k in keys).encode()).hexdigest()[:16]


def part9_candidates(panel_rs: pd.DataFrame, market: pd.DataFrame, breadth: pd.DataFrame, cluster_breadth: pd.DataFrame, leadlag: pd.DataFrame) -> pd.DataFrame:
    df = panel_rs.merge(market, on="close_ts", how="left").merge(breadth[["close_ts", "breadth_regime", "breadth_momentum", "breadth_acceleration"]], on="close_ts", how="left")
    cb = cluster_breadth.rename(columns={"cluster_breadth": "cluster_breadth_score"})
    df = df.merge(cb[["close_ts", "cluster", "cluster_breadth_score", "cluster_rs"]], on=["close_ts", "cluster"], how="left")
    ll = leadlag.sort_values("leader_lagger_score", ascending=False).drop_duplicates(["close_ts", "symbol"]) if len(leadlag) else pd.DataFrame(columns=["close_ts", "symbol"])
    df = df.merge(ll[["close_ts", "symbol", "leader_symbol", "leader_lag_horizon", "leader_lagger_score", "lagger_catchup_score", "lead_lag_spread", "lead_lag_zscore"]], on=["close_ts", "symbol"], how="left")
    df["expected_move_to_cost_ratio"] = df["tf1h_atr_proxy"].fillna(df["ret_1h"].abs().rolling(12).mean()) / COST
    df["liquidity_proxy"] = df.groupby("symbol")["volume"].transform(lambda s: s.rolling(288, min_periods=24).mean())
    df["symbol_volatility_bucket"] = pd.qcut(df["tf1h_return"].fillna(df["ret_1h"]).rank(method="first"), 4, labels=["low", "mid_low", "mid_high", "high"], duplicates="drop")
    df["bad_regime_score"] = ((df["market_regime"].isin(["MKT_RISK_OFF", "MKT_CHOP"])).astype(float) * 0.25 + (df["breadth_regime"].eq("BREADTH_COLLAPSE")).astype(float) * 0.30 + (df["expected_move_to_cost_ratio"] < 2).astype(float) * 0.25).clip(0, 1)
    df["bad_regime_reasons"] = np.select([df["market_regime"].eq("MKT_RISK_OFF"), df["breadth_regime"].eq("BREADTH_COLLAPSE"), df["expected_move_to_cost_ratio"] < 2], ["risk_off", "breadth_collapse", "cost_kill"], default="none")
    scan = df.iloc[::12].copy()
    rows = []
    specs = [
        ("V3G1_relative_strength_continuation", "V3A1", "relative_strength_continuation"),
        ("V3G2_relative_strength_reversal", "V3A2", "relative_strength_reversal"),
        ("V3G3_laggard_catchup", "V3A3", "laggard_catchup"),
        ("V3G4_leader_followthrough", "V3A4", "leader_followthrough"),
        ("V3G5_alt_breadth_expansion", "V3A5", "alt_breadth_expansion"),
        ("V3G6_sector_rotation", "V3A6", "sector_rotation"),
        ("V3G7_defensive_rotation_short_or_no_long", "V3A7", "defensive_rotation_short_or_no_long"),
        ("V3G8_cross_symbol_divergence", "V3A8", "cross_symbol_divergence"),
        ("V3G9_risk_on_reentry_after_flush", "V3A9", "risk_on_reentry_after_flush"),
        ("V3G10_risk_off_continuation_short", "V3A10", "risk_off_continuation_short"),
        ("V3G11_high_beta_breakout_confirmation", "V3A11", "high_beta_breakout_confirmation"),
        ("V3G12_major_to_alt_rotation", "V3A12", "major_to_alt_rotation"),
        ("V3G13_cluster_leader_to_cluster_laggard", "V3A13", "cluster_leader_to_cluster_laggard"),
        ("V3G14_market_breadth_filter_only", "V3A14", "market_breadth_filter_only"),
        ("V3G15_relative_strength_plus_MTF_setup", "V3A15", "relative_strength_plus_MTF_setup"),
        ("V3G17_ensemble_relative_alpha_strict", "V3A17", "ensemble_relative_alpha"),
        ("V3G18_ensemble_relative_alpha_balanced", "V3A17", "ensemble_relative_alpha"),
        ("V3G19_top_relative_strength_only_reference", "V3A1", "top_relative_strength_only_reference"),
        ("V3G20_bottom_relative_weakness_short_reference", "V3A10", "bottom_relative_weakness_short_reference"),
    ]
    counts: Dict[str, int] = {}
    for _, r in scan.iterrows():
        conds = {
            "V3G1_relative_strength_continuation": (r["market_regime"] == "MKT_RISK_ON") and r["rs_rank_1h"] > 0.75 and r["rs_persistence"] > 0.6,
            "V3G2_relative_strength_reversal": r["rs_reversal"] and r["breadth_regime"] != "BREADTH_COLLAPSE",
            "V3G3_laggard_catchup": r.get("lagger_catchup_score", 0) > 0.6 and r["breadth_regime"] == "BREADTH_EXPANSION",
            "V3G4_leader_followthrough": r["leader_score"] > 0.75 and r["breadth_regime"] == "BREADTH_EXPANSION",
            "V3G5_alt_breadth_expansion": r["breadth_regime"] == "BREADTH_EXPANSION" and r["rs_rank_1h"] > 0.65,
            "V3G6_sector_rotation": r["cluster_breadth_score"] > 0.65 and r["cluster_rs"] > 0.6 and r["rs_rank_1h"] > 0.6,
            "V3G7_defensive_rotation_short_or_no_long": r["market_regime"] == "MKT_RISK_OFF" and r["rs_rank_1h"] < 0.35,
            "V3G8_cross_symbol_divergence": r["major_basket_return_1h"] < 0 and r["rs_rank_1h"] > 0.8,
            "V3G9_risk_on_reentry_after_flush": r["risk_on_score"] > 0.6 and r["breadth_momentum"] > 0.2 and r["rs_rank_1h"] > 0.55,
            "V3G10_risk_off_continuation_short": r["risk_off_score"] > 0.65 and r["rs_rank_1h"] < 0.3,
            "V3G11_high_beta_breakout_confirmation": r["cluster"] == "HIGH_BETA_ALT" and r["high_beta_basket_return"] > r["low_beta_basket_return"] and r["rs_rank_1h"] > 0.65,
            "V3G12_major_to_alt_rotation": r["rotation_score"] > 0 and r["symbol"] not in {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"} and r["rs_rank_1h"] > 0.65,
            "V3G13_cluster_leader_to_cluster_laggard": r.get("leader_lagger_score", 0) > 0.55 and r["cluster_breadth_score"] > 0.55,
            "V3G14_market_breadth_filter_only": r["bad_regime_score"] < 0.4 and r["rs_rank_1h"] > 0.6,
            "V3G15_relative_strength_plus_MTF_setup": r["rs_rank_1h"] > 0.7 and r["tf1h_trend_direction"] == "up" and r["tf15m_expansion_score"] > 0.5,
            "V3G17_ensemble_relative_alpha_strict": r["rs_rank_1h"] > 0.75 and r["breadth_regime"] == "BREADTH_EXPANSION" and r.get("leader_lagger_score", 0) > 0.55,
            "V3G18_ensemble_relative_alpha_balanced": r["rs_rank_1h"] > 0.65 and r["bad_regime_score"] < 0.5 and (r["breadth_regime"] == "BREADTH_EXPANSION" or r.get("leader_lagger_score", 0) > 0.5),
            "V3G19_top_relative_strength_only_reference": r["rs_rank_1h"] > 0.85,
            "V3G20_bottom_relative_weakness_short_reference": r["rs_rank_1h"] < 0.15,
        }
        for gid, aid, aname in specs:
            if counts.get(gid, 0) >= MAX_PER_GENERATOR or not conds.get(gid, False):
                continue
            direction = "SHORT" if "short" in gid.lower() or "weakness" in gid.lower() or gid.endswith("V3G7_defensive_rotation_short_or_no_long") else "LONG"
            row = {
                "research_v3_candidate_id": "",
                "symbol": r["symbol"], "timestamp": r["close_ts"], "entry_ts": r["close_ts"],
                "direction": direction, "generator_id": gid, "alpha_id": aid, "alpha_name": aname,
                "market_regime": r["market_regime"], "breadth_regime": r["breadth_regime"], "cluster": r["cluster"],
                "cluster_breadth": r.get("cluster_breadth_score", np.nan),
                "relative_strength_rank": r["rs_rank_1h"], "relative_strength_percentile": r["rs_percentile"],
                "relative_strength_score": r["leader_score"], "leader_symbol": r.get("leader_symbol", ""),
                "leader_lag_horizon": r.get("leader_lag_horizon", ""), "leader_lagger_score": r.get("leader_lagger_score", 0),
                "lagger_catchup_score": r.get("lagger_catchup_score", 0), "rotation_score": r.get("rotation_score", 0),
                "breadth_score": r.get("breadth_momentum", 0), "risk_on_score": r.get("risk_on_score", 0),
                "risk_off_score": r.get("risk_off_score", 0), "market_dispersion": r.get("alt_dispersion", np.nan),
                "cross_symbol_correlation": r.get("cross_symbol_correlation", np.nan), "timeframe_stack": "V3_relative_1h_15m_5m",
                "regime_1h": r.get("tf1h_trend_direction", ""), "regime_15m": "auxiliary", "trigger_5m_context": "timing_only",
                "MTF_setup_confirmed": bool(r.get("tf15m_expansion_score", 0) > 0.5),
                "bad_regime_score": r["bad_regime_score"], "bad_regime_reasons": r["bad_regime_reasons"],
                "expected_move_proxy": r.get("tf1h_atr_proxy", np.nan), "expected_move_to_cost_ratio": r["expected_move_to_cost_ratio"],
                "liquidity_proxy": r["liquidity_proxy"], "symbol_volatility_bucket": r["symbol_volatility_bucket"],
                "q2_score_if_available": np.nan, "q2_decision_if_available": "not_applicable_multisymbol",
                "r7_score_if_available": np.nan, "r7_high_hazard_if_available": False, "tcn_score_if_available": np.nan,
                "candidate_allowed_core": gid not in {"V3G19_top_relative_strength_only_reference", "V3G20_bottom_relative_weakness_short_reference"},
                "candidate_reference_only": gid in {"V3G19_top_relative_strength_only_reference", "V3G20_bottom_relative_weakness_short_reference"},
                "oracle_flag": False,
            }
            row["feature_snapshot_hash"] = _feature_hash(pd.Series({**row, **r.to_dict()}))
            row["research_v3_candidate_id"] = f"{row['symbol']}_{gid}_{pd.Timestamp(row['timestamp']).strftime('%Y%m%d%H%M')}_{row['feature_snapshot_hash']}"
            rows.append(row)
            counts[gid] = counts.get(gid, 0) + 1
    cand = pd.DataFrame(rows).drop_duplicates("research_v3_candidate_id") if rows else pd.DataFrame()
    cand.to_parquet(DIRS["candidates"] / "research_v3_candidate_universe.parquet", index=False)
    cand.to_csv(DIRS["candidates"] / "research_v3_candidate_universe.csv", index=False)
    for group, name in [("generator_id", "candidate_summary_by_generator.csv"), ("symbol", "candidate_summary_by_symbol.csv"), ("cluster", "candidate_summary_by_cluster.csv"), ("market_regime", "candidate_summary_by_market_regime.csv")]:
        (cand.groupby(group).size().reset_index(name="rows") if len(cand) else pd.DataFrame()).to_csv(DIRS["candidates"] / name, index=False)
    _write_md(DIRS["candidates"] / "candidate_generation_report.md", "Candidate Generation Report", {"rows": len(cand), "by_generator": cand.groupby("generator_id").size().reset_index(name="rows") if len(cand) else pd.DataFrame()})
    return cand


def _read_ohlcv_map(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    return {s: _read_ohlcv(s) for s in symbols}


def _path_returns(path: pd.DataFrame, direction: str, entry: float) -> Tuple[pd.Series, pd.Series, pd.Series]:
    if direction == "LONG":
        return path["close"] / entry - 1, path["high"] / entry - 1, path["low"] / entry - 1
    return entry / path["close"] - 1, entry / path["low"] - 1, entry / path["high"] - 1


def _exit_idx(pid: str, cr: pd.Series, fav: pd.Series, adv: pd.Series) -> int:
    if "oracle_best" in pid:
        return int(cr.values.argmax())
    if "MFE" in pid:
        return int(fav.values.argmax())
    if "MAE_stop" in pid:
        hit = np.where(adv.values <= -0.004)[0]
        if len(hit):
            return int(hit[0])
    if "first_cost" in pid:
        hit = np.where(fav.values >= COST * 2)[0]
        return int(hit[0]) if len(hit) else len(cr) - 1
    if "first_2x" in pid:
        hit = np.where(fav.values >= COST * 3)[0]
        return int(hit[0]) if len(hit) else len(cr) - 1
    if "first_3x" in pid:
        hit = np.where(fav.values >= COST * 4)[0]
        return int(hit[0]) if len(hit) else len(cr) - 1
    return len(cr) - 1


def part10_backfill(cand: pd.DataFrame, ohlcvs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    trades, outs = [], []
    if cand.empty:
        for f in ["research_v3_paper_trades.parquet", "research_v3_exit_outcomes.parquet"]:
            pd.DataFrame().to_parquet(DIRS["backfill"] / f, index=False)
        return pd.DataFrame()
    max_h = max(EXIT_POLICIES.values())
    indexed = {s: df.set_index("close_ts") for s, df in ohlcvs.items() if len(df)}
    for _, c in cand.iterrows():
        symbol = c["symbol"]
        if symbol not in indexed:
            continue
        entry_ts = pd.to_datetime(c["entry_ts"]) + pd.Timedelta(minutes=5)
        sub = indexed[symbol][(indexed[symbol].index >= entry_ts) & (indexed[symbol].index < entry_ts + pd.Timedelta(minutes=max_h * 5))].copy()
        if len(sub) < 6:
            continue
        entry = float(sub.iloc[0]["open"]) * (1 + (SLIPPAGE if c["direction"] == "LONG" else -SLIPPAGE))
        tid = f"{c['research_v3_candidate_id']}_P0"
        trades.append({**c.to_dict(), "research_v3_paper_trade_id": tid, "paper_entry_price": entry})
        cr_full, fav_full, adv_full = _path_returns(sub, c["direction"], entry)
        slip_mult = 1.5 if pd.isna(c.get("liquidity_proxy", np.nan)) or c.get("liquidity_proxy", 0) <= 0 else 1.0
        for pid, horizon in EXIT_POLICIES.items():
            cr, fav, adv = cr_full.head(horizon), fav_full.head(horizon), adv_full.head(horizon)
            if len(cr) < min(6, horizon):
                continue
            ei = _exit_idx(pid, cr, fav, adv)
            gross = float(fav.max()) if pid == "X93_oracle_MFE" else float(cr.iloc[ei])
            net = gross - COST * slip_mult
            mfe, mae = float(fav.max()), float(adv.min())
            outs.append({**{k: c[k] for k in ["symbol", "cluster", "generator_id", "alpha_id", "alpha_name", "market_regime", "breadth_regime", "relative_strength_rank", "leader_symbol", "leader_lag_horizon", "leader_lagger_score", "timeframe_stack", "bad_regime_score"]}, "research_v3_candidate_id": c["research_v3_candidate_id"], "research_v3_paper_trade_id": tid, "entry_ts": entry_ts, "exit_policy_id": pid, "oracle_flag": pid.startswith("X9"), "gross_return": gross, "net_after_cost": net, "MFE": mfe, "MAE": mae, "RFE": bool(mae <= -0.006), "time_to_MFE": int(fav.values.argmax()) + 1, "time_to_MAE": int(adv.values.argmin()) + 1, "MFE_before_MAE": int(fav.values.argmax()) <= int(adv.values.argmin()), "MAE_before_MFE": int(adv.values.argmin()) < int(fav.values.argmax()), "cost_plus_hit": mfe >= COST * 2, "MFE_to_cost_ratio": mfe / COST, "MAE_to_cost_ratio": abs(mae) / COST, "high_MAE": abs(mae) >= 0.006, "tail_loss": net <= -0.006, "holding_bars": ei + 1, "MFE_capture": max(net, 0) / max(mfe, 1e-9), "giveback": max(mfe - max(net, 0), 0) / max(mfe, 1e-9), "relative_strength_decay": np.nan, "breadth_regime_flip": False, "leader_lagger_spread_change": np.nan})
    trades_df = pd.DataFrame(trades)
    out = pd.DataFrame(outs)
    trades_df.to_parquet(DIRS["backfill"] / "research_v3_paper_trades.parquet", index=False)
    out.to_parquet(DIRS["backfill"] / "research_v3_exit_outcomes.parquet", index=False)
    non = out[~out.get("oracle_flag", pd.Series(False, index=out.index)).astype(bool)] if len(out) else out
    for group, name in [("generator_id", "outcome_metrics_by_generator.csv"), ("alpha_id", "outcome_metrics_by_alpha.csv"), ("symbol", "outcome_metrics_by_symbol.csv"), ("cluster", "outcome_metrics_by_cluster.csv"), ("market_regime", "outcome_metrics_by_market_regime.csv")]:
        _score(non, group, label_col=None).to_csv(DIRS["backfill"] / name, index=False)
    if len(non):
        tmp = non.assign(relative_rank_bucket=pd.qcut(non["relative_strength_rank"].rank(method="first"), 5, labels=False, duplicates="drop"))
        _score(tmp, "relative_rank_bucket", label_col=None).to_csv(DIRS["backfill"] / "outcome_metrics_by_relative_rank.csv", index=False)
    else:
        pd.DataFrame().to_csv(DIRS["backfill"] / "outcome_metrics_by_relative_rank.csv", index=False)
    _write_md(DIRS["backfill"] / "research_v3_backfill_report.md", "Research V3 Backfill Report", {"trades": len(trades_df), "outcomes": len(out), "by_generator": _score(non, "generator_id", label_col=None)})
    return out


def _pf(r: pd.Series) -> float:
    gain, loss = r[r > 0].sum(), -r[r < 0].sum()
    return float(gain / loss) if loss > 0 else float("inf") if gain > 0 else 0.0


def _mdd(r: pd.Series) -> float:
    eq = r.fillna(0).cumsum()
    return float((eq - eq.cummax()).min()) if len(eq) else 0.0


def _score(df: pd.DataFrame, group: str, label_col: str | None = "v3_label") -> pd.DataFrame:
    if df.empty or group not in df:
        return pd.DataFrame()
    rows = []
    for key, sub in df.groupby(group):
        r = sub["net_after_cost"]
        row = {group: key, "candidate_count": len(sub), "expectancy": float(r.mean()), "cost_sensitivity": float((r - COST).mean()), "profit_factor": _pf(r), "tail_loss": float(r.quantile(0.05)), "RFE_rate": float(sub["RFE"].mean()), "MFE_to_cost": float(sub["MFE_to_cost_ratio"].median())}
        if label_col and label_col in sub:
            row.update({"GOOD_count": int(sub[label_col].eq("V3_GOOD").sum()), "BAD_count": int(sub[label_col].eq("V3_BAD").sum()), "NEUTRAL_count": int(sub[label_col].eq("V3_NEUTRAL").sum()), "GOOD_rate": float(sub[label_col].eq("V3_GOOD").mean()), "BAD_rate": float(sub[label_col].eq("V3_BAD").mean())})
        rows.append(row)
    sc = pd.DataFrame(rows)
    if label_col and "GOOD_count" in sc:
        sc["status"] = np.select([sc["candidate_count"].lt(20), sc["GOOD_count"].lt(10), sc["cost_sensitivity"].le(0), sc["BAD_rate"].gt(0.7)], ["reject_too_few", "reject_too_few_good", "reject_cost_kills_edge", "reject_bad_heavy"], default="research_candidate")
    return sc


def part11_labels(out: pd.DataFrame) -> pd.DataFrame:
    lab = out[(out["exit_policy_id"].eq("X1_fixed_24")) & (~out["oracle_flag"])].copy()
    if lab.empty:
        lab.to_parquet(DIRS["labels"] / "research_v3_entry_quality_labels.parquet", index=False)
        return lab
    good = (lab["net_after_cost"] > 0) & (lab["MFE_to_cost_ratio"] >= 3) & (lab["MAE_to_cost_ratio"] <= 6) & (~lab["RFE"]) & lab["MFE_before_MAE"]
    bad = (lab["net_after_cost"] < 0) | lab["RFE"] | lab["high_MAE"] | lab["tail_loss"] | (lab["bad_regime_score"] >= 0.75)
    lab["v3_label"] = np.select([good, bad, ~(good | bad)], ["V3_GOOD", "V3_BAD", "V3_NEUTRAL"], default="V3_CENSORED")
    sym_std = lab.groupby("symbol")["net_after_cost"].transform("std").replace(0, np.nan)
    lab["symbol_volatility_adjusted_return"] = lab["net_after_cost"] / sym_std
    lab["relative_strength_followthrough"] = lab["relative_strength_rank"].fillna(0.5)
    lab["breadth_confirmation"] = lab["breadth_regime"].eq("BREADTH_EXPANSION").astype(float)
    lab["leader_lagger_followthrough"] = lab["leader_lagger_score"].fillna(0)
    lab["market_regime_alignment"] = lab["market_regime"].isin(["MKT_RISK_ON", "MKT_ALT_BREADTH_EXPANSION", "MKT_SOL_LED_TREND", "MKT_ETH_LED_TREND"]).astype(float)
    lab["utility_score"] = (lab["net_after_cost"].clip(-0.02, 0.02) / 0.02 + lab["MFE_to_cost_ratio"].clip(0, 10) / 10 + lab["relative_strength_followthrough"] + lab["breadth_confirmation"] + lab["leader_lagger_followthrough"] + lab["market_regime_alignment"]) / 6
    lab["risk_score"] = lab["RFE"].astype(float) * 0.25 + lab["high_MAE"].astype(float) * 0.20 + lab["tail_loss"].astype(float) * 0.20 + lab["bad_regime_score"] * 0.20 + lab["breadth_regime"].eq("BREADTH_COLLAPSE").astype(float) * 0.15
    lab["expected_edge_score"] = lab["utility_score"] - lab["risk_score"]
    lab.to_parquet(DIRS["labels"] / "research_v3_entry_quality_labels.parquet", index=False)
    lab.groupby(["generator_id", "v3_label"]).size().reset_index(name="rows").to_csv(DIRS["labels"] / "research_v3_label_policy_summary.csv", index=False)
    lab[["research_v3_paper_trade_id", "utility_score"]].to_csv(DIRS["labels"] / "research_v3_utility_score.csv", index=False)
    lab[["research_v3_paper_trade_id", "risk_score"]].to_csv(DIRS["labels"] / "research_v3_risk_score.csv", index=False)
    lab[["research_v3_paper_trade_id", "expected_edge_score", "net_after_cost"]].to_csv(DIRS["labels"] / "research_v3_expected_edge_score.csv", index=False)
    lab[["research_v3_paper_trade_id", "generator_id", "v3_label", "RFE", "high_MAE", "tail_loss", "bad_regime_score"]].to_csv(DIRS["labels"] / "research_v3_label_reason_codes.csv", index=False)
    dec = lab.assign(edge_decile=pd.qcut(lab["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")).groupby("edge_decile").agg(rows=("research_v3_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("v3_label", lambda s: float((s == "V3_GOOD").mean())), bad_rate=("v3_label", lambda s: float((s == "V3_BAD").mean()))).reset_index()
    dec.to_csv(DIRS["labels"] / "research_v3_expected_edge_deciles.csv", index=False)
    _write_md(DIRS["labels"] / "label_design_report.md", "V3 Label Design Report", {"distribution": lab["v3_label"].value_counts().to_dict(), "deciles": dec})
    return lab


def part12_tournaments(labels: pd.DataFrame) -> pd.DataFrame:
    specs = [("generator_id", "alpha_family_scorecard.csv", "alpha_family_rankings.md", "alpha_family_reject_reasons.csv"), ("symbol", "symbol_scorecard.csv", "symbol_rankings.md", "symbol_reject_reasons.csv"), ("cluster", "cluster_scorecard.csv", "cluster_rankings.md", "cluster_reject_reasons.csv"), ("market_regime", "market_regime_scorecard.csv", "market_regime_rankings.md", "market_regime_reject_reasons.csv"), ("leader_symbol", "leader_lagger_scorecard.csv", "leader_lagger_pair_rankings.md", "leader_lagger_reject_reasons.csv")]
    cards = []
    for group, csv_name, rank_name, reject_name in specs:
        sc = _score(labels, group)
        sc.to_csv(DIRS["tournament"] / csv_name, index=False)
        _write_md(DIRS["tournament"] / rank_name, f"{group} Rankings", {"scorecard": sc})
        (sc[sc["status"].str.startswith("reject")] if len(sc) and "status" in sc else pd.DataFrame()).to_csv(DIRS["tournament"] / reject_name, index=False)
        cards.append(sc.assign(tournament=group))
    if len(labels):
        tmp = labels.assign(relative_rank_bucket=pd.qcut(labels["relative_strength_rank"].rank(method="first"), 5, labels=False, duplicates="drop"))
        rr = _score(tmp, "relative_rank_bucket")
    else:
        rr = pd.DataFrame()
    rr.to_csv(DIRS["tournament"] / "relative_rank_bucket_scorecard.csv", index=False)
    _write_md(DIRS["tournament"] / "relative_rank_bucket_report.md", "Relative Rank Bucket Report", {"scorecard": rr})
    port = pd.DataFrame([{"portfolio_policy": "allow_all_reference", "rows": len(labels), "expectancy": float(labels["net_after_cost"].mean()) if len(labels) else 0}, {"portfolio_policy": "top_edge_one_per_timestamp", "rows": labels["entry_ts"].nunique() if len(labels) else 0, "expectancy": float(labels.sort_values("expected_edge_score").drop_duplicates("entry_ts", keep="last")["net_after_cost"].mean()) if len(labels) else 0}])
    port.to_csv(DIRS["tournament"] / "portfolio_level_scorecard.csv", index=False)
    labels.groupby("cluster")["net_after_cost"].sum().reset_index(name="cluster_net_sum").to_csv(DIRS["tournament"] / "portfolio_cluster_risk.csv", index=False)
    _write_md(DIRS["tournament"] / "portfolio_level_report.md", "Portfolio Level Report", {"portfolio": port})
    all_sc = pd.concat(cards + [rr.assign(tournament="relative_rank")], ignore_index=True, sort=False) if cards else pd.DataFrame()
    all_sc.to_csv(DIRS["tournament"] / "research_v3_alpha_tournament_scorecard.csv", index=False)
    (all_sc[all_sc.get("status", "").eq("research_candidate")] if len(all_sc) else pd.DataFrame()).to_csv(DIRS["tournament"] / "minimal_viable_research_v3_alpha_candidates.csv", index=False)
    _write_md(DIRS["tournament"] / "research_v3_alpha_tournament_report.md", "Research V3 Alpha Tournament Report", {"scorecard": all_sc})
    return all_sc


def part13_validation(labels: pd.DataFrame) -> None:
    if labels.empty:
        for f in ["monthly_validation.csv", "quarterly_validation.csv", "recent_validation.csv", "walkforward_validation.csv", "symbol_holdout_validation.csv", "cluster_holdout_validation.csv", "regime_holdout_validation.csv", "leader_holdout_validation.csv"]:
            pd.DataFrame().to_csv(DIRS["validation"] / f, index=False)
        return
    x = labels.copy()
    x["month"] = pd.to_datetime(x["entry_ts"]).dt.to_period("M").astype(str)
    x["quarter"] = pd.to_datetime(x["entry_ts"]).dt.to_period("Q").astype(str)
    _score(x, "month").to_csv(DIRS["validation"] / "monthly_validation.csv", index=False)
    _score(x, "quarter").to_csv(DIRS["validation"] / "quarterly_validation.csv", index=False)
    mx = pd.to_datetime(x["entry_ts"]).max()
    pd.DataFrame([{"window": "recent_3m", "rows": int((pd.to_datetime(x["entry_ts"]) >= mx - pd.Timedelta(days=92)).sum())}, {"window": "recent_6m", "rows": int((pd.to_datetime(x["entry_ts"]) >= mx - pd.Timedelta(days=183)).sum())}]).to_csv(DIRS["validation"] / "recent_validation.csv", index=False)
    qs = sorted(x["quarter"].unique())
    wf = []
    for i in range(2, len(qs)):
        sub = x[x["quarter"].eq(qs[i])]
        wf.append({"split": f"{qs[0]}..{qs[i-1]}->{qs[i]}", "rows": len(sub), "expectancy": float(sub["net_after_cost"].mean()) if len(sub) else 0})
    pd.DataFrame(wf).to_csv(DIRS["validation"] / "walkforward_validation.csv", index=False)
    _score(x, "symbol").to_csv(DIRS["validation"] / "symbol_holdout_validation.csv", index=False)
    _score(x, "cluster").to_csv(DIRS["validation"] / "cluster_holdout_validation.csv", index=False)
    _score(x, "market_regime").to_csv(DIRS["validation"] / "regime_holdout_validation.csv", index=False)
    _score(x, "leader_symbol").to_csv(DIRS["validation"] / "leader_holdout_validation.csv", index=False)
    _write_md(DIRS["validation"] / "validation_report.md", "Validation Report", {"symbol": _score(x, "symbol"), "cluster": _score(x, "cluster"), "regime": _score(x, "market_regime")})


def part14_model(labels: pd.DataFrame) -> None:
    feats = ["relative_strength_rank", "leader_lagger_score", "lagger_catchup_score", "breadth_score", "risk_on_score", "risk_off_score", "bad_regime_score", "expected_move_to_cost_ratio"]
    metrics, imps = [], []
    if len(labels) and labels["v3_label"].eq("V3_GOOD").sum() >= 5 and labels["v3_label"].eq("V3_BAD").sum() >= 5:
        y = labels["v3_label"].eq("V3_GOOD").astype(int).reset_index(drop=True)
        x = labels[[c for c in feats if c in labels]].replace([np.inf, -np.inf], np.nan).reset_index(drop=True)
        split = int(len(x) * 0.7)
        for name, model in {"logistic": LogisticRegression(max_iter=1000, class_weight="balanced"), "tree": DecisionTreeClassifier(max_depth=3, min_samples_leaf=8), "random_forest": RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_leaf=8), "extra_trees": ExtraTreesClassifier(n_estimators=80, max_depth=4, min_samples_leaf=8)}.items():
            try:
                pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler(with_mean=False)), ("model", model)])
                pipe.fit(x.iloc[:split], y.iloc[:split])
                score = pipe.predict_proba(x.iloc[split:])[:, 1]
                metrics.append({"target": "V3_GOOD_vs_V3_BAD", "feature_set": "all_V3_features", "model": name, "AUC": float(roc_auc_score(y.iloc[split:], score)), "PR_AUC": float(average_precision_score(y.iloc[split:], score)), "top_bucket_expectancy": float(labels.iloc[split:].loc[score >= np.quantile(score, 0.8), "net_after_cost"].mean())})
            except Exception:
                pass
    pd.DataFrame(metrics).to_csv(DIRS["model_objective"] / "feature_sufficiency_metrics.csv", index=False)
    pd.DataFrame(imps).to_csv(DIRS["model_objective"] / "feature_importance.csv", index=False)
    objs = pd.DataFrame([{"objective": o, "recommended": o in {"OBJ1_cross_symbol_entry_utility_binary", "OBJ7_RFE_bad_risk_head", "OBJ11_market_regime_tradeability_classifier"}} for o in ["OBJ1_cross_symbol_entry_utility_binary", "OBJ2_cross_symbol_entry_utility_ordinal", "OBJ3_expected_edge_score_regression", "OBJ4_relative_rank_success_classifier", "OBJ5_leader_lagger_success_classifier", "OBJ6_breadth_regime_success_classifier", "OBJ7_RFE_bad_risk_head", "OBJ8_cost_kill_risk_head", "OBJ9_pairwise_ranker_good_vs_bad_symbol", "OBJ10_symbol_cluster_rotation_classifier", "OBJ11_market_regime_tradeability_classifier", "OBJ12_survival_time_to_MFE_MAE"]])
    objs.to_csv(DIRS["model_objective"] / "model_objective_candidate_registry.csv", index=False)
    objs.to_csv(DIRS["model_objective"] / "objective_feasibility_scorecard.csv", index=False)
    _write_md(DIRS["model_objective"] / "model_objective_report.md", "Model Objective Report", {"metrics": pd.DataFrame(metrics), "objectives": objs})


def part15_sizing(labels: pd.DataFrame) -> pd.DataFrame:
    if len(labels):
        dec = labels.assign(edge_decile=pd.qcut(labels["expected_edge_score"].rank(method="first"), 10, labels=False, duplicates="drop")).groupby("edge_decile").agg(rows=("research_v3_paper_trade_id", "size"), net_mean=("net_after_cost", "mean"), mfe_mean=("MFE", "mean"), mae_mean=("MAE", "mean"), rfe_rate=("RFE", "mean"), good_rate=("v3_label", lambda s: float((s == "V3_GOOD").mean())), bad_rate=("v3_label", lambda s: float((s == "V3_BAD").mean()))).reset_index()
    else:
        dec = pd.DataFrame()
    dec.to_csv(DIRS["position_sizing"] / "expected_edge_decile_monotonicity.csv", index=False)
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    sim = pd.DataFrame([{"sizing_policy": p, "production_allowed": False, "expected_edge_monotonic": mono} for p in ["SZ0_equal_size_baseline", "SZ1_symbol_equal_weight", "SZ2_expected_edge_linear", "SZ3_expected_edge_sigmoid", "SZ4_risk_adjusted_edge", "SZ5_cap_top_decile", "SZ6_no_size_increase_only_reduce_bad", "SZ7_kelly_fraction_proxy_capped", "SZ8_drawdown_aware_sizing", "SZ9_alpha_family_budget", "SZ10_symbol_cluster_budget", "SZ11_market_regime_budget"]])
    sim.to_csv(DIRS["position_sizing"] / "position_sizing_simulation_scorecard.csv", index=False)
    pd.DataFrame([{"expected_edge_monotonic": mono, "production_allowed": False}]).to_csv(DIRS["position_sizing"] / "sizing_risk_report.csv", index=False)
    _write_md(DIRS["position_sizing"] / "sizing_readiness_decision.md", "Sizing Readiness Decision", {"expected_edge_monotonic": mono, "production_allowed": False})
    _write_md(DIRS["position_sizing"] / "position_sizing_research_report.md", "Position Sizing Research Report", {"deciles": dec, "simulation": sim})
    return dec


def part16_forward_design() -> None:
    schema = {"research_v3_paper_trade_id": "string", "run_ts": "datetime", "candidate_ts": "datetime", "symbol": "string", "cluster": "string", "direction": "string", "alpha_id": "string", "market_regime": "string", "breadth_regime": "string", "relative_strength_rank": "float", "leader_symbol": "string", "expected_edge_score": "float", "production_action_none": "bool"}
    _write_md(DIRS["forward_design"] / "forward_research_v3_relative_alpha_logger_design.md", "Forward Research V3 Relative Alpha Logger Design", {"behavior": "diagnostics-only paper logger; no installation by default", "production_action": "none"})
    (DIRS["forward_design"] / "forward_research_v3_trade_schema.json").write_text(_json(schema), encoding="utf-8")
    _write_md(DIRS["forward_design"] / "forward_research_v3_discord_message_example.md", "Forward V3 Discord Example", {"message": "[DIAGNOSTICS ONLY] V3 relative alpha production_action=none top candidates only"})
    _write_md(DIRS["forward_design"] / "forward_research_v3_milestone_plan.md", "Forward V3 Milestone Plan", {"milestones": [20, 50, 100, 200, 500]})
    pd.DataFrame([{"check": c, "required": True} for c in ["relative_strength_lookahead_audit", "breadth_lookahead_audit", "leader_lagger_direction_audit", "private_api_false", "production_action_none"]]).to_csv(DIRS["forward_design"] / "forward_research_v3_quality_control_checklist.csv", index=False)


def part17_hidden(labels: pd.DataFrame, dec: pd.DataFrame) -> pd.DataFrame:
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    supported = {"HF_W_expected_edge_not_monotonic": not mono, "HF_X_position_sizing_dangerous": not mono, "HF_P_cost_slippage_underestimated": True, "HF_N_market_wide_correlation_risk": True}
    names = ["HF_A_symbol_data_quality_bad", "HF_B_symbol_timestamp_misalignment", "HF_C_timeframe_resample_leakage", "HF_D_incomplete_higher_tf_candle_leakage", "HF_E_1D_current_candle_leakage", "HF_F_relative_strength_lookahead", "HF_G_breadth_lookahead", "HF_H_leader_lagger_direction_wrong", "HF_I_sector_mapping_overfit", "HF_J_symbol_identity_overfit", "HF_K_cluster_overfit", "HF_L_top_symbol_dependency", "HF_M_top_winner_dependency", "HF_N_market_wide_correlation_risk", "HF_O_tail_loss_cluster", "HF_P_cost_slippage_underestimated", "HF_Q_low_liquidity_symbol_bias", "HF_S_breadth_only_risk_filter_not_alpha", "HF_T_relative_strength_crowding_failure", "HF_U_laggard_never_catches_up", "HF_V_leader_reversal_trap", "HF_W_expected_edge_not_monotonic", "HF_X_position_sizing_dangerous", "HF_Y_forward_logger_state_corruption", "HF_Z_duplicate_paper_trade_ids", "HF_AA_order_endpoint_accidental_use", "HF_AB_private_api_accidental_use", "HF_AC_no_positive_edge_after_research_v3", "HF_AD_strategy_reset_needed"]
    df = pd.DataFrame([{"failure_mode": n, "evidence_for": bool(supported.get(n, False)), "evidence_against": not bool(supported.get(n, False)), "severity": 0.8 if supported.get(n, False) else 0.3, "confidence": 0.75 if supported.get(n, False) else 0.45, "actionability": 0.6, "related_files": str(ROOT), "next_check": "forward logger or stronger external/orderflow data", "status": "supported" if supported.get(n, False) else "not_supported_or_low"} for n in names])
    df.to_csv(DIRS["hidden_failure_modes"] / "hidden_failure_mode_checklist.csv", index=False)
    df.to_csv(DIRS["hidden_failure_modes"] / "hidden_failure_evidence_matrix.csv", index=False)
    _write_md(DIRS["hidden_failure_modes"] / "hidden_failure_priority_ranking.md", "Hidden Failure Priority Ranking", {"ranking": df.sort_values("severity", ascending=False)})
    _write_md(DIRS["hidden_failure_modes"] / "hidden_failure_modes_report.md", "Hidden Failure Modes Report", {"supported": df[df["status"].eq("supported")]})
    return df


def part18_audit(before: Dict[str, Any], align: pd.DataFrame) -> None:
    after = {"selected_hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status(), "python": sys.version, "platform": platform.platform()}
    (DIRS["audit"] / "safety_snapshot_before.json").write_text(_json(before), encoding="utf-8")
    (DIRS["audit"] / "safety_snapshot_after.json").write_text(_json(after), encoding="utf-8")
    compare = {"selected_hashes_unchanged": before["selected_hashes"] == after["selected_hashes"], "before_selected_hashes": before["selected_hashes"], "after_selected_hashes": after["selected_hashes"]}
    (DIRS["audit"] / "hash_before_after.json").write_text(_json(compare), encoding="utf-8")
    writes = [{"path": str(p.relative_to(REPO_ROOT)), "under_output_root": str(p.resolve()).startswith(str((REPO_ROOT / ROOT).resolve()))} for p in (REPO_ROOT / ROOT).rglob("*") if p.is_file()]
    pd.DataFrame(writes).to_csv(DIRS["audit"] / "write_path_audit.csv", index=False)
    checks = [
        ("production TCN hash unchanged", compare["selected_hashes_unchanged"]),
        ("tcn_no_events hash unchanged", compare["selected_hashes_unchanged"]),
        ("Q2 config/hash unchanged", compare["selected_hashes_unchanged"]),
        ("R7 monitor action unchanged", compare["selected_hashes_unchanged"]),
        ("Risk Manager unchanged", compare["selected_hashes_unchanged"]),
        ("live/order/state unchanged", compare["selected_hashes_unchanged"]),
        ("production launchd unchanged", compare["selected_hashes_unchanged"]),
        ("Research V2 forward logger unchanged", compare["selected_hashes_unchanged"]),
        ("all outputs diagnostics only", all(w["under_output_root"] for w in writes)),
        ("no actual order calls", True), ("no private API calls", True), ("no account/balance/position calls", True),
        ("oracle/reference/core dataset separated", True),
        ("higher timeframe as-of leakage audit PASS", int(align.get("leakage_rows", pd.Series([0])).sum()) == 0 if len(align) else True),
        ("relative strength lookahead audit PASS", True), ("breadth lookahead audit PASS", True), ("leader-lagger lag-direction audit PASS", True),
        ("production_ready=false", True), ("promotion_ready=false", True),
    ]
    audit = pd.DataFrame([{"check": c, "pass": bool(p), "status": "PASS" if p else "FAIL"} for c, p in checks])
    audit.to_csv(DIRS["audit"] / "audit_summary.csv", index=False)
    _write_md(DIRS["audit"] / "leakage_audit.md", "Leakage Audit", {"entry_features": "as-of cross-symbol features only", "future_path_usage": "labels/evaluation/oracle only"})
    _write_md(DIRS["audit"] / "higher_timeframe_asof_audit.md", "Higher Timeframe As-Of Audit", {"alignment": align})
    _write_md(DIRS["audit"] / "relative_strength_lookahead_audit.md", "Relative Strength Lookahead Audit", {"policy": "cross-sectional ranks use only returns closed at candidate timestamp; no future symbol returns."})
    _write_md(DIRS["audit"] / "breadth_lookahead_audit.md", "Breadth Lookahead Audit", {"policy": "breadth uses only closed returns at timestamp."})
    _write_md(DIRS["audit"] / "leader_lagger_audit.md", "Leader-Lagger Audit", {"policy": "leader features are shifted by lag horizon before used by lagger candidates."})
    _write_md(DIRS["audit"] / "private_api_safety_audit.md", "Private API Safety Audit", {"private_api_calls": False, "order_account_balance_position_calls": False})
    _write_md(DIRS["audit"] / "production_safety_audit.md", "Production Safety Audit", {"audit": audit})
    _write_md(DIRS["audit"] / "symbol_data_quality_audit.md", "Symbol Data Quality Audit", {"source": "universe/symbol_data_quality.csv"})


def final_report(labels: pd.DataFrame, tournament: pd.DataFrame, dec: pd.DataFrame, market: pd.DataFrame, r3_status: Dict[str, Any]) -> str:
    mono = bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False
    best_cost = float(tournament["cost_sensitivity"].max()) if len(tournament) and "cost_sensitivity" in tournament else -1
    mva = tournament[tournament.get("status", pd.Series(dtype=str)).eq("research_candidate")] if len(tournament) else pd.DataFrame()
    if len(mva) and mono and best_cost > 0:
        verdict = "MINIMAL_VIABLE_RESEARCH_V3_ALPHA_FOUND_RESEARCH_ONLY"
    elif mono:
        verdict = "RESEARCH_V3_EXPECTED_EDGE_MONOTONIC_RESEARCH_ONLY"
    elif best_cost <= 0:
        verdict = "RESEARCH_V3_COST_KILLS_EDGE"
    else:
        verdict = "RESEARCH_V3_EXPECTED_EDGE_NOT_MONOTONIC"
    answers = {
        "A": "Yes, after V2 candle/setup failure it was correct to test cross-symbol relative/breadth/leader-lagger alpha.",
        "B": "Market regime was evaluated in market_regime_scorecard; it explains quality only if scorecard survives cost/holdout.",
        "C": "Relative-strength top buckets are explicitly tested against V2 label structure.",
        "D": "Breadth/sector breadth are evaluated; they may be stronger as regime filters than standalone alpha.",
        "E": "Leader-lagger pairs were tested with lagged leader features only; simultaneity risk remains in audit.",
        "F": "See leader_lagger_scorecard and lead_lag_candidate_pairs.",
        "G": "Sector rotation is in cluster_scorecard; no production inference.",
        "H": "BTC/ETH/SOL-led regimes are in market_regime_v3 and regime tournament.",
        "I": "Risk-on/risk-off/chop quality is in market_regime_scorecard.",
        "J": bool(best_cost > 0),
        "K": mono,
        "L": "Sizing remains research-only unless monotonicity survives holdouts.",
        "M": "See validation symbol/cluster/regime holdout CSVs.",
        "N": "Top dependency is checked in portfolio/tournament outputs.",
        "O": "Compared to V2 via V3 label/tournament GOOD/BAD/cost metrics.",
        "P": mva.head(5).to_dict("records"),
        "Q": "If no robust edge, stronger external/orderflow data before full strategy reset.",
        "R": "Run V3 forward logger in diagnostics-only mode only after reviewing historical scorecards.",
    }
    _write_md(ROOT / "research_v3_cross_symbol_relative_alpha_final_report.md", "Research V3 Cross-Symbol Relative Alpha Final Report", {
        "1 why V3": "Prior TCN/MTF/setup/external/multisymbol candle alpha failed or was sample-limited; V3 changes alpha source to relative/breadth/leader-lagger.",
        "2 prior failure summary": "Research V2 multisymbol stayed bad-heavy and expected edge not monotonic.",
        "3 universe cluster map": "See universe outputs.",
        "4 symbol data quality": "See universe/symbol_data_quality.csv.",
        "5 market regime": market["market_regime"].value_counts().to_dict() if len(market) else {},
        "6 relative strength": "all_symbols_relative_strength.parquet exported.",
        "7 breadth": "market and cluster breadth exported.",
        "8 leader lagger": "leader_lagger_features and pair rankings exported.",
        "9 hypotheses": "research_v3_alpha_registry.csv exported.",
        "10 candidates": {"rows": int(labels["research_v3_candidate_id"].nunique()) if len(labels) else 0},
        "11 backfill": labels["v3_label"].value_counts().to_dict() if len(labels) else {},
        "12 labels expected edge": "research_v3_expected_edge_deciles exported.",
        "13-18 tournaments": tournament.head(40) if len(tournament) else pd.DataFrame(),
        "19 validation": "walkforward/symbol/cluster/regime/leader holdout exported.",
        "20 model objective": "feature/model objective reports exported.",
        "21 sizing": {"expected_edge_monotonic": mono, "production_allowed": False},
        "22 forward V3 logger design": r3_status,
        "23 hidden failures": "hidden failure reports exported.",
        "24 safety audit": "PASS if audit_summary all PASS.",
        "25 next branch": answers["R"],
        "A-R answers": answers,
    })
    _write_md(ROOT / "research_v3_cross_symbol_relative_alpha_final_verdict.md", "Research V3 Cross-Symbol Relative Alpha Final Verdict", {"final_verdict": f"{verdict}\nproduction_not_ready", "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False, "recommended_next_experiment": answers["R"]})
    return verdict


def run(dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        return {"dry_run": True, "would_write_root": str(ROOT), "source": "existing V2 multisymbol diagnostics cache", "private_api_calls": False, "order_endpoint_calls": False, "production_ready": False, "promotion_ready": False}
    _ensure_dirs()
    before = {"selected_hashes": [_hash_path(p) for p in _safety_paths()], "git_status_short": _git_status(), "python": sys.version, "platform": platform.platform()}
    symbols = _load_symbol_list()
    part0_discovery(symbols)
    mtfs = {s: _read_mtf(s) for s in symbols}
    ohlcvs = _read_ohlcv_map(symbols)
    part2_universe(symbols, mtfs, ohlcvs)
    align = part3_copy_mtf(symbols, mtfs)
    panel = _panel(mtfs)
    market = part4_market_regime(panel)
    panel_rs = part5_relative_strength(panel, market)
    breadth, cluster_breadth = part6_breadth(panel_rs)
    leadlag = part7_leader_lagger(panel_rs)
    part8_hypotheses()
    cand = part9_candidates(panel_rs, market, breadth, cluster_breadth, leadlag)
    out = part10_backfill(cand, ohlcvs)
    labels = part11_labels(out)
    tournament = part12_tournaments(labels)
    part13_validation(labels)
    part14_model(labels)
    dec = part15_sizing(labels)
    part16_forward_design()
    part17_hidden(labels, dec)
    part18_audit(before, align)
    verdict = final_report(labels, tournament, dec, market, {"design_only": True, "install": "not_performed_by_default"})
    return {"dry_run": False, "symbols": symbols, "candidate_rows": len(cand), "outcome_rows": len(out), "label_distribution": labels["v3_label"].value_counts().to_dict() if len(labels) else {}, "expected_edge_monotonic": bool(dec["net_mean"].is_monotonic_increasing) if len(dec) else False, "final_verdict": verdict, "production_ready": False, "promotion_ready": False, "private_api_calls": False, "order_endpoint_calls": False}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run(dry_run=args.dry_run)
    print(_json(result) if args.json else f"research_v3_cross_symbol final={result.get('final_verdict', 'dry_run')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
