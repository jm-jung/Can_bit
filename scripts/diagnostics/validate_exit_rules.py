"""
Exit 구조 검증 (분석 전용). 운영 전략/엔진 코드를 변경하지 않는다.

실행: 프로젝트 루트에서
  python -m scripts.diagnostics.validate_exit_rules
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_IDS: Tuple[str, ...] = (
    "20260428_233713",
    "20260429_135041",
    "20260501_215331",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MON_DIR = REPO_ROOT / "data" / "monitoring"
OHLCV_PATH = REPO_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"
OUT_DIR = REPO_ROOT / "data" / "diagnostics" / "exit_validation"

FIXED_EXIT_BARS: Tuple[int, ...] = (4, 6, 8, 10, 12, 16, 24)

PROFIT_LOCK_RULES: Tuple[Dict[str, Any], ...] = (
    {"rule_name": "A_tp0.0015_sl-0.0030", "take_profit": 0.0015, "stop_loss": -0.0030},
    {"rule_name": "B_tp0.0020_sl-0.0030", "take_profit": 0.0020, "stop_loss": -0.0030},
    {"rule_name": "C_tp0.0025_sl-0.0035", "take_profit": 0.0025, "stop_loss": -0.0035},
    {"rule_name": "D_tp0.0030_sl-0.0040", "take_profit": 0.0030, "stop_loss": -0.0040},
    {"rule_name": "E_tp0.0015_sl-0.0020", "take_profit": 0.0015, "stop_loss": -0.0020},
)

RETRACE_RULES: Tuple[Dict[str, Any], ...] = (
    {"rule_name": "R1_mfe0.0020_gb0.0010", "activate_mfe": 0.0020, "giveback": 0.0010},
    {"rule_name": "R2_mfe0.0025_gb0.0012", "activate_mfe": 0.0025, "giveback": 0.0012},
    {"rule_name": "R3_mfe0.0030_gb0.0015", "activate_mfe": 0.0030, "giveback": 0.0015},
)

LATEST_RUN = "20260501_215331"


# ---------------------------------------------------------------------------
# Time / IO helpers
# ---------------------------------------------------------------------------


def _parse_ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _jsonl_path(run_id: str) -> Path:
    return MON_DIR / f"monitor_guard_stage2_{run_id}.jsonl"


def _summary_path(run_id: str) -> Path:
    return MON_DIR / f"monitor_guard_stage2_summary_{run_id}.json"


def load_summary_trades(run_id: str) -> Optional[int]:
    p = _summary_path(run_id)
    if not p.is_file():
        return None
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return int(data.get("total_trades", -1))


def load_ohlcv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"OHLCV not found: {path}")
    df = pd.read_csv(path)
    # timestamp column autodetect
    ts_col = None
    for c in df.columns:
        if c.lower() in ("timestamp", "time", "ts", "datetime", "date"):
            ts_col = c
            break
    if ts_col is None:
        # first column fallback
        ts_col = df.columns[0]
    ohlc = {"open": None, "high": None, "low": None, "close": None}
    for c in df.columns:
        cl = c.lower()
        if cl in ohlc and ohlc[cl] is None:
            ohlc[cl] = c
    miss = [k for k, v in ohlc.items() if v is None]
    if miss:
        raise ValueError(f"OHLCV missing columns: {miss} in {path}")
    out = pd.DataFrame(
        {
            "ts": pd.to_datetime(df[ts_col], errors="coerce"),
            "open": pd.to_numeric(df[ohlc["open"]], errors="coerce"),
            "high": pd.to_numeric(df[ohlc["high"]], errors="coerce"),
            "low": pd.to_numeric(df[ohlc["low"]], errors="coerce"),
            "close": pd.to_numeric(df[ohlc["close"]], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    dup_before = len(out)
    out = out.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    dup_after = len(out)
    if dup_before != dup_after:
        # 로그용 메타 — 호출부에서 집계 가능
        out.attrs["dedupe_removed"] = dup_before - dup_after
    return out


def rebuild_trades_from_jsonl(jsonl_path: Path, run_id: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """ENTRY/EXIT 매칭으로 trade row 복원. 문제 발생 시 messages 반환."""
    messages: List[str] = []
    entries: Dict[int, Dict[str, Any]] = {}
    exits: Dict[int, Dict[str, Any]] = {}
    if not jsonl_path.is_file():
        messages.append(f"missing_jsonl:{jsonl_path}")
        return [], messages

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError as e:
                messages.append(f"json_decode_error:{e}")
                continue
            if ev.get("run_id") and str(ev.get("run_id")) != run_id:
                continue
            if ev.get("event") == "ENTRY":
                tid = int(ev["trade_id"])
                if tid in entries:
                    messages.append(f"duplicate_ENTRY_trade_id={tid}")
                entries[tid] = ev
            elif ev.get("event") == "EXIT":
                tid = int(ev["trade_id"])
                if tid in exits:
                    messages.append(f"duplicate_EXIT_trade_id={tid}")
                exits[tid] = ev
            else:
                pass

    trade_ids = sorted(set(entries.keys()) & set(exits.keys()))
    only_e = set(entries.keys()) - set(exits.keys())
    only_x = set(exits.keys()) - set(entries.keys())
    if only_e:
        messages.append(f"ENTRY_without_EXIT:{sorted(only_e)}")
    if only_x:
        messages.append(f"EXIT_without_ENTRY:{sorted(only_x)}")

    rows: List[Dict[str, Any]] = []
    for tid in trade_ids:
        en = entries[tid]
        ex = exits[tid]
        side = str(en.get("side") or en.get("direction") or "LONG").upper()
        if side not in ("LONG", "SHORT"):
            side = "LONG"
        gc = en.get("guard_components") or {}
        p_long = gc.get("p_long")
        p_short = gc.get("p_short")
        if p_long is not None and p_short is None:
            p_short = 1.0 - float(p_long)
        max_proba = None
        if p_long is not None and p_short is not None:
            max_proba = max(float(p_long), float(p_short))
        elif p_long is not None:
            max_proba = float(p_long)

        rp = float(ex.get("realized_profit") or ex.get("profit") or math.nan)
        ep = float(en["entry_price"])
        xp = float(ex["exit_price"])
        raw_exit = directional_raw_profit(ep, xp, side)
        row = {
            "run_id": run_id,
            "trade_id": tid,
            "direction": side,
            "entry_ts": _parse_ts(str(en["ts"])),
            "exit_ts": _parse_ts(str(ex["ts"])),
            "entry_price": ep,
            "exit_price": xp,
            "original_bars_held": int(ex.get("holding_bars") or ex.get("bars_held") or -1),
            "original_profit": rp,
            "baseline_raw_profit": raw_exit,
            "original_scaled_profit": ex.get("scaled_realized_profit") or ex.get("scaled_profit"),
            "original_result_win": bool(float(ex.get("realized_profit") or 0) > 0),
            "p_long": float(p_long) if p_long is not None else math.nan,
            "p_short": float(p_short) if p_short is not None else math.nan,
            "max_proba": float(max_proba) if max_proba is not None else math.nan,
            "margin": float(gc["margin"]) if gc.get("margin") is not None else math.nan,
            "entropy": float(gc["entropy"]) if gc.get("entropy") is not None else math.nan,
            "guard_scale": float(en["guard_scale"]) if en.get("guard_scale") is not None else math.nan,
            "final_scale": float(en["final_scale"]) if en.get("final_scale") is not None else math.nan,
        }
        rows.append(row)
    rows.sort(key=lambda r: r["trade_id"])
    return rows, messages


# ---------------------------------------------------------------------------
# PnL math
# ---------------------------------------------------------------------------


def directional_raw_profit(entry: float, exit_px: float, direction: str) -> float:
    if direction == "LONG":
        return exit_px / entry - 1.0
    return entry / exit_px - 1.0


def calc_mfe_mae_slice(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    last_idx: int,
    entry_price: float,
    direction: str,
) -> Tuple[float, float]:
    """last_idx inclusive 구간에서 MFE/MAE (비율)."""
    sl = ohlcv.iloc[entry_idx : last_idx + 1]
    if sl.empty:
        return math.nan, math.nan
    if direction == "LONG":
        highs = sl["high"].to_numpy()
        lows = sl["low"].to_numpy()
        mfe = float(np.max(highs / entry_price - 1.0))
        mae = float(np.min(lows / entry_price - 1.0))
        return mfe, mae
    lows = sl["low"].to_numpy()
    highs = sl["high"].to_numpy()
    mfe = float(np.max(entry_price / lows - 1.0))
    mae = float(np.min(entry_price / highs - 1.0))
    return mfe, mae


def simulate_fixed_bars(
    ohlcv: pd.DataFrame,
    ts_index: pd.Series,
    entry_ts: pd.Timestamp,
    entry_price: float,
    direction: str,
    exit_bars: int,
) -> Tuple[Optional[float], str]:
    """close(entry_idx+exit_bars) 기준 수익률. 불가 시 (None, reason)."""
    if exit_bars < 1:
        return None, "invalid_exit_bars"
    entry_idx = resolve_bar_index(ts_index, pd.Timestamp(entry_ts))
    if entry_idx < 0:
        return None, "entry_ts_not_in_ohlcv"
    exit_idx = entry_idx + exit_bars
    if exit_idx >= len(ohlcv):
        return None, "ohlcv_horizon_insufficient"
    c = float(ohlcv.iloc[exit_idx]["close"])
    return directional_raw_profit(entry_price, c, direction), "ok"


def simulate_profit_lock(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    direction: str,
    take_profit: float,
    stop_loss: float,
    max_offset: int = 12,
) -> Tuple[float, str, int, int, int, int]:
    """
    entry 이후 max_offset 개의 봉(1..max_offset)에서 순차 탐색 후,
    미체결 시 entry_idx+max_offset 종가 청산.
    반환: (profit, exit_kind, bars_held, tp_hit, sl_hit, time_exit)
    exit_kind: tp | sl | time
    """
    tp_hit = sl_hit = time_exit = 0
    if entry_idx + max_offset >= len(ohlcv):
        return math.nan, "insufficient_bars", -1, 0, 0, 0

    sl_thr = abs(stop_loss)

    for off in range(1, max_offset + 1):
        bi = entry_idx + off
        row = ohlcv.iloc[bi]
        hi = float(row["high"])
        lo = float(row["low"])
        if direction == "LONG":
            tp_px = entry_price * (1.0 + take_profit)
            sl_px = entry_price * (1.0 + stop_loss)
            tp_hit_bar = hi >= tp_px
            sl_hit_bar = lo <= sl_px
            if tp_hit_bar and sl_hit_bar:
                prof = stop_loss
                sl_hit = 1
                return prof, "sl", off, tp_hit, sl_hit, time_exit
            if sl_hit_bar:
                sl_hit = 1
                return stop_loss, "sl", off, tp_hit, sl_hit, time_exit
            if tp_hit_bar:
                tp_hit = 1
                return take_profit, "tp", off, tp_hit, sl_hit, time_exit
        else:
            tp_px = entry_price / (1.0 + take_profit)
            sl_px = entry_price / (1.0 + sl_thr)
            tp_hit_bar = lo <= tp_px
            sl_hit_bar = hi >= sl_px
            if tp_hit_bar and sl_hit_bar:
                sl_hit = 1
                return stop_loss, "sl", off, tp_hit, sl_hit, time_exit
            if sl_hit_bar:
                sl_hit = 1
                return stop_loss, "sl", off, tp_hit, sl_hit, time_exit
            if tp_hit_bar:
                tp_hit = 1
                return take_profit, "tp", off, tp_hit, sl_hit, time_exit

    # time exit at close of entry_idx + max_offset
    time_exit = 1
    exit_idx = entry_idx + max_offset
    c = float(ohlcv.iloc[exit_idx]["close"])
    prof = directional_raw_profit(entry_price, c, direction)
    return prof, "time", max_offset, tp_hit, sl_hit, time_exit


def simulate_retracement(
    ohlcv: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
    direction: str,
    activate_mfe: float,
    giveback: float,
    max_offset: int = 12,
) -> Tuple[float, str, int, int, int]:
    """반환: profit, exit_kind (retrace|time), bars_held, retrace_exit, time_exit"""
    retrace_exit = time_exit = 0
    if entry_idx + max_offset >= len(ohlcv):
        return math.nan, "insufficient_bars", -1, 0, 0

    activated = False
    peak_high = -math.inf
    trough_low = math.inf

    for off in range(1, max_offset + 1):
        bi = entry_idx + off
        row = ohlcv.iloc[bi]
        hi = float(row["high"])
        lo = float(row["low"])
        cl = float(row["close"])

        if direction == "LONG":
            peak_high = max(peak_high, hi)
            mfe = peak_high / entry_price - 1.0
            cur = cl / entry_price - 1.0
            if mfe >= activate_mfe:
                activated = True
            if activated and cur <= mfe - giveback:
                retrace_exit = 1
                return cur, "retrace", off, retrace_exit, time_exit
        else:
            trough_low = min(trough_low, lo)
            mfe = entry_price / trough_low - 1.0
            cur = entry_price / cl - 1.0
            if mfe >= activate_mfe:
                activated = True
            if activated and cur <= mfe - giveback:
                retrace_exit = 1
                return cur, "retrace", off, retrace_exit, time_exit

    time_exit = 1
    exit_idx = entry_idx + max_offset
    c = float(ohlcv.iloc[exit_idx]["close"])
    prof = directional_raw_profit(entry_price, c, direction)
    return prof, "time", max_offset, retrace_exit, time_exit


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def summarize_profits(
    profits: Sequence[float],
) -> Dict[str, Any]:
    arr = np.array([p for p in profits if not math.isnan(p)], dtype=float)
    n = len(arr)
    if n == 0:
        return {
            "trade_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "win_rate": math.nan,
            "total_profit": math.nan,
            "mean_profit": math.nan,
            "median_profit": math.nan,
            "max_win": math.nan,
            "max_loss": math.nan,
            "p10": math.nan,
            "p25": math.nan,
            "p50": math.nan,
            "p75": math.nan,
            "p90": math.nan,
        }
    wins = arr > 0
    return {
        "trade_count": n,
        "win_count": int(wins.sum()),
        "loss_count": int((~wins).sum()),
        "win_rate": float(wins.mean()),
        "total_profit": float(arr.sum()),
        "mean_profit": float(arr.mean()),
        "median_profit": float(np.median(arr)),
        "max_win": float(arr.max()),
        "max_loss": float(arr.min()),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


def trade_key(row: Mapping[str, Any], use_exit: bool) -> Tuple[str, str, str]:
    d = str(row["direction"])
    et = pd.Timestamp(row["entry_ts"]).isoformat()
    if use_exit:
        xt = pd.Timestamp(row["exit_ts"]).isoformat()
        return (d, et, xt)
    return (d, et, "")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_ts_index(ohlcv: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    msgs: List[str] = []
    ts = ohlcv["ts"]
    if ts.duplicated().any():
        msgs.append("unexpected_duplicate_ts_after_dedupe")
    idx = pd.Series(np.arange(len(ohlcv)), index=ts)
    return idx, msgs


def resolve_bar_index(ts_index: pd.Series, ts: pd.Timestamp) -> int:
    """timestamp 인덱스가 중복이면 첫 행을 사용."""
    try:
        loc = ts_index.loc[ts]
    except KeyError:
        return -1
    if isinstance(loc, pd.Series):
        return int(loc.iloc[0])
    return int(loc)


def attach_entry_indices(
    trades: List[Dict[str, Any]],
    ts_index: pd.Series,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    msgs: List[str] = []
    for t in trades:
        ets = pd.Timestamp(t["entry_ts"])
        bi = resolve_bar_index(ts_index, ets)
        if bi < 0:
            msgs.append(f"trade_id={t['trade_id']}:entry_ts_missing_in_ohlcv:{ets}")
        t["entry_idx"] = bi
    return trades, msgs


def main() -> None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ohlcv = load_ohlcv(OHLCV_PATH)
    ts_index, idx_msgs = build_ts_index(ohlcv)

    all_trades: Dict[str, List[Dict[str, Any]]] = {}
    run_logs: Dict[str, List[str]] = {}

    for rid in RUN_IDS:
        path = _jsonl_path(rid)
        trades, msgs = rebuild_trades_from_jsonl(path, rid)
        trades, map_msgs = attach_entry_indices(trades, ts_index)
        summary_n = load_summary_trades(rid)
        match_ok = summary_n is not None and summary_n == len(trades)
        run_logs[rid] = msgs + map_msgs + [
            f"summary_total_trades={summary_n}",
            f"reconstructed_trades={len(trades)}",
            f"match={'yes' if match_ok else 'mismatch'}",
        ]
        all_trades[rid] = trades

    # Step 3: original vs calculated
    repro_rows: List[Dict[str, Any]] = []
    for rid, trades in all_trades.items():
        diffs: List[float] = []
        calc_total = 0.0
        orig_total = 0.0
        for t in trades:
            cp = directional_raw_profit(t["entry_price"], t["exit_price"], t["direction"])
            op = t["original_profit"]
            if not math.isnan(op):
                diffs.append(cp - op)
            if not math.isnan(cp):
                calc_total += cp
            if not math.isnan(op):
                orig_total += op
            repro_rows.append(
                {
                    "run_id": rid,
                    "trade_id": t["trade_id"],
                    "calc_profit": cp,
                    "original_profit": op,
                    "diff": cp - op if not math.isnan(op) else math.nan,
                }
            )
        abs_mean = float(np.mean(np.abs(diffs))) if diffs else math.nan
        abs_max = float(np.max(np.abs(diffs))) if diffs else math.nan
        run_logs[rid].append(
            f"reproduce:orig_total={orig_total:.8f},calc_total={calc_total:.8f},"
            f"diff_abs_mean={abs_mean},diff_abs_max={abs_max}"
        )

    proxy_note = any(
        any("entry_ts_missing" in m for m in run_logs[rid]) for rid in RUN_IDS
    )

    # Simulations
    fixed_rows: List[Dict[str, Any]] = []
    lock_rows: List[Dict[str, Any]] = []
    retrace_rows: List[Dict[str, Any]] = []

    per_trade_sim: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)

    for rid, trades in all_trades.items():
        for t in trades:
            tid = int(t["trade_id"])
            entry_price = t["entry_price"]
            direction = t["direction"]
            entry_ts = pd.Timestamp(t["entry_ts"])
            entry_idx = int(t["entry_idx"])
            orig_p = t["baseline_raw_profit"]

            bag: Dict[str, Any] = {
                "fixed": {},
                "lock": {},
                "retrace": {},
            }

            # fixed bars
            for eb in FIXED_EXIT_BARS:
                p, reason = simulate_fixed_bars(ohlcv, ts_index, entry_ts, entry_price, direction, eb)
                bag["fixed"][eb] = p
                row = {
                    "run_id": rid,
                    "exit_bars": eb,
                    "trade_id": tid,
                    "profit": p if p is not None else math.nan,
                    "status": reason if p is None else "ok",
                }
                fixed_rows.append(row)

            # MFE/MAE on original 12-bar window
            mfe = mae = math.nan
            if entry_idx >= 0 and entry_idx + 12 < len(ohlcv):
                mfe, mae = calc_mfe_mae_slice(ohlcv, entry_idx, entry_idx + 12, entry_price, direction)

            # profit lock
            for rule in PROFIT_LOCK_RULES:
                if entry_idx < 0:
                    prof = math.nan
                    kind = "no_entry_idx"
                    bh = -1
                    tph = slh = te = 0
                else:
                    prof, kind, bh, tph, slh, te = simulate_profit_lock(
                        ohlcv,
                        entry_idx,
                        entry_price,
                        direction,
                        float(rule["take_profit"]),
                        float(rule["stop_loss"]),
                    )
                bag["lock"][rule["rule_name"]] = prof
                lock_rows.append(
                    {
                        "run_id": rid,
                        "rule_name": rule["rule_name"],
                        "trade_id": tid,
                        "profit": prof,
                        "exit_kind": kind,
                        "bars_held": bh,
                    }
                )

            # retracement
            for rule in RETRACE_RULES:
                if entry_idx < 0:
                    prof = math.nan
                    kind = "no_entry_idx"
                    bh = -1
                    re_x = t_x = 0
                else:
                    prof, kind, bh, re_x, t_x = simulate_retracement(
                        ohlcv,
                        entry_idx,
                        entry_price,
                        direction,
                        float(rule["activate_mfe"]),
                        float(rule["giveback"]),
                    )
                bag["retrace"][rule["rule_name"]] = prof
                retrace_rows.append(
                    {
                        "run_id": rid,
                        "rule_name": rule["rule_name"],
                        "trade_id": tid,
                        "profit": prof,
                        "exit_kind": kind,
                        "bars_held": bh,
                    }
                )

            # collect alternatives for "best per trade"
            alt_specs: List[Tuple[str, float, str, int]] = []
            for eb, p in bag["fixed"].items():
                if p is not None and not math.isnan(p):
                    alt_specs.append((f"fixed_{eb}", float(p), "fixed", int(eb)))
            for rn, p in bag["lock"].items():
                if not math.isnan(p):
                    alt_specs.append((rn, float(p), "lock", -1))
            for rn, p in bag["retrace"].items():
                if not math.isnan(p):
                    alt_specs.append((rn, float(p), "retrace", -1))

            best_name = ""
            best_p = math.nan
            best_type = ""
            best_bars = -1
            if alt_specs:
                best_name, best_p, best_type, best_bars = max(alt_specs, key=lambda x: x[1])

            per_trade_sim[rid][tid] = {
                "baseline_raw_profit": orig_p,
                "mfe": mfe,
                "mae": mae,
                "best_alt_profit": best_p,
                "best_alt_name": best_name,
                "best_alt_type": best_type,
                "best_alt_bars_hint": best_bars,
                "bag": bag,
            }

    # Aggregate fixed by run + exit_bars
    fixed_agg: List[Dict[str, Any]] = []
    for rid in RUN_IDS:
        for eb in FIXED_EXIT_BARS:
            profits = [
                r["profit"]
                for r in fixed_rows
                if r["run_id"] == rid and r["exit_bars"] == eb and not math.isnan(r["profit"])
            ]
            stats = summarize_profits(profits)
            fixed_agg.append({"run_id": rid, "exit_bars": eb, **stats})

    # Aggregate lock / retrace
    lock_agg: List[Dict[str, Any]] = []
    for rid in RUN_IDS:
        for rule in PROFIT_LOCK_RULES:
            rn = rule["rule_name"]
            profits = [
                r["profit"]
                for r in lock_rows
                if r["run_id"] == rid and r["rule_name"] == rn and not math.isnan(r["profit"])
            ]
            extra_tp = sum(
                1
                for r in lock_rows
                if r["run_id"] == rid and r["rule_name"] == rn and r.get("exit_kind") == "tp"
            )
            extra_sl = sum(
                1
                for r in lock_rows
                if r["run_id"] == rid and r["rule_name"] == rn and r.get("exit_kind") == "sl"
            )
            extra_time = sum(
                1
                for r in lock_rows
                if r["run_id"] == rid and r["rule_name"] == rn and r.get("exit_kind") == "time"
            )
            bars_list = [
                int(r["bars_held"])
                for r in lock_rows
                if r["run_id"] == rid and r["rule_name"] == rn and int(r["bars_held"]) > 0
            ]
            stats = summarize_profits(profits)
            stats.update(
                {
                    "tp_hit_count": extra_tp,
                    "sl_hit_count": extra_sl,
                    "time_exit_count": extra_time,
                    "avg_bars_held": float(np.mean(bars_list)) if bars_list else math.nan,
                    "median_bars_held": float(np.median(bars_list)) if bars_list else math.nan,
                }
            )
            lock_agg.append({"run_id": rid, "rule_name": rn, **stats})

    retrace_agg: List[Dict[str, Any]] = []
    for rid in RUN_IDS:
        for rule in RETRACE_RULES:
            rn = rule["rule_name"]
            profits = [
                r["profit"]
                for r in retrace_rows
                if r["run_id"] == rid and r["rule_name"] == rn and not math.isnan(r["profit"])
            ]
            re_ct = sum(
                1
                for r in retrace_rows
                if r["run_id"] == rid and r["rule_name"] == rn and r.get("exit_kind") == "retrace"
            )
            te_ct = sum(
                1
                for r in retrace_rows
                if r["run_id"] == rid and r["rule_name"] == rn and r.get("exit_kind") == "time"
            )
            bars_list = [
                int(r["bars_held"])
                for r in retrace_rows
                if r["run_id"] == rid and r["rule_name"] == rn and int(r["bars_held"]) > 0
            ]
            stats = summarize_profits(profits)
            stats.update(
                {
                    "retrace_exit_count": re_ct,
                    "time_exit_count": te_ct,
                    "avg_bars_held": float(np.mean(bars_list)) if bars_list else math.nan,
                    "median_bars_held": float(np.median(bars_list)) if bars_list else math.nan,
                }
            )
            retrace_agg.append({"run_id": rid, "rule_name": rn, **stats})

    # Best rules per category for latest run (by total_profit)
    def best_in(rows: List[Dict[str, Any]], key_fn) -> Optional[Dict[str, Any]]:
        sub = [x for x in rows if x["run_id"] == LATEST_RUN]
        if not sub:
            return None
        return max(sub, key=key_fn)

    best_fixed = best_in(fixed_agg, lambda x: x["total_profit"])
    best_lock = best_in(lock_agg, lambda x: x["total_profit"])
    best_retr = best_in(retrace_agg, lambda x: x["total_profit"])

    # original metrics from log for latest (not OHLCV fixed-12)
    latest_trades = all_trades[LATEST_RUN]
    orig_lp = [t["baseline_raw_profit"] for t in latest_trades if not math.isnan(t["baseline_raw_profit"])]
    orig_stats = summarize_profits(orig_lp)

    # Step 7 detail rows
    detail_rows: List[Dict[str, Any]] = []
    imp_sum = 0.0
    loss_to_win = 0
    win_to_loss = 0
    loss_less = 0
    worsen = 0
    for t in latest_trades:
        tid = int(t["trade_id"])
        sim = per_trade_sim[LATEST_RUN][tid]
        op = t["baseline_raw_profit"]
        bp = sim["best_alt_profit"]
        imp = bp - op if not math.isnan(bp) and not math.isnan(op) else math.nan
        if not math.isnan(imp):
            imp_sum += imp
            if op < 0 and bp > 0:
                loss_to_win += 1
            elif op > 0 and bp < 0:
                win_to_loss += 1
            elif op < 0 and bp < 0 and bp > op:
                loss_less += 1
            elif imp < 0:
                worsen += 1
        ow = op > 0
        bw = bp > 0 if not math.isnan(bp) else False
        detail_rows.append(
            {
                "trade_id": tid,
                "entry_ts": pd.Timestamp(t["entry_ts"]).isoformat(),
                "direction": t["direction"],
                "log_realized_profit": t["original_profit"],
                "baseline_raw_profit_at_logged_exit": op,
                "best_rule_profit": bp,
                "improvement": imp,
                "original_win_raw_baseline": ow,
                "best_rule_win": bw,
                "log_original_win": bool(float(t["original_profit"]) > 0) if not math.isnan(t["original_profit"]) else False,
                "MFE": sim["mfe"],
                "MAE": sim["mae"],
                "best_rule_exit_type": sim["best_alt_name"],
                "best_rule_bars_held": sim["best_alt_bars_hint"],
            }
        )

    # Trade set classification
    keys_exit = {rid: {trade_key(t, True) for t in all_trades[rid]} for rid in RUN_IDS}
    keys_entry_only = {rid: {trade_key(t, False) for t in all_trades[rid]} for rid in RUN_IDS}
    k_latest = keys_exit[LATEST_RUN]
    k_prev_union = keys_exit[RUN_IDS[0]] | keys_exit[RUN_IDS[1]]
    common_exit = k_latest & keys_exit[RUN_IDS[0]] & keys_exit[RUN_IDS[1]]
    latest_only_exit = k_latest - k_prev_union
    previous_only_exit = k_prev_union - k_latest

    def subset_improvement(keyset: set, use_exit_key: bool) -> Dict[str, float]:
        """common / latest_only 서브셋에서 best-original 합."""
        s = 0.0
        n = 0
        for t in latest_trades:
            k = trade_key(t, use_exit_key)
            if k not in keyset:
                continue
            tid = int(t["trade_id"])
            sim = per_trade_sim[LATEST_RUN][tid]
            op = t["baseline_raw_profit"]
            bp = sim["best_alt_profit"]
            if math.isnan(op) or math.isnan(bp):
                continue
            s += bp - op
            n += 1
        return {"n": float(n), "net_improvement": float(s)}

    sub_common = subset_improvement(common_exit, True)
    sub_latest_only = subset_improvement(latest_only_exit, True)

    # latest_only 10 — 손실 개선 여부 요약
    lo_trades = [t for t in latest_trades if trade_key(t, True) in latest_only_exit]
    lo_imp = []
    for t in lo_trades:
        tid = int(t["trade_id"])
        sim = per_trade_sim[LATEST_RUN][tid]
        op = t["baseline_raw_profit"]
        bp = sim["best_alt_profit"]
        if not math.isnan(op) and not math.isnan(bp):
            lo_imp.append(bp - op)

    # Verdict
    def improves_vs_original(alt: Dict[str, Any]) -> Tuple[bool, bool, bool]:
        if not alt:
            return False, False, False
        tp = alt["total_profit"] > orig_stats["total_profit"]
        wr = alt["win_rate"] > orig_stats["win_rate"]
        mp = alt["mean_profit"] > orig_stats["mean_profit"]
        return tp, wr, mp

    best_overall = None
    best_label = ""
    candidates = [best_fixed, best_lock, best_retr]
    candidates = [c for c in candidates if c is not None]
    if candidates:
        best_overall = max(candidates, key=lambda x: x["total_profit"])
        if best_overall is best_fixed:
            best_label = f"fixed_{int(best_overall['exit_bars'])}"
        else:
            best_label = str(best_overall.get("rule_name", ""))

    imp_flags = improves_vs_original(best_overall) if best_overall else (False, False, False)
    imp_ct = sum(1 for x in imp_flags if x)

    if best_overall:
        still_neg = best_overall["total_profit"] < 0 and orig_stats["total_profit"] < 0
    else:
        still_neg = True

    lo_avg_imp = float(np.mean(lo_imp)) if lo_imp else math.nan
    lo_neg_after_best = sum(
        1
        for t in lo_trades
        if not math.isnan(per_trade_sim[LATEST_RUN][int(t["trade_id"])]["best_alt_profit"])
        and per_trade_sim[LATEST_RUN][int(t["trade_id"])]["best_alt_profit"] < 0
    )

    if imp_ct >= 2 and LATEST_RUN in [r["run_id"] for r in fixed_agg]:
        verdict = "A. exit 문제 강함"
        verdict_code = "A"
    elif (imp_ct >= 1 or imp_sum > 0) and still_neg:
        verdict = "B. exit 문제 일부 있음"
        verdict_code = "B"
    else:
        verdict = "C. exit만으로 해결 어려움"
        verdict_code = "C"

    # CSV outputs
    def write_csv(name: str, rows: List[Dict[str, Any]]) -> Path:
        p = OUT_DIR / f"{name}_{stamp}.csv"
        if not rows:
            p.write_text("", encoding="utf-8")
            return p
        keys = list(rows[0].keys())
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
        return p

    p_fixed_cmp = write_csv("exit_fixed_bars_comparison", fixed_agg)
    p_lock_cmp = write_csv("exit_profit_lock_comparison", lock_agg)
    p_ret_cmp = write_csv("exit_retracement_comparison", retrace_agg)
    p_detail = write_csv("latest_trade_exit_detail", detail_rows)

    # Markdown report
    lines: List[str] = []
    lines.append("# Can_bit Exit Validation Report")
    lines.append("")
    lines.append(f"- 생성 시각(로컬): {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- 산출 스탬프: `{stamp}`")
    if proxy_note:
        lines.append(
            "- **주의**: 일부 trade의 `entry_ts`가 OHLCV에 없어 해당 trade 시뮬레이션은 `계산 불가`로 처리됨. 이 경우 결과는 **proxy simulation 제한**."
        )
    lines.append("")
    lines.append("## 1. 목적")
    lines.append("")
    lines.append(
        "- 동일 entry set(동일 `entry_ts`, `entry_price`, `direction`)에서 **exit 규칙만** 바꿨을 때 "
        "`total_profit`, `win_rate`, `mean_profit` 등이 개선되는지 검증한다."
    )
    lines.append("")
    lines.append("## 2. 대상 run 무결성")
    lines.append("")
    lines.append("| run_id | summary_total_trades | reconstructed | match |")
    lines.append("| --- | --- | --- | --- |")
    for rid in RUN_IDS:
        sn = load_summary_trades(rid)
        rc = len(all_trades[rid])
        ok = sn == rc
        lines.append(f"| {rid} | {sn} | {rc} | {'yes' if ok else 'mismatch'} |")
    lines.append("")
    for rid in RUN_IDS:
        lines.append(f"### 로그: `{rid}`")
        for m in run_logs[rid][:20]:
            lines.append(f"- {m}")
        if len(run_logs[rid]) > 20:
            lines.append(f"- ... ({len(run_logs[rid]) - 20} more)")
        lines.append("")

    lines.append("## 3. Original vs Fixed Bars Exit")
    lines.append("")
    lines.append(
        "| run_id | exit_bars | trade_count | win_rate | total_profit | mean_profit | max_loss |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in fixed_agg:
        lines.append(
            f"| {r['run_id']} | {r['exit_bars']} | {r['trade_count']} | {r['win_rate']:.6f} | "
            f"{r['total_profit']:.8f} | {r['mean_profit']:.8f} | {r['max_loss']:.8f} |"
        )
    lines.append("")

    lines.append("## 4. Original vs Profit Lock Exit")
    lines.append("")
    lines.append(
        "| run_id | rule_name | win_rate | total_profit | mean_profit | max_loss | tp_hit | sl_hit | time_exit | avg_bars |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in lock_agg:
        lines.append(
            f"| {r['run_id']} | {r['rule_name']} | {r['win_rate']:.6f} | {r['total_profit']:.8f} | "
            f"{r['mean_profit']:.8f} | {r['max_loss']:.8f} | {r['tp_hit_count']} | {r['sl_hit_count']} | "
            f"{r['time_exit_count']} | {r['avg_bars_held']:.4f} |"
        )
    lines.append("")

    lines.append("## 5. Original vs Retracement Exit")
    lines.append("")
    lines.append(
        "| run_id | rule_name | win_rate | total_profit | mean_profit | max_loss | retrace_exit | time_exit | avg_bars |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in retrace_agg:
        lines.append(
            f"| {r['run_id']} | {r['rule_name']} | {r['win_rate']:.6f} | {r['total_profit']:.8f} | "
            f"{r['mean_profit']:.8f} | {r['max_loss']:.8f} | {r['retrace_exit_count']} | {r['time_exit_count']} | "
            f"{r['avg_bars_held']:.4f} |"
        )
    lines.append("")

    lines.append("## 6. 최신 run 상세")
    lines.append("")
    lines.append(f"- run_id: `{LATEST_RUN}`")
    lines.append(
        "- **비교 단위**: 대안 exit 시뮬레이션은 OHLCV 종가 기준 **비용 미반영 raw 수익률**이다. "
        "동일 단위로 `baseline_raw_profit`(로그의 entry/exit 가격으로 계산한 raw)을 baseline으로 사용한다. "
        "로그 필드 `realized_profit` 합은 수수료·스케일 등으로 raw 합과 다를 수 있으며, "
        "`monitor_guard_stage2_summary` 의 `total_return` 은 trade 평균 기반으로 raw 합과 일치하지 않을 수 있다."
    )
    lines.append(
        f"- **baseline (logged exit, raw)**: win_rate={orig_stats['win_rate']:.6f}, "
        f"total_profit={orig_stats['total_profit']:.8f}, mean_profit={orig_stats['mean_profit']:.8f}"
    )
    if best_overall:
        lines.append(
            f"- **best alternative (전체 후보 중 total_profit 최대)**: `{best_label}` → "
            f"win_rate={best_overall['win_rate']:.6f}, total_profit={best_overall['total_profit']:.8f}, "
            f"mean_profit={best_overall['mean_profit']:.8f}"
        )
        lines.append(
            f"- **대비 개선 여부 (total_profit, win_rate, mean_profit)**: "
            f"{imp_flags[0]}, {imp_flags[1]}, {imp_flags[2]} (개선 개수={imp_ct})"
        )
    else:
        lines.append("- **best alternative**: 계산 불가(후보 없음)")
    lines.append(f"- **net improvement (trade별 best−original 합)**: {imp_sum:.8f}")
    lines.append(f"- 손실→승리 전환 trade 수: {loss_to_win}")
    lines.append(f"- 손실 폭 축소(op<0, bp>op) trade 수: {loss_less}")
    lines.append(f"- 승리→손실 전환 trade 수: {win_to_loss}")
    lines.append(f"- 악화(개선<0) trade 수: {worsen}")
    lines.append("")
    lines.append("### 집합 분석 (최신 run 기준 키: direction+entry_ts+exit_ts)")
    lines.append("")
    lines.append(f"- common_trade 수: {len(common_exit)}")
    lines.append(f"- latest_only_trade 수: {len(latest_only_exit)}")
    lines.append(f"- previous_only_trade 수: {len(previous_only_exit)}")
    lines.append(
        f"- common에서 exit 대안 net_improvement 합: {sub_common['net_improvement']:.8f} (n={int(sub_common['n'])})"
    )
    lines.append(
        f"- latest_only에서 exit 대안 net_improvement 합: {sub_latest_only['net_improvement']:.8f} (n={int(sub_latest_only['n'])})"
    )
    if math.isnan(lo_avg_imp):
        lines.append("- latest_only 평균 개선: 계산 불가")
    else:
        lines.append(f"- latest_only 평균 개선(best−original): {lo_avg_imp:.8f}")
    lines.append(
        f"- latest_only 중 best exit 이후에도 음수인 trade 수: {lo_neg_after_best} / {len(lo_trades)}"
    )
    lines.append("")

    lines.append("## 7. 판정")
    lines.append("")
    lines.append(f"- **판정**: {verdict}")
    lines.append(
        f"- 근거 요약: 최신 run 대비 개선 지표 개수={imp_ct}, best 대안 total_profit 부호·크기, "
        f"latest_only 서브셋 음수 잔존 비율 등을 숫자로 종합."
    )
    lines.append("")

    lines.append("## 8. 다음 액션 제안 (코드 변경 없음)")
    lines.append("")
    lines.append("- 고정 bars 후보: 상기 표에서 total_profit이 큰 exit_bars 검토.")
    lines.append("- profit lock 후보: 동일.")
    lines.append("- retracement 후보: 동일.")
    lines.append("- 운영 로그에 `exit_reason`, intrabar TP/SL 충돌 여부를 남기면 본 분석과 교차검증 가능.")
    lines.append("- 본 분석은 가격 경로 단순화 모델이므로, entry selection 필터와 병행 여부는 별도 실험 필요.")
    lines.append("")

    lines.append("## 부록: 확인된 사실 vs 추측")
    lines.append("")
    lines.append(
        "- **확인된 사실**: jsonl에서 ENTRY/EXIT 매칭, summary trade 수, OHLCV 정렬·중복 제거, "
        "대안 exit 규칙에 따른 통계 산출."
    )
    lines.append(
        "- **추측 아님/주의**: 로그 `realized_profit`과 순수 가격비(`directional_raw_profit`) 차이는 "
        "수수료·스케일·슬리피지 등으로 설명될 수 있음 — 본 스크립트는 차이를 `reproduce` 로그로만 기록."
    )
    lines.append("")

    report_path = OUT_DIR / f"exit_validation_report_{stamp}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Console summary
    def impr(orig: Dict[str, Any], alt: Optional[Dict[str, Any]]) -> str:
        if alt is None:
            return "계산 불가"
        return f"{alt['total_profit'] - orig['total_profit']:.8f}"

    print("\n[EXIT VALIDATION SUMMARY]\n")
    print("1. original 최신 run (baseline_raw @ logged exit, OHLCV 시뮬과 동일 단위):")
    print(f"   - win_rate: {orig_stats['win_rate']:.6f}")
    print(f"   - total_profit: {orig_stats['total_profit']:.8f}")
    print(f"   - mean_profit: {orig_stats['mean_profit']:.8f}")
    print("")
    print("2. best fixed bars:")
    if best_fixed:
        print(f"   - rule: fixed_{int(best_fixed['exit_bars'])}")
        print(f"   - win_rate: {best_fixed['win_rate']:.6f}")
        print(f"   - total_profit: {best_fixed['total_profit']:.8f}")
        print(f"   - improvement vs original: {impr(orig_stats, best_fixed)}")
    else:
        print("   - 계산 불가")
    print("")
    print("3. best profit lock:")
    if best_lock:
        print(f"   - rule: {best_lock['rule_name']}")
        print(f"   - win_rate: {best_lock['win_rate']:.6f}")
        print(f"   - total_profit: {best_lock['total_profit']:.8f}")
        print(f"   - improvement vs original: {impr(orig_stats, best_lock)}")
    else:
        print("   - 계산 불가")
    print("")
    print("4. best retracement:")
    if best_retr:
        print(f"   - rule: {best_retr['rule_name']}")
        print(f"   - win_rate: {best_retr['win_rate']:.6f}")
        print(f"   - total_profit: {best_retr['total_profit']:.8f}")
        print(f"   - improvement vs original: {impr(orig_stats, best_retr)}")
    else:
        print("   - 계산 불가")
    print("")
    print("5. 최종 판정:")
    print(f"   - {verdict}")
    print("")
    print("6. 생성 파일:")
    print(f"   - {report_path}")
    print(f"   - {p_fixed_cmp}")
    print(f"   - {p_lock_cmp}")
    print(f"   - {p_ret_cmp}")
    print(f"   - {p_detail}")
    print("")


if __name__ == "__main__":
    main()
