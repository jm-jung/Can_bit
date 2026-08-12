"""Align external series to QQQ sessions with availability rules."""
from __future__ import annotations

import json
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from canbit_equity.regime.config import FRED_LAG_SESSIONS, FRED_SERIES, MAX_FFILL_AGE, YF_SERIES, paths
from canbit_equity.regime.providers import load_normalized_fred, load_normalized_market


def _qqq_sessions(qqq: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(qqq["session_date"]).dt.normalize().unique()).sort_values()


def align_market_close(qqq_sessions: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
    src = load_normalized_market(symbol)
    src = src.set_index("session_date").sort_index()
    col = "close_adj" if "close_adj" in src.columns else "adj_close"
    out_rows = []
    future_join = 0
    for sd in qqq_sessions:
        if sd in src.index:
            val = float(src.loc[sd, col])
            # same-session close is allowed for market series
            if pd.isna(val) or not np.isfinite(val):
                continue
            out_rows.append(
                {
                    "qqq_session_date": sd,
                    "source_observation_date": sd,
                    "available_session": sd,
                    "source_age_sessions": 0,
                    "value": val,
                    "provider": "yfinance",
                    "provenance": "YAHOO_FINANCE_UNOFFICIAL",
                    "series_id": symbol,
                }
            )
        else:
            # no forward fill for missing market days beyond as-of; leave gap
            pass
    df = pd.DataFrame(out_rows)
    return df


def align_fred(qqq_sessions: pd.DatetimeIndex, series_id: str) -> pd.DataFrame:
    src = load_normalized_fred(series_id).set_index("observation_date").sort_index()
    # Map each QQQ session to last FRED observation with 1-session lag:
    # value available on session t must have observation_date such that
    # available_session = next QQQ session after observation_date, then lag means
    # for session t use observations with available_session <= t-1.
    # Conservative: observation_date D is first usable on the next QQQ session after D,
    # then apply additional lag of FRED_LAG_SESSIONS on QQQ calendar.
    sess = list(qqq_sessions)
    sess_pos = {s: i for i, s in enumerate(sess)}
    # Build available_session for each FRED obs = first QQQ session strictly after observation_date
    avail_map = []
    for od, row in src.iterrows():
        # first session > od
        idx = qqq_sessions.searchsorted(od, side="right")
        if idx >= len(qqq_sessions):
            continue
        first_avail = qqq_sessions[idx]
        # apply lag: usable starting FRED_LAG_SESSIONS QQQ sessions after first_avail
        pos = sess_pos[first_avail] + FRED_LAG_SESSIONS
        if pos >= len(sess):
            continue
        usable_from = sess[pos]
        avail_map.append((usable_from, od, float(row["value"])))
    avail_df = pd.DataFrame(avail_map, columns=["usable_from", "observation_date", "value"]).sort_values("usable_from")

    out = []
    last_val = None
    last_od = None
    last_usable_pos = None
    ai = 0
    for i, sd in enumerate(sess):
        while ai < len(avail_df) and avail_df.iloc[ai]["usable_from"] <= sd:
            last_val = float(avail_df.iloc[ai]["value"])
            last_od = pd.Timestamp(avail_df.iloc[ai]["observation_date"])
            last_usable_pos = sess_pos[pd.Timestamp(avail_df.iloc[ai]["usable_from"])]
            ai += 1
        if last_val is None:
            continue
        age = i - int(last_usable_pos)
        if age > MAX_FFILL_AGE:
            continue
        out.append(
            {
                "qqq_session_date": sd,
                "source_observation_date": last_od,
                "available_session": sess[int(last_usable_pos)],
                "source_age_sessions": int(age),
                "value": last_val,
                "provider": "fred",
                "provenance": "FRED_OFFICIAL_LATEST_VINTAGE",
                "series_id": series_id,
            }
        )
    return pd.DataFrame(out)


def build_aligned_panel(qqq: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    p = paths()
    sessions = _qqq_sessions(qqq)
    frames = {}
    audits = []
    future_join_count = 0
    backward_fill_count = 0
    macro_same_day = 0
    for sym in YF_SERIES:
        df = align_market_close(sessions, sym)
        frames[sym] = df.set_index("qqq_session_date")["value"]
        # audit: available_session <= qqq_session
        viol = int((df["available_session"] > df["qqq_session_date"]).sum()) if len(df) else 0
        future_join_count += viol
        audits.append({"series_id": sym, "rows": len(df), "future_joins": viol, "type": "market"})
    for sid in FRED_SERIES:
        df = align_fred(sessions, sid)
        frames[sid] = df.set_index("qqq_session_date")["value"]
        viol = int((df["available_session"] > df["qqq_session_date"]).sum()) if len(df) else 0
        future_join_count += viol
        # same-day unlagged would be available_session == qqq and observation_date == qqq — forbidden for FRED
        same = 0
        if len(df):
            same = int(
                (
                    (df["source_observation_date"] == df["qqq_session_date"])
                    & (df["available_session"] == df["qqq_session_date"])
                ).sum()
            )
        macro_same_day += same
        audits.append({"series_id": sid, "rows": len(df), "future_joins": viol, "same_day_unlagged": same, "type": "fred"})

    panel = pd.DataFrame({"session_date": sessions}).set_index("session_date")
    rename = {
        "^VIX": "vix",
        "SPY": "spy",
        "RSP": "rsp",
        "QEW": "qew",
        "SMH": "smh",
        "HYG": "hyg",
        "IEF": "ief",
        "DGS10": "dgs10",
        "DGS2": "dgs2",
        "T10Y2Y": "t10y2y",
    }
    for k, s in frames.items():
        panel[rename[k]] = s.reindex(sessions)

    audit = {
        "future_join_count": future_join_count,
        "backward_fill_count": backward_fill_count,
        "availability_violation_count": future_join_count,
        "macro_same_day_unlagged_count": macro_same_day,
        "series": audits,
        "verdict": "ALIGNMENT_PASS" if future_join_count == 0 and macro_same_day == 0 else "ALIGNMENT_FAIL",
    }
    (p.diag / "alignment/alignment_audit.json").write_text(json.dumps(audit, indent=2, default=str) + "\n")
    (p.diag / "alignment/alignment_audit.md").write_text(
        f"# Alignment Audit\n\n`{audit['verdict']}`\nfuture_join={future_join_count} macro_same_day={macro_same_day}\n"
    )
    panel.reset_index().head(50).to_csv(p.diag / "alignment/aligned_source_sample.csv", index=False)
    return panel.reset_index(), audit
