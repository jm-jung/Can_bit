"""5/10/20 session outcome maturity — separate from continuous shadow equity."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from canbit_equity.calendar import get_calendar
from canbit_equity.regime.prospective.config import HORIZONS, PRIMARY_HORIZON, PROSP_ROOT


def _forward_sessions_after(cal, signal_session: pd.Timestamp, min_sessions: int = 20) -> pd.DatetimeIndex:
    """Return forward XNYS sessions after signal, clamped to calendar bounds."""
    sd = pd.Timestamp(pd.Timestamp(signal_session).date())
    start = sd + pd.Timedelta(days=1)
    last = pd.Timestamp(pd.Timestamp(cal.last_session).date())
    if start > last:
        return pd.DatetimeIndex([])
    # Enough calendar days for min_sessions trading days, but never past calendar end.
    end = min(sd + pd.Timedelta(days=max(min_sessions * 3, 60)), last)
    return cal.sessions_in_range(start, end)


def mature_outcomes(live: pd.DataFrame, strict: pd.DataFrame, late: pd.DataFrame) -> Dict[str, Any]:
    preds = pd.concat([strict, late], ignore_index=True) if (len(strict) or len(late)) else pd.DataFrame()
    path = PROSP_ROOT / "outcomes/matured_outcomes.parquet"
    if path.exists():
        existing = pd.read_parquet(path)
    else:
        existing = pd.DataFrame(columns=["signal_session", "horizon"])

    if len(preds) == 0:
        if not path.exists():
            existing.to_parquet(path, index=False)
        return {"status": "NO_PREDICTIONS", "new": 0, "total": int(len(existing))}

    live = live.copy()
    live["session_date"] = pd.to_datetime(live["session_date"]).dt.normalize()
    live_idx = live.set_index("session_date").sort_index()
    cal = get_calendar("XNYS")
    preds = preds.copy()
    preds["signal_session"] = pd.to_datetime(preds["signal_session"]).dt.normalize()

    existing_keys = set()
    if len(existing):
        for _, r in existing.iterrows():
            existing_keys.add((str(pd.Timestamp(r["signal_session"]).date()), int(r["horizon"])))

    new_rows: List[Dict[str, Any]] = []
    for _, pr in preds.iterrows():
        sd = pd.Timestamp(pr["signal_session"])
        forward = _forward_sessions_after(cal, sd, min_sessions=max(HORIZONS))
        if len(forward) == 0:
            continue
        entry_sess = pd.Timestamp(pd.Timestamp(forward[0]).date())
        for h in HORIZONS:
            key = (str(sd.date()), int(h))
            if key in existing_keys:
                continue
            if len(forward) < h:
                continue
            maturity = pd.Timestamp(pd.Timestamp(forward[h - 1]).date())
            if maturity not in live_idx.index or entry_sess not in live_idx.index:
                continue
            entry_open = float(live_idx.loc[entry_sess, "open_adj"])
            exit_close = float(live_idx.loc[maturity, "close_adj"])
            gross = exit_close / entry_open - 1.0
            net = gross - 2.0 * (5.0 / 10000.0)
            pred_sig = str(pr.get("target_signal") or ("LONG" if float(pr.get("target_position", 0)) == 1 else "FLAT"))
            label_long = int(net > 0)
            decision_correct = int((pred_sig == "LONG" and label_long == 1) or (pred_sig == "FLAT" and label_long == 0))
            cand_ret = net if pred_sig == "LONG" else 0.0
            dual_pos = pr.get("dual_target_position")
            dual_dec = (
                None
                if dual_pos is None or (isinstance(dual_pos, float) and pd.isna(dual_pos))
                else ("LONG" if float(dual_pos) == 1 else "FLAT")
            )
            new_rows.append(
                {
                    "signal_session": sd,
                    "next_entry_session": entry_sess,
                    "horizon": int(h),
                    "maturity_session": maturity,
                    "entry_open_adj": entry_open,
                    "exit_close_adj": exit_close,
                    "market_forward_return_gross": gross,
                    "market_forward_return_net": net,
                    "label_long": label_long,
                    "predicted_probability": float(pr.get("probability_long"))
                    if pr.get("probability_long") is not None
                    else None,
                    "predicted_signal": pred_sig,
                    "decision_correct": decision_correct,
                    "candidate_decision_return": cand_ret,
                    "opportunity_cost": net - cand_ret,
                    "dual_decision": dual_dec,
                    "buy_hold_reference": net,
                    "model_hash": pr.get("model_hash"),
                    "feature_order_hash": pr.get("feature_order_hash"),
                    "provenance_tier": pr.get("provenance_tier"),
                    "outcome_generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "provider_revision_flag": False,
                    "candidate_id": pr.get("candidate_id"),
                    "primary_horizon": h == PRIMARY_HORIZON,
                }
            )

    if new_rows:
        existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        existing.to_parquet(path, index=False)

    matured20 = int((existing["horizon"] == PRIMARY_HORIZON).sum()) if len(existing) else 0
    return {
        "status": "UPDATED",
        "new": len(new_rows),
        "total": int(len(existing)),
        "matured_5d": int((existing["horizon"] == 5).sum()) if len(existing) else 0,
        "matured_10d": int((existing["horizon"] == 10).sum()) if len(existing) else 0,
        "matured_20d": matured20,
    }
