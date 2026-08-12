#!/usr/bin/env python3
"""Read-only live collection stability probe (samples every --interval-sec)."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
STATUS = REPO / "data/diagnostics/new_market_microstructure_data_pipeline/live/status/microstructure_public_collector_status.json"
KLINE_DIR = REPO / "data/diagnostics/new_market_microstructure_data_pipeline/live/normalized/ws_kline_1m/symbol=BTCUSDT"


def read_status():
    if not STATUS.exists():
        return {}
    return json.loads(STATUS.read_text())


def kline_file_stats(day: str | None = None):
    day = day or pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    p = KLINE_DIR / f"date={day}" / "events.jsonl"
    if not p.exists():
        return {"path": str(p), "exists": False, "size": 0, "mtime": None, "closed_rows_tail": 0}
    st = p.stat()
    # count closed rows near end cheaply: last 200 lines
    lines = p.read_text(errors="ignore").splitlines()[-200:]
    closed = 0
    last_ot = None
    for line in lines:
        if not line.strip():
            continue
        o = json.loads(line)
        if o.get("is_closed") is False:
            continue
        closed += 1
        last_ot = o.get("open_time") or o.get("event_ts")
    return {
        "path": str(p),
        "exists": True,
        "size": st.st_size,
        "mtime": pd.Timestamp(st.st_mtime, unit="s", tz="UTC").isoformat(),
        "closed_rows_in_tail200": closed,
        "last_open_time_in_tail": last_ot,
    }


def sample_once():
    now = pd.Timestamp.now(tz="UTC")
    s = read_status()
    kf = kline_file_stats(now.strftime("%Y-%m-%d"))
    return {
        "wall_clock_utc": now.isoformat(),
        "pid": s.get("pid"),
        "collector_instance_id": s.get("collector_instance_id"),
        "connection_generation": s.get("ws_connection_generation"),
        "ws_connection_state": s.get("ws_connection_state"),
        "reconnect_in_progress": s.get("reconnect_in_progress"),
        "health_quorum_pass": s.get("health_quorum_pass"),
        "aggtrade_task_alive": s.get("aggtrade_task_alive"),
        "kline_task_alive": s.get("kline_task_alive"),
        "markprice_task_alive": s.get("markprice_task_alive"),
        "forceorder_task_alive": s.get("forceorder_task_alive"),
        "ws_connection_open": s.get("ws_connection_open"),
        "last_aggtrade_time_utc": s.get("last_aggtrade_time_utc"),
        "last_kline_time_utc": s.get("last_kline_time_utc"),
        "last_markprice_time_utc": s.get("last_markprice_time_utc"),
        "last_oi_poll_time_utc": s.get("last_oi_poll_time_utc"),
        "aggtrade_age_seconds": s.get("aggtrade_age_seconds"),
        "kline_age_seconds": s.get("kline_age_seconds"),
        "markprice_age_seconds": s.get("markprice_age_seconds"),
        "oi_age_seconds": s.get("oi_age_seconds"),
        "total_reconnect_attempts": s.get("total_reconnect_attempts"),
        "total_successful_self_heals": s.get("total_successful_self_heals"),
        "kline_file": kf,
    }


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"verdict": "LIVE_PROBE_UNRESOLVED", "samples": 0}
    df = pd.DataFrame(rows)
    pid_n = df["pid"].nunique(dropna=True)
    gen_n = df["connection_generation"].nunique(dropna=True)
    gen_delta = (
        int(df["connection_generation"].iloc[-1] - df["connection_generation"].iloc[0])
        if df["connection_generation"].notna().all()
        else None
    )
    reconnect_delta = int(
        (df["total_reconnect_attempts"].iloc[-1] or 0) - (df["total_reconnect_attempts"].iloc[0] or 0)
    )
    heal_delta = int(
        (df["total_successful_self_heals"].iloc[-1] or 0) - (df["total_successful_self_heals"].iloc[0] or 0)
    )
    sizes = [r.get("kline_file", {}).get("size") or 0 for r in rows]
    size_growth = sizes[-1] - sizes[0]
    # advancement: kline age mostly low
    ages = pd.to_numeric(df["kline_age_seconds"], errors="coerce")
    fresh_ratio = float((ages <= 120).mean()) if len(ages) else 0.0
    states = df["ws_connection_state"].astype(str)
    healthy_ratio = float((states == "HEALTHY").mean())
    # expected minutes vs unique last_open_time observed in tails is weak; use size growth + fresh
    start = pd.Timestamp(rows[0]["wall_clock_utc"])
    end = pd.Timestamp(rows[-1]["wall_clock_utc"])
    elapsed_min = max((end - start).total_seconds() / 60.0, 1e-9)

    verdict = "LIVE_PROBE_STABLE"
    if reconnect_delta >= 3 or gen_delta and gen_delta >= 3:
        verdict = "LIVE_PROBE_RECONNECT_STORM"
    elif fresh_ratio < 0.7 or size_growth <= 0:
        if healthy_ratio > 0.8 and fresh_ratio < 0.5:
            verdict = "LIVE_PROBE_FALSE_HEALTHY"
        elif size_growth <= 0 and fresh_ratio > 0.7:
            verdict = "LIVE_PROBE_WRITER_STALL"
        else:
            verdict = "LIVE_PROBE_INTERMITTENT"
    elif reconnect_delta == 1 or (gen_delta and gen_delta == 1):
        verdict = "LIVE_PROBE_INTERMITTENT"

    return {
        "label": label,
        "verdict": verdict,
        "samples": len(rows),
        "elapsed_minutes": round(elapsed_min, 2),
        "pid_unique": int(pid_n),
        "pid_start": rows[0].get("pid"),
        "pid_end": rows[-1].get("pid"),
        "generation_unique": int(gen_n),
        "generation_delta": gen_delta,
        "reconnect_attempt_delta": reconnect_delta,
        "self_heal_delta": heal_delta,
        "kline_file_size_growth_bytes": size_growth,
        "kline_age_fresh_ratio_le_120s": round(fresh_ratio, 4),
        "healthy_state_ratio": round(healthy_ratio, 4),
        "final_state": rows[-1].get("ws_connection_state"),
        "final_kline_age_seconds": rows[-1].get("kline_age_seconds"),
        "final_aggtrade_age_seconds": rows[-1].get("aggtrade_age_seconds"),
        "final_markprice_age_seconds": rows[-1].get("markprice_age_seconds"),
        "start_utc": rows[0]["wall_clock_utc"],
        "end_utc": rows[-1]["wall_clock_utc"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration-sec", type=int, default=600)
    p.add_argument("--interval-sec", type=int, default=10)
    p.add_argument("--label", default="pre_fix")
    p.add_argument(
        "--out-dir",
        default="data/diagnostics/microstructure_collection_stability_forensics/live_probe",
    )
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / f"{args.label}_probe.jsonl"
    rows = []
    t_end = time.time() + args.duration_sec
    with jsonl.open("w") as f:
        while time.time() < t_end:
            row = sample_once()
            rows.append(row)
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            time.sleep(args.interval_sec)
    summary = summarize(rows, args.label)
    (out / f"{args.label}_probe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    md = [
        f"# Live Probe ({args.label})",
        "",
        f"**Verdict:** `{summary['verdict']}`",
        "",
        f"- samples: {summary['samples']}",
        f"- elapsed_min: {summary['elapsed_minutes']}",
        f"- pid unique: {summary['pid_unique']} ({summary['pid_start']}→{summary['pid_end']})",
        f"- generation delta: {summary['generation_delta']}",
        f"- reconnect delta: {summary['reconnect_attempt_delta']}",
        f"- kline file growth bytes: {summary['kline_file_size_growth_bytes']}",
        f"- kline fresh ratio: {summary['kline_age_fresh_ratio_le_120s']}",
        f"- healthy ratio: {summary['healthy_state_ratio']}",
    ]
    (out / f"{args.label}_probe_summary.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
