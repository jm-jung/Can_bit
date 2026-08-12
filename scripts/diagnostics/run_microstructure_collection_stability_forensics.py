#!/usr/bin/env python3
"""Read-only collection stability timeline forensics."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

OUT = Path("data/diagnostics/microstructure_collection_stability_forensics")
LIVE = Path("data/diagnostics/new_market_microstructure_data_pipeline/live/normalized")
WATCHDOG = Path("data/diagnostics/microstructure_ws_self_healing/logs/ws_watchdog_events.jsonl")
GAP = Path("data/diagnostics/microstructure_gap_backfill/data/gap_ledger.parquet")
PMSET = OUT / "timeline/pmset_sleep_wake.log"

START = pd.Timestamp("2026-07-21T00:00:00", tz="UTC")
NOW = pd.Timestamp.now(tz="UTC")


def parse_ts(v):
    if v is None:
        return None
    t = pd.Timestamp(v)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def cycle_key(e):
    return (e.get("collector_instance_id"), e.get("connection_generation"))


def scan_kline_day(day: str):
    p = LIVE / f"ws_kline_1m/symbol=BTCUSDT/date={day}/events.jsonl"
    mins = set()
    if not p.exists():
        return mins, 0, None, None, 0, None
    n = 0
    first = last = None
    with p.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            if o.get("is_closed") is False:
                continue
            ts = parse_ts(o.get("open_time") or o.get("event_ts"))
            if ts is None:
                continue
            mins.add(ts.floor("min"))
            n += 1
            first = ts if first is None else min(first, ts)
            last = ts if last is None else max(last, ts)
    st = p.stat()
    return mins, n, first, last, st.st_size, pd.Timestamp(st.st_mtime, unit="s", tz="UTC")


def parse_pmset_intervals(path: Path):
    """Return list of (sleep_start_utc, sleep_end_utc) from pmset -g log."""
    if not path.exists():
        return []
    # Lines look like: 2026-07-28 13:36:01 +0900 Sleep ... 1038 secs
    sleep_pat = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) [+-]\d{4}\s+(Sleep|DarkWake|Wake)\b.*?(\d+)\s+secs?"
    )
    # DarkWake/Wake lines may not have duration in same way
    event_pat = re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) ([+-]\d{4})\s+(Sleep|DarkWake|Wake)\b"
    )
    dur_pat = re.compile(r"(\d+)\s+secs")
    events = []
    for line in path.read_text(errors="ignore").splitlines():
        m = event_pat.match(line)
        if not m:
            continue
        local = pd.Timestamp(f"{m.group(1)} {m.group(2)}")
        utc = local.tz_convert("UTC")
        kind = m.group(3)
        dur_m = dur_pat.search(line)
        dur = int(dur_m.group(1)) if dur_m else None
        events.append((utc, kind, dur, line[:120]))
    intervals = []
    i = 0
    while i < len(events):
        ts, kind, dur, _ = events[i]
        if kind == "Sleep" and dur:
            intervals.append((ts, ts + pd.Timedelta(seconds=dur)))
        i += 1
    return intervals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "timeline").mkdir(exist_ok=True)
    (OUT / "root_cause").mkdir(exist_ok=True)

    events = []
    with WATCHDOG.open() as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            ts = parse_ts(o.get("event_utc"))
            if ts is None or ts < START:
                continue
            o["_ts"] = ts
            events.append(o)
    print(f"watchdog events: {len(events)}")

    days = pd.date_range(START.floor("D"), NOW.floor("D"), freq="D", tz="UTC")
    daily_rows = []
    all_kline = set()
    for d in days:
        day = d.strftime("%Y-%m-%d")
        end = min(d + pd.Timedelta(days=1), NOW)
        cal = (
            list(pd.date_range(d, end - pd.Timedelta(minutes=1), freq="1min", tz="UTC"))
            if end > d
            else []
        )
        mins, n, first, last, fsize, fmtime = scan_kline_day(day)
        mins = {m for m in mins if d <= m < end}
        all_kline |= mins
        day_ev = [e for e in events if d <= e["_ts"] < end]
        live = [e for e in day_ev if e["event_type"] == "LIVE_RECOVERY_COMPLETE"]
        hard = [e for e in day_ev if e["event_type"] == "HARD_STALE_DETECTED"]
        req = [e for e in day_ev if e["event_type"] == "RECONNECT_REQUESTED"]
        types = Counter(e["event_type"] for e in day_ev)
        daily_rows.append(
            {
                "date_utc": day,
                "calendar_minutes": len(cal),
                "is_partial": bool(end < d + pd.Timedelta(days=1)),
                "closed_kline_minutes": len(mins),
                "strict_coverage_pct": round(100 * len(mins) / max(len(cal), 1), 4),
                "kline_rows": n,
                "first_kline": str(first) if first else None,
                "last_kline": str(last) if last else None,
                "file_size_bytes": fsize,
                "file_mtime": str(fmtime) if fmtime else None,
                "watchdog_events": len(day_ev),
                "hard_stale_incidents": len({cycle_key(e) for e in hard}),
                "reconnect_requests": len({cycle_key(e) for e in req}),
                "completed_self_heal_cycles": len({cycle_key(e) for e in live}),
                "live_recovery_events": len(live),
                "unique_instances": len({e.get("collector_instance_id") for e in day_ev}),
                "max_connection_generation": max(
                    (e.get("connection_generation") or 0) for e in day_ev
                )
                if day_ev
                else 0,
                "event_type_top": json.dumps(dict(types.most_common(8))),
            }
        )

    pd.DataFrame(daily_rows).to_csv(OUT / "timeline/daily_collection_stability.csv", index=False)
    print(pd.DataFrame(daily_rows)[
        [
            "date_utc",
            "calendar_minutes",
            "closed_kline_minutes",
            "strict_coverage_pct",
            "hard_stale_incidents",
            "completed_self_heal_cycles",
            "reconnect_requests",
        ]
    ].to_string(index=False))

    # reconnect incidents
    live_by_gen = defaultdict(list)
    for e in events:
        if e["event_type"] == "LIVE_RECOVERY_COMPLETE":
            live_by_gen[cycle_key(e)].append(e)
    incidents = []
    for e in sorted(
        [e for e in events if e["event_type"] == "RECONNECT_REQUESTED"], key=lambda x: x["_ts"]
    ):
        k = cycle_key(e)
        lives = live_by_gen.get(k, [])
        recover_ts = min(x["_ts"] for x in lives) if lives else None
        incidents.append(
            {
                "incident_key": f"{k[0]}|{k[1]}",
                "collector_instance_id": k[0],
                "connection_generation": k[1],
                "reconnect_requested_at": e["_ts"].isoformat(),
                "trigger_reason": e.get("trigger_reason"),
                "trigger_streams": json.dumps(e.get("trigger_streams")),
                "completed": bool(lives),
                "first_live_recovery_at": recover_ts.isoformat() if recover_ts else None,
                "live_recovery_event_count": len(lives),
                "recovery_latency_seconds": (recover_ts - e["_ts"]).total_seconds()
                if recover_ts
                else None,
            }
        )
    pd.DataFrame(incidents).to_csv(OUT / "timeline/reconnect_incidents.csv", index=False)

    # healthy sessions
    hard_events = sorted(
        [e for e in events if e["event_type"] == "HARD_STALE_DETECTED"], key=lambda x: x["_ts"]
    )
    sessions = []
    for k, lives in live_by_gen.items():
        t0 = min(x["_ts"] for x in lives)
        nxt = next((h for h in hard_events if h["_ts"] > t0), None)
        end = nxt["_ts"] if nxt else NOW
        sessions.append(
            {
                "start_utc": t0.isoformat(),
                "end_utc": end.isoformat(),
                "duration_seconds": (end - t0).total_seconds(),
                "ended_by_hard_stale": nxt is not None,
                "instance": k[0],
                "generation": k[1],
            }
        )
    # also FIRST_LIVE
    for e in events:
        if e["event_type"] != "FIRST_LIVE_EVENT_RECEIVED":
            continue
        t0 = e["_ts"]
        nxt = next((h for h in hard_events if h["_ts"] > t0), None)
        end = nxt["_ts"] if nxt else NOW
        sessions.append(
            {
                "start_utc": t0.isoformat(),
                "end_utc": end.isoformat(),
                "duration_seconds": (end - t0).total_seconds(),
                "ended_by_hard_stale": nxt is not None,
                "instance": e.get("collector_instance_id"),
                "generation": e.get("connection_generation"),
            }
        )
    sdf = pd.DataFrame(sessions)
    if not sdf.empty:
        sdf = sdf.sort_values("start_utc").drop_duplicates(
            subset=["instance", "generation", "start_utc"], keep="first"
        )
    sdf.to_csv(OUT / "timeline/healthy_session_durations.csv", index=False)
    if len(sdf):
        durs = sdf["duration_seconds"]
        print(
            "healthy sessions",
            len(sdf),
            "median",
            float(durs.median()),
            "p10",
            float(durs.quantile(0.1)),
            "p90",
            float(durs.quantile(0.9)),
            "max",
            float(durs.max()),
        )
        print(
            "sessions <60s",
            int((durs < 60).sum()),
            "<120s",
            int((durs < 120).sum()),
            "<300s",
            int((durs < 300).sum()),
        )

    # raw gaps
    cal_all = pd.date_range(START, NOW.floor("min") - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
    missing = [t for t in cal_all if t not in all_kline]
    clusters = []
    if missing:
        s = prev = missing[0]
        for t in missing[1:]:
            if t - prev > pd.Timedelta(minutes=1):
                clusters.append((s, prev, (prev - s) / pd.Timedelta(minutes=1) + 1))
                s = t
            prev = t
        clusters.append((s, prev, (prev - s) / pd.Timedelta(minutes=1) + 1))

    gap = pd.read_parquet(GAP) if GAP.exists() else pd.DataFrame()
    if not gap.empty:
        for c in ("gap_start_utc", "gap_end_utc"):
            gap[c] = pd.to_datetime(gap[c], utc=True)
        gap_win = gap[(gap["gap_end_utc"] > START) & (gap["gap_start_utc"] < NOW)]
    else:
        gap_win = gap

    raw_gap_rows = []
    for a, b, m in clusters:
        overlap = 0
        if not gap_win.empty:
            for _, g in gap_win.iterrows():
                if g.gap_end_utc > a and g.gap_start_utc <= b + pd.Timedelta(minutes=1):
                    overlap = 1
                    break
        raw_gap_rows.append(
            {
                "gap_start": a.isoformat(),
                "gap_end": b.isoformat(),
                "minutes": m,
                "ledger_overlap": overlap,
            }
        )
    rg = pd.DataFrame(raw_gap_rows)
    rg.to_csv(OUT / "timeline/raw_gap_vs_ledger.csv", index=False)

    # sleep overlap
    sleep_intervals = parse_pmset_intervals(PMSET)
    sleep_minutes = set()
    for a, b in sleep_intervals:
        if b <= START or a >= NOW:
            continue
        a2 = max(a.floor("min"), START)
        b2 = min(b.floor("min"), NOW.floor("min"))
        if b2 > a2:
            sleep_minutes.update(pd.date_range(a2, b2 - pd.Timedelta(minutes=1), freq="1min", tz="UTC"))
    miss_set = set(missing)
    overlap_sleep = len(miss_set & sleep_minutes)
    outside_sleep = len(miss_set - sleep_minutes)
    print(
        "sleep_intervals",
        len(sleep_intervals),
        "sleep_minutes_in_window",
        len(sleep_minutes),
        "no_raw_overlap_sleep",
        overlap_sleep,
        "no_raw_outside_sleep",
        outside_sleep,
    )

    summary = {
        "audit_start": str(START),
        "audit_end": str(NOW),
        "no_raw_minutes": len(missing),
        "no_raw_hours": round(len(missing) / 60, 2),
        "raw_gap_clusters": len(clusters),
        "raw_gaps_without_ledger": int((rg.ledger_overlap == 0).sum()) if len(rg) else 0,
        "hard_stale_incidents": len(
            {cycle_key(e) for e in events if e["event_type"] == "HARD_STALE_DETECTED"}
        ),
        "reconnect_cycles": len(incidents),
        "completed_self_heals": len(
            {cycle_key(e) for e in events if e["event_type"] == "LIVE_RECOVERY_COMPLETE"}
        ),
        "sleep_intervals_parsed": len(sleep_intervals),
        "sleep_minutes_in_window": len(sleep_minutes),
        "no_raw_minutes_overlapping_host_sleep": overlap_sleep,
        "no_raw_minutes_outside_host_sleep": outside_sleep,
        "sleep_explained_gap_pct": round(100 * overlap_sleep / max(len(missing), 1), 2),
        "non_sleep_gap_pct": round(100 * outside_sleep / max(len(missing), 1), 2),
        "healthy_session_stats": {
            "n": int(len(sdf)),
            "median_s": float(sdf.duration_seconds.median()) if len(sdf) else None,
            "p10_s": float(sdf.duration_seconds.quantile(0.1)) if len(sdf) else None,
            "p90_s": float(sdf.duration_seconds.quantile(0.9)) if len(sdf) else None,
            "max_s": float(sdf.duration_seconds.max()) if len(sdf) else None,
            "pct_under_120s": float((sdf.duration_seconds < 120).mean() * 100) if len(sdf) else None,
        },
        "note_pmset": "pmset -g log is ring buffer; sleep coverage may undercount older days",
    }
    (OUT / "timeline/forensics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    md = [
        "# Recent 7d Collection Stability Timeline",
        "",
        f"- Window: `{START}` → `{NOW}`",
        f"- NO_RAW minutes: {summary['no_raw_minutes']} ({summary['no_raw_hours']}h)",
        f"- Sleep-overlapping NO_RAW: {overlap_sleep} ({summary['sleep_explained_gap_pct']}%)",
        f"- Non-sleep NO_RAW: {outside_sleep} ({summary['non_sleep_gap_pct']}%)",
        f"- Hard stale incidents: {summary['hard_stale_incidents']}",
        f"- Reconnect cycles: {summary['reconnect_cycles']}",
        f"- Completed self-heals (unique gen): {summary['completed_self_heals']}",
        f"- Healthy session median/p10/max s: {summary['healthy_session_stats'].get('median_s')}/"
        f"{summary['healthy_session_stats'].get('p10_s')}/{summary['healthy_session_stats'].get('max_s')}",
        f"- Sessions <120s: {summary['healthy_session_stats'].get('pct_under_120s')}%",
        "",
        "Note: pmset log is a limited ring buffer; older sleep intervals may be missing.",
    ]
    (OUT / "timeline/recent_7d_timeline.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2, default=str)[:2500])


if __name__ == "__main__":
    main()
