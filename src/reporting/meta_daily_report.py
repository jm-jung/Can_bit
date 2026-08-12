"""
B_with_meta 일일 운영 리포트 생성.

데이터: state_log.csv, meta_state_snapshot.json, logs/trades.log, logs/risk.log
출력: B_WITH_META_DAILY_REPORT_YYYYMMDD.md, (선택) b_with_meta_daily_summary.json
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import csv
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FR2_DIR = PROJECT_ROOT / "data" / "diagnostics" / "fr2"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_LOG_PATH = FR2_DIR / "state_log.csv"
SNAPSHOT_PATH = FR2_DIR / "meta_state_snapshot.json"
METRICS_SNAPSHOT_PATH = FR2_DIR / "meta_metrics_snapshot.json"
TRADES_LOG = LOG_DIR / "trades.log"
RISK_LOG = LOG_DIR / "risk.log"


def _parse_ts(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        s = str(s).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _safe_int(x: Any) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def _read_state_log(path: Path, since: Optional[datetime] = None) -> list[dict]:
    if not path.exists():
        return []
    try:
        rows = []
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ts = _parse_ts(row.get("timestamp", ""))
                if ts is None:
                    continue
                if since is not None and ts < since:
                    continue
                row["_parsed_ts"] = ts
                rows.append(row)
        return rows
    except Exception:
        return []


def _state_column(rows: list[dict]) -> str:
    if not rows:
        return "current_state"
    for c in ("current_state", "state"):
        if c in rows[0]:
            return c
    return "current_state"


def _read_snapshot(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_json_lines(path: Path, since: Optional[datetime] = None, limit: int = 50000) -> list[dict]:
    if not path.exists() or limit <= 0:
        return []
    out = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ts = obj.get("timestamp")
                    if since and ts:
                        dt = _parse_ts(ts)
                        if dt is not None and dt < since:
                            continue
                    out.append(obj)
                    if len(out) >= limit:
                        break
                except json.JSONDecodeError:
                    continue
    except Exception:
        pass
    return out


@dataclass
class DailyReportData:
    report_date: str
    state_log_path: Path
    snapshot_path: Path
    state_log_latest_ts: Optional[str] = None
    snapshot_latest_ts: Optional[str] = None
    state_log_rows: list = field(default_factory=list)
    snapshot: dict = field(default_factory=dict)
    metrics_snapshot: dict = field(default_factory=dict)
    trade_events: list = field(default_factory=list)
    risk_events: list = field(default_factory=list)
    window_start: Optional[datetime] = None
    window_end: Optional[datetime] = None

    @property
    def state_col(self) -> str:
        return _state_column(self.state_log_rows)


def load_report_data(
    report_date: Optional[datetime] = None,
    state_log_path: Optional[Path] = None,
    snapshot_path: Optional[Path] = None,
    lookback_hours: float = 24.0,
) -> DailyReportData:
    report_date = report_date or datetime.now(timezone.utc)
    if report_date.tzinfo is None:
        report_date = report_date.replace(tzinfo=timezone.utc)
    window_end = report_date
    window_start = report_date - timedelta(hours=lookback_hours)

    state_log_path = state_log_path or STATE_LOG_PATH
    snapshot_path = snapshot_path or SNAPSHOT_PATH

    state_log_rows = _read_state_log(state_log_path, since=window_start)
    state_log_latest_ts = None
    if state_log_rows:
        state_log_latest_ts = state_log_rows[-1].get("timestamp", "")

    snapshot = _read_snapshot(snapshot_path)
    snapshot_latest_ts = snapshot.get("last_eval_ts") or snapshot.get("last_transition_ts")

    metrics_snapshot = {}
    try:
        if METRICS_SNAPSHOT_PATH.exists():
            metrics_snapshot = json.loads(METRICS_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    except Exception:
        metrics_snapshot = {}

    trade_events = _read_json_lines(TRADES_LOG, since=window_start)
    risk_events = _read_json_lines(RISK_LOG, since=window_start)

    return DailyReportData(
        report_date=report_date.strftime("%Y-%m-%d"),
        state_log_path=state_log_path,
        snapshot_path=snapshot_path,
        state_log_latest_ts=state_log_latest_ts,
        snapshot_latest_ts=snapshot_latest_ts,
        state_log_rows=state_log_rows,
        snapshot=snapshot,
        metrics_snapshot=metrics_snapshot,
        trade_events=trade_events,
        risk_events=risk_events,
        window_start=window_start,
        window_end=window_end,
    )


def _metric_col(rows: list[dict], *names: str) -> Optional[str]:
    if not rows:
        return None
    for n in names:
        if n in rows[0]:
            return n
    return None


def build_report(data: DailyReportData) -> tuple[str, dict]:
    """Returns (markdown_content, summary_dict)."""
    lines: list[str] = []
    summary: dict[str, Any] = {
        "report_date": data.report_date,
        "current_state": data.snapshot.get("current_state", "REDUCED"),
        "executive_summary": {},
        "log_health": {},
        "meta_metrics_latest": {},
        "meta_metrics_health": {},
        "off_evaluation": {},
        "conclusion": "",
    }

    rows = data.state_log_rows
    state_col = data.state_col
    window_start = data.window_start
    window_end = data.window_end

    # ---- A. Executive Summary ----
    current_state = (data.snapshot.get("current_state") or "REDUCED").upper()
    mult = data.snapshot.get("position_multiplier")
    if mult is None:
        cfg = data.snapshot.get("config") or {}
        if current_state == "FULL":
            mult = cfg.get("full_multiplier", 1.0)
        elif current_state == "REDUCED":
            mult = cfg.get("reduced_multiplier", 0.4)
        else:
            mult = 0.0

    score_latest = alpha_latest = float("nan")
    if rows:
        score_col = _metric_col(rows, "score")
        alpha_col = _metric_col(rows, "alpha_score")
        if score_col:
            score_latest = _safe_float(rows[-1].get(score_col))
        if alpha_col:
            alpha_latest = _safe_float(rows[-1].get(alpha_col))

    # Latest rolling metrics + health (from meta_metrics_snapshot.json)
    meta_metrics_health = data.metrics_snapshot or {}
    metrics_stale_flag = meta_metrics_health.get("metrics_stale_flag")
    metrics_stale_duration = meta_metrics_health.get("stale_duration_minutes")

    latest_row = rows[-1] if rows else {}
    meta_metrics_latest = {
        "cost_on_60d": _safe_float(latest_row.get("cost_on_60d")),
        "cost_on_90d": _safe_float(latest_row.get("cost_on_90d")),
        "alpha_fee_ratio_60d": _safe_float(latest_row.get("alpha_fee_ratio_60d")),
        "alpha_fee_ratio_90d": _safe_float(latest_row.get("alpha_fee_ratio_90d")),
        "score": _safe_float(latest_row.get("score")),
        "alpha_score": _safe_float(latest_row.get("alpha_score")),
        "trades_60d": _safe_int(latest_row.get("trades_60d")),
    }
    summary["meta_metrics_latest"] = meta_metrics_latest
    summary["meta_metrics_health"] = {
        "metrics_stale_flag": metrics_stale_flag,
        "stale_duration_minutes": metrics_stale_duration,
    }

    full_pct = red_pct = off_pct = 0.0
    if rows and state_col in rows[0]:
        total = len(rows)
        full_pct = sum(1 for r in rows if str(r.get(state_col, "")).upper() == "FULL") / total * 100
        red_pct = sum(1 for r in rows if str(r.get(state_col, "")).upper() == "REDUCED") / total * 100
        off_pct = sum(1 for r in rows if str(r.get(state_col, "")).upper() == "OFF") / total * 100

    flow_summary = f"FULL {full_pct:.0f}% / REDUCED {red_pct:.0f}% / OFF {off_pct:.0f}%"
    verdict = "정상"
    if off_pct >= 50 and score_latest == score_latest and score_latest < 0:
        verdict = "위험"
    elif score_latest == score_latest and score_latest < -0.1:
        verdict = "약세"
    elif data.state_log_latest_ts is None or (data.snapshot_latest_ts and data.state_log_latest_ts):
        try:
            if data.state_log_latest_ts:
                lt = _parse_ts(data.state_log_latest_ts)
                if lt and (window_end - lt).total_seconds() > 86400 * 2:
                    verdict = "시스템 문제(로그 지연)"
            if data.snapshot_latest_ts:
                st = _parse_ts(data.snapshot_latest_ts)
                if st and (window_end - st).total_seconds() > 86400 * 2:
                    verdict = "시스템 문제(스냅샷 지연)"
        except Exception:
            pass

    lines.append("# B_with_meta 일일 운영 리포트")
    lines.append(f"\n**리포트 일자**: {data.report_date} (최근 24시간 기준)")
    lines.append("")
    lines.append("## A. Executive Summary (핵심 5줄 요약)")
    lines.append("")
    lines.append(f"- **현재 상태**: {current_state}")
    lines.append(f"- **현재 multiplier**: {mult}")
    lines.append(f"- **최근 24h 상태 흐름**: {flow_summary}")
    lines.append(f"- **score / alpha_score 최신값**: {score_latest:.4f} / {alpha_latest:.4f}")
    if metrics_stale_flag:
        lines.append(f"- **metrics health**: {metrics_stale_flag} (stale {metrics_stale_duration}m)")
    lines.append(f"- **운영 판단**: {verdict}")
    lines.append("")
    summary["executive_summary"] = {
        "current_state": current_state,
        "current_multiplier": mult,
        "flow_summary": flow_summary,
        "score_latest": score_latest,
        "alpha_score_latest": alpha_latest,
        "verdict": verdict,
    }

    # ---- B. 현재 운영 상태 ----
    lines.append("## B. 현재 운영 상태")
    lines.append("")
    lines.append(f"- strategy_name: B_with_meta")
    lines.append(f"- mode: (API에서 확인: GET /strategy/meta-state)")
    lines.append(f"- current_state: {current_state}")
    lines.append(f"- position_multiplier: {mult}")
    lines.append(f"- last_transition_ts: {data.snapshot.get('last_transition_ts', 'N/A')}")
    lines.append(f"- last_eval_ts: {data.snapshot.get('last_eval_ts', 'N/A')}")
    lines.append(f"- transition_reason: {data.snapshot.get('transition_reason', 'N/A')}")
    lines.append("")

    # ---- C. Meta 상태 흐름 (최근 24h) ----
    lines.append("## C. Meta 상태 흐름 (최근 24시간)")
    lines.append("")
    lines.append(f"- FULL 상태 비율: {full_pct:.1f}%")
    lines.append(f"- REDUCED 상태 비율: {red_pct:.1f}%")
    lines.append(f"- OFF 상태 비율: {off_pct:.1f}%")
    lines.append("")

    transitions = 0
    full_to_red = red_to_full = red_to_off = off_to_red = 0
    if rows and state_col in (rows[0] or {}) and len(rows) >= 2:
        for i in range(1, len(rows)):
            prev = str(rows[i - 1].get(state_col, "")).upper()
            cur = str(rows[i].get(state_col, "")).upper()
            if prev != cur:
                transitions += 1
                if prev == "FULL" and cur == "REDUCED":
                    full_to_red += 1
                elif prev == "REDUCED" and cur == "FULL":
                    red_to_full += 1
                elif prev == "REDUCED" and cur == "OFF":
                    red_to_off += 1
                elif prev == "OFF" and cur == "REDUCED":
                    off_to_red += 1

    lines.append(f"- 상태 전환 횟수: {transitions}")
    lines.append(f"- FULL → REDUCED: {full_to_red}")
    lines.append(f"- REDUCED → FULL: {red_to_full}")
    lines.append(f"- REDUCED → OFF: {red_to_off}")
    lines.append(f"- OFF → REDUCED: {off_to_red}")
    lines.append("")
    lines.append("최근 상태 타임라인 (최근 10개):")
    if rows:
        ts_col = _metric_col(rows, "timestamp") or "timestamp"
        for row in rows[-10:]:
            ts = row.get(ts_col, "")
            st = row.get(state_col, "")
            lines.append(f"- {ts} | {st}")
    else:
        lines.append("- (데이터 없음)")
    lines.append("")

    # ---- D. 핵심 지표 ----
    lines.append("## D. 핵심 지표 (Meta 입력, 최근 24h)")
    lines.append("")
    if not rows:
        lines.append("(state_log 없음)")
    else:
        c60 = _metric_col(rows, "cost_on_60d")
        c90 = _metric_col(rows, "cost_on_90d")
        a60 = _metric_col(rows, "alpha_fee_ratio_60d") or _metric_col(rows, "alpha_60d")
        a90 = _metric_col(rows, "alpha_fee_ratio_90d") or _metric_col(rows, "alpha_90d")
        sc = _metric_col(rows, "score")
        al = _metric_col(rows, "alpha_score")
        t60 = _metric_col(rows, "trades_60d")
        for name, col in [
            ("cost_on_60d", c60),
            ("cost_on_90d", c90),
            ("alpha_fee_ratio_60d", a60),
            ("alpha_fee_ratio_90d", a90),
            ("score", sc),
            ("alpha_score", al),
            ("trades_60d", t60),
        ]:
            if col and col in (rows[0] or {}):
                vals = [_safe_float(r.get(col)) for r in rows]
                valid = [v for v in vals if v == v]
                if valid:
                    lines.append(f"- {name}: latest={vals[-1]:.4f}, min={min(valid):.4f}, max={max(valid):.4f}")
            elif name == "trades_60d" and t60 and rows and t60 in rows[0]:
                lines.append(f"- trades_60d (latest): {_safe_int(rows[-1].get(t60))}")
    lines.append("")

    # ---- E. 거래/액션 요약 ----
    lines.append("## E. 거래/액션 요약 (최근 24시간)")
    lines.append("")
    entries = sum(1 for e in data.trade_events if e.get("type") == "entry")
    exits = sum(1 for e in data.trade_events if e.get("type") == "exit")
    blocked_by_meta = sum(
        1 for e in data.risk_events
        if e.get("event") == "blocked" and ("meta" in str(e.get("reason", "")).lower() or "meta_state" in str(e.get("reason", "")))
    )
    reduced_entries = sum(1 for e in data.trade_events if e.get("type") == "entry" and (e.get("meta_state") or "").upper() == "REDUCED")
    full_entries = sum(1 for e in data.trade_events if e.get("type") == "entry" and (e.get("meta_state") or "").upper() == "FULL")
    lines.append(f"- 신규 진입 횟수: {entries}")
    lines.append(f"- 청산 횟수: {exits}")
    lines.append(f"- **blocked_by_meta 횟수**: {blocked_by_meta}")
    lines.append(f"- REDUCED multiplier 적용 진입: {reduced_entries}")
    lines.append(f"- FULL multiplier 적용 진입: {full_entries}")
    lines.append(f"- OFF 상태에서 차단된 진입: (risk.log의 meta 차단 포함 위 수)")
    lines.append(f"- 총 trade 수: {entries + exits}")
    lines.append("")

    # ---- F. 로그 건강도 ----
    lines.append("## F. 로그 건강도 (CRITICAL)")
    lines.append("")
    state_log_ts = data.state_log_latest_ts
    snap_ts = data.snapshot_latest_ts
    metrics_stale_flag = (data.metrics_snapshot or {}).get("metrics_stale_flag")
    stale_duration_minutes = (data.metrics_snapshot or {}).get("stale_duration_minutes")
    lines.append(f"- state_log.csv 최신 timestamp: {state_log_ts or 'N/A'}")
    lines.append(f"- snapshot 최신 timestamp: {snap_ts or 'N/A'}")
    if metrics_stale_flag:
        lines.append(f"- metrics health: {metrics_stale_flag} (stale {stale_duration_minutes}m)")

    log_verdict = "정상"
    if not state_log_ts and not snap_ts:
        log_verdict = "중단"
    elif window_end:
        try:
            for label, ts in [("state_log", state_log_ts), ("snapshot", snap_ts)]:
                if ts:
                    dt = _parse_ts(ts)
                    if dt and (window_end - dt).total_seconds() > 86400 * 1.5:
                        log_verdict = "지연"
                        break
        except Exception:
            pass
    lines.append(f"- 최근 24h 로그 append: {log_verdict}")
    lines.append(f"- **판정**: {log_verdict}")
    lines.append("")
    summary["log_health"] = {
        "state_log_latest": state_log_ts,
        "snapshot_latest": snap_ts,
        "metrics_stale_flag": metrics_stale_flag,
        "stale_duration_minutes": stale_duration_minutes,
        "verdict": log_verdict,
    }

    # ---- G. 이상 징후 ----
    lines.append("## G. 이상 징후 감지")
    lines.append("")
    warnings: list[str] = []
    if metrics_stale_flag == "STALE":
        warnings.append("WARNING: rolling metrics가 STALE 상태입니다(데이터 갱신 지연 가능).")
    elif metrics_stale_flag == "CRITICAL":
        warnings.append("ALERT: rolling metrics가 CRITICAL 상태입니다(운영 파이프라인 점검 필요).")
    sc = _metric_col(rows, "score") if rows else None
    al = _metric_col(rows, "alpha_score") if rows else None
    if off_pct == 0 and rows and sc and (_safe_float(rows[-1].get(sc)) < 0):
        warnings.append("WARNING: OFF 0회인데 score 음수 지속 → OFF 조건 둔함 의심")
    if red_pct == 100 and full_pct == 0 and off_pct == 0:
        warnings.append("WARNING: REDUCED만 지속")
    if rows and sc is not None and _safe_float(rows[-1].get(sc)) < 0:
        try:
            if all(_safe_float(r.get(sc)) < 0 for r in rows):
                warnings.append("WARNING: score 지속 음수")
        except Exception:
            pass
    if rows and al is not None and _safe_float(rows[-1].get(al)) < 1.0:
        try:
            if all(_safe_float(r.get(al)) < 1.0 for r in rows):
                warnings.append("WARNING: alpha_score 지속 1 미만")
        except Exception:
            pass
    if entries == 0 and blocked_by_meta == 0 and rows:
        warnings.append("WARNING: blocked_by_meta 없음 (진입 시도 없었을 수 있음)")
    if entries == 0 and exits == 0:
        warnings.append("WARNING: trade 없음")
    if log_verdict != "정상":
        warnings.append("ALERT: 로그 갱신 중단/지연")
    t60_col = _metric_col(rows, "trades_60d")
    if t60_col and rows:
        t60_val = _safe_int(rows[-1].get(t60_col))
        if 0 < t60_val < 55:
            warnings.append(f"WARNING: trades_60d={t60_val} (50 근접)")
    if not warnings:
        lines.append("탐지된 이상 징후 없음.")
    else:
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")

    # ---- H. OFF 민감도 ----
    lines.append("## H. OFF 민감도 상태 평가")
    lines.append("")
    off_count = int(off_pct * len(rows) / 100) if rows and state_col in (rows[0] or {}) else 0
    off_verdict = "정상 범위"
    if off_count == 0 and rows and sc and _safe_float(rows[-1].get(sc)) < 0:
        off_verdict = "OFF 너무 둔함"
    elif rows and off_count > len(rows) * 0.5:
        off_verdict = "OFF 과민"
    lines.append(f"- 현재 OFF 발생 여부: {'예' if off_count > 0 else '아니오'}")
    lines.append(f"- OFF 발생 빈도(24h 내): {off_count}회")
    lines.append(f"- **자동 판단**: {off_verdict}")
    lines.append("")
    summary["off_evaluation"] = {"off_count_24h": off_count, "verdict": off_verdict}

    # ---- I. 오늘의 운영 결론 ----
    lines.append("## I. 오늘의 운영 결론")
    lines.append("")
    if log_verdict != "정상":
        conclusion = "로그 갱신 이상. 운영 파이프라인 점검 필요."
    elif verdict == "위험":
        conclusion = "OFF 조건 미발동 상태에서 손실 지속. OFF cutoff 조정 필요."
    elif verdict == "약세":
        conclusion = "최근 score/alpha_score 약화. REDUCED 유지 권장."
    elif verdict == "정상":
        conclusion = "현재 Meta Layer 정상 작동. REDUCED 유지 또는 FULL 확대 검토 가능."
    else:
        conclusion = "현재 Meta Layer 상태 유지. 지표 지속 관찰 권장."
    lines.append(conclusion)
    summary["conclusion"] = conclusion
    lines.append("")

    return "\n".join(lines), summary


def generate_daily_report(
    report_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
    write_json: bool = True,
    lookback_hours: float = 24.0,
) -> tuple[Optional[Path], Optional[Path]]:
    """
    일일 리포트 생성. (md_path, json_path) 반환. 실패 시 (None, None).
    """
    output_dir = output_dir or FR2_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_report_data(report_date=report_date, lookback_hours=lookback_hours)
    md_content, summary = build_report(data)

    date_str = (report_date or datetime.now(timezone.utc)).strftime("%Y%m%d")
    md_path = output_dir / f"B_WITH_META_DAILY_REPORT_{date_str}.md"
    json_path = output_dir / f"b_with_meta_daily_summary.json" if write_json else None

    try:
        md_path.write_text(md_content, encoding="utf-8")
    except Exception:
        return None, None

    if write_json:
        try:
            full_summary = {**summary, "generated_at": datetime.now(timezone.utc).isoformat()}

            def _sanitize(o: Any) -> Any:
                if isinstance(o, float) and o != o:
                    return None
                if isinstance(o, dict):
                    return {k: _sanitize(v) for k, v in o.items()}
                if isinstance(o, list):
                    return [_sanitize(x) for x in o]
                return o
            full_summary = _sanitize(full_summary)
            (output_dir / f"b_with_meta_daily_summary_{date_str}.json").write_text(
                json.dumps(full_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            json_path = output_dir / f"b_with_meta_daily_summary_{date_str}.json"
        except Exception:
            json_path = None

    return md_path, json_path
