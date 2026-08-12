#!/usr/bin/env python3
"""
주간 모니터링 운영 리포트 생성 스크립트

data/monitoring/에 쌓이는 monitor_guard_stage2_summary_<run_id>.json 파일들을 읽어서
주간(weekly) 운영 리포트 Markdown을 자동 생성합니다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _aggregate_metrics_from_trend(performance_trend: List[Dict[str, Any]]) -> Dict[str, Any]:
    """performance_trend 리스트에서 최근 7일 집계 메트릭 계산 (비율 단위)."""
    from src.decision.decision_utils import (
        aggregate_returns_for_sharpe,
        normalize_ratio,
        safe_float,
    )

    total_trades = sum(int(safe_float(p.get("total_trades")) or 0) for p in performance_trend)
    returns = []
    for p in performance_trend:
        r = p.get("return") or p.get("total_return")
        v = normalize_ratio(r)
        if v is not None:
            returns.append(v)
    total_return = (sum(returns) / len(returns)) if returns else None
    mdd_list = []
    for p in performance_trend:
        m = p.get("max_drawdown")
        v = normalize_ratio(m)
        if v is not None:
            mdd_list.append(abs(v))
    max_drawdown = max(mdd_list) if mdd_list else None
    if max_drawdown is not None:
        max_drawdown = -max_drawdown  # convention: MDD 음수로 저장 가능

    wr_list = [(safe_float(p.get("win_rate")), int(safe_float(p.get("total_trades")) or 0)) for p in performance_trend]
    wr_list = [(w, t) for w, t in wr_list if w is not None and t > 0]
    if wr_list:
        total_w = sum(w * t for w, t in wr_list)
        total_t = sum(t for _, t in wr_list)
        win_rate = total_w / total_t
    else:
        win_rate = None
    sharpe = aggregate_returns_for_sharpe(returns)
    return {
        "total_trades": total_trades,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "sharpe": sharpe,
    }


def parse_run_id_date(run_id: str) -> Optional[datetime]:
    """run_id (YYYYMMDD_HHMMSS)에서 날짜 파싱"""
    try:
        date_part = run_id.split("_")[0]
        return datetime.strptime(date_part, "%Y%m%d")
    except (ValueError, IndexError):
        return None


def load_summary_files(start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> List[Dict[str, Any]]:
    """모니터링 summary JSON 파일들을 로드하고 필터링"""
    monitoring_dir = PROJECT_ROOT / "data" / "monitoring"
    if not monitoring_dir.exists():
        return []
    
    summary_files = sorted(monitoring_dir.glob("monitor_guard_stage2_summary_*.json"))
    summaries = []
    skipped_files = []
    
    for file_path in summary_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # 날짜 필터링
            run_id = data.get("run_id", "")
            run_date = parse_run_id_date(run_id)
            
            # timestamp가 있으면 우선 사용
            if "timestamp" in data:
                try:
                    run_date = datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass
            
            if run_date:
                if start_date and run_date < start_date:
                    continue
                if end_date and run_date > end_date:
                    continue
            
            data["_file_path"] = str(file_path)
            data["_run_date"] = run_date
            summaries.append(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            skipped_files.append((str(file_path), str(e)))
    
    # 날짜순 정렬
    summaries.sort(key=lambda x: x.get("_run_date") or datetime.min)
    
    return summaries, skipped_files


def generate_weekly_report(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output_dir: Optional[Path] = None,
    decision_thresholds: Optional[Any] = None,
) -> Optional[Path]:
    """주간 리포트 생성"""
    if output_dir is None:
        output_dir = PROJECT_ROOT / "data" / "monitoring_reports"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 기본값: 최근 7일
    if start_date is None and end_date is None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=7)
    elif end_date is None:
        end_date = start_date + timedelta(days=7)
    elif start_date is None:
        start_date = end_date - timedelta(days=7)
    
    summaries, skipped_files = load_summary_files(start_date, end_date)
    
    if not summaries:
        print(f"경고: {start_date.date()} ~ {end_date.date()} 기간에 summary 파일이 없습니다.")
        return None
    
    # 리포트 생성
    report_md = generate_markdown_report(summaries, start_date, end_date, skipped_files)
    report_json = generate_json_report(
        summaries, start_date, end_date, skipped_files,
        decision_thresholds=decision_thresholds,
    )
    
    # 파일 저장
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    
    md_path = output_dir / f"weekly_report_{start_str}_{end_str}.md"
    json_path = output_dir / f"weekly_report_{start_str}_{end_str}.json"
    
    if not md_path.exists():
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report_md)
    else:
        print(f"[WeeklyReport] MD already exists, skip: {md_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False, default=str)

    print(f"주간 리포트 생성 완료:")
    print(f"  - Markdown: {md_path}")
    print(f"  - JSON: {json_path}")
    
    return md_path


def generate_markdown_report(
    summaries: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    skipped_files: List[tuple],
) -> str:
    """Markdown 리포트 생성"""
    lines = []
    
    # 헤더
    lines.append(f"# 주간 모니터링 운영 리포트")
    lines.append("")
    lines.append(f"**기간**: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    lines.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # ======================================================================
    # A) 범위/메타
    # ======================================================================
    lines.append("## A) 범위/메타")
    lines.append("")
    
    run_ids = [s.get("run_id", "N/A") for s in summaries]
    lines.append(f"- **총 run 수**: {len(summaries)}")
    lines.append(f"- **run_id 리스트** (시간순):")
    for i, run_id in enumerate(run_ids, 1):
        run_date = summaries[i-1].get("_run_date")
        date_str = run_date.strftime("%Y-%m-%d %H:%M") if run_date else "N/A"
        lines.append(f"  {i}. `{run_id}` ({date_str})")
    lines.append("")
    
    # mode 분포
    modes = [s.get("mode", "unknown") for s in summaries]
    mode_counter = Counter(modes)
    lines.append("- **mode 분포**:")
    for mode, count in mode_counter.items():
        lines.append(f"  - {mode}: {count}개")
    lines.append("")
    
    if skipped_files:
        lines.append(f"- **건너뛴 파일**: {len(skipped_files)}개")
        for file_path, error in skipped_files[:5]:  # 최대 5개만 표시
            lines.append(f"  - `{Path(file_path).name}`: {error}")
        if len(skipped_files) > 5:
            lines.append(f"  - ... 외 {len(skipped_files) - 5}개")
        lines.append("")
    
    # ======================================================================
    # Alerts (경고)
    # ======================================================================
    alerts = detect_alerts(summaries)
    if alerts:
        lines.append("## ⚠️ Alerts")
        lines.append("")
        for alert in alerts:
            lines.append(f"- **{alert['type']}**: {alert['message']}")
        lines.append("")
    
    # ======================================================================
    # B) 성능 추이 (런 단위)
    # ======================================================================
    lines.append("## B) 성능 추이 (런 단위)")
    lines.append("")
    
    # 테이블 헤더
    lines.append("| run_id | Return | MaxDD | Trades | WinRate | entries_executed | total_checks | final_scale_mean | final_scale_median |")
    lines.append("|--------|--------|-------|--------|---------|------------------|--------------|------------------|-------------------|")
    
    returns = []
    max_dds = []
    total_trades = []
    entries_executed_list = []
    total_checks_list = []
    final_scale_means = []
    final_scale_medians = []
    
    for summary in summaries:
        run_id = summary.get("run_id", "N/A")
        
        # 성능 지표 추출 (summary에 직접 없으면 config나 다른 곳에서 찾기)
        return_val = summary.get("total_return")  # 없을 수 있음
        maxdd_val = summary.get("max_drawdown")  # 없을 수 있음
        trades_val = summary.get("total_trades", 0)
        winrate_val = summary.get("win_rate")  # 없을 수 있음
        entries_executed = summary.get("entries_executed", 0)
        total_checks = summary.get("total_checks", 0)
        
        # final_scale 통계
        final_scale_stats = summary.get("final_scale_stats", {})
        final_scale_mean = final_scale_stats.get("mean") if final_scale_stats else None
        final_scale_median = final_scale_stats.get("median") if final_scale_stats else None
        
        # 테이블 행
        return_str = f"{return_val:.2%}" if return_val is not None else "N/A"
        maxdd_str = f"{maxdd_val:.2%}" if maxdd_val is not None else "N/A"
        trades_str = str(trades_val) if trades_val else "N/A"
        winrate_str = f"{winrate_val:.2%}" if winrate_val is not None else "N/A"
        entries_str = str(entries_executed) if entries_executed else "N/A"
        checks_str = str(total_checks) if total_checks else "N/A"
        final_mean_str = f"{final_scale_mean:.4f}" if final_scale_mean is not None else "N/A"
        final_median_str = f"{final_scale_median:.4f}" if final_scale_median is not None else "N/A"
        
        lines.append(f"| {run_id} | {return_str} | {maxdd_str} | {trades_str} | {winrate_str} | {entries_executed} | {total_checks} | {final_mean_str} | {final_median_str} |")
        
        # 집계용
        if return_val is not None:
            returns.append(return_val)
        if maxdd_val is not None:
            max_dds.append(maxdd_val)
        if trades_val:
            total_trades.append(trades_val)
        if entries_executed:
            entries_executed_list.append(entries_executed)
        if total_checks:
            total_checks_list.append(total_checks)
        if final_scale_mean is not None:
            final_scale_means.append(final_scale_mean)
        if final_scale_median is not None:
            final_scale_medians.append(final_scale_median)
    
    lines.append("")
    
    # 주간 합산/평균 요약
    lines.append("### 주간 합산/평균 요약")
    lines.append("")
    lines.append("| 지표 | 값 |")
    lines.append("|------|-----|")
    
    if returns:
        avg_return = sum(returns) / len(returns)
        lines.append(f"| 평균 Return | {avg_return:.2%} |")
    
    if max_dds:
        worst_maxdd = max(max_dds)
        lines.append(f"| 최악 MaxDD | {worst_maxdd:.2%} |")
    
    if total_trades:
        total_trades_sum = sum(total_trades)
        avg_trades = sum(total_trades) / len(total_trades)
        lines.append(f"| 총 Trades | {total_trades_sum} |")
        lines.append(f"| 평균 Trades | {avg_trades:.1f} |")
    
    if entries_executed_list:
        avg_entries = sum(entries_executed_list) / len(entries_executed_list)
        total_entries = sum(entries_executed_list)
        lines.append(f"| 평균 entries_executed | {avg_entries:.1f} |")
        lines.append(f"| 총 entries_executed | {total_entries} |")
    
    if final_scale_means:
        avg_final_mean = sum(final_scale_means) / len(final_scale_means)
        lines.append(f"| 평균 final_scale_mean | {avg_final_mean:.4f} |")
    
    lines.append("")
    
    # ======================================================================
    # C) 분포/드리프트 요약
    # ======================================================================
    lines.append("## C) 분포/드리프트 요약 (주간 전체 합산)")
    lines.append("")
    
    # Guard scale histogram 합산
    guard_histograms = [s.get("guard_scale_histogram", {}) for s in summaries if s.get("guard_scale_histogram")]
    if guard_histograms:
        guard_combined = defaultdict(int)
        for hist in guard_histograms:
            for bucket, count in hist.items():
                guard_combined[bucket] += count
        
        lines.append("### Guard Scale Histogram (합산)")
        lines.append("")
        lines.append("| 버킷 | 개수 |")
        lines.append("|------|------|")
        
        # 상위 3개 버킷
        sorted_buckets = sorted(guard_combined.items(), key=lambda x: x[1], reverse=True)
        for bucket, count in sorted_buckets[:3]:
            lines.append(f"| {bucket} | {count} |")
        lines.append("")
    
    # Stage-2 CAP histogram 합산
    cap_histograms = [s.get("stage2_cap_histogram", {}) for s in summaries if s.get("stage2_cap_histogram")]
    if cap_histograms:
        cap_combined = defaultdict(int)
        total_caps = 0
        for hist in cap_histograms:
            cap_combined["cap_1_0"] += hist.get("cap_1_0", 0)
            cap_combined["cap_0_8"] += hist.get("cap_0_8", 0)
            cap_combined["cap_0_6"] += hist.get("cap_0_6", 0)
            total_caps += hist.get("total", 0)
        
        lines.append("### Stage-2 CAP Histogram (합산)")
        lines.append("")
        lines.append("| CAP 값 | 개수 | 비율 |")
        lines.append("|--------|------|------|")
        
        if total_caps > 0:
            for cap_key in ["cap_1_0", "cap_0_8", "cap_0_6"]:
                count = cap_combined[cap_key]
                pct = count / total_caps * 100
                cap_val = cap_key.replace("cap_", "").replace("_", ".")
                lines.append(f"| {cap_val} | {count} | {pct:.1f}% |")
        lines.append("")
    
    # Final scale histogram 합산
    final_histograms = [s.get("final_scale_histogram", {}) for s in summaries if s.get("final_scale_histogram")]
    if final_histograms:
        final_combined = defaultdict(int)
        for hist in final_histograms:
            for bucket, count in hist.items():
                final_combined[bucket] += count
        
        lines.append("### Final Scale Histogram (합산)")
        lines.append("")
        lines.append("| 버킷 | 개수 |")
        lines.append("|------|------|")
        
        # 상위 3개 버킷
        sorted_buckets = sorted(final_combined.items(), key=lambda x: x[1], reverse=True)
        for bucket, count in sorted_buckets[:3]:
            lines.append(f"| {bucket} | {count} |")
        lines.append("")
    
    # Cap reason counts 합산
    reason_counts_list = [s.get("cap_reason_counts", {}) for s in summaries if s.get("cap_reason_counts")]
    if reason_counts_list:
        reason_combined = defaultdict(int)
        for reason_dict in reason_counts_list:
            for reason, count in reason_dict.items():
                reason_combined[reason] += count
        
        lines.append("### Cap Reason Counts (합산, 상위 10개)")
        lines.append("")
        lines.append("| Reason | 개수 |")
        lines.append("|--------|------|")
        
        sorted_reasons = sorted(reason_combined.items(), key=lambda x: x[1], reverse=True)
        for reason, count in sorted_reasons[:10]:
            lines.append(f"| {reason} | {count} |")
        lines.append("")
    
    # ======================================================================
    # D) Config Snapshot 변화 감지
    # ======================================================================
    lines.append("## D) Config Snapshot 변화 감지")
    lines.append("")
    
    config_changes = detect_config_changes(summaries)
    if config_changes:
        lines.append("### Config 변화 이력")
        lines.append("")
        lines.append("| run_id | 변경 항목 | 이전 값 | 새 값 |")
        lines.append("|--------|----------|---------|-------|")
        
        for change in config_changes:
            run_id = change["run_id"]
            field = change["field"]
            old_val = change["old_value"]
            new_val = change["new_value"]
            lines.append(f"| {run_id} | {field} | {old_val} | {new_val} |")
    else:
        lines.append("**No config drift detected**")
        lines.append("")
    
    return "\n".join(lines)


def detect_alerts(summaries: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """경고 감지"""
    alerts = []
    
    if len(summaries) < 2:
        return alerts
    
    # cap_0_6_pct 급변 감지
    cap_0_6_pcts = []
    for summary in summaries:
        cap_hist = summary.get("stage2_cap_histogram", {})
        if cap_hist:
            cap_0_6_pct = cap_hist.get("cap_0_6_pct", 0)
            cap_0_6_pcts.append(cap_0_6_pct)
    
    if len(cap_0_6_pcts) >= 2:
        for i in range(1, len(cap_0_6_pcts)):
            diff = abs(cap_0_6_pcts[i] - cap_0_6_pcts[i-1])
            if diff >= 10.0:
                run_id = summaries[i].get("run_id", "N/A")
                alerts.append({
                    "type": "CAP 0.6 급변",
                    "message": f"run_id={run_id}: cap_0_6_pct가 {cap_0_6_pcts[i-1]:.1f}% → {cap_0_6_pcts[i]:.1f}% ({diff:.1f}%p 변화)"
                })
    
    # guard_scale_median 급변 감지
    guard_medians = []
    for summary in summaries:
        guard_stats = summary.get("guard_scale_stats", {})
        if guard_stats:
            median = guard_stats.get("median")
            if median is not None:
                guard_medians.append(median)
    
    if guard_medians:
        avg_median = sum(guard_medians) / len(guard_medians)
        for i, median in enumerate(guard_medians):
            if median > 0 and avg_median > 0:
                ratio = median / avg_median
                if ratio >= 2.0 or ratio <= 0.5:
                    run_id = summaries[i].get("run_id", "N/A")
                    alerts.append({
                        "type": "Guard Scale Median 급변",
                        "message": f"run_id={run_id}: guard_scale_median={median:.4f} (주간 평균 대비 {ratio:.2f}배)"
                    })
    
    # trades/entries_executed 급변 감지
    trades_list = [s.get("total_trades", 0) for s in summaries if s.get("total_trades")]
    entries_list = [s.get("entries_executed", 0) for s in summaries if s.get("entries_executed")]
    
    if trades_list:
        avg_trades = sum(trades_list) / len(trades_list)
        for i, trades in enumerate(trades_list):
            if avg_trades > 0:
                ratio = trades / avg_trades
                if ratio >= 2.0:
                    run_id = summaries[i].get("run_id", "N/A")
                    alerts.append({
                        "type": "Trades 급증",
                        "message": f"run_id={run_id}: total_trades={trades} (주간 평균 대비 {ratio:.2f}배)"
                    })
    
    if entries_list:
        avg_entries = sum(entries_list) / len(entries_list)
        for i, entries in enumerate(entries_list):
            if avg_entries > 0:
                ratio = entries / avg_entries
                if ratio >= 2.0:
                    run_id = summaries[i].get("run_id", "N/A")
                    alerts.append({
                        "type": "Entries Executed 급증",
                        "message": f"run_id={run_id}: entries_executed={entries} (주간 평균 대비 {ratio:.2f}배)"
                    })
    
    return alerts


def detect_config_changes(summaries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Config snapshot 변화 감지"""
    changes = []
    
    if len(summaries) < 2:
        return changes
    
    prev_config = None
    
    for summary in summaries:
        run_id = summary.get("run_id", "N/A")
        config = summary.get("config_snapshot", {})
        
        if not config:
            continue
        
        if prev_config is None:
            prev_config = config
            continue
        
        # Stage-2 config 비교
        stage2_prev = prev_config.get("stage2", {})
        stage2_curr = config.get("stage2", {})
        
        for key in ["cap_entropy_high_th", "cap_entropy_mid_th", "cap_pdiff_tiny_th", "cap_pdiff_small_th", "cap_low", "cap_mid", "cap_default"]:
            prev_val = stage2_prev.get(key)
            curr_val = stage2_curr.get(key)
            if prev_val is not None and curr_val is not None and prev_val != curr_val:
                changes.append({
                    "run_id": run_id,
                    "field": f"stage2.{key}",
                    "old_value": prev_val,
                    "new_value": curr_val,
                })
        
        # Anti-overtrading config 비교
        anti_prev = prev_config.get("anti_overtrading", {})
        anti_curr = config.get("anti_overtrading", {})
        
        for key in ["min_hold_bars", "cooldown_bars"]:
            prev_val = anti_prev.get(key)
            curr_val = anti_curr.get(key)
            if prev_val is not None and curr_val is not None and prev_val != curr_val:
                changes.append({
                    "run_id": run_id,
                    "field": f"anti_overtrading.{key}",
                    "old_value": prev_val,
                    "new_value": curr_val,
                })
        
        # Guard v2 config 비교
        guard_prev = prev_config.get("guard_v2", {})
        guard_curr = config.get("guard_v2", {})
        
        for key in ["scale_floor", "block_if_scale_below", "min_margin", "max_entropy"]:
            prev_val = guard_prev.get(key)
            curr_val = guard_curr.get(key)
            if prev_val is not None and curr_val is not None and prev_val != curr_val:
                changes.append({
                    "run_id": run_id,
                    "field": f"guard_v2.{key}",
                    "old_value": prev_val,
                    "new_value": curr_val,
                })
        
        prev_config = config
    
    return changes


def generate_json_report(
    summaries: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
    skipped_files: List[tuple],
    decision_thresholds: Optional[Any] = None,
) -> Dict[str, Any]:
    """JSON 리포트 생성 (기계가 읽기 쉬운 형식)"""
    report = {
        "report_period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat(),
        },
        "generated_at": datetime.now().isoformat(),
        "total_runs": len(summaries),
        "run_ids": [s.get("run_id") for s in summaries],
        "mode_distribution": dict(Counter(s.get("mode", "unknown") for s in summaries)),
        "skipped_files": [{"file": f, "error": e} for f, e in skipped_files],
        "performance_trend": [],
        "aggregated_distributions": {},
        "alerts": detect_alerts(summaries),
        "config_changes": detect_config_changes(summaries),
    }
    
    # 성능 추이
    for summary in summaries:
        perf = {
            "run_id": summary.get("run_id"),
            "return": summary.get("total_return"),
            "max_drawdown": summary.get("max_drawdown"),
            "total_trades": summary.get("total_trades"),
            "win_rate": summary.get("win_rate"),
            "entries_executed": summary.get("entries_executed"),
            "total_checks": summary.get("total_checks"),
            "final_scale_stats": summary.get("final_scale_stats"),
        }
        report["performance_trend"].append(perf)
    
    # 합산 분포
    guard_histograms = [s.get("guard_scale_histogram", {}) for s in summaries if s.get("guard_scale_histogram")]
    if guard_histograms:
        guard_combined = defaultdict(int)
        for hist in guard_histograms:
            for bucket, count in hist.items():
                guard_combined[bucket] += count
        report["aggregated_distributions"]["guard_scale_histogram"] = dict(guard_combined)
    
    cap_histograms = [s.get("stage2_cap_histogram", {}) for s in summaries if s.get("stage2_cap_histogram")]
    if cap_histograms:
        cap_combined = defaultdict(int)
        for hist in cap_histograms:
            cap_combined["cap_1_0"] += hist.get("cap_1_0", 0)
            cap_combined["cap_0_8"] += hist.get("cap_0_8", 0)
            cap_combined["cap_0_6"] += hist.get("cap_0_6", 0)
        report["aggregated_distributions"]["stage2_cap_histogram"] = dict(cap_combined)
    
    final_histograms = [s.get("final_scale_histogram", {}) for s in summaries if s.get("final_scale_histogram")]
    if final_histograms:
        final_combined = defaultdict(int)
        for hist in final_histograms:
            for bucket, count in hist.items():
                final_combined[bucket] += count
        report["aggregated_distributions"]["final_scale_histogram"] = dict(final_combined)
    
    reason_counts_list = [s.get("cap_reason_counts", {}) for s in summaries if s.get("cap_reason_counts")]
    if reason_counts_list:
        reason_combined = defaultdict(int)
        for reason_dict in reason_counts_list:
            for reason, count in reason_dict.items():
                reason_combined[reason] += count
        report["aggregated_distributions"]["cap_reason_counts"] = dict(reason_combined)

    # Decision Engine: 최근 7일 집계 기반 판정
    aggregated = _aggregate_metrics_from_trend(report["performance_trend"])
    report["aggregated_metrics"] = aggregated
    from src.decision.decision_engine import decide
    from src.config.decision_config import thresholds_from_env
    thresholds = decision_thresholds if decision_thresholds is not None else thresholds_from_env()
    decision_result = decide(aggregated, thresholds)
    report["decision"] = decision_result.to_dict()

    return report


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="주간 모니터링 운영 리포트 생성")
    parser.add_argument(
        "--start",
        type=str,
        help="시작 날짜 (YYYY-MM-DD). 기본값: 최근 7일 전",
    )
    parser.add_argument(
        "--end",
        type=str,
        help="종료 날짜 (YYYY-MM-DD). 기본값: 오늘",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=None,
        help="최근 N일 (지정 시 --start/--end 무시, start=오늘-N일, end=오늘)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="출력 디렉토리. 기본값: data/monitoring_reports/",
    )
    # Decision 기준값 (기본값은 config에서, 여기서 override)
    parser.add_argument("--min-trades", type=int, default=None, help="최소 거래 수 (데이터 부족 판정)")
    parser.add_argument("--discard-mdd", type=float, default=None, help="폐기 MDD 임계값 (비율, 0.12=12%%)")
    parser.add_argument("--discard-return", type=float, default=None, help="폐기 수익률 하한 (비율)")
    parser.add_argument("--discard-sharpe", type=float, default=None, help="폐기 Sharpe 하한")
    parser.add_argument("--candidate-mdd", type=float, default=None, help="실전 후보 MDD 상한 (비율)")
    parser.add_argument("--candidate-win-rate", type=float, default=None, help="실전 후보 승률 하한")
    parser.add_argument("--improve-win-rate", type=float, default=None, help="개선 필요 승률 하한")

    args = parser.parse_args()

    start_date = None
    end_date = None
    if getattr(args, "days", None) is not None:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
    else:
        if args.start:
            start_date = datetime.strptime(args.start, "%Y-%m-%d")
        if args.end:
            end_date = datetime.strptime(args.end, "%Y-%m-%d")
        if start_date is None and end_date is None:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)

    from src.config.decision_config import thresholds_from_env, thresholds_from_dict
    th = thresholds_from_env()
    th = thresholds_from_dict({
        "min_trades": getattr(args, "min_trades", None),
        "discard_mdd": getattr(args, "discard_mdd", None),
        "discard_return": getattr(args, "discard_return", None),
        "discard_sharpe": getattr(args, "discard_sharpe", None),
        "candidate_mdd": getattr(args, "candidate_mdd", None),
        "candidate_win_rate": getattr(args, "candidate_win_rate", None),
        "improve_win_rate": getattr(args, "improve_win_rate", None),
    })

    report_path = generate_weekly_report(
        start_date, end_date, output_dir, decision_thresholds=th
    )
    
    if report_path:
        print(f"\n리포트 생성 완료: {report_path}")
    else:
        print("\n경고: 리포트가 생성되지 않았습니다.")
        sys.exit(1)


if __name__ == "__main__":
    main()
