#!/usr/bin/env python3
"""
Discord Webhook 기반 알림 모듈

일일 운영 파이프라인의 성공/실패 알림을 Discord로 전송합니다.
"""
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List

# 타임존 처리 (pytz 없으면 zoneinfo 사용)
try:
    import pytz
    HAS_PYTZ = True
    HAS_ZONEINFO = False
except ImportError:
    HAS_PYTZ = False
    try:
        from zoneinfo import ZoneInfo
        HAS_ZONEINFO = True
    except ImportError:
        HAS_ZONEINFO = False

try:
    from dotenv import load_dotenv
    # .env 파일이 없어도 에러가 나지 않도록 처리
    try:
        load_dotenv()
    except (PermissionError, FileNotFoundError):
        pass  # .env 파일이 없거나 접근 불가능한 경우 무시
except ImportError:
    pass  # dotenv가 없어도 동작 (환경변수 직접 설정 가능)

PROJECT_ROOT = Path(__file__).parent.parent.parent


def get_cache_freshness(
    cache_path: str,
    tz: str = "Asia/Seoul",
    stale_days: float = 2.0
) -> Dict[str, Any]:
    """
    예측 캐시 파일의 신선도 체크
    
    Args:
        cache_path: 캐시 파일 경로 (절대 경로 또는 프로젝트 루트 기준 상대 경로)
        tz: 타임존 (기본값: "Asia/Seoul")
        stale_days: 오래된 것으로 판단하는 일수 (기본값: 2.0일)
    
    Returns:
        {
            "path": "...",
            "exists": true/false,
            "mtime_kst": "2026-02-16 13:10:00",
            "age_days": 21.3,
            "status": "FRESH|STALE|MISSING|ERROR",
            "message": "✅ 최신" / "⚠️ 21.3일 전 갱신" / "❌ 캐시 없음" / "❌ 캐시 확인 실패"
        }
    """
    result = {
        "path": cache_path,
        "exists": False,
        "mtime_kst": "N/A",
        "age_days": 0.0,
        "status": "ERROR",
        "message": "❌ 캐시 확인 실패"
    }
    
    try:
        # 경로 정규화 (프로젝트 루트 기준 상대 경로 처리)
        cache_file = Path(cache_path)
        if not cache_file.is_absolute():
            cache_file = PROJECT_ROOT / cache_file
        
        # 파일 존재 확인
        if not cache_file.exists():
            result["status"] = "MISSING"
            result["message"] = "❌ 캐시 없음"
            return result
        
        result["exists"] = True
        
        # 파일 수정 시간 가져오기
        mtime_ts = os.path.getmtime(cache_file)
        mtime_dt = datetime.fromtimestamp(mtime_ts, tz=timezone.utc)
        
        # KST로 변환
        try:
            if HAS_PYTZ:
                kst_tz = pytz.timezone(tz)
                mtime_kst = mtime_dt.astimezone(kst_tz)
            elif HAS_ZONEINFO:
                from zoneinfo import ZoneInfo
                kst_tz = ZoneInfo(tz)
                mtime_kst = mtime_dt.astimezone(kst_tz)
            else:
                # 타임존 라이브러리가 없으면 UTC 사용
                mtime_kst = mtime_dt
        except Exception:
            # 타임존 변환 실패 시 UTC 사용
            mtime_kst = mtime_dt
        
        result["mtime_kst"] = mtime_kst.strftime("%Y-%m-%d %H:%M:%S KST")
        
        # 나이 계산 (일 단위)
        now_utc = datetime.now(timezone.utc)
        age_delta = now_utc - mtime_dt
        age_days = age_delta.total_seconds() / 86400.0
        result["age_days"] = round(age_days, 1)
        
        # 상태 판정
        if age_days <= stale_days:
            result["status"] = "FRESH"
            result["message"] = "✅ 최신"
        else:
            result["status"] = "STALE"
            result["message"] = f"⚠️ {age_days:.1f}일 전 갱신"
        
    except Exception as e:
        # 예외 발생 시 ERROR 상태로 반환 (raise 금지)
        result["status"] = "ERROR"
        result["message"] = f"❌ 캐시 확인 실패: {str(e)[:50]}"
    
    return result


def send_discord_message(
    title: str,
    message: str,
    level: str = "INFO"
) -> None:
    """
    Discord Webhook으로 메시지 전송
    
    Args:
        title: 알림 제목
        message: 알림 메시지 본문
        level: 알림 레벨 (INFO, WARN, ERROR)
    
    Returns:
        None (실패 시에도 예외를 던지지 않음)
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url:
        print(
            "[DiscordNotify][WARN] DISCORD_WEBHOOK_URL not found in environment. Skipping notification.",
            file=sys.stderr
        )
        return
    
    # 레벨별 색상 매핑
    color_map = {
        "INFO": 0x3498db,  # 파랑
        "WARN": 0xf1c40f,  # 노랑
        "ERROR": 0xe74c3c,  # 빨강
    }
    color = color_map.get(level.upper(), 0x3498db)
    
    # Discord Embed 포맷
    embed = {
        "title": title,
        "description": message,
        "color": color,
        "footer": {
            "text": f"Can_bit Auto Monitor | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    payload = {
        "embeds": [embed]
    }
    
    try:
        import requests
        
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        
        # 성공 시 로그 출력하지 않음 (운영 로그 오염 방지)
        
    except ImportError:
        print(
            "[DiscordNotify][WARN] Failed to send message: requests module not installed.",
            file=sys.stderr
        )
    except Exception as e:
        print(
            f"[DiscordNotify][WARN] Failed to send message: {e}",
            file=sys.stderr
        )


def _load_decision_from_weekly_report(weekly_report_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """주간 리포트 경로(.md 또는 .json)에서 decision 객체 로드. 단일 진실 소스."""
    if not weekly_report_path:
        return None
    p = Path(weekly_report_path)
    if not p.exists():
        return None
    json_path = p.with_suffix(".json") if p.suffix.lower() == ".md" else p
    if not json_path.exists():
        return None
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("decision")
    except Exception:
        return None


def load_recent_summaries(days: int = 7) -> List[Dict[str, Any]]:
    """최근 N일간의 summary JSON 파일들을 로드"""
    monitoring_dir = PROJECT_ROOT / "data" / "monitoring"
    if not monitoring_dir.exists():
        return []
    
    summary_files = sorted(monitoring_dir.glob("monitor_guard_stage2_summary_*.json"), reverse=True)
    summaries = []
    cutoff_date = datetime.now() - timedelta(days=days)
    
    for file_path in summary_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # run_id에서 날짜 파싱 (YYYYMMDD_HHMMSS)
            run_id = data.get("run_id", "")
            if run_id:
                try:
                    date_part = run_id.split("_")[0]
                    run_date = datetime.strptime(date_part, "%Y%m%d")
                    if run_date >= cutoff_date:
                        data["_run_date"] = run_date
                        summaries.append(data)
                except (ValueError, IndexError):
                    pass
        except Exception:
            continue
    
    return summaries


def send_daily_report(
    summary_json_path: Optional[str] = None,
    ohlcv_info: Optional[Dict[str, Any]] = None,
    weekly_report_path: Optional[str] = None,
    pipeline_status: str = "정상 완료",
    cache_path: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    일일 실험 리포트를 Discord로 전송
    
    Args:
        summary_json_path: monitor_guard_stage2_summary_*.json 파일 경로
        ohlcv_info: OHLCV 데이터 정보 (symbol, timeframe, latest_ts, new_candles 등)
        weekly_report_path: 주간 리포트 파일 경로 (존재 시; .md면 동일 stem .json에서 decision 로드)
        pipeline_status: 파이프라인 상태 ("정상 완료" / "부분 실패" / "실패")
        cache_path: 예측 캐시 파일 경로 (선택적, 없으면 캐시 체크 생략)
        dry_run: True면 payload만 출력하고 전송하지 않음
    
    Returns:
        None (실패 시에도 예외를 던지지 않음)
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    
    if not webhook_url and not dry_run:
        print(
            "[DiscordNotify][WARN] DISCORD_WEBHOOK_URL not found in environment. Skipping notification.",
            file=sys.stderr
        )
        return
    
    # Summary JSON 로드
    summary_data = {}
    if summary_json_path and Path(summary_json_path).exists():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception as e:
            print(
                f"[DiscordNotify][WARN] Failed to load summary JSON: {e}",
                file=sys.stderr
            )
    
    # 최근 7일 summary 데이터 로드
    recent_summaries = load_recent_summaries(days=7)
    
    # 기본값 설정
    ohlcv_info = ohlcv_info or {}
    
    # 캐시 신선도 체크 (선택적)
    cache_info = None
    cache_stale = False
    if cache_path:
        try:
            cache_info = get_cache_freshness(cache_path, stale_days=2.0)
            if cache_info["status"] in ["STALE", "MISSING", "ERROR"]:
                cache_stale = True
        except Exception:
            # 캐시 체크 실패해도 파이프라인 계속 진행
            cache_info = {
                "path": cache_path,
                "exists": False,
                "mtime_kst": "N/A",
                "age_days": 0.0,
                "status": "ERROR",
                "message": "❌ 캐시 확인 실패"
            }
            cache_stale = True
    
    # Embed Fields 구성
    fields: List[Dict[str, Any]] = []
    
    # [1] 데이터 상태
    symbol = ohlcv_info.get("symbol", "BTCUSDT")
    timeframe = ohlcv_info.get("timeframe", "5m")
    latest_ts = ohlcv_info.get("latest_ts", "N/A")
    new_candles = ohlcv_info.get("new_candles", 0)
    
    data_status = f"**심볼**: {symbol}\n**타임프레임**: {timeframe}\n**최신 데이터 시각**: {latest_ts}"
    if new_candles > 0:
        data_status += f"\n**오늘 추가된 캔들 수**: {new_candles}개"
    else:
        data_status += "\n**오늘 추가된 캔들 수**: 없음 (이미 최신 상태)"
    
    # 데이터 상태 판단
    if latest_ts != "N/A" and new_candles >= 0:
        data_status += "\n**데이터 상태 판단**: ✅ 정상"
        data_interpretation = "오늘 데이터는 정상적으로 수집되었으며 전략 실험에 사용 가능한 상태입니다."
    elif latest_ts == "N/A":
        data_status += "\n**데이터 상태 판단**: ⚠️ 데이터 부족"
        data_interpretation = "데이터 로드에 실패했거나 데이터가 존재하지 않습니다."
    else:
        data_status += "\n**데이터 상태 판단**: ⚠️ 갱신 이상"
        data_interpretation = "데이터 갱신 과정에서 문제가 발생했을 수 있습니다."
    
    data_status += f"\n\n**해석 문장**: {data_interpretation}"
    
    # 예측 캐시 상태 추가 (캐시 정보가 있는 경우)
    if cache_info:
        cache_status_text = f"\n\n**예측 캐시 상태**:\n"
        cache_status_text += f"- 파일: `{Path(cache_info['path']).name}`\n"
        cache_status_text += f"- 수정: {cache_info['mtime_kst']}\n"
        cache_status_text += f"- 나이: {cache_info['age_days']}일\n"
        cache_status_text += f"- 판정: {cache_info['message']}"
        
        if cache_stale:
            cache_status_text += "\n- 안내: 예측 캐시 갱신 후 다시 관찰 권장"
        
        data_status += cache_status_text
    
    fields.append({
        "name": "📊 [1] 데이터 상태",
        "value": data_status,
        "inline": False
    })
    
    # [2] 전략 실험 결과 (Paper/Shadow)
    total_return = summary_data.get("total_return")
    total_trades = summary_data.get("total_trades", 0)
    win_rate = summary_data.get("win_rate")
    max_drawdown = summary_data.get("max_drawdown")
    
    strategy_result = ""
    if total_return is not None:
        strategy_result += f"**누적 수익률**: {total_return:.2%}\n"
    if total_trades > 0:
        strategy_result += f"**거래 수**: {total_trades}회\n"
    if win_rate is not None:
        strategy_result += f"**승률**: {win_rate:.2%}\n"
    if max_drawdown is not None:
        strategy_result += f"**최대 낙폭 (MDD)**: {max_drawdown:.2%}\n"
    
    # 성과 해석
    if total_return is not None:
        if total_return > 0.01:
            performance_interpretation = "✅ 수익 발생"
        elif total_return > -0.01:
            performance_interpretation = "⚠️ 소폭 손실"
        else:
            performance_interpretation = "❌ 명확한 손실"
        strategy_result += f"\n**성과 해석**: {performance_interpretation}"
    else:
        strategy_result += "\n**성과 해석**: 데이터 부족"
    
    strategy_result += "\n**주의**: 단일 일자 결과이므로 성과 판단에는 누적 관찰이 필요합니다."
    
    if strategy_result:
        fields.append({
            "name": "📈 [2] 전략 실험 결과 (Paper / Shadow)",
            "value": strategy_result,
            "inline": False
        })
    
    # [3] 리스크 판단 (Guard / Stage-2)
    guard_stats = summary_data.get("guard_scale_stats", {})
    stage2_cap_hist = summary_data.get("stage2_cap_histogram", {})
    entries_executed = summary_data.get("entries_executed", 0)
    entries_attempted = summary_data.get("entries_attempted", 0)
    total_checks = summary_data.get("total_checks", 0)
    blocked_by_guard = summary_data.get("blocked_by_guard_hard", 0)
    
    risk_judgment = ""
    
    # Guard 결과
    if entries_attempted > 0:
        block_rate = blocked_by_guard / entries_attempted if entries_attempted > 0 else 0
        if block_rate > 0.3:
            risk_judgment += f"**Guard 결과**: ⚠️ BLOCK 빈번 ({block_rate:.1%})\n"
            risk_judgment += "**사유**: Guard 스케일 임계값 미달로 인한 진입 차단\n"
        else:
            risk_judgment += f"**Guard 결과**: ✅ ALLOW ({entries_executed}/{entries_attempted} 진입 허용)\n"
    elif total_checks > 0:
        risk_judgment += f"**Guard 결과**: ✅ 정상 (진입 시도 없음, 총 {total_checks}회 체크)\n"
    else:
        risk_judgment += "**Guard 결과**: 데이터 부족\n"
    
    # Stage-2 CAP 분포
    if stage2_cap_hist:
        cap_1_0 = stage2_cap_hist.get("cap_1_0", 0)
        cap_0_8 = stage2_cap_hist.get("cap_0_8", 0)
        cap_0_6 = stage2_cap_hist.get("cap_0_6", 0)
        total_caps = cap_1_0 + cap_0_8 + cap_0_6
        
        if total_caps > 0:
            risk_judgment += f"\n**Stage-2 CAP 분포** (포지션 사이즈 제한):\n"
            risk_judgment += f"- 100% (무제한): {cap_1_0}회 ({cap_1_0/total_caps:.1%})\n"
            risk_judgment += f"- 80% (중간 제한): {cap_0_8}회 ({cap_0_8/total_caps:.1%})\n"
            risk_judgment += f"- 60% (강한 제한): {cap_0_6}회 ({cap_0_6/total_caps:.1%})\n"
    
    # 리스크 해석
    block_rate = blocked_by_guard / entries_attempted if entries_attempted > 0 else 0
    
    if entries_executed > 0 and stage2_cap_hist and total_caps > 0:
        cap_0_6_pct = (cap_0_6 / total_caps * 100)
        if cap_0_6_pct > 30:
            risk_interpretation = "리스크 제어 로직이 적극적으로 작동하여 과도한 진입을 제한하고 있습니다."
        elif block_rate > 0.3:
            risk_interpretation = "Guard가 빈번히 작동하여 리스크를 관리하고 있습니다."
        else:
            risk_interpretation = "리스크 제어 로직은 정상 작동 중이며 과도한 진입은 제한되었습니다."
    elif entries_attempted > 0:
        if block_rate > 0.3:
            risk_interpretation = "Guard가 빈번히 작동하여 리스크를 관리하고 있습니다."
        else:
            risk_interpretation = "리스크 제어 로직은 정상 작동 중이며 과도한 진입은 제한되었습니다."
    else:
        risk_interpretation = "데이터 부족으로 리스크 제어 효과를 판단하기 어렵습니다."
    
    risk_judgment += f"\n**리스크 해석**: {risk_interpretation}"
    
    if risk_judgment:
        fields.append({
            "name": "🛡️ [3] 리스크 판단 (Guard / Stage-2)",
            "value": risk_judgment,
            "inline": False
        })
    
    # [4] Decision (최근 7일 집계 기반, 단일 진실 소스: 주간 리포트 JSON)
    short_term_judgment = ""
    decision_data = _load_decision_from_weekly_report(weekly_report_path)
    if decision_data:
        label = decision_data.get("label", "데이터 부족(보류)")
        reasons = decision_data.get("reasons") or []
        metrics_snapshot = decision_data.get("metrics_snapshot") or {}
        short_term_judgment += f"**Decision**: {label}\n"
        short_term_judgment += "**Reasons**:\n"
        for r in reasons[:3]:
            short_term_judgment += f"- {r}\n"
        t = metrics_snapshot.get("total_trades")
        r = metrics_snapshot.get("total_return")
        mdd = metrics_snapshot.get("max_drawdown")
        wr = metrics_snapshot.get("win_rate")
        sh = metrics_snapshot.get("sharpe")
        short_term_judgment += "**Key metrics**: "
        parts = []
        if t is not None:
            parts.append(f"trades={t}")
        if r is not None:
            parts.append(f"return={r:.2%}")
        if mdd is not None:
            parts.append(f"mdd={mdd:.2%}" if isinstance(mdd, (int, float)) else f"mdd={mdd}")
        if wr is not None:
            parts.append(f"win_rate={wr:.2%}")
        if sh is not None:
            parts.append(f"sharpe={sh:.2f}")
        short_term_judgment += ", ".join(parts) if parts else "N/A"
    else:
        short_term_judgment += "**Decision**: 데이터 부족(보류)\n"
        short_term_judgment += "**Reasons**: 주간 리포트 없음 또는 decision 미생성. 주간 리포트 생성 후 다시 확인하세요."

    fields.append({
        "name": "🔍 [4] Decision (최근 7일 집계)",
        "value": short_term_judgment,
        "inline": False
    })
    
    # [5] 최근 누적 성과 요약 (추세 판단)
    cumulative_performance = ""
    
    if len(recent_summaries) > 0:
        returns = [s.get("total_return") for s in recent_summaries if s.get("total_return") is not None]
        if returns:
            cumulative_return = sum(returns)
            avg_daily_return = cumulative_return / len(returns)
            worst_day = min(returns)
            
            cumulative_performance += f"**최근 7일 누적 수익률**: {cumulative_return:.2%}\n"
            cumulative_performance += f"**최근 7일 평균 일수익률**: {avg_daily_return:.2%}\n"
            cumulative_performance += f"**최근 7일 최대 손실일**: {worst_day:.2%}\n"
            
            # 추세 해석
            if len(returns) < 3:
                trend_interpretation = "아직 추세 없음 (관찰 기간 부족)"
            elif abs(avg_daily_return) < 0.005 and max(returns) - min(returns) < 0.02:
                trend_interpretation = "안정적 개선 (변동성 낮음)"
            elif max(returns) - min(returns) > 0.05:
                trend_interpretation = "변동성 과다 (추가 관찰 필요)"
            else:
                trend_interpretation = "아직 추세 없음 (관찰 지속 필요)"
            
            cumulative_performance += f"\n**추세 해석**: {trend_interpretation}"
        else:
            cumulative_performance = "**최근 7일 데이터**: 수익률 데이터 부족"
    else:
        cumulative_performance = "**최근 7일 데이터**: 없음"
    
    fields.append({
        "name": "📊 [5] 최근 누적 성과 요약 (추세 판단)",
        "value": cumulative_performance,
        "inline": False
    })
    
    # [6] 매매 밀도 및 과매매 체크
    trading_density = ""
    
    if len(recent_summaries) > 0:
        today_trades = total_trades
        recent_trades = [s.get("total_trades", 0) for s in recent_summaries if s.get("total_trades", 0) > 0]
        
        if recent_trades:
            avg_recent_trades = sum(recent_trades) / len(recent_trades)
            trading_density += f"**오늘 거래 수**: {today_trades}회\n"
            trading_density += f"**최근 7일 평균 거래 수**: {avg_recent_trades:.1f}회\n"
            
            # 수수료 부담 추정 (거래당 0.04% 수수료 가정)
            estimated_commission = today_trades * 0.0004 * 2  # 매수+매도
            trading_density += f"**수수료 부담 추정**: {estimated_commission:.2%}\n"
            
            # 해석
            if today_trades > avg_recent_trades * 1.5:
                density_interpretation = "거래 빈도가 높아 수수료 민감 구간에 진입했습니다."
            elif today_trades < avg_recent_trades * 0.5:
                density_interpretation = "거래 빈도가 낮아 기회 포착이 제한적일 수 있습니다."
            else:
                density_interpretation = "거래 빈도는 정상 범위 내입니다."
            
            trading_density += f"\n**해석**: {density_interpretation}"
        else:
            trading_density = f"**오늘 거래 수**: {today_trades}회\n**최근 7일 데이터**: 거래 데이터 부족"
    else:
        trading_density = f"**오늘 거래 수**: {total_trades}회\n**최근 7일 데이터**: 없음"
    
    fields.append({
        "name": "📈 [6] 매매 밀도 및 과매매 체크",
        "value": trading_density,
        "inline": False
    })
    
    # [7] 리스크 제어 효과 요약
    risk_control_summary = ""
    
    if len(recent_summaries) > 0:
        total_allows = sum(s.get("entries_executed", 0) for s in recent_summaries)
        total_blocks = sum(s.get("blocked_by_guard_hard", 0) for s in recent_summaries)
        total_attempts = sum(s.get("entries_attempted", 0) for s in recent_summaries)
        
        avg_cap_rates = []
        for s in recent_summaries:
            cap_hist = s.get("stage2_cap_histogram", {})
            if cap_hist:
                cap_1_0 = cap_hist.get("cap_1_0", 0)
                cap_0_8 = cap_hist.get("cap_0_8", 0)
                cap_0_6 = cap_hist.get("cap_0_6", 0)
                total_caps = cap_1_0 + cap_0_8 + cap_0_6
                if total_caps > 0:
                    avg_cap_rate = (cap_1_0 * 1.0 + cap_0_8 * 0.8 + cap_0_6 * 0.6) / total_caps
                    avg_cap_rates.append(avg_cap_rate)
        
        risk_control_summary += f"**최근 7일 Guard 통계**: ALLOW {total_allows}회 / BLOCK {total_blocks}회\n"
        
        if avg_cap_rates:
            avg_cap_application = sum(avg_cap_rates) / len(avg_cap_rates)
            risk_control_summary += f"**평균 Stage-2 CAP 적용률**: {avg_cap_application:.1%}\n"
        else:
            risk_control_summary += "**평균 Stage-2 CAP 적용률**: 데이터 부족\n"
        
        # 해석
        if total_attempts > 0:
            block_rate_7d = total_blocks / total_attempts
            if block_rate_7d > 0.3:
                control_interpretation = "리스크 제어는 전략 폭주를 효과적으로 억제하고 있습니다."
            elif block_rate_7d > 0.1:
                control_interpretation = "리스크 제어가 적절히 작동하고 있습니다."
            else:
                control_interpretation = "리스크 제어는 정상 작동 중이며 추가 관찰이 필요합니다."
        else:
            control_interpretation = "데이터 부족으로 리스크 제어 효과를 판단하기 어렵습니다."
        
        risk_control_summary += f"\n**해석**: {control_interpretation}"
    else:
        risk_control_summary = "**최근 7일 데이터**: 없음"
    
    fields.append({
        "name": "🛡️ [7] 리스크 제어 효과 요약",
        "value": risk_control_summary,
        "inline": False
    })
    
    # [8] 생성된 결과물
    artifacts = ""
    if summary_json_path:
        summary_file = Path(summary_json_path).name
        artifacts += f"**최신 Summary JSON**: `{summary_file}`\n"
    
    if weekly_report_path and Path(weekly_report_path).exists():
        weekly_file = Path(weekly_report_path).name
        artifacts += f"**주간 리포트**: `{weekly_file}`\n"
    else:
        artifacts += "**주간 리포트**: 없음 (월요일이 아님)\n"
    
    if artifacts:
        fields.append({
            "name": "📁 [8] 생성된 결과물",
            "value": artifacts,
            "inline": False
        })
    
    # [0] 공통 헤더
    header = f"**생성 시간**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S KST')}\n"
    header += f"**파이프라인 상태**: {pipeline_status}"
    
    # 캐시가 오래됐으면 경고 문장 추가
    if cache_stale and cache_info:
        if cache_info["status"] == "STALE":
            header += f"\n\n⚠️ **경고**: 예측 캐시가 {cache_info['age_days']:.1f}일 전에 갱신되었습니다. 캐시 갱신 후 다시 관찰 권장."
        elif cache_info["status"] == "MISSING":
            header += "\n\n⚠️ **경고**: 예측 캐시 파일이 없습니다. 캐시 생성 후 다시 관찰 권장."
        elif cache_info["status"] == "ERROR":
            header += "\n\n⚠️ **경고**: 예측 캐시 확인 중 오류가 발생했습니다."
    
    fields.insert(0, {
        "name": "📋 [0] 공통 헤더",
        "value": header,
        "inline": False
    })
    
    # Discord Embed 생성 (같은 웹훅으로 Shadow Ops 메시지가 이어질 때 제목으로 구분)
    embed = {
        "title": "【일일 파이프라인】Can_bit 일일 실험 리포트",
        "description": (
            "OHLCV·FR2 메타·Paper/Shadow 백테스트 등 **기존 daily_run** 결과입니다. "
            "이어서 **【Shadow Ops】** 제목의 메시지가 오면 Production 확정 파이프라인 배치 요약입니다."
        ),
        "color": 0x3498db if pipeline_status == "정상 완료" else 0xf1c40f,  # 파랑 또는 노랑
        "fields": fields,
        "footer": {
            "text": f"Can_bit 일일 파이프라인 (daily_run) | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        },
        "timestamp": datetime.utcnow().isoformat()
    }
    
    payload = {
        "embeds": [embed]
    }

    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return

    try:
        import requests

        response = requests.post(
            webhook_url,
            json=payload,
            timeout=10
        )
        response.raise_for_status()

    except ImportError:
        print(
            "[DiscordNotify][WARN] Failed to send message: requests module not installed.",
            file=sys.stderr
        )
    except Exception as e:
        print(
            f"[DiscordNotify][WARN] Failed to send message: {e}",
            file=sys.stderr
        )


def send_b_with_meta_daily_report(
    daily_report_md_path: Optional[str] = None,
    summary_json_path: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """
    B_with_meta 일일 운영 리포트를 Discord 웹훅으로 전송 (일일 cadence용).

    daily_report_md_path 또는 summary_json_path 중 하나 이상 제공.
    summary_json_path가 있으면 그걸로 embed 필드 구성, 없으면 md에서 요약만 추출.
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url and not dry_run:
        print(
            "[DiscordNotify][WARN] DISCORD_WEBHOOK_URL not set. Skipping B_with_meta daily report.",
            file=sys.stderr
        )
        return

    summary = {}
    report_date = datetime.now().strftime("%Y-%m-%d")
    if summary_json_path and Path(summary_json_path).exists():
        try:
            with open(summary_json_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            report_date = summary.get("report_date", report_date)
        except Exception:
            summary = {}
    if not summary and daily_report_md_path and Path(daily_report_md_path).exists():
        try:
            text = Path(daily_report_md_path).read_text(encoding="utf-8")
            for line in text.splitlines():
                if line.startswith("- **현재 상태**:"):
                    summary.setdefault("executive_summary", {})["current_state"] = line.replace("- **현재 상태**:", "").strip()
                if line.startswith("- **운영 판단**:"):
                    summary.setdefault("executive_summary", {})["verdict"] = line.replace("- **운영 판단**:", "").strip()
                if line.startswith("## I."):
                    break
                if "**자동 판단**: " in line:
                    summary.setdefault("off_evaluation", {})["verdict"] = line.split("**자동 판단**: ")[-1].strip()
                if line.strip().startswith("현재 Meta") or line.strip().startswith("로그 갱신") or line.strip().startswith("OFF 조건"):
                    summary["conclusion"] = line.strip()
        except Exception:
            pass

    exec_sum = summary.get("executive_summary") or {}
    log_health = summary.get("log_health") or {}
    off_eval = summary.get("off_evaluation") or {}
    meta_latest = summary.get("meta_metrics_latest") or {}
    meta_health = summary.get("meta_metrics_health") or {}
    conclusion = summary.get("conclusion", "N/A")

    title = f"B_with_meta 일일 운영 리포트 ({report_date})"
    fields = [
        {
            "name": "Executive Summary",
            "value": f"상태: {exec_sum.get('current_state', 'N/A')}\nmultiplier: {exec_sum.get('current_multiplier', 'N/A')}\n판단: {exec_sum.get('verdict', 'N/A')}",
            "inline": True,
        },
        {
            "name": "로그 건강도",
            "value": f"{log_health.get('verdict', 'N/A')}\nmetrics: {meta_health.get('metrics_stale_flag', 'N/A')}",
            "inline": True,
        },
        {
            "name": "OFF 민감도",
            "value": off_eval.get("verdict", "N/A"),
            "inline": True,
        },
        {
            "name": "Rolling Metrics (latest)",
            "value": (
                f"score/alpha_score: {meta_latest.get('score', 'N/A')} / {meta_latest.get('alpha_score', 'N/A')}\n"
                f"cost_on 60/90d: {meta_latest.get('cost_on_60d', 'N/A')} / {meta_latest.get('cost_on_90d', 'N/A')}\n"
                f"alpha_fee_ratio 60/90d: {meta_latest.get('alpha_fee_ratio_60d', 'N/A')} / {meta_latest.get('alpha_fee_ratio_90d', 'N/A')}\n"
                f"trades_60d: {meta_latest.get('trades_60d', 'N/A')}"
            ),
            "inline": False,
        },
        {
            "name": "오늘의 결론",
            "value": conclusion[:500] + ("..." if len(conclusion) > 500 else ""),
            "inline": False,
        },
    ]

    embed = {
        "title": title,
        "color": 0x3498db,
        "fields": fields,
        "footer": {"text": f"Can_bit B_with_meta Daily | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"},
        "timestamp": datetime.utcnow().isoformat(),
    }
    payload = {"embeds": [embed]}

    if dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return

    try:
        import requests
        response = requests.post(webhook_url, json=payload, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"[DiscordNotify][WARN] Failed to send B_with_meta daily report: {e}", file=sys.stderr)


if __name__ == "__main__":
    # 테스트용
    import argparse
    
    parser = argparse.ArgumentParser(description="Discord 알림 테스트")
    parser.add_argument("--title", type=str, default="Test", help="알림 제목")
    parser.add_argument("--message", type=str, default="Hello Discord", help="알림 메시지")
    parser.add_argument("--level", type=str, default="INFO", choices=["INFO", "WARN", "ERROR"], help="알림 레벨")
    
    args = parser.parse_args()
    send_discord_message(args.title, args.message, args.level)
