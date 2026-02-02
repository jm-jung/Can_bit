#!/usr/bin/env python3
"""
Discord Webhook 기반 알림 모듈

일일 운영 파이프라인의 성공/실패 알림을 Discord로 전송합니다.
"""
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

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
) -> None:
    """
    일일 실험 리포트를 Discord로 전송
    
    Args:
        summary_json_path: monitor_guard_stage2_summary_*.json 파일 경로
        ohlcv_info: OHLCV 데이터 정보 (symbol, timeframe, latest_ts, new_candles 등)
        weekly_report_path: 주간 리포트 파일 경로 (존재 시)
        pipeline_status: 파이프라인 상태 ("정상 완료" / "부분 실패" / "실패")
    
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
    
    # [4] 단기 판단 (오늘 기준)
    short_term_judgment = ""
    
    if total_return is not None and entries_executed > 0:
        if total_return > 0.01 and (win_rate or 0) > 0.5 and total_trades >= 10:
            recommendation = "✅ 권장"
            reason = "양의 수익률, 높은 승률, 충분한 거래 수를 보이고 있습니다."
        elif total_return > 0 and total_trades >= 5:
            recommendation = "⚠️ 판단 보류"
            reason = "소폭 수익이지만 데이터 기간이 짧아 실전 투입은 아직 권장되지 않습니다."
        elif total_return < -0.01:
            recommendation = "❌ 비권장"
            reason = "손실이 발생하여 실전 투입은 권장되지 않습니다."
        else:
            recommendation = "⚠️ 판단 보류"
            reason = "거래 수가 부족하여 신뢰할 만한 판단이 어렵습니다."
    else:
        recommendation = "⚠️ 판단 보류"
        reason = "데이터 부족으로 판단이 불가능합니다."
    
    short_term_judgment += f"**실거래 권장 여부**: {recommendation}\n"
    short_term_judgment += f"**사유**: {reason}"
    
    fields.append({
        "name": "🔍 [4] 단기 판단 (오늘 기준)",
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
    
    fields.insert(0, {
        "name": "📋 [0] 공통 헤더",
        "value": header,
        "inline": False
    })
    
    # Discord Embed 생성
    embed = {
        "title": "Can_bit 일일 실험 리포트",
        "description": "일일 운영 파이프라인 실행 결과입니다.",
        "color": 0x3498db if pipeline_status == "정상 완료" else 0xf1c40f,  # 파랑 또는 노랑
        "fields": fields,
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


if __name__ == "__main__":
    # 테스트용
    import argparse
    
    parser = argparse.ArgumentParser(description="Discord 알림 테스트")
    parser.add_argument("--title", type=str, default="Test", help="알림 제목")
    parser.add_argument("--message", type=str, default="Hello Discord", help="알림 메시지")
    parser.add_argument("--level", type=str, default="INFO", choices=["INFO", "WARN", "ERROR"], help="알림 레벨")
    
    args = parser.parse_args()
    send_discord_message(args.title, args.message, args.level)
