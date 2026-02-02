# Stage-2 + StrategyGuard 장기구간 평가 리포트

**생성일:** 2025-01-22  
**목적:** Guard BLOCK 유도 및 장기구간 성능 평가

## 평가 케이스

### 공통 설정
- **전략:** ml_tcn
- **심볼:** BTCUSDT
- **타임프레임:** 5m
- **방향:** long-only
- **최적화 임계값:** ON
- **Signal confirmation bars:** 1

## 실행 결과

| RunID | Period | Case | Direction | Trades | TotalReturn | WinRate | MaxDD | Sharpe | BlockReasons | Stage2 | Guard | Notes | CSVPaths |
|-------|--------|------|-----------|--------|-------------|---------|-------|--------|--------------|--------|-------|-------|----------|


|20260120_113406|2023-01-01~2024-12-31|GUARD_BLOCKRUN|long|20|-13.15|10.00|13.03|-0.6819|hysteresis=111189|OFF (N/A)|Checks=521793; ALLOW=521793; BLOCK=0; Decision=ALLOW; Tracked=10|Guard BLOCK 유도 실패(BLOCK=0). 다음 실행에서 Guard decision 근거(최근 win_rate/avg_return 등) 로그 덤프 추가하여 원인 규명|trades_guard_blockrun_20260120_113406.csv|

|20260120_120314|2023-01-01~2024-12-31|GUARD_BLOCKRUN|long|20|-13.15|10.00|13.03|-0.6819|hysteresis=111189|OFF (N/A)|Checks=521803; ALLOW=521803; BLOCK=0; Decision=ALLOW; Tracked=10|Guard decision debug enabled (로그에서 샘플 파싱 실패)|trades_guard_blockrun_20260120_120314.csv|

|20260120_131820_test|2025-01-01~2025-03-01|GUARD_BLOCKRUN|long|20|-13.15|10.00|13.03|-0.6819|hysteresis=111189|OFF (N/A)|Checks=521813; ALLOW=521813; BLOCK=0; Decision=ALLOW; Tracked=10|Guard decision debug enabled; Sample: event=EXIT decision=ALLOW recent_trades_count=10 win_rate=0.0000 avg_return=0.000000 min_win_rate=0.8000 min_avg_return=0.005000 is_blocked=False block_trades_remaining=0; BLOCK=0 유지. 근거값: win_rate=0.0000 (th=0.8000), avg_return=0.000000 (th=0.005000)|trades_guard_blockrun_short_20260120_131820.csv|

|20260120_132047|2023-01-01~2024-12-31|GUARD_BLOCKRUN|long|20|-13.15|10.00|13.03|-0.6819|hysteresis=111189|OFF (N/A)|Checks=521813; ALLOW=521813; BLOCK=0; Decision=ALLOW; Tracked=10|Guard decision debug enabled; Sample: event=EXIT decision=ALLOW recent_trades_count=10 win_rate=0.0000 avg_return=0.000000 min_win_rate=0.8000 min_avg_return=0.005000 is_blocked=False block_trades_remaining=0; BLOCK=0 유지. 근거값: win_rate=0.0000 (th=0.8000), avg_return=0.000000 (th=0.005000)|trades_guard_blockrun_20260120_132047.csv|
