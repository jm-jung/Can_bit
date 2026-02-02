# 하루 1회 운영 루틴

이 문서는 Can_bit 전략의 일일 모니터링 및 데이터 업데이트 운영 프로토콜을 설명합니다.

## 개요

매일 1회 실행하여:
1. 최신 OHLCV 데이터를 증분 업데이트
2. Guard v2 + Stage-2 v2.2 CAP 로직으로 paper/shadow 실행 (실주문 없음)
3. 모니터링 summary JSON 생성
4. 주간 리포트 생성 (선택)

## (A) 데이터 업데이트 CLI

로컬 데이터의 마지막 timestamp 이후부터 현재까지 OHLCV를 증분 fetch하여 병합합니다.

### 실행 명령

```bash
# 기본 실행 (BTCUSDT, 5m, 현재까지) - CCXT 경로 사용
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now

# 다른 심볼/타임프레임
python -m src.data.update_ohlcv --symbol ETHUSDT --timeframe 1m --end now

# 특정 날짜까지
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end 2026-01-26

# 고급 옵션 (필요한 경우만)
# 타임아웃 조정
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now --timeout-ms 20000

# SSL 인증서 검증 비활성화 (비추천, 필요한 경우만)
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now --insecure-ssl 1

# 직접 klines API 사용 (최후 수단, CCXT 실패 시)
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now --fallback-direct-klines 1
```

### 옵션 설명

- `--timeout-ms`: 네트워크 타임아웃 (밀리초, 기본값: 30000)
- `--insecure-ssl`: SSL 인증서 검증 비활성화 (비추천, 기본값: 0)
- `--fallback-direct-klines`: CCXT 대신 직접 klines API 사용 (최후 수단, 기본값: 0)

**환경변수 지원:**
- `CANBIT_INSECURE_SSL=1`: SSL 인증서 검증 비활성화 (CLI 인자보다 우선순위 낮음)

### 출력 예시

```
[Update] 기존 데이터: 2609255 rows, 마지막 timestamp: 2025-12-18 08:27:00
[Update] 증분 업데이트 시작: 2025-12-18 08:27 ~ 2026-01-26 11:00
[Update] ✅ 업데이트 완료:
  - 이전 마지막: 2025-12-18 08:27:00
  - 현재 마지막: 2026-01-26 10:59:00
  - 가져온 행 수: 12345
  - 총 행 수: 2621600
  - 저장 경로: data/ohlcv/BTCUSDT_5m_full.csv
```

### 주의사항

- API 키 없이 public OHLCV fetch로 동작 (네트워크 키/시크릿 불필요)
- Rate limit 고려하여 batch로 나누어 가져옴
- 중복 제거 후 append/merge 수행
- 업데이트 로그는 `data/monitoring/ohlcv_update_<timestamp>.json`에 저장

## (B) Paper/Shadow 실행 CLI

Guard v2 + Stage-2 v2.2 CAP 로직으로 최신 데이터에서 paper/shadow(로깅-only) 실행합니다.
**실주문은 절대 하지 않으며**, 모니터링 summary JSON만 생성합니다.

### 실행 명령

```bash
# 기본 실행 (데이터 업데이트 포함)
python -m src.monitoring.run_paper_shadow \
  --symbol BTCUSDT \
  --timeframe 5m \
  --lookback-days 30 \
  --update-data 1

# 데이터 업데이트 스킵
python -m src.monitoring.run_paper_shadow \
  --symbol BTCUSDT \
  --timeframe 5m \
  --lookback-days 30 \
  --update-data 0

# 다른 전략/방향
python -m src.monitoring.run_paper_shadow \
  --symbol BTCUSDT \
  --timeframe 5m \
  --lookback-days 30 \
  --strategy ml_xgb \
  --direction long \
  --update-data 1
```

### 파라미터

- `--symbol`: 거래 심볼 (default: BTCUSDT)
- `--timeframe`: 타임프레임 (default: 5m)
- `--lookback-days`: 최근 N일 데이터 사용 (default: 30)
- `--update-data`: 데이터 업데이트 여부 (1=업데이트, 0=스킵, default: 1)
- `--strategy`: 전략 이름 (ml_xgb, ml_lstm_attn, ml_tcn, default: ml_tcn)
- `--direction`: 방향 (long, short, default: long)

### 출력 예시

```
[Paper/Shadow] Paper/Shadow 실행 시작:
  - 심볼: BTCUSDT
  - 타임프레임: 5m
  - 전략: ml_tcn
  - 방향: long
  - 기간: 2025-12-27 ~ 2026-01-26 (최근 30일)
  - 모드: paper (실주문 없음, 모니터링만)
...
[Paper/Shadow] ✅ Paper/Shadow 실행 완료
[Paper/Shadow] 결과: Return=-1.79%, Trades=174
[Paper/Shadow] 모니터링 summary JSON이 생성되었습니다:
  - data/monitoring/monitor_guard_stage2_summary_20260126_110906.json
```

### 생성되는 파일

- `data/monitoring/monitor_guard_stage2_<run_id>.jsonl`: 상세 이벤트 로그 (CHECK/ENTRY/EXIT)
- `data/monitoring/monitor_guard_stage2_summary_<run_id>.json`: 집계 요약 (주간 리포트에 사용)

## (C) 주간 리포트 생성 CLI

주간 모니터링 운영 리포트를 생성합니다.

### 실행 명령

```bash
# 최근 7일 (기본값)
python -m src.monitoring.generate_weekly_report

# 특정 기간 지정
python -m src.monitoring.generate_weekly_report \
  --start 2026-01-19 \
  --end 2026-01-26
```

### 생성되는 파일

- `data/monitoring_reports/weekly_report_<start>_<end>.md`: Markdown 리포트
- `data/monitoring_reports/weekly_report_<start>_<end>.json`: JSON 리포트 (기계 읽기용)

## 일일 운영 프로토콜

### 매일 1회 실행 (권장: 오전 9시)

```bash
# 1. 데이터 업데이트
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now

# 2. Paper/Shadow 실행 (데이터 업데이트 포함)
python -m src.monitoring.run_paper_shadow \
  --symbol BTCUSDT \
  --timeframe 5m \
  --lookback-days 30 \
  --update-data 1
```

### 주간 리포트 생성 (매주 월요일)

```bash
# 지난 주 리포트 생성
python -m src.monitoring.generate_weekly_report \
  --start 2026-01-19 \
  --end 2026-01-26
```

## 검증 절차

### 1. 데이터 업데이트 검증

```bash
# 실행 전 마지막 timestamp 확인
python -c "from src.services.ohlcv_service import load_ohlcv_df; df=load_ohlcv_df(); print(f'Before: {df[\"timestamp\"].max()}')"

# 업데이트 실행
python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now

# 실행 후 마지막 timestamp 확인
python -c "from src.services.ohlcv_service import load_ohlcv_df; df=load_ohlcv_df(); print(f'After: {df[\"timestamp\"].max()}')"
```

**기대 결과**: `after_last_ts`가 현재 날짜(2026-01-26) 근처까지 갱신됨

### 2. Paper/Shadow 실행 검증

```bash
# 실행
python -m src.monitoring.run_paper_shadow \
  --symbol BTCUSDT \
  --timeframe 5m \
  --lookback-days 30 \
  --update-data 0

# 생성된 summary 확인
ls -lt data/monitoring/monitor_guard_stage2_summary_*.json | head -1
```

**기대 결과**: `data/monitoring/monitor_guard_stage2_summary_<run_id>.json` 파일 생성

### 3. 주간 리포트 검증

```bash
# 리포트 생성
python -m src.monitoring.generate_weekly_report \
  --start 2026-01-19 \
  --end 2026-01-26

# 리포트 확인
cat data/monitoring_reports/weekly_report_2026-01-19_2026-01-26.md | head -30
```

**기대 결과**: 방금 실행한 run이 주간 리포트에 포함됨

## 중요 사항

### 절대 원칙

- **Guard v2 / Stage-2 CAP 로직 변경 금지**: 모든 로직은 동결 상태 유지
- **트레이딩 엔진의 진입/청산 로직 변경 금지**: 백테스트 실행 경로를 그대로 사용
- **실주문 금지**: 네트워크 키/시크릿 요구 없음, API 호출 없음
- **모니터링만 수행**: 로깅 및 summary JSON 생성만 수행

### 허용 사항

- 데이터 업데이트 모듈/CLI 추가
- Paper/Shadow runner 추가
- 모니터링 로깅 연결 (기존 MonitoringLogger 사용)

## 문제 해결

### 데이터 업데이트 실패

- 네트워크 연결 확인
- Rate limit 확인 (너무 빠른 요청 시 잠시 대기)
- 기존 데이터 파일 경로 확인

### Paper/Shadow 실행 실패

- 데이터 범위 확인 (lookback-days가 너무 크면 데이터 부족)
- 전략 모델 파일 확인 (ml_tcn 모델 존재 여부)
- 임계값 파일 확인 (data/thresholds/ 폴더)

### 주간 리포트에 run이 포함되지 않음

- run_id의 날짜가 리포트 기간 내에 있는지 확인
- summary JSON 파일이 올바른 위치에 있는지 확인

---

## 자동화 (launchd)

macOS의 `launchd`를 사용하여 매일 자동으로 운영 루틴을 실행할 수 있습니다.

### 설치

1. launchd plist 파일을 LaunchAgents 디렉토리로 복사:

```bash
mkdir -p ~/Library/LaunchAgents
cp scripts/com.canbit.daily.plist ~/Library/LaunchAgents/com.canbit.daily.plist
```

2. 기존에 로드된 경우 먼저 언로드:

```bash
launchctl unload ~/Library/LaunchAgents/com.canbit.daily.plist 2>/dev/null || true
```

3. launchd에 로드:

```bash
launchctl load ~/Library/LaunchAgents/com.canbit.daily.plist
```

### 상태 확인

launchd에 등록된 작업 목록에서 확인:

```bash
launchctl list | grep com.canbit.daily
```

정상적으로 로드되면 다음과 같은 출력이 표시됩니다:

```
PID   Status  Label
123   -       com.canbit.daily
```

### 즉시 실행 테스트

설치 직후 `RunAtLoad=true`로 설정되어 있어 자동으로 1회 실행됩니다.
또는 수동으로 즉시 실행하려면:

```bash
launchctl start com.canbit.daily
```

### 로그 확인

launchd의 표준 출력/에러 로그:

```bash
# 표준 출력
tail -200 data/ops_logs/launchd_out.log

# 표준 에러
tail -200 data/ops_logs/launchd_err.log
```

일일 실행 스크립트의 상세 로그:

```bash
# 오늘 날짜의 로그
tail -200 data/ops_logs/daily_run_$(date +%Y-%m-%d).log

# 특정 날짜의 로그
tail -200 data/ops_logs/daily_run_2026-01-26.log
```

### 실행 스케줄

- **기본 설정**: 매일 오전 9시 (KST 기준)
- **수정 방법**: `scripts/com.canbit.daily.plist`의 `StartCalendarInterval` 섹션 수정 후 재로드

```bash
# plist 수정 후
launchctl unload ~/Library/LaunchAgents/com.canbit.daily.plist
launchctl load ~/Library/LaunchAgents/com.canbit.daily.plist
```

### 제거

자동 실행을 중지하고 제거하려면:

```bash
# launchd에서 언로드
launchctl unload ~/Library/LaunchAgents/com.canbit.daily.plist

# plist 파일 삭제
rm ~/Library/LaunchAgents/com.canbit.daily.plist
```

### 실행 순서

자동화 스크립트(`scripts/daily_run.sh`)는 다음 순서로 실행됩니다:

1. **OHLCV 업데이트**: `python -m src.data.update_ohlcv --symbol BTCUSDT --timeframe 5m --end now`
2. **Paper/Shadow 실행**: `python -m src.monitoring.run_paper_shadow --symbol BTCUSDT --timeframe 5m --lookback-days 30 --update-data 0`
3. **주간 리포트 생성** (월요일만): `python -m src.monitoring.generate_weekly_report --start <last_monday> --end <last_sunday>`

어떤 단계든 실패(exit code != 0)하면 즉시 종료되며, 에러 로그에 원인이 기록됩니다.

### 문제 해결

#### 실행되지 않는 경우

1. launchd 상태 확인:
   ```bash
   launchctl list | grep com.canbit.daily
   ```

2. 에러 로그 확인:
   ```bash
   tail -50 data/ops_logs/launchd_err.log
   ```

3. 수동 실행으로 테스트:
   ```bash
   bash scripts/daily_run.sh
   ```

#### 권한 문제

스크립트에 실행 권한이 있는지 확인:

```bash
chmod +x scripts/daily_run.sh
```

#### Python 경로 문제

`.venv/bin/python`이 존재하는지 확인:

```bash
ls -la .venv/bin/python
```
