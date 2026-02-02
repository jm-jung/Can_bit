# 백테스트 아키텍처 감사 보고서

**작성 기준**: 코드, 문서, 실행 가능한 구현만을 기준으로 작성  
**작성일**: 2025-12-23  
**범위**: `src/backtest/` 디렉토리 및 관련 문서

---

## [FACT SUMMARY]

### 우리가 실제로 구현한 것 (코드 기준)

1. **ML 모델 기반 시그널 생성 시스템**
   - 파일: `src/backtest/ml_backtest_engine_impl.py`
   - 구현: `XgbBacktestEngine`, `LstmAttnBacktestEngine` 클래스
   - 기능: 모델 확률(`proba_long`, `proba_short`)을 입력받아 시그널(`LONG`/`SHORT`/`HOLD`) 생성
   - 시그널 생성 함수: `generate_signals()` (각 엔진별 구현)
   - 시그널 생성 SSOT: `src/strategies/ml_signal_policy.py::decide_action_3class()` (LSTM-Attn용)

2. **시그널 → 매매 실행 파이프라인**
   - 파일: `src/backtest/ml_backtest_engines.py::execute_trades()`
   - 기능: DataFrame의 `signal` 컬럼을 순회하며 포지션 진입/청산 실행
   - 직접 매매 트리거: YES (시그널이 `LONG`/`SHORT`이면 즉시 진입 시도)

3. **Stage-2 게이팅 필터**
   - 파일: `src/backtest/ml_backtest_engine_impl.py::generate_signals()` (라인 566-675)
   - 기능: Stage-1(`raw_signal`) 생성 후, `stage2_trade` 이진 판정
   - 개입 지점: `generate_signals()` 내부, `execute_trades()` 호출 전
   - 역할: 시그널 필터 (독립 전략 아님)
   - 로직: `stage2_trade=False`면 `signal=HOLD`로 변환

4. **Direction 필터 (both/long/short)**
   - 파일: `src/backtest/run_ml_xgb_backtest.py` (CLI), `src/backtest/ml_backtest_engines.py` (실행)
   - 기능: `long_only`/`short_only` 파라미터로 특정 방향 시그널 차단
   - 개입 지점:
     - `generate_signals()`: `ActionDecisionConfig`에 `long_only`/`short_only` 전달 (라인 447-448)
     - `execute_trades()`: signal 처리 직전 안전장치 (라인 279-284)
   - 역할: 실행 단계 필터 (전략 선택 아님)

5. **전략 선택 메커니즘 (Factory Pattern)**
   - 파일: `src/backtest/ml_backtest_engine_impl.py::get_ml_backtest_engine()`
   - 기능: `strategy_name` 문자열로 엔진 인스턴스 생성
   - 지원 전략: `ml_xgb`, `ml_lstm_attn`, `ml_tcn` (라인 1041-1062)
   - 구조: 단순 factory 함수 (전략 등록/관리 시스템 아님)

6. **Anti-Overtrading 파라미터**
   - 파일: `src/backtest/ml_backtest_engines.py::execute_trades()`
   - 기능: `min_hold_bars`, `cooldown_bars`, `enter_long_th`/`exit_long_th` 등으로 거래 억제
   - 개입 지점: `execute_trades()` 내부, 진입/청산 결정 시점

### 우리가 아직 구현하지 않은 것

1. **전략 등록/관리 시스템**
   - 코드에 없음: 전략을 동적으로 등록/제거하는 메커니즘
   - 현재: `get_ml_backtest_engine()`에 하드코딩된 if-elif 분기만 존재

2. **조건부 전략 ON/OFF**
   - 코드에 없음: 런타임에 전략을 활성화/비활성화하는 메커니즘
   - 현재: CLI에서 `--strategy`로 한 번만 선택, 이후 변경 불가

3. **여러 전략 동시 실행/앙상블**
   - 코드에 없음: 여러 전략의 시그널을 조합하는 로직
   - 현재: 한 번에 하나의 전략만 실행

4. **전략별 독립적인 실행 로직**
   - 코드에 없음: 전략마다 다른 매매 규칙을 가진 구조
   - 현재: `execute_trades()`는 공통 구현, 전략별 차이는 `generate_signals()`에만 존재

### 우리가 의도했으나 코드로는 존재하지 않는 것

1. **"전략 관리 시스템"**
   - 문서/코멘트에 언급 없음: "전략 관리"라는 개념 자체가 코드에 없음
   - 확인 불가: 의도 여부를 코드로 판단 불가

2. **"전략 컨트롤러"**
   - 코드에 없음: 전략을 관리하는 상위 컨트롤러 클래스/모듈
   - 현재: 단순 factory 함수만 존재

---

## [ARCHITECTURE MAP]

### 시그널 생성 단계

**파일**: `src/backtest/ml_backtest_engine_impl.py`

1. **`load_predictions()`** (라인 286-386)
   - 입력: 캐시 파일 또는 `proba_long_cache`/`proba_short_cache`
   - 출력: `proba_long_arr`, `proba_short_arr`, `df_aligned`
   - 전략별 차이: XGBoost는 예외 발생, LSTM-Attn은 parquet 로드

2. **`generate_signals()`** (라인 388-720)
   - 입력: `proba_long_arr`, `proba_short_arr`, `df`, 임계값, 필터 파라미터
   - 출력: DataFrame with `raw_signal`, `stage2_trade`, `stage2_reason`, `signal` 컬럼
   - 내부 흐름:
     ```
     for each row:
       - flat_max_th/margin_th 체크 (라인 481-496)
       - decide_action_3class() 호출 → raw_signal 생성 (라인 502-508)
       - Stage-2 게이팅 (if use_stage2): raw_signal → stage2_trade 판정 (라인 569-647)
       - final_signal = stage2_trade ? raw_signal : "HOLD" (라인 649)
     ```
   - 전략별 차이:
     - XGBoost: 이진 임계값 로직 (라인 109-135)
     - LSTM-Attn: `decide_action_3class()` 사용 (라인 502)

### 필터 / Stage-2 개입 지점

1. **Stage-2 게이팅**
   - 위치: `generate_signals()` 내부 (라인 566-675)
   - 입력: `raw_signal`, `proba_long`, `proba_short`, `stage2_trade_th`, `stage2_min_edge`
   - 출력: `stage2_trade` (bool), `stage2_reason` (str)
   - 효과: `stage2_trade=False`면 `signal=HOLD`로 변환

2. **Direction 필터**
   - 위치 1: `generate_signals()` 내부 (라인 447-448)
     - `ActionDecisionConfig`에 `long_only`/`short_only` 전달
     - `decide_action_3class()` 내부에서 필터링
   - 위치 2: `execute_trades()` 내부 (라인 279-284)
     - 안전장치: signal 처리 직전 재차 필터링

3. **Anti-Overtrading 필터**
   - 위치: `execute_trades()` 내부
   - 종류:
     - `min_hold_bars`: 최소 보유 기간 (라인 354-357)
     - `cooldown_bars`: 재진입 쿨다운 (라인 607-617)
     - `enter_long_th`/`exit_long_th`: 히스테리시스 (라인 632-649)
     - `signal_confirmation_bars`: 신호 확인 (라인 677-720)

### 실행(매매) 단계

**파일**: `src/backtest/ml_backtest_engines.py::execute_trades()` (라인 124-1040)

1. **입력**: DataFrame with `signal` 컬럼
2. **순회**: `df.itertuples()` (라인 264)
3. **처리 순서**:
   ```
   for each row:
     - Direction 필터 체크 (라인 279-284)
     - Stage-2 체크 (라인 289-301)
     - Daily loss limit 체크 (라인 304-350)
     - Exit 조건 체크 (if position exists) (라인 352-520)
     - Entry 시도 (if position is None) (라인 600-750)
   ```
4. **출력**: `BacktestResult` (trades, equity_curve, metrics)

---

## [MISMATCH CHECK]

### 사용자가 "했다고 생각한 것" vs "코드에 실제로 있는 것"

1. **"전략 선택 시스템"**
   - 생각: 여러 전략을 등록하고 선택하는 시스템
   - 실제: 단순 factory 함수 (`get_ml_backtest_engine()`)
   - 혼동 가능성: 중간 (factory 패턴이지만 등록 시스템은 아님)

2. **"Stage-2는 독립 전략"**
   - 생각: Stage-2가 별도의 전략
   - 실제: Stage-2는 시그널 필터 (라인 566-675에서 `raw_signal`을 필터링)
   - 혼동 가능성: 높음 (Stage-2가 "2단계"라는 이름으로 독립성처럼 보임)

3. **"direction은 전략 선택"**
   - 생각: `--direction long`이 LONG 전략을 선택
   - 실제: `direction`은 실행 필터 (기존 전략의 시그널을 필터링)
   - 혼동 가능성: 중간 (CLI 옵션 이름이 모호할 수 있음)

4. **"양방향 판단 → 단방향 실행"**
   - 생각: 모델이 양방향 판단하고, 실행 단계에서 단방향만 선택
   - 실제: YES (코드에 구현됨)
     - `generate_signals()`: 양방향 시그널 생성 (LONG/SHORT 모두 가능)
     - `direction` 필터: 실행 단계에서 특정 방향 차단
   - 혼동 가능성: 낮음 (의도대로 구현됨)

---

## [Q&A - 코드 기준 답변]

### Q1. 지금 레포에 "전략(strategy)"라고 부를 수 있는 것은 정확히 무엇인가?

**답변 (코드 기준)**:

1. **전략 식별자**: `strategy_name` 문자열 (`ml_xgb`, `ml_lstm_attn`, `ml_tcn`)
   - 위치: `src/backtest/run_ml_xgb_backtest.py::parse_args()` (라인 27-32)
   - 역할: Factory 함수에서 엔진 선택에 사용

2. **전략 엔진**: `MLBacktestEngine` 구현체
   - `XgbBacktestEngine`: XGBoost 모델용 (이진 분류)
   - `LstmAttnBacktestEngine`: LSTM-Attn/TCN 모델용 (3-class 분류)
   - 위치: `src/backtest/ml_backtest_engine_impl.py`

3. **시그널 생성 로직**: 각 엔진의 `generate_signals()` 메서드
   - 가격 기반 ML 시그널 생성: YES
   - 시그널이 직접 매매 트리거: YES (`execute_trades()`에서 `signal` 컬럼 사용)
   - 단방향/양방향 실행 로직: 양방향 기본, `direction` 필터로 단방향 제한 가능

### Q2. 우리가 구현한 것은 아래 중 무엇에 해당하는가?

**답변 (코드 기준)**:

- **단일 양방향 수익 전략**: YES
  - 기본 동작: LONG/SHORT 모두 가능
  - 코드: `execute_trades()`에서 `signal in ("LONG", "SHORT")` 모두 처리

- **단방향(LONG-only / SHORT-only) 실행 전략**: YES
  - 구현: `direction` 필터로 특정 방향 차단
  - 코드: `ml_backtest_engines.py::execute_trades()` (라인 279-284)

- **전략 선택/차단을 위한 필터**: PARTIAL
  - 구현: `direction` 필터는 있음
  - 미구현: 전략 자체를 선택/차단하는 메커니즘은 없음 (단순 factory만 존재)

- **전략을 관리하는 상위 컨트롤러**: NO
  - 코드에 없음: 전략 등록/관리/생명주기 관리 시스템 없음
  - 현재: 단순 factory 함수만 존재

- **단순 백테스트 엔진 확장**: YES
  - 구현: `MLBacktestEngine` 추상 클래스, 구체 구현체들
  - 확장: 새로운 전략은 `MLBacktestEngine` 상속 후 `get_ml_backtest_engine()`에 분기 추가

### Q3. Stage-2는 코드상에서 어떤 역할을 하는가?

**답변 (코드 기준)**:

- **독립적인 전략**: NO
  - 코드: `generate_signals()` 내부에 구현됨 (라인 566-675)
  - 구조: 전략이 아닌 필터

- **시그널 필터**: YES
  - 입력: `raw_signal` (Stage-1 결과)
  - 출력: `stage2_trade` (bool), `final_signal` (HOLD 또는 `raw_signal`)
  - 로직: `stage2_trade=False`면 `signal=HOLD`로 변환 (라인 649)

- **실행 허용 조건**: YES
  - 개입 지점: `execute_trades()`에서 `stage2_trade` 체크 (라인 289-301)
  - 효과: `stage2_trade=False`면 진입/청산 시도 안 함

- **어느 함수/단계에서 개입하는가**:
  1. `generate_signals()` 내부: `raw_signal` 생성 후 `stage2_trade` 판정 (라인 569-647)
  2. `execute_trades()` 내부: signal 처리 직전 `stage2_trade` 체크 (라인 289-301)

### Q4. direction 옵션(both/long/short)은 무엇인가?

**답변 (코드 기준)**:

- **"전략 선택"**: NO
  - 코드: `get_ml_backtest_engine()`에서 전략 선택은 `strategy_name`으로만 수행
  - `direction`은 전략 선택과 무관

- **"실행 단계 필터"**: YES
  - 위치 1: `generate_signals()` 내부 (라인 447-448)
    - `ActionDecisionConfig`에 `long_only`/`short_only` 전달
    - `decide_action_3class()` 내부에서 필터링
  - 위치 2: `execute_trades()` 내부 (라인 279-284)
    - 안전장치로 signal 처리 직전 재차 필터링

- **"실험용 옵션"**: 확인 불가
  - 코드만으로는 실험용인지 프로덕션용인지 판단 불가
  - 기능: 실행 필터로 동작함

**코드 흐름**:
```
CLI: --direction long
  → run_ml_xgb_backtest.py: long_only = True (라인 340)
  → engine.run_backtest(long_only=True, ...)
    → generate_signals(long_only=True, ...)  [신호 생성 단계 필터링]
    → execute_trades(long_only=True, ...)    [실행 단계 안전장치]
```

### Q5. 현재 구조에서 다음 중 실제로 가능한 것은 무엇인가?

**답변 (코드 기준)**:

- **조건부로 전략을 ON/OFF**: NO
  - 코드에 없음: 런타임에 전략을 활성화/비활성화하는 메커니즘 없음
  - 현재: CLI에서 `--strategy`로 한 번만 선택, 이후 변경 불가

- **전략을 여러 개 등록하고 선택**: PARTIAL
  - 등록: NO (하드코딩된 if-elif 분기만 존재)
  - 선택: YES (`--strategy` 옵션으로 선택 가능)

- **양방향 판단 → 단방향 실행**: YES
  - 구현: `generate_signals()`는 양방향 시그널 생성, `direction` 필터로 단방향 제한
  - 코드: `ml_backtest_engines.py::execute_trades()` (라인 279-284)

- **단순히 하나의 전략을 다른 설정으로 반복 실행**: YES
  - 구현: CLI에서 `--strategy`와 다양한 파라미터 조합으로 실행 가능
  - 예: `--strategy ml_tcn --direction long --use-stage2 --stage2-trade-th 0.75`

---

## [CONCLUSION]

### 이 프로젝트는 현재 기준으로 무엇인가?

**답변: 2) 전략 실험 프레임워크**

**이유 (코드 기준)**:

1. **여러 전략을 동일 인터페이스로 실행 가능**
   - `MLBacktestEngine` 추상 클래스로 인터페이스 통일
   - Factory 패턴으로 전략 선택 (`get_ml_backtest_engine()`)
   - 코드: `src/backtest/ml_backtest_engine_impl.py::get_ml_backtest_engine()` (라인 1023-1064)

2. **다양한 파라미터 조합으로 실험 가능**
   - CLI 옵션: `--direction`, `--use-stage2`, `--stage2-trade-th`, `--min-hold-bars` 등
   - 코드: `src/backtest/run_ml_xgb_backtest.py::parse_args()` (라인 21-249)

3. **전략 개발 프로젝트가 아닌 이유**
   - 전략 자체의 로직은 모델 학습/추론 코드에 있음 (이 레포 범위 밖)
   - 이 레포는 "전략을 백테스트하는 도구"를 제공

4. **전략 관리 시스템의 초기 형태가 아닌 이유**
   - 전략 등록/관리 메커니즘 없음
   - 전략 생명주기 관리 없음
   - 단순 factory 함수만 존재

**결론**: 현재 코드는 "여러 ML 전략을 백테스트하고 파라미터를 실험하는 프레임워크"에 해당함.

---

## [참고: 코드 인용]

### 전략 선택 (Factory Pattern)
```python
# src/backtest/ml_backtest_engine_impl.py:1023-1064
def get_ml_backtest_engine(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    feature_preset: str = "extended_safe",
) -> MLBacktestEngine:
    if strategy_name == "ml_xgb":
        return XgbBacktestEngine(...)
    elif strategy_name == "ml_lstm_attn":
        return LstmAttnBacktestEngine(...)
    elif strategy_name == "ml_tcn":
        return LstmAttnBacktestEngine(...)  # TCN은 LSTM-Attn과 동일 엔진 사용
```

### Stage-2 게이팅
```python
# src/backtest/ml_backtest_engine_impl.py:566-675
if use_stage2:
    # Stage-1 결과(raw_signal)에 대해 Stage-2 판정
    for i in range(n):
        raw_sig = signals[i]
        # ... stage2_trade 판정 로직 ...
    final_signals = np.where(stage2_trade, signals, "HOLD")
    df["signal"] = final_signals
```

### Direction 필터
```python
# src/backtest/ml_backtest_engines.py:279-284
if long_only and signal == "SHORT":
    signal = "HOLD"
    block_reasons["direction_filter"] = block_reasons.get("direction_filter", 0) + 1
elif short_only and signal == "LONG":
    signal = "HOLD"
    block_reasons["direction_filter"] = block_reasons.get("direction_filter", 0) + 1
```

---

**보고서 종료**

