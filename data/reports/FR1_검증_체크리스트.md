# Feature Research Round 1 검증 체크리스트

**결론**: GPT 의견과 동일. **반박할 로컬 근거는 없고, “너무 좋게 나왔으니 검증 모드”가 맞다.**

---

## 1. 코드 상으로 확인된 것 (누수 아님으로 보이는 부분)

### Multi-timeframe (15m, 1h)
- `extended_features.py` → `resample(period, label="right", closed="right").last()`
- **label="right", closed="right**: 상위 TF 봉의 **종료 시점**만 라벨로 쓰고, 해당 구간에 **이미 닫힌** 5m 봉만 포함.
- `.reindex(df.index).ffill()`: 5m 시점 t에는 **t 이전에 종료된** 상위 TF 값만 전파. t 시점에 아직 안 닫힌 상위 TF는 reindex 후 NaN → ffill로 **이전 종료봉** 값이 채워짐.
- **→ 구현만 보면 “미래 상위 TF 봉”은 쓰이지 않음.** 다만 resample 구간 경계·타임존이 꼬이면 버그 가능성은 있으니, **단위 테스트(알고 있는 시점에서 값 수동 계산)** 로 한 번 더 검증 권장.

### Realized volatility
- `log_ret = log(close / close.shift(1))` → **과거~현재** 수익률만 사용.
- `rolling(12).std()` 등은 **과거 12봉** (현재 봉 포함, 미래 미포함).
- **→ shift/윈도우 기준으로는 누수 없음.**

### Volume z-score / ATR·true range
- `rolling(20).mean()`, `rolling(14).mean()` 등 모두 **과거+현재** 구간만 사용. 미래 봉 없음.
- **→ 구현만 보면 누수 없음.**

### Label vs feature 시점
- 학습/추론 모두 `create_sequences` / `build_ml_dataset` 경로: `future_return = close[t+horizon]/close[t] - 1` 형태로 **t 시점에서 t+horizon 미래**만 사용. feature는 t 시점까지의 OHLCV로만 구성.
- **→ label 계산 시점과 feature 시점 오프셋은 일치.**

---

## 2. 확인 필요한 것 (반박 불가, 검증 필요)

### Backtest metric 정의
- **cost_on = backtest 결과의 `total_return`**.
- 엔진 내부: `total_return = balance - 1.0` (단순 누적 수익률, 1.0 = 100%).
- **정의**: 720d 전체 구간 **단순 누적 수익률**. 로그수익 아님, 연율화 아님.
- base -0.1121, FR1 1.8176 → **같은 정의로 비교된 값**이지만, **폭이 너무 커서** 실수·경로 차이 가능성은 배제 못 함. **동일 스크립트·동일 옵션으로 base vs FR1 한 번 더 돌려서 재현** 권장.

### Train/valid/test split
- `train_tcn` → `create_sequences` → `make_time_series_splits(..., train_ratio=0.7, valid_ratio=0.15)`.
- **시간순 분리**인지는 `make_time_series_splits` 구현 확인 필요. (shuffle 여부, 시계열 순서 유지 여부.)

### 180d / 365d / 720d 일관성
- **아직 미실행.** 720d만 좋고 180d·365d가 나쁘면 과적합/기간 특이 버그 가능.
- **→ 6번 체크: 180d, 365d, 720d 각각 동일 백테스트 실행 후 비교** 필수.

---

## 3. 정리

| 항목 | 상태 | 비고 |
|------|------|------|
| Multi-TF 미래 봉 사용 여부 | 코드상 없음 | 단위 테스트로 재확인 권장 |
| Rolling/shift 누수 | 코드상 없음 | - |
| Label/feature 시점 | 코드상 일치 | - |
| cost_on 정의 | total_return = balance - 1.0 | base와 동일 정의, 재실행으로 검증 |
| 시계열 split | 확인 필요 | make_time_series_splits 검토 |
| 180d/365d/720d | 미실행 | **우선 실행할 검증** |

**솔직한 입장**:  
- “로컬 자료로 GPT 의견을 반박할 만한 내용”은 **없음.**  
- 구현만 보면 multi-TF·realized vol·volume·ATR은 **의도상 누수 없이** 짜여 있지만, **개선 폭이 커서** leakage/정렬/메트릭 이슈를 **먼저 배제하는 게 맞다.**  
- 따라서 **“성공 확정”이 아니라 “검증 모드”**로 두고,  
  1) 180d/365d/720d 백테스트,  
  2) cost_on 재실행 검증,  
  3) 필요 시 multi-TF·resample 단위 테스트  
순으로 진행하는 걸 권장.
