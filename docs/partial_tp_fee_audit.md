# Partial TP 수수료/슬리피지 안전 점검 (Phase C8)

## 점검 항목

1. **부분 청산 시 수수료/슬리피지 중복 적용 여부**
2. **전체 청산(잔여 포지션) 시 수수료/PnL 정상 여부**

## 정상 로직 (기대)

- 부분 청산: `partial_close_size = position_size * ratio` → fee = partial_close_size * price * (commission + slippage). 수익률 기준: `partial_fee_impact = ratio * (current_price/entry_price) * fee_ratio`.
- 잔여 청산: 진입 시 이미 전체 포지션에 대해 entry 수수료 지불됨. 잔여 청산 시에는 **exit 수수료만** 잔여 notional 기준: `remaining_ratio * (exit_price/entry_price) * fee_ratio`.

## 점검 결과

- **부분 청산**: 기존 코드는 `partial_fee_impact = fee_ratio * (1.0 + current_price/entry_price)` 사용. 비율(ratio) 미적용·과다 적용에 해당. **옵션 보정** 추가: `partial_tp_fee_correct=True` 시 `partial_fee_impact = partial_tp_ratio * (current_price/entry_price) * fee_ratio` 적용.
- **잔여 청산**: 기존은 `fee_impact_remaining = fee_ratio * (1.0 + exit_price/entry_price)` → entry 비용이 잔여 청산 시 다시 반영됨. **옵션 보정** 시 `fee_impact_remaining = remaining_ratio * (exit_price/entry_price) * fee_ratio` (exit만 잔여 notional 기준).

## 적용 방식

- **기본값**: `partial_tp_fee_correct=False` 유지 → 기존 백테스트 결과 변경 없음.
- 보정 적용이 필요하면 백테스트 호출 시 `partial_tp_fee_correct=True` 전달 (현재는 `execute_trades` 인자로만 가능; CLI는 미노출).

## 결론

- Partial TP 기능은 옵션으로 유지. 수수료/슬리피지 보정은 feature flag로 선택 적용 가능.
