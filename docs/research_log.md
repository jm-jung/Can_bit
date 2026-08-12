# Research log

## Phase C9 (Break-even stop) — 공식 결론

**pinned end_date**: 2026-03-03

| run_id        | cost_on (365d) | MDD (365d) | 판정   |
|---------------|----------------|------------|--------|
| phase_c9_base | -0.0582206     | 0.1146812  | baseline |
| phase_c9_be002 | -0.2026050   | 0.2092240  | REJECT |
| phase_c9_be003 | -0.1582152   | 0.1739844  | REJECT |
| phase_c9_be004 | -0.1335606   | 0.1497917  | REJECT |

**결론**: Phase C9 **KEEP_BASELINE**. Break-even stop은 현 전략에서 REJECT, 추가 튜닝 중단. 운영 기본값: time_stop=72, early_exit ON, partial_tp OFF, **break_even OFF**.

---

## Phase C8 (Partial TP combo)

- baseline 대비 cost_on -0.10 이상 악화, MDD +0.056 이상 악화.
- 모든 Partial TP variants REJECT. **KEEP_BASELINE**, partial_tp OFF.

---

## Phase C7 (Partial TP)

- 365d pinned 기준 cost_on·MDD 악화로 REJECT.

---

## Exit 실험 종료 선언

Phase C4~C9 결론 정리. Exit류(Partial TP, Break-even stop) 추가 탐색 **중단**.  
다음 단계: **Phase D1** (모델/학습 개선, entry edge).
