# Phase C5: Time-stop micro sweep

## 목적
C4에서 ts72 채택. C5에서는 time_stop_bars만 60/72/84/96으로 미세 조정해 가장 안정적인 값 확정. (변경은 time_stop_bars 1개뿐.)

## 한 줄 실행
```bash
cd /Users/jeongminjun/Projects/Can_bit && source .venv/bin/activate && ./scripts/run_phase_checks.sh C5
```

## Run 구성
- baseline: phase_c5_base_ts72 (bars=72)
- sweep: phase_c5_ts60, phase_c5_ts84, phase_c5_ts96

## 결과
- 요약: `data/diagnostics/phase_c5_time_stop_sweep_summary.md`
- 채택 bars: `docs/phase_c5_adopt.md`

## 운영 반영
- default OFF 유지. 운영 preset은 phase_c5_adopt.md의 최종 time_stop_bars 사용.
