#!/usr/bin/env bash
# 백그라운드 Phase 실험 완료 여부 빠르게 확인
# 사용: ./scripts/check_phase_complete.sh [d3|d4|d5|d6]
set -e
cd "$(dirname "$0")/.."
PHASE="${1:-d4}"

case "$PHASE" in
  d3)
    REPORT="data/reports/phase_d3_regime_filter_summary.md"
    DIAG_GLOB="phase_d3"
    EXPORTS="data/backtests/phase_d3_best_run.json data/backtests/phase_d3_best_trades.csv"
    ;;
  d4)
    REPORT="data/reports/phase_d4_position_scaling_summary.md"
    DIAG_GLOB="phase_d4"
    EXPORTS="data/backtests/phase_d4_best_run.json data/backtests/phase_d4_best_trades.csv"
    ;;
  d5)
    REPORT="data/reports/phase_d5_entry_micro_tuning_summary.md"
    DIAG_GLOB="phase_d5"
    EXPORTS="data/backtests/phase_d5_best_run.json data/backtests/phase_d5_best_trades.csv"
    ;;
  d6)
    REPORT="data/reports/phase_d6_stability_summary.md"
    DIAG_GLOB="phase_d6"
    EXPORTS="data/backtests/phase_d6_180d_trades.csv data/backtests/phase_d6_365d_trades.csv data/backtests/phase_d6_720d_trades.csv"
    ;;
  d7)
    REPORT="data/reports/phase_d7_entry_tightening_summary.md"
    DIAG_GLOB="phase_d7"
    EXPORTS="data/backtests/phase_d7_best_run.json data/backtests/phase_d7_best_trades_720d.csv"
    ;;
  d8)
    REPORT="data/reports/phase_d8_exit_sweep_summary.md"
    DIAG_GLOB="phase_d8"
    EXPORTS="data/backtests/phase_d8_best_run.json data/backtests/phase_d8_best_trades_720d.csv"
    ;;
  *)
    echo "Usage: $0 [d3|d4|d5|d6|d7|d8]"
    exit 1
    ;;
esac

echo "=== Phase D${PHASE#d} 완료 여부 확인 ==="
echo ""

if [ -f "$REPORT" ]; then
  echo "[OK] 요약 보고서 존재: $REPORT"
  echo "     최종 수정: $(ls -l "$REPORT" | awk '{print $6, $7, $8}')"
else
  echo "[--] 요약 보고서 없음: $REPORT"
fi

echo "[OK] diagnostics JSON (phase 관련): $(ls data/diagnostics/*${DIAG_GLOB}*.json 2>/dev/null | wc -l | tr -d ' ') 개"

for f in $EXPORTS; do
  if [ -f "$f" ]; then
    echo "[OK] Export: $f"
  else
    echo "[--] Export 없음: $f"
  fi
done

echo ""
echo "--- 요약본 마지막 5줄 ---"
[ -f "$REPORT" ] && tail -5 "$REPORT"
echo ""
echo "완료 시: 위에 [OK]만 있고, 요약본에 BEST run_id / verdict가 있으면 정상 종료."
