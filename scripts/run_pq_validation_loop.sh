#!/usr/bin/env bash
# PQ 검증 루프: Run A (Baseline) → Run B (PQ min_proba 0.56) → Run C (분기)
# 1차 목표: entries_blocked > 0 인 PQ run 확보
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_LOG="data/diagnostics/PQ_VALIDATION_RUN_LOG.txt"
mkdir -p data/diagnostics
echo "=== PQ validation loop $(date -Iseconds) ===" | tee "$RUN_LOG"

echo ""
echo "=== torch smoke test ==="
python -X faulthandler -c "import torch; print('torch', torch.__version__); print('threads', torch.get_num_threads())" || { echo "Segfault/error: fix env or reinstall torch (see README)"; exit 1; }

BASE="python -X faulthandler -m scripts.run_tcn_candidate_validation"
COMMON="--id h15_t0p004 --symbol BTCUSDT --timeframe 5m --max-entropy 1.35 --min-hold 48 --cooldown 24 --days-list 30,365"

echo ""
echo "=== Run A: Baseline (regime off, min_max_proba 0.58) ==="
$BASE $COMMON --min-max-proba 0.58 --regime-filter off
A_JSON=$(ls -t data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_*.json 2>/dev/null | head -1)
echo "Run A JSON: $A_JSON" | tee -a "$RUN_LOG"

echo ""
echo "=== Run B: PQ 작동 확인용 (min_max_proba 0.56, q 0.95, q_window 8640) ==="
$BASE $COMMON --min-max-proba 0.56 --regime-filter proba_quantile --q-window 8640 --q 0.95 --p-floor 0.55
B_JSON=$(ls -t data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_*.json 2>/dev/null | head -1)
echo "Run B JSON: $B_JSON" | tee -a "$RUN_LOG"

# B 결과에서 365d blocked_ratio 확인
BLOCKED_RATIO=$(python -c "
import json, sys
p = '$B_JSON'
try:
    d = json.load(open(p))
    for r in d.get('results', []):
        if r.get('days') == 365:
            print(r.get('blocked_ratio') or 0)
            break
    else:
        print(0)
except Exception:
    print(0)
" 2>/dev/null || echo "0")

echo "Run B (365d) blocked_ratio: $BLOCKED_RATIO" | tee -a "$RUN_LOG"

# Run C 분기 (부동소수 비교는 bc 또는 python으로)
NEED_C=$(python -c "
br = float('$BLOCKED_RATIO')
if br == 0:
    print('tighten')   # q 0.975
elif br > 0.40:
    print('loosen')    # q 0.925
else:
    print('ok')        # 0.05~0.30 구간 또는 그 외
" 2>/dev/null || echo "tighten")

echo "Run C 분기: $NEED_C" | tee -a "$RUN_LOG"

if [ "$NEED_C" = "tighten" ]; then
  echo ""
  echo "=== Run C: PQ 타이트 (q 0.975) ==="
  $BASE $COMMON --min-max-proba 0.56 --regime-filter proba_quantile --q-window 8640 --q 0.975 --p-floor 0.55
  C_JSON=$(ls -t data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_*.json 2>/dev/null | head -1)
  echo "Run C JSON: $C_JSON" | tee -a "$RUN_LOG"
elif [ "$NEED_C" = "loosen" ]; then
  echo ""
  echo "=== Run C: PQ 완화 (q 0.925) ==="
  $BASE $COMMON --min-max-proba 0.56 --regime-filter proba_quantile --q-window 8640 --q 0.925 --p-floor 0.55
  C_JSON=$(ls -t data/diagnostics/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_*.json 2>/dev/null | head -1)
  echo "Run C JSON: $C_JSON" | tee -a "$RUN_LOG"
else
  echo ""
  echo "=== Run C 생략 (B가 이상 구간 또는 선택적 14d) ==="
  echo "선택: q_window=4032(14d) 1회 추가하려면:"
  echo "  $BASE $COMMON --min-max-proba 0.56 --regime-filter proba_quantile --q-window 4032 --q 0.95 --p-floor 0.55"
fi

echo ""
echo "=== Compare baseline vs PQ (Baseline = Run A) ==="
if [ -n "$A_JSON" ] && [ -f "$A_JSON" ]; then
  python -m scripts.compare_vol_compress_vs_proba_quantile --baseline "$A_JSON" 2>&1 | tee -a "$RUN_LOG"
else
  python -m scripts.compare_vol_compress_vs_proba_quantile 2>&1 | tee -a "$RUN_LOG"
fi

echo ""
echo "=== 최종 리포트 생성 ==="
python -m scripts.report_baseline_vs_pq 2>&1 | tee -a "$RUN_LOG"

echo ""
echo "Done. Run A: $A_JSON | Run B: $B_JSON | Log: $RUN_LOG"
echo "Report: data/diagnostics/REPORT_BASELINE_VS_PQ.md"
