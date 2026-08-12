#!/usr/bin/env bash
# Phase C7 partial_tp 검증: 동일 run_id 1회 재실행 후 count 확인, 0이면 threshold=0.0020 run 1회 추가.
set -e
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG="${PROJECT_ROOT}/data/diagnostics"
export PYTHONFAULTHANDLER=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

cd "${PROJECT_ROOT}" || exit 1
source .venv/bin/activate

COMMON=(
  --id h15_t0p004 --symbol BTCUSDT --timeframe 5m
  --min-max-proba 0.58 --max-entropy 1.35 --min-hold 48 --cooldown 24
  --regime-filter off --position-scaling off
  --time-stop on --time-stop-bars 72
  --early-exit on --early-exit-lookback 12 --early-exit-p-floor 0.55 --early-exit-bad-k 8
  --days-list 30,365
)

run() {
  python -X faulthandler -m scripts.run_tcn_candidate_validation "${COMMON[@]}" "$@"
}

echo "[1] 동일 run_id phase_c7_tp025_r50 1회 재실행 (threshold=0.0025)"
run --run-id phase_c7_tp025_r50 --partial-tp on --partial-tp-threshold 0.0025 --partial-tp-ratio 0.5

JSON="${DIAG}/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c7_tp025_r50.json"
if [[ ! -f "$JSON" ]]; then
  echo "ERROR: $JSON not found"
  exit 1
fi

# 30d / 365d partial_tp_count 추출 (jq)
COUNT_30=$(python3 -c "
import json
with open('$JSON') as f:
    d = json.load(f)
for r in d.get('results', []):
    if r.get('days') == 30:
        print(r.get('partial_tp_count'), end='')
        break
else:
    print('N/A', end='')
")
COUNT_365=$(python3 -c "
import json
with open('$JSON') as f:
    d = json.load(f)
for r in d.get('results', []):
    if r.get('days') == 365:
        print(r.get('partial_tp_count'), end='')
        break
else:
    print('N/A', end='')
")

echo "[2] partial_tp_count 확인: 30d=$COUNT_30, 365d=$COUNT_365"
if [[ "$COUNT_365" == "0" ]] || [[ "$COUNT_365" == "N/A" ]]; then
  echo "[3] 365d count가 0/N/A → threshold=0.0020 run 1회 추가 실행"
  run --run-id phase_c7_tp020_r50 --partial-tp on --partial-tp-threshold 0.0020 --partial-tp-ratio 0.5
  JSON2="${DIAG}/tcn_candidate_validation_BTCUSDT_5m_h15_t0p004_phase_c7_tp020_r50.json"
  COUNT2=$(python3 -c "
import json
with open('$JSON2') as f:
    d = json.load(f)
for r in d.get('results', []):
    if r.get('days') == 365:
        print(r.get('partial_tp_count'), end='')
        break
else:
    print('N/A', end='')
")
  echo "    phase_c7_tp020_r50 365d partial_tp_count=$COUNT2"
else
  echo "[3] 365d count 비영 → threshold 0.0020 run 생략"
fi
echo "Done."
exit 0
