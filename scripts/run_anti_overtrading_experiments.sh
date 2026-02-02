#!/usr/bin/env bash
set -euo pipefail

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="data/experiments/anti_overtrading_${TS}"
mkdir -p "${OUT_DIR}"

run_case () {
  local name="$1"
  shift

  local log_file="${OUT_DIR}/${name}.log"
  echo "============================================================" | tee "${log_file}"
  echo "[RUN] ${name}" | tee -a "${log_file}"
  echo "[CMD] python -m src.backtest.run_ml_xgb_backtest $*" | tee -a "${log_file}"
  echo "============================================================" | tee -a "${log_file}"

  python -m src.backtest.run_ml_xgb_backtest "$@" 2>&1 | tee -a "${log_file}"

  local report_path
  report_path="$(grep -Eo 'INFO:src\\.backtest\\.backtest_report:Saved backtest report to .*\\.json' "${log_file}" | sed 's/^INFO:src\\.backtest\\.backtest_report:Saved backtest report to //g' | tail -n 1 || true)"
  if [[ -n "${report_path}" ]]; then
    echo "[REPORT] ${report_path}" | tee -a "${log_file}"
    echo "${report_path}" > "${OUT_DIR}/${name}.report_path.txt"
  else
    echo "[REPORT] (not found in log)" | tee -a "${log_file}"
  fi

  echo "" | tee -a "${log_file}"
}

BASE_ARGS=(--strategy ml_lstm_attn --symbol BTCUSDT --timeframe 5m --use-optimized-threshold)

run_case "00_baseline_loose" \
  "${BASE_ARGS[@]}" \
  --signal-confirmation-bars 1

run_case "01_medium" \
  "${BASE_ARGS[@]}" \
  --signal-confirmation-bars 5 \
  --min-hold-bars 6 \
  --cooldown-bars 4 \
  --enter-long-th 0.65 \
  --exit-long-th 0.52 \
  --enter-short-th 0.65 \
  --exit-short-th 0.52 \
  --margin-th 0.06 \
  --apply-confirmation-to-flips

run_case "02_conservative" \
  "${BASE_ARGS[@]}" \
  --signal-confirmation-bars 8 \
  --min-hold-bars 12 \
  --cooldown-bars 8 \
  --enter-long-th 0.70 \
  --exit-long-th 0.55 \
  --enter-short-th 0.70 \
  --exit-short-th 0.55 \
  --margin-th 0.08 \
  --apply-confirmation-to-flips

run_case "03_conservative_no_flip_confirmation" \
  "${BASE_ARGS[@]}" \
  --signal-confirmation-bars 8 \
  --min-hold-bars 12 \
  --cooldown-bars 8 \
  --enter-long-th 0.70 \
  --exit-long-th 0.55 \
  --enter-short-th 0.70 \
  --exit-short-th 0.55 \
  --margin-th 0.08 \
  --no-confirmation-to-flips

echo "============================================================"
echo "[DONE] Logs saved under: ${OUT_DIR}"
echo "============================================================"

echo ""
echo "==================== QUICK SUMMARY ===================="
for f in "${OUT_DIR}"/*.log; do
  name="$(basename "${f}" .log)"
  echo ""
  echo "---- ${name} ----"
  grep -nE "\[ANTI-OVERTRADING\]|Total Return:|Total Trades:|flip_count|block_reasons|fee_decay_estimate|total_fee(s|_ratio)|Average holding period" "${f}" | tail -n 120 || true
done

