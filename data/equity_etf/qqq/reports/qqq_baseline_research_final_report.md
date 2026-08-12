# QQQ Baseline Research Final Report

## Verdict
`QQQ_BASELINE_RESEARCH_INCONCLUSIVE`

## Purpose
Independent NASDAQ-100 ETF (QQQ) LONG/FLAT regime research inside CAN_BIT repo, fully isolated from BTC observation/collection systems.

## Data
- Provider: yfinance / YAHOO_FINANCE_UNOFFICIAL
- Period: 1999-03-10 → 2026-07-30
- Rows: 6890
- Quality: QQQ_DATA_PIPELINE_PASS
- Adjustment: adj_close factor total-return proxy (no dividend double-count)

## Method
- Features: past-only daily
- Signal: session t close
- Execution: session t+1 open
- Costs: LOW/BASE/HIGH bps per side
- Walk-forward expanding window + locked final holdout

## Selected candidate
- DUAL_TREND_FILTER threshold=None

## Final holdout (BASE cost)
- Strategy CAGR/Sharpe/MDD: 0.0445 / 0.3858 / -0.1491
- Buy&Hold CAGR/Sharpe/MDD: 0.2346 / 1.0610 / -0.2264
- CAGR preservation: 18.948538140888513
- MDD improvement: 34.1575546112281

## Safety
- production_ready=false
- promotion_ready=false
- btc_pipeline_modified=false
- No live trading / broker API / launchd for QQQ in this stage
