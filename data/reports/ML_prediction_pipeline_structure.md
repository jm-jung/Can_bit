# ML Prediction Pipeline — Structural Inspection

**Scope:** Analysis only; no code was modified.

---

## A) Model architecture summary

| Item | Detail |
|------|--------|
| **Model** | TCN (Temporal Convolutional Network) |
| **Classification** | **Multi-class (3 classes)**, not binary |
| **Classes** | FLAT (0), LONG (1), SHORT (2) — see `LstmClassIndex` in `src/dl/data/labels.py` |
| **Output (raw)** | Logits, shape `(batch_size, 3)`; **no** softmax inside the model |
| **Output (inference)** | Probabilities via `F.softmax(logits, dim=-1)` → shape `(batch_size, 3)` |
| **Probability derivation** | **Softmax** over the 3 logits (not sigmoid) |

**Code references:**
- **Model definition:** `src/dl/models/tcn.py`
  - `TCNModel.forward()` returns raw logits `(batch, num_classes)`; comment states softmax is applied at inference (lines 195–226).
  - `num_classes=3` (default), classifier: `nn.Linear(num_channels[-1], num_classes)`.
- **Inference wrapper:** `src/dl/tcn_model.py`
  - `TCNSignalModel.predict_proba_batch()` (lines 395–432): `logits = self.model(batch_sequences)` then `probs = torch.nn.functional.softmax(logits, dim=-1)`; extracts LONG/SHORT via `LstmClassIndex.LONG`, `LstmClassIndex.SHORT`.
- **Training:** `src/dl/train/train_tcn.py` uses `num_classes=3` and expects label range [0, 2] (FLAT/LONG/SHORT).

---

## B) Prediction output structure

- **Per-bar output:** Two arrays used downstream:
  - `proba_long_arr`: P(LONG) = `probs[:, LstmClassIndex.LONG]`
  - `proba_short_arr`: P(SHORT) = `probs[:, LstmClassIndex.SHORT]`
- **Implicit:** P(FLAT) = `1 - p_long - p_short` (clamped to [0,1] in backtest).
- **Tensor shape:** At inference, after softmax: `(batch_size, 3)`; indices 0=FLAT, 1=LONG, 2=SHORT.

**Code references:**
- `src/dl/tcn_model.py` lines 415–419: `probs = F.softmax(logits, dim=-1)` then `proba_long_batch = probs[:, LstmClassIndex.LONG].cpu().numpy()`, `proba_short_batch = probs[:, LstmClassIndex.SHORT].cpu().numpy()`.
- `src/dl/data/labels.py`: `LstmClassIndex` FLAT=0, LONG=1, SHORT=2.

---

## C) Entry decision pipeline

High-level flow:

1. **Model prediction**  
   TCN forward → logits → softmax → `(proba_long_arr, proba_short_arr)` (and implicitly FLAT).

2. **Signal generation**  
   `generate_signals(proba_long_arr, proba_short_arr, df, long_threshold, short_threshold, ...)` adds a **`signal`** column to the DataFrame: per bar, LONG / SHORT / HOLD (and FLAT where used).  
   Uses 3-class policy (e.g. `decide_action_3class`) with `long_threshold` and `short_threshold` (from `resolve_ml_thresholds` for `ml_tcn`).

3. **Backtest / trade execution**  
   `execute_trades(df, ...)` iterates bars; for each bar it reads `signal` from the row and applies **entry filters** in this order (all can block entry):
   - Flat gate (optional): `p_flat > max_flat_proba` → skip.
   - **min_max_proba:** `current_max_proba = max(p_long, p_short, p_flat)`; if `current_max_proba < min_max_proba` → `skip_entry_filter = True`.
   - **max_entropy:** entropy `H = -sum(p*log2(p))` over (p_long, p_short, p_flat); if `H > max_entropy` → `skip_entry_filter = True`.
   - **min_proba_gap (Phase D10):** `spread = top1_proba - top2_proba`; if `spread < min_proba_gap` → `skip_entry_filter = True`.
   - Cooldown, hysteresis (enter_long_th / enter_short_th), Guard v2, Stage-2 cap, etc.  
   If `skip_entry_filter` is True → `entry_allowed = False` → **no position is opened** for that bar.

4. **Hysteresis (threshold) check**  
   After the above, entry still requires: for LONG `p_long >= enter_long_th`, for SHORT `p_short >= enter_short_th` (otherwise blocked by hysteresis).

**Variables used for entry (summary):**
- **Probabilities:** `p_long`, `p_short`, `p_flat` (from `proba_long_arr[i]`, `proba_short_arr[i]`, and `1 - p_long - p_short`).
- **Derived:** `current_max_proba = max(p_long, p_short, p_flat)`, `current_entropy`, `entry_proba_gap_val = top1 - top2`.
- **Thresholds:** `min_max_proba`, `max_entropy`, `min_proba_gap`, `enter_long_th`, `enter_short_th`.

**Code references:**
- Signal generation: `src/backtest/ml_backtest_engine_impl.py` — `generate_signals()` (e.g. LstmAttnBacktestEngine, ~406–498); `run_backtest()` calls `generate_signals()` then `execute_trades()` (1068–1095).
- Entry filters and `skip_entry_filter` / `entry_allowed`: `src/backtest/ml_backtest_engines.py`
  - Lines 2147–2171: flat gate, min_max_proba, max_entropy.
  - Lines 2173–2181: min_proba_gap (spread) filter.
  - Lines 2367–2369: `if skip_entry_filter: entry_allowed = False`; then `if entry_allowed:` block for opening position.
  - Lines 2304–2319: hysteresis (enter_long_th / enter_short_th).

---

## D) Location of spread filter (Phase D10)

- **Where applied:** `src/backtest/ml_backtest_engines.py`, inside the entry-decision block of `execute_trades()`, **before** any position is opened.
- **Exact logic (snippet):**
  - Sort the three probabilities: `_sorted_probs = sorted([p_long, p_short, p_flat], reverse=True)`.
  - `entry_top1_val = _sorted_probs[0]`, `entry_top2_val = _sorted_probs[1]`.
  - `entry_proba_gap_val = entry_top1_val - entry_top2_val` (i.e. spread = top1 − top2).
  - If `min_proba_gap is not None and min_proba_gap > 0 and entry_proba_gap_val < min_proba_gap`: set `skip_entry_filter = True`, increment `skipped_by_proba_gap`, append to `spread_rejected_list`.
- **Parameter flow:**  
  `run_backtest_7d(..., min_proba_gap=...)` → `engine.run_backtest(..., min_proba_gap=min_proba_gap)` → passed into `execute_trades(..., min_proba_gap=min_proba_gap)` (see `scripts/run_tcn_label_sweep_v2.py` 272, 344; `ml_backtest_engine_impl.py` 880, 1113; `ml_backtest_engines.py` 927).
- **Statistics:** Same file: `skipped_by_proba_gap`, `filtered_trade_count`, `filtered_trade_ratio`, `mean_spread_of_executed_trades`, `mean_spread_of_rejected_trades` are computed and returned (e.g. 3229, 3236–3242). So the spread filter is used **both to gate entry and for statistics**.

**Code references:**
- `src/backtest/ml_backtest_engines.py`  
  - Lines 2173–2181: spread computation and gate (`min_proba_gap` check).  
  - Lines 2697–2718: when a trade is actually opened, `entry_proba_gap_val` is stored on the position and appended to `spread_executed_list`.  
  - Lines 3229, 3236–3242: result dict with `min_proba_gap`, `by_proba_gap`, `filtered_trade_count`, `filtered_trade_ratio`, mean spread of executed/rejected.

---

## E) Conclusion: Does the spread filter actually affect trading decisions?

**Yes.** The confidence spread filter (`min_proba_gap`) **directly gates trade entry**:

1. It is applied in the same entry-decision path as `min_max_proba` and `max_entropy`, **before** position creation.
2. When `entry_proba_gap_val < min_proba_gap`, the code sets `skip_entry_filter = True`, which leads to `entry_allowed = False` (lines 2367–2369), so **no position is opened** for that bar.
3. Only entries that pass the spread check (and other filters) can open a position; when they do, their spread is recorded in `spread_executed_list` and on the position for stats and export.

So the spread filter is not only for statistics — it is an **active entry filter** that prevents opening a trade when the confidence spread (top1 − top2) is below `min_proba_gap`. Phase D10 results (e.g. unchanged 720d cost_on with higher `filtered_trade_ratio` for gap 0.03–0.10) are consistent with the filter reducing the number of executed trades while not improving 720d performance in the tested setup.
