"""
3-class signal decision policy (SSOT - Single Source of Truth).

This module provides a unified function for converting 3-class model outputs
(probabilities or logits) into trading signals (LONG/FLAT/SHORT).

All backtest engines and real-time strategies should use this function
to ensure consistent signal generation logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

logger = logging.getLogger(__name__)

Signal = Literal["LONG", "SHORT", "FLAT"]
ReasonCode = Literal[
    "ARGMAX",
    "MARGIN_TO_FLAT",
    "LOWCONF_TO_FLAT",
    "COOLDOWN_BLOCK",
    "FLIP_BLOCK",
    "THRESHOLD_LONG",
    "THRESHOLD_SHORT",
    "FLAT_THRESHOLD",
]


@dataclass
class ActionDecisionConfig:
    """Configuration for 3-class action decision policy."""

    # Threshold-based rules
    long_threshold: float = 0.5
    short_threshold: Optional[float] = None

    # Margin-based rules (for argmax with margin)
    margin: float = 0.0  # If top1_prob - top2_prob < margin, choose FLAT
    min_confidence: float = 0.0  # If top1_prob < min_confidence, choose FLAT

    # Flat threshold (uncertainty filter)
    flat_threshold: Optional[float] = None  # If p_flat >= flat_threshold, force FLAT

    # Position stability rules
    cooldown_bars: int = 0  # N bars after trade before new entry allowed
    require_flip_via_flat: bool = False  # If True, LONG→SHORT must go through FLAT

    # Strategy mode
    long_only: bool = False
    short_only: bool = False


@dataclass
class ActionDecisionState:
    """State tracking for action decision (cooldown, last action, etc.)."""

    last_action: Optional[Signal] = None
    last_trade_idx: Optional[int] = None
    current_idx: int = 0


@dataclass
class ActionDecisionResult:
    """Result of action decision with debug information."""

    action: Signal
    reason_code: ReasonCode
    debug_info: dict


def decide_action_3class(
    proba: dict[str, float] | tuple[float, float, float],
    config: ActionDecisionConfig,
    state: Optional[ActionDecisionState] = None,
    idx: Optional[int] = None,
    ts: Optional[str] = None,
) -> ActionDecisionResult:
    """
    Decide action from 3-class probabilities (SSOT function).

    This is the single source of truth for converting 3-class model outputs
    into trading signals. All backtest engines and real-time strategies
    should use this function.

    Args:
        proba: Either:
            - dict with keys "p_long", "p_flat", "p_short"
            - tuple of (p_long, p_flat, p_short)
        config: Policy configuration
        state: Optional state tracking (for cooldown/flip rules)
        idx: Optional index for logging
        ts: Optional timestamp for logging

    Returns:
        ActionDecisionResult with action, reason_code, and debug_info

    Policy Rules (in order):
    1. Flat threshold: If p_flat >= flat_threshold, force FLAT
    2. Cooldown: If within cooldown_bars after last trade, block new entry
    3. Flip rule: If require_flip_via_flat and action would flip position, block
    4. Margin rule: If top1_prob - top2_prob < margin, choose FLAT
    5. Confidence rule: If top1_prob < min_confidence, choose FLAT
    6. Threshold rule: Apply long_threshold/short_threshold
    7. Argmax: Default to argmax(prob)

    Debug info includes:
    - p_long, p_flat, p_short
    - top1_label, top1_prob, top2_label, top2_prob
    - margin, min_confidence, cooldown_left
    - reason_code, final_action
    """
    # Parse probabilities
    if isinstance(proba, dict):
        p_long = proba.get("p_long", 0.0)
        p_flat = proba.get("p_flat", 0.0)
        p_short = proba.get("p_short", 0.0)
    elif isinstance(proba, tuple) and len(proba) == 3:
        p_long, p_flat, p_short = proba
    else:
        raise ValueError(
            f"proba must be dict or tuple of 3 floats, got {type(proba)}"
        )

    # Validate probabilities
    proba_sum = p_long + p_flat + p_short
    if abs(proba_sum - 1.0) > 0.01:
        logger.warning(
            f"[POLICY] Probabilities don't sum to 1: sum={proba_sum:.4f}, "
            f"p_long={p_long:.4f}, p_flat={p_flat:.4f}, p_short={p_short:.4f}"
        )
        # Normalize
        total = p_long + p_flat + p_short
        if total > 0:
            p_long /= total
            p_flat /= total
            p_short /= total
        else:
            # All zero - default to FLAT
            p_long = 0.0
            p_flat = 1.0
            p_short = 0.0

    # Check for NaN/Inf
    if np.isnan(p_long) or np.isnan(p_flat) or np.isnan(p_short):
        logger.error(
            f"[POLICY] NaN detected in probabilities: p_long={p_long}, "
            f"p_flat={p_flat}, p_short={p_short}. Defaulting to FLAT."
        )
        return ActionDecisionResult(
            action="FLAT",
            reason_code="LOWCONF_TO_FLAT",
            debug_info={
                "p_long": p_long,
                "p_flat": p_flat,
                "p_short": p_short,
                "top1_label": "FLAT",
                "top1_prob": 1.0,
                "top2_label": None,
                "top2_prob": 0.0,
                "margin": 0.0,
                "min_confidence": config.min_confidence,
                "cooldown_left": 0,
                "reason_code": "LOWCONF_TO_FLAT",
                "final_action": "FLAT",
            },
        )

    # Find top 2 probabilities
    proba_dict = {"LONG": p_long, "FLAT": p_flat, "SHORT": p_short}
    sorted_proba = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)
    top1_label, top1_prob = sorted_proba[0]
    top2_label, top2_prob = sorted_proba[1] if len(sorted_proba) > 1 else (None, 0.0)

    # Calculate margin
    margin_actual = top1_prob - top2_prob

    # Initialize state if not provided
    if state is None:
        state = ActionDecisionState()
    if idx is None:
        idx = state.current_idx

    # Calculate cooldown left
    cooldown_left = 0
    if state.last_trade_idx is not None and config.cooldown_bars > 0:
        bars_since_trade = idx - state.last_trade_idx
        cooldown_left = max(0, config.cooldown_bars - bars_since_trade)

    # Rule 1: Flat threshold (uncertainty filter)
    if config.flat_threshold is not None and p_flat >= config.flat_threshold:
        return ActionDecisionResult(
            action="FLAT",
            reason_code="FLAT_THRESHOLD",
            debug_info={
                "p_long": p_long,
                "p_flat": p_flat,
                "p_short": p_short,
                "top1_label": top1_label,
                "top1_prob": top1_prob,
                "top2_label": top2_label,
                "top2_prob": top2_prob,
                "margin": margin_actual,
                "min_confidence": config.min_confidence,
                "cooldown_left": cooldown_left,
                "reason_code": "FLAT_THRESHOLD",
                "final_action": "FLAT",
            },
        )

    # Rule 2: Cooldown check (only blocks new entries, not exits)
    if cooldown_left > 0:
        # Cooldown only blocks new entries, not exits to FLAT
        # If we're in a position and want to exit, allow it
        if state.last_action in ("LONG", "SHORT"):
            # Allow exit to FLAT even during cooldown
            if top1_label == "FLAT":
                return ActionDecisionResult(
                    action="FLAT",
                    reason_code="ARGMAX",
                    debug_info={
                        "p_long": p_long,
                        "p_flat": p_flat,
                        "p_short": p_short,
                        "top1_label": top1_label,
                        "top1_prob": top1_prob,
                        "top2_label": top2_label,
                        "top2_prob": top2_prob,
                        "margin": margin_actual,
                        "min_confidence": config.min_confidence,
                        "cooldown_left": cooldown_left,
                        "reason_code": "ARGMAX",
                        "final_action": "FLAT",
                    },
                )
        # Block new entry during cooldown
        return ActionDecisionResult(
            action="FLAT",
            reason_code="COOLDOWN_BLOCK",
            debug_info={
                "p_long": p_long,
                "p_flat": p_flat,
                "p_short": p_short,
                "top1_label": top1_label,
                "top1_prob": top1_prob,
                "top2_label": top2_label,
                "top2_prob": top2_prob,
                "margin": margin_actual,
                "min_confidence": config.min_confidence,
                "cooldown_left": cooldown_left,
                "reason_code": "COOLDOWN_BLOCK",
                "final_action": "FLAT",
            },
        )

    # Rule 3: Flip rule check
    if config.require_flip_via_flat and state.last_action is not None:
        if state.last_action == "LONG" and top1_label == "SHORT":
            return ActionDecisionResult(
                action="FLAT",
                reason_code="FLIP_BLOCK",
                debug_info={
                    "p_long": p_long,
                    "p_flat": p_flat,
                    "p_short": p_short,
                    "top1_label": top1_label,
                    "top1_prob": top1_prob,
                    "top2_label": top2_label,
                    "top2_prob": top2_prob,
                    "margin": margin_actual,
                    "min_confidence": config.min_confidence,
                    "cooldown_left": cooldown_left,
                    "reason_code": "FLIP_BLOCK",
                    "final_action": "FLAT",
                },
            )
        elif state.last_action == "SHORT" and top1_label == "LONG":
            return ActionDecisionResult(
                action="FLAT",
                reason_code="FLIP_BLOCK",
                debug_info={
                    "p_long": p_long,
                    "p_flat": p_flat,
                    "p_short": p_short,
                    "top1_label": top1_label,
                    "top1_prob": top1_prob,
                    "top2_label": top2_label,
                    "top2_prob": top2_prob,
                    "margin": margin_actual,
                    "min_confidence": config.min_confidence,
                    "cooldown_left": cooldown_left,
                    "reason_code": "FLIP_BLOCK",
                    "final_action": "FLAT",
                },
            )

    # Rule 4: Margin rule
    if config.margin > 0 and margin_actual < config.margin:
        return ActionDecisionResult(
            action="FLAT",
            reason_code="MARGIN_TO_FLAT",
            debug_info={
                "p_long": p_long,
                "p_flat": p_flat,
                "p_short": p_short,
                "top1_label": top1_label,
                "top1_prob": top1_prob,
                "top2_label": top2_label,
                "top2_prob": top2_prob,
                "margin": margin_actual,
                "min_confidence": config.min_confidence,
                "cooldown_left": cooldown_left,
                "reason_code": "MARGIN_TO_FLAT",
                "final_action": "FLAT",
            },
        )

    # Rule 5: Confidence rule
    if config.min_confidence > 0 and top1_prob < config.min_confidence:
        return ActionDecisionResult(
            action="FLAT",
            reason_code="LOWCONF_TO_FLAT",
            debug_info={
                "p_long": p_long,
                "p_flat": p_flat,
                "p_short": p_short,
                "top1_label": top1_label,
                "top1_prob": top1_prob,
                "top2_label": top2_label,
                "top2_prob": top2_prob,
                "margin": margin_actual,
                "min_confidence": config.min_confidence,
                "cooldown_left": cooldown_left,
                "reason_code": "LOWCONF_TO_FLAT",
                "final_action": "FLAT",
            },
        )

    # Rule 6: Threshold-based decision
    # Apply thresholds if set, otherwise use argmax
    if top1_label == "LONG":
        if config.long_threshold is not None and p_long >= config.long_threshold:
            if config.long_only:
                action = "LONG"
                reason_code = "THRESHOLD_LONG"
            elif config.short_only:
                action = "FLAT"
                reason_code = "MARGIN_TO_FLAT"
            else:
                action = "LONG"
                reason_code = "THRESHOLD_LONG"
        else:
            # Below threshold, check if SHORT qualifies
            if config.short_threshold is not None and p_short >= config.short_threshold:
                if config.short_only:
                    action = "SHORT"
                    reason_code = "THRESHOLD_SHORT"
                elif config.long_only:
                    action = "FLAT"
                    reason_code = "MARGIN_TO_FLAT"
                else:
                    action = "SHORT"
                    reason_code = "THRESHOLD_SHORT"
            else:
                # Neither qualifies, use argmax
                action = top1_label
                reason_code = "ARGMAX"
    elif top1_label == "SHORT":
        if config.short_threshold is not None and p_short >= config.short_threshold:
            if config.short_only:
                action = "SHORT"
                reason_code = "THRESHOLD_SHORT"
            elif config.long_only:
                action = "FLAT"
                reason_code = "MARGIN_TO_FLAT"
            else:
                action = "SHORT"
                reason_code = "THRESHOLD_SHORT"
        else:
            # Below threshold, check if LONG qualifies
            if config.long_threshold is not None and p_long >= config.long_threshold:
                if config.long_only:
                    action = "LONG"
                    reason_code = "THRESHOLD_LONG"
                elif config.short_only:
                    action = "FLAT"
                    reason_code = "MARGIN_TO_FLAT"
                else:
                    action = "LONG"
                    reason_code = "THRESHOLD_LONG"
            else:
                # Neither qualifies, use argmax
                action = top1_label
                reason_code = "ARGMAX"
    else:  # top1_label == "FLAT"
        # FLAT is top, but check if LONG or SHORT qualify by threshold
        if config.long_threshold is not None and p_long >= config.long_threshold:
            if config.long_only:
                action = "LONG"
                reason_code = "THRESHOLD_LONG"
            elif config.short_only:
                action = "FLAT"
                reason_code = "ARGMAX"
            else:
                action = "LONG"
                reason_code = "THRESHOLD_LONG"
        elif config.short_threshold is not None and p_short >= config.short_threshold:
            if config.short_only:
                action = "SHORT"
                reason_code = "THRESHOLD_SHORT"
            elif config.long_only:
                action = "FLAT"
                reason_code = "ARGMAX"
            else:
                action = "SHORT"
                reason_code = "THRESHOLD_SHORT"
        else:
            # FLAT is top and no threshold qualifies
            action = "FLAT"
            reason_code = "ARGMAX"

    # Final check: long_only / short_only filters
    if config.long_only and action == "SHORT":
        action = "FLAT"
        reason_code = "MARGIN_TO_FLAT"
    if config.short_only and action == "LONG":
        action = "FLAT"
        reason_code = "MARGIN_TO_FLAT"

    return ActionDecisionResult(
        action=action,
        reason_code=reason_code,
        debug_info={
            "p_long": p_long,
            "p_flat": p_flat,
            "p_short": p_short,
            "top1_label": top1_label,
            "top1_prob": top1_prob,
            "top2_label": top2_label,
            "top2_prob": top2_prob,
            "margin": margin_actual,
            "min_confidence": config.min_confidence,
            "cooldown_left": cooldown_left,
            "reason_code": reason_code,
            "final_action": action,
        },
    )

