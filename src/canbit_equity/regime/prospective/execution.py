"""Execution timing helpers (shadow only — no broker)."""
from __future__ import annotations

# Position effective at next session open; no order/private calls.
EXECUTION_ENABLED = False
ORDER_CALLS = 0
PRIVATE_CALLS = 0
