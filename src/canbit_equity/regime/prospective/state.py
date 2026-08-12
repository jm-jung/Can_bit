"""State helpers."""
from __future__ import annotations

from typing import Any, Dict

from canbit_equity.regime.prospective.config import PROSP_ROOT
from canbit_equity.regime.prospective.file_lock import atomic_write_json


def load_state() -> Dict[str, Any]:
    path = PROSP_ROOT / "state/prospective_state.json"
    if not path.exists():
        return {}
    import json

    return json.loads(path.read_text())


def save_state(state: Dict[str, Any]) -> None:
    atomic_write_json(PROSP_ROOT / "state/prospective_state.json", state)
