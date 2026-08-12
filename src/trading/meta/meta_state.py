from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


MetaStateName = str  # "FULL" | "REDUCED" | "OFF"


@dataclass
class FR2MetaState:
    current_state: MetaStateName = "REDUCED"
    full_condition_streak: int = 0
    off_condition_streak: int = 0
    reduced_streak: int = 0
    last_transition_ts: str | None = None
    last_eval_ts: str | None = None
    transition_reason: str | None = None

    def mark_eval(self, now: datetime) -> None:
        self.last_eval_ts = now.isoformat()

    def can_eval(self, now: datetime, min_interval_seconds: int) -> bool:
        if not self.last_eval_ts:
            return True
        try:
            last = datetime.fromisoformat(self.last_eval_ts)
        except Exception:
            return True
        return (now - last).total_seconds() >= float(min_interval_seconds)


def load_state(snapshot_path: str, default_state: MetaStateName) -> FR2MetaState:
    p = Path(snapshot_path)
    if not p.exists():
        return FR2MetaState(current_state=default_state)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        st = FR2MetaState(**{k: data.get(k) for k in asdict(FR2MetaState()).keys()})
        # Ensure current_state always set
        if not st.current_state:
            st.current_state = default_state
        return st
    except Exception:
        return FR2MetaState(current_state=default_state)


def save_state(snapshot_path: str, state: FR2MetaState) -> None:
    p = Path(snapshot_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")

