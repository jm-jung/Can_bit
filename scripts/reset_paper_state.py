"""
Paper trading state를 정식 관찰 1일차 baseline으로 초기화.

Usage:
    python -m scripts.reset_paper_state

기존 state → data/state/archive/paper_trading_state_YYYYMMDD_HHMMSS.json
Discord 미전송, 콘솔 요약만 출력.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_daily_paper import _print_reset_summary, reset_paper_observation_state


def main() -> None:
    result = reset_paper_observation_state()
    _print_reset_summary(result)


if __name__ == "__main__":
    main()
