#!/usr/bin/env python3
"""
1m 최신 데이터를 기준으로 5m_full을 동기화한다.

목적:
- meta metrics freshness drift 방지
- refresh/fr2 최신 eval이 사용할 5m 입력이 1m과 동기화되게 보장

사용:
python -m scripts.sync_ohlcv_5m
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.resample_ohlcv import resample_ohlcv


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _max_ts_from_csv(csv_path: Path) -> pd.Timestamp:
    df = pd.read_csv(csv_path, usecols=["timestamp"])
    ts = pd.to_datetime(df["timestamp"])
    return ts.max()


def sync_5m(symbol: str = "BTCUSDT") -> None:
    path_1m = PROJECT_ROOT / "src" / "data" / "btc_ohlcv.csv"
    path_5m = PROJECT_ROOT / "data" / "ohlcv" / "BTCUSDT_5m_full.csv"

    path_5m.parent.mkdir(parents=True, exist_ok=True)
    if not path_1m.exists():
        raise FileNotFoundError(f"1m source not found: {path_1m}")
    if not path_5m.exists():
        # If 5m file is missing, just build from scratch.
        print("[SYNC] 5m_full missing; rebuilding from 1m...")
        resample_ohlcv(
            from_csv=path_1m,
            to_csv=path_5m,
            from_timeframe="1m",
            to_timeframe="5m",
            symbol=symbol,
        )
        return

    t1 = _max_ts_from_csv(path_1m)
    t5 = _max_ts_from_csv(path_5m)

    print(f"[SYNC] 1m max: {t1}")
    print(f"[SYNC] 5m max: {t5}")

    if t1 <= t5:
        print("[SYNC] already up-to-date")
        return

    print("[SYNC] rebuilding 5m from 1m...")
    resample_ohlcv(
        from_csv=path_1m,
        to_csv=path_5m,
        from_timeframe="1m",
        to_timeframe="5m",
        symbol=symbol,
    )

    t5_new = _max_ts_from_csv(path_5m)
    print(f"[SYNC] 5m updated to: {t5_new}")


def main() -> int:
    sync_5m()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

