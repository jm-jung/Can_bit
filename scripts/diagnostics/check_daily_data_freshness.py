"""
Check BTCUSDT 5m data freshness for the R7 daily warning-only monitor.

Diagnostics-only: writes freshness status under
data/diagnostics/false_high_r7_daily_monitor/freshness/ and never changes
production, Q2, live execution, orders, or state.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None

KST = ZoneInfo("Asia/Seoul") if ZoneInfo else timezone.utc
DEFAULT_OUTPUT = Path("data/diagnostics/false_high_r7_daily_monitor/freshness")
CANONICAL_PATHS_JSON = Path("data/diagnostics/data_sync/canonical_data_paths.json")
FEATURE_REFRESH_ROOT = Path("data/diagnostics/feature_proba_refresh")


def _json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _candidate_paths() -> List[Path]:
    candidates: List[Path] = []
    canonical = REPO_ROOT / CANONICAL_PATHS_JSON
    if canonical.exists():
        try:
            data = json.loads(canonical.read_text(encoding="utf-8"))
            c5 = data.get("canonical_5m_path")
            if c5:
                candidates.append(Path(c5))
        except Exception:
            pass
    candidates.extend([
        Path("data/market/btcusdt_5m.parquet"),
        Path("data/ohlcv/BTCUSDT_5m_full.csv"),
        Path("data/ohlcv/BTCUSDT_5m.csv"),
        Path("data/cache/BTCUSDT_5m_full.csv"),
        Path("data/cache/BTCUSDT_5m.csv"),
        Path("data/cache/ml_predictions/ml_tcn_BTCUSDT_5m_proba.parquet"),
    ])
    out: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _status_from_age(age_hours: float | None, max_age_hours: float, warn_age_hours: float) -> str:
    if age_hours is None:
        return "MISSING"
    if age_hours <= warn_age_hours:
        return "FRESH"
    if age_hours <= max_age_hours:
        return "WARN"
    return "STALE"


def _age_hours(ts: pd.Timestamp | None, now_utc: datetime) -> float | None:
    if ts is None or pd.isna(ts):
        return None
    latest_naive = pd.Timestamp(ts).tz_localize(None).to_pydatetime()
    return max(0.0, (now_utc.replace(tzinfo=None) - latest_naive).total_seconds() / 3600.0)


def _component(path: Path, now_utc: datetime, max_age_hours: float, warn_age_hours: float) -> Dict[str, Any]:
    ts = _read_latest_ts(path)
    age = _age_hours(ts, now_utc)
    return {
        "path": str(path),
        "latest_ts": "" if ts is None or pd.isna(ts) else str(pd.Timestamp(ts)),
        "age_hours": age,
        "status": _status_from_age(age, max_age_hours, warn_age_hours),
        "exists": (REPO_ROOT / path if not path.is_absolute() else path).exists(),
    }


def _read_latest_ts(path: Path) -> pd.Timestamp | None:
    full = REPO_ROOT / path if not path.is_absolute() else path
    if not full.exists():
        return None
    try:
        if full.suffix.lower() == ".parquet":
            df = pd.read_parquet(full, columns=None)
            cols = [c for c in ["timestamp", "_ts", "ts", "datetime", "open_time"] if c in df.columns]
            if not cols:
                return None
            return pd.to_datetime(df[cols[0]], errors="coerce").max()
        # CSV: only timestamp column is needed, but reading full is acceptable for this small diagnostics check.
        df = pd.read_csv(full)
        cols = [c for c in ["timestamp", "_ts", "ts", "datetime", "open_time"] if c in df.columns]
        if not cols:
            return None
        return pd.to_datetime(df[cols[0]], errors="coerce").max()
    except Exception:
        return None


def check(max_age_hours: float, warn_age_hours: float, output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime.now(timezone.utc)
    now_kst = now_utc.astimezone(KST)
    latest_ts = None
    source_path = ""
    for rel in _candidate_paths():
        ts = _read_latest_ts(rel)
        if ts is not None and not pd.isna(ts):
            latest_ts = ts
            source_path = str(rel)
            break

    age_hours = _age_hours(latest_ts, now_utc)
    ohlcv_status = _status_from_age(age_hours, max_age_hours, warn_age_hours)
    latest_str = "" if latest_ts is None or pd.isna(latest_ts) else str(pd.Timestamp(latest_ts))

    feature = _component(FEATURE_REFRESH_ROOT / "latest_features.parquet", now_utc, max_age_hours, warn_age_hours)
    proba = _component(FEATURE_REFRESH_ROOT / "latest_tcn_proba.parquet", now_utc, max_age_hours, warn_age_hours)
    q2 = _component(FEATURE_REFRESH_ROOT / "latest_q2_diagnostics.parquet", now_utc, max_age_hours, warn_age_hours)
    r7_input = _component(FEATURE_REFRESH_ROOT / "latest_r7_input_frame.parquet", now_utc, max_age_hours, warn_age_hours)
    if ohlcv_status in {"STALE", "MISSING"}:
        overall_status = "OHLCV_STALE" if ohlcv_status == "STALE" else "MISSING"
    elif proba["status"] == "MISSING":
        overall_status = "PROBA_CACHE_MISSING"
    elif q2["status"] == "MISSING":
        overall_status = "Q2_CACHE_MISSING"
    elif r7_input["status"] in {"STALE", "MISSING"}:
        overall_status = "FEATURE_CACHE_STALE"
    elif r7_input["status"] == "WARN" or ohlcv_status == "WARN":
        overall_status = "WARN"
    else:
        overall_status = "FRESH"
    can_forward = overall_status in {"FRESH", "WARN"} and r7_input["status"] in {"FRESH", "WARN"}

    result = {
        "run_ts_utc": now_utc.isoformat(),
        "run_ts_kst": now_kst.isoformat(),
        "latest_data_ts": latest_str,
        "latest_source_path": source_path,
        "age_hours": age_hours,
        "status": overall_status,
        "overall_status": overall_status,
        "ohlcv_freshness_status": ohlcv_status,
        "feature_freshness_status": feature["status"],
        "proba_freshness_status": proba["status"],
        "q2_freshness_status": q2["status"],
        "r7_input_freshness_status": r7_input["status"],
        "latest_ohlcv_ts": latest_str,
        "latest_feature_ts": feature["latest_ts"],
        "latest_proba_ts": proba["latest_ts"],
        "latest_q2_ts": q2["latest_ts"],
        "latest_r7_input_ts": r7_input["latest_ts"],
        "ohlcv_age_hours": age_hours,
        "feature_age_hours": feature["age_hours"],
        "proba_age_hours": proba["age_hours"],
        "q2_age_hours": q2["age_hours"],
        "r7_input_age_hours": r7_input["age_hours"],
        "can_use_for_forward_validation": can_forward,
        "components": {
            "ohlcv": {"path": source_path, "latest_ts": latest_str, "age_hours": age_hours, "status": ohlcv_status},
            "feature": feature,
            "proba": proba,
            "q2": q2,
            "r7_input": r7_input,
        },
        "max_age_hours": max_age_hours,
        "warn_age_hours": warn_age_hours,
        "production_changed": False,
        "q2_changed": False,
        "state_changed": False,
    }
    latest_path = output_dir / "data_freshness_latest.json"
    latest_path.write_text(_json(result), encoding="utf-8")
    (output_dir / "data_freshness_latest.md").write_text(
        "# R7 Daily Data Freshness\n\n```json\n" + _json(result) + "\n```\n",
        encoding="utf-8",
    )
    hist_path = output_dir / "data_freshness_history.csv"
    old = pd.read_csv(hist_path) if hist_path.exists() else pd.DataFrame()
    hist = pd.concat([old, pd.DataFrame([result])], ignore_index=True)
    hist.to_csv(hist_path, index=False)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Check BTCUSDT 5m data freshness.")
    parser.add_argument("--max-age-hours", type=float, default=2.0)
    parser.add_argument("--warn-age-hours", type=float, default=1.0)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = check(args.max_age_hours, args.warn_age_hours, Path(args.output_dir))
    if args.json:
        print(_json(result))
    else:
        print(f"freshness_status={result['status']} latest_data_ts={result['latest_data_ts']} age_hours={result['age_hours']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
