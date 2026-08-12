from __future__ import annotations

import csv
import os
import json
from dataclasses import dataclass
from pathlib import Path
from math import isnan


@dataclass(frozen=True)
class FR2RollingMetrics:
    asof_ts: str
    cost_on_60d: float
    cost_on_90d: float
    alpha_fee_ratio_60d: float
    alpha_fee_ratio_90d: float
    trades_60d: int


def _safe_float(x) -> float:
    try:
        v = float(x)
        return v
    except Exception:
        return float("nan")


def _safe_int(x) -> int:
    try:
        return int(x)
    except Exception:
        return 0


def _read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            return [c.strip() for c in row if c is not None]
    return []


def _read_last_valid_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    last: dict | None = None
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades = _safe_int(row.get("trades_60d"))
                if trades <= 0:
                    continue

                cost60 = _safe_float(row.get("cost_on_60d"))
                cost90 = _safe_float(row.get("cost_on_90d"))
                a60 = _safe_float(row.get("alpha_fee_ratio_60d") or row.get("alpha_60d"))
                a90 = _safe_float(row.get("alpha_fee_ratio_90d") or row.get("alpha_90d"))

                # Avoid NaN metrics source rows (which would propagate to score/alpha_score).
                if isnan(cost60) or isnan(cost90) or isnan(a60) or isnan(a90):
                    continue

                last = row
    except Exception:
        return None
    return last


def is_fr2_offline_legacy_state_log(path: Path) -> bool:
    """
    scripts/run_fr2_meta_layer.py 가 쓰는 오프라인 레거시 스키마인지 판별한다.
    운영용으로 export된 state_log_legacy_*.csv(strategy_name/current_state 등)는 제외한다.
    """
    try:
        header = _read_csv_header(path)
    except Exception:
        return False
    hs = {str(c).strip() for c in header if c}
    if "strategy_name" in hs or "current_state" in hs:
        return False
    need = {"state", "alpha_60d", "trades_60d", "cost_on_60d", "cost_on_90d"}
    return need.issubset(hs)


def resolve_latest_fr2_offline_legacy_path(parent_dir: Path) -> Path | None:
    """mtime 기준 최신 오프라인 레거시 CSV. 없으면 None."""
    cands = [p for p in parent_dir.glob("state_log_legacy_*.csv") if is_fr2_offline_legacy_state_log(p)]
    if not cands:
        return None
    cands.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return cands[0]


def _resolve_metrics_source_path(state_log_path: str) -> Path:
    """
    운영 상태 로그(state_log.csv)는 meta_layer가 append하면서 스키마/값이 변할 수 있음.
    반면 rolling metrics는 기존 스크립트(scripts/run_fr2_meta_layer.py)가 만든 legacy CSV에 들어있음.

    따라서 state_log.csv가 'operational schema'(current_state/position_multiplier 등)를 가지면,
    가장 최근의 state_log_legacy_*.csv를 rolling metrics source로 사용한다.
    """
    p = Path(state_log_path)
    if not p.exists():
        return p

    try:
        header = _read_csv_header(p)
    except Exception:
        return p

    operational_markers = {"current_state", "position_multiplier", "next_state_candidate"}
    if any(m in header for m in operational_markers):
        dir_ = p.parent
        legacy_latest = resolve_latest_fr2_offline_legacy_path(dir_)
        if legacy_latest is not None:
            return legacy_latest

    return p


def try_load_latest_metrics_from_state_log(state_log_path: str) -> FR2RollingMetrics | None:
    """
    운영 경로에서 rolling 60d/90d 백테스트를 매번 돌리는 건 비용이 크다.
    따라서 기본 구현은 '이미 생성된 state_log.csv'에서 최신 지표를 읽어오는 경로를 제공한다.
    (state_log.csv는 별도 스케줄러/운영 루프에서 주기적으로 갱신될 수 있음)
    """
    # Prefer dedicated metrics source snapshot (operational fresh supply),
    # to avoid any self-referential coupling with state_log.csv append rows.
    state_log_path_obj = Path(state_log_path)
    metrics_source_json = state_log_path_obj.parent / "meta_metrics_source_latest.json"
    if metrics_source_json.exists():
        try:
            payload = json.loads(metrics_source_json.read_text(encoding="utf-8"))
            # Support both "timestamp" and "asof_ts"
            asof_ts = payload.get("asof_ts") or payload.get("timestamp")
            if asof_ts:
                trades_60d = _safe_int(payload.get("trades_60d"))
                cost60 = _safe_float(payload.get("cost_on_60d"))
                cost90 = _safe_float(payload.get("cost_on_90d"))
                a60 = _safe_float(payload.get("alpha_fee_ratio_60d") or payload.get("alpha_60d"))
                a90 = _safe_float(payload.get("alpha_fee_ratio_90d") or payload.get("alpha_90d"))
                # Avoid NaN propagation
                if trades_60d > 0 and not any(isnan(x) for x in [cost60, cost90, a60, a90]):
                    return FR2RollingMetrics(
                        asof_ts=str(asof_ts),
                        cost_on_60d=cost60,
                        cost_on_90d=cost90,
                        alpha_fee_ratio_60d=a60,
                        alpha_fee_ratio_90d=a90,
                        trades_60d=trades_60d,
                    )
        except Exception:
            pass

    p = _resolve_metrics_source_path(state_log_path)
    if not p.exists():
        return None
    try:
        last = _read_last_valid_row(p)
        if not last:
            return None

        trades_60d = _safe_int(last.get("trades_60d"))
        if trades_60d <= 0:
            return None

        return FR2RollingMetrics(
            asof_ts=str(last.get("timestamp") or ""),
            cost_on_60d=_safe_float(last.get("cost_on_60d")),
            cost_on_90d=_safe_float(last.get("cost_on_90d")),
            # Legacy file uses alpha_60d/alpha_90d; operational uses alpha_fee_ratio_60d/alpha_fee_ratio_90d.
            alpha_fee_ratio_60d=_safe_float(
                last.get("alpha_fee_ratio_60d") or last.get("alpha_60d")
            ),
            alpha_fee_ratio_90d=_safe_float(
                last.get("alpha_fee_ratio_90d") or last.get("alpha_90d")
            ),
            trades_60d=trades_60d,
        )
    except Exception:
        return None


def compute_scores(
    metrics: FR2RollingMetrics,
    w_60d: float,
    w_90d: float,
) -> tuple[float, float]:
    cost60 = metrics.cost_on_60d if not isnan(metrics.cost_on_60d) else 0.0
    cost90 = metrics.cost_on_90d if not isnan(metrics.cost_on_90d) else 0.0
    a60 = metrics.alpha_fee_ratio_60d if not isnan(metrics.alpha_fee_ratio_60d) else 0.0
    a90 = metrics.alpha_fee_ratio_90d if not isnan(metrics.alpha_fee_ratio_90d) else 0.0
    score = w_60d * cost60 + w_90d * cost90
    alpha_score = w_60d * a60 + w_90d * a90
    return score, alpha_score

