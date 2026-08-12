from __future__ import annotations

import json
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from src.trading.meta.meta_config import FR2MetaConfig
from src.trading.meta.meta_metrics import (
    FR2RollingMetrics,
    compute_scores,
    try_load_latest_metrics_from_state_log,
)
from src.trading.meta.meta_state import FR2MetaState, load_state, save_state


MetaState = Literal["FULL", "REDUCED", "OFF"]


def _ensure_state_log_header(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        # Keep append-compatible behavior: if required fields are present, do not rotate/recreate file.
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing = next(reader, [])
            if existing and all(col in existing for col in fieldnames):
                return
            # Missing required columns: extend header while preserving existing rows.
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            merged = list(existing) if existing else []
            for col in fieldnames:
                if col not in merged:
                    merged.append(col)
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=merged)
                w.writeheader()
                for row in rows:
                    out = {k: row.get(k, "") for k in merged}
                    w.writerow(out)
            return
        except Exception:
            # If we cannot validate, do not block writes.
            return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()


def _parse_iso_ts_to_utc(s: str) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class FR2MetaLayer:
    """
    운영 경로에서 호출되는 Meta Layer.

    - 기본 상태는 REDUCED (reduced-first)
    - 신규 진입만 제어 (OFF면 entry 차단). Exit는 막지 않는다.
    - 상태/스트릭은 snapshot으로 persistence.
    - metrics는 기본적으로 state_log.csv(offline/주기 갱신)에서 읽는 경로를 우선 제공.
    """

    def __init__(self, config: FR2MetaConfig):
        self.cfg = config
        self.state = load_state(self.cfg.snapshot_path, default_state=self.cfg.initial_state)
        self._last_metrics_snapshot: dict = {}

    def get_current_state(self) -> MetaState:
        s = (self.state.current_state or self.cfg.initial_state).upper()
        if s not in ("FULL", "REDUCED", "OFF"):
            return "REDUCED"
        return s  # type: ignore[return-value]

    def _candidate_state(
        self,
        score: float,
        alpha_score: float,
        trades_60d: int,
        alpha_off_triggered: bool,
        reduced_streak_off_triggered: bool,
    ) -> tuple[MetaState, str]:
        # Trade protection: if too few trades, avoid FULL.
        if trades_60d < self.cfg.trades_60d_min:
            return "REDUCED", "trade_protection(trades_60d<min)"

        # Optional alpha OFF rule
        if alpha_off_triggered:
            return "OFF", "alpha_off_rule"

        # Optional reduced-streak OFF rule
        if reduced_streak_off_triggered:
            return "OFF", "reduced_streak_rule"

        # Base OFF rule
        if score < self.cfg.off_score_cutoff:
            return "OFF", "score_off_rule"

        # FULL / REDUCED
        if score >= self.cfg.full_score_min and alpha_score >= self.cfg.full_alpha_min:
            return "FULL", "full_rule"
        if score >= self.cfg.reduced_score_min:
            return "REDUCED", "reduced_rule"
        # Fallback
        return "OFF", "fallback"

    def evaluate(self) -> dict:
        """
        Evaluate meta state at most once per cfg.eval_min_interval_seconds.
        Returns snapshot dict for API/status usage.
        """
        now = datetime.utcnow()
        if not self.state.can_eval(now, self.cfg.eval_min_interval_seconds):
            return self.snapshot(extra={"skipped": True})

        now_utc = now.replace(tzinfo=timezone.utc)

        metrics = try_load_latest_metrics_from_state_log(self.cfg.state_log_path)
        if metrics is None:
            # If no metrics, stay reduced-first and log reason.
            self.state.mark_eval(now)
            self.state.transition_reason = "metrics_source_missing"
            save_state(self.cfg.snapshot_path, self.state)

            # Write metrics snapshot for API/health.
            self._last_metrics_snapshot = {
                "metrics_available": False,
                "metrics_timestamp": None,
                "metrics_stale_flag": "MISSING",
                "stale_duration_minutes": None,
            }
            try:
                p = Path(self.cfg.metrics_snapshot_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(self._last_metrics_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
            except Exception:
                pass

            return self.snapshot(extra={"skipped": False, "metrics_available": False})

        # Metric freshness / health
        metrics_ts = _parse_iso_ts_to_utc(metrics.asof_ts)
        stale_duration_minutes: float | None = None
        if metrics_ts is not None:
            stale_duration_minutes = (now_utc - metrics_ts).total_seconds() / 60.0

        if stale_duration_minutes is None:
            metrics_stale_flag = "UNKNOWN"
        elif stale_duration_minutes >= self.cfg.metrics_stale_critical_minutes:
            metrics_stale_flag = "CRITICAL"
        elif stale_duration_minutes >= self.cfg.metrics_stale_warning_minutes:
            metrics_stale_flag = "STALE"
        else:
            metrics_stale_flag = "FRESH"

        score, alpha_score = compute_scores(metrics, self.cfg.w_60d, self.cfg.w_90d)

        # Persist metrics snapshot (for API + Daily Report + Discord)
        self._last_metrics_snapshot = {
            "metrics_available": True,
            "metrics_timestamp": metrics.asof_ts,
            "metrics_stale_flag": metrics_stale_flag,
            "stale_duration_minutes": stale_duration_minutes,
            "cost_on_60d": metrics.cost_on_60d,
            "cost_on_90d": metrics.cost_on_90d,
            "alpha_fee_ratio_60d": metrics.alpha_fee_ratio_60d,
            "alpha_fee_ratio_90d": metrics.alpha_fee_ratio_90d,
            "score": score,
            "alpha_score": alpha_score,
            "trades_60d": metrics.trades_60d,
            "warning_minutes": self.cfg.metrics_stale_warning_minutes,
            "critical_minutes": self.cfg.metrics_stale_critical_minutes,
            "evaluated_at": now_utc.isoformat(),
        }
        try:
            p = Path(self.cfg.metrics_snapshot_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._last_metrics_snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

        # Track alpha-off streak
        alpha_off_triggered = False
        if self.cfg.alpha_off_enabled:
            if alpha_score < self.cfg.alpha_off_cutoff:
                self.state.off_condition_streak += 1
            else:
                self.state.off_condition_streak = 0
            if self.state.off_condition_streak >= self.cfg.alpha_off_consecutive_needed:
                alpha_off_triggered = True

        # reduced streak tracking (based on score range)
        if score < self.cfg.full_score_min:
            self.state.reduced_streak += 1
        else:
            self.state.reduced_streak = 0

        reduced_streak_off_triggered = (
            self.cfg.reduced_streak_rule_enabled
            and self.state.reduced_streak >= self.cfg.reduced_streak_off_threshold
        )

        next_candidate, reason = self._candidate_state(
            score=score,
            alpha_score=alpha_score,
            trades_60d=metrics.trades_60d,
            alpha_off_triggered=alpha_off_triggered,
            reduced_streak_off_triggered=reduced_streak_off_triggered,
        )

        current = self.get_current_state()

        # Hysteresis for FULL/OFF: require consecutive confirmations.
        full_ok = next_candidate == "FULL"
        off_ok = next_candidate == "OFF"

        if full_ok:
            self.state.full_condition_streak += 1
        else:
            self.state.full_condition_streak = 0
        if off_ok:
            # Note: alpha-off uses off_condition_streak too; keep separate by using the same streak
            # counter only for OFF confirmations from the candidate evaluation.
            self.state.off_condition_streak += 1
        else:
            # do not reset if alpha-off is enabled and is tracking; reset only if not in alpha-off mode
            if not self.cfg.alpha_off_enabled:
                self.state.off_condition_streak = 0

        decided: MetaState = current
        transition_reason: str | None = None

        if next_candidate == "FULL" and self.state.full_condition_streak >= self.cfg.hysteresis_needed:
            decided = "FULL"
            transition_reason = f"hysteresis_full({reason})"
        elif next_candidate == "OFF" and self.state.off_condition_streak >= self.cfg.hysteresis_needed:
            decided = "OFF"
            transition_reason = f"hysteresis_off({reason})"
        else:
            # default fallback
            decided = "REDUCED" if next_candidate != "FULL" else current

        # Startup safety: never jump to FULL if initial state and no prior transitions.
        if self.state.last_transition_ts is None and decided == "FULL":
            decided = "REDUCED"
            transition_reason = "startup_reduced_first"

        final_reason = transition_reason or reason
        latest_invalid_probe = False

        # Expose why we are not using the very latest metrics source.
        # - 최신 probe가 invalid이면 refresh 파이프라인이 source_info에 이유를 남긴다.
        # - 구형 payload(추가 키 없음)는 timestamp mismatch로 fallback 여부를 추정한다.
        try:
            metrics_source_latest_path = Path(self.cfg.state_log_path).parent / "meta_metrics_source_latest.json"
            if metrics_source_latest_path.exists():
                payload = json.loads(metrics_source_latest_path.read_text(encoding="utf-8"))
                latest_asof = payload.get("asof_ts") or payload.get("timestamp")
                src_info = payload.get("source_info") or {}

                first_probe_valid = src_info.get("first_probe_valid")
                fb_reason = src_info.get("fallback_reason")
                latest_probe_trades_60d = src_info.get("latest_probe_trades_60d")
                latest_probe_alpha = src_info.get("latest_probe_alpha_fee_ratio_60d")

                # Preferred path: explicit probe flags from refresh pipeline.
                if first_probe_valid is False:
                    latest_invalid_probe = True
                    final_reason = (
                        f"{final_reason}; latest_invalid"
                        + (f"(reason={fb_reason})" if fb_reason else "")
                    )

                    if latest_probe_trades_60d is not None:
                        try:
                            if int(latest_probe_trades_60d) == 0:
                                final_reason = f"{final_reason}; zero_trades"
                        except Exception:
                            pass

                    # refresh json sanitizer turns NaN/Inf into null, so None means alpha_nan.
                    if latest_probe_alpha is None:
                        final_reason = f"{final_reason}; alpha_nan"
                    else:
                        try:
                            a = float(latest_probe_alpha)
                            if not np.isfinite(a):
                                final_reason = f"{final_reason}; alpha_nan"
                        except Exception:
                            pass
                else:
                    # Fallback path: infer from timestamp mismatch (legacy behavior).
                    if latest_asof and str(latest_asof) != str(metrics.asof_ts):
                        fb_reason2 = src_info.get("fallback_reason")
                        latest_invalid_probe = True
                        final_reason = (
                            f"{final_reason}; latest_invalid"
                            + (f"(reason={fb_reason2})" if fb_reason2 else "")
                        )
        except Exception:
            # Reason annotation should never break trading logic.
            pass

        # If metrics are critical but the input latest-probe itself was invalid,
        # prefer surfacing that rather than a stale-only message.
        if metrics_stale_flag and metrics_stale_flag not in ("FRESH", ""):
            if not (metrics_stale_flag == "CRITICAL" and latest_invalid_probe):
                final_reason = f"{final_reason}; metrics_{str(metrics_stale_flag).lower()}"

        # Apply transition
        if decided != current:
            self.state.current_state = decided
            self.state.last_transition_ts = now.isoformat()

        # Always update transition_reason so "no_metrics_available" doesn't linger forever.
        self.state.transition_reason = final_reason

        self.state.mark_eval(now)
        save_state(self.cfg.snapshot_path, self.state)

        # Append state log row (operational)
        self._append_state_log(
            now=now,
            metrics=metrics,
            score=score,
            alpha_score=alpha_score,
            current_state=current,
            next_state_candidate=next_candidate,
            decided_state=decided,
            reason=final_reason,
        )

        return self.snapshot(
            extra={
                "skipped": False,
                "metrics_available": True,
                "metrics_stale_flag": metrics_stale_flag,
                "score": score,
                "alpha_score": alpha_score,
                "candidate": next_candidate,
            }
        )

    def evaluate_force(self) -> dict:
        """
        운영 검증/디버깅용: meta eval의 최소 간격(rate-limit)을 무시하고 즉시 evaluate 수행.
        """
        self.state.last_eval_ts = None
        save_state(self.cfg.snapshot_path, self.state)
        return self.evaluate()

    def position_multiplier(self) -> float:
        s = self.get_current_state()
        if s == "FULL":
            return self.cfg.full_multiplier
        if s == "REDUCED":
            # downshift if reduced streak rule is enabled
            if self.cfg.reduced_streak_rule_enabled and self.state.reduced_streak >= self.cfg.reduced_streak_off_threshold:
                return self.cfg.reduced_streak_downshift_multiplier
            return self.cfg.reduced_multiplier
        return self.cfg.off_multiplier

    def can_open_new_position(self) -> tuple[bool, str]:
        s = self.get_current_state()
        if s == "OFF":
            return False, "meta_state_off"
        return True, f"meta_state_{s.lower()}"

    def snapshot(self, extra: dict | None = None) -> dict:
        snap = {
            "strategy_name": self.cfg.strategy_name,
            "current_state": self.get_current_state(),
            "position_multiplier": self.position_multiplier(),
            "full_condition_streak": self.state.full_condition_streak,
            "off_condition_streak": self.state.off_condition_streak,
            "reduced_streak": self.state.reduced_streak,
            "last_transition_ts": self.state.last_transition_ts,
            "transition_reason": self.state.transition_reason,
            "last_eval_ts": self.state.last_eval_ts,
            "config": asdict(self.cfg),
        }
        if extra:
            snap.update(extra)
        return snap

    def _append_state_log(
        self,
        now: datetime,
        metrics: FR2RollingMetrics,
        score: float,
        alpha_score: float,
        current_state: MetaState,
        next_state_candidate: MetaState,
        decided_state: MetaState,
        reason: str,
    ) -> None:
        p = Path(self.cfg.state_log_path)
        fields = [
            "timestamp",
            "strategy_name",
            "current_state",
            "next_state_candidate",
            "position_multiplier",
            "cost_on_60d",
            "cost_on_90d",
            "alpha_fee_ratio_60d",
            "alpha_fee_ratio_90d",
            "score",
            "alpha_score",
            "trades_60d",
            "full_condition_streak",
            "off_condition_streak",
            "reduced_streak",
            "last_transition_ts",
            "transition_reason",
        ]
        _ensure_state_log_header(p, fields)
        with p.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writerow(
                {
                    "timestamp": now.isoformat(),
                    "strategy_name": self.cfg.strategy_name,
                    "current_state": decided_state,
                    "next_state_candidate": next_state_candidate,
                    "position_multiplier": self.position_multiplier(),
                    "cost_on_60d": metrics.cost_on_60d,
                    "cost_on_90d": metrics.cost_on_90d,
                    "alpha_fee_ratio_60d": metrics.alpha_fee_ratio_60d,
                    "alpha_fee_ratio_90d": metrics.alpha_fee_ratio_90d,
                    "score": score,
                    "alpha_score": alpha_score,
                    "trades_60d": metrics.trades_60d,
                    "full_condition_streak": self.state.full_condition_streak,
                    "off_condition_streak": self.state.off_condition_streak,
                    "reduced_streak": self.state.reduced_streak,
                    "last_transition_ts": self.state.last_transition_ts,
                    "transition_reason": reason,
                }
            )

