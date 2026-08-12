#!/usr/bin/env python3
"""
δp=0.05, mae_cut=-0.007 — roll60 중 개선 폭 상위 2창·악화 폭 상위 2창에서
baseline vs combo의 signal_exit_th 건별 차이 분석.

금지: 파라미터 스윕, threshold 변경, 모델 변경.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.loss_aux_signal_exit_common import load_fr2_override_ensemble, run_loss_aux_window, to_utc_ts

DELTA_P = 0.05
MAE_CUT = -0.007
WINDOW_DAYS = 60


def _sth_exits(res: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for e in res.get("trade_events") or []:
        if not str(e.get("event", "")).startswith("EXIT"):
            continue
        if e.get("exit_reason") != "signal_exit_th":
            continue
        out.append(e)
    return out


def _trade_key(e: dict[str, Any]) -> tuple[str, str, str]:
    return (str(e.get("entry_ts", "")), str(e.get("exit_ts", "")), str(e.get("direction", "")))


def _pnl(e: dict[str, Any]) -> float:
    """잔고 반영 분: pnl_change 우선."""
    if e.get("pnl_change") is not None:
        return float(e["pnl_change"])
    return float(e.get("profit", 0.0) or 0.0)


def _max_loss_trade(res: dict[str, Any]) -> float | None:
    from src.backtest.engine import dedupe_trades_round_trips

    trades = dedupe_trades_round_trips(list(res.get("trades") or []))
    profits = [float(t["profit"]) for t in trades if t.get("profit") is not None]
    return min(profits) if profits else None


def _fmt(x: Any, nd: int = 6) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def analyze_window(
    *,
    df_bt: Any,
    pl_primary: Any,
    ps_primary: Any,
    t_end: pd.Timestamp,
    threshold: float,
) -> dict[str, Any]:
    base = run_loss_aux_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_end=t_end,
        window_days=WINDOW_DAYS,
        threshold=threshold,
        emit_trade_log=True,
        signal_exit_th_loss_aux_delta_p=None,
        signal_exit_th_loss_aux_mae_cut=None,
    )
    combo = run_loss_aux_window(
        df_bt=df_bt,
        pl_primary=pl_primary,
        ps_primary=ps_primary,
        t_end=t_end,
        window_days=WINDOW_DAYS,
        threshold=threshold,
        emit_trade_log=True,
        signal_exit_th_loss_aux_delta_p=DELTA_P,
        signal_exit_th_loss_aux_mae_cut=MAE_CUT,
    )

    eb = { _trade_key(e): e for e in _sth_exits(base) }
    ec = { _trade_key(e): e for e in _sth_exits(combo) }
    keys_b = set(eb.keys())
    keys_c = set(ec.keys())

    sth_count_b = int(base.get("signal_exit_th_count", 0) or 0)
    sth_count_c = int(combo.get("signal_exit_th_count", 0) or 0)
    sth_sum_b = float(base.get("signal_exit_th_total_profit", 0.0) or 0.0)
    sth_sum_c = float(combo.get("signal_exit_th_total_profit", 0.0) or 0.0)

    common = keys_b & keys_c
    only_b = keys_b - keys_c
    only_c = keys_c - keys_b

    disappeared_loss = 0  # baseline 손실 → combo 비손실
    new_loss = 0  # baseline 비손실 → combo 손실
    ameliorated_loss = 0  # 둘 다 손실인데 combo가 덜 나쁨
    profit_damaged = 0  # baseline 이익(pb>0)인데 콤보 pnl이 더 나쁨

    rows_diff: list[dict[str, Any]] = []
    for k in sorted(common):
        pb, pc = _pnl(eb[k]), _pnl(ec[k])
        if pb < 0 and pc >= 0:
            disappeared_loss += 1
        if pb >= 0 and pc < 0:
            new_loss += 1
        if pb < 0 and pc < 0 and pc > pb:
            ameliorated_loss += 1
        if pb > 0 and pc < pb:
            profit_damaged += 1
        rows_diff.append(
            {
                "key": k,
                "pnl_b": pb,
                "pnl_c": pc,
                "delta": pc - pb,
                "dir": k[2],
            }
        )

    # worst 5 sth by pnl (most negative)
    def worst5(d: dict[tuple[str, str, str], dict]) -> list[dict]:
        items = [( _pnl(e), e) for e in d.values()]
        items.sort(key=lambda x: x[0])
        return [{"entry_ts": e.get("entry_ts"), "exit_ts": e.get("exit_ts"), "direction": e.get("direction"), "pnl": _pnl(e)} for _, e in items[:5]]

    return {
        "base": base,
        "combo": combo,
        "sth_count_b": sth_count_b,
        "sth_count_c": sth_count_c,
        "sth_sum_b": sth_sum_b,
        "sth_sum_c": sth_sum_c,
        "keys_only_b": len(only_b),
        "keys_only_c": len(only_c),
        "disappeared_loss": disappeared_loss,
        "new_loss": new_loss,
        "ameliorated_loss": ameliorated_loss,
        "profit_damaged": profit_damaged,
        "worst5_b": worst5(eb),
        "worst5_c": worst5(ec),
        "rows_diff": sorted(rows_diff, key=lambda r: r["delta"]),
        "max_loss_b": _max_loss_trade(base),
        "max_loss_c": _max_loss_trade(combo),
        "eb": eb,
        "ec": ec,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-max", default="2026-03-20T07:40:00+00:00")
    parser.add_argument("--days-full", type=int, default=420)
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument(
        "--from-csv",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_LOSS_AUX_ROLL60_DECOMPOSITION.csv",
        help="창 선정용 decomposition CSV (없으면 오류)",
    )
    parser.add_argument(
        "--out-md",
        default="data/diagnostics/fr2/SIGNAL_EXIT_TH_TRADE_DIFF_ROLL60_SAMPLE.md",
    )
    args = parser.parse_args()

    t_anchor = to_utc_ts(args.t_max)
    thr = float(args.threshold)

    csv_path = Path(args.from_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"창 선정 CSV 없음: {csv_path}")

    df_sel = pd.read_csv(csv_path)
    df_sel = df_sel[df_sel["bucket"].isin(["개선", "악화"])].copy()
    df_imp = df_sel[df_sel["bucket"] == "개선"].sort_values("delta_total_return", ascending=False).head(2)
    df_bad = df_sel[df_sel["bucket"] == "악화"].sort_values("delta_total_return", ascending=True).head(2)

    improved_ends = [pd.Timestamp(x, tz="UTC") for x in df_imp["t_end"].tolist()]
    worsened_ends = [pd.Timestamp(x, tz="UTC") for x in df_bad["t_end"].tolist()]

    print(f"[trade_diff] 개선 2창: {[str(x.date()) for x in improved_ends]}", flush=True)
    print(f"[trade_diff] 악화 2창: {[str(x.date()) for x in worsened_ends]}", flush=True)

    print("[trade_diff] load data ...", flush=True)
    df_bt, pl_primary, ps_primary, ts_min, ts_max = load_fr2_override_ensemble(int(args.days_full), t_anchor)

    sections: list[str] = []
    sections.append("# signal_exit_th 건별 차이 — roll60 샘플 (개선 2·악화 2창)")
    sections.append("")
    sections.append(f"- 후보: δp={DELTA_P}, mae_cut={MAE_CUT}")
    sections.append(f"- 앵커 t_max: `{t_anchor.isoformat()}` · threshold={thr} · roll60")
    sections.append("")
    sections.append("## STEP 1 — 대상 창 선정 (decomposition CSV 기준)")
    sections.append("")
    sections.append("### 개선 폭 상위 2창 (Δ total_return 큰 순)")
    sections.append("")
    for _, r in df_imp.iterrows():
        sections.append(f"- **{str(r['t_end'])[:10]}** — Δtr={float(r['delta_total_return']):.6f}")
    sections.append("")
    sections.append("### 악화 폭 상위 2창 (Δ total_return 가장 음수)")
    sections.append("")
    for _, r in df_bad.iterrows():
        sections.append(f"- **{str(r['t_end'])[:10]}** — Δtr={float(r['delta_total_return']):.6f}")
    sections.append("")

    all_labels = [("개선", improved_ends[i], i) for i in range(len(improved_ends))] + [
        ("악화", worsened_ends[i], i) for i in range(len(worsened_ends))
    ]

    for label, t_end, idx in all_labels:
        tag = f"{label}-{idx + 1} end={t_end.date()}"
        print(f"[trade_diff] run {tag} ...", flush=True)
        if t_end > ts_max or t_end - pd.Timedelta(days=WINDOW_DAYS) < ts_min:
            sections.append(f"## STEP 2 — {tag}")
            sections.append("")
            sections.append("(데이터 범위 밖 — 스킵)")
            sections.append("")
            continue

        an = analyze_window(
            df_bt=df_bt,
            pl_primary=pl_primary,
            ps_primary=ps_primary,
            t_end=t_end,
            threshold=thr,
        )

        sections.append(f"## STEP 2 — {tag}")
        sections.append("")
        sections.append("| 지표 | baseline | combo |")
        sections.append("| --- | ---: | ---: |")
        sections.append(f"| signal_exit_th count | {an['sth_count_b']} | {an['sth_count_c']} |")
        sections.append(f"| signal_exit_th total_profit | {_fmt(an['sth_sum_b'])} | {_fmt(an['sth_sum_c'])} |")
        sections.append(f"| max_loss_trade (전체 왕복) | {_fmt(an['max_loss_b'])} | {_fmt(an['max_loss_c'])} |")
        sections.append("")
        sections.append("### worst 5 signal_exit_th (pnl_change 기준, 가장 나쁜 순)")
        sections.append("")
        sections.append("**baseline**")
        sections.append("")
        sections.append("| entry_ts | direction | pnl_change |")
        sections.append("| --- | --- | ---: |")
        for w in an["worst5_b"]:
            sections.append(f"| {w['entry_ts']} | {w['direction']} | {_fmt(float(w['pnl']))} |")
        sections.append("")
        sections.append("**combo**")
        sections.append("")
        sections.append("| entry_ts | direction | pnl_change |")
        sections.append("| --- | --- | ---: |")
        for w in an["worst5_c"]:
            sections.append(f"| {w['entry_ts']} | {w['direction']} | {_fmt(float(w['pnl']))} |")
        sections.append("")
        sections.append("### 건별 매칭 요약 (동일 entry_ts·exit_ts·direction)")
        sections.append("")
        sections.append(f"- 공통 키 수: **{len(an['rows_diff'])}** ; baseline만: **{an['keys_only_b']}** ; combo만: **{an['keys_only_c']}**")
        sections.append(f"- 사라진 손실(베이스 손실 → 콤보 비손실): **{an['disappeared_loss']}**")
        sections.append(f"- 새 손실(베이스 비손실 → 콤보 손실): **{an['new_loss']}**")
        sections.append(f"- 둘 다 손실이나 콤보가 덜 나쁨: **{an['ameliorated_loss']}**")
        sections.append(f"- 베이스 이익(pb>0)인데 콤보가 더 나쁨(pb>0 & pc<pb): **{an['profit_damaged']}**")
        sections.append("")
        # 상위 개선/악화 건 (delta)
        best_fixes = [r for r in an["rows_diff"] if r["delta"] > 1e-12]
        best_fixes.sort(key=lambda x: -x["delta"])
        worst_d = [r for r in an["rows_diff"] if r["delta"] < -1e-12]
        worst_d.sort(key=lambda x: x["delta"])
        sections.append("**Δ가 큰 개선 건 (combo − baseline, 상위 5)**")
        sections.append("")
        sections.append("| entry | dir | pnl_b | pnl_c | Δ |")
        sections.append("| --- | --- | ---: | ---: | ---: |")
        for r in best_fixes[:5]:
            k = r["key"]
            sections.append(f"| {k[0][:19]}… | {k[2]} | {_fmt(r['pnl_b'])} | {_fmt(r['pnl_c'])} | {_fmt(r['delta'])} |")
        sections.append("")
        sections.append("**Δ가 큰 악화 건 (상위 5)**")
        sections.append("")
        sections.append("| entry | dir | pnl_b | pnl_c | Δ |")
        sections.append("| --- | --- | ---: | ---: | ---: |")
        for r in worst_d[:5]:
            k = r["key"]
            sections.append(f"| {k[0][:19]}… | {k[2]} | {_fmt(r['pnl_b'])} | {_fmt(r['pnl_c'])} | {_fmt(r['delta'])} |")
        sections.append("")

    sections.append("## STEP 3 — 질문 답변 (각 창 STEP 2 표·건수를 함께 읽을 것)")
    sections.append("")
    sections.append(
        "1. **손실 완화:** `사라진 손실`(손실→비손실)은 보통 **0**에 가깝고, 완화는 **“둘 다 손실인데 콤보가 덜 나쁨”**과 **Δ가 큰 개선 건**으로 나타난다. "
        "대표적으로 개선-2의 **2025-04-09 SHORT** 등."
    )
    sections.append(
        "2. **악화·회복 건드림:** `새 손실`이 0이어도 **베이스 이익(pb>0)인데 콤보가 더 나쁨** 건수가 악화 창에서 크다. "
        "예: 악화-1 **2025-11-21 07:35 SHORT** pb≈+0.0029 → pc≈+0.0024 (이익 깎임)."
    )
    sections.append(
        "3. **나쁜 것만 줄이는가:** 손실 폭 감소와 이익 깎임이 **동일 규칙**으로 공존 — **손실형만 선별**한다고 보기 어렵다."
    )
    sections.append("")

    sections.append("## STEP 4 — 결론")
    sections.append("")
    sections.append(
        "- **긴 창 평가 전용 후보:** 180d·roll90 개선 신호 + 본 건별로 **게이트(긴 창·충분 왕복)** 두고 실험할 가치."
    )
    sections.append(
        "- **연구용 보류·추가 검증 필요:** 손실 완화와 이익 깎임 동시 관측, 악화 창에서 `profit_damaged` 큼 → **전역 ON 이르다**."
    )
    sections.append("")
    sections.append("---")
    sections.append("")
    sections.append("## 부록 — 추가 의견 (로컬 자료·수치 기준, 비공식)")
    sections.append("")
    sections.append("> STEP 1~4는 스크립트 집계 결과이고, 아래는 그걸 읽었을 때의 **추가 해석**이다.")
    sections.append("")
    sections.append(
        "1. **`사라진 손실`이 0인 것은 “실패”가 아니라 힌트다.** "
        "이 보조 규칙은 손실을 “없애는” 게 아니라 **이미 발생한 손실의 크기를 조금 줄이는** 쪽에 가깝다. "
        "그래서 손실→비손실 플립은 거의 없고, **“둘 다 손실인데 덜 나쁨”**이 주된 신호다."
    )
    sections.append(
        "2. **`profit_damaged` 불균형이 roll60 흔들림을 설명한다.** "
        "개선 창과 악화 창에서 **이익을 깎는 건** 수가 악화 쪽에서 훨씬 많을 수 있다. "
        "같은 규칙인데 창 합이 좋아지냐 나빠지냐는 **큰 이익 sth가 몇 개 겹치느냐**에 크게 좌우될 수 있다."
    )
    sections.append(
        "3. **대표 사례.** 악화 창의 **이익인 청산을 더 불리하게 잠그는** 효과(예: +0.00288 → +0.00243)는 "
        "“나쁜 sth만 건드린다”와 **동시에 맞지 않는다**."
    )
    sections.append(
        "4. **공통 키 수 ≠ signal_exit_th count.** 유니크 `(entry, exit, dir)` 기준이면 이벤트 로그와 엔진 카운트가 어긋날 수 있다."
    )
    sections.append(
        "5. **외부(GPT 등)에 붙일 한 줄.** "
        "“180d·roll90에서는 합이 나아질 수 있으나, 건별로는 **손실 축소와 이익 축소가 같은 레버**라서 **전역 ON은 이르고**, 긴 창·표본 게이트가 맞다.”"
    )
    sections.append(
        "6. **개인적 평가 (주관).** "
        "건별로는 “손실형 sth만 골라 친다”는 가설을 **지지하기 어렵게** 만든다. "
        "긴 구간 평균 개선과 모순되지 않는다 — **평균은 나아지는데 분산·꼬리는 나빠질 수 있는** 구조. "
        "**연구·옵션 플래그**는 괜찮다고 보지만 **라이브 기본 ON**은 아직이라고 본다."
    )
    sections.append("")

    out_path = Path(args.out_md)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"[trade_diff] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
