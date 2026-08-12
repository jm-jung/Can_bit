#!/usr/bin/env python3
"""
FR2 (Microstructure) 파이프라인 CLI.

사용법:
  # 전체 실행 (마이크로구조 수집 제외: 데이터 없으면 feature 0으로 채움)
  python scripts/run_fr2.py --build-dataset --train --diagnostics

  # 마이크로구조 데이터까지 수집 후 전체 실행
  python scripts/run_fr2.py --fetch-microstructure --build-dataset --train --diagnostics

  # 한 단계만
  python scripts/run_fr2.py --build-dataset
  python scripts/run_fr2.py --train --epochs 30
  python scripts/run_fr2.py --diagnostics
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "data" / "diagnostics" / "models"
FR2_MODEL_PT = MODELS_DIR / "tcn_h15_micro_v1.pt"


def run(cmd: list[str], step_name: str, timeout: int | None = None) -> bool:
    print(f"[FR2] {step_name}: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), timeout=timeout)
    if r.returncode != 0:
        print(f"[FR2] {step_name} failed (exit {r.returncode})", flush=True)
        return False
    return True


def do_fetch_microstructure(args) -> bool:
    days = getattr(args, "microstructure_days", 720)
    since = getattr(args, "microstructure_since", "2023-01-01")
    cmd = [
        sys.executable, "-m", "src.data.fetch_futures_microstructure",
        "--since", since,
        "--days", str(days),
    ]
    return run(cmd, "fetch_microstructure")


def do_build_dataset(_args) -> bool:
    cmd = [sys.executable, "scripts/run_fr2_dataset_build.py"]
    return run(cmd, "build_dataset")


def do_train(args) -> bool:
    epochs = getattr(args, "epochs", 50)
    cmd = [
        sys.executable, "-m", "src.dl.train.train_tcn",
        "--symbol", "BTCUSDT",
        "--timeframe", "5m",
        "--feature-preset", "microstructure_v1",
        "--seq-len", "60",
        "--horizon-bars", "15",
        "--pos-threshold", "0.004",
        "--neg-threshold", "0.004",
        "--out-model", str(FR2_MODEL_PT),
        "--epochs", str(epochs),
    ]
    return run(cmd, "train")


def do_diagnostics(_args) -> bool:
    cmd = [sys.executable, "scripts/run_fr2_diagnostics.py"]
    return run(cmd, "diagnostics")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FR2: Microstructure feature pipeline. Run steps in order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_fr2.py --build-dataset --train --diagnostics
  python scripts/run_fr2.py --fetch-microstructure --build-dataset --train --diagnostics
  python scripts/run_fr2.py --diagnostics   # only run diagnostics (need existing model)
        """,
    )
    parser.add_argument("--fetch-microstructure", action="store_true", help="Fetch Binance futures taker/funding data → data/ohlcv/BTCUSDT_5m_microstructure.parquet")
    parser.add_argument("--microstructure-since", default="2023-01-01", help="Start date for fetch (default: 2023-01-01)")
    parser.add_argument("--microstructure-days", type=int, default=720, help="Days to fetch (default: 720)")
    parser.add_argument("--build-dataset", action="store_true", help="Build tcn_microstructure_v1.parquet + feature_summary.csv")
    parser.add_argument("--train", action="store_true", help="Train TCN h15_micro_v1 (microstructure_v1 preset)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--diagnostics", action="store_true", help="Run alignment/backtest/rolling/walk-forward → fr2_*.csv, fr2_summary.md")
    args = parser.parse_args()

    if not any([args.fetch_microstructure, args.build_dataset, args.train, args.diagnostics]):
        parser.print_help()
        print("\n[FR2] No step selected. Use e.g. --build-dataset --train --diagnostics", flush=True)
        return 0

    if args.fetch_microstructure and not do_fetch_microstructure(args):
        return 1
    if args.build_dataset and not do_build_dataset(args):
        return 1
    if args.train and not do_train(args):
        return 1
    if args.diagnostics:
        if not FR2_MODEL_PT.exists() and not (MODELS_DIR / "tcn_h15_micro_v1.pt").exists():
            print(f"[FR2] Diagnostics need FR2 model at {FR2_MODEL_PT}. Run --train first.", flush=True)
            return 1
        if not do_diagnostics(args):
            return 1

    print("[FR2] Done.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
