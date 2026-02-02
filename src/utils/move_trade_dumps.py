#!/usr/bin/env python3
"""
Trade dump CSV 파일을 /tmp 또는 /private/tmp에서 data/backtest_dumps/로 이동하는 스크립트.

대상 파일:
- /tmp/*.csv 또는 /private/tmp/*.csv
- 파일명에 다음 키워드 중 하나 포함:
  - trades_baseline
  - trades_stage2
  - trades_guard
  - trades_stage2_guard

규칙:
- 목적지: data/backtest_dumps/
- 동일 파일명 존재 시 suffix 추가 (덮어쓰기 방지)
- 이동 후 원본 삭제
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

# 대상 키워드
TARGET_KEYWORDS = [
    "trades_baseline",
    "trades_stage2",
    "trades_guard",
    "trades_stage2_guard",
]

# 검색 경로
SEARCH_PATHS = ["/tmp", "/private/tmp"]

# 목적지 디렉토리
DEST_DIR = "data/backtest_dumps"


def is_target_file(filename: str) -> bool:
    """파일명이 이동 대상인지 확인"""
    return any(keyword in filename for keyword in TARGET_KEYWORDS)


def get_unique_dest_path(dest_dir: Path, filename: str) -> Path:
    """덮어쓰기 방지를 위한 고유한 목적지 경로 생성"""
    dest_path = dest_dir / filename
    
    if not dest_path.exists():
        return dest_path
    
    # 파일명에서 확장자 분리
    name, ext = os.path.splitext(filename)
    
    # suffix 추가
    counter = 2
    while True:
        new_filename = f"{name}_v{counter}{ext}"
        new_dest_path = dest_dir / new_filename
        if not new_dest_path.exists():
            return new_dest_path
        counter += 1


def move_trade_dumps():
    """Trade dump CSV 파일들을 이동"""
    # 목적지 디렉토리 생성
    dest_dir = Path(DEST_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    moved_files = []
    skipped_files = []
    errors = []
    
    # 각 검색 경로에서 파일 찾기
    for search_path in SEARCH_PATHS:
        search_dir = Path(search_path)
        
        if not search_dir.exists():
            continue
        
        # CSV 파일 검색
        for csv_file in search_dir.glob("*.csv"):
            if not is_target_file(csv_file.name):
                continue
            
            try:
                # 파일 크기 확인
                if csv_file.stat().st_size == 0:
                    skipped_files.append((str(csv_file), "empty file"))
                    continue
                
                # 고유한 목적지 경로 생성
                dest_path = get_unique_dest_path(dest_dir, csv_file.name)
                
                # 파일 이동
                shutil.move(str(csv_file), str(dest_path))
                
                moved_files.append((str(csv_file), str(dest_path)))
                print(f"✓ Moved: {csv_file} → {dest_path}")
                
            except Exception as e:
                errors.append((str(csv_file), str(e)))
                print(f"✗ Error moving {csv_file}: {e}")
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("Move Summary")
    print("=" * 60)
    print(f"Total moved: {len(moved_files)}")
    print(f"Skipped: {len(skipped_files)}")
    print(f"Errors: {len(errors)}")
    
    if moved_files:
        print("\nMoved files:")
        for src, dst in moved_files:
            print(f"  {src} → {dst}")
    
    if skipped_files:
        print("\nSkipped files:")
        for path, reason in skipped_files:
            print(f"  {path} ({reason})")
    
    if errors:
        print("\nErrors:")
        for path, error in errors:
            print(f"  {path}: {error}")
    
    return len(moved_files), len(skipped_files), len(errors)


if __name__ == "__main__":
    print("=" * 60)
    print("Trade Dump CSV File Mover")
    print("=" * 60)
    print(f"Search paths: {', '.join(SEARCH_PATHS)}")
    print(f"Destination: {DEST_DIR}")
    print(f"Target keywords: {', '.join(TARGET_KEYWORDS)}")
    print("=" * 60)
    print()
    
    moved, skipped, errors = move_trade_dumps()
    
    if errors > 0:
        exit(1)
    else:
        exit(0)

