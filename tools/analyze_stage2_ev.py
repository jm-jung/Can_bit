"""
Stage-2 통과 구간 EV(Expected Value) 분석 도구

이 스크립트는 Stage-2에서 Trade=True로 통과한 시점들만 모아서
그 시점 이후 N-bar forward return의 기대값(EV)을 계산합니다.
수수료/슬리피지까지 포함한 "실제 거래 가능성"을 판단하는 데 사용됩니다.

사용 예시:
    python tools/analyze_stage2_ev.py \
        --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \
        --pred-parquet data/cache/ml_predictions/ml_lstm_attn_BTCUSDT_5m_proba.parquet \
        --horizon-bars 6 \
        --commission-rate 0.0004 \
        --slippage-rate 0.0005 \
        --use-stage2 \
        --stage2-trade-th 0.58 \
        --stage2-min-edge 0.05
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    """
    OHLCV CSV 파일을 로드합니다.
    
    Args:
        csv_path: OHLCV CSV 파일 경로
        
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    logger.info(f"Loading OHLCV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # timestamp 컬럼을 datetime으로 변환
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    elif "datetime" in df.columns:
        df["timestamp"] = pd.to_datetime(df["datetime"])
        df = df.rename(columns={"datetime": "timestamp"})
    else:
        raise ValueError("OHLCV CSV must have 'timestamp' or 'datetime' column")
    
    # 필수 컬럼 확인
    required_cols = ["timestamp", "close"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"OHLCV CSV missing required columns: {missing_cols}")
    
    # 정렬 (timestamp 기준)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(f"Loaded OHLCV: {len(df)} rows, timestamp range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    return df


def load_predictions(parquet_path: Path) -> pd.DataFrame:
    """
    Prediction cache parquet 파일을 로드합니다.
    
    Args:
        parquet_path: Prediction cache parquet 파일 경로
        
    Returns:
        DataFrame with columns: timestamp, proba_long, proba_short, (optional) raw_signal, stage2_trade, stage2_reason
    """
    logger.info(f"Loading predictions from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # 필수 컬럼 확인
    required_cols = ["proba_long", "proba_short"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Prediction parquet missing required columns: {missing_cols}")
    
    # timestamp 컬럼 확인 및 변환
    if "timestamp" not in df.columns:
        raise ValueError("Prediction parquet must have 'timestamp' column")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # 정렬 (timestamp 기준)
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    logger.info(
        f"Loaded predictions: {len(df)} rows, "
        f"timestamp range: {df['timestamp'].min()} ~ {df['timestamp'].max()}, "
        f"columns: {list(df.columns)}"
    )
    
    return df


def normalize_price_columns(df: pd.DataFrame, ohlcv_suffix: str = "_ohlcv", pred_suffix: str = "_pred") -> pd.DataFrame:
    """
    Join 후 중복된 price 컬럼(close 등)을 단일 SSOT로 정규화합니다.
    
    Args:
        df: Join된 DataFrame (close_x, close_y 같은 중복 컬럼 가능)
        ohlcv_suffix: OHLCV 쪽 suffix (기본: "_ohlcv" 또는 merge에서 "_x")
        pred_suffix: Prediction 쪽 suffix (기본: "_pred" 또는 merge에서 "_y")
        
    Returns:
        정규화된 DataFrame (df["close"]가 보장됨)
    """
    df = df.copy()
    
    # close 후보 컬럼 수집
    close_candidates = []
    for col in df.columns:
        if col == "close" or col.startswith("close_"):
            close_candidates.append(col)
    
    logger.info(f"Close column candidates: {close_candidates}")
    
    if len(close_candidates) == 0:
        raise ValueError("No 'close' column found in DataFrame")
    
    if len(close_candidates) == 1:
        # 후보가 1개면 그걸 close로 rename (이미 close면 pass)
        if close_candidates[0] != "close":
            df = df.rename(columns={close_candidates[0]: "close"})
            logger.info(f"Renamed '{close_candidates[0]}' to 'close'")
    else:
        # 후보가 2개 이상: OHLCV 쪽 우선 선택
        # 우선순위: close_ohlcv > close_x > close > close_pred > close_y
        priority_order = [
            "close_ohlcv",
            "close_x",  # merge 기본 suffix (왼쪽=OHLCV)
            "close",
            "close_pred",
            "close_y",  # merge 기본 suffix (오른쪽=pred)
        ]
        
        chosen_close = None
        for candidate in priority_order:
            if candidate in close_candidates:
                chosen_close = candidate
                break
        
        # priority_order에 없으면 첫 번째 후보 선택
        if chosen_close is None:
            chosen_close = close_candidates[0]
            logger.warning(f"No priority match found, using first candidate: {chosen_close}")
        
        logger.info(f"Selected close column: '{chosen_close}' (from {len(close_candidates)} candidates)")
        
        # 선택한 close를 df["close"]로 통일
        if chosen_close != "close":
            # 다른 close 후보들과 값 비교
            other_candidates = [c for c in close_candidates if c != chosen_close]
            
            if other_candidates:
                # 값 불일치 체크
                max_diff = 0.0
                mean_diff = 0.0
                diff_samples = []
                
                for other_col in other_candidates:
                    # NaN 제외하고 비교
                    valid_mask = ~(df[chosen_close].isna() | df[other_col].isna())
                    if valid_mask.sum() > 0:
                        diff = np.abs(df.loc[valid_mask, chosen_close] - df.loc[valid_mask, other_col])
                        rel_diff = diff / (df.loc[valid_mask, chosen_close].abs() + 1e-10)
                        
                        max_diff = max(max_diff, float(diff.max()))
                        mean_diff = float(diff.mean())
                        max_rel_diff = float(rel_diff.max())
                        
                        # 상대 오차 0.1% 이상이면 샘플 수집
                        if max_rel_diff > 0.001:
                            large_diff_mask = rel_diff > 0.001
                            if large_diff_mask.sum() > 0:
                                # Index 객체는 .head()가 없으므로 슬라이싱 사용
                                # valid_mask가 True인 행의 원본 인덱스
                                valid_indices = df.loc[valid_mask].index
                                # large_diff_mask는 rel_diff(Series)에 대한 boolean mask
                                # rel_diff의 인덱스는 valid_indices와 동일
                                large_diff_indices = valid_indices[large_diff_mask]
                                sample_indices = large_diff_indices[:5].tolist()
                                
                                # 샘플 DataFrame 생성 (사용자 요구사항에 맞게)
                                if len(sample_indices) > 0:
                                    sample_df = df.loc[sample_indices, [chosen_close, other_col, "timestamp"]].copy()
                                    sample_df["abs_diff"] = (sample_df[chosen_close] - sample_df[other_col]).abs()
                                    sample_df["rel_diff"] = sample_df["abs_diff"] / (sample_df[chosen_close].abs() + 1e-12)
                                    
                                    # diff_samples에 추가
                                    for idx in sample_indices:
                                        diff_samples.append({
                                            "idx": int(idx),
                                            "timestamp": str(sample_df.loc[idx, "timestamp"]),
                                            f"{chosen_close}": float(sample_df.loc[idx, chosen_close]),
                                            f"{other_col}": float(sample_df.loc[idx, other_col]),
                                            "abs_diff": float(sample_df.loc[idx, "abs_diff"]),
                                            "rel_diff": float(sample_df.loc[idx, "rel_diff"]),
                                        })
                
                # 경고 로그
                if max_diff > 1e-6 or (diff_samples and len(diff_samples) > 0):
                    logger.warning(
                        f"Close column mismatch detected: "
                        f"max_abs_diff={max_diff:.6f}, mean_abs_diff={mean_diff:.6f}"
                    )
                    if diff_samples:
                        logger.warning(f"Sample mismatches (first {min(5, len(diff_samples))}):")
                        for sample in diff_samples[:5]:
                            # 샘플 정보를 명확하게 출력
                            chosen_val = sample.get(chosen_close, "N/A")
                            other_val = sample.get(other_col, "N/A")
                            if isinstance(chosen_val, (int, float)) and isinstance(other_val, (int, float)):
                                logger.warning(
                                    f"  idx={sample['idx']}, ts={sample['timestamp']}, "
                                    f"{chosen_close}={chosen_val:.2f}, "
                                    f"{other_col}={other_val:.2f}, "
                                    f"abs_diff={sample['abs_diff']:.6f}, rel_diff={sample['rel_diff']:.4%}"
                                )
                            else:
                                logger.warning(
                                    f"  idx={sample['idx']}, ts={sample['timestamp']}, "
                                    f"{chosen_close}={chosen_val}, "
                                    f"{other_col}={other_val}, "
                                    f"abs_diff={sample['abs_diff']:.6f}, rel_diff={sample['rel_diff']:.4%}"
                                )
            
            # rename
            df = df.rename(columns={chosen_close: "close"})
            logger.info(f"Renamed '{chosen_close}' to 'close'")
        
        # 나머지 close 후보는 drop (rename 후 재계산하여 stale 방지)
        # rename 후 실제 df.columns에 존재하는 close 후보만 필터링
        remaining_close_candidates = [c for c in df.columns if (c == "close" or c.startswith("close_")) and c != "close"]
        if remaining_close_candidates:
            # 실제 존재하는 컬럼만 drop (안전장치)
            existing_close_cols = [c for c in remaining_close_candidates if c in df.columns]
            if existing_close_cols:
                df = df.drop(columns=existing_close_cols, errors="ignore")
                logger.info(f"Dropped other close columns: {existing_close_cols}")
            else:
                logger.debug(f"No close columns to drop (already removed or renamed)")
    
    # 최종 확인 및 로깅
    if "close" not in df.columns:
        available_cols = list(df.columns)
        close_related = [c for c in available_cols if "close" in c.lower()]
        raise ValueError(
            f"Failed to normalize 'close' column. "
            f"Available columns: {available_cols[:20]}{'...' if len(available_cols) > 20 else ''}, "
            f"Close-related: {close_related}"
        )
    
    # 최종 컬럼 목록 로깅
    final_close_related = [c for c in df.columns if "close" in c.lower()]
    logger.info(f"Final DataFrame columns (close-related): {final_close_related}")
    logger.info(f"Final DataFrame total columns: {len(df.columns)} (showing first 10: {list(df.columns[:10])})")
    
    return df


def compute_forward_returns(
    df: pd.DataFrame,
    horizon_bars: int,
    commission_rate: float,
    slippage_rate: float,
    direction: str = "both",
) -> pd.DataFrame:
    """
    Forward return을 계산합니다.
    
    Args:
        df: OHLCV와 prediction이 join된 DataFrame (close 컬럼 보장)
        horizon_bars: Forward return 계산용 horizon (bars)
        commission_rate: 수수료율
        slippage_rate: 슬리피지율
        direction: 분석 방향 ("long", "short", "both"). 기본값: "both"
        
    Returns:
        DataFrame with additional columns:
        - forward_return: close[t+horizon] / close[t] - 1
        - directional_return: LONG이면 +forward_return, SHORT이면 -forward_return
        - net_return: directional_return - 2 * (commission_rate + slippage_rate)
    """
    df = df.copy()
    
    # 방어 코드: close 컬럼 확인 및 fallback
    if "close" not in df.columns:
        # Fallback: close 후보 찾기
        close_candidates = [c for c in df.columns if "close" in c.lower()]
        if close_candidates:
            logger.warning(f"'close' column not found, using fallback: {close_candidates[0]}")
            df["close"] = df[close_candidates[0]]
        else:
            raise ValueError("No 'close' column found in DataFrame for forward return calculation")
    
    # Forward return 계산
    # close[t+horizon] / close[t] - 1
    # NaN 값 방어: close가 0이거나 NaN인 경우 처리
    close_valid = df["close"].notna() & (df["close"] > 0)
    df["forward_return"] = np.nan
    
    # shift는 전체 DataFrame에 대해 수행한 후 유효한 행만 선택
    close_shifted = df["close"].shift(-horizon_bars)
    forward_return = close_shifted / df["close"] - 1.0
    df.loc[close_valid, "forward_return"] = forward_return.loc[close_valid]
    
    # direction에 따라 raw_signal 조정 (long-only면 SHORT를 HOLD로, short-only면 LONG을 HOLD로)
    if direction == "long":
        # SHORT 신호를 HOLD로 처리
        df["raw_signal"] = np.where(df["raw_signal"] == "SHORT", "HOLD", df["raw_signal"])
    elif direction == "short":
        # LONG 신호를 HOLD로 처리
        df["raw_signal"] = np.where(df["raw_signal"] == "LONG", "HOLD", df["raw_signal"])
    
    # 방향성 수익 계산
    # LONG: +forward_return
    # SHORT: -forward_return
    df["directional_return"] = np.where(
        df["raw_signal"] == "LONG",
        df["forward_return"],
        np.where(
            df["raw_signal"] == "SHORT",
            -df["forward_return"],
            np.nan  # HOLD는 NaN
        )
    )
    
    # 왕복 비용 차감
    roundtrip_cost = 2 * (commission_rate + slippage_rate)
    df["net_return"] = df["directional_return"] - roundtrip_cost
    
    # Forward return이 계산 가능한 행만 필터링 (horizon_bars 이후 데이터 필요)
    valid_mask = ~df["forward_return"].isna()
    valid_count = valid_mask.sum()
    logger.info(
        f"Forward return calculation: {valid_count} valid samples out of {len(df)} "
        f"(horizon={horizon_bars} bars, {len(df) - valid_count} samples dropped at end)"
    )
    
    return df


def filter_stage2_trade(
    df: pd.DataFrame,
    use_stage2: bool,
    stage2_trade_th: Optional[float],
    stage2_min_edge: float,
    direction: str = "both",
) -> pd.DataFrame:
    """
    Stage-2 Trade=True인 시점만 필터링합니다.
    
    Args:
        df: DataFrame with proba_long, proba_short, (optional) stage2_trade, raw_signal
        use_stage2: Stage-2 필터링 사용 여부
        stage2_trade_th: Stage-2 절대 임계값
        stage2_min_edge: Stage-2 마진 임계값
        direction: 분석 방향 ("long", "short", "both"). 기본값: "both"
        
    Returns:
        필터링된 DataFrame
    """
    if not use_stage2:
        # Stage-2 비활성: 모든 샘플 사용 (direction 필터만 적용)
        if direction == "long":
            mask = df["raw_signal"] == "LONG"
            logger.info(f"Stage-2 disabled, direction=long: {mask.sum()} LONG samples out of {len(df)}")
            return df[mask].reset_index(drop=True)
        elif direction == "short":
            mask = df["raw_signal"] == "SHORT"
            logger.info(f"Stage-2 disabled, direction=short: {mask.sum()} SHORT samples out of {len(df)}")
            return df[mask].reset_index(drop=True)
        else:
            logger.info("Stage-2 filtering disabled: using all samples")
            return df
    
    # Stage-2 Trade=True인 경우만 필터링
    if "stage2_trade" in df.columns:
        stage2_mask = df["stage2_trade"] == True
        
        # direction 필터 적용
        if direction == "long":
            stage2_mask = stage2_mask & (df["raw_signal"] == "LONG")
        elif direction == "short":
            stage2_mask = stage2_mask & (df["raw_signal"] == "SHORT")
        # both는 stage2_mask 그대로 사용
        
        stage2_count = stage2_mask.sum()
        logger.info(
            f"Stage-2 filtering (direction={direction}): {stage2_count} samples with stage2_trade=True "
            f"out of {len(df)} total"
        )
        return df[stage2_mask].reset_index(drop=True)
    
    # stage2_trade 컬럼이 없으면 Stage-2 로직을 직접 계산
    logger.info(f"stage2_trade column not found, computing Stage-2 logic on-the-fly (direction={direction})")
    
    stage2_trade = np.full(len(df), False, dtype=bool)
    
    # iterrows()는 원본 인덱스를 반환하므로 위치 기반 인덱싱을 위해 enumerate 사용
    for pos_idx, (orig_idx, row) in enumerate(df.iterrows()):
        raw_sig = row.get("raw_signal", "HOLD")
        p_long = row["proba_long"]
        p_short = row["proba_short"]
        
        if raw_sig == "HOLD" or raw_sig == "FLAT":
            continue  # HOLD는 Trade=False
        
        # direction 필터: long이면 LONG만, short이면 SHORT만 처리
        if direction == "long" and raw_sig != "LONG":
            continue
        if direction == "short" and raw_sig != "SHORT":
            continue
        
        trade_ok_abs = False
        trade_ok_edge = False
        
        if raw_sig == "LONG":
            if stage2_trade_th is not None:
                trade_ok_abs = p_long >= stage2_trade_th
            if stage2_min_edge > 0.0:
                trade_ok_edge = (p_long - p_short) >= stage2_min_edge
        elif raw_sig == "SHORT":
            if stage2_trade_th is not None:
                trade_ok_abs = p_short >= stage2_trade_th
            if stage2_min_edge > 0.0:
                trade_ok_edge = (p_short - p_long) >= stage2_min_edge
        
        # 둘 중 하나라도 만족하면 Trade=True
        stage2_trade[pos_idx] = trade_ok_abs or trade_ok_edge
    
    stage2_count = stage2_trade.sum()
    logger.info(
        f"Stage-2 filtering (computed, direction={direction}): {stage2_count} samples with Trade=True "
        f"out of {len(df)} total"
    )
    
    return df[stage2_trade].reset_index(drop=True)


def compute_statistics(df: pd.DataFrame, direction: str = "both") -> dict:
    """
    LONG/SHORT 각각 및 mixed 통계를 계산합니다.
    
    Args:
        df: Forward return이 계산된 DataFrame (net_return 컬럼 포함)
        direction: 분석 방향 ("long", "short", "both"). 기본값: "both"
        
    Returns:
        통계 딕셔너리
    """
    stats = {}
    
    # 필수 컬럼 확인
    required_cols = ["net_return", "directional_return", "raw_signal"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame missing required columns for statistics: {missing_cols}")
    
    # 전체 (LONG + SHORT)
    valid_mask = ~df["net_return"].isna()
    if valid_mask.sum() == 0:
        logger.warning("No valid net_return values found for statistics calculation")
        return stats
    
    # direction에 따라 계산할 통계 결정
    compute_long = (direction == "long" or direction == "both")
    compute_short = (direction == "short" or direction == "both")
    compute_mixed = (direction == "both")
    
    if compute_mixed and valid_mask.sum() > 0:
        net_returns = df.loc[valid_mask, "net_return"]
        stats["mixed"] = {
            "sample_count": len(net_returns),
            "mean_gross_return": float(df.loc[valid_mask, "directional_return"].mean()),
            "mean_net_return": float(net_returns.mean()),
            "median_net_return": float(net_returns.median()),
            "std_net_return": float(net_returns.std()),
            "winrate_net": float((net_returns > 0).sum() / len(net_returns)),
            "p10_net_return": float(net_returns.quantile(0.10)),
            "p90_net_return": float(net_returns.quantile(0.90)),
            "min_net_return": float(net_returns.min()),
            "max_net_return": float(net_returns.max()),
        }
    
    # LONG만
    if compute_long:
        long_mask = (df["raw_signal"] == "LONG") & valid_mask
        if long_mask.sum() > 0:
            long_net_returns = df.loc[long_mask, "net_return"]
            stats["LONG"] = {
                "sample_count": len(long_net_returns),
                "mean_gross_return": float(df.loc[long_mask, "directional_return"].mean()),
                "mean_net_return": float(long_net_returns.mean()),
                "median_net_return": float(long_net_returns.median()),
                "std_net_return": float(long_net_returns.std()),
                "winrate_net": float((long_net_returns > 0).sum() / len(long_net_returns)),
                "p10_net_return": float(long_net_returns.quantile(0.10)),
                "p90_net_return": float(long_net_returns.quantile(0.90)),
                "min_net_return": float(long_net_returns.min()),
                "max_net_return": float(long_net_returns.max()),
            }
        elif direction == "long":
            # long-only인데 LONG 샘플이 없으면 빈 통계
            stats["LONG"] = {
                "sample_count": 0,
                "mean_gross_return": 0.0,
                "mean_net_return": 0.0,
                "median_net_return": 0.0,
                "std_net_return": 0.0,
                "winrate_net": 0.0,
                "p10_net_return": 0.0,
                "p90_net_return": 0.0,
                "min_net_return": 0.0,
                "max_net_return": 0.0,
            }
    
    # SHORT만
    if compute_short:
        short_mask = (df["raw_signal"] == "SHORT") & valid_mask
        if short_mask.sum() > 0:
            short_net_returns = df.loc[short_mask, "net_return"]
            stats["SHORT"] = {
                "sample_count": len(short_net_returns),
                "mean_gross_return": float(df.loc[short_mask, "directional_return"].mean()),
                "mean_net_return": float(short_net_returns.mean()),
                "median_net_return": float(short_net_returns.median()),
                "std_net_return": float(short_net_returns.std()),
                "winrate_net": float((short_net_returns > 0).sum() / len(short_net_returns)),
                "p10_net_return": float(short_net_returns.quantile(0.10)),
                "p90_net_return": float(short_net_returns.quantile(0.90)),
                "min_net_return": float(short_net_returns.min()),
                "max_net_return": float(short_net_returns.max()),
            }
        elif direction == "short":
            # short-only인데 SHORT 샘플이 없으면 빈 통계
            stats["SHORT"] = {
                "sample_count": 0,
                "mean_gross_return": 0.0,
                "mean_net_return": 0.0,
                "median_net_return": 0.0,
                "std_net_return": 0.0,
                "winrate_net": 0.0,
                "p10_net_return": 0.0,
                "p90_net_return": 0.0,
                "min_net_return": 0.0,
                "max_net_return": 0.0,
            }
    
    return stats


def print_statistics(stats: dict, commission_rate: float, slippage_rate: float, direction: str = "both"):
    """
    통계를 포맷팅하여 출력합니다.
    
    Args:
        stats: compute_statistics()의 결과
        commission_rate: 수수료율
        slippage_rate: 슬리피지율
        direction: 분석 방향 ("long", "short", "both"). 기본값: "both"
    """
    roundtrip_cost = 2 * (commission_rate + slippage_rate)
    
    print("=" * 80)
    print(f"Stage-2 EV Analysis Results (direction={direction})")
    print("=" * 80)
    print(f"Roundtrip cost (2 * (commission + slippage)): {roundtrip_cost:.6f} ({roundtrip_cost*100:.4f}%)")
    print()
    
    # direction에 따라 출력할 섹션 결정
    if direction == "long":
        sections = ["LONG"]
    elif direction == "short":
        sections = ["SHORT"]
    else:
        sections = ["LONG", "SHORT", "mixed"]
    
    for section in sections:
        if section not in stats:
            continue
        
        s = stats[section]
        print(f"--- {section} ---")
        print(f"  Sample count: {s['sample_count']:,}")
        print(f"  Mean gross return: {s['mean_gross_return']:.6f} ({s['mean_gross_return']*100:.4f}%)")
        print(f"  Mean net return: {s['mean_net_return']:.6f} ({s['mean_net_return']*100:.4f}%)")
        print(f"  Median net return: {s['median_net_return']:.6f} ({s['median_net_return']*100:.4f}%)")
        print(f"  Std net return: {s['std_net_return']:.6f}")
        print(f"  Win rate (net > 0): {s['winrate_net']:.4f} ({s['winrate_net']*100:.2f}%)")
        print(f"  P10 net return: {s['p10_net_return']:.6f} ({s['p10_net_return']*100:.4f}%)")
        print(f"  P90 net return: {s['p90_net_return']:.6f} ({s['p90_net_return']*100:.4f}%)")
        print(f"  Min net return: {s['min_net_return']:.6f} ({s['min_net_return']*100:.4f}%)")
        print(f"  Max net return: {s['max_net_return']:.6f} ({s['max_net_return']*100:.4f}%)")
        
        # EV 판단
        if s['mean_net_return'] > 0:
            print(f"  ✓ EV > 0: Positive expected value")
        else:
            print(f"  ✗ EV <= 0: Negative expected value")
        print()
    
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Stage-2 통과 구간 EV 분석 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python tools/analyze_stage2_ev.py \\
      --ohlcv-csv data/ohlcv/BTCUSDT_5m_full.csv \\
      --pred-parquet data/cache/ml_predictions/ml_lstm_attn_BTCUSDT_5m_proba.parquet \\
      --horizon-bars 6 \\
      --commission-rate 0.0004 \\
      --slippage-rate 0.0005 \\
      --use-stage2 \\
      --stage2-trade-th 0.58 \\
      --stage2-min-edge 0.05
        """
    )
    
    parser.add_argument(
        "--ohlcv-csv",
        type=Path,
        required=True,
        help="OHLCV CSV 파일 경로 (timestamp, open, high, low, close, volume 컬럼 필요)",
    )
    parser.add_argument(
        "--pred-parquet",
        type=Path,
        required=True,
        help="Prediction cache parquet 파일 경로 (timestamp, proba_long, proba_short 컬럼 필요)",
    )
    parser.add_argument(
        "--horizon-bars",
        type=int,
        default=6,
        help="Forward return 계산용 horizon (bars). 기본값: 6",
    )
    parser.add_argument(
        "--commission-rate",
        type=float,
        default=0.0004,
        help="수수료율. 기본값: 0.0004 (0.04%%)",
    )
    parser.add_argument(
        "--slippage-rate",
        type=float,
        default=0.0005,
        help="슬리피지율. 기본값: 0.0005 (0.05%%)",
    )
    parser.add_argument(
        "--use-stage2",
        action="store_true",
        default=False,
        help="Stage-2 Trade=True인 시점만 분석. 기본값: False (모든 시점 분석)",
    )
    parser.add_argument(
        "--stage2-trade-th",
        type=float,
        default=None,
        help="Stage-2 절대 임계값 (use-stage2가 True일 때 사용). 기본값: None",
    )
    parser.add_argument(
        "--stage2-min-edge",
        type=float,
        default=0.0,
        help="Stage-2 마진 임계값 (use-stage2가 True일 때 사용). 기본값: 0.0",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=None,
        help="추가 필터: |p_long - p_short| >= min_edge인 시점만 분석. 기본값: None (비활성)",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["long", "short", "both"],
        default="both",
        help="분석 방향: 'long' (LONG-only), 'short' (SHORT-only), 'both' (LONG+SHORT). 기본값: both",
    )
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not args.ohlcv_csv.exists():
        raise FileNotFoundError(f"OHLCV CSV not found: {args.ohlcv_csv}")
    if not args.pred_parquet.exists():
        raise FileNotFoundError(f"Prediction parquet not found: {args.pred_parquet}")
    
    # 데이터 로드
    ohlcv_df = load_ohlcv(args.ohlcv_csv)
    pred_df = load_predictions(args.pred_parquet)
    
    # Timestamp 기준 join
    logger.info("Joining OHLCV and predictions on timestamp...")
    df = pd.merge(
        ohlcv_df[["timestamp", "close"]],
        pred_df,
        on="timestamp",
        how="inner",
        suffixes=("_ohlcv", "_pred"),
    )
    
    if len(df) == 0:
        raise ValueError("No matching timestamps between OHLCV and predictions")
    
    logger.info(f"Joined DataFrame: {len(df)} rows after inner join")
    logger.info(f"Joined DataFrame columns: {list(df.columns)}")
    
    # Price 컬럼 정규화 (close 중복 해결)
    df = normalize_price_columns(df, ohlcv_suffix="_ohlcv", pred_suffix="_pred")
    
    # Timestamp alignment 경고
    ohlcv_ts_set = set(ohlcv_df["timestamp"])
    pred_ts_set = set(pred_df["timestamp"])
    overlap = len(ohlcv_ts_set & pred_ts_set)
    ohlcv_only = len(ohlcv_ts_set - pred_ts_set)
    pred_only = len(pred_ts_set - ohlcv_ts_set)
    
    if ohlcv_only > 0 or pred_only > 0:
        logger.warning(
            f"Timestamp alignment: {overlap} overlapping, "
            f"{ohlcv_only} OHLCV-only, {pred_only} prediction-only timestamps"
        )
    
    # raw_signal이 없으면 생성 (argmax 기반)
    if "raw_signal" not in df.columns:
        logger.info("raw_signal column not found, computing from proba_long/proba_short")
        # p_flat 계산
        df["proba_flat"] = 1.0 - df["proba_long"] - df["proba_short"]
        df["proba_flat"] = df["proba_flat"].clip(0.0, 1.0)
        
        # argmax로 방향 결정
        proba_matrix = df[["proba_flat", "proba_long", "proba_short"]].values
        argmax_indices = np.argmax(proba_matrix, axis=1)
        df["raw_signal"] = np.where(
            argmax_indices == 0, "HOLD",
            np.where(argmax_indices == 1, "LONG", "SHORT")
        )
    
    # Stage-2 필터링
    df_filtered = filter_stage2_trade(
        df,
        use_stage2=args.use_stage2,
        stage2_trade_th=args.stage2_trade_th,
        stage2_min_edge=args.stage2_min_edge,
        direction=args.direction,
    )
    
    # min_edge 필터 (추가) - direction에 따라 다르게 적용
    if args.min_edge is not None and args.min_edge > 0.0:
        if args.direction == "long":
            edge_mask = (df_filtered["proba_long"] - df_filtered["proba_short"]) >= args.min_edge
        elif args.direction == "short":
            edge_mask = (df_filtered["proba_short"] - df_filtered["proba_long"]) >= args.min_edge
        else:
            edge_mask = np.abs(df_filtered["proba_long"] - df_filtered["proba_short"]) >= args.min_edge
        
        edge_count = edge_mask.sum()
        logger.info(
            f"Min-edge filter (direction={args.direction}): {edge_count} samples "
            f"out of {len(df_filtered)}"
        )
        df_filtered = df_filtered[edge_mask].reset_index(drop=True)
    
    # Forward return 계산
    df_with_returns = compute_forward_returns(
        df_filtered,
        horizon_bars=args.horizon_bars,
        commission_rate=args.commission_rate,
        slippage_rate=args.slippage_rate,
        direction=args.direction,
    )
    
    # 통계 계산
    stats = compute_statistics(df_with_returns, direction=args.direction)
    
    # 결과 출력
    print_statistics(stats, args.commission_rate, args.slippage_rate, direction=args.direction)
    
    # 요약
    if "mixed" in stats:
        mixed_ev = stats["mixed"]["mean_net_return"]
        if mixed_ev > 0:
            logger.info(f"✓ Overall EV is positive: {mixed_ev:.6f} ({mixed_ev*100:.4f}%)")
        else:
            logger.warning(f"✗ Overall EV is negative: {mixed_ev:.6f} ({mixed_ev*100:.4f}%)")
            logger.warning("This may indicate model signal quality issues rather than threshold tuning problems.")


if __name__ == "__main__":
    main()

