"""
Overfitting test utilities for debugging deep learning models.

이 모듈은 작은 데이터셋에 대한 오버핏 테스트를 수행하여
모델/옵티마이저/데이터 플로우에 구조적인 버그가 있는지 검증합니다.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

logger = logging.getLogger(__name__)


def print_label_stats(
    loader_or_dataset: DataLoader | Dataset,
    prefix: str = "[DEBUG][LABEL]",
) -> None:
    """
    데이터셋/로더의 label 분포를 출력합니다.
    
    Args:
        loader_or_dataset: DataLoader 또는 Dataset 인스턴스
        prefix: 로그 출력 접두사
    """
    labels = []
    
    if isinstance(loader_or_dataset, DataLoader):
        for _, y_batch in loader_or_dataset:
            if isinstance(y_batch, torch.Tensor):
                labels.extend(y_batch.cpu().numpy().flatten().tolist())
            else:
                labels.extend(np.array(y_batch).flatten().tolist())
    elif isinstance(loader_or_dataset, Dataset):
        for i in range(len(loader_or_dataset)):
            _, y = loader_or_dataset[i]
            if isinstance(y, torch.Tensor):
                labels.append(y.item() if y.numel() == 1 else y.cpu().numpy().flatten()[0])
            else:
                labels.append(float(y))
    else:
        logger.warning(f"{prefix} Unsupported type: {type(loader_or_dataset)}")
        return
    
    if not labels:
        logger.warning(f"{prefix} No labels found")
        return
    
    labels_array = np.array(labels)
    num_pos = int((labels_array == 1).sum())
    num_total = len(labels_array)
    y_mean = float(labels_array.mean())
    
    logger.info(f"{prefix} y.mean={y_mean:.4f}, #pos={num_pos}, #total={num_total}")


def get_subset_dataloader(
    base_dataset: Dataset,
    n_samples: int,
    require_both_labels: bool = False,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
) -> DataLoader:
    """
    작은 subset 데이터로더를 생성합니다.
    
    Args:
        base_dataset: 기본 Dataset
        n_samples: 선택할 샘플 수
        require_both_labels: True이면 label 0과 1이 최소 1개씩 포함되도록 선택
        batch_size: 배치 크기 (None이면 n_samples 전체를 한 배치로)
        shuffle: 셔플 여부
        
    Returns:
        작은 subset을 포함하는 DataLoader
    """
    dataset_size = len(base_dataset)
    if n_samples > dataset_size:
        logger.warning(
            f"[DEBUG] Requested {n_samples} samples but dataset has only {dataset_size}. "
            f"Using all {dataset_size} samples."
        )
        n_samples = dataset_size
    
    # require_both_labels가 True인 경우, label 0과 1을 각각 최소 1개씩 포함
    if require_both_labels:
        indices_0 = []
        indices_1 = []
        
        for i in range(min(dataset_size, n_samples * 2)):  # 충분히 탐색
            _, y = base_dataset[i]
            if isinstance(y, torch.Tensor):
                label = int(y.item() if y.numel() == 1 else y.cpu().numpy().flatten()[0])
            else:
                label = int(y)
            
            if label == 0 and len(indices_0) < n_samples // 2:
                indices_0.append(i)
            elif label == 1 and len(indices_1) < n_samples // 2:
                indices_1.append(i)
            
            if len(indices_0) + len(indices_1) >= n_samples:
                break
        
        # 부족한 경우 나머지로 채움
        remaining = n_samples - len(indices_0) - len(indices_1)
        if remaining > 0:
            for i in range(dataset_size):
                if i not in indices_0 and i not in indices_1:
                    if remaining <= 0:
                        break
                    _, y = base_dataset[i]
                    if isinstance(y, torch.Tensor):
                        label = int(y.item() if y.numel() == 1 else y.cpu().numpy().flatten()[0])
                    else:
                        label = int(y)
                    if label == 0:
                        indices_0.append(i)
                    else:
                        indices_1.append(i)
                    remaining -= 1
        
        indices = indices_0 + indices_1
    else:
        # 단순히 처음 n_samples개 선택
        indices = list(range(min(n_samples, dataset_size)))
    
    subset = Subset(base_dataset, indices)
    
    if batch_size is None:
        batch_size = n_samples  # 전체를 한 배치로
    
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle)


def check_backbone_feature_std(
    model: nn.Module,
    sample_input: torch.Tensor,
    device: torch.device,
    prefix: str = "[DEBUG][STD]",
) -> float:
    """
    Backbone의 마지막 feature 표준편차를 확인합니다.
    
    Args:
        model: 모델 인스턴스
        sample_input: 샘플 입력 텐서
        device: 디바이스
        prefix: 로그 출력 접두사
        
    Returns:
        feature 표준편차 값
    """
    model.eval()
    with torch.no_grad():
        sample_input = sample_input.to(device)
        
        # LSTM + Attention 모델의 경우, context vector를 추출
        # 모델 구조에 따라 조정 필요
        if hasattr(model, 'lstm') and hasattr(model, 'attention'):
            lstm_out, _ = model.lstm(sample_input)
            attention_scores = model.attention(lstm_out)
            attention_weights = torch.nn.functional.softmax(attention_scores, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1)
            h = context
        elif hasattr(model, 'classifier'):
            # classifier 이전의 feature를 추출하려고 시도
            # 실제 구조에 맞게 조정 필요
            logits = model(sample_input)
            # 여기서는 logits를 feature로 간주 (실제로는 더 이전 레이어를 봐야 함)
            h = logits
        else:
            # 일반적인 경우: forward의 중간 출력을 hook으로 추출
            # 간단하게 전체 forward 출력을 사용
            logits = model(sample_input)
            h = logits
        
        h_std = h.detach().float().std().item()
        
        if h_std < 1e-6:
            logger.warning(
                f"{prefix} feature std is too small ({h_std:.6f}) → "
                f"활성 함수/초기화/gradient 문제 가능성"
            )
        elif np.isnan(h_std) or np.isinf(h_std):
            logger.error(
                f"{prefix} feature std is NaN/Inf → 모델 구조 문제 가능성"
            )
        else:
            logger.info(f"{prefix} backbone feature std={h_std:.4f}")
        
        return h_std


def run_single_sample_overfit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    base_dataset: Dataset,
    device: torch.device,
    max_steps: int = 1000,
    lr_override: Optional[float] = None,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[bool, dict]:
    """
    샘플 1개에 대한 오버핏 테스트를 실행합니다.
    
    Args:
        model: 모델 인스턴스
        optimizer: 옵티마이저
        base_dataset: 기본 데이터셋
        loss_fn: 손실 함수 (None이면 FocalLoss 사용)
        device: 디바이스
        max_steps: 최대 학습 스텝
        lr_override: 학습률 오버라이드 (None이면 optimizer의 lr 사용)
        
    Returns:
        (성공 여부, 결과 딕셔너리)
    """
    logger.info("=" * 60)
    logger.info("[DEBUG][OVERFIT-1] Starting single sample overfit test")
    logger.info("=" * 60)
    
    # Label 분포 출력
    print_label_stats(base_dataset, prefix="[DEBUG][OVERFIT-1][LABEL]")
    
    # Label이 확실한 샘플 1개 선택
    sample_idx = None
    sample_label = None
    
    for i in range(len(base_dataset)):
        _, y = base_dataset[i]
        if isinstance(y, torch.Tensor):
            label = int(y.item() if y.numel() == 1 else y.cpu().numpy().flatten()[0])
        else:
            label = int(y)
        
        if label in [0, 1]:  # 유효한 label
            sample_idx = i
            sample_label = label
            break
    
    if sample_idx is None:
        logger.error("[ERROR][OVERFIT-1] No valid sample found")
        return False, {"error": "No valid sample"}
    
    logger.info(f"[DEBUG][OVERFIT-1] Selected sample_idx={sample_idx}, label={sample_label}")
    
    # 1개 샘플만 포함하는 dataloader 생성
    subset_loader = get_subset_dataloader(
        base_dataset,
        n_samples=1,
        require_both_labels=False,
        batch_size=1,
        shuffle=False,
    )
    
    # 학습률 설정
    if lr_override is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_override
        logger.info(f"[DEBUG][OVERFIT-1] Learning rate overridden to {lr_override}")
    
    # 손실 함수 설정
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()
    
    model.train()
    
    # Backbone feature std 확인
    X_sample, _ = base_dataset[sample_idx]
    if isinstance(X_sample, torch.Tensor):
        X_sample_batch = X_sample.unsqueeze(0).to(device)
    else:
        X_sample_batch = torch.tensor(X_sample).unsqueeze(0).to(device)
    check_backbone_feature_std(model, X_sample_batch, device, prefix="[DEBUG][OVERFIT-1][STD]")
    
    # 학습 루프
    best_loss = float('inf')
    best_prob = None
    
    for step in range(max_steps):
        for X_batch, y_batch in subset_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # y_batch shape 조정 (필요시)
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            
            optimizer.zero_grad()
            
            logits = model(X_batch)  # (B, 1) raw logits
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            # 확률 계산 (로깅용)
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                prob_up = float(probs.cpu().item())
            
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_prob = prob_up
            
            # 주기적으로 로그 출력
            if step % 100 == 0 or step == max_steps - 1:
                logger.info(
                    f"[DEBUG][OVERFIT-1] step={step}, loss={loss.item():.6f}, "
                    f"prob_up={prob_up:.4f}, label={sample_label}"
                )
    
    # 성공 기준 확인
    success = False
    if sample_label == 1:
        success = best_loss < 1e-3 and best_prob >= 0.99
    else:  # label == 0
        success = best_loss < 1e-3 and best_prob <= 0.01
    
    result = {
        "success": success,
        "final_loss": best_loss,
        "final_prob": best_prob,
        "label": sample_label,
        "steps": max_steps,
    }
    
    if success:
        logger.info(f"[DEBUG][OVERFIT-1] ✓ PASSED: loss={best_loss:.6f}, prob_up={best_prob:.4f}")
    else:
        logger.error(
            f"[ERROR][OVERFIT-1] ✗ FAILED: loss={best_loss:.6f}, prob_up={best_prob:.4f}, "
            f"label={sample_label} → 모델/옵티마이저/데이터 플로우 버그 가능성 매우 높음"
        )
    
    return success, result


def run_two_sample_overfit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    base_dataset: Dataset,
    device: torch.device,
    max_steps: int = 2000,
    lr_override: Optional[float] = None,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[bool, dict]:
    """
    샘플 2개(라벨 0/1)에 대한 오버핏 테스트를 실행합니다.

    Args:
        model: 모델 인스턴스
        optimizer: 옵티마이저
        base_dataset: 기본 데이터셋
        device: 디바이스
        max_steps: 최대 학습 스텝
        lr_override: 학습률 오버라이드
        loss_fn: 손실 함수

    Returns:
        (성공 여부, 결과 딕셔너리)
    """
    logger.info("=" * 60)
    logger.info("[DEBUG][OVERFIT-2] Starting two sample overfit test")
    logger.info("=" * 60)

    # Dropout 설정 확인
    dropout_ps = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    if dropout_ps:
        logger.info(f"[CHECK][OVERFIT-2][DROPOUT] dropout_ps={dropout_ps}")
    else:
        logger.info("[CHECK][OVERFIT-2][DROPOUT] No Dropout layers found")

    # Label 분포 출력
    print_label_stats(base_dataset, prefix="[DEBUG][OVERFIT-2][LABEL]")

    # Label이 서로 다른 두 샘플 선택
    sample_0_idx: Optional[int] = None
    sample_1_idx: Optional[int] = None

    for i in range(len(base_dataset)):
        _, y = base_dataset[i]
        if isinstance(y, torch.Tensor):
            label = int(y.item() if y.numel() == 1 else y.cpu().numpy().flatten()[0])
        else:
            label = int(y)

        if label == 0 and sample_0_idx is None:
            sample_0_idx = i
        elif label == 1 and sample_1_idx is None:
            sample_1_idx = i

        if sample_0_idx is not None and sample_1_idx is not None:
            break

    if sample_0_idx is None or sample_1_idx is None:
        logger.error("[ERROR][OVERFIT-2] Could not find samples with both labels (0 and 1)")
        return False, {"error": "Could not find samples with both labels"}

    logger.info(
        f"[DEBUG][OVERFIT-2] Selected sample_0_idx={sample_0_idx} (label=0), "
        f"sample_1_idx={sample_1_idx} (label=1)"
    )

    # ------------------------------------------------------------
    # [CHECK][OVERFIT-2][INPUT_DIFF]
    # 두 샘플의 원본 입력(X_0, X_1)이 실제로 얼마나 다른지 확인하는 디버그 로그
    #  - 전체 abs diff의 mean/std/max
    #  - feature_dim 방향(per-feature) 평균 abs diff (앞 몇 개만)
    # ------------------------------------------------------------
    X_0_raw, _ = base_dataset[sample_0_idx]
    X_1_raw, _ = base_dataset[sample_1_idx]

    # Tensor가 아니면 Tensor로 변환
    if not isinstance(X_0_raw, torch.Tensor):
        X_0 = torch.tensor(X_0_raw)
    else:
        X_0 = X_0_raw

    if not isinstance(X_1_raw, torch.Tensor):
        X_1 = torch.tensor(X_1_raw)
    else:
        X_1 = X_1_raw

    # float32로 캐스팅
    X_0 = X_0.float()
    X_1 = X_1.float()

    # shape 체크
    if X_0.shape != X_1.shape:
        logger.warning(
            f"[CHECK][OVERFIT-2][INPUT_SHAPE_MISMATCH] "
            f"X_0.shape={tuple(X_0.shape)}, X_1.shape={tuple(X_1.shape)}"
        )
    else:
        diff = (X_1 - X_0).abs()

        diff_mean = diff.mean().item()
        diff_std = diff.std().item() if diff.numel() > 1 else 0.0
        diff_max = diff.max().item()

        logger.info(
            "[CHECK][OVERFIT-2][INPUT_DIFF] "
            f"abs_diff: mean={diff_mean:.6f}, std={diff_std:.6f}, max={diff_max:.6f}, "
            f"shape={tuple(diff.shape)}"
        )

        # 보통 (seq_len, feature_dim) 형태라고 가정하고
        # 마지막 차원(feature_dim)에 대해 per-feature mean abs diff 계산
        if diff.dim() >= 2:
            # 마지막 축 기준 feature-wise mean
            per_feat_mean = diff.mean(dim=0) if diff.dim() == 2 else diff.view(-1, diff.shape[-1]).mean(dim=0)
            feat_dim = per_feat_mean.shape[0]
            k = min(8, feat_dim)  # 앞에서 최대 8개만 출력

            first_k_vals = ", ".join(
                f"{per_feat_mean[i].item():.6f}" for i in range(k)
            )
            logger.info(
                "[CHECK][OVERFIT-2][INPUT_DIFF_PER_FEATURE] "
                f"first_{k}_feature_mean_abs_diff=[{first_k_vals}] "
                f"(feature_dim={feat_dim})"
            )
        else:
            logger.info(
                "[CHECK][OVERFIT-2][INPUT_DIFF_PER_FEATURE] "
                f"skipped (unexpected diff.dim={diff.dim()})"
            )

    # 선택된 2개 샘플에 대한 backbone feature std 체크
    X_0, _ = base_dataset[sample_0_idx]
    X_1, _ = base_dataset[sample_1_idx]
    if not isinstance(X_0, torch.Tensor):
        X_0 = torch.tensor(X_0)
    if not isinstance(X_1, torch.Tensor):
        X_1 = torch.tensor(X_1)
    X_pair = torch.stack([X_0, X_1], dim=0).to(device)
    check_backbone_feature_std(
        model,
        X_pair,
        device,
        prefix="[DEBUG][OVERFIT-2][STD]",
    )

    # ★ 핵심 수정 포인트 ★
    # get_subset_dataloader(require_both_labels=True)를 사용하는 대신,
    # 이미 찾은 sample_0_idx / sample_1_idx를 직접 사용해서 Subset/DataLoader를 구성한다.
    subset_indices = [sample_0_idx, sample_1_idx]
    subset = Subset(base_dataset, subset_indices)
    subset_loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=True,  # 순서는 섞어도 되지만, 두 개 샘플만 포함되도록 보장됨
    )

    # [작업 1] overfit-2용 라벨이 처음부터 올바른지 확인하는 로그
    temp_iter = iter(subset_loader)
    X_temp, y_overfit = next(temp_iter)
    y_overfit = y_overfit.to(device)
    if y_overfit.dim() == 1:
        y_overfit = y_overfit.unsqueeze(1)
    logger.info(
        f"[CHECK][OVERFIT-2][RAW_LABEL] y_overfit={y_overfit.detach().cpu().tolist()}, "
        f"shape={tuple(y_overfit.shape)}, dtype={y_overfit.dtype}, "
        f"unique={torch.unique(y_overfit).cpu().tolist()}"
    )

    # RAW_LABEL 확인 후, 다시 학습용 DataLoader를 생성해도 되지만
    # 여기서는 동일한 subset으로 다시 한 번 DataLoader를 만들어 명시적으로 사용한다.
    subset_loader = DataLoader(
        subset,
        batch_size=2,
        shuffle=True,
    )

    # 학습률 설정
    if lr_override is not None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_override
        logger.info(f"[DEBUG][OVERFIT-2] Learning rate overridden to {lr_override}")

    # 손실 함수 설정
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()

    model.train()

    # 학습 루프
    best_loss = float("inf")
    best_probs = None

    for step in range(max_steps):
        for X_batch, y_batch in subset_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)

            # [작업 2] DataLoader 루프 초반에 y_batch의 실체를 확인하는 로그 추가
            if step < 3:
                logger.info(
                    f"[CHECK][OVERFIT-2][DATALOADER] step={step}, "
                    f"y_batch={y_batch.detach().cpu().tolist()}, "
                    f"unique={torch.unique(y_batch).cpu().tolist()}, "
                    f"shape={tuple(y_batch.shape)}, dtype={y_batch.dtype}"
                )

            # [목표 1] y_batch 디버그 로그 추가
            logger.info(
                f"[DEBUG][OVERFIT-2] y_batch={y_batch.detach().cpu().tolist()}, "
                f"shape={tuple(y_batch.shape)}, dtype={y_batch.dtype}"
            )

            optimizer.zero_grad()

            # logits: (B, 1) 또는 (B,) 형태의 raw logits라고 가정
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            # [작업 3] loss 계산에 쓰이는 y와 로그에서 표시하는 label이 동일한 텐서인지 확인
            logger.info(
                f"[CHECK][OVERFIT-2][LOSS-INPUT] loss_y={y_batch.detach().cpu().tolist()}"
            )

            loss.backward()
            # FC 레이어 초기 상태 체크
            if hasattr(model, "fc_out"):
                with torch.no_grad():
                    w = model.fc_out.weight
                    b = model.fc_out.bias
                    logger.info(
                        f"[CHECK][OVERFIT-2][FC_BEFORE] "
                        f"w_mean={w.mean().item():.6f}, w_std={w.std().item():.6f}, "
                        f"b_mean={b.mean().item():.6f}, b_std={b.std().item():.6f}"
                    )

            optimizer.step()

            # [목표 2] best_loss와 best_probs 업데이트
            if loss.item() < best_loss:
                best_loss = loss.item()

                # 각 샘플별 확률 계산 및 저장
                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    # probs shape이 (B, 1)일 경우를 대비해 .view(-1)로 평탄화
                    probs_flat = probs.view(-1)
                    best_probs = {
                        "sample_0": float(probs_flat[0].item()),
                        "sample_1": float(probs_flat[1].item()),
                    }

                # 성공 조건 충족 시 즉시 반환
                if (
                    best_probs["sample_0"] <= 0.01
                    and best_probs["sample_1"] >= 0.99
                    and best_loss < 1e-3
                ):
                    logger.info(
                        f"[DEBUG][OVERFIT-2] ✓ PASSED: loss={best_loss:.6f}, "
                        f"best_prob_0={best_probs['sample_0']:.4f}, "
                        f"best_prob_1={best_probs['sample_1']:.4f}"
                    )
                    return True, {
                        "success": True,
                        "final_loss": best_loss,
                        "sample_0": {"prob": best_probs["sample_0"], "label": 0},
                        "sample_1": {"prob": best_probs["sample_1"], "label": 1},
                        "steps": step + 1,
                    }

            # 주기적으로 로그 출력
            if step % 200 == 0 or step == max_steps - 1:
                with torch.no_grad():
                    # [작업 3] y_batch에서 직접 label 추출하여 일관성 확인
                    label_0 = float(y_batch[0].item())
                    label_1 = float(y_batch[1].item())

                    # base_dataset에서 직접 샘플을 꺼내서 forward 결과 확인
                    X_0, _ = base_dataset[sample_0_idx]
                    X_1, _ = base_dataset[sample_1_idx]

                    if isinstance(X_0, torch.Tensor):
                        X_0_batch = X_0.unsqueeze(0).to(device)
                    else:
                        X_0_batch = torch.tensor(X_0).unsqueeze(0).to(device)

                    if isinstance(X_1, torch.Tensor):
                        X_1_batch = X_1.unsqueeze(0).to(device)
                    else:
                        X_1_batch = torch.tensor(X_1).unsqueeze(0).to(device)

                    logits_0 = model(X_0_batch)
                    logits_1 = model(X_1_batch)
                    prob_0 = float(torch.sigmoid(logits_0).view(-1)[0].item())
                    prob_1 = float(torch.sigmoid(logits_1).view(-1)[0].item())

                    logger.info(
                        f"[DEBUG][OVERFIT-2] step={step}, loss={loss.item():.6f}, "
                        f"sample_0: prob_up={prob_0:.4f} (label={label_0:.0f}), "
                        f"sample_1: prob_up={prob_1:.4f} (label={label_1:.0f})"
                    )

    # [목표 2] 실패 시 로그 출력 및 반환
    # FC 레이어 학습 후 상태 체크
    if hasattr(model, "fc_out"):
        with torch.no_grad():
            w = model.fc_out.weight
            b = model.fc_out.bias
            logger.info(
                f"[CHECK][OVERFIT-2][FC_AFTER] "
                f"w_mean={w.mean().item():.6f}, w_std={w.std().item():.6f}, "
                f"b_mean={b.mean().item():.6f}, b_std={b.std().item():.6f}"
            )

    if best_probs is None:
        logger.error("[ERROR][OVERFIT-2] ✗ FAILED: no valid step recorded")
    else:
        logger.error(
            f"[ERROR][OVERFIT-2] ✗ FAILED: loss={best_loss:.6f}, "
            f"best_prob_0={best_probs['sample_0']:.4f}, "
            f"best_prob_1={best_probs['sample_1']:.4f}"
        )

    return False, None


def run_small_batch_overfit(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    base_dataset: Dataset,
    device: torch.device,
    n_samples: int = 32,
    max_steps: int = 5000,
    lr_override: Optional[float] = None,
    loss_fn: Optional[nn.Module] = None,
) -> Tuple[bool, dict]:
    """
    작은 배치(32개 또는 64개)에 대한 오버핏 테스트를 실행합니다.
    
    Args:
        model: 모델 인스턴스
        optimizer: 옵티마이저
        base_dataset: 기본 데이터셋
        device: 디바이스
        n_samples: 샘플 수 (32 또는 64)
        max_steps: 최대 학습 스텝
        lr_override: 학습률 오버라이드
        loss_fn: 손실 함수
        
    Returns:
        (성공 여부, 결과 딕셔너리)
    """
    logger.info("=" * 60)
    logger.info(f"[DEBUG][OVERFIT-{n_samples}] Starting {n_samples}-sample overfit test")
    logger.info("=" * 60)
    
    # Label 분포 출력
    print_label_stats(base_dataset, prefix=f"[DEBUG][OVERFIT-{n_samples}][LABEL]")
    
    # n_samples개 샘플 포함하는 dataloader 생성
    subset_loader = get_subset_dataloader(
        base_dataset,
        n_samples=n_samples,
        require_both_labels=True,
        batch_size=min(32, n_samples),
        shuffle=True,
    )
    
    # 학습률 설정
    if lr_override is not None:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_override
    
    # 손실 함수 설정
    if loss_fn is None:
        loss_fn = nn.BCEWithLogitsLoss()
    
    model.train()
    
    # 학습 루프
    best_loss = float('inf')
    best_acc = 0.0
    
    for step in range(max_steps):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for X_batch, y_batch in subset_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            if y_batch.dim() == 1:
                y_batch = y_batch.unsqueeze(1)
            
            optimizer.zero_grad()
            
            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 정확도 계산
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                correct += (predictions == y_batch).sum().item()
                total += y_batch.size(0)
        
        avg_loss = epoch_loss / len(subset_loader)
        acc = correct / total if total > 0 else 0.0
        
        if avg_loss < best_loss:
            best_loss = avg_loss
        if acc > best_acc:
            best_acc = acc
        
        # 주기적으로 로그 출력
        if step % 500 == 0 or step == max_steps - 1:
            logger.info(
                f"[DEBUG][OVERFIT-{n_samples}] step={step}, loss={avg_loss:.6f}, "
                f"acc={acc:.4f}"
            )
    
    # 성공 기준: loss가 충분히 줄고 accuracy가 높은 수준
    success = best_loss < 0.1 and best_acc >= 0.9
    
    result = {
        "success": success,
        "final_loss": best_loss,
        "final_acc": best_acc,
        "steps": max_steps,
    }
    
    if success:
        logger.info(
            f"[DEBUG][OVERFIT-{n_samples}] ✓ PASSED: loss={best_loss:.6f}, acc={best_acc:.4f}"
        )
    else:
        logger.warning(
            f"[WARN][OVERFIT-{n_samples}] Partially passed: loss={best_loss:.6f}, acc={best_acc:.4f}"
        )
    
    return success, result

