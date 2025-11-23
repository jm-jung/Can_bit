"""
Gradient flow and detach/no_grad/numpy usage checker.

학습 경로에서 gradient를 끊을 수 있는 구문을 검사합니다.
"""
from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def check_detach_no_grad_numpy_usage(file_path: str | Path) -> List[Tuple[int, str, str]]:
    """
    파일에서 detach(), no_grad(), numpy() 사용을 검사합니다.
    
    Args:
        file_path: 검사할 파일 경로
        
    Returns:
        (line_number, issue_type, code_snippet) 리스트
    """
    file_path = Path(file_path)
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return []
    
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content, filename=str(file_path))
        
        for node in ast.walk(tree):
            # .detach() 호출 검사
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr == 'detach':
                        line_no = node.lineno
                        # 간단한 코드 스니펫 추출 (실제로는 더 정교하게 할 수 있음)
                        lines = content.split('\n')
                        if line_no <= len(lines):
                            code_snippet = lines[line_no - 1].strip()
                            issues.append((line_no, 'detach', code_snippet))
                    
                    elif node.func.attr == 'numpy':
                        line_no = node.lineno
                        lines = content.split('\n')
                        if line_no <= len(lines):
                            code_snippet = lines[line_no - 1].strip()
                            issues.append((line_no, 'numpy', code_snippet))
            
            # with torch.no_grad() 검사
            if isinstance(node, ast.With):
                for item in node.items:
                    if isinstance(item.context_expr, ast.Call):
                        if isinstance(item.context_expr.func, ast.Attribute):
                            if (item.context_expr.func.attr == 'no_grad' and
                                isinstance(item.context_expr.func.value, ast.Name) and
                                item.context_expr.func.value.id == 'torch'):
                                line_no = node.lineno
                                lines = content.split('\n')
                                if line_no <= len(lines):
                                    code_snippet = lines[line_no - 1].strip()
                                    issues.append((line_no, 'no_grad', code_snippet))
    
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {e}")
        return []
    
    return issues


def report_gradient_issues(file_paths: List[str | Path]) -> None:
    """
    여러 파일에 대해 gradient 이슈를 검사하고 리포트합니다.
    
    Args:
        file_paths: 검사할 파일 경로 리스트
    """
    logger.info("=" * 60)
    logger.info("[DEBUG][GRADIENT] Checking for detach/no_grad/numpy usage")
    logger.info("=" * 60)
    
    all_issues = []
    
    for file_path in file_paths:
        file_path = Path(file_path)
        issues = check_detach_no_grad_numpy_usage(file_path)
        
        if issues:
            logger.warning(f"[DEBUG][GRADIENT] Found {len(issues)} potential issues in {file_path}:")
            for line_no, issue_type, code_snippet in issues:
                logger.warning(
                    f"  Line {line_no}: {issue_type} - {code_snippet[:80]}"
                )
                all_issues.append((file_path, line_no, issue_type, code_snippet))
        else:
            logger.info(f"[DEBUG][GRADIENT] No issues found in {file_path}")
    
    if all_issues:
        logger.warning(
            f"[WARN][GRADIENT] Total {len(all_issues)} potential gradient flow issues found. "
            f"Please review these carefully - they may break gradient flow in training."
        )
    else:
        logger.info("[DEBUG][GRADIENT] ✓ No gradient flow issues detected")

