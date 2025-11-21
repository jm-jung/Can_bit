"""
Logging configuration
"""
import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler

from src.core.config import settings


def setup_logging():
    """
    Setup application logging configuration
    
    uvicorn 환경에서도 src.* 로거가 터미널에 출력되도록 설정합니다.
    uvicorn이 먼저 로깅을 설정할 수 있으므로, root 로거에 직접 핸들러를 추가합니다.
    """
    # Create logs directory if it doesn't exist
    log_file_path = Path(settings.LOG_FILE)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 로그 포맷 정의
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Root 로거 가져오기
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # 기존 핸들러 확인 및 추가
    # uvicorn이 이미 핸들러를 설정했을 수 있으므로, 우리 핸들러가 없으면 추가
    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout
        for h in root_logger.handlers
    )
    has_file_handler = any(
        isinstance(h, RotatingFileHandler) and h.baseFilename == str(log_file_path.resolve())
        for h in root_logger.handlers
    )
    
    # Console handler 추가 (없는 경우만)
    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(console_handler)
    
    # File handler 추가 (없는 경우만)
    if not has_file_handler:
        file_handler = RotatingFileHandler(
            settings.LOG_FILE,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        root_logger.addHandler(file_handler)
    
    # src.* 패키지 로거 설정
    # propagate=True (기본값)로 설정하여 root 로거로 전파되도록 보장
    src_logger = logging.getLogger("src")
    src_logger.setLevel(log_level)
    src_logger.propagate = True  # root 로거로 전파 (명시적으로 설정)
    
    # uvicorn 로거 레벨 설정 (기존 핸들러 유지)
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    # 설정 완료 로그
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully (root level=%s, src.* propagate=True)", log_level)
    
    return logger

