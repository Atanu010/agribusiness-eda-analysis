"""
Utility module for logging, directory management, runtime timers, and
data formatting across the agribusiness EDA analytics framework.
"""

import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional
import pandas as pd


def setup_logger(
    name: str = "agribusiness_eda",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns a centralized, standardized logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | [%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def ensure_directory(path: Path) -> Path:
    """Ensures that a specified directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def timeit(func: Callable) -> Callable:
    """Decorator to measure and log the execution runtime of an analytical step."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logger.info(f"Starting '{func.__name__}'...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        logger.info(f"Finished '{func.__name__}' in {elapsed:.4f} seconds.")
        return result
    return wrapper
