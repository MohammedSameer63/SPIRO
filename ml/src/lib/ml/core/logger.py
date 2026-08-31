"""
SPIRO ML — Logger
Wraps loguru with rich formatting and optional file output.
All modules call get_logger(__name__) for consistent structured logs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from loguru import logger as _logger


def get_logger(name: str = "spiro_ml", log_file: Optional[Path] = None):
    """
    Return a configured loguru logger.

    Parameters
    ----------
    name : str
        Module name shown in logs.
    log_file : Path, optional
        If provided, also write logs to this file.

    Returns
    -------
    loguru.Logger
    """
    # Remove default sink
    _logger.remove()

    fmt = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level:<8}</level> | "
        f"<cyan>{name}</cyan> | "
        "<level>{message}</level>"
    )

    # Console sink
    _logger.add(
        sys.stdout,
        format=fmt,
        level="DEBUG",
        colorize=True,
    )

    # File sink
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        _logger.add(
            str(log_file),
            format=fmt,
            level="DEBUG",
            rotation="50 MB",
            retention="30 days",
            compression="zip",
            colorize=False,
        )

    return _logger.bind(module=name)


# Module-level default logger
log = get_logger("spiro_ml")
