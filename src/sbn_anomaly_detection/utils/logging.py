"""Structured logging setup for the SBN anomaly detection pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger with console (and optionally file) handlers.

    Parameters
    ----------
    level:
        Minimum log level (e.g. ``logging.DEBUG``, ``logging.INFO``).
    log_file:
        If provided, also write logs to this file path.
    fmt:
        Log-record format string.
    datefmt:
        Date/time format used in log records.
    """
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any pre-existing handlers to avoid duplicate output
    root.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Optional file handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root.info("Logging initialised (level=%s)", logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """Return a named logger scoped to this package.

    Usage::

        logger = get_logger(__name__)
        logger.info("Training started")
    """
    return logging.getLogger(name)
