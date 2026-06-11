"""Logging configuration helpers."""

from __future__ import annotations

import logging
import sys
from colorlog import ColoredFormatter

def setup_logging(
    level: str = "INFO",
    fmt: str = "%(asctime)s %(levelname)-8s %(name)s – %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> None:
    """Configure root logger with a StreamHandler to stdout.

    Parameters
    ----------
    level:
        Logging level string (``'DEBUG'``, ``'INFO'``, etc.).
    fmt:
        Log message format string.
    datefmt:
        Date/time format string.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(ColoredFormatter(
        fmt="%(log_color)s" + fmt,
        datefmt=datefmt,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    ))    
    root_logger = logging.getLogger()
    # Avoid adding duplicate handlers in interactive environments.
    if not root_logger.handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)
