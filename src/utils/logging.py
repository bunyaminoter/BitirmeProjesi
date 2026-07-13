"""
Rich-based logging setup.

Configures Python logging with Rich handlers for beautiful
console output with colors, formatting, and tracebacks.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    use_rich: bool = True,
) -> logging.Logger:
    """Configure logging for the project.

    Args:
        level: Logging level string ('DEBUG', 'INFO', 'WARNING', etc.).
        log_file: Optional path to a log file.
        use_rich: Whether to use Rich for console formatting.

    Returns:
        Configured root logger.
    """
    logger = logging.getLogger("asl")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_path=False,
            )
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the 'asl' namespace.

    Args:
        name: Logger name (will be prefixed with 'asl.').

    Returns:
        Logger instance.
    """
    return logging.getLogger(f"asl.{name}")
