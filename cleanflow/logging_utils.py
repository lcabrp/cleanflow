"""Small opt-in logging helpers used by CleanFlow library code."""

from __future__ import annotations

import logging


def get_logger(name: str = "cleanflow", level: int = logging.INFO, enabled: bool = False) -> logging.Logger | None:
    """Return a configured logger when logging is requested.

    Library functions stay quiet by default; callers can opt in without every
    function carrying print statements.
    """
    if not enabled:
        return None

    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(handler)
    return logger


def log(logger: logging.Logger | None, message: str) -> None:
    """Log only when an opt-in logger was provided."""
    if logger is not None:
        logger.info(message)

