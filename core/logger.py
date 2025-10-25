"""Minimal logging helpers shared across the code base."""

from __future__ import annotations

import logging

__all__ = [
    "get_logger",
    "log_debug",
    "log_error",
    "log_info",
    "log_step",
    "log_warn",
    "set_level",
]


_LOGGER_NAME = "evaluador_informes"
_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"


def _configure_logger(logger: logging.Logger) -> None:
    """Attach a default stream handler to the provided logger if needed."""

    if logger.handlers:
        # The logger was already configured elsewhere (e.g. by tests).  Avoid
        # adding duplicate handlers that would result in duplicated messages.
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
    logger.addHandler(handler)
    logger.propagate = False

    try:  # Import lazily to avoid circular imports during test discovery.
        from core.config import LOG_LEVEL
    except Exception:  # pragma: no cover - only triggered in broken setups.
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(LOG_LEVEL)


def get_logger() -> logging.Logger:
    """Return the shared application logger.

    The first time the logger is requested we attach a console handler and use
    the log level specified in :mod:`core.config`.  Subsequent calls reuse the
    same instance, allowing callers to customise handlers if needed without the
    helper reconfiguring it behind the scenes.
    """

    logger = logging.getLogger(_LOGGER_NAME)
    _configure_logger(logger)
    return logger


def set_level(level: int | str) -> None:
    """Update the logging level for the shared logger."""

    get_logger().setLevel(level)


def _log(level: int, message: str) -> None:
    """Internal helper used by the convenience wrappers below."""

    get_logger().log(level, message)


def log_debug(message: str) -> None:
    _log(logging.DEBUG, message)


def log_info(message: str) -> None:
    _log(logging.INFO, message)


def log_warn(message: str) -> None:
    _log(logging.WARNING, message)


def log_error(message: str) -> None:
    _log(logging.ERROR, message)


def log_step(title: str) -> None:
    """Highlight pipeline milestones with a clear, multi-line banner."""

    separator = "=" * 50
    for line in ("", separator, title.upper(), separator):
        _log(logging.INFO, line)