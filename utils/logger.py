"""
utils/logger.py — Structured, levelled logging for the entire system.

Every module obtains its logger via `get_logger(__name__)` so log records
carry the originating module path automatically.  In production this can
be swapped for JSON-structured logging (e.g. structlog) without touching
any call-sites.
"""

import logging
import sys
from functools import lru_cache

from config import settings

# ── Formatter ────────────────────────────────────────────────────────────────

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _build_handler() -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    return handler


# ── Public API ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger configured to the level set in AppConfig.

    Results are cached so calling get_logger('foo') many times returns
    the exact same object — no duplicate handlers accumulate.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.addHandler(_build_handler())

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    logger.propagate = False  # prevent double-printing through the root logger
    return logger
