"""Structured logging shared by every pipeline stage.

Logs go to the console and to a rotating file in logs/pipeline.log so failures in
scheduled (Airflow) runs can be investigated after the fact.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler

from .config import LOG_DIR

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    global _configured
    if not _configured:
        root = logging.getLogger("cyber_pipeline")
        root.setLevel(logging.INFO)
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(_FORMAT))
        file_handler = RotatingFileHandler(LOG_DIR / "pipeline.log", maxBytes=5_000_000, backupCount=3)
        file_handler.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)
        root.addHandler(file_handler)
        _configured = True
    return logging.getLogger(f"cyber_pipeline.{name}")


@contextmanager
def log_duration(logger: logging.Logger, step: str):
    """Log how long a pipeline step took, and log (then re-raise) any error."""
    start = time.perf_counter()
    logger.info("START %s", step)
    try:
        yield
    except Exception:
        logger.exception("FAILED %s after %.2fs", step, time.perf_counter() - start)
        raise
    logger.info("DONE %s in %.2fs", step, time.perf_counter() - start)
