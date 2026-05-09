"""Tiny shared helpers for the rover pipeline."""
from __future__ import annotations

import logging
import sys


def get_logger(name: str = "zarc") -> logging.Logger:
    """Return a configured stderr logger; reused across modules."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s",
                              datefmt="%Y-%m-%dT%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
