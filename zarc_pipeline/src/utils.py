"""
Small helpers shared across modules: logging, safe numeric coercion,
and a uniform 'group key' string for spectra.
"""
import logging
import sys

import numpy as np
import pandas as pd


def get_logger(name: str = "zarc") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
    return logger


def to_numeric_strip(series: pd.Series) -> pd.Series:
    """Strip whitespace and quotes, then coerce to float (NaN on failure)."""
    if series.dtype == object:
        series = series.astype(str).str.strip().str.strip("'\"")
    return pd.to_numeric(series, errors="coerce")


def spectrum_key(ionic_radius: float, temperature: float) -> str:
    return f"r={ionic_radius:.3e}_T={temperature:g}"


def safe_log10(x):
    """log10 that returns NaN for non-positive values instead of warning."""
    arr = np.asarray(x, dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    pos = arr > 0
    out[pos] = np.log10(arr[pos])
    return out
