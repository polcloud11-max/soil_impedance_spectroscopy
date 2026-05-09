"""
Live-sensor ingestion for the rover impedance probe.

The bus delivers each frequency sweep as a JSON-serialisable payload.
``load_sensor_sweep`` accepts both supported shapes and returns a
:class:`Spectrum` ready for :func:`zarc_fitting.fit_spectrum`.

Wire formats accepted
---------------------
1. **Per-point list** (one dict per frequency sample)::

       [
         {"frequency_hz": 1.0e6, "re_z_ohm": 3160.0, "neg_im_z_ohm": 9670.0},
         {"frequency_hz": 9.26e5, "re_z_ohm": 3480.0, "neg_im_z_ohm": 10200.0},
         ...
       ]

2. **Column-array dict** (one dict for the whole sweep)::

       {
         "sample_id":     "soil-001",                     # optional
         "timestamp":     "2026-05-09T14:21:03Z",         # optional
         "frequency_hz":  [1.0e6, 9.26e5, ...],
         "re_z_ohm":      [3160.0, 3480.0, ...],
         "neg_im_z_ohm":  [9670.0, 10200.0, ...]
       }

A raw JSON string in either of the above shapes is also accepted.

Sign convention
---------------
``neg_im_z_ohm`` is **-Im(Z)** as transmitted by the probe (positive for
capacitive media). Internally we store ``Z = Re(Z) + j*Im(Z)`` with
``Im(Z) <= 0`` so downstream maths matches standard EIS textbooks.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .preprocessing import Spectrum
from .utils import get_logger
from config import MIN_POINTS_PER_SPECTRUM

log = get_logger()

# Required field names on the wire
_REQUIRED_KEYS: Tuple[str, str, str] = (
    "frequency_hz", "re_z_ohm", "neg_im_z_ohm"
)

PointDict = Mapping[str, float]
SweepPayload = Union[str, bytes, Mapping[str, Any], Sequence[PointDict]]


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------
def load_sensor_sweep(data_payload: SweepPayload) -> Spectrum:
    """Convert a single sensor-bus payload into a :class:`Spectrum`.

    Parameters
    ----------
    data_payload
        Raw payload from the rover sensor bus. May be a JSON string, a
        list of point-dicts, or a column-array dict (see module docstring).

    Returns
    -------
    Spectrum
        Validated, frequency-sorted spectrum with ``Im(Z) <= 0``.

    Raises
    ------
    ValueError
        If required fields are missing, arrays are mis-aligned, or the
        sweep contains fewer than :data:`config.MIN_POINTS_PER_SPECTRUM`
        valid points.
    """
    payload = _coerce_to_python(data_payload)

    if isinstance(payload, Mapping):
        f_hz, re_z, neg_im_z, sample_id, timestamp = _from_column_dict(payload)
    elif isinstance(payload, Sequence):
        f_hz, re_z, neg_im_z, sample_id, timestamp = _from_point_list(payload)
    else:
        raise ValueError(
            f"Unsupported payload type: {type(payload).__name__}. "
            "Expected JSON string, dict of arrays, or list of point-dicts."
        )

    # Sign convention: probe sends -Im(Z) as positive; store internal Im <= 0
    Z = re_z.astype(np.float64) - 1j * np.abs(neg_im_z.astype(np.float64))

    spectrum = Spectrum(
        f=f_hz.astype(np.float64),
        Z=Z,
        sample_id=sample_id,
        timestamp=timestamp,
    )

    if spectrum.n_points < MIN_POINTS_PER_SPECTRUM:
        raise ValueError(
            f"Sweep has only {spectrum.n_points} valid points; "
            f"need at least {MIN_POINTS_PER_SPECTRUM} for a stable ZARC fit."
        )

    log.info(
        f"Loaded sweep "
        f"(sample_id={spectrum.sample_id}, n={spectrum.n_points}, "
        f"f={spectrum.f.min():.3g}-{spectrum.f.max():.3g} Hz)"
    )
    return spectrum


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _coerce_to_python(payload: SweepPayload) -> Union[Mapping, Sequence]:
    """Decode bytes/str payloads to native Python objects."""
    if isinstance(payload, (bytes, bytearray)):
        payload = payload.decode("utf-8")
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Payload is not valid JSON: {exc}") from exc
    return payload


def _from_column_dict(
    payload: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    """Extract aligned arrays from a column-array dict."""
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"Payload is missing required keys: {missing}")

    try:
        f_hz = np.asarray(payload["frequency_hz"], dtype=np.float64)
        re_z = np.asarray(payload["re_z_ohm"], dtype=np.float64)
        neg_im_z = np.asarray(payload["neg_im_z_ohm"], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not coerce payload arrays to float64: {exc}") from exc

    if not (f_hz.shape == re_z.shape == neg_im_z.shape):
        raise ValueError(
            f"Array length mismatch: f={f_hz.shape}, "
            f"re={re_z.shape}, neg_im={neg_im_z.shape}"
        )
    if f_hz.ndim != 1:
        raise ValueError(f"Expected 1-D arrays, got ndim={f_hz.ndim}")

    sample_id = payload.get("sample_id")
    timestamp = payload.get("timestamp")
    return f_hz, re_z, neg_im_z, _opt_str(sample_id), _opt_str(timestamp)


def _from_point_list(
    payload: Sequence[PointDict],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[str], Optional[str]]:
    """Extract aligned arrays from a list of per-point dicts."""
    if len(payload) == 0:
        raise ValueError("Empty sweep payload")

    n = len(payload)
    f_hz = np.empty(n, dtype=np.float64)
    re_z = np.empty(n, dtype=np.float64)
    neg_im_z = np.empty(n, dtype=np.float64)

    for i, pt in enumerate(payload):
        if not isinstance(pt, Mapping):
            raise ValueError(
                f"Point {i} is not a dict: got {type(pt).__name__}"
            )
        missing = [k for k in _REQUIRED_KEYS if k not in pt]
        if missing:
            raise ValueError(f"Point {i} is missing keys: {missing}")
        try:
            f_hz[i] = float(pt["frequency_hz"])
            re_z[i] = float(pt["re_z_ohm"])
            neg_im_z[i] = float(pt["neg_im_z_ohm"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Point {i} has non-numeric field: {exc}"
            ) from exc

    return f_hz, re_z, neg_im_z, None, None


def _opt_str(value: Any) -> Optional[str]:
    return None if value is None else str(value)
