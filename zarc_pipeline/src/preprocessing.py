"""
Spectrum container used throughout the rover pipeline.

A ``Spectrum`` holds one frequency sweep:

* ``f``     - frequencies in Hz (positive, descending after construction)
* ``Z``     - complex impedance Re(Z) + j*Im(Z) with ``Im(Z) <= 0``
              (capacitive convention; the sensor sends ``-Im(Z)`` as a
              positive number, the loader negates it on the way in)

Optional metadata fields ``sample_id`` and ``timestamp`` are propagated
into the JSON output but never affect the fit.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Spectrum:
    """Sorted, validated impedance sweep ready for ZARC fitting."""

    __slots__ = ("sample_id", "timestamp", "f", "omega", "Z", "n_points")

    def __init__(
        self,
        f: np.ndarray,
        Z: np.ndarray,
        sample_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        if f.shape != Z.shape:
            raise ValueError(
                f"frequency and impedance arrays must match shape: "
                f"{f.shape} vs {Z.shape}"
            )

        # Drop non-finite points and non-positive frequencies
        mask = np.isfinite(f) & np.isfinite(Z) & (f > 0)
        f = f[mask]
        Z = Z[mask]

        # Sort high -> low frequency (typical EIS convention)
        order = np.argsort(-f)
        self.f: np.ndarray = f[order].astype(np.float64)
        self.omega: np.ndarray = 2.0 * np.pi * self.f
        self.Z: np.ndarray = Z[order].astype(np.complex128)
        self.n_points: int = int(self.f.size)
        self.sample_id: Optional[str] = sample_id
        self.timestamp: Optional[str] = timestamp

    # ----- derived views ---------------------------------------------
    @property
    def re(self) -> np.ndarray:
        return self.Z.real

    @property
    def im(self) -> np.ndarray:
        """Im(Z) (<= 0 for RC-type systems)."""
        return self.Z.imag

    @property
    def neg_im(self) -> np.ndarray:
        """-Im(Z) (>= 0; what would go on the Nyquist y-axis)."""
        return -self.Z.imag

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.Z)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.degrees(np.angle(self.Z))
