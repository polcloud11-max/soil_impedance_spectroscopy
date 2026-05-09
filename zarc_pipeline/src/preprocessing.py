"""
Group EIS rows into individual spectra and build the complex-impedance array.

Sign convention
---------------
The CSV stores Img(Z) as POSITIVE values that grow at low frequency. For
RC-like systems the standard physics convention is Im(Z) <= 0, so we treat
the stored value as |Im(Z)|. Internally we keep Z = Re + j*Im with Im <= 0,
and Nyquist plots show -Im(Z) on the y-axis (positive, upward arcs).
"""
from typing import Dict, List

import numpy as np
import pandas as pd

from .utils import get_logger, spectrum_key
from config import EIS_IM_STORED_AS_POSITIVE, MIN_POINTS_PER_SPECTRUM

log = get_logger()


class Spectrum:
    """Container for a single EIS spectrum sorted by descending frequency."""

    __slots__ = ("key", "ionic_radius", "temperature",
                 "f", "omega", "Z", "n_points")

    def __init__(self, ionic_radius: float, temperature: float,
                 f: np.ndarray, Z: np.ndarray):
        self.key = spectrum_key(ionic_radius, temperature)
        self.ionic_radius = float(ionic_radius)
        self.temperature = float(temperature)
        # Sort high -> low frequency (typical EIS convention)
        order = np.argsort(-f)
        self.f = f[order]
        self.omega = 2 * np.pi * self.f
        self.Z = Z[order]
        self.n_points = int(self.f.size)

    @property
    def re(self):
        return self.Z.real

    @property
    def im(self):
        return self.Z.imag       # negative for capacitive systems

    @property
    def neg_im(self):
        return -self.Z.imag      # what goes on the Nyquist y-axis

    @property
    def magnitude(self):
        return np.abs(self.Z)

    @property
    def phase_deg(self):
        return np.degrees(np.angle(self.Z))


def build_spectra(df: pd.DataFrame) -> List[Spectrum]:
    """Group EIS dataframe by (ionic_radius, temperature) into Spectrum objs."""
    spectra: List[Spectrum] = []
    skipped = 0

    for (rad, T), grp in df.groupby(["ionic_radius", "temperature"], sort=True):
        f = grp["frequency"].to_numpy(dtype=float)
        re = grp["re_z"].to_numpy(dtype=float)
        im_raw = grp["im_z"].to_numpy(dtype=float)

        # Apply sign convention: file stores positive |Im(Z)| -> store as -|Im|
        im = -np.abs(im_raw) if EIS_IM_STORED_AS_POSITIVE else im_raw

        Z = re + 1j * im

        # Drop any non-finite or non-positive frequencies
        ok = np.isfinite(f) & np.isfinite(Z) & (f > 0)
        f, Z = f[ok], Z[ok]

        if f.size < MIN_POINTS_PER_SPECTRUM:
            skipped += 1
            log.warning(f"Skipping {spectrum_key(rad, T)}: only {f.size} points")
            continue

        spectra.append(Spectrum(rad, T, f, Z))

    log.info(f"Built {len(spectra)} spectra (skipped {skipped})")
    return spectra


def spectra_summary(spectra: List[Spectrum]) -> pd.DataFrame:
    rows = []
    for s in spectra:
        rows.append({
            "key": s.key,
            "ionic_radius": s.ionic_radius,
            "temperature": s.temperature,
            "n_points": s.n_points,
            "f_min_hz": float(s.f.min()),
            "f_max_hz": float(s.f.max()),
            "Z_min_ohm": float(np.abs(s.Z).min()),
            "Z_max_ohm": float(np.abs(s.Z).max()),
        })
    return pd.DataFrame(rows)
