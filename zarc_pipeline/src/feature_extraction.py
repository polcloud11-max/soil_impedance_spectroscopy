"""
Build a feature table per spectrum that combines:

* ZARC-fit parameters (Rs, R, tau, alpha) and derived quantities
* Spectral descriptors computed directly from the data (frequency-agnostic
  summaries that survive even if a ZARC fit fails)

These features compress an EIS curve into a fixed-size vector usable for
clustering or classification.
"""
from typing import List

import numpy as np
import pandas as pd

from .preprocessing import Spectrum
from .utils import get_logger, safe_log10

log = get_logger()


# Friendly description of every feature -> printed in the report
FEATURE_DESCRIPTIONS = {
    "Rs": "Series resistance (Ohm) — high-frequency intercept",
    "R": "Polarization resistance (Ohm) — Nyquist arc diameter",
    "tau": "Relaxation time (s) — characteristic time-constant",
    "alpha": "Cole-Cole depression exponent (1=ideal, <1=depressed)",
    "f_char_hz": "Characteristic frequency 1/(2*pi*tau)",
    "log10_R_over_Rs": "log10(R/Rs) — relative size of polarization arc",
    "Z_min_mag": "min |Z| over the spectrum",
    "Z_max_mag": "max |Z| over the spectrum",
    "log10_Z_range": "log10(max|Z| / min|Z|) — dynamic range of |Z|",
    "phase_min_deg": "Most negative measured phase",
    "phase_max_deg": "Highest (least negative) measured phase",
    "phase_range_deg": "phase_max - phase_min",
    "f_at_phase_min_hz": "Frequency at which phase is most negative",
    "neg_im_peak": "Maximum -Im(Z) (peak height of Nyquist arc)",
    "f_at_neg_im_peak_hz": "Frequency at the Nyquist arc apex",
    "re_at_neg_im_peak": "Re(Z) at the Nyquist arc apex (~ Rs + R/2)",
}


def _spectral_descriptors(s: Spectrum) -> dict:
    mag = np.abs(s.Z)
    phase = s.phase_deg
    neg_im = s.neg_im

    idx_phase_min = int(np.argmin(phase))
    idx_peak = int(np.argmax(neg_im))

    return {
        "Z_min_mag": float(mag.min()),
        "Z_max_mag": float(mag.max()),
        "log10_Z_range": float(safe_log10(mag.max() / max(mag.min(), 1e-12))),
        "phase_min_deg": float(phase.min()),
        "phase_max_deg": float(phase.max()),
        "phase_range_deg": float(phase.max() - phase.min()),
        "f_at_phase_min_hz": float(s.f[idx_phase_min]),
        "neg_im_peak": float(neg_im.max()),
        "f_at_neg_im_peak_hz": float(s.f[idx_peak]),
        "re_at_neg_im_peak": float(s.re[idx_peak]),
    }


def build_feature_table(spectra: List[Spectrum],
                        fits_df: pd.DataFrame) -> pd.DataFrame:
    fits_by_key = fits_df.set_index("key")
    rows = []

    for s in spectra:
        f = fits_by_key.loc[s.key] if s.key in fits_by_key.index else None
        d = {
            "key": s.key,
            "ionic_radius": s.ionic_radius,
            "temperature": s.temperature,
            "n_points": s.n_points,
        }

        if f is not None and bool(f["success"]) and np.isfinite(f["tau"]):
            d.update({
                "Rs": float(f["Rs"]),
                "R": float(f["R"]),
                "tau": float(f["tau"]),
                "alpha": float(f["alpha"]),
                "f_char_hz": float(1.0 / (2 * np.pi * f["tau"])),
                "log10_R_over_Rs": float(safe_log10(f["R"] / max(f["Rs"], 1e-12))),
                "fit_rmse_rel": float(f["rmse_rel"]),
                "fit_ok": True,
            })
        else:
            d.update({
                "Rs": np.nan, "R": np.nan, "tau": np.nan, "alpha": np.nan,
                "f_char_hz": np.nan, "log10_R_over_Rs": np.nan,
                "fit_rmse_rel": np.nan, "fit_ok": False,
            })

        d.update(_spectral_descriptors(s))
        rows.append(d)

    df = pd.DataFrame(rows)
    log.info(f"Feature table: {df.shape[0]} rows x {df.shape[1]} cols")
    return df
