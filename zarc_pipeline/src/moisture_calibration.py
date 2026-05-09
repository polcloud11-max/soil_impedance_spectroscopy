"""
Empirical conversion of ZARC equivalent-circuit parameters into estimated
Volumetric Water Content (VWC) and bulk capacitance.

Physical principle
------------------
For mineral/regolith soils, both the series resistance ``Rs`` and the
polarization resistance ``R`` decrease as moisture rises (water is far
more conductive than dry soil grains or pore air). A power-law of the DC
limit ``R_total = Rs + R`` therefore approximates VWC well over a useful
working range; the constants below MUST be tuned via probe-in-the-loop
calibration in soil samples of known gravimetric water content.

Bulk capacitance is recovered directly from the time constant
``tau = R * C_bulk`` so ``C_bulk = tau / R``. Reported in farads.

The calibration constants are intentionally exposed as module-level
attributes (not hard-coded in the function) so that flight software can
overwrite them at boot from a calibration file without touching code.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# ---------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------
# Power-law parameters mapping R_total (ohm) -> VWC (%).
# Default values are PLACEHOLDERS taken from the project specification
# and MUST be replaced after probe characterisation in lab soil samples.
CALIBRATION_A: float = 1500.0
CALIBRATION_B: float = -0.45
CALIBRATION_OFFSET: float = 2.0


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def calculate_vwc(Rs: float, R: float, tau: float, alpha: float) -> Dict[str, float]:
    """Map ZARC parameters to estimated VWC and bulk capacitance.

    Parameters
    ----------
    Rs : float
        Series resistance [ohm].
    R : float
        Polarization resistance / Nyquist arc diameter [ohm].
    tau : float
        Characteristic relaxation time [s].
    alpha : float
        Cole-Cole depression exponent (kept in the signature for API
        completeness; not used directly by this calibration model, but
        downstream models may consume it).

    Returns
    -------
    dict
        ``vwc_percent`` (clamped 0-100), ``bulk_capacitance_farads``,
        ``r_total_ohms``.

    Notes
    -----
    All inputs are validated; non-finite values produce NaN outputs and a
    VWC of 0% (the rover should treat NaN ``r_total`` as "fit failed,
    do not act on this reading").
    """
    # Defensive numeric checks ------------------------------------------------
    Rs_f = float(Rs)
    R_f = float(R)
    tau_f = float(tau)
    _ = float(alpha)  # validated, not used in this empirical model

    # Bulk capacitance from tau = R * C  =>  C = tau / R
    if not np.isfinite(R_f) or R_f <= 0.0 or not np.isfinite(tau_f):
        bulk_capacitance: float = float("nan")
    else:
        bulk_capacitance = tau_f / max(R_f, 1e-6)

    # DC limit (real part of Z as omega -> 0)
    if not (np.isfinite(Rs_f) and np.isfinite(R_f)):
        r_total: float = float("nan")
    else:
        r_total = Rs_f + R_f

    # VWC via empirical power law on R_total
    if not np.isfinite(r_total) or r_total <= 0.0:
        vwc_clamped: float = 0.0
    else:
        estimated_vwc_percent = (
            CALIBRATION_A * (r_total ** CALIBRATION_B) + CALIBRATION_OFFSET
        )
        vwc_clamped = float(np.clip(estimated_vwc_percent, 0.0, 100.0))

    return {
        "vwc_percent": vwc_clamped,
        "bulk_capacitance_farads": float(bulk_capacitance),
        "r_total_ohms": float(r_total),
    }
