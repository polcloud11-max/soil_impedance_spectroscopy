"""
Cole-Cole / ZARC equivalent-circuit fit.

Model
-----
.. math::  Z(\\omega) = R_s + \\frac{R}{1 + (j\\omega\\tau)^{\\alpha}}

* ``Rs``    series (high-frequency) resistance              [ohm]
* ``R``     polarization resistance / Nyquist arc diameter  [ohm]
* ``tau``   characteristic relaxation time                  [s]
* ``alpha`` Cole-Cole depression exponent (1 = ideal Debye, < 1 = depressed)

Solver: ``scipy.optimize.least_squares`` (Trust Region Reflective) with
multi-start initial guesses and residuals weighted by ``1/|Z|`` so the fit
is not dominated by a few high-magnitude points across decades of |Z|.

Failures (non-convergence, solver exceptions) are reported in the returned
:class:`ZarcFit` object via ``success=False``; they are never silently
swallowed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.optimize import least_squares

from .preprocessing import Spectrum
from .utils import get_logger
from config import ZARC_BOUNDS

log = get_logger()


# ---------------------------------------------------------------------
# Forward model
# ---------------------------------------------------------------------
def zarc_impedance(
    omega: np.ndarray,
    Rs: float,
    R: float,
    tau: float,
    alpha: float,
) -> np.ndarray:
    """Return complex Z(omega) for the ZARC element."""
    j_omega_tau = 1j * np.asarray(omega, dtype=np.float64) * tau
    return Rs + R / (1.0 + j_omega_tau ** alpha)


# ---------------------------------------------------------------------
# Residuals & initial-guess generator
# ---------------------------------------------------------------------
def _residuals(
    theta: np.ndarray,
    omega: np.ndarray,
    Z_meas: np.ndarray,
) -> np.ndarray:
    Rs, R, tau, alpha = theta
    Z_mod = zarc_impedance(omega, Rs, R, tau, alpha)
    w = 1.0 / np.maximum(np.abs(Z_meas), 1e-12)
    return np.concatenate([
        (Z_mod.real - Z_meas.real) * w,
        (Z_mod.imag - Z_meas.imag) * w,
    ])


def _initial_guesses(spectrum: Spectrum) -> List[List[float]]:
    """Generate multiple starting points to improve robustness."""
    re = spectrum.re
    neg_im = spectrum.neg_im
    f = spectrum.f

    Rs0 = float(max(re.min(), 1e-6))
    R0 = float(max(re.max() - re.min(), 1e-3))

    idx_peak = int(np.argmax(neg_im))
    f_peak = float(f[idx_peak]) if f[idx_peak] > 0 else float(f[len(f) // 2])
    tau0 = 1.0 / (2.0 * np.pi * f_peak)

    starts: List[List[float]] = []
    for alpha_guess in (0.95, 0.85, 0.70):
        for tau_mult in (1.0, 0.1, 10.0):
            starts.append([Rs0, R0, tau0 * tau_mult, alpha_guess])
    return starts


def _bounds_arrays() -> Tuple[np.ndarray, np.ndarray]:
    lo = np.asarray([
        ZARC_BOUNDS["Rs"][0], ZARC_BOUNDS["R"][0],
        ZARC_BOUNDS["tau"][0], ZARC_BOUNDS["alpha"][0],
    ], dtype=np.float64)
    hi = np.asarray([
        ZARC_BOUNDS["Rs"][1], ZARC_BOUNDS["R"][1],
        ZARC_BOUNDS["tau"][1], ZARC_BOUNDS["alpha"][1],
    ], dtype=np.float64)
    return lo, hi


# ---------------------------------------------------------------------
# Fit result container
# ---------------------------------------------------------------------
@dataclass
class ZarcFit:
    """Outcome of a single ZARC fit. Always returned (success or failure)."""
    Rs: float
    R: float
    tau: float
    alpha: float
    rmse_real: float
    rmse_imag: float
    rmse_rel: float
    n_points: int
    success: bool
    message: str


# ---------------------------------------------------------------------
# Public driver
# ---------------------------------------------------------------------
def fit_spectrum(spectrum: Spectrum) -> ZarcFit:
    """Fit the ZARC model to a single spectrum.

    Returns a :class:`ZarcFit`. On failure the parameter fields are NaN,
    ``success=False`` and ``message`` carries the reason.
    """
    omega = spectrum.omega
    Z_meas = spectrum.Z

    lo, hi = _bounds_arrays()
    best_result = None
    best_cost = np.inf
    last_message = ""

    for x0 in _initial_guesses(spectrum):
        x0_clipped = np.clip(x0, lo + 1e-12, hi - 1e-12)
        try:
            result = least_squares(
                _residuals,
                x0_clipped,
                args=(omega, Z_meas),
                bounds=(lo, hi),
                method="trf",
                max_nfev=4000,
                xtol=1e-10,
                ftol=1e-10,
            )
        except Exception as exc:           # solver blew up; record and try next
            last_message = f"solver error: {exc}"
            continue

        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result
            last_message = result.message

    if best_result is None:
        log.warning(f"ZARC fit failed: {last_message}")
        return ZarcFit(
            Rs=float("nan"), R=float("nan"),
            tau=float("nan"), alpha=float("nan"),
            rmse_real=float("nan"), rmse_imag=float("nan"),
            rmse_rel=float("nan"),
            n_points=spectrum.n_points,
            success=False,
            message=last_message or "no solver returned a solution",
        )

    Rs, R, tau, alpha = best_result.x
    Z_mod = zarc_impedance(omega, Rs, R, tau, alpha)
    rmse_real = float(np.sqrt(np.mean((Z_mod.real - Z_meas.real) ** 2)))
    rmse_imag = float(np.sqrt(np.mean((Z_mod.imag - Z_meas.imag) ** 2)))
    rmse_rel = float(np.sqrt(2.0 * best_result.cost / len(best_result.fun)))

    return ZarcFit(
        Rs=float(Rs),
        R=float(R),
        tau=float(tau),
        alpha=float(alpha),
        rmse_real=rmse_real,
        rmse_imag=rmse_imag,
        rmse_rel=rmse_rel,
        n_points=spectrum.n_points,
        success=bool(best_result.success),
        message=str(best_result.message),
    )
