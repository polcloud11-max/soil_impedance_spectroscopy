"""
Cole-Cole / ZARC equivalent-circuit fitting.

Model
-----
    Z(omega) = Rs + R / (1 + (j * omega * tau)^alpha)

* Rs    : series (high-frequency) resistance
* R     : polarization resistance (arc diameter)
* tau   : characteristic relaxation time -> peak frequency f_c = 1 / (2*pi*tau)
* alpha : depression exponent (1 -> ideal Debye semicircle, <1 -> depressed)

We fit Re(Z) and -Im(Z) jointly via nonlinear least squares with sensible
bounds and multi-start initial guesses. Failures are reported, NOT hidden.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .preprocessing import Spectrum
from .utils import get_logger
from .visualization import plot_fit_comparison
from config import ZARC_BOUNDS, FIG_DIR

log = get_logger()


# ------------------------- Model ----------------------------------
def zarc_impedance(omega: np.ndarray,
                   Rs: float, R: float, tau: float, alpha: float) -> np.ndarray:
    """Return complex Z(omega) for the ZARC element."""
    j_omega_tau = 1j * omega * tau
    # Use complex power explicitly to avoid numpy real-power issues
    return Rs + R / (1.0 + j_omega_tau ** alpha)


def _residuals(theta, omega, Z_meas):
    Rs, R, tau, alpha = theta
    Z_mod = zarc_impedance(omega, Rs, R, tau, alpha)
    # Stack real and imag residuals; weight by 1/|Z| so different scales
    # don't bias the fit.
    w = 1.0 / np.maximum(np.abs(Z_meas), 1e-12)
    return np.concatenate([
        (Z_mod.real - Z_meas.real) * w,
        (Z_mod.imag - Z_meas.imag) * w,
    ])


# ------------------------- Init guess -----------------------------
def _initial_guess(spectrum: Spectrum) -> List[List[float]]:
    """Generate a few starting points to improve convergence robustness."""
    re = spectrum.re
    neg_im = spectrum.neg_im
    f = spectrum.f

    Rs0 = max(re.min(), 1e-6)
    R0 = max(re.max() - re.min(), 1e-3)

    # tau ~ 1 / (2*pi*f_peak) where f_peak = freq of max -Im(Z)
    idx_peak = int(np.argmax(neg_im))
    f_peak = f[idx_peak] if f[idx_peak] > 0 else f[len(f) // 2]
    tau0 = 1.0 / (2 * np.pi * f_peak)

    # Multi-start: vary alpha and tau decade
    starts = []
    for a in (0.95, 0.85, 0.7):
        for tau_mult in (1.0, 0.1, 10.0):
            starts.append([Rs0, R0, tau0 * tau_mult, a])
    return starts


# ------------------------- Fit driver -----------------------------
@dataclass
class ZarcFit:
    key: str
    ionic_radius: float
    temperature: float
    Rs: float
    R: float
    tau: float
    alpha: float
    rmse_real: float
    rmse_imag: float
    rmse_rel: float           # weighted RMSE on |Z|-normalised residuals
    n_points: int
    success: bool
    message: str

    def as_row(self):
        return asdict(self)


def _bounds_arrays():
    lo = [ZARC_BOUNDS["Rs"][0], ZARC_BOUNDS["R"][0],
          ZARC_BOUNDS["tau"][0], ZARC_BOUNDS["alpha"][0]]
    hi = [ZARC_BOUNDS["Rs"][1], ZARC_BOUNDS["R"][1],
          ZARC_BOUNDS["tau"][1], ZARC_BOUNDS["alpha"][1]]
    return np.asarray(lo), np.asarray(hi)


def fit_spectrum(spectrum: Spectrum,
                 plot: bool = False,
                 plot_dir: Optional[Path] = None) -> ZarcFit:
    omega = spectrum.omega
    Z_meas = spectrum.Z

    lo, hi = _bounds_arrays()
    best = None
    best_cost = np.inf
    last_msg = ""

    for x0 in _initial_guess(spectrum):
        x0 = np.clip(x0, lo + 1e-12, hi - 1e-12)
        try:
            res = least_squares(
                _residuals, x0,
                args=(omega, Z_meas),
                bounds=(lo, hi),
                method="trf",
                max_nfev=4000,
                xtol=1e-10, ftol=1e-10,
            )
        except Exception as exc:                       # solver blew up
            last_msg = f"solver error: {exc}"
            continue

        if res.cost < best_cost:
            best_cost = res.cost
            best = res
            last_msg = res.message

    if best is None:
        log.warning(f"{spectrum.key}: all starts failed ({last_msg})")
        return ZarcFit(
            key=spectrum.key, ionic_radius=spectrum.ionic_radius,
            temperature=spectrum.temperature,
            Rs=np.nan, R=np.nan, tau=np.nan, alpha=np.nan,
            rmse_real=np.nan, rmse_imag=np.nan, rmse_rel=np.nan,
            n_points=spectrum.n_points,
            success=False, message=last_msg or "no solution",
        )

    Rs, R, tau, alpha = best.x
    Z_mod = zarc_impedance(omega, Rs, R, tau, alpha)
    rmse_real = float(np.sqrt(np.mean((Z_mod.real - Z_meas.real) ** 2)))
    rmse_imag = float(np.sqrt(np.mean((Z_mod.imag - Z_meas.imag) ** 2)))
    rmse_rel = float(np.sqrt(2 * best.cost / len(best.fun)))

    if plot and plot_dir is not None:
        plot_fit_comparison(
            spectrum, Z_mod,
            {"Rs": Rs, "R": R, "tau": tau, "alpha": alpha},
            plot_dir / f"fit_{spectrum.key.replace('=', '').replace('.', 'p')}.png",
        )

    return ZarcFit(
        key=spectrum.key, ionic_radius=spectrum.ionic_radius,
        temperature=spectrum.temperature,
        Rs=float(Rs), R=float(R), tau=float(tau), alpha=float(alpha),
        rmse_real=rmse_real, rmse_imag=rmse_imag, rmse_rel=rmse_rel,
        n_points=spectrum.n_points,
        success=bool(best.success),
        message=str(best.message),
    )


def fit_all(spectra: List[Spectrum],
            plot_each: bool = False) -> pd.DataFrame:
    fits = []
    plot_dir = FIG_DIR / "fits"
    if plot_each:
        plot_dir.mkdir(exist_ok=True)
    for s in spectra:
        fit = fit_spectrum(s, plot=plot_each, plot_dir=plot_dir)
        fits.append(fit.as_row())

    df = pd.DataFrame(fits)
    n_ok = int(df["success"].sum())
    log.info(f"ZARC fit: {n_ok}/{len(df)} converged")
    return df
