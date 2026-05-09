"""
Nyquist and Bode plots for EIS spectra. Saves PNGs into outputs/figures.
"""
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from .preprocessing import Spectrum
from .utils import get_logger
from config import FIG_DIR, FIG_DPI, SAMPLE_SPECTRA_TO_PLOT

log = get_logger()


# ------------------------------------------------------------------
# Nyquist
# ------------------------------------------------------------------
def plot_nyquist(spectra: List[Spectrum],
                 out_path: Optional[Path] = None,
                 max_n: int = SAMPLE_SPECTRA_TO_PLOT,
                 title: str = "Nyquist plot (Re Z vs -Im Z)") -> Path:
    """
    Standard Nyquist convention: x = Re(Z), y = -Im(Z). For RC-type systems
    arcs appear as positive humps. Each curve = one (ionic_radius, T) sample.
    """
    pick = _pick_representative(spectra, max_n)
    fig, ax = plt.subplots(figsize=(7, 6))
    for s in pick:
        ax.plot(s.re, s.neg_im, "-o", ms=3, lw=1, label=s.key)
    ax.set_xlabel(r"Re$(Z)$ [$\Omega$]")
    ax.set_ylabel(r"$-$Im$(Z)$ [$\Omega$]")
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    out_path = out_path or (FIG_DIR / "nyquist_overview.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    log.info(f"Saved {out_path}")
    return out_path


# ------------------------------------------------------------------
# Bode
# ------------------------------------------------------------------
def plot_bode(spectra: List[Spectrum],
              out_path: Optional[Path] = None,
              max_n: int = SAMPLE_SPECTRA_TO_PLOT,
              title: str = "Bode plot |Z|, phase vs frequency") -> Path:
    pick = _pick_representative(spectra, max_n)

    fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)
    ax_mag, ax_phase = axes
    for s in pick:
        ax_mag.loglog(s.f, np.abs(s.Z), "-o", ms=3, lw=1, label=s.key)
        ax_phase.semilogx(s.f, s.phase_deg, "-o", ms=3, lw=1, label=s.key)

    ax_mag.set_ylabel(r"$|Z|$ [$\Omega$]")
    ax_mag.set_title(title)
    ax_mag.grid(True, which="both", alpha=0.3)
    ax_mag.legend(fontsize=7, loc="best")

    ax_phase.set_xlabel("Frequency [Hz]")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.grid(True, which="both", alpha=0.3)

    out_path = out_path or (FIG_DIR / "bode_overview.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    log.info(f"Saved {out_path}")
    return out_path


# ------------------------------------------------------------------
# Per-spectrum fit comparison (called from the fitting stage)
# ------------------------------------------------------------------
def plot_fit_comparison(spectrum: Spectrum,
                        Z_model: np.ndarray,
                        params: dict,
                        out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_n, ax_b = axes

    ax_n.plot(spectrum.re, spectrum.neg_im, "o", ms=4,
              label="measured", color="C0")
    ax_n.plot(Z_model.real, -Z_model.imag, "-", lw=1.5,
              label="ZARC fit", color="C3")
    ax_n.set_xlabel(r"Re$(Z)$ [$\Omega$]")
    ax_n.set_ylabel(r"$-$Im$(Z)$ [$\Omega$]")
    ax_n.set_title(f"Nyquist | {spectrum.key}")
    ax_n.set_aspect("equal", adjustable="datalim")
    ax_n.grid(True, alpha=0.3)
    ax_n.legend(fontsize=8)

    ax_b.loglog(spectrum.f, np.abs(spectrum.Z), "o", ms=4,
                label="|Z| measured", color="C0")
    ax_b.loglog(spectrum.f, np.abs(Z_model), "-",
                label="|Z| fit", color="C3")
    ax_b.set_xlabel("Frequency [Hz]")
    ax_b.set_ylabel(r"$|Z|$ [$\Omega$]")
    ax_b.set_title(
        f"Bode |Z| | Rs={params.get('Rs', np.nan):.3g}, "
        f"R={params.get('R', np.nan):.3g}, "
        f"tau={params.get('tau', np.nan):.3g}, "
        f"alpha={params.get('alpha', np.nan):.3f}"
    )
    ax_b.grid(True, which="both", alpha=0.3)
    ax_b.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    return out_path


# ------------------------------------------------------------------
def _pick_representative(spectra: List[Spectrum], n: int) -> List[Spectrum]:
    """Evenly sample n spectra so plots remain readable."""
    if len(spectra) <= n:
        return spectra
    idx = np.linspace(0, len(spectra) - 1, n, dtype=int)
    return [spectra[i] for i in idx]
