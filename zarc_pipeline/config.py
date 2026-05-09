"""
Onboard configuration for the rover impedance-spectroscopy pipeline.

Only fitting bounds and sweep-validation thresholds live here. All static
demonstration paths (CSV files, output directories, plot directories) have
been removed for the embedded build.
"""
from typing import Dict, Tuple

# ---------- Sweep-validation thresholds ------------------------------
# Minimum number of valid (frequency, Z) points required to attempt a fit.
MIN_POINTS_PER_SPECTRUM: int = 20

# ---------- ZARC fitting bounds --------------------------------------
# (lower, upper) bounds for the parameter set (Rs, R, tau, alpha) used by
# scipy.optimize.least_squares.  These are intentionally wide; tighten them
# if the rover's probe geometry restricts the realistic range.
ZARC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Rs":    (0.0,   1e9),
    "R":     (1e-3,  1e10),
    "tau":   (1e-12, 1e3),
    "alpha": (0.3,   1.0),
}
