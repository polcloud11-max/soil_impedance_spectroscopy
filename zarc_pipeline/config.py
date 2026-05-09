"""
Project-wide configuration: paths, plotting settings, fitting parameters.
Edit values here rather than scattering magic numbers across modules.
"""
from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parent                       # CSVs live one level up
EIS_CSV = DATA_DIR / "Dataset.csv"
SOIL_CSV = DATA_DIR / "soil_impedance_all.csv"

OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
FEAT_DIR = OUT_DIR / "features"
REP_DIR = OUT_DIR / "reports"

for _d in (FIG_DIR, FEAT_DIR, REP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------- EIS dataset conventions ----------
# Column names in Dataset.csv (after cleaning).
EIS_COLS = {
    "ionic_radius": "ionic_radius",
    "temperature": "temperature",
    "frequency": "frequency",
    "re_z": "re_z",
    "im_z": "im_z",
}

# Sign convention for the imaginary part of Z stored in the file.
# Inspecting Dataset.csv shows positive Img(Z) values that grow at low
# frequency — meaning the file stores |Im(Z)| (positive). Standard physics
# convention has Im(Z) <= 0 for capacitive/RC behaviour, and Nyquist plots
# put -Im(Z) on the y-axis. We therefore treat the stored value as -Im(Z).
EIS_IM_STORED_AS_POSITIVE = True

# Minimum number of points needed to attempt a ZARC fit.
MIN_POINTS_PER_SPECTRUM = 20

# ---------- ZARC fitting ----------
# Bounds and initial guesses (lower, upper) for the parameter set
# (Rs, R, tau, alpha) used in zarc_fitting.py.
ZARC_BOUNDS = {
    "Rs":   (0.0,   1e9),
    "R":    (1e-3,  1e10),
    "tau":  (1e-12, 1e3),
    "alpha":(0.3,   1.0),
}

# ---------- Plot styling ----------
FIG_DPI = 130
SAMPLE_SPECTRA_TO_PLOT = 6   # number of representative spectra to overlay
