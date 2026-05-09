# Impedance Spectroscopy Pipeline

Physics-informed pipeline for processing electrical impedance spectroscopy
(EIS) data, fitting the Cole-Cole / ZARC equivalent circuit, extracting
physical features, and running a defensible classification workflow.

Built around two CSVs in the parent folder:

| File | Role | Frequency-resolved? | ZARC fitting? |
|---|---|---|---|
| `Dataset.csv` | EIS core (28 spectra, 2 ionic radii × 9–19 temperatures) | yes | yes |
| `soil_impedance_all.csv` | soil-sensing context (1944 scalar measurements) | **no** | **no** — would be scientifically invalid |

The two datasets are **not row-merged**. They are complementary modules: the
EIS module performs the spectroscopy work; the soil module performs honest
exploratory analysis only.

---

## Folder layout

```
zarc_pipeline/
├── README.md
├── requirements.txt
├── config.py             # paths, sign convention, fit bounds
├── main.py               # end-to-end driver
├── src/
│   ├── data_loading.py       # cleaned CSV loaders + inspectors
│   ├── preprocessing.py      # group rows -> Spectrum objects
│   ├── visualization.py      # Nyquist + Bode + per-fit plots
│   ├── zarc_fitting.py       # Cole-Cole / ZARC nonlinear least-squares fit
│   ├── feature_extraction.py # ZARC params + spectral descriptors
│   ├── soil_analysis.py      # EDA + K-Means on soil dataset
│   ├── classification.py     # RF / LogReg / SVM with grouped CV
│   └── utils.py
└── outputs/
    ├── figures/              # PNG plots (Nyquist, Bode, per-spectrum fits, soil)
    ├── features/             # CSVs (ZARC params, features, classification, soil clusters)
    └── reports/              # JSON pipeline summary
```

## How to run

```bash
pip install -r requirements.txt
python main.py
```

Both CSVs must sit one directory above `zarc_pipeline/` (paths set in
`config.py`).

---

## Pipeline overview

```
Dataset.csv                              soil_impedance_all.csv
    │                                          │
    ▼                                          ▼
  load + clean                             load + clean
    │                                          │
  group by (ionic_radius, T)              EDA: histograms, scatter, boxplot
    │                                          │
  Nyquist + Bode plots                     K-Means soil-state clustering
    │                                          │
  ZARC nonlinear LS fit                    save outputs
    │
  spectral descriptors
    │
  feature table
    │
  classification (LeaveOneTempOut CV)
    │
  JSON report
```

### Sign convention (important)

`Dataset.csv` stores `Img(Z)` as positive values that grow at low frequency.
For RC-type systems standard physics has Im(Z) ≤ 0, so the file value is
treated as `|Im(Z)|`. Internally `Z = Re + j*Im` with `Im ≤ 0`, and Nyquist
plots show `-Im(Z)` on the y-axis (positive arcs).

### ZARC model

```
Z(ω) = Rs + R / (1 + (jωτ)^α)
```

Parameters fit per spectrum:

* `Rs` — series / high-frequency intercept (Ω)
* `R`  — polarization resistance / arc diameter (Ω)
* `τ`  — relaxation time (s) → characteristic frequency `f_c = 1 / (2π τ)`
* `α`  — Cole-Cole depression exponent (1 = ideal Debye, < 1 = depressed)

Fit details:

* `scipy.optimize.least_squares` (TRF), bounds from `config.ZARC_BOUNDS`.
* Multi-start initial guesses (3 α values × 3 τ decades) — best cost wins.
* Residuals weighted by `1/|Z|` so high- and low-frequency points contribute
  comparably across decades of impedance.
* Failures are reported, never hidden — `success` and `message` columns in
  `outputs/features/zarc_fits.csv`.

### Feature table

Per spectrum, both fit-derived and model-free descriptors are stored. See
`feature_descriptions` in the JSON report. Highlights:

| Feature | Meaning |
|---|---|
| `Rs`, `R`, `tau`, `alpha` | ZARC parameters |
| `f_char_hz` | `1/(2πτ)`, where the Nyquist arc peaks |
| `log10_R_over_Rs` | how dominant the polarization arc is |
| `Z_min_mag`, `Z_max_mag`, `log10_Z_range` | magnitude span |
| `phase_min_deg`, `phase_max_deg`, `phase_range_deg` | phase behaviour |
| `neg_im_peak`, `f_at_neg_im_peak_hz`, `re_at_neg_im_peak` | apex of the Nyquist arc |

### Classification (defensible)

`ionic_radius` is a real, factory-stamped grouping in `Dataset.csv` (two
distinct values), not a fabricated label. The task is: from the EIS
features, recover which species was measured.

* Models: Random Forest, Logistic Regression (scaled), SVM-RBF (scaled).
* Cross-validation: **Leave-One-Temperature-Out** — each fold holds out
  one temperature so test points are unseen operating conditions, not just
  shuffled rows. With 19 temperatures total this gives 19 folds.
* Metrics: accuracy, macro precision / recall / F1.

**Honesty note:** with only 28 spectra this is a small-data demonstration.
Reported metrics indicate the *separability of the feature space*, not a
production benchmark.

### Soil dataset handling

`soil_impedance_all.csv` has only `TOTAL_IMPEDANCE` per observation — no
frequency sweep. ZARC fitting would be scientifically invalid here, so the
module does:

1. EDA: distributions, impedance vs temperature, impedance vs probe depth.
2. Unsupervised K-Means clustering on `(impedance, temperature, depth)` for
   exploratory soil-state grouping. **This is not a calibrated moisture
   model** — the dataset has no moisture labels, and supervised learning
   would require fabricating them.

If real moisture or class labels are added later (gravimetric water
content, soil-type code, etc.), the same `classification.py` plumbing
applies — only the label column changes.

---

## Connection to the research paper

Umar & Setiadi (AIP Conf. Proc. 1656, 040005, 2015) demonstrate that soil
electrical impedance varies systematically with moisture across frequency
(10 kHz–10 MHz), and that frequency-domain analysis carries more
information than a single-frequency conductivity reading. Their eq. (19)
is itself a small linear correction model: `Z(f, θw) = (a₀₀ + a₀₁·f) + B·θw`.

This pipeline generalises that idea:

* **Impedance spectroscopy is useful for soil sensing** because dry soil is
  capacitive (air-dominated, low conductivity) while wet soil is resistive
  (water-dominated). Different frequencies probe different physical
  processes (electrode polarization, double-layer, bulk conduction), so a
  full spectrum carries far more information than one impedance number.
* **Frequency-domain analysis matters** because the same gravimetric
  moisture can give different scalar impedances depending on temperature,
  texture, salinity. The shape of the spectrum (arc depression, peak
  frequency, Rs/R ratio) disentangles these.
* **Cole-Cole / ZARC fitting compresses each spectrum into 4 physically
  meaningful parameters** — exactly the kind of low-dimensional,
  interpretable input a downstream classifier or rover-side ML model
  needs to operate under tight bandwidth and compute budgets.
* **For a Mars rover application**: the rover would run the same
  preprocessing → ZARC fit → feature extraction onboard, transmit only
  the small feature vector home, and use models trained on labeled
  Earth-soil/regolith analogues for in-situ inference. The honest gap
  here is that this requires *labeled* analogue data — which the public
  `soil_impedance_all.csv` does not provide.

---

## Outputs produced

```
outputs/figures/
    nyquist_overview.png             # Re vs -Im, several spectra overlaid
    bode_overview.png                # |Z| and phase vs frequency
    soil_overview.png                # impedance distributions + scatter/boxplot
    soil_clusters.png                # K-Means soil-state clusters
    fits/
        fit_r1p820e-10_T-10.png      # measured vs ZARC-modelled per spectrum
        ... (28 files)

outputs/features/
    spectra_summary.csv              # per-spectrum metadata
    zarc_fits.csv                    # ZARC parameters, RMSE, success flag
    features.csv                     # full feature table for classification
    classification_metrics.csv       # accuracy / precision / recall / F1
    soil_clusters.csv                # soil rows + K-Means cluster id

outputs/reports/
    pipeline_report.json             # everything in one machine-readable file
```

---

## Limitations (read before drawing conclusions)

* `Dataset.csv` is not soil data. It looks like ionic / solid-state
  electrolyte impedance (cation radii 1.82 Å, 2.06 Å — Na⁺, K⁺ scale).
  The EIS *methodology* is identical and that is the part this pipeline
  exercises; do not reinterpret these spectra as soil moisture.
* `soil_impedance_all.csv` has scalar impedance only. Treating it as EIS
  is not possible.
* 28 spectra is small. Classification metrics are demonstrative.
* No frequency-error model is propagated through the fits. Fit RMSE is
  reported, but full uncertainty intervals on `(Rs, R, τ, α)` are not.
