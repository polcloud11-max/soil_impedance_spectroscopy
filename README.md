# Rover Onboard Impedance-Spectroscopy Pipeline

Standalone sensory processing unit for an autonomous planetary rover. One
frequency sweep in (JSON) → one soil reading out (JSON). Fits a Cole-Cole
(ZARC) equivalent-circuit model, then maps the parameters to estimated
Volumetric Water Content (VWC) and bulk capacitance.

No CSV demo data, no plotting, no machine learning.

---

## Pipeline (data flow per sweep)

```
   Sensor bus payload (JSON)
   { frequency_hz[], re_z_ohm[], neg_im_z_ohm[],
     sample_id?, timestamp? }
                │
                ▼
   ┌──────────────────────────────┐
   │  load_sensor_sweep()         │   zarc_pipeline/src/data_loading.py
   │  • validate required keys    │
   │  • align array shapes        │
   │  • Im(Z) := -|neg_im_z_ohm|  │   (sign convention)
   │  • drop non-finite, f<=0     │
   │  • require >= MIN_POINTS     │
   └──────────────┬───────────────┘
                  │ Spectrum (f sorted high→low, Z = Re + jIm)
                  ▼
   ┌──────────────────────────────┐
   │  fit_spectrum()              │   zarc_pipeline/src/zarc_fitting.py
   │  Z(ω) = Rs + R/(1+(jωτ)^α)   │
   │  • scipy least_squares (TRF) │
   │  • multi-start (3 α × 3 τ)   │
   │  • residuals weighted by 1/|Z|│
   │  • bounded by ZARC_BOUNDS    │
   └──────────────┬───────────────┘
                  │ ZarcFit { Rs, R, tau, alpha,
                  │           rmse_real, rmse_imag, rmse_rel,
                  │           n_points, success, message }
                  ▼
   ┌──────────────────────────────┐
   │  calculate_vwc()             │   zarc_pipeline/src/moisture_calibration.py
   │  R_total = Rs + R            │
   │  VWC%   = A·R_total^B+OFFSET │   (clamped 0..100)
   │  C_bulk = tau / max(R, 1e-6) │
   └──────────────┬───────────────┘
                  │ { vwc_percent, bulk_capacitance_farads, r_total_ohms }
                  ▼
   ┌──────────────────────────────┐
   │  build_output_record()       │   zarc_pipeline/main.py
   │  assemble final JSON record  │
   └──────────────┬───────────────┘
                  │
                  ▼
        stdout (NDJSON) + optional --log-file
```

---

## Module architecture

```
                     ┌──────────┐
                     │ main.py  │  CLI, stream/batch I/O,
                     └────┬─────┘  output assembly
                          │
        ┌─────────────────┼─────────────────────┐
        │                 │                     │
        ▼                 ▼                     ▼
┌────────────────┐ ┌────────────────┐ ┌────────────────────────┐
│ data_loading   │ │ zarc_fitting   │ │ moisture_calibration   │
│ load_sensor_   │ │ fit_spectrum   │ │ calculate_vwc          │
│ sweep          │ │ ZarcFit        │ │ (CALIBRATION_A,B,OFFSET│
│                │ │ zarc_impedance │ │  module-level, override-│
│                │ │                │ │  able at boot)         │
└──────┬─────────┘ └──────┬─────────┘ └────────────────────────┘
       │                  │                      ▲
       │                  │                      │ (numpy only)
       ▼                  ▼                      │
┌──────────────────────────────┐                 │
│ preprocessing.Spectrum       │                 │
│ • f, omega, Z (Im<=0)        │                 │
│ • re, im, neg_im, magnitude, │                 │
│   phase_deg                  │                 │
└──────────────────────────────┘                 │
       ▲                  ▲                      │
       │                  │                      │
       └──────────────────┴──────────────────────┘
                          │
                ┌─────────┴──────────┐
                │ config.py          │  MIN_POINTS_PER_SPECTRUM,
                │                    │  ZARC_BOUNDS
                │ utils.get_logger() │  shared stderr logger
                └────────────────────┘
```

External deps: **numpy**, **scipy** only.

---

## Folder layout

```
Zarc/
├── README.md
└── zarc_pipeline/
    ├── requirements.txt
    ├── config.py                   # MIN_POINTS_PER_SPECTRUM, ZARC_BOUNDS
    ├── main.py                     # CLI + processing loop
    └── src/
        ├── __init__.py
        ├── data_loading.py         # load_sensor_sweep()
        ├── preprocessing.py        # Spectrum
        ├── zarc_fitting.py         # fit_spectrum(), ZarcFit, zarc_impedance()
        ├── moisture_calibration.py # calculate_vwc()
        └── utils.py                # get_logger()
```

---

## Install

```bash
pip install -r zarc_pipeline/requirements.txt
```

## Run

**Stream mode (default)** — one JSON payload per stdin line, one JSON per sweep on stdout:

```bash
python zarc_pipeline/main.py < live_bus.ndjson
```

**Batch mode** — file containing a JSON array OR newline-delimited JSON:

```bash
python zarc_pipeline/main.py --input sweeps.json
```

**Mirror output to a log file:**

```bash
python zarc_pipeline/main.py --input sweeps.json --log-file outputs.ndjson
```

---

## Wire format (input)

Two payload shapes are accepted. Each represents **one** frequency sweep.

### Column-array dict (recommended)

```json
{
  "sample_id":    "soil-001",
  "timestamp":    "2026-05-09T14:21:03Z",
  "frequency_hz": [1.0e6, 9.26e5, 8.58e5, ...],
  "re_z_ohm":     [3160.0, 3480.0, 3840.0, ...],
  "neg_im_z_ohm": [9670.0, 10200.0, 10300.0, ...]
}
```

`sample_id` and `timestamp` are optional; both are propagated to the output.

### Per-point list

```json
[
  {"frequency_hz": 1.0e6,  "re_z_ohm": 3160.0, "neg_im_z_ohm": 9670.0},
  {"frequency_hz": 9.26e5, "re_z_ohm": 3480.0, "neg_im_z_ohm": 10200.0}
]
```

Per-point list does NOT carry `sample_id` / `timestamp` (they're dict-level
fields). Use the column-array form if you need those.

`neg_im_z_ohm` is `-Im(Z)` as transmitted by the probe (positive for
capacitive media). Internally the loader stores `Z = Re + jIm` with `Im <= 0`.

### Validation

`load_sensor_sweep` rejects a payload (raises `ValueError`, surfaced as an
error record on the output stream) when:

* required keys are missing
* arrays are mis-aligned or non-1-D
* fewer than `MIN_POINTS_PER_SPECTRUM` (default 20) finite, positive-frequency points remain

---

## Output schema

One JSON record per sweep, written to stdout (and optionally `--log-file`):

```json
{
  "timestamp": "2026-05-09T14:21:03Z",
  "sample_id": "soil-001",
  "zarc_parameters": {
    "Rs":    1247.3,
    "R":     58420.5,
    "tau":   2.31e-4,
    "alpha": 0.812
  },
  "soil_metrics": {
    "vwc_percent":             14.7,
    "bulk_capacitance_farads": 3.95e-9
  },
  "fit_health": {
    "success":  true,
    "rmse_rel": 0.038,
    "n_points": 100,
    "message":  "`ftol` termination condition is satisfied."
  }
}
```

Field semantics:

| Field | Meaning |
|---|---|
| `timestamp` | `spectrum.timestamp` if provided, else current UTC ISO-8601 |
| `sample_id` | Echo of the input `sample_id` (`null` if not provided) |
| `zarc_parameters.Rs` | Series / high-frequency resistance (Ω) |
| `zarc_parameters.R` | Polarization resistance / Nyquist arc diameter (Ω) |
| `zarc_parameters.tau` | Characteristic relaxation time (s) |
| `zarc_parameters.alpha` | Cole-Cole depression exponent (0.3..1.0) |
| `soil_metrics.vwc_percent` | Estimated volumetric water content (0..100) |
| `soil_metrics.bulk_capacitance_farads` | `tau / R` (farads) |
| `fit_health.success` | Solver converged |
| `fit_health.rmse_rel` | Weighted RMS residual on `(Re, Im)/|Z|` |
| `fit_health.n_points` | Frequency points actually used by the fit |
| `fit_health.message` | Solver termination message |

### Error record shape (validation failure)

```json
{
  "timestamp": "2026-05-09T14:21:03+00:00",
  "sample_id": "soil-001",
  "error": "Payload is missing required keys: ['neg_im_z_ohm']",
  "fit_health": {
    "success": false,
    "rmse_rel": null,
    "n_points": 0,
    "message": "Payload is missing required keys: ['neg_im_z_ohm']"
  }
}
```

The pipeline never silently drops a malformed sweep — the consumer stays
in lockstep with the input stream.

---

## Math

### Forward model

```
Z(ω) = Rs + R / (1 + (jωτ)^α)
```

| Symbol | Meaning | Units |
|---|---|---|
| `Rs` | series resistance (high-frequency intercept) | Ω |
| `R` | polarization resistance (arc diameter) | Ω |
| `τ` | relaxation time | s |
| `α` | Cole-Cole depression exponent | — |

### Fitting

* Solver: `scipy.optimize.least_squares`, method `trf`
* Free parameters: `(Rs, R, τ, α)`
* Bounds: `config.ZARC_BOUNDS`
* Multi-start: 3 `α` initial values × 3 decade-spaced `τ` values; lowest-cost solution wins
* Residuals weighted by `1/|Z|` so the fit is not biased toward high-magnitude points across decades of |Z|
* `rmse_rel = sqrt(2·cost / n_residuals)` on the weighted residual vector

Failures (solver exception, non-convergence) return `ZarcFit(success=False)`
with NaN parameters and a populated `message` field.

### Moisture calibration

```
R_total = Rs + R
VWC%    = CALIBRATION_A · R_total^CALIBRATION_B + CALIBRATION_OFFSET   (clamped to [0, 100])
C_bulk  = τ / max(R, 1e-6)                                             (farads)
```

Defaults (PLACEHOLDERS — replace via probe-in-the-loop calibration):

```
CALIBRATION_A      = 1500.0
CALIBRATION_B      = -0.45
CALIBRATION_OFFSET = 2.0
```

The constants are module-level attributes in `zarc_pipeline/src/moisture_calibration.py`,
overwritable at boot from a calibration file without code changes.

---

## Configuration (`zarc_pipeline/config.py`)

| Name | Default | Purpose |
|---|---|---|
| `MIN_POINTS_PER_SPECTRUM` | `20` | Minimum points required for a fit |
| `ZARC_BOUNDS["Rs"]` | `(0, 1e9)` | Optimiser bounds (Ω) |
| `ZARC_BOUNDS["R"]` | `(1e-3, 1e10)` | Optimiser bounds (Ω) |
| `ZARC_BOUNDS["tau"]` | `(1e-12, 1e3)` | Optimiser bounds (s) |
| `ZARC_BOUNDS["alpha"]` | `(0.3, 1.0)` | Optimiser bounds (—) |

---

## Limitations

* `CALIBRATION_A/B/OFFSET` are placeholders. `vwc_percent` is *qualitatively*
  correct (decreases with rising resistance) but not absolute until
  calibrated against soil samples of known gravimetric water content.
* `bulk_capacitance_farads` is `τ/R` — a direct, calibration-free relative
  moisture indicator.
* No uncertainty quantification on `(Rs, R, τ, α)`. Only fit RMSE is reported.
* `alpha` is not consumed by `calculate_vwc`. Retained in the signature so a
  future model that uses arc depression (e.g. soil-texture inference) can
  drop in without changing call sites.
* No pre-fit DSP layer (Hampel filter, Kramers-Kronig validation). Bad
  sweeps surface as elevated `rmse_rel`; downstream consumers should gate on it.

