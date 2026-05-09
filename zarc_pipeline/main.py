"""
End-to-end driver for the impedance-spectroscopy pipeline.

Pipeline:
    Dataset.csv               soil_impedance_all.csv
        |                              |
   load + clean                load + clean
        |                              |
   group spectra              EDA + clustering
        |
   Nyquist + Bode plots
        |
   ZARC (Cole-Cole) fits
        |
   Feature table
        |
   Classification (ionic radius)
        |
   JSON report saved to outputs/reports
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (EIS_CSV, SOIL_CSV, FIG_DIR, FEAT_DIR, REP_DIR)
from src.classification import run_classification
from src.data_loading import (inspect_eis, inspect_soil, load_eis, load_soil)
from src.feature_extraction import FEATURE_DESCRIPTIONS, build_feature_table
from src.preprocessing import build_spectra, spectra_summary
from src.soil_analysis import (cluster_soil_states, plot_soil_distributions,
                               soil_summary)
from src.utils import get_logger
from src.visualization import plot_bode, plot_nyquist
from src.zarc_fitting import fit_all

log = get_logger()


def _np_safe(o):
    """Make numpy types JSON-serialisable."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    raise TypeError(f"not serialisable: {type(o)}")


def run_eis_pipeline() -> dict:
    log.info("=" * 60)
    log.info("EIS pipeline (Dataset.csv)")
    log.info("=" * 60)

    df = load_eis(EIS_CSV)
    eis_info = inspect_eis(df)
    log.info(f"EIS info: {eis_info}")

    spectra = build_spectra(df)
    summary = spectra_summary(spectra)
    summary.to_csv(FEAT_DIR / "spectra_summary.csv", index=False)

    plot_nyquist(spectra)
    plot_bode(spectra)

    fits_df = fit_all(spectra, plot_each=True)
    fits_df.to_csv(FEAT_DIR / "zarc_fits.csv", index=False)

    features_df = build_feature_table(spectra, fits_df)
    features_df.to_csv(FEAT_DIR / "features.csv", index=False)

    clf_metrics = run_classification(features_df)
    if not clf_metrics.empty:
        clf_metrics.to_csv(FEAT_DIR / "classification_metrics.csv", index=False)

    return {
        "eis_info": eis_info,
        "n_spectra": len(spectra),
        "n_fits_ok": int(fits_df["success"].sum()) if not fits_df.empty else 0,
        "n_fits_total": int(len(fits_df)),
        "feature_columns": features_df.columns.tolist(),
        "classification_metrics":
            clf_metrics.to_dict(orient="records") if not clf_metrics.empty
            else "skipped",
    }


def run_soil_pipeline() -> dict:
    log.info("=" * 60)
    log.info("Soil pipeline (soil_impedance_all.csv)")
    log.info("=" * 60)

    df = load_soil(SOIL_CSV)
    info = inspect_soil(df)
    log.info(f"Soil info: {info}")

    plot_soil_distributions(df)
    soil_stats = soil_summary(df)
    cluster_soil_states(df, k=3)

    return {
        "soil_info": info,
        "soil_stats": soil_stats,
        "note": ("Soil dataset has only scalar TOTAL_IMPEDANCE per row "
                 "(no frequency sweep). ZARC fitting is NOT applied here."),
    }


def write_report(report: dict) -> Path:
    out = REP_DIR / "pipeline_report.json"
    with out.open("w") as f:
        json.dump(report, f, indent=2, default=_np_safe)
    log.info(f"Saved {out}")
    return out


def main():
    report = {
        "feature_descriptions": FEATURE_DESCRIPTIONS,
        "eis": run_eis_pipeline(),
        "soil": run_soil_pipeline(),
        "scientific_notes": [
            "Dataset.csv is treated as the EIS core: real frequency-resolved "
            "spectra suitable for Nyquist/Bode/ZARC fitting and feature "
            "extraction.",
            "soil_impedance_all.csv contains only scalar impedance per "
            "observation -- it is used for context and exploratory clustering, "
            "NEVER for ZARC fitting.",
            "Sign convention: stored Img(Z) is treated as |Im(Z)|; internally "
            "Im(Z) <= 0 (capacitive); Nyquist plots show -Im(Z) on the y-axis.",
            "Classification uses ionic_radius as a real (not fabricated) label "
            "and Leave-One-Temperature-Out CV. With only 28 spectra this is a "
            "small-data demo, not a deployment benchmark.",
        ],
    }
    write_report(report)
    print("\nDone. See outputs/figures, outputs/features, outputs/reports.\n")


if __name__ == "__main__":
    main()
