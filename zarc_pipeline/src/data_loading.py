"""
Loaders + sanity inspectors for the two input CSVs.

Dataset.csv  -> frequency-resolved EIS spectra (real data for the core
                pipeline: Nyquist/Bode/ZARC).
soil_impedance_all.csv -> scalar soil impedance + metadata (no frequency
                sweep, so NOT usable for ZARC fitting).
"""
from pathlib import Path
from typing import Dict

import pandas as pd

from .utils import get_logger, to_numeric_strip

log = get_logger()


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().strip('"').strip("'") for c in df.columns]
    return df


# ----------------------------- EIS dataset ------------------------------
def load_eis(path: Path) -> pd.DataFrame:
    """Load Dataset.csv and return a cleaned dataframe ready for grouping."""
    df = pd.read_csv(path, dtype=str)        # read as text first
    df = _clean_columns(df)

    rename = {
        "Ionic radius": "ionic_radius",
        "Temperature": "temperature",
        "Frequency": "frequency",
        "Re(Z)": "re_z",
        "Img(Z)": "im_z",
    }
    df = df.rename(columns=rename)

    expected = ["ionic_radius", "temperature", "frequency", "re_z", "im_z"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset.csv is missing columns: {missing}")

    for col in expected:
        df[col] = to_numeric_strip(df[col])

    n_before = len(df)
    df = df.dropna(subset=expected).drop_duplicates()
    log.info(f"EIS rows: {len(df)} (dropped {n_before - len(df)} bad/duplicate)")
    return df


def inspect_eis(df: pd.DataFrame) -> Dict:
    """Return summary statistics about the EIS dataframe."""
    info = {
        "n_rows": int(len(df)),
        "n_ionic_radii": int(df["ionic_radius"].nunique()),
        "n_temperatures": int(df["temperature"].nunique()),
        "n_spectra": int(df.groupby(["ionic_radius", "temperature"]).ngroups),
        "freq_min_hz": float(df["frequency"].min()),
        "freq_max_hz": float(df["frequency"].max()),
        "re_z_range": (float(df["re_z"].min()), float(df["re_z"].max())),
        "im_z_range": (float(df["im_z"].min()), float(df["im_z"].max())),
    }
    return info


# ----------------------------- Soil dataset -----------------------------
def load_soil(path: Path) -> pd.DataFrame:
    """Load soil_impedance_all.csv. NOTE: only one impedance value per row."""
    df = pd.read_csv(path, dtype=str)
    df = _clean_columns(df)

    # Strip embedded quotes from string fields
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().str.strip("'\"").str.strip()

    numeric_cols = ["SAMPLE_DISTANCE", "PROBE_DEPTH", "SOIL_TEMP",
                    "TOTAL_IMPEDANCE"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = to_numeric_strip(df[c])

    if "OBS_DATE" in df.columns:
        df["OBS_DATE"] = pd.to_datetime(df["OBS_DATE"],
                                        errors="coerce", format="%d-%b-%y")

    log.info(f"Soil rows: {len(df)} | unique stations: "
             f"{df.get('STATION_ID', pd.Series()).nunique()}")
    return df


def inspect_soil(df: pd.DataFrame) -> Dict:
    info = {
        "n_rows": int(len(df)),
        "n_missing_impedance": int(df["TOTAL_IMPEDANCE"].isna().sum()),
        "n_missing_temp": int(df["SOIL_TEMP"].isna().sum()),
        "impedance_stats": {
            "min": float(df["TOTAL_IMPEDANCE"].min()),
            "median": float(df["TOTAL_IMPEDANCE"].median()),
            "mean": float(df["TOTAL_IMPEDANCE"].mean()),
            "max": float(df["TOTAL_IMPEDANCE"].max()),
        },
        "depths_present": sorted(df["PROBE_DEPTH"].dropna().unique().tolist()),
        "n_stations": int(df["STATION_ID"].nunique())
            if "STATION_ID" in df.columns else None,
    }
    return info
