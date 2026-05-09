"""
Classification on the EIS feature table.

Why this is justified
---------------------
Dataset.csv contains a clean grouping variable: `ionic_radius` takes only
two values (1.82e-10 m and 2.06e-10 m). That is a real, factory-stamped
label, NOT a fabricated proxy. We can therefore ask:

    "Given the impedance features of an unknown spectrum, can we recover
     which ionic species was measured?"

This is a defensible supervised problem. We use:
* Random Forest (non-linear, robust to feature scale)
* Logistic Regression (linear, interpretable baseline)
* SVM with RBF kernel

We deliberately split by `temperature` so the test set contains
temperatures the model has not seen — a tougher, more honest evaluation
than random row splits when there is one spectrum per (radius, T).

Important limitation: with only 28 spectra total, this is a small-data
demonstration. Reported metrics indicate the *signal*, not a deployment
benchmark.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .utils import get_logger

log = get_logger()

FEATURE_COLS = [
    "Rs", "R", "tau", "alpha",
    "f_char_hz", "log10_R_over_Rs",
    "Z_min_mag", "Z_max_mag", "log10_Z_range",
    "phase_min_deg", "phase_max_deg", "phase_range_deg",
    "f_at_phase_min_hz", "neg_im_peak",
    "f_at_neg_im_peak_hz", "re_at_neg_im_peak",
]


def _prepare(features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray, List[str]]:
    """
    Return X, y, groups, feature_names. Drop rows with any NaN feature.
    `groups` = temperature, used for grouped CV.
    """
    df = features_df.dropna(subset=FEATURE_COLS + ["ionic_radius",
                                                    "temperature"]).copy()
    X = df[FEATURE_COLS].to_numpy(dtype=float)
    y = df["ionic_radius"].astype(str).to_numpy()
    groups = df["temperature"].to_numpy()
    return X, y, groups, FEATURE_COLS


def run_classification(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Leave-One-Group-Out CV across temperatures: each fold holds out one
    temperature group, trains on the rest, evaluates. Macro-averaged
    metrics are reported because classes are roughly balanced.
    """
    X, y, groups, _ = _prepare(features_df)
    if len(np.unique(y)) < 2:
        log.warning("Only one class present — skipping classification")
        return pd.DataFrame()
    if len(X) < 6:
        log.warning("Too few samples for CV — skipping classification")
        return pd.DataFrame()

    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, random_state=42),
        "LogReg": Pipeline([("scaler", StandardScaler()),
                            ("clf", LogisticRegression(max_iter=2000))]),
        "SVM-RBF": Pipeline([("scaler", StandardScaler()),
                             ("clf", SVC(kernel="rbf", C=10, gamma="scale"))]),
    }

    logo = LeaveOneGroupOut()
    rows = []
    confusions = {}

    for name, model in models.items():
        y_true_all, y_pred_all = [], []
        for tr, te in logo.split(X, y, groups=groups):
            model.fit(X[tr], y[tr])
            y_pred_all.extend(model.predict(X[te]))
            y_true_all.extend(y[te])
        y_true_all = np.asarray(y_true_all)
        y_pred_all = np.asarray(y_pred_all)

        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_true_all, y_pred_all),
            "precision_macro": precision_score(
                y_true_all, y_pred_all, average="macro", zero_division=0),
            "recall_macro": recall_score(
                y_true_all, y_pred_all, average="macro", zero_division=0),
            "f1_macro": f1_score(
                y_true_all, y_pred_all, average="macro", zero_division=0),
        })
        confusions[name] = confusion_matrix(y_true_all, y_pred_all,
                                            labels=np.unique(y))
        log.info(f"\n--- {name} ---\n"
                 + classification_report(y_true_all, y_pred_all,
                                         zero_division=0))

    metrics = pd.DataFrame(rows)
    metrics.attrs["confusions"] = confusions
    metrics.attrs["classes"] = np.unique(y).tolist()
    return metrics
