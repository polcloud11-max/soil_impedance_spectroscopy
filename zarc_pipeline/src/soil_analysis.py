"""
Exploratory analysis of soil_impedance_all.csv.

Important scientific note
-------------------------
This dataset contains a SINGLE TOTAL_IMPEDANCE value per observation, with
no frequency information. It therefore cannot be used for true Cole-Cole/
ZARC fitting or any frequency-domain spectroscopy. We treat it strictly as
an environmental/contextual dataset that complements the EIS pipeline.

Outputs:
* histograms and scatter plots for impedance vs temperature / depth
* aggregated statistics by station and depth
* an exploratory K-Means clustering on (impedance, temperature, depth) for
  unsupervised soil-state grouping (NOT a supervised soil-moisture model —
  there are no moisture labels in this CSV).
"""
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .utils import get_logger
from config import FIG_DIR, FEAT_DIR, FIG_DPI

log = get_logger()


def soil_summary(df: pd.DataFrame) -> Dict:
    return {
        "n_rows": int(len(df)),
        "by_depth": df.groupby("PROBE_DEPTH")["TOTAL_IMPEDANCE"]
                      .agg(["count", "mean", "median", "std"])
                      .to_dict(orient="index"),
        "impedance_quantiles": df["TOTAL_IMPEDANCE"]
                                 .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                                 .to_dict(),
        "corr_imp_temp": float(
            df[["TOTAL_IMPEDANCE", "SOIL_TEMP"]].corr().iloc[0, 1]
        ),
    }


def plot_soil_distributions(df: pd.DataFrame,
                            out_path: Optional[Path] = None) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].hist(df["TOTAL_IMPEDANCE"].dropna(), bins=50,
                    color="C0", edgecolor="k", alpha=0.8)
    axes[0, 0].set_xlabel("Total impedance [Ohm]")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Soil impedance distribution")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(df["SOIL_TEMP"].dropna(), bins=40,
                    color="C2", edgecolor="k", alpha=0.8)
    axes[0, 1].set_xlabel("Soil temperature [C]")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Soil temperature distribution")
    axes[0, 1].grid(alpha=0.3)

    sub = df.dropna(subset=["TOTAL_IMPEDANCE", "SOIL_TEMP"])
    axes[1, 0].scatter(sub["SOIL_TEMP"], sub["TOTAL_IMPEDANCE"],
                       s=8, alpha=0.4)
    axes[1, 0].set_xlabel("Soil temperature [C]")
    axes[1, 0].set_ylabel("Total impedance [Ohm]")
    axes[1, 0].set_title("Impedance vs temperature")
    axes[1, 0].grid(alpha=0.3)

    by_depth = df.dropna(subset=["TOTAL_IMPEDANCE", "PROBE_DEPTH"])
    depths = sorted(by_depth["PROBE_DEPTH"].unique())
    data = [by_depth.loc[by_depth["PROBE_DEPTH"] == d, "TOTAL_IMPEDANCE"]
            for d in depths]
    axes[1, 1].boxplot(data, labels=[f"{d:g}" for d in depths])
    axes[1, 1].set_xlabel("Probe depth [cm]")
    axes[1, 1].set_ylabel("Total impedance [Ohm]")
    axes[1, 1].set_title("Impedance by probe depth")
    axes[1, 1].grid(alpha=0.3, axis="y")

    out_path = out_path or (FIG_DIR / "soil_overview.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=FIG_DPI)
    plt.close(fig)
    log.info(f"Saved {out_path}")
    return out_path


def cluster_soil_states(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """
    Unsupervised grouping on (impedance, temperature, depth). Useful for
    exploratory soil-state segmentation. NOT a calibrated moisture model.
    """
    X = df[["TOTAL_IMPEDANCE", "SOIL_TEMP", "PROBE_DEPTH"]].dropna()
    if len(X) < k * 5:
        log.warning("Not enough rows for clustering")
        return df.assign(cluster=np.nan)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_std)

    out = df.copy()
    out["cluster"] = np.nan
    out.loc[X.index, "cluster"] = labels

    cluster_path = FEAT_DIR / "soil_clusters.csv"
    out.to_csv(cluster_path, index=False)
    log.info(f"Saved {cluster_path}")

    # Visualise clusters in (T, Z) space
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(k):
        sel = out["cluster"] == c
        ax.scatter(out.loc[sel, "SOIL_TEMP"],
                   out.loc[sel, "TOTAL_IMPEDANCE"],
                   s=10, alpha=0.5, label=f"cluster {c}")
    ax.set_xlabel("Soil temperature [C]")
    ax.set_ylabel("Total impedance [Ohm]")
    ax.set_title(f"K-Means soil-state clusters (k={k})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "soil_clusters.png", dpi=FIG_DPI)
    plt.close(fig)

    return out
