from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def knn_density(latent: np.ndarray, k: int = 25, eps: float = 1e-12) -> np.ndarray:
    """Estimate local density as inverse kNN radius in latent space."""
    latent = np.asarray(latent, dtype=float)
    k = min(k, max(2, latent.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    dist, _ = nn.kneighbors(latent)
    radius = dist[:, -1]
    density = 1.0 / (radius ** latent.shape[1] + eps)
    density = density / np.sum(density)
    return density


def quasi_potential_from_density(density: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    phi = -np.log(density + eps)
    return phi - np.nanmin(phi)


def make_microstates(latent: np.ndarray, n_microstates: int = 24, random_state: int = 7):
    n_microstates = min(n_microstates, max(2, latent.shape[0] // 10))
    km = KMeans(n_clusters=n_microstates, n_init=20, random_state=random_state)
    labels = km.fit_predict(latent)
    return labels, km.cluster_centers_


def summarize_microstates(latent, meta: pd.DataFrame, labels, phi, disease_axis=None) -> pd.DataFrame:
    rows = []
    labels = np.asarray(labels)
    for state in sorted(np.unique(labels)):
        idx = labels == state
        row = {
            "microstate": int(state),
            "n_cells": int(idx.sum()),
            "latent_1": float(np.mean(latent[idx, 0])),
            "latent_2": float(np.mean(latent[idx, 1])) if latent.shape[1] > 1 else 0.0,
            "quasi_potential_mean": float(np.mean(phi[idx])),
            "quasi_potential_median": float(np.median(phi[idx])),
            "donor_count": int(meta.loc[idx, "donor_id"].nunique()),
        }
        if disease_axis is not None:
            row["disease_axis_mean"] = float(np.nanmean(disease_axis[idx]))
        cond_counts = meta.loc[idx, "condition"].value_counts(normalize=True).to_dict()
        for cond, val in cond_counts.items():
            row[f"frac_{cond}"] = float(val)
        rows.append(row)
    return pd.DataFrame(rows)
