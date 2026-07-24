from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def normalize_log1p(X: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    """Library-size normalize and log1p transform non-negative count-like data.

    If the input already looks log-transformed, this remains reasonably stable but
    users should preferably provide counts or normalized expression consistently.
    """
    X = np.asarray(X, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X[X < 0] = 0.0
    lib = X.sum(axis=1, keepdims=True)
    lib[lib == 0] = 1.0
    return np.log1p(X / lib * target_sum)


def select_hvg(X: np.ndarray, genes: list[str], n_top: int = 2000):
    """Select highly variable genes by variance after log-normalization."""
    n_top = min(n_top, X.shape[1])
    var = np.var(X, axis=0)
    idx = np.argsort(var)[::-1][:n_top]
    return X[:, idx], [genes[i] for i in idx], idx


def compute_pca(X: np.ndarray, n_components: int = 10, random_state: int = 7) -> np.ndarray:
    n_components = min(n_components, X.shape[0] - 1, X.shape[1])
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    return PCA(n_components=n_components, random_state=random_state).fit_transform(Xs)
