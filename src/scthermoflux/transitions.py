from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_transition_matrix(
    latent: np.ndarray,
    microstates: np.ndarray,
    disease_axis: np.ndarray | None = None,
    k: int = 12,
    beta: float = 2.5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Build a microstate transition matrix from a cell-level kNN graph.

    If disease_axis is provided, edges are biased in the direction of increasing
    disease-axis score. This does not prove time progression; it encodes an
    inferred disease-associated transition orientation.
    """
    n_states = int(np.max(microstates)) + 1
    K_counts = np.zeros((n_states, n_states), dtype=float)
    k = min(k, max(2, latent.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(latent)
    dist, idx = nn.kneighbors(latent)
    scale = np.median(dist[:, 1:]) + eps
    if disease_axis is None:
        disease_axis = np.zeros(latent.shape[0])
    disease_axis = np.asarray(disease_axis, dtype=float)
    for i in range(latent.shape[0]):
        si = int(microstates[i])
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            sj = int(microstates[j])
            if si == sj:
                continue
            spatial_w = np.exp(-d / scale)
            directional_w = 1.0 / (1.0 + np.exp(-beta * (disease_axis[j] - disease_axis[i])))
            K_counts[si, sj] += spatial_w * directional_w
    # add weak self loops for stability
    for i in range(n_states):
        K_counts[i, i] += eps + 0.01 * K_counts[i].sum()
    row_sum = K_counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    K = K_counts / row_sum
    return K


def stationary_distribution(K: np.ndarray, max_iter: int = 10000, tol: float = 1e-12) -> np.ndarray:
    n = K.shape[0]
    pi = np.ones(n) / n
    for _ in range(max_iter):
        new = pi @ K
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    pi = np.maximum(pi, 0)
    return pi / pi.sum()
