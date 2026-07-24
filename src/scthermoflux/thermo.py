from __future__ import annotations

import numpy as np
import pandas as pd


def probability_current(K: np.ndarray, pi: np.ndarray) -> np.ndarray:
    flow = pi[:, None] * K
    return flow - flow.T


def entropy_production(K: np.ndarray, pi: np.ndarray, eps: float = 1e-12) -> float:
    flow = pi[:, None] * K
    ratio = np.log((flow + eps) / (flow.T + eps))
    J = flow - flow.T
    sigma = 0.5 * np.sum(J * ratio)
    return float(max(sigma, 0.0))


def edgewise_flux_table(K: np.ndarray, pi: np.ndarray, microstate_table: pd.DataFrame, eps: float = 1e-12) -> pd.DataFrame:
    flow = pi[:, None] * K
    J = flow - flow.T
    rows = []
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            if i == j:
                continue
            if J[i, j] > 0:
                affinity = np.log((flow[i, j] + eps) / (flow[j, i] + eps))
                rows.append({
                    "from_microstate": int(i),
                    "to_microstate": int(j),
                    "net_current": float(J[i, j]),
                    "forward_flow": float(flow[i, j]),
                    "reverse_flow": float(flow[j, i]),
                    "affinity": float(affinity),
                    "edge_entropy_production": float(J[i, j] * affinity),
                })
    if not rows:
        return pd.DataFrame(columns=[
            "from_microstate", "to_microstate", "net_current", "forward_flow",
            "reverse_flow", "affinity", "edge_entropy_production"
        ])
    out = pd.DataFrame(rows).sort_values("edge_entropy_production", ascending=False)
    return out


def condition_level_entropy(meta: pd.DataFrame, microstates: np.ndarray, K: np.ndarray, condition_col: str = "condition") -> pd.DataFrame:
    rows = []
    n_states = K.shape[0]
    for condition, sub in meta.groupby(condition_col):
        idx = sub.index.to_numpy()
        counts = np.bincount(microstates[idx], minlength=n_states).astype(float)
        pi = counts + 1e-6
        pi = pi / pi.sum()
        rows.append({
            "condition": condition,
            "n_cells": int(len(idx)),
            "entropy_production": entropy_production(K, pi),
            "state_entropy": float(-np.sum(pi * np.log(pi + 1e-12))),
        })
    return pd.DataFrame(rows)
