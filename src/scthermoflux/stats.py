from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind
from .thermo import entropy_production


def donor_level_statistics(meta: pd.DataFrame, microstates: np.ndarray, phi: np.ndarray, disease_axis: np.ndarray, K: np.ndarray) -> pd.DataFrame:
    rows = []
    n_states = K.shape[0]
    for donor, sub in meta.groupby("donor_id"):
        idx = sub.index.to_numpy()
        condition = sub["condition"].mode().iloc[0]
        counts = np.bincount(microstates[idx], minlength=n_states).astype(float)
        pi = counts + 1e-6
        pi = pi / pi.sum()
        rows.append({
            "donor_id": donor,
            "condition": condition,
            "n_cells": int(len(idx)),
            "entropy_production": entropy_production(K, pi),
            "state_entropy": float(-np.sum(pi * np.log(pi + 1e-12))),
            "quasi_potential_mean": float(np.mean(phi[idx])),
            "quasi_potential_median": float(np.median(phi[idx])),
            "disease_axis_mean": float(np.mean(disease_axis[idx])),
            "disease_axis_median": float(np.median(disease_axis[idx])),
        })
    return pd.DataFrame(rows)


def compare_conditions(donor_df: pd.DataFrame, condition_a: str = "ND", condition_b: str = "T2D") -> pd.DataFrame:
    metrics = [
        "entropy_production", "state_entropy", "quasi_potential_mean",
        "quasi_potential_median", "disease_axis_mean", "disease_axis_median"
    ]
    rows = []
    for metric in metrics:
        a = donor_df.loc[donor_df.condition == condition_a, metric].dropna().to_numpy()
        b = donor_df.loc[donor_df.condition == condition_b, metric].dropna().to_numpy()
        if len(a) < 2 or len(b) < 2:
            p_mwu = np.nan
            p_t = np.nan
        else:
            p_mwu = mannwhitneyu(a, b, alternative="two-sided").pvalue
            p_t = ttest_ind(a, b, equal_var=False).pvalue
        rows.append({
            "metric": metric,
            "condition_a": condition_a,
            "condition_b": condition_b,
            "n_a": int(len(a)),
            "n_b": int(len(b)),
            "mean_a": float(np.mean(a)) if len(a) else np.nan,
            "mean_b": float(np.mean(b)) if len(b) else np.nan,
            "delta_b_minus_a": float(np.mean(b) - np.mean(a)) if len(a) and len(b) else np.nan,
            "mannwhitney_p": p_mwu,
            "welch_t_p": p_t,
        })
    return pd.DataFrame(rows)
