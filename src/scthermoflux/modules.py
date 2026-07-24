from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from .utils import zscore


def load_modules(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return {name: spec["genes"] for name, spec in y["modules"].items()}


def score_modules(X: np.ndarray, genes: list[str], modules: dict[str, list[str]]) -> pd.DataFrame:
    """Score gene modules as mean expression over available genes."""
    gene_to_idx = {g.upper(): i for i, g in enumerate(genes)}
    out = {}
    for name, module_genes in modules.items():
        idx = [gene_to_idx[g.upper()] for g in module_genes if g.upper() in gene_to_idx]
        if len(idx) == 0:
            # Missing modules are encoded as zeros so that composite axes remain finite.
            out[name] = np.zeros(X.shape[0], dtype=float)
        else:
            out[name] = np.nanmean(X[:, idx], axis=1)
    df = pd.DataFrame(out)
    return df


def add_disease_axis(module_scores: pd.DataFrame) -> pd.DataFrame:
    """Add a conservative disease-axis score from curated modules."""
    df = module_scores.copy()
    def col(name):
        return df[name].to_numpy() if name in df else np.zeros(len(df))
    beta = zscore(col("BetaIdentitySecretion"))
    immune = zscore(col("ImmuneStress"))
    er = zscore(col("ERStress_UPR"))
    dediff = zscore(col("DedifferentiationStress"))
    df["DiseaseAxis"] = immune + er + dediff - beta
    return df
