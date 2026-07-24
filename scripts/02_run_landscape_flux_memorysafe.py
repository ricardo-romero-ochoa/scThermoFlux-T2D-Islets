#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import anndata as ad
except ImportError as exc:
    raise ImportError("Install anndata to read .h5ad files: pip install anndata") from exc

try:
    from scipy import sparse
    from scipy.stats import mannwhitneyu, ttest_ind
except ImportError as exc:
    raise ImportError("Install scipy: pip install scipy") from exc

from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - mu) / sd).astype(np.float32)


def is_sparse_matrix(X) -> bool:
    return sparse.issparse(X)


def as_float32_sparse_or_dense(X):
    if is_sparse_matrix(X):
        return X.tocsr().astype(np.float32)
    return np.asarray(X, dtype=np.float32)


# ---------------------------------------------------------------------
# Data loading and subsetting
# ---------------------------------------------------------------------
def read_and_subset_h5ad(
    infile: str | Path,
    cell_type: str,
    conditions: list[str],
    min_cells_per_donor: int = 100,
    max_cells_per_donor: int = 1500,
    seed: int = 7,
):
    print(f"Reading: {infile}")
    adata = ad.read_h5ad(infile)

    required = {"donor_id", "condition", "cell_type"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing required obs columns in h5ad: {sorted(missing)}")

    mask = np.ones(adata.n_obs, dtype=bool)
    if cell_type and cell_type.lower() != "all":
        mask &= adata.obs["cell_type"].astype(str).str.lower().eq(cell_type.lower()).to_numpy()
    if conditions:
        mask &= adata.obs["condition"].astype(str).isin(conditions).to_numpy()

    adata = adata[mask].copy()
    if adata.n_obs < 20:
        raise ValueError(f"Subset contains only {adata.n_obs} cells; too few for analysis.")

    # Filter donors with too few cells first.
    donor_counts = adata.obs["donor_id"].astype(str).value_counts()
    keep_donors = donor_counts[donor_counts >= min_cells_per_donor].index
    adata = adata[adata.obs["donor_id"].astype(str).isin(keep_donors)].copy()

    if adata.n_obs < 20:
        raise ValueError(
            "No usable cells after donor filtering. Lower --min-cells-per-donor."
        )

    # Optional donor-balanced downsampling. This prevents hidden memory explosions
    # and also avoids over-weighting donors with very many cells.
    if max_cells_per_donor and max_cells_per_donor > 0:
        rng = np.random.default_rng(seed)
        selected = []
        obs = adata.obs.reset_index(drop=False)
        for donor, sub in obs.groupby("donor_id", observed=True):
            idx = sub.index.to_numpy()
            if len(idx) > max_cells_per_donor:
                idx = rng.choice(idx, size=max_cells_per_donor, replace=False)
            selected.append(idx)
        selected = np.concatenate(selected)
        selected.sort()
        adata = adata[selected].copy()

    adata.X = as_float32_sparse_or_dense(adata.X)
    adata.obs = adata.obs.copy().reset_index(drop=False).rename(columns={"index": "cell_id"})
    adata.obs_names = adata.obs["cell_id"].astype(str).to_numpy()
    adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)
    adata.obs["condition"] = adata.obs["condition"].astype(str)
    adata.obs["cell_type"] = adata.obs["cell_type"].astype(str)

    print("Subset after donor filtering/downsampling:")
    print(adata)
    print("Cells by condition:")
    print(adata.obs["condition"].value_counts())
    print("Donors by condition:")
    print(adata.obs[["donor_id", "condition"]].drop_duplicates()["condition"].value_counts())

    return adata


# ---------------------------------------------------------------------
# Sparse-safe preprocessing
# ---------------------------------------------------------------------
def normalize_log1p_sparse_safe(X, target_sum: float = 1e4):
    """Library-size normalize and log1p without densifying sparse matrices."""
    X = as_float32_sparse_or_dense(X)
    if is_sparse_matrix(X):
        lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float32)
        lib[lib <= 0] = 1.0
        scale = (target_sum / lib).astype(np.float32)
        Xn = sparse.diags(scale).dot(X).tocsr()
        Xn.data = np.log1p(Xn.data).astype(np.float32)
        return Xn

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    X[X < 0] = 0.0
    lib = X.sum(axis=1, keepdims=True)
    lib[lib <= 0] = 1.0
    return np.log1p(X / lib * target_sum).astype(np.float32)


def select_hvg_sparse_safe(X, genes: list[str], n_top: int = 2000):
    """Variance-based HVG selection without densifying sparse matrices."""
    n_top = int(min(n_top, X.shape[1]))
    if is_sparse_matrix(X):
        mean = np.asarray(X.mean(axis=0)).ravel().astype(np.float64)
        mean_sq = np.asarray(X.power(2).mean(axis=0)).ravel().astype(np.float64)
        var = mean_sq - mean**2
    else:
        var = np.var(X, axis=0)
    var = np.nan_to_num(var, nan=0.0, posinf=0.0, neginf=0.0)
    idx = np.argsort(var)[::-1][:n_top]
    return X[:, idx], [genes[i] for i in idx], idx


def compute_latent_svd(X, n_components: int = 10, seed: int = 7) -> np.ndarray:
    """Sparse-safe latent representation.

    TruncatedSVD does not center features, which is acceptable here because the
    objective is a memory-stable transcriptomic manifold rather than exact PCA.
    """
    n_components = int(min(n_components, X.shape[0] - 1, X.shape[1]))
    model = TruncatedSVD(n_components=n_components, random_state=seed)
    latent = model.fit_transform(X).astype(np.float32)
    return latent


# ---------------------------------------------------------------------
# Module scoring
# ---------------------------------------------------------------------
def load_modules(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        y = yaml.safe_load(f)
    return {name: spec["genes"] for name, spec in y["modules"].items()}


def score_modules_sparse_safe(X, genes: list[str], modules: dict[str, list[str]]) -> pd.DataFrame:
    gene_to_idx = {str(g).upper(): i for i, g in enumerate(genes)}
    out = {}
    for name, module_genes in modules.items():
        idx = [gene_to_idx[g.upper()] for g in module_genes if g.upper() in gene_to_idx]
        if len(idx) == 0:
            out[name] = np.zeros(X.shape[0], dtype=np.float32)
        else:
            vals = X[:, idx]
            if is_sparse_matrix(vals):
                score = np.asarray(vals.mean(axis=1)).ravel()
            else:
                score = np.nanmean(vals, axis=1)
            out[name] = np.asarray(score, dtype=np.float32)
    df = pd.DataFrame(out)
    beta = zscore(df.get("BetaIdentitySecretion", pd.Series(np.zeros(X.shape[0]))).to_numpy())
    immune = zscore(df.get("ImmuneStress", pd.Series(np.zeros(X.shape[0]))).to_numpy())
    er = zscore(df.get("ERStress_UPR", pd.Series(np.zeros(X.shape[0]))).to_numpy())
    dediff = zscore(df.get("DedifferentiationStress", pd.Series(np.zeros(X.shape[0]))).to_numpy())
    df["DiseaseAxis"] = immune + er + dediff - beta
    return df


# ---------------------------------------------------------------------
# Landscape, microstates, transitions, thermodynamics
# ---------------------------------------------------------------------
def knn_density(latent_2d: np.ndarray, k: int = 25, eps: float = 1e-12) -> np.ndarray:
    latent_2d = np.asarray(latent_2d, dtype=np.float32)
    k = min(k, max(2, latent_2d.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(latent_2d)
    dist, _ = nn.kneighbors(latent_2d)
    radius = dist[:, -1].astype(np.float64)
    density = 1.0 / (radius ** latent_2d.shape[1] + eps)
    density = density / np.sum(density)
    return density


def quasi_potential_from_density(density: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    phi = -np.log(density + eps)
    return (phi - np.nanmin(phi)).astype(np.float32)


def make_microstates(latent: np.ndarray, n_microstates: int = 24, seed: int = 7):
    n_microstates = int(min(n_microstates, max(2, latent.shape[0] // 10)))
    km = MiniBatchKMeans(
        n_clusters=n_microstates,
        random_state=seed,
        n_init="auto",
        batch_size=min(4096, max(256, latent.shape[0])),
    )
    labels = km.fit_predict(latent).astype(int)
    return labels, km.cluster_centers_.astype(np.float32)


def summarize_microstates(latent, meta: pd.DataFrame, labels, phi, disease_axis=None) -> pd.DataFrame:
    rows = []
    labels = np.asarray(labels)
    meta = meta.reset_index(drop=True)
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


def build_transition_matrix(latent, microstates, disease_axis=None, k: int = 12, beta: float = 2.5, eps: float = 1e-12):
    n_states = int(np.max(microstates)) + 1
    K_counts = np.zeros((n_states, n_states), dtype=np.float64)
    k = min(k, max(2, latent.shape[0] - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(latent)
    dist, idx = nn.kneighbors(latent)
    scale = np.median(dist[:, 1:]) + eps
    if disease_axis is None:
        disease_axis = np.zeros(latent.shape[0], dtype=np.float32)
    disease_axis = np.asarray(disease_axis, dtype=np.float32)

    for i in range(latent.shape[0]):
        si = int(microstates[i])
        for d, j in zip(dist[i, 1:], idx[i, 1:]):
            sj = int(microstates[j])
            if si == sj:
                continue
            spatial_w = np.exp(-float(d) / scale)
            directional_w = 1.0 / (1.0 + np.exp(-beta * float(disease_axis[j] - disease_axis[i])))
            K_counts[si, sj] += spatial_w * directional_w

    for i in range(n_states):
        K_counts[i, i] += eps + 0.01 * K_counts[i].sum()
    row_sum = K_counts.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return K_counts / row_sum


def stationary_distribution(K: np.ndarray, max_iter: int = 10000, tol: float = 1e-12) -> np.ndarray:
    n = K.shape[0]
    pi = np.ones(n, dtype=np.float64) / n
    for _ in range(max_iter):
        new = pi @ K
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    pi = np.maximum(pi, 0)
    return pi / pi.sum()


def entropy_production(K: np.ndarray, pi: np.ndarray, eps: float = 1e-12) -> float:
    flow = pi[:, None] * K
    ratio = np.log((flow + eps) / (flow.T + eps))
    J = flow - flow.T
    sigma = 0.5 * np.sum(J * ratio)
    return float(max(sigma, 0.0))


def edgewise_flux_table(K: np.ndarray, pi: np.ndarray, eps: float = 1e-12) -> pd.DataFrame:
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
    return pd.DataFrame(rows).sort_values("edge_entropy_production", ascending=False)


def condition_level_entropy(meta: pd.DataFrame, microstates: np.ndarray, K: np.ndarray) -> pd.DataFrame:
    rows = []
    n_states = K.shape[0]
    meta = meta.reset_index(drop=True)
    for condition, sub in meta.groupby("condition", observed=True):
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


def donor_level_statistics(meta: pd.DataFrame, microstates, phi, disease_axis, K) -> pd.DataFrame:
    rows = []
    n_states = K.shape[0]
    meta = meta.reset_index(drop=True)
    for donor, sub in meta.groupby("donor_id", observed=True):
        idx = sub.index.to_numpy()
        condition = sub["condition"].mode().iloc[0]
        counts = np.bincount(microstates[idx], minlength=n_states).astype(float)
        pi = counts + 1e-6
        pi = pi / pi.sum()
        row = {
            "donor_id": donor,
            "condition": condition,
            "n_cells": int(len(idx)),
            "entropy_production": entropy_production(K, pi),
            "state_entropy": float(-np.sum(pi * np.log(pi + 1e-12))),
            "quasi_potential_mean": float(np.mean(phi[idx])),
            "quasi_potential_median": float(np.median(phi[idx])),
            "disease_axis_mean": float(np.mean(disease_axis[idx])),
            "disease_axis_median": float(np.median(disease_axis[idx])),
        }
        for col in ["sex", "age_years", "BMI", "HbA1c", "assay", "batch_assay"]:
            if col in meta.columns:
                row[col] = sub[col].iloc[0]
        rows.append(row)
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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Memory-safe transcriptomic landscape/flux/entropy analysis for large h5ad files."
    )
    p.add_argument("--input", required=True, help="Standardized .h5ad file")
    p.add_argument("--dataset", required=True)
    p.add_argument("--cell-type", default="beta")
    p.add_argument("--conditions", default="ND,T2D")
    p.add_argument("--modules", default="data/metadata/core_modules.yaml")
    p.add_argument("--outdir", required=True)
    p.add_argument("--n-latent", type=int, default=10)
    p.add_argument("--n-hvg", type=int, default=2000)
    p.add_argument("--n-microstates", type=int, default=24)
    p.add_argument("--density-k", type=int, default=25)
    p.add_argument("--knn-k", type=int, default=12)
    p.add_argument("--transition-beta", type=float, default=2.5)
    p.add_argument("--min-cells-per-donor", type=int, default=100)
    p.add_argument(
        "--max-cells-per-donor",
        type=int,
        default=1500,
        help="Donor-balanced downsampling. Use 0 to disable, but 1000-1500 is recommended on laptops.",
    )
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    outdir = ensure_dir(args.outdir)
    tables = ensure_dir(outdir / "tables")

    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]

    adata = read_and_subset_h5ad(
        args.input,
        cell_type=args.cell_type,
        conditions=conditions,
        min_cells_per_donor=args.min_cells_per_donor,
        max_cells_per_donor=args.max_cells_per_donor,
        seed=args.seed,
    )
    adata.obs["dataset"] = args.dataset
    genes = list(map(str, adata.var_names))
    meta = adata.obs.reset_index(drop=True).copy()

    print("Normalizing/log-transforming without densifying...")
    Xn = normalize_log1p_sparse_safe(adata.X)

    print("Scoring modules...")
    modules = load_modules(args.modules)
    module_scores = score_modules_sparse_safe(Xn, genes, modules)
    module_scores.to_csv(tables / "cell_module_scores.csv", index=False)

    print("Selecting highly variable genes...")
    Xh, hvg_genes, hvg_idx = select_hvg_sparse_safe(Xn, genes, n_top=args.n_hvg)
    pd.DataFrame({"gene": hvg_genes}).to_csv(tables / "hvg_genes.csv", index=False)

    print("Computing sparse-safe latent representation with TruncatedSVD...")
    latent = compute_latent_svd(Xh, n_components=args.n_latent, seed=args.seed)
    pd.DataFrame(
        latent,
        columns=[f"latent_{i+1}" for i in range(latent.shape[1])]
    ).to_csv(tables / "cell_latent_coordinates.csv", index=False)

    print("Estimating density/quasi-potential...")
    density = knn_density(latent[:, :2], k=args.density_k)
    phi = quasi_potential_from_density(density)

    print("Building microstates...")
    microstates, centers = make_microstates(
        latent[:, :min(5, latent.shape[1])],
        n_microstates=args.n_microstates,
        seed=args.seed,
    )

    print("Writing cell-level state scores...")
    cell_state = pd.concat([meta.reset_index(drop=True), module_scores.reset_index(drop=True)], axis=1)
    cell_state["microstate"] = microstates
    cell_state["quasi_potential"] = phi
    cell_state["latent_1"] = latent[:, 0]
    cell_state["latent_2"] = latent[:, 1] if latent.shape[1] > 1 else 0.0
    cell_state.to_csv(tables / "cell_state_scores.csv", index=False)

    microstate_table = summarize_microstates(
        latent, meta, microstates, phi,
        disease_axis=module_scores["DiseaseAxis"].to_numpy()
    )
    microstate_table.to_csv(tables / "microstate_table.csv", index=False)

    print("Estimating transition matrix...")
    K = build_transition_matrix(
        latent[:, :min(5, latent.shape[1])],
        microstates,
        disease_axis=module_scores["DiseaseAxis"].to_numpy(),
        k=args.knn_k,
        beta=args.transition_beta,
    )
    np.savetxt(tables / "transition_matrix.csv", K, delimiter=",")

    pi = stationary_distribution(K)
    np.savetxt(tables / "stationary_distribution.csv", pi, delimiter=",")

    print("Computing entropy-production-like observables...")
    edge_table = edgewise_flux_table(K, pi)
    edge_table.to_csv(tables / "edge_flux_table.csv", index=False)
    pd.DataFrame({"global_entropy_production": [entropy_production(K, pi)]}).to_csv(
        tables / "global_thermo.csv", index=False
    )

    condition_thermo = condition_level_entropy(meta, microstates, K)
    condition_thermo.to_csv(tables / "condition_level_thermo.csv", index=False)

    donor_df = donor_level_statistics(
        meta, microstates, phi,
        module_scores["DiseaseAxis"].to_numpy(), K
    )
    donor_df.to_csv(tables / "donor_level_thermo.csv", index=False)

    comp = compare_conditions(
        donor_df,
        condition_a=conditions[0],
        condition_b=conditions[1] if len(conditions) > 1 else conditions[0],
    )
    comp.to_csv(tables / "condition_comparison.csv", index=False)

    run_info = pd.DataFrame([
        {
            "input": str(args.input),
            "dataset": args.dataset,
            "cell_type": args.cell_type,
            "conditions": ",".join(conditions),
            "n_cells_used": int(adata.n_obs),
            "n_genes_total": int(adata.n_vars),
            "n_hvg": int(len(hvg_genes)),
            "n_latent": int(latent.shape[1]),
            "n_microstates": int(K.shape[0]),
            "min_cells_per_donor": int(args.min_cells_per_donor),
            "max_cells_per_donor": int(args.max_cells_per_donor),
            "global_entropy_production": float(entropy_production(K, pi)),
        }
    ])
    run_info.to_csv(tables / "run_info.csv", index=False)

    print(f"Analysis complete. Tables written to: {tables}")
    print("Run info:")
    print(run_info.to_string(index=False))


if __name__ == "__main__":
    main()
