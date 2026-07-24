#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Make package importable when running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scthermoflux.dataio import load_expression, subset_cells
from scthermoflux.preprocess import normalize_log1p, select_hvg, compute_pca
from scthermoflux.modules import load_modules, score_modules, add_disease_axis
from scthermoflux.landscape import knn_density, quasi_potential_from_density, make_microstates, summarize_microstates
from scthermoflux.transitions import build_transition_matrix, stationary_distribution
from scthermoflux.thermo import edgewise_flux_table, entropy_production, condition_level_entropy
from scthermoflux.stats import donor_level_statistics, compare_conditions
from scthermoflux.utils import ensure_dir


def main():
    p = argparse.ArgumentParser(description="Run transcriptomic landscape/flux/entropy analysis.")
    p.add_argument("--input", required=True, help="Standardized .h5ad or CSV/TSV cell table")
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
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    outdir = ensure_dir(args.outdir)
    tables = ensure_dir(outdir / "tables")

    X, meta, genes = load_expression(args.input)
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    X, meta, genes = subset_cells(X, meta, genes, args.cell_type, conditions)
    meta["dataset"] = args.dataset

    Xn = normalize_log1p(X)
    modules = load_modules(args.modules)
    module_scores = add_disease_axis(score_modules(Xn, genes, modules))
    module_scores.to_csv(tables / "cell_module_scores.csv", index=False)

    Xh, hvg_genes, hvg_idx = select_hvg(Xn, genes, n_top=args.n_hvg)
    latent = compute_pca(Xh, n_components=args.n_latent, random_state=args.seed)
    pd.DataFrame(latent, columns=[f"latent_{i+1}" for i in range(latent.shape[1])]).to_csv(tables / "cell_latent_coordinates.csv", index=False)

    density = knn_density(latent[:, :2], k=args.density_k)
    phi = quasi_potential_from_density(density)
    microstates, centers = make_microstates(latent[:, :min(5, latent.shape[1])], n_microstates=args.n_microstates, random_state=args.seed)

    cell_state = pd.concat([meta.reset_index(drop=True), module_scores.reset_index(drop=True)], axis=1)
    cell_state["microstate"] = microstates
    cell_state["quasi_potential"] = phi
    cell_state["latent_1"] = latent[:, 0]
    cell_state["latent_2"] = latent[:, 1] if latent.shape[1] > 1 else 0.0
    cell_state.to_csv(tables / "cell_state_scores.csv", index=False)

    microstate_table = summarize_microstates(latent, meta, microstates, phi, disease_axis=module_scores["DiseaseAxis"].to_numpy())
    microstate_table.to_csv(tables / "microstate_table.csv", index=False)

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

    edge_table = edgewise_flux_table(K, pi, microstate_table)
    edge_table.to_csv(tables / "edge_flux_table.csv", index=False)
    pd.DataFrame({"global_entropy_production": [entropy_production(K, pi)]}).to_csv(tables / "global_thermo.csv", index=False)

    condition_thermo = condition_level_entropy(meta, microstates, K)
    condition_thermo.to_csv(tables / "condition_level_thermo.csv", index=False)

    donor_df = donor_level_statistics(meta, microstates, phi, module_scores["DiseaseAxis"].to_numpy(), K)
    donor_df.to_csv(tables / "donor_level_thermo.csv", index=False)
    comp = compare_conditions(donor_df, condition_a=conditions[0], condition_b=conditions[1] if len(conditions) > 1 else conditions[0])
    comp.to_csv(tables / "condition_comparison.csv", index=False)

    print(f"Analysis complete. Tables written to: {tables}")


if __name__ == "__main__":
    main()
