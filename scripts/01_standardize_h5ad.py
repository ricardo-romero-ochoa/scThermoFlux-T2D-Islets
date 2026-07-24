#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Standardize h5ad obs columns for scThermoFlux.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--condition-col", required=True)
    parser.add_argument("--donor-col", required=True)
    parser.add_argument("--celltype-col", required=True)
    parser.add_argument("--sex-col", default=None)
    parser.add_argument("--age-col", default=None)
    parser.add_argument("--bmi-col", default=None)
    parser.add_argument("--hba1c-col", default=None)
    args = parser.parse_args()

    try:
        import anndata as ad
    except ImportError as exc:
        raise SystemExit("Install anndata to use this script.") from exc

    adata = ad.read_h5ad(args.input)
    obs = adata.obs.copy()
    colmap = {
        args.condition_col: "condition",
        args.donor_col: "donor_id",
        args.celltype_col: "cell_type",
    }
    optional = [(args.sex_col, "sex"), (args.age_col, "age"), (args.bmi_col, "BMI"), (args.hba1c_col, "HbA1c")]
    for src, dst in optional:
        if src and src in obs.columns:
            colmap[src] = dst
    missing = [c for c in colmap if c not in obs.columns]
    if missing:
        raise SystemExit(f"Missing columns in input h5ad: {missing}")
    obs = obs.rename(columns=colmap)
    obs["dataset"] = args.dataset
    adata.obs = obs
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(args.output)
    print(f"Wrote standardized h5ad: {args.output}")


if __name__ == "__main__":
    main()
