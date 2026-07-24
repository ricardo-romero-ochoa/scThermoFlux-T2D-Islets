import re
from pathlib import Path

import pandas as pd
import scanpy as sc


# ---------------------------------------------------------------------
# Input / output paths
# ---------------------------------------------------------------------
infile = "C:/Users/ricar/Desktop/scThermo/data/processed/GSE221156_beta_cellxgene_raw.h5ad"
outfile = "C:/Users/ricar/Desktop/scThermo/data/processed/GSE221156_beta_standardized.h5ad"
summary_out = "C:/Users/ricar/Desktop/scThermo/data/metadata/GSE221156_beta_donor_summary.tsv"

Path(outfile).parent.mkdir(parents=True, exist_ok=True)
Path(summary_out).parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
adata = sc.read_h5ad(infile)

print("Original object:")
print(adata)

print("\nAvailable obs columns:")
print(adata.obs.columns.tolist())

print("\nAvailable var columns:")
print(adata.var.columns.tolist())

print("\nDisease labels:")
print(adata.obs["disease"].value_counts(dropna=False))

print("\nCell types:")
print(adata.obs["cell_type"].value_counts(dropna=False))

print("\nAssays:")
print(adata.obs["assay"].value_counts(dropna=False))


# ---------------------------------------------------------------------
# Preserve original gene identifiers
# ---------------------------------------------------------------------
adata.var["ensembl_id"] = adata.var_names.astype(str)

# Use gene symbols as var_names when available.
# CELLxGENE stores Ensembl IDs as var_names and gene symbols in feature_name.
if "feature_name" in adata.var.columns:
    gene_symbols = adata.var["feature_name"].astype(str)
    gene_symbols = gene_symbols.replace({"nan": ""})
    gene_symbols = gene_symbols.where(gene_symbols != "", adata.var["ensembl_id"])

    adata.var["gene_symbol_original"] = gene_symbols.values
    adata.var_names = gene_symbols.values
    adata.var_names_make_unique()

    # Critical fix:
    # Prevent AnnData write_h5ad error when var.index.name conflicts with
    # an existing column such as feature_name.
    adata.var.index.name = "gene_symbol"


# ---------------------------------------------------------------------
# Required standardized metadata
# ---------------------------------------------------------------------
required_obs = ["donor_id", "disease", "cell_type"]
missing = [c for c in required_obs if c not in adata.obs.columns]

if missing:
    raise ValueError(f"Missing required obs columns: {missing}")

adata.obs["dataset"] = "GSE221156"
adata.obs["donor_id"] = adata.obs["donor_id"].astype(str)

adata.obs["cell_type_original"] = adata.obs["cell_type"].astype(str)
adata.obs["cell_type"] = "beta"

adata.obs["condition_original"] = adata.obs["disease"].astype(str).str.strip()

disease_map = {
    "normal": "ND",
    "Normal": "ND",
    "non-diabetic": "ND",
    "non diabetic": "ND",
    "healthy": "ND",
    "type 2 diabetes mellitus": "T2D",
    "type II diabetes mellitus": "T2D",
    "T2D": "T2D",
    "pre-diabetes mellitus": "PD",
    "prediabetes": "PD",
    "pre-diabetic": "PD",
    "PD": "PD",
}

adata.obs["condition"] = (
    adata.obs["condition_original"]
    .map(disease_map)
    .fillna(adata.obs["condition_original"])
)


# ---------------------------------------------------------------------
# Clean numerical covariates
# ---------------------------------------------------------------------
numeric_cols = [
    "BMI",
    "HbA1c",
    "nCount_RNA",
    "nFeature_RNA",
    "percent.mt",
    "Viability",
    "Purity",
    "Viability_2",
    "Purity_2",
]

for col in numeric_cols:
    if col in adata.obs.columns:
        adata.obs[col] = pd.to_numeric(adata.obs[col], errors="coerce")


# Extract numerical age from strings such as "51-year-old stage"
if "development_stage" in adata.obs.columns:
    adata.obs["age_years"] = (
        adata.obs["development_stage"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
    )
    adata.obs["age_years"] = pd.to_numeric(adata.obs["age_years"], errors="coerce")
else:
    adata.obs["age_years"] = pd.NA


# ---------------------------------------------------------------------
# Keep useful batch/source columns with standardized names
# ---------------------------------------------------------------------
if "assay" in adata.obs.columns:
    adata.obs["batch_assay"] = adata.obs["assay"].astype(str)
else:
    adata.obs["batch_assay"] = "unknown"

if "LibraryID" in adata.obs.columns:
    adata.obs["library_id"] = adata.obs["LibraryID"].astype(str)
else:
    adata.obs["library_id"] = "unknown"

if "Center" in adata.obs.columns:
    adata.obs["center"] = adata.obs["Center"].astype(str)
else:
    adata.obs["center"] = "unknown"


# ---------------------------------------------------------------------
# Main analysis subset: ND vs T2D beta cells
# ---------------------------------------------------------------------
adata_main = adata[adata.obs["condition"].isin(["ND", "T2D"])].copy()

if adata_main.n_obs == 0:
    raise ValueError(
        "No cells remained after filtering for condition in ['ND', 'T2D']. "
        "Inspect adata.obs['condition_original'].value_counts()."
    )


# ---------------------------------------------------------------------
# Optional removal of genes flagged by CELLxGENE as filtered
# ---------------------------------------------------------------------
if "feature_is_filtered" in adata_main.var.columns:
    filtered = adata_main.var["feature_is_filtered"].astype(bool)
    adata_main = adata_main[:, ~filtered].copy()


# ---------------------------------------------------------------------
# Ensure unique names and safe index names for h5ad writing
# ---------------------------------------------------------------------
adata_main.obs_names_make_unique()
adata_main.var_names_make_unique()

adata_main.obs.index.name = "cell_id"
adata_main.var.index.name = "gene_symbol"


# ---------------------------------------------------------------------
# Donor-level summary
# ---------------------------------------------------------------------
summary = (
    adata_main.obs
    .groupby(["donor_id", "condition"], observed=True)
    .agg(
        n_cells=("condition", "size"),
        sex=("sex", "first") if "sex" in adata_main.obs.columns else ("condition", "first"),
        age_years=("age_years", "first"),
        BMI=("BMI", "first") if "BMI" in adata_main.obs.columns else ("condition", "first"),
        HbA1c=("HbA1c", "first") if "HbA1c" in adata_main.obs.columns else ("condition", "first"),
        assay=("assay", lambda x: ";".join(sorted(set(map(str, x)))))
        if "assay" in adata_main.obs.columns
        else ("condition", "first"),
        library_ids=("library_id", lambda x: ";".join(sorted(set(map(str, x))))),
    )
    .reset_index()
    .sort_values(["condition", "donor_id"])
)

summary.to_csv(summary_out, sep="\t", index=False)


# ---------------------------------------------------------------------
# Save standardized AnnData object
# ---------------------------------------------------------------------
adata_main.write_h5ad(outfile, compression="gzip")


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------
print("\nStandardized object:")
print(adata_main)

print("\nCondition counts:")
print(adata_main.obs["condition"].value_counts())

print("\nNumber of donors by condition:")
print(
    adata_main.obs[["donor_id", "condition"]]
    .drop_duplicates()["condition"]
    .value_counts()
)

print("\nDonor-level summary:")
print(summary)

print(f"\nWrote standardized h5ad: {outfile}")
print(f"Wrote donor summary: {summary_out}")