from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

METADATA_COLUMNS = {
    "cell_id", "dataset", "donor_id", "condition", "cell_type",
    "sex", "age", "BMI", "HbA1c", "batch"
}


def load_cell_table(path: str | Path):
    """Load a simple cell-by-gene CSV/TSV table.

    Required metadata columns: donor_id, condition, cell_type.
    Gene columns are all non-metadata numeric columns.
    """
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)
    required = {"donor_id", "condition", "cell_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns in {path}: {sorted(missing)}")
    gene_cols = [c for c in df.columns if c not in METADATA_COLUMNS and pd.api.types.is_numeric_dtype(df[c])]
    if len(gene_cols) < 5:
        raise ValueError("Too few numeric gene columns detected. Check input format.")
    meta = df[[c for c in df.columns if c in METADATA_COLUMNS]].copy()
    if "cell_id" not in meta:
        meta["cell_id"] = [f"cell_{i}" for i in range(len(df))]
    X = df[gene_cols].astype(float).to_numpy()
    return X, meta, gene_cols


def load_h5ad(path: str | Path):
    """Load an AnnData file if anndata is installed."""
    try:
        import anndata as ad
    except ImportError as exc:
        raise ImportError("Install anndata to load .h5ad files.") from exc
    adata = ad.read_h5ad(path)
    required = {"donor_id", "condition", "cell_type"}
    missing = required - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing required obs columns in h5ad: {sorted(missing)}")
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    gene_cols = list(map(str, adata.var_names))
    meta = adata.obs.reset_index().rename(columns={"index": "cell_id"})
    if "cell_id" not in meta:
        meta["cell_id"] = meta.iloc[:, 0].astype(str)
    return np.asarray(X, dtype=float), meta, gene_cols


def load_expression(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".h5ad":
        return load_h5ad(path)
    return load_cell_table(path)


def subset_cells(X, meta: pd.DataFrame, genes, cell_type: str | None = None, conditions: list[str] | None = None):
    mask = np.ones(len(meta), dtype=bool)
    if cell_type and cell_type.lower() != "all":
        mask &= meta["cell_type"].astype(str).str.lower().eq(cell_type.lower()).to_numpy()
    if conditions:
        wanted = {c.strip() for c in conditions}
        mask &= meta["condition"].astype(str).isin(wanted).to_numpy()
    if mask.sum() < 20:
        raise ValueError(f"Subset contains only {mask.sum()} cells; too few for analysis.")
    return X[mask], meta.loc[mask].reset_index(drop=True), genes
