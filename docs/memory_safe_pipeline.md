# Memory-safe pipeline for GSE221156 beta cells

The original `scripts/02_run_landscape_flux.py` loads `.h5ad` data through `src/scthermoflux/dataio.py`, where sparse `adata.X` is converted to a dense NumPy array with `X.toarray()`. For GSE221156 beta cells this can require >13 GB for the expression matrix alone and can cause the process to be killed.

Use:

```bash
python scripts/02_run_landscape_flux_memorysafe.py \
  --input data/processed/GSE221156_beta_standardized.h5ad \
  --dataset GSE221156 \
  --cell-type beta \
  --conditions ND,T2D \
  --min-cells-per-donor 250 \
  --max-cells-per-donor 1000 \
  --n-hvg 1500 \
  --n-latent 8 \
  --n-microstates 20 \
  --density-k 20 \
  --knn-k 8 \
  --outdir results/GSE221156_beta_memorysafe
```

Then generate figures:

```bash
python scripts/03_make_figures.py \
  --tables results/GSE221156_beta_memorysafe/tables \
  --outdir results/GSE221156_beta_memorysafe/figures
```

For stronger machines, increase `--max-cells-per-donor` to 1500 or set it to 0 to disable downsampling. For laptops, 500-1000 cells per donor is recommended.
