# Real-data harmonization template

For each real dataset, create a standardized object with:

```text
obs/cell metadata:
- cell_id
- dataset
- donor_id
- condition: ND, PD, T2D, T1D, etc.
- cell_type: beta, alpha, delta, gamma/PP, endothelial, immune, acinar, ductal, stellate
- sex, age, BMI, HbA1c if available
```

Recommended harmonization rules:

1. Map `healthy`, `control`, `non-diabetic`, `ND` to `ND`.
2. Map `type 2 diabetes`, `T2D`, `diabetic` to `T2D` only when metadata are unambiguous.
3. Keep `PD` or `prediabetic` separate; do not merge with ND or T2D in the main analysis.
4. Use only beta cells for the primary test.
5. Run donor-level summaries before any p-values.
6. Record excluded donors/cells in `results/<dataset>/tables/exclusion_log.csv`.

Example command:

```bash
python scripts/01_standardize_h5ad.py \
  --input data/raw/GSE221156.h5ad \
  --output data/processed/GSE221156_standardized.h5ad \
  --dataset GSE221156 \
  --condition-col disease_state \
  --donor-col donor_id \
  --celltype-col cell_type
```
