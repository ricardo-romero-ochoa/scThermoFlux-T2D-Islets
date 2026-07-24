# scThermoFlux-T2D-Islets

A reproducible research repository for the manuscript concept:

> **A nonequilibrium thermodynamic framework for transcriptomic disease-state transitions in type 2 diabetes pancreatic islets**

This repo implements a conservative, coarse-grained thermodynamic analysis of single-cell transcriptomic disease states. It treats the transcriptomic “landscape” as a **dimensionless quasi-potential** inferred from cell-state probability density, not as directly measured physical free energy. The main outputs are landscape deformation, graph-based transition structure, probability currents, detailed-balance violation, and entropy-production-like irreversibility scores.

## Core claims supported by this repo

1. T2D islet cell states can be represented on a transcriptomic manifold.
2. Disease-associated changes can be decomposed into:
   - state occupancy shifts,
   - quasi-potential changes,
   - inferred transition asymmetry,
   - probability currents,
   - entropy-production-like irreversibility.
3. Thermodynamic observables should be aggregated at the **donor level**, not treated as independent cell-level p-values.
4. Cross-sectional data support **inferred transition structure**, not direct longitudinal progression.

## Recommended dataset hierarchy

The repository is configured around public human islet datasets:

| role | accession | reason |
|---|---|---|
| Primary | GSE221156 | Largest T2D islet scRNA-seq atlas; 245,878 cells from 48 donors: 17 ND, 14 PD, 17 T2D. |
| Validation | GSE153855 | High-quality Smart-seq2 dataset; 6 control and 5 T2D donors. |
| Validation | E-MTAB-5061 | Segerstolpe/Palasantza human pancreas/islet Smart-seq2 resource; 2,942 cells, healthy and T2D. |
| Validation | GSE81608 | Xin et al. single-cell endocrine islet dataset; 1,492 annotated α, β, δ, PP cells from ND and T2D donors. |
| Validation | GSE86469 | Lawlor et al.; 638 single cells from 5 ND and 3 T2D donors. |
| Cautious validation | GSE83139 | Includes juvenile, adult control, T1D and T2D donors; useful but confounded by donor class and age. |
| Bulk module anchor | GSE164416 | Whole-islet RNA-seq, 133 donors across ND/IGT/T3cD/T2D; useful for module validation, not single-cell flux. |

See `data/metadata/dataset_inventory.tsv` and `docs/dataset_notes.md` for details.

## What is included

```text
scThermoFlux_T2D_islets/
├── config/                         # YAML configuration files
├── data/
│   ├── metadata/                   # dataset inventory and module gene sets
│   ├── raw/                        # local raw/processed external datasets, not versioned
│   └── processed/                  # standardized h5ad/csv outputs, not versioned
├── docs/                           # dataset and method notes
├── manuscript/                     # manuscript outline, captions, methods draft
├── notebooks/                      # placeholder notebooks with execution order
├── results/                        # output tables, figures, models
├── scripts/                        # command-line entry points
├── src/scthermoflux/               # reusable Python package
└── tests/                          # minimal smoke tests
```

## Quick start: fully reproducible demo

The demo creates synthetic control/T2D beta-cell-like data and runs the full landscape-flux-entropy workflow.

```bash
conda env create -f environment.yml
conda activate scthermoflux
python scripts/00_make_demo_data.py
python scripts/02_run_landscape_flux.py \
  --input data/processed/demo_t2d_islet_cells.csv \
  --dataset demo \
  --outdir results/demo
python scripts/03_make_figures.py \
  --tables results/demo/tables \
  --outdir results/demo/figures
python scripts/04_audit_outputs.py --results results/demo
```

Expected outputs:

```text
results/demo/tables/cell_state_scores.csv
results/demo/tables/microstate_table.csv
results/demo/tables/edge_flux_table.csv
results/demo/tables/donor_level_thermo.csv
results/demo/tables/condition_comparison.csv
results/demo/figures/figure_01_landscape.png
results/demo/figures/figure_02_flux_network.png
results/demo/figures/figure_03_donor_statistics.png
results/demo/AUDIT_REPORT.md
```

## Running on real single-cell data

The repository accepts either:

1. a standardized `.h5ad` file, or
2. a simple cell-by-gene `.csv`/`.tsv` table with metadata columns.

For real datasets, first standardize your object so that `obs` contains:

| column | required | meaning |
|---|---:|---|
| `donor_id` | yes | biological donor |
| `condition` | yes | `ND`, `PD`, `T2D`, or other |
| `cell_type` | yes | beta, alpha, delta, etc. |
| `dataset` | recommended | accession or cohort label |
| `sex`, `age`, `BMI`, `HbA1c` | optional | covariates |

Then run:

```bash
python scripts/01_standardize_h5ad.py \
  --input data/raw/GSE221156.h5ad \
  --output data/processed/GSE221156_standardized.h5ad \
  --dataset GSE221156 \
  --condition-col disease_state \
  --donor-col donor_id \
  --celltype-col cell_type

python scripts/02_run_landscape_flux.py \
  --input data/processed/GSE221156_standardized.h5ad \
  --dataset GSE221156 \
  --cell-type beta \
  --conditions ND,T2D \
  --outdir results/GSE221156_beta
```

## Method summary

For each cell, the workflow computes curated biological module scores and a disease-axis coordinate:

\[
\text{DiseaseAxis}
= z(\text{ImmuneStress}) + z(\text{ERStress}) - z(\text{BetaIdentitySecretion}).
\]

A low-dimensional state representation is built from PCA or provided latent coordinates. Cell-state density is estimated and converted into a dimensionless quasi-potential:

\[
\Phi(x) = -\log[P(x)+\epsilon].
\]

Microstates are constructed using k-means. A Markov kernel is inferred on the microstate graph with optional bias along the disease axis. The probability current is:

\[
J_{ij} = \pi_iK_{ij} - \pi_jK_{ji},
\]

and the entropy-production-like score is:

\[
\sigma = \frac{1}{2}\sum_{i,j} J_{ij}\log\frac{\pi_iK_{ij}+\epsilon}{\pi_jK_{ji}+\epsilon}.
\]

## Important interpretation limits

- `quasi_potential` is not physical free energy.
- `entropy_production` is a coarse-grained irreversibility proxy, not total heat dissipation.
- Cross-sectional ND/T2D data do not prove chronological progression.
- RNA velocity is not required for this first version. If spliced/unspliced layers are available, velocity-based transition matrices can be added as a later module.
- Donor-level aggregation is mandatory for valid biological inference.

## Suggested manuscript figures

- **Figure 1:** Conceptual framework: DE versus landscape-flux thermodynamics.
- **Figure 2:** Synthetic validation: equilibrium, driven, cyclic, and null systems.
- **Figure 3:** GSE221156 beta-cell quasi-potential landscape.
- **Figure 4:** Disease-associated probability currents and detailed-balance violation.
- **Figure 5:** Donor-level thermodynamic statistics and robustness controls.
- **Figure 6:** Biological module attribution of high-flux transitions.

## License

MIT License. See `LICENSE`.
