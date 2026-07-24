#!/usr/bin/env bash
set -euo pipefail
python scripts/00_make_demo_data.py
python scripts/02_run_landscape_flux.py \
  --input data/processed/demo_t2d_islet_cells.csv \
  --dataset demo \
  --cell-type beta \
  --conditions ND,T2D \
  --outdir results/demo
python scripts/03_make_figures.py \
  --tables results/demo/tables \
  --outdir results/demo/figures
python scripts/04_audit_outputs.py --results results/demo
