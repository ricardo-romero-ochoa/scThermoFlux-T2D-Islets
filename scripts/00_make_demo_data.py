#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
outdir = Path("data/processed")
outdir.mkdir(parents=True, exist_ok=True)

n_donors = {"ND": 6, "T2D": 6}
cells_per_donor = 120

beta_genes = ["INS", "IAPP", "PDX1", "MAFA", "NKX6-1", "SLC2A2", "GABRA2", "PPP1R1A", "ADCYAP1", "ENTPD3"]
stress_genes = ["HSPA5", "XBP1", "DDIT3", "ATF4", "HERPUD1", "IL1R2", "HLA-DRA", "HLA-DPA1", "GBP2", "TNFRSF10A"]
oxphos_genes = ["NDUFA1", "NDUFB5", "SDHA", "UQCRC1", "COX4I1", "ATP5F1A"]
other_genes = [f"GENE{i:03d}" for i in range(80)]
genes = beta_genes + stress_genes + oxphos_genes + other_genes

rows = []
for condition, nd in n_donors.items():
    for d in range(nd):
        donor = f"{condition}_donor_{d+1:02d}"
        donor_shift = rng.normal(0, 0.25)
        for c in range(cells_per_donor):
            # latent disease coordinate: T2D shifted toward stress with overlap
            disease_axis = rng.normal(-0.7 if condition == "ND" else 0.8, 0.8) + donor_shift
            beta_strength = 7.0 - 1.6 * disease_axis + rng.normal(0, 0.2)
            stress_strength = 2.2 + 1.7 * disease_axis + rng.normal(0, 0.2)
            ox_strength = 4.0 - 0.5 * disease_axis + rng.normal(0, 0.2)
            expr = {}
            for g in beta_genes:
                expr[g] = max(0, rng.poisson(np.exp(beta_strength / 2.4)))
            for g in stress_genes:
                expr[g] = max(0, rng.poisson(np.exp(stress_strength / 2.2)))
            for g in oxphos_genes:
                expr[g] = max(0, rng.poisson(np.exp(ox_strength / 2.3)))
            for g in other_genes:
                expr[g] = max(0, rng.poisson(rng.uniform(0.5, 3.0)))
            row = {
                "cell_id": f"{donor}_cell_{c:03d}",
                "dataset": "demo",
                "donor_id": donor,
                "condition": condition,
                "cell_type": "beta",
            }
            row.update(expr)
            rows.append(row)

pd.DataFrame(rows).to_csv(outdir / "demo_t2d_islet_cells.csv", index=False)
print(f"Wrote {outdir / 'demo_t2d_islet_cells.csv'}")
