#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scthermoflux.plotting import plot_landscape, plot_flux_network, plot_donor_stats
from scthermoflux.utils import ensure_dir


def main():
    p = argparse.ArgumentParser(description="Create publication-style figures from scThermoFlux tables.")
    p.add_argument("--tables", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    tables = Path(args.tables)
    outdir = ensure_dir(args.outdir)

    cell = pd.read_csv(tables / "cell_state_scores.csv")
    micro = pd.read_csv(tables / "microstate_table.csv")
    edges = pd.read_csv(tables / "edge_flux_table.csv")
    donor = pd.read_csv(tables / "donor_level_thermo.csv")
    latent = cell[["latent_1", "latent_2"]].to_numpy()
    plot_landscape(latent, cell, cell["quasi_potential"].to_numpy(), outdir / "figure_01_landscape.png")
    plot_flux_network(micro, edges, outdir / "figure_02_flux_network.png")
    plot_donor_stats(donor, outdir / "figure_03_donor_statistics.png")
    print(f"Figures written to {outdir}")


if __name__ == "__main__":
    main()
