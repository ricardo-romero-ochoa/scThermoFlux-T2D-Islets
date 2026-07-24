#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from scthermoflux.utils import write_manifest

REQUIRED_TABLES = [
    "cell_state_scores.csv",
    "microstate_table.csv",
    "edge_flux_table.csv",
    "donor_level_thermo.csv",
    "condition_comparison.csv",
]
REQUIRED_FIGURES = [
    "figure_01_landscape.png",
    "figure_02_flux_network.png",
    "figure_03_donor_statistics.png",
]


def main():
    p = argparse.ArgumentParser(description="Audit scThermoFlux outputs.")
    p.add_argument("--results", required=True)
    args = p.parse_args()
    root = Path(args.results)
    lines = ["# Audit report", ""]
    ok = True
    for f in REQUIRED_TABLES:
        path = root / "tables" / f
        exists = path.exists() and path.stat().st_size > 0
        ok &= exists
        lines.append(f"- Table `{f}`: {'OK' if exists else 'MISSING'}")
        if exists:
            try:
                df = pd.read_csv(path)
                lines.append(f"  - rows: {len(df)}, columns: {len(df.columns)}")
            except Exception as exc:
                ok = False
                lines.append(f"  - read error: {exc}")
    for f in REQUIRED_FIGURES:
        path = root / "figures" / f
        exists = path.exists() and path.stat().st_size > 0
        ok &= exists
        lines.append(f"- Figure `{f}`: {'OK' if exists else 'MISSING'}")
    lines.append("")
    lines.append(f"Overall status: {'PASS' if ok else 'FAIL'}")
    report = root / "AUDIT_REPORT.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    write_manifest(root, root / "REPOSITORY_MANIFEST.csv")
    print(report.read_text())
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
