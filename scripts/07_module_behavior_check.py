#!/usr/bin/env python

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind


DEFAULT_MODULES = [
    "BetaIdentitySecretion",
    "ImmuneStress",
    "ERStress_UPR",
    "OxidativePhosphorylation",
    "DedifferentiationStress",
    "DiseaseAxis",
    "disease_axis",
]


def compare_groups(df, value_col, condition_col="condition", a="ND", b="T2D"):
    x = pd.to_numeric(
        df.loc[df[condition_col] == a, value_col],
        errors="coerce",
    ).dropna()

    y = pd.to_numeric(
        df.loc[df[condition_col] == b, value_col],
        errors="coerce",
    ).dropna()

    if len(x) < 2 or len(y) < 2:
        return {
            "metric": value_col,
            "condition_a": a,
            "condition_b": b,
            "n_a": len(x),
            "n_b": len(y),
            "mean_a": np.nan,
            "mean_b": np.nan,
            "median_a": np.nan,
            "median_b": np.nan,
            "delta_b_minus_a": np.nan,
            "mannwhitney_p": np.nan,
            "welch_t_p": np.nan,
        }

    try:
        mw_p = mannwhitneyu(x, y, alternative="two-sided").pvalue
    except Exception:
        mw_p = np.nan

    try:
        tt_p = ttest_ind(x, y, equal_var=False, nan_policy="omit").pvalue
    except Exception:
        tt_p = np.nan

    return {
        "metric": value_col,
        "condition_a": a,
        "condition_b": b,
        "n_a": len(x),
        "n_b": len(y),
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "median_a": float(x.median()),
        "median_b": float(y.median()),
        "delta_b_minus_a": float(y.mean() - x.mean()),
        "mannwhitney_p": float(mw_p) if pd.notna(mw_p) else np.nan,
        "welch_t_p": float(tt_p) if pd.notna(tt_p) else np.nan,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check biological module behavior in scThermoFlux cell-level state scores."
    )
    parser.add_argument(
        "--cell-scores",
        default="results/GSE221156_beta_memorysafe/tables/cell_state_scores.csv",
        help="Path to cell_state_scores.csv",
    )
    parser.add_argument(
        "--outdir",
        default="results/GSE221156_beta_memorysafe/module_behavior",
        help="Output directory",
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=DEFAULT_MODULES,
        help="Module columns to analyze. Missing modules are skipped.",
    )
    parser.add_argument(
        "--condition-a",
        default="ND",
        help="Reference condition",
    )
    parser.add_argument(
        "--condition-b",
        default="T2D",
        help="Comparison condition",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cell = pd.read_csv(args.cell_scores)

    required = ["condition", "donor_id"]
    missing = [c for c in required if c not in cell.columns]
    if missing:
        raise ValueError(f"cell_state_scores.csv is missing required columns: {missing}")

    cell["condition"] = cell["condition"].astype(str)
    cell["donor_id"] = cell["donor_id"].astype(str)

    available_modules = [m for m in args.modules if m in cell.columns]

    if not available_modules:
        numeric_cols = cell.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {
            "microstate",
            "quasi_potential",
            "latent_1",
            "latent_2",
            "latent_3",
            "latent_4",
            "latent_5",
            "latent_6",
            "latent_7",
            "latent_8",
        }
        available_modules = [c for c in numeric_cols if c not in exclude]
        warnings.warn(
            "None of the requested module names were found. "
            f"Using available numeric columns instead: {available_modules}"
        )

    print("Modules to analyze:")
    for m in available_modules:
        print(f"  - {m}")

    for m in available_modules:
        cell[m] = pd.to_numeric(cell[m], errors="coerce")

    # ------------------------------------------------------------------
    # Cell-level descriptive means.
    # These are descriptive only, not the main statistical inference.
    # ------------------------------------------------------------------
    cell_condition_means = (
        cell
        .groupby("condition", observed=True)[available_modules]
        .mean()
        .reset_index()
    )

    cell_condition_path = outdir / "module_cell_means_by_condition.csv"
    cell_condition_means.to_csv(cell_condition_path, index=False)

    # ------------------------------------------------------------------
    # Donor-level module means.
    # This is the correct unit for condition-level inference.
    # ------------------------------------------------------------------
    donor_means = (
        cell
        .groupby(["donor_id", "condition"], observed=True)[available_modules]
        .mean()
        .reset_index()
    )

    donor_means_path = outdir / "module_donor_means.csv"
    donor_means.to_csv(donor_means_path, index=False)

    # ------------------------------------------------------------------
    # Donor-level condition comparisons.
    # ------------------------------------------------------------------
    comparisons = []
    for m in available_modules:
        comparisons.append(
            compare_groups(
                donor_means,
                value_col=m,
                condition_col="condition",
                a=args.condition_a,
                b=args.condition_b,
            )
        )

    comparisons = pd.DataFrame(comparisons)
    comparisons = comparisons.sort_values("delta_b_minus_a", ascending=False)

    comparison_path = outdir / "module_donor_condition_comparison.csv"
    comparisons.to_csv(comparison_path, index=False)

    # ------------------------------------------------------------------
    # Disease-direction sanity check.
    # ------------------------------------------------------------------
    expected = {
        "DiseaseAxis": "increase_in_T2D",
        "disease_axis": "increase_in_T2D",
        "ImmuneStress": "increase_in_T2D",
        "ERStress_UPR": "increase_in_T2D",
        "DedifferentiationStress": "increase_in_T2D",
        "BetaIdentitySecretion": "decrease_in_T2D",
    }

    sanity_rows = []

    for _, row in comparisons.iterrows():
        module = row["metric"]
        delta = row["delta_b_minus_a"]

        if module in expected:
            if expected[module] == "increase_in_T2D":
                expected_direction = "T2D > ND"
                matches = delta > 0
            elif expected[module] == "decrease_in_T2D":
                expected_direction = "T2D < ND"
                matches = delta < 0
            else:
                expected_direction = "not specified"
                matches = np.nan
        else:
            expected_direction = "not specified"
            matches = np.nan

        sanity_rows.append({
            "module": module,
            "delta_T2D_minus_ND": delta,
            "expected_direction": expected_direction,
            "matches_expectation": matches,
        })

    sanity = pd.DataFrame(sanity_rows)
    sanity_path = outdir / "module_direction_sanity_check.csv"
    sanity.to_csv(sanity_path, index=False)

    # ------------------------------------------------------------------
    # Markdown report.
    # ------------------------------------------------------------------
    report_path = outdir / "module_behavior_report.md"

    lines = []
    lines.append("# Biological module behavior report")
    lines.append("")
    lines.append(f"Input cell scores: `{args.cell_scores}`")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- Cell-level means are descriptive only.")
    lines.append("- Donor-level means are the appropriate unit for ND vs T2D comparisons.")
    lines.append("- `delta_b_minus_a > 0` means the module is higher in T2D than ND.")
    lines.append("- `delta_b_minus_a < 0` means the module is lower in T2D than ND.")
    lines.append("")
    lines.append("## Cell-level descriptive means")
    lines.append("")
    lines.append(cell_condition_means.to_markdown(index=False))
    lines.append("")
    lines.append("## Donor-level condition comparison")
    lines.append("")
    show_cols = [
        "metric",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "delta_b_minus_a",
        "mannwhitney_p",
        "welch_t_p",
    ]
    lines.append(comparisons[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Direction sanity check")
    lines.append("")
    lines.append(sanity.to_markdown(index=False))
    lines.append("")
    lines.append("## Suggested interpretation")
    lines.append("")
    lines.append("A biologically coherent T2D beta-cell result should generally show:")
    lines.append("")
    lines.append("- higher DiseaseAxis in T2D;")
    lines.append("- higher ImmuneStress and/or ERStress_UPR in T2D;")
    lines.append("- lower BetaIdentitySecretion in T2D;")
    lines.append("- possible increase in DedifferentiationStress;")
    lines.append("- OxidativePhosphorylation may decrease, increase, or become heterogeneous depending on donor state and module definition.")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {cell_condition_path}")
    print(f"Wrote: {donor_means_path}")
    print(f"Wrote: {comparison_path}")
    print(f"Wrote: {sanity_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()