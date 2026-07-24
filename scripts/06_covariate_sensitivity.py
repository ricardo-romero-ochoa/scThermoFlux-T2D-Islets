#!/usr/bin/env python

from pathlib import Path
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


DEFAULT_METRICS = [
    "disease_axis_mean",
    "disease_axis_median",
    "quasi_potential_mean",
    "quasi_potential_median",
    "state_entropy",
    "entropy_production",
]


def clean_numeric(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def usable_numeric_covariate(df, col, min_nonmissing=15):
    if col not in df.columns:
        return False
    x = pd.to_numeric(df[col], errors="coerce")
    return x.notna().sum() >= min_nonmissing and x.nunique(dropna=True) > 1


def usable_categorical_covariate(df, col, min_nonmissing=15):
    if col not in df.columns:
        return False
    x = df[col].astype(str).replace({"nan": np.nan, "None": np.nan})
    return x.notna().sum() >= min_nonmissing and x.nunique(dropna=True) > 1


def fit_model(df, metric, formula_rhs, model_name):
    formula = f"{metric} ~ {formula_rhs}"

    needed_cols = [metric, "condition"]
    for token in ["age_years", "BMI", "HbA1c", "sex", "assay", "batch_assay"]:
        if token in formula:
            needed_cols.append(token)

    needed_cols = [c for c in needed_cols if c in df.columns]
    sub = df[needed_cols].copy()

    for col in ["age_years", "BMI", "HbA1c"]:
        if col in sub.columns:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")

    sub = sub.dropna()

    if sub["condition"].nunique() < 2:
        return None

    if len(sub) < 8:
        return None

    try:
        model = smf.ols(formula, data=sub).fit(cov_type="HC3")
    except Exception as exc:
        warnings.warn(f"Could not fit {model_name} for {metric}: {exc}")
        return None

    condition_terms = [p for p in model.params.index if "condition" in p and "T2D" in p]

    if condition_terms:
        term = condition_terms[0]
        beta = model.params[term]
        pval = model.pvalues[term]
        ci_low, ci_high = model.conf_int().loc[term].tolist()
    else:
        beta = np.nan
        pval = np.nan
        ci_low = np.nan
        ci_high = np.nan

    return {
        "metric": metric,
        "model": model_name,
        "formula": formula,
        "n_donors_used": int(model.nobs),
        "r_squared": model.rsquared,
        "condition_T2D_beta": beta,
        "condition_T2D_p": pval,
        "condition_T2D_ci_low": ci_low,
        "condition_T2D_ci_high": ci_high,
        "aic": model.aic,
        "bic": model.bic,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run donor-level covariate sensitivity models for scThermoFlux results."
    )
    parser.add_argument(
        "--donor-table",
        default="results/GSE221156_beta_memorysafe/tables/donor_level_thermo.csv",
        help="Path to donor_level_thermo.csv",
    )
    parser.add_argument(
        "--metadata",
        default="data/metadata/GSE221156_beta_donor_summary.tsv",
        help="Path to donor-level metadata TSV",
    )
    parser.add_argument(
        "--outdir",
        default="results/GSE221156_beta_memorysafe/covariate_sensitivity",
        help="Output directory",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metrics to analyze",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    donor_table = pd.read_csv(args.donor_table)

    if "condition" not in donor_table.columns:
        raise ValueError("donor_level_thermo.csv must contain a 'condition' column.")

    if "donor_id" not in donor_table.columns:
        raise ValueError("donor_level_thermo.csv must contain a 'donor_id' column.")

    donor_table["condition"] = pd.Categorical(
        donor_table["condition"],
        categories=["ND", "T2D"],
        ordered=True,
    )

    if Path(args.metadata).exists():
        metadata = pd.read_csv(args.metadata, sep="\t")
        df = donor_table.merge(
            metadata,
            on=["donor_id", "condition"],
            how="left",
            suffixes=("", "_meta"),
        )
    else:
        warnings.warn(f"Metadata file not found: {args.metadata}. Running unadjusted models only.")
        df = donor_table.copy()

    df = clean_numeric(df, ["age_years", "BMI", "HbA1c", "n_cells"])

    # Prefer assay if available; otherwise use batch_assay.
    assay_col = None
    if usable_categorical_covariate(df, "assay"):
        assay_col = "assay"
    elif usable_categorical_covariate(df, "batch_assay"):
        assay_col = "batch_assay"

    rows = []

    for metric in args.metrics:
        if metric not in df.columns:
            warnings.warn(f"Metric not found and will be skipped: {metric}")
            continue

        df[metric] = pd.to_numeric(df[metric], errors="coerce")

        # Model 1: unadjusted
        result = fit_model(
            df=df,
            metric=metric,
            formula_rhs="C(condition)",
            model_name="unadjusted",
        )
        if result is not None:
            rows.append(result)

        # Model 2: demographic / technical sensitivity, excluding HbA1c
        covariates = []

        if usable_numeric_covariate(df, "age_years"):
            covariates.append("age_years")

        if usable_numeric_covariate(df, "BMI"):
            covariates.append("BMI")

        if usable_categorical_covariate(df, "sex"):
            covariates.append("C(sex)")

        if assay_col is not None:
            covariates.append(f"C({assay_col})")

        if covariates:
            rhs = "C(condition) + " + " + ".join(covariates)
            result = fit_model(
                df=df,
                metric=metric,
                formula_rhs=rhs,
                model_name="adjusted_age_BMI_sex_assay",
            )
            if result is not None:
                rows.append(result)

        # Model 3: HbA1c sensitivity.
        # HbA1c is partly downstream of T2D, so this is not the primary model.
        covariates_hba1c = covariates.copy()

        if usable_numeric_covariate(df, "HbA1c"):
            covariates_hba1c.append("HbA1c")

        if covariates_hba1c:
            rhs = "C(condition) + " + " + ".join(covariates_hba1c)
            result = fit_model(
                df=df,
                metric=metric,
                formula_rhs=rhs,
                model_name="HbA1c_sensitivity",
            )
            if result is not None:
                rows.append(result)

    results = pd.DataFrame(rows)

    results_path = outdir / "covariate_sensitivity_models.csv"
    results.to_csv(results_path, index=False)

    merged_path = outdir / "donor_thermo_with_metadata.csv"
    df.to_csv(merged_path, index=False)

    report_path = outdir / "covariate_sensitivity_report.md"

    lines = []
    lines.append("# Covariate sensitivity report")
    lines.append("")
    lines.append(f"Input donor table: `{args.donor_table}`")
    lines.append(f"Metadata table: `{args.metadata}`")
    lines.append("")
    lines.append("## Interpretation guide")
    lines.append("")
    lines.append("- `condition_T2D_beta > 0`: metric is higher in T2D than ND.")
    lines.append("- `condition_T2D_beta < 0`: metric is lower in T2D than ND.")
    lines.append("- The unadjusted model is the primary disease comparison.")
    lines.append("- The HbA1c model is a sensitivity analysis, not the primary model, because HbA1c is partly downstream of T2D.")
    lines.append("")
    lines.append("## Model results")
    lines.append("")

    if not results.empty:
        show_cols = [
            "metric",
            "model",
            "n_donors_used",
            "condition_T2D_beta",
            "condition_T2D_ci_low",
            "condition_T2D_ci_high",
            "condition_T2D_p",
            "r_squared",
        ]
        lines.append(results[show_cols].to_markdown(index=False))
    else:
        lines.append("No models were successfully fitted.")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote: {results_path}")
    print(f"Wrote: {merged_path}")
    print(f"Wrote: {report_path}")


if __name__ == "__main__":
    main()