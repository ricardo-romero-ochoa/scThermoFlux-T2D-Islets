from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def read_table(root, name):
    path = Path(root) / "tables" / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return np.nan, np.nan
    if method == "spearman":
        return spearmanr(x[ok], y[ok])
    return pearsonr(x[ok], y[ok])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--original", required=True)
    p.add_argument("--memorysafe", required=True)
    p.add_argument("--out", default="results/pipeline_comparison_report.md")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("# scThermoFlux pipeline comparison")
    lines.append("")
    lines.append(f"- Original: `{args.original}`")
    lines.append(f"- Memory-safe: `{args.memorysafe}`")
    lines.append("")

    # ------------------------------------------------------------------
    # Donor-level comparison
    # ------------------------------------------------------------------
    d1 = read_table(args.original, "donor_level_thermo.csv")
    d2 = read_table(args.memorysafe, "donor_level_thermo.csv")

    key = ["donor_id", "condition"]
    merged = d1.merge(d2, on=key, suffixes=("_original", "_memorysafe"))

    lines.append("## Donor-level comparison")
    lines.append("")
    lines.append(f"- Donors in original: {d1['donor_id'].nunique()}")
    lines.append(f"- Donors in memory-safe: {d2['donor_id'].nunique()}")
    lines.append(f"- Matched donors: {merged['donor_id'].nunique()}")
    lines.append("")

    metrics = [
        "entropy_production",
        "state_entropy",
        "quasi_potential_mean",
        "quasi_potential_median",
        "disease_axis_mean",
        "disease_axis_median",
    ]

    rows = []
    for m in metrics:
        a = f"{m}_original"
        b = f"{m}_memorysafe"
        if a in merged.columns and b in merged.columns:
            rho, sp = safe_corr(merged[a], merged[b], "spearman")
            r, pp = safe_corr(merged[a], merged[b], "pearson")
            rows.append({
                "metric": m,
                "spearman_rho": rho,
                "spearman_p": sp,
                "pearson_r": r,
                "pearson_p": pp,
                "mean_original": merged[a].mean(),
                "mean_memorysafe": merged[b].mean(),
                "mean_difference_memorysafe_minus_original": (merged[b] - merged[a]).mean(),
            })

    corr_df = pd.DataFrame(rows)
    corr_path = out.parent / "pipeline_donor_metric_correlations.csv"
    corr_df.to_csv(corr_path, index=False)

    lines.append("Donor-level metric correlations:")
    lines.append("")
    lines.append(corr_df.to_markdown(index=False))
    lines.append("")

    # ------------------------------------------------------------------
    # Condition-level comparison
    # ------------------------------------------------------------------
    c1 = read_table(args.original, "condition_comparison.csv")
    c2 = read_table(args.memorysafe, "condition_comparison.csv")

    cc = c1.merge(c2, on="metric", suffixes=("_original", "_memorysafe"))
    cc_path = out.parent / "pipeline_condition_comparison_merged.csv"
    cc.to_csv(cc_path, index=False)

    lines.append("## Condition-level comparison")
    lines.append("")
    show_cols = [
        "metric",
        "delta_b_minus_a_original",
        "mannwhitney_p_original",
        "welch_t_p_original",
        "delta_b_minus_a_memorysafe",
        "mannwhitney_p_memorysafe",
        "welch_t_p_memorysafe",
    ]
    show_cols = [c for c in show_cols if c in cc.columns]
    lines.append(cc[show_cols].to_markdown(index=False))
    lines.append("")

    # ------------------------------------------------------------------
    # Microstate composition sanity
    # ------------------------------------------------------------------
    ms = read_table(args.memorysafe, "microstate_table.csv")
    lines.append("## Memory-safe microstate sanity checks")
    lines.append("")
    lines.append(f"- Number of microstates: {ms['microstate'].nunique()}")
    lines.append(f"- Median cells per microstate: {ms['n_cells'].median():.1f}")
    if "donor_count" in ms.columns:
        lines.append(f"- Median donor count per microstate: {ms['donor_count'].median():.1f}")
        low_donor = (ms["donor_count"] < 3).sum()
        lines.append(f"- Microstates with donor_count < 3: {low_donor}")
    if "frac_T2D" in ms.columns:
        lines.append("")
        lines.append("Most T2D-enriched microstates:")
        lines.append("")
        lines.append(
            ms.sort_values("frac_T2D", ascending=False)
              .head(10)
              .to_markdown(index=False)
        )
    lines.append("")

    # ------------------------------------------------------------------
    # Edge flux sanity
    # ------------------------------------------------------------------
    edge = read_table(args.memorysafe, "edge_flux_table.csv")
    lines.append("## Top memory-safe irreversible edges")
    lines.append("")
    lines.append(
        edge.sort_values("edge_entropy_production", ascending=False)
            .head(15)
            .to_markdown(index=False)
    )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote report: {out}")
    print(f"Wrote donor correlations: {corr_path}")
    print(f"Wrote condition comparison: {cc_path}")


if __name__ == "__main__":
    main()