from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _panel(ax, letter):
    ax.text(-0.12, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold", va="top")


def plot_landscape(latent, meta: pd.DataFrame, phi, outpath: str | Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    ax = axes[0]
    conds = list(pd.unique(meta["condition"]))
    for cond in conds:
        idx = meta["condition"].eq(cond).to_numpy()
        ax.scatter(latent[idx, 0], latent[idx, 1], s=6, alpha=0.6, label=str(cond))
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.legend(frameon=False, markerscale=2)
    _panel(ax, "A")

    ax = axes[1]
    sc = ax.scatter(latent[:, 0], latent[:, 1], c=phi, s=6, alpha=0.8)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("Quasi-potential")
    _panel(ax, "B")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_flux_network(microstate_table: pd.DataFrame, edge_table: pd.DataFrame, outpath: str | Path, top_n: int = 40):
    fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
    x = microstate_table["latent_1"].to_numpy()
    y = microstate_table["latent_2"].to_numpy()
    size = 30 + 5 * microstate_table["n_cells"].to_numpy()
    ax.scatter(x, y, s=size, alpha=0.7)
    top = edge_table.head(top_n)
    lookup = microstate_table.set_index("microstate")[["latent_1", "latent_2"]].to_dict("index")
    for _, row in top.iterrows():
        i = int(row["from_microstate"]); j = int(row["to_microstate"])
        if i not in lookup or j not in lookup:
            continue
        x0, y0 = lookup[i]["latent_1"], lookup[i]["latent_2"]
        x1, y1 = lookup[j]["latent_1"], lookup[j]["latent_2"]
        lw = 0.5 + 4 * row["edge_entropy_production"] / (top["edge_entropy_production"].max() + 1e-12)
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=lw, alpha=0.45))
    ax.set_xlabel("Microstate latent 1")
    ax.set_ylabel("Microstate latent 2")
    _panel(ax, "A")
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_donor_stats(donor_df: pd.DataFrame, outpath: str | Path):
    metrics = ["entropy_production", "disease_axis_mean", "quasi_potential_mean"]
    labels = ["Entropy-production-like score", "Disease-axis mean", "Quasi-potential mean"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric, label, letter in zip(axes, metrics, labels, ["A", "B", "C"]):
        conds = list(pd.unique(donor_df["condition"]))
        data = [donor_df.loc[donor_df.condition == c, metric].dropna().to_numpy() for c in conds]
        ax.boxplot(data, labels=conds, showfliers=False)
        for i, vals in enumerate(data, start=1):
            if len(vals):
                jitter = np.linspace(-0.08, 0.08, len(vals)) if len(vals) > 1 else np.array([0.0])
                ax.scatter(np.full(len(vals), i) + jitter, vals, s=20, alpha=0.8)
        ax.set_ylabel(label)
        _panel(ax, letter)
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
