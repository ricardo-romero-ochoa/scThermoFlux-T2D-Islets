#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Simple donor-level condition-label permutation null.")
    p.add_argument("--donor-table", required=True)
    p.add_argument("--metric", default="entropy_production")
    p.add_argument("--condition-a", default="ND")
    p.add_argument("--condition-b", default="T2D")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(args.donor_table)
    mask = df["condition"].isin([args.condition_a, args.condition_b])
    sub = df.loc[mask].copy()
    obs = sub.groupby("condition")[args.metric].mean()
    observed = obs.get(args.condition_b, np.nan) - obs.get(args.condition_a, np.nan)
    labels = sub["condition"].to_numpy().copy()
    vals = sub[args.metric].to_numpy()
    deltas = []
    for _ in range(args.n_perm):
        perm = rng.permutation(labels)
        a = vals[perm == args.condition_a]
        b = vals[perm == args.condition_b]
        deltas.append(np.mean(b) - np.mean(a))
    deltas = np.asarray(deltas)
    pval = (np.sum(np.abs(deltas) >= abs(observed)) + 1) / (len(deltas) + 1)
    out = pd.DataFrame({"observed_delta": [observed], "permutation_p": [pval], "n_perm": [args.n_perm]})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(out)


if __name__ == "__main__":
    main()
