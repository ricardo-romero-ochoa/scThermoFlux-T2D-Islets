from __future__ import annotations

from pathlib import Path
import yaml
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def zscore(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.all(~np.isfinite(x)):
        return np.zeros_like(x, dtype=float)
    mean = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < eps:
        return np.zeros_like(x, dtype=float)
    return (x - mean) / (sd + eps)


def safe_log_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log((num + eps) / (den + eps))


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.empty(n, dtype=float)
    cumulative = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        idx = order[i]
        cumulative = min(cumulative, pvalues[idx] * n / rank)
        ranked[idx] = cumulative
    return np.minimum(ranked, 1.0)


def write_manifest(root: str | Path, outpath: str | Path) -> None:
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            rows.append({
                "path": str(path.relative_to(root)),
                "size_bytes": path.stat().st_size,
            })
    pd.DataFrame(rows).to_csv(outpath, index=False)
