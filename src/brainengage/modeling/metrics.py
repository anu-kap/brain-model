from __future__ import annotations

import numpy as np
from scipy import stats


def spearman(y: np.ndarray, p: np.ndarray) -> float:
    if np.std(p) == 0 or np.std(y) == 0:
        return 0.0
    return float(stats.spearmanr(y, p).statistic)


def pearson(y: np.ndarray, p: np.ndarray) -> float:
    if np.std(p) == 0 or np.std(y) == 0:
        return 0.0
    return float(stats.pearsonr(y, p).statistic)


def bootstrap_ci(y: np.ndarray, preds: dict[str, np.ndarray], fn, n_boot: int, seed: int,
                 pair: tuple[str, str] | None = None) -> dict:
    """Bootstrap over videos. If ``pair=(a, b)`` returns the CI of fn(b) - fn(a);
    otherwise returns the CI for the single key in ``preds``."""
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if pair:
            a, b = pair
            vals.append(fn(y[idx], preds[b][idx]) - fn(y[idx], preds[a][idx]))
        else:
            (k,) = preds
            vals.append(fn(y[idx], preds[k][idx]))
    vals = np.asarray(vals)
    if pair:
        point = fn(y, preds[pair[1]]) - fn(y, preds[pair[0]])
    else:
        point = fn(y, preds[next(iter(preds))])
    return {"point": float(point), "lo": float(np.percentile(vals, 2.5)), "hi": float(np.percentile(vals, 97.5)),
            "p_le_0": float(np.mean(vals <= 0))}
