"""ONE temporal summary function, applied identically to brain parcels (Model C)
and to PCA components of the input embeddings (Model A). This is what makes the
A-vs-B comparison fair: any gain from B cannot come from richer summaries."""
from __future__ import annotations

import numpy as np

ALL_STATS = ("mean", "peak", "time_to_peak", "slope", "var", "early")


def temporal_stats(ts: np.ndarray, early_window_s: int = 3, stats=ALL_STATS, fs: float = 1.0) -> np.ndarray:
    """ts: (T, C) time series sampled at ``fs`` Hz, index 0 = stimulus onset.
    Returns a flat vector ordered channel-major: [c0_stat0, c0_stat1, ..., c1_stat0, ...]."""
    ts = np.asarray(ts, dtype=np.float64)
    if ts.ndim != 2 or ts.shape[0] == 0:
        raise ValueError(f"expected (T, C) with T>0, got {ts.shape}")
    T, C = ts.shape
    t = np.arange(T) / fs
    n_early = max(1, int(round(early_window_s * fs)))
    cols = {}
    cols["mean"] = ts.mean(0)
    cols["peak"] = ts.max(0)
    cols["time_to_peak"] = t[ts.argmax(0)]
    if T > 1:
        tc = t - t.mean()
        cols["slope"] = (tc[:, None] * (ts - ts.mean(0))).sum(0) / (tc ** 2).sum()
    else:
        cols["slope"] = np.zeros(C)
    cols["var"] = ts.var(0)
    cols["early"] = ts[:n_early].mean(0)
    mat = np.stack([cols[s] for s in stats], axis=1)  # (C, n_stats)
    return mat.reshape(-1)


def stat_names(channels: list[str], stats=ALL_STATS) -> list[str]:
    return [f"{c}__{s}" for c in channels for s in stats]
