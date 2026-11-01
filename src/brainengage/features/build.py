"""Assemble per-video feature tables from cached TRIBE outputs + simple descriptors.

Fixed tables (written to data/features/*.parquet, rows aligned to the manifest):
  simple            duration, loudness, motion, cuts, transcript words
  brain_yeo7        Yeo-7 networks + GFP, temporal stats                      (Model C)
  brain_yeo7_w5     same, stats restricted to stimulus [0, 5 s)               (secondary)
  brain_hcp         HCP-MMP 180 parcels + GFP (only if labels available)      (exploratory)
  brain_mismatch    another same-duration-bin video's brain series            (control)
  rand_readout      random linear projection of the SAME input embeddings     (control)
  emb_raw           temporal stats of raw (unreduced) embeddings per modality (Model A, wide)
Raw embeddings (variable T) are kept in data/features/embeddings_raw.npz for the
per-fold PCA variants of Model A (fit inside each training draw, see modeling/cv.py).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from brainengage.features.parcellate import YEO7_NAMES, parcel_timeseries, yeo7_vertex_labels
from brainengage.features.simple import simple_features
from brainengage.features.temporal_stats import stat_names, temporal_stats

log = logging.getLogger(__name__)
MODALITIES = ("video", "audio", "text")


def load_preds(preds_dir: Path, vid: str) -> tuple[np.ndarray, dict]:
    p = np.load(preds_dir / f"{vid}.npy").astype(np.float32)
    meta = json.loads((preds_dir / f"{vid}.meta.json").read_text())
    return p, meta


def load_embeddings(emb_dir: Path, vid: str) -> dict[str, np.ndarray]:
    f = emb_dir / f"{vid}.npz"
    if not f.exists():
        return {}
    with np.load(f) as z:
        return {k: z[k].astype(np.float32) for k in z.files}


def window(ts: np.ndarray, w: int | None) -> np.ndarray:
    return ts if w is None else ts[: max(1, w)]


def stats_table(series: dict[str, np.ndarray], channels: list[str], ids: list[str], early: int,
                stats, w: int | None = None) -> pd.DataFrame:
    rows = [temporal_stats(window(series[i], w), early_window_s=early, stats=stats) for i in ids]
    return pd.DataFrame(np.vstack(rows), columns=stat_names(channels, stats), index=ids)


def build_simple(man: pd.DataFrame, meta: dict[str, dict], early: int) -> pd.DataFrame:
    rows = []
    for r in man.itertuples(index=False):
        f = simple_features(Path(r.path), early_window_s=early)
        f["n_words"] = float(meta.get(r.id, {}).get("n_words", 0))
        f["log_n_words"] = float(np.log1p(f["n_words"]))
        rows.append(f)
    return pd.DataFrame(rows, index=man.id.tolist())


def mismatch_partner(man: pd.DataFrame, seed: int) -> dict[str, str]:
    """Derangement within duration bins: each video gets another video's brain series."""
    rng = np.random.default_rng(seed)
    out = {}
    for _, g in man.groupby("duration_bin"):
        ids = g.id.tolist()
        if len(ids) < 2:
            for i in ids:
                out[i] = i
            continue
        perm = ids[:]
        for _ in range(100):
            rng.shuffle(perm)
            if all(a != b for a, b in zip(ids, perm)):
                break
        out.update(dict(zip(ids, perm)))
    return out


def random_readout(emb: dict[str, np.ndarray], dim: int, seed: int) -> np.ndarray:
    """(T, D_total) -> (T, dim+1): fixed-seed Gaussian projection + RMS channel (GFP analogue)."""
    mods = [m for m in MODALITIES if m in emb]
    T = min(emb[m].shape[0] for m in mods)
    x = np.concatenate([emb[m][:T] for m in mods], axis=1)
    x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((x.shape[1], dim)) / np.sqrt(x.shape[1])
    z = x @ W
    gfp = np.sqrt((z ** 2).mean(1, keepdims=True))
    return np.concatenate([z, gfp], axis=1)


def build_all(cfg: dict, man: pd.DataFrame, preds_dir: Path, emb_dir: Path, out_dir: Path, atlas_dir: Path) -> dict:
    fc = cfg["features"]
    early, stats = int(fc["early_window_s"]), tuple(fc["stats"])
    w5 = int(fc["label_window_s"])
    ids = man.id.tolist()
    out_dir.mkdir(parents=True, exist_ok=True)

    preds, meta, emb = {}, {}, {}
    for vid in ids:
        preds[vid], meta[vid] = load_preds(preds_dir, vid)
        emb[vid] = load_embeddings(emb_dir, vid)
    # truncate brain series to the clip duration (+1 s), keep index 0 = onset
    for vid in ids:
        T = int(np.ceil(meta[vid]["duration_s"])) + 1
        preds[vid] = preds[vid][:T]

    tables: dict[str, pd.DataFrame] = {}
    tables["simple"] = build_simple(man, meta, early)

    yeo = yeo7_vertex_labels(atlas_dir)
    ch_yeo = YEO7_NAMES + ["gfp"]
    yeo_ts = {v: parcel_timeseries(preds[v], yeo, 7) for v in ids}
    tables["brain_yeo7"] = stats_table(yeo_ts, ch_yeo, ids, early, stats)
    tables["brain_yeo7_w5"] = stats_table(yeo_ts, ch_yeo, ids, early, stats, w=w5)

    partner = mismatch_partner(man, cfg["seed"])
    mis_ts = {v: yeo_ts[partner[v]][: yeo_ts[v].shape[0]] for v in ids}
    tables["brain_mismatch"] = stats_table(mis_ts, ch_yeo, ids, early, stats)

    try:
        from brainengage.features.parcellate import hcp_vertex_labels

        hcp, names = hcp_vertex_labels()
        hcp_ts = {v: parcel_timeseries(preds[v], hcp, len(names)) for v in ids}
        tables["brain_hcp"] = stats_table(hcp_ts, names + ["gfp"], ids, early, stats)
    except Exception as e:  # noqa: BLE001
        log.warning("HCP parcellation skipped (%s)", e)

    have_emb = [v for v in ids if emb[v]]
    if len(have_emb) == len(ids):
        rd_ts = {v: random_readout(emb[v], int(fc["random_readout_dim"]), cfg["seed"]) for v in ids}
        rd_ch = [f"rand{i}" for i in range(int(fc["random_readout_dim"]))] + ["gfp"]
        tables["rand_readout"] = stats_table(rd_ts, rd_ch, ids, early, stats)
        tables["rand_readout_w5"] = stats_table(rd_ts, rd_ch, ids, early, stats, w=w5)
        for m in MODALITIES:
            if all(m in emb[v] for v in ids):
                D = emb[ids[0]][m].shape[1]
                ch = [f"{m}{j}" for j in range(D)]
                tables[f"emb_raw_{m}"] = stats_table({v: emb[v][m] for v in ids}, ch, ids, early, stats)
        np.savez_compressed(out_dir / "embeddings_raw.npz",
                            **{f"{v}__{m}": emb[v][m].astype(np.float16) for v in ids for m in emb[v]})
    else:
        log.warning("embeddings missing for %d/%d videos: A/B/random-readout tables not built",
                    len(ids) - len(have_emb), len(ids))

    for name, df in tables.items():
        df.index.name = "id"
        df.reset_index().to_parquet(out_dir / f"{name}.parquet", index=False)
    man[["id", "ecr", "duration_s", "duration_bin"]].to_parquet(out_dir / "targets.parquet", index=False)
    log.info("wrote %s", {k: v.shape for k, v in tables.items()})
    return tables
