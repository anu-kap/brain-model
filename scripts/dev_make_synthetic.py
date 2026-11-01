"""Development only: fabricate TRIBE-like outputs for the videos in a manifest so that
steps 3-5 can be exercised end to end on a CPU machine before any GPU run.

The synthetic generator plants a latent per-video signal correlated with ECR into the
input embeddings and then derives the "brain" predictions as a fixed linear map of
those embeddings plus noise, mimicking the real situation (brain = f(inputs)).
Outputs go to a separate tree (data/synthetic/...) and never mix with real caches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from brainengage.config import load_config, resolve
from brainengage.features.parcellate import N_VERT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--signal", type=float, default=0.6, help="corr(latent, ECR)")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config()
    man = pd.read_parquet(resolve(args.manifest or cfg["data"]["manifest"]))
    if args.limit:
        man = man.iloc[: args.limit]
    out = resolve(args.out)
    preds_dir, emb_dir = out / "preds", out / "embeddings"
    preds_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    dims = {"video": 64, "audio": 32, "text": 48}
    Wb = {m: rng.standard_normal((d, N_VERT)) / np.sqrt(d) for m, d in dims.items()}
    ecr_z = (man.ecr - man.ecr.mean()) / (man.ecr.std() + 1e-9)
    for r, ez in zip(man.itertuples(index=False), ecr_z):
        T = int(np.ceil(r.duration_s)) + 1
        latent = args.signal * ez + np.sqrt(1 - args.signal ** 2) * rng.standard_normal()
        emb = {}
        for m, d in dims.items():
            base = rng.standard_normal((T, d))
            base[:, :4] += latent  # planted signal in the first few dims
            emb[m] = base.astype(np.float16)
        brain = sum(emb[m].astype(np.float32) @ Wb[m] for m in dims) + 0.5 * rng.standard_normal((T, N_VERT))
        brain += np.linspace(0, 1, T)[:, None] * 0.2  # generic ramp (temporal-shape artifact)
        np.save(preds_dir / f"{r.id}.npy", brain.astype(np.float16))
        meta = {"id": r.id, "n_timesteps": T, "n_vertices": N_VERT, "duration_s": float(r.duration_s),
                "segment_starts": list(range(T)), "n_words": int(rng.integers(0, 40)), "text_stream": True,
                "timing": {"total_s": 0.0}, "embedding_shapes": {m: [T, d] for m, d in dims.items()}, "synthetic": True}
        (preds_dir / f"{r.id}.meta.json").write_text(json.dumps(meta))
        np.savez_compressed(emb_dir / f"{r.id}.npz", **emb)
    print(f"synthetic outputs for {len(man)} videos in {out}")


if __name__ == "__main__":
    main()
