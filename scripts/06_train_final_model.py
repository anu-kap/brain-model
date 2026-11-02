"""Step 6: train the final B_pca8 model on all available data, save to results/model.joblib.

B_pca8 = simple descriptors + PCA-8 of TRIBE input embeddings + Yeo-7 brain parcel features.
Run after step 03 (features built). Optionally run step 04 first so the CV OOF score
is available to log alongside the (optimistic) resubstitution score.

    python scripts/06_train_final_model.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from brainengage.config import load_config, resolve
from brainengage.modeling.cv import FEATURE_SETS, FeatureStore, make_ridge
from brainengage.modeling.metrics import spearman

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train_final")

FEATURE_SET = "B_pca8"
PCA_DIM = 8


def main():
    cfg = load_config()
    seed = int(cfg["seed"])
    mc = cfg["modeling"]
    feat_dir = resolve(cfg["data"].get("features_dir", "data/features"))
    store = FeatureStore(feat_dir, cfg)
    N = len(store.ids)
    y = store.y
    log.info("Training %s on %d videos, modalities=%s", FEATURE_SET, N, list(store.raw))

    if not store.set_available(FEATURE_SET):
        raise RuntimeError(f"{FEATURE_SET} components not available — run step 03 first")

    # Fit one PCA per modality on ALL data and save the fitted objects for inference.
    # This mirrors FeatureStore.pca_blocks but exposes (scaler, pca) for later use.
    kmax = max(store.pca_dims)
    pca_fitters: dict[str, tuple] = {}
    for m, series in store.raw.items():
        pooled = np.concatenate(series, axis=0)
        scaler = StandardScaler().fit(pooled)
        kk = min(kmax, pooled.shape[0] - 1, pooled.shape[1])
        pca = PCA(n_components=kk, svd_solver="randomized", random_state=seed).fit(
            scaler.transform(pooled)
        )
        pca_fitters[m] = (scaler, pca)
        log.info("PCA %s: pooled shape %s -> n_components=%d (using %d)", m, pooled.shape, kk, PCA_DIM)

    # Assemble B_pca8 features for all videos using the same pca_blocks path as run_cv.
    blocks = store.pca_blocks(np.arange(N), seed)
    X = store.assemble(FEATURE_SET, blocks)
    log.info("Feature matrix: %s", X.shape)

    alphas = np.logspace(*mc["ridge_alphas_log10"])
    ridge = make_ridge(alphas)
    ridge.fit(X, y)
    train_spearman = spearman(y, ridge.predict(X))
    log.info("Resubstitution Spearman (optimistic upper bound): %.3f", train_spearman)

    # Load CV OOF score if available (the honest number to report).
    oof_spearman = None
    oof_path = Path("results") / "cv_pooled_results.csv"
    if oof_path.exists():
        import pandas as pd
        oof = pd.read_csv(oof_path, dtype={"n": str})
        row = oof[(oof["set"] == FEATURE_SET) & (oof["learner"] == "ridge") & (oof["n"] == "all")]
        if len(row):
            oof_spearman = float(row["spearman"].mean())
            log.info("CV OOF Spearman (%s, honest): %.3f", FEATURE_SET, oof_spearman)

    artifact = {
        "feature_set": FEATURE_SET,
        "pca_dim": PCA_DIM,
        "modalities": list(store.raw.keys()),   # ordered list used during training
        "pca_fitters": pca_fitters,             # {m: (StandardScaler, PCA)} — for inference
        "ridge": ridge,                          # make_pipeline(StandardScaler, RidgeCV)
        "cfg": cfg,
        "n_train": N,
        "train_ids": store.ids,
        "train_ecr": y.tolist(),                # needed for percentile calculation in predict
        "train_spearman_resubstitution": float(train_spearman),
        "cv_oof_spearman": oof_spearman,
        "no_text": "text" not in store.raw,
    }
    out = Path("results") / "model.joblib"
    joblib.dump(artifact, out)
    log.info("Saved model -> %s", out)

    print(f"\nModel saved to {out}")
    print(f"  Feature set : {FEATURE_SET}")
    print(f"  Modalities  : {list(store.raw.keys())}")
    print(f"  N train     : {N}")
    print(f"  Spearman (resubstitution, optimistic) : {train_spearman:.3f}")
    if oof_spearman is not None:
        print(f"  Spearman (CV OOF, honest)             : {oof_spearman:.3f}")


if __name__ == "__main__":
    main()
