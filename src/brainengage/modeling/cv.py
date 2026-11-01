"""Repeated K-fold evaluation with within-fold subsampling (learning curve), per-draw PCA
for the baseline embeddings, one ridge learner for every feature set, pooled out-of-fold metrics.

Feature *sets* are concatenations of *components*. Fixed components come from
data/features/*.parquet. ``pca{k}`` components are fit on the training draw only
(StandardScaler + PCA on pooled 1 Hz timepoints of the training videos), projected,
then summarised with the same temporal_stats as the brain parcels.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from brainengage.features.temporal_stats import temporal_stats
from brainengage.modeling.metrics import pearson, spearman

log = logging.getLogger(__name__)
MODALITIES = ("video", "audio", "text")

# name -> list of components. "pca{k}" / "pca{k}_w5" expand to one block per modality.
FEATURE_SETS: dict[str, list[str]] = {
    "trivial": ["simple"],
    "A_pca8": ["simple", "pca8"],
    "A_pca32": ["simple", "pca32"],
    "A_raw": ["simple", "emb_raw"],
    "C": ["simple", "brain_yeo7"],
    "C_brain_only": ["brain_yeo7"],
    "B_pca8": ["simple", "pca8", "brain_yeo7"],
    "B_pca32": ["simple", "pca32", "brain_yeo7"],
    "B_raw": ["simple", "emb_raw", "brain_yeo7"],
    # controls
    "C_rand": ["simple", "rand_readout"],
    "B_rand_pca8": ["simple", "pca8", "rand_readout"],
    "C_mismatch": ["simple", "brain_mismatch"],
    "B_mismatch_pca8": ["simple", "pca8", "brain_mismatch"],
    # secondary: first 5 s only (ECR window)
    "A_pca8_w5": ["simple", "pca8_w5"],
    "C_w5": ["simple", "brain_yeo7_w5"],
    "B_pca8_w5": ["simple", "pca8_w5", "brain_yeo7_w5"],
    "C_rand_w5": ["simple", "rand_readout_w5"],
    # exploratory
    "C_hcp": ["simple", "brain_hcp"],
    "B_hcp_pca8": ["simple", "pca8", "brain_hcp"],
}
HGB_SETS = ["trivial", "A_pca8", "B_pca8", "C"]


class FeatureStore:
    def __init__(self, feat_dir: Path, cfg: dict):
        feat_dir = Path(feat_dir)
        fc = cfg["features"]
        self.early, self.stats, self.w5 = int(fc["early_window_s"]), tuple(fc["stats"]), int(fc["label_window_s"])
        self.pca_dims = sorted(int(k) for k in fc["baseline_pca_dims"])
        t = pd.read_parquet(feat_dir / "targets.parquet")
        self.ids = t.id.tolist()
        self.y = t.ecr.to_numpy(float)
        self.targets = t
        self.fixed: dict[str, np.ndarray] = {}
        for f in sorted(feat_dir.glob("*.parquet")):
            if f.stem == "targets":
                continue
            df = pd.read_parquet(f).set_index("id").loc[self.ids]
            self.fixed[f.stem] = np.nan_to_num(df.to_numpy(np.float64))
        self.raw: dict[str, list[np.ndarray]] = {}
        z = feat_dir / "embeddings_raw.npz"
        if z.exists():
            with np.load(z) as zz:
                for m in MODALITIES:
                    keys = [f"{v}__{m}" for v in self.ids]
                    if all(k in zz.files for k in keys):
                        self.raw[m] = [zz[k].astype(np.float32) for k in keys]
        log.info("features: fixed=%s raw_modalities=%s n=%d", list(self.fixed), list(self.raw), len(self.ids))

    # ---- components -------------------------------------------------------------
    def available(self, comp: str) -> bool:
        if comp.startswith("pca"):
            return bool(self.raw)
        if comp == "emb_raw":
            return any(k.startswith("emb_raw_") for k in self.fixed)
        return comp in self.fixed

    def set_available(self, name: str) -> bool:
        return all(self.available(c) for c in FEATURE_SETS[name])

    def pca_blocks(self, fit_idx: np.ndarray, seed: int) -> dict[str, np.ndarray]:
        """Fit one PCA(max_k) per modality on the training draw; return blocks for every k and window."""
        out: dict[str, np.ndarray] = {}
        kmax = max(self.pca_dims)
        for m, series in self.raw.items():
            pooled = np.concatenate([series[i] for i in fit_idx], axis=0)
            scaler = StandardScaler().fit(pooled)
            pca = PCA(n_components=min(kmax, pooled.shape[0] - 1, pooled.shape[1]),
                      svd_solver="randomized", random_state=seed).fit(scaler.transform(pooled))
            proj = [pca.transform(scaler.transform(s)) for s in series]
            for k in self.pca_dims:
                kk = min(k, proj[0].shape[1])
                out[f"pca{k}__{m}"] = np.vstack([temporal_stats(p[:, :kk], self.early, self.stats) for p in proj])
                out[f"pca{k}_w5__{m}"] = np.vstack([temporal_stats(p[:self.w5, :kk], self.early, self.stats) for p in proj])
        return out

    def assemble(self, name: str, pca_blocks: dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for comp in FEATURE_SETS[name]:
            if comp.startswith("pca"):
                parts += [pca_blocks[f"{comp}__{m}"] for m in self.raw]
            elif comp == "emb_raw":
                parts += [v for k, v in self.fixed.items() if k.startswith("emb_raw_")]
            else:
                parts.append(self.fixed[comp])
        return np.concatenate(parts, axis=1)


def make_ridge(alphas: np.ndarray):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))


def make_hgb(seed: int):
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                         min_samples_leaf=10, random_state=seed)


def run_cv(store: FeatureStore, cfg: dict, sets: list[str] | None = None, repeats: int | None = None,
           draws: int | None = None, permute_y: bool = False, learners=("ridge",)) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (fold_rows, oof) where oof has one row per (set, learner, n, repeat, draw, video)."""
    mc = cfg["modeling"]
    seed = int(cfg["seed"])
    K, R = int(mc["outer_folds"]), int(repeats or mc["outer_repeats"])
    D = int(draws or mc["draws_per_n"])
    alphas = np.logspace(*mc["ridge_alphas_log10"])
    sizes = mc["learning_curve_n"]
    sets = [s for s in (sets or list(FEATURE_SETS)) if store.set_available(s)]
    skipped = [s for s in (sets or []) if s not in sets]
    log.info("sets=%s", sets)
    N = len(store.ids)
    y = store.y.copy()
    if permute_y:
        y = np.random.default_rng(seed + 999).permutation(y)
    rows, oof_rows = [], []
    for rep in range(R):
        kf = KFold(K, shuffle=True, random_state=seed + rep)
        for fold, (tr, te) in enumerate(kf.split(np.arange(N))):
            for n in sizes:
                n_eff = len(tr) if n == "all" else min(int(n), len(tr))
                n_draws = 1 if n_eff == len(tr) else D
                for d in range(n_draws):
                    rng = np.random.default_rng([seed, rep, fold, n_eff, d])
                    sub = tr if n_eff == len(tr) else rng.choice(tr, n_eff, replace=False)
                    blocks = store.pca_blocks(sub, seed) if store.raw else {}
                    for sname in sets:
                        X = store.assemble(sname, blocks)
                        for lname in learners:
                            if lname == "hgb" and (sname not in HGB_SETS or n_eff != len(tr)):
                                continue
                            model = make_ridge(alphas) if lname == "ridge" else make_hgb(seed)
                            model.fit(X[sub], y[sub])
                            p = model.predict(X[te])
                            rows.append({"set": sname, "learner": lname, "n": n, "n_train": n_eff, "repeat": rep,
                                         "fold": fold, "draw": d, "n_features": X.shape[1],
                                         "spearman": spearman(y[te], p), "pearson": pearson(y[te], p),
                                         "alpha": float(model[-1].alpha_) if lname == "ridge" else np.nan})
                            for i, pi in zip(te, p):
                                oof_rows.append((sname, lname, str(n), rep, d, store.ids[i], float(pi)))
        log.info("repeat %d/%d done (%d fold rows)", rep + 1, R, len(rows))
    oof = pd.DataFrame(oof_rows, columns=["set", "learner", "n", "repeat", "draw", "id", "pred"])
    return pd.DataFrame(rows), oof


def pooled_metrics(oof: pd.DataFrame, y_by_id: dict[str, float]) -> pd.DataFrame:
    """Spearman/Pearson over all N videos per (set, learner, n, repeat, draw)."""
    out = []
    for keys, g in oof.groupby(["set", "learner", "n", "repeat", "draw"]):
        yy = np.array([y_by_id[i] for i in g.id])
        out.append(dict(zip(["set", "learner", "n", "repeat", "draw"], keys),
                        spearman=spearman(yy, g.pred.to_numpy()), pearson=pearson(yy, g.pred.to_numpy()), n_videos=len(g)))
    return pd.DataFrame(out)


def ensemble_oof(oof: pd.DataFrame, ids: list[str]) -> dict[tuple, np.ndarray]:
    """Per (set, learner, n): per-video prediction averaged over repeats and draws, aligned to ids."""
    out = {}
    for keys, g in oof.groupby(["set", "learner", "n"]):
        m = g.groupby("id").pred.mean()
        out[keys] = m.reindex(ids).to_numpy()
    return out
