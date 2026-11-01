"""Leakage / protocol tests on a tiny synthetic FeatureStore."""
import numpy as np
import pandas as pd
import pytest

from brainengage.config import load_config
from brainengage.features.temporal_stats import ALL_STATS
from brainengage.modeling.cv import FeatureStore, pooled_metrics, run_cv


@pytest.fixture
def store(tmp_path):
    rng = np.random.default_rng(0)
    n = 40
    ids = [f"v{i}" for i in range(n)]
    y = rng.uniform(0, 1, n)
    pd.DataFrame({"id": ids, "ecr": y, "duration_s": 10.0, "duration_bin": 0}).to_parquet(tmp_path / "targets.parquet")
    simple = pd.DataFrame({"id": ids, "duration_s": 10.0, "loud": rng.standard_normal(n)})
    simple.to_parquet(tmp_path / "simple.parquet")
    brain = pd.DataFrame(rng.standard_normal((n, 8)), columns=[f"b{i}" for i in range(8)])
    brain.insert(0, "id", ids)
    brain["b0"] += 6 * y  # planted signal (strong, so a 30-video ridge fit recovers it)
    brain.to_parquet(tmp_path / "brain_yeo7.parquet")
    emb = {}
    for i, v in enumerate(ids):
        for m in ("video", "audio", "text"):
            x = rng.standard_normal((11, 6)).astype(np.float16)
            x[:, 0] += 6 * y[i]
            emb[f"{v}__{m}"] = x
    np.savez_compressed(tmp_path / "embeddings_raw.npz", **emb)
    cfg = load_config()
    cfg["modeling"].update(outer_folds=4, outer_repeats=1, draws_per_n=1, learning_curve_n=[20, "all"])
    cfg["features"]["baseline_pca_dims"] = [8, 32]
    return FeatureStore(tmp_path, cfg), cfg


def test_pca_fit_depends_only_on_training_draw(store):
    st, cfg = store
    b1 = st.pca_blocks(np.arange(0, 20), seed=1)  # embeddings are 6-dim so pca8 keeps 6 comps
    b2 = st.pca_blocks(np.arange(20, 40), seed=1)
    b1b = st.pca_blocks(np.arange(0, 20), seed=1)
    assert b1["pca8__video"].shape == (40, 6 * len(ALL_STATS))
    assert np.allclose(b1["pca8__video"], b1b["pca8__video"])
    assert not np.allclose(b1["pca8__video"], b2["pca8__video"])


def test_run_cv_recovers_planted_signal_and_permutation_is_null(store):
    st, cfg = store
    rows, oof = run_cv(st, cfg, sets=["trivial", "A_pca8", "C"], learners=("ridge",))
    pooled = pooled_metrics(oof, dict(zip(st.ids, st.y)))
    c = pooled[(pooled.set == "C") & (pooled.n == "all")].spearman.mean()
    t = pooled[(pooled.set == "trivial") & (pooled.n == "all")].spearman.mean()
    a = pooled[(pooled.set == "A_pca8") & (pooled.n == "all")].spearman.mean()
    assert c > 0.5 and a > 0.5 and abs(t) < 0.5
    assert set(oof[oof.n == "all"].groupby(["set"]).id.nunique()) == {40}  # every video predicted OOF once
    _, poof = run_cv(st, cfg, sets=["C"], permute_y=True, learners=("ridge",))
    yperm = np.random.default_rng(cfg["seed"] + 999).permutation(st.y)
    pp = pooled_metrics(poof, dict(zip(st.ids, yperm)))
    assert abs(pp[pp.n == "all"].spearman.mean()) < 0.35
