"""Step 7: predict engagement (ECR) for a new video using the saved model.
GPU env required (TRIBE inference). Run after step 06.

    python scripts/07_predict.py --video path/to/ad.mp4
    python scripts/07_predict.py --video path/to/ad.mp4 -u data.video_feature.image.batch_size=2  # OOM fix
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np

from brainengage.config import load_config, resolve
from brainengage.features.parcellate import YEO7_NAMES, parcel_timeseries, yeo7_vertex_labels
from brainengage.features.simple import simple_features
from brainengage.features.temporal_stats import temporal_stats
from brainengage.features.tribe_extract import load_model, parse_config_update, run_video

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predict")

INTENSITY_BARS = {"low": "▒▒", "med": "████", "high": "██████████"}


def _intensity(v: float, vmax: float) -> str:
    frac = v / (vmax + 1e-9)
    return "high" if frac > 0.66 else "med" if frac > 0.33 else "low"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="path to the ad video (mp4/mov/…)")
    ap.add_argument("--model", default="results/model.joblib")
    ap.add_argument("--config", default=None)
    ap.add_argument("-u", "--config-update", action="append", default=None,
                    help="TRIBE config overrides, e.g. -u data.video_feature.image.batch_size=2")
    args = ap.parse_args()

    artifact = joblib.load(args.model)
    pca_fitters: dict = artifact["pca_fitters"]     # {m: (StandardScaler, PCA)}
    ridge = artifact["ridge"]                        # make_pipeline(StandardScaler, RidgeCV)
    pca_dim: int = artifact["pca_dim"]
    train_modalities: list[str] = artifact["modalities"]
    train_ecr = np.array(artifact["train_ecr"])
    no_text: bool = artifact.get("no_text", False)

    cfg = load_config(args.config)
    tc = cfg["tribe"]
    fc = cfg["features"]
    early, stats = int(fc["early_window_s"]), tuple(fc["stats"])
    w5 = int(fc["label_window_s"])

    # text_stream is disabled if the model was trained without text
    text_stream = tc["text_stream"] and not no_text

    video = Path(args.video)
    if not video.exists():
        raise FileNotFoundError(video)

    # --- run TRIBE ---
    log.info("Loading TRIBE model (this takes ~30 s on first call)…")
    tribe = load_model(tc["repo_id"], resolve(tc["cache_folder"]),
                       config_update=parse_config_update(args.config_update))
    log.info("Running TRIBE on %s…", video.name)
    res = run_video(tribe, video, text_stream=text_stream)
    preds = res["preds"]          # (T, 20484) float32
    emb = res["embeddings"]       # {m: (T, D) float16}
    duration_s = res["duration_s"]
    n_words = res["n_words"]
    log.info("TRIBE done: preds=%s duration=%.1fs words=%d timing=%s",
             preds.shape, duration_s, n_words,
             {k: f"{v:.0f}s" for k, v in res["timing"].items()})

    # Truncate brain series to clip duration + 1 s (mirrors build.py)
    T = int(np.ceil(duration_s)) + 1
    preds = preds[:T]

    # --- simple features (same key order as build_simple in build.py) ---
    sf = simple_features(video, early_window_s=early)
    sf["n_words"] = float(n_words)
    sf["log_n_words"] = float(np.log1p(n_words))
    x_simple = np.array(list(sf.values()), dtype=np.float64).reshape(1, -1)

    # --- PCA embedding features (per modality, same order as training) ---
    pca_parts = []
    for m in train_modalities:
        scaler, pca_obj = pca_fitters[m]
        kk = min(pca_dim, pca_obj.n_components_)
        if m in emb:
            proj = pca_obj.transform(scaler.transform(emb[m].astype(np.float32)))[:, :kk]
        else:
            log.warning("Modality %s missing; substituting zeros", m)
            proj = np.zeros((max(1, T), kk))
        pca_parts.append(temporal_stats(proj, early, stats).reshape(1, -1))
    x_pca = np.concatenate(pca_parts, axis=1) if pca_parts else np.zeros((1, 0))

    # --- Yeo-7 brain parcel features ---
    atlas_dir = resolve(tc["cache_folder"]) / "atlas"
    yeo = yeo7_vertex_labels(atlas_dir)
    yeo_ts = parcel_timeseries(preds, yeo, 7)       # (T, 8): 7 networks + GFP
    x_brain = temporal_stats(yeo_ts, early, stats).reshape(1, -1)

    # Assemble in the same order as FeatureStore.assemble for B_pca8:
    # [simple | pca8_mod0 | pca8_mod1 | brain_yeo7]
    X = np.concatenate([x_simple, x_pca, x_brain], axis=1).astype(np.float64)
    log.info("Feature vector: %d dims", X.shape[1])

    ecr_hat = float(np.clip(ridge.predict(X)[0], 0.0, 1.0))
    percentile = float(np.mean(train_ecr <= ecr_hat) * 100)

    # --- brain region summary over first 5 s (ECR window) ---
    w5_ts = yeo_ts[:w5].astype(np.float64)          # (≤5, 8)
    ch_names = YEO7_NAMES + ["GFP (overall)"]
    peak_t = w5_ts.argmax(axis=0)                   # time-of-peak per channel
    mean_act = w5_ts.mean(axis=0)
    max_act = w5_ts.max(axis=0)
    vmax = max_act.max()
    intensities = [_intensity(v, vmax) for v in max_act]

    modality_label = "video + audio" if no_text else "video + audio + text"

    print()
    print("=" * 56)
    print("  Brain-Model Engagement Prediction")
    print("=" * 56)
    print(f"  Video      : {video.name}  ({duration_s:.1f} s)")
    print(f"  Modalities : {modality_label}")
    print(f"  Predicted ECR  : {ecr_hat:.3f}")
    print(f"  Percentile     : top {100 - percentile:.0f}% of {len(train_ecr)} training videos")
    print()
    print("  Brain response — Yeo-7 networks (first 5 s):")
    for name, pt, inten, act in zip(ch_names, peak_t, intensities, mean_act):
        bar = INTENSITY_BARS[inten]
        print(f"    {name:<22} peak at t={pt}s  {bar:<12}  mean={act:.4f}  ({inten})")
    print()
    if no_text:
        print("  Note: model trained without text stream (Llama 3.2-3B access pending).")
        print("        Re-run after Llama access is granted for the full three-modality result.")
    print("=" * 56)
    print()


if __name__ == "__main__":
    main()
