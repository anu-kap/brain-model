"""Run TRIBE v2 on one video and persist (a) the predicted cortical time series and
(b) the intermediate input embeddings TRIBE computed on the way (the baseline features).

GPU environment only (Lightning AI L4 Studio). See README "GPU setup".

Facts verified against the tribev2 source (commit of 2026-11-01):
- ``TribeModel.from_pretrained(repo, cache_folder=..., device=..., config_update=...)``
- ``model.get_events_dataframe(video_path=...)`` runs ExtractAudioFromVideo (writes
  ``<video>.wav`` next to the mp4), chunks, WhisperX via ``uvx whisperx`` (caches
  ``<video>.tsv``), then text transforms. ``get_audio_and_text_events(df, audio_only=True)``
  skips transcription (text stream absent -> zeros via allow_missing).
- ``preds, segments = model.predict(events)`` -> (n_TR, 20484) float32; TR = 1 s.
- The 5 s hemodynamic offset is applied on the fMRI side at training time
  (neuralset FmriExtractor: ``start = ta.start - offset``), so ``preds[t]`` is the
  lag-compensated response to stimulus second ``t``. Index 0 = stimulus onset.
- Extractors are callable after ``predict``: ``ext(events, start, duration)`` returns a
  tensor (L, D, T2) at 2 Hz read from the exca cache (``_model`` is freed, cache persists).
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

MODALITIES = ("video", "audio", "text")


def parse_config_update(items: list[str] | None) -> dict | None:
    """["data.video_feature.image.batch_size=2", ...] -> dict with typed values."""
    if not items:
        return None
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out[k] = v
    return out


def load_model(repo_id: str, cache_folder: Path, device: str = "auto", config_update: dict | None = None):
    from tribev2 import TribeModel

    t = time.time()
    model = TribeModel.from_pretrained(repo_id, cache_folder=str(cache_folder), device=device,
                                       config_update=config_update)
    log.info("loaded %s in %.1fs (device=%s)", repo_id, time.time() - t, device)
    return model


def build_events(model, video_path: Path, text_stream: bool = True) -> pd.DataFrame:
    if text_stream:
        return model.get_events_dataframe(video_path=str(video_path))
    from tribev2.demo_utils import get_audio_and_text_events

    event = {"type": "Video", "filepath": str(video_path), "start": 0, "timeline": "default", "subject": "default"}
    return get_audio_and_text_events(pd.DataFrame([event]), audio_only=True)


def n_words(events: pd.DataFrame) -> int:
    return int((events["type"] == "Word").sum()) if "type" in events else 0


def _to_1hz(x: np.ndarray, src_hz: float) -> np.ndarray:
    """(T_src, D) -> (T_1hz, D) by averaging consecutive samples."""
    k = int(round(src_hz))
    if k <= 1:
        return x
    T = (x.shape[0] // k) * k
    if T == 0:
        return x[:1]
    return x[:T].reshape(T // k, k, x.shape[1]).mean(1)


def extract_embeddings(model, events: pd.DataFrame, duration_s: float) -> dict[str, np.ndarray]:
    """Return {modality: (T_1hz, L*D) float16} for each modality TRIBE used, read from its cache."""
    import torch

    out: dict[str, np.ndarray] = {}
    for modality in model.data.features_to_use:
        ext = getattr(model.data, f"{modality}_feature", None)
        if ext is None:
            continue
        try:
            with torch.inference_mode():
                x = ext(events, start=0.0, duration=float(duration_s))
        except Exception as e:  # noqa: BLE001
            log.warning("embedding read failed for %s: %s", modality, e)
            continue
        x = x.detach().cpu().float().numpy()
        if x.ndim == 3:  # (L, D, T)
            L, D, T = x.shape
            x = x.transpose(2, 0, 1).reshape(T, L * D)
        elif x.ndim == 2:  # (D, T)
            x = x.T
        else:
            log.warning("unexpected embedding ndim %d for %s", x.ndim, modality)
            continue
        hz = float(getattr(ext, "frequency", 1.0) or 1.0)
        out[modality] = _to_1hz(x, hz).astype(np.float16)
    return out


def run_video(model, video_path: Path, text_stream: bool = True) -> dict:
    """Full per-video pipeline. Returns dict with preds (T, V) float32, segments, events, embeddings, timing."""
    timing = {}
    t0 = time.time()
    events = build_events(model, video_path, text_stream=text_stream)
    timing["events_s"] = time.time() - t0
    t1 = time.time()
    preds, segments = model.predict(events=events, verbose=False)
    timing["predict_s"] = time.time() - t1
    vid = events[events["type"] == "Video"]
    duration = float((vid["start"] + vid["duration"]).max()) if len(vid) else float(preds.shape[0])
    t2 = time.time()
    emb = extract_embeddings(model, events, duration)
    timing["embeddings_s"] = time.time() - t2
    timing["total_s"] = time.time() - t0
    return {"preds": preds.astype(np.float32), "segments": segments, "events": events,
            "embeddings": emb, "duration_s": duration, "n_words": n_words(events), "timing": timing}


def save_outputs(result: dict, video_id: str, preds_dir: Path, emb_dir: Path, text_stream: bool) -> None:
    preds_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / f"{video_id}.npy", result["preds"].astype(np.float16))
    starts = [float(getattr(s, "start", np.nan)) for s in result["segments"]]
    meta = {"id": video_id, "n_timesteps": int(result["preds"].shape[0]), "n_vertices": int(result["preds"].shape[1]),
            "duration_s": result["duration_s"], "segment_starts": starts, "n_words": result["n_words"],
            "text_stream": text_stream, "timing": result["timing"],
            "embedding_shapes": {k: list(v.shape) for k, v in result["embeddings"].items()}}
    (preds_dir / f"{video_id}.meta.json").write_text(json.dumps(meta, indent=1))
    if result["embeddings"]:
        np.savez_compressed(emb_dir / f"{video_id}.npz", **result["embeddings"])


def is_done(video_id: str, preds_dir: Path) -> bool:
    return (preds_dir / f"{video_id}.npy").exists() and (preds_dir / f"{video_id}.meta.json").exists()
