"""Simple video/audio descriptors with PyAV (no ffmpeg binary needed).

All are cheap and deterministic. They go into *every* feature set so that
duration is never a hidden covariate of the brain features.
"""
from __future__ import annotations

from pathlib import Path

import av
import imagehash
import numpy as np
from PIL import Image

_SMALL = 64  # frame side for motion/cut features


def probe_video(path: Path) -> dict:
    with av.open(str(path)) as c:
        vs = next((s for s in c.streams if s.type == "video"), None)
        as_ = next((s for s in c.streams if s.type == "audio"), None)
        dur = (c.duration / av.time_base) if c.duration is not None else None
        if dur is None and vs is not None and vs.duration is not None:
            dur = float(vs.duration * vs.time_base)
        fps = float(vs.average_rate) if vs is not None and vs.average_rate else float("nan")
        return {
            "duration_s": float(dur) if dur is not None else float("nan"),
            "fps": fps,
            "width": int(vs.width) if vs else 0,
            "height": int(vs.height) if vs else 0,
            "has_video": vs is not None,
            "has_audio": as_ is not None,
        }


def _gray_frames(path: Path, max_fps: float = 10.0):
    """Yield (t_seconds, uint8 64x64 gray) at most max_fps frames per second."""
    with av.open(str(path)) as c:
        vs = next(s for s in c.streams if s.type == "video")
        vs.thread_type = "AUTO"
        last_t = -1e9
        for frame in c.decode(vs):
            t = float(frame.pts * vs.time_base) if frame.pts is not None else last_t + 1.0 / max_fps
            if t - last_t < 1.0 / max_fps - 1e-6:
                continue
            last_t = t
            g = frame.reformat(width=_SMALL, height=_SMALL, format="gray").to_ndarray()
            yield t, g


def frame_phashes(path: Path, fracs=(0.1, 0.5, 0.9)) -> str:
    """Concatenated perceptual hashes of frames at given fractions of the clip."""
    frames = [(t, g) for t, g in _gray_frames(path, max_fps=4.0)]
    if not frames:
        return "none"
    hashes = []
    for f in fracs:
        t, g = frames[min(len(frames) - 1, int(f * len(frames)))]
        hashes.append(str(imagehash.phash(Image.fromarray(g))))
    return "-".join(hashes)


def _audio_mono(path: Path, sr: int = 16000) -> np.ndarray | None:
    with av.open(str(path)) as c:
        as_ = next((s for s in c.streams if s.type == "audio"), None)
        if as_ is None:
            return None
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sr)
        chunks = []
        for frame in c.decode(as_):
            for rf in resampler.resample(frame):
                chunks.append(rf.to_ndarray().reshape(-1))
        for rf in resampler.resample(None):
            chunks.append(rf.to_ndarray().reshape(-1))
    if not chunks:
        return None
    return np.concatenate(chunks).astype(np.float32)


def _db(x: float) -> float:
    return float(20 * np.log10(x + 1e-6))


def simple_features(path: Path, early_window_s: float = 3.0, sr: int = 16000) -> dict:
    info = probe_video(path)
    dur = info["duration_s"]
    out = {"duration_s": dur, "has_audio": float(info["has_audio"])}

    # --- audio loudness ---
    a = _audio_mono(path, sr)
    if a is None or a.size == 0:
        out.update(loudness_db=-120.0, loudness_early_db=-120.0, loudness_var=0.0)
    else:
        n_early = int(early_window_s * sr)
        out["loudness_db"] = _db(float(np.sqrt(np.mean(a ** 2))))
        out["loudness_early_db"] = _db(float(np.sqrt(np.mean(a[:n_early] ** 2)))) if n_early > 0 else out["loudness_db"]
        # per-second RMS variance (dynamics)
        n_sec = max(1, a.size // sr)
        per_sec = np.sqrt(np.mean(a[: n_sec * sr].reshape(n_sec, sr) ** 2, axis=1))
        out["loudness_var"] = float(np.var(20 * np.log10(per_sec + 1e-6)))

    # --- motion energy and cut rate ---
    prev, diffs, times = None, [], []
    for t, g in _gray_frames(path, max_fps=10.0):
        g = g.astype(np.float32)
        if prev is not None:
            diffs.append(float(np.mean(np.abs(g - prev))))
            times.append(t)
        prev = g
    diffs_a, times_a = np.asarray(diffs), np.asarray(times)
    if diffs_a.size == 0:
        out.update(motion_energy=0.0, motion_early=0.0, cut_rate=0.0, n_cuts=0.0)
    else:
        out["motion_energy"] = float(np.mean(diffs_a))
        early = diffs_a[times_a < early_window_s]
        out["motion_early"] = float(np.mean(early)) if early.size else out["motion_energy"]
        thr = max(25.0, float(np.mean(diffs_a) + 3 * np.std(diffs_a)))
        n_cuts = int(np.sum(diffs_a > thr))
        out["n_cuts"] = float(n_cuts)
        out["cut_rate"] = n_cuts / max(dur, 1e-3)
    return out
