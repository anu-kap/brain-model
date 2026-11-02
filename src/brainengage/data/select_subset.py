"""Download, probe, deduplicate and balance the SnapUGC subset.

Used by scripts/01_build_subset.py. The label CSV has columns:
  Id, Title, Description, Download_link, ECR  (lowercased on load)
Dead CDN links return a 29-byte 404 body.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from brainengage.features.simple import frame_phashes, probe_video

log = logging.getLogger(__name__)

_DEAD_MAX_BYTES = 29  # CDN 404 responses are at most 29 bytes


def ensure_labels_csv(path: Path, gdrive_id: str | None = None) -> Path:
    path = Path(path)
    if path.exists() and path.stat().st_size > 10_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not gdrive_id:
        raise FileNotFoundError(f"Labels CSV not found at {path}. Provide a gdrive_id to auto-fetch.")
    log.info("Downloading train_data.csv from Google Drive (%s) -> %s", gdrive_id, path)
    import gdown
    gdown.download(id=gdrive_id, output=str(path), quiet=False)
    return path


def load_labels(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"Id": str})
    df.columns = [c.lower() for c in df.columns]
    if "ecr" not in df.columns:
        raise ValueError(f"ECR column not found in {path}; got: {df.columns.tolist()}")
    return df


def _download_one(row, videos_dir: Path, timeout: int) -> dict:
    dst = videos_dir / f"{row.id}.mp4"
    if dst.exists() and dst.stat().st_size > _DEAD_MAX_BYTES:
        return {"id": row.id, "status": "cached", "bytes": dst.stat().st_size}
    try:
        r = requests.get(row.download_link, timeout=timeout, stream=False)
        data = r.content
        if len(data) <= _DEAD_MAX_BYTES:
            return {"id": row.id, "status": "dead", "bytes": len(data)}
        dst.write_bytes(data)
        return {"id": row.id, "status": "ok", "bytes": len(data)}
    except Exception as e:  # noqa: BLE001
        return {"id": row.id, "status": f"error:{type(e).__name__}", "bytes": 0}


def download_ids(labels_df: pd.DataFrame, videos_dir: Path,
                 workers: int = 8, timeout: int = 60) -> pd.DataFrame:
    """Download exactly the videos in labels_df. Returns a log DataFrame with id and status."""
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)
    rows = list(labels_df.itertuples(index=False))
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_download_one, r, videos_dir, timeout): r.id for r in rows}
        for i, fut in enumerate(as_completed(futs), 1):
            results.append(fut.result())
            if i % 20 == 0 or i == len(rows):
                n_ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
                log.info("[%d/%d] live so far: %d", i, len(rows), n_ok)
    return pd.DataFrame(results)


def download_until(labels: pd.DataFrame, videos_dir: Path, n_live: int, seed: int,
                   workers: int = 8, timeout: int = 60) -> pd.DataFrame:
    """Shuffle labels by seed, download in batches until n_live clips are on disk."""
    videos_dir = Path(videos_dir)
    shuffled = labels.sample(frac=1, random_state=seed).reset_index(drop=True)
    all_logs: list[pd.DataFrame] = []
    batch = 200
    for start in range(0, len(shuffled), batch):
        chunk = shuffled.iloc[start: start + batch]
        log_chunk = download_ids(chunk, videos_dir, workers=workers, timeout=timeout)
        all_logs.append(log_chunk)
        combined = pd.concat(all_logs)
        n_live_now = int(combined["status"].isin(["ok", "cached"]).sum())
        log.info("live: %d / target %d (scanned %d/%d rows)", n_live_now, n_live, start + len(chunk), len(shuffled))
        if n_live_now >= n_live:
            break
    return pd.concat(all_logs, ignore_index=True)


def probe_all(ids: list[str], videos_dir: Path, min_duration: float, max_duration: float) -> pd.DataFrame:
    """Probe each video for duration, dimensions, codec. Returns DataFrame with probe_ok flag."""
    videos_dir = Path(videos_dir)
    rows = []
    for vid in ids:
        p = videos_dir / f"{vid}.mp4"
        if not p.exists():
            rows.append({"id": vid, "path": str(p), "probe_ok": False,
                         "duration_s": float("nan"), "fps": float("nan"),
                         "width": 0, "height": 0, "has_audio": False})
            continue
        try:
            info = probe_video(p)
            dur = info["duration_s"]
            ok = info["has_video"] and not np.isnan(dur) and min_duration <= dur <= max_duration
            rows.append({"id": vid, "path": str(p), "probe_ok": bool(ok),
                         "duration_s": dur, "fps": info["fps"],
                         "width": info["width"], "height": info["height"],
                         "has_audio": info["has_audio"]})
        except Exception as e:  # noqa: BLE001
            log.warning("probe failed for %s: %s", vid, e)
            rows.append({"id": vid, "path": str(p), "probe_ok": False,
                         "duration_s": float("nan"), "fps": float("nan"),
                         "width": 0, "height": 0, "has_audio": False})
    return pd.DataFrame(rows)


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop near-duplicate clips using perceptual hashes of frames at 10%, 50%, 90%."""
    df = df.sort_values("id").reset_index(drop=True)
    seen: dict[str, str] = {}
    keep: list[str] = []
    for row in df.itertuples(index=False):
        try:
            h = frame_phashes(Path(row.path))
        except Exception:  # noqa: BLE001
            h = f"nohash_{row.id}"  # unique fallback — never deduplicates on failure
        if h not in seen:
            seen[h] = row.id
            keep.append(row.id)
        else:
            log.info("dedupe: %s is near-duplicate of %s (hash=%s)", row.id, seen[h], h)
    return df[df.id.isin(keep)].reset_index(drop=True)


def balance(df: pd.DataFrame, n_target: int, duration_bins: list, seed: int) -> pd.DataFrame:
    """Stratified sample of n_target rows across duration bins; fills shortfall from other bins."""
    df = df.copy()
    df["duration_bin"] = pd.cut(
        df["duration_s"], bins=duration_bins, right=False, labels=False
    ).astype(int)
    n_bins = len(duration_bins) - 1
    per_bin = n_target // n_bins
    rng = np.random.default_rng(seed)

    selected: list[pd.DataFrame] = []
    leftover: list[pd.DataFrame] = []
    for b in range(n_bins):
        pool = df[df.duration_bin == b]
        take = min(len(pool), per_bin)
        chosen = pool.sample(n=take, random_state=int(rng.integers(1 << 31)))
        selected.append(chosen)
        leftover.append(pool.drop(index=chosen.index))

    shortfall = n_target - sum(len(s) for s in selected)
    if shortfall > 0:
        extra = pd.concat(leftover)
        take = min(shortfall, len(extra))
        if take > 0:
            selected.append(extra.sample(n=take, random_state=int(rng.integers(1 << 31))))

    return pd.concat(selected).reset_index(drop=True)
