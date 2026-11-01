"""Step 1: pick, download, verify, dedupe and balance the SnapUGC subset -> data/manifest.parquet"""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from brainengage.config import load_config, resolve
from brainengage.data.select_subset import balance, dedupe, download_until, load_labels, probe_all

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-target", type=int, default=None)
    ap.add_argument("--n-candidates", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    d = cfg["data"]
    n_target = args.n_target or d["n_target"]
    n_cand = args.n_candidates or d["n_candidates_min"]
    videos_dir = resolve(d["videos_dir"])

    labels = load_labels(resolve(d["labels_csv"]))
    if d.get("drop_ecr_zero"):
        labels = labels[labels.ecr > 0]
    log_df = download_until(labels, videos_dir, n_live=n_cand, seed=cfg["seed"],
                            workers=d["download_workers"], timeout=d["download_timeout_s"])
    log_df.to_csv(videos_dir.parent / "download_log.csv", index=False)
    live_ids = sorted(p.stem for p in videos_dir.glob("*.mp4"))
    probed = probe_all(live_ids, videos_dir, d["min_duration"], d["max_duration"])
    probed = probed.merge(labels[["id", "title", "description", "ecr"]], on="id", how="left")
    probed.to_csv(videos_dir.parent / "probe_log.csv", index=False)
    ok = probed[probed.probe_ok].copy()
    logging.info("live=%d probe_ok=%d", len(live_ids), len(ok))
    ok = dedupe(ok)
    man = balance(ok, n_target, d["duration_bins"], cfg["seed"])
    cols = ["id", "path", "duration_s", "duration_bin", "ecr", "title", "description", "fps", "width", "height", "has_audio"]
    man = man[cols]
    man.to_parquet(resolve(d["manifest"]), index=False)
    print(man.describe(include="all").T.head(12))
    print(f"wrote {resolve(d['manifest'])} with {len(man)} rows")


if __name__ == "__main__":
    main()
