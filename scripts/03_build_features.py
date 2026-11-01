"""Step 3: cached TRIBE outputs + videos -> feature tables in data/features/ (CPU)."""
from __future__ import annotations

import argparse
import logging

import pandas as pd

from brainengage.config import load_config, resolve
from brainengage.features.build import build_all
from brainengage.features.tribe_extract import is_done

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default=None, help="override manifest path (e.g. a synthetic one)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["tribe"]
    man = pd.read_parquet(resolve(args.manifest or cfg["data"]["manifest"]))
    preds_dir = resolve(tc["preds_dir"])
    done = man[[is_done(i, preds_dir) for i in man.id]]
    logging.info("%d/%d videos have TRIBE outputs; building features for those", len(done), len(man))
    build_all(cfg, done.reset_index(drop=True), preds_dir, resolve(tc["embeddings_dir"]),
              resolve(cfg["data"].get("features_dir", "data/features")), resolve(tc["cache_folder"]) / "atlas")


if __name__ == "__main__":
    main()
