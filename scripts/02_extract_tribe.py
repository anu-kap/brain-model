"""Step 2: run TRIBE v2 over the manifest (resumable). GPU env only.
    python scripts/02_extract_tribe.py                 # all videos in manifest
    python scripts/02_extract_tribe.py --limit 5       # first 5 (throughput check)
    python scripts/02_extract_tribe.py --no-text       # text stream disabled (not for the real run)
"""
from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import pandas as pd

from brainengage.config import load_config, resolve
from brainengage.features.tribe_extract import is_done, load_model, parse_config_update, run_video, save_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--no-text", action="store_true")
    ap.add_argument("-u", "--config-update", action="append", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["tribe"]
    text_stream = tc["text_stream"] and not args.no_text
    preds_dir, emb_dir = resolve(tc["preds_dir"]), resolve(tc["embeddings_dir"])
    man = pd.read_parquet(resolve(cfg["data"]["manifest"]))
    if args.ids:
        man = man[man.id.isin(args.ids)]
    todo = man[[not is_done(i, preds_dir) for i in man.id]]
    if args.limit:
        todo = todo.iloc[: args.limit]
    log.info("%d videos in manifest, %d to do", len(man), len(todo))
    if todo.empty:
        return
    model = load_model(tc["repo_id"], resolve(tc["cache_folder"]), config_update=parse_config_update(args.config_update))
    log_path = preds_dir.parent / "extract_log.csv"
    new_log = not log_path.exists()
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["id", "status", "n_timesteps", "duration_s", "n_words", "total_s", "ts"])
        for i, row in enumerate(todo.itertuples(index=False), 1):
            t = time.time()
            try:
                res = run_video(model, Path(row.path), text_stream=text_stream)
                save_outputs(res, row.id, preds_dir, emb_dir, text_stream)
                w.writerow([row.id, "ok", res["preds"].shape[0], f"{res['duration_s']:.2f}", res["n_words"],
                            f"{res['timing']['total_s']:.1f}", time.strftime("%F %T")])
                log.info("[%d/%d] %s T=%d words=%d %.0fs", i, len(todo), row.id, res["preds"].shape[0], res["n_words"], time.time() - t)
            except Exception as e:  # noqa: BLE001
                w.writerow([row.id, f"error:{type(e).__name__}:{str(e)[:120]}", "", "", "", f"{time.time()-t:.1f}", time.strftime("%F %T")])
                log.exception("[%d/%d] %s FAILED", i, len(todo), row.id)
            f.flush()


if __name__ == "__main__":
    main()
