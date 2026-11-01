"""Step 0: run TRIBE v2 on ONE video end to end and print shapes, timing and sanity checks.
GPU env only. Example:
    python scripts/00_smoke_test.py --video data/videos/<id>.mp4
    python scripts/00_smoke_test.py --video x.mp4 --no-text           # Llama access not yet granted
    python scripts/00_smoke_test.py --video x.mp4 -u data.video_feature.image.batch_size=2
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from brainengage.config import load_config, resolve
from brainengage.features.tribe_extract import load_model, parse_config_update, run_video, save_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-text", action="store_true", help="skip WhisperX/Llama text stream")
    ap.add_argument("-u", "--config-update", action="append", default=None, help="key=value overrides for TRIBE config")
    ap.add_argument("--save", action="store_true", help="also save outputs like 02_extract_tribe.py does")
    args = ap.parse_args()
    cfg = load_config(args.config)
    tc = cfg["tribe"]
    text_stream = tc["text_stream"] and not args.no_text

    model = load_model(tc["repo_id"], resolve(tc["cache_folder"]), config_update=parse_config_update(args.config_update))
    video = Path(args.video)
    res = run_video(model, video, text_stream=text_stream)
    preds = res["preds"]

    checks = {}
    checks["shape"] = list(preds.shape)
    checks["n_vertices_ok"] = preds.shape[1] == tc["n_vertices"]
    checks["duration_s"] = res["duration_s"]
    checks["n_timesteps_minus_duration"] = preds.shape[0] - res["duration_s"]
    checks["finite"] = bool(np.isfinite(preds).all())
    checks["std_first5_rows"] = float(preds[:5].std())
    checks["std_rest_rows"] = float(preds[5:].std()) if preds.shape[0] > 5 else None
    checks["row0_vs_mean_corr"] = float(np.corrcoef(preds[0], preds.mean(0))[0, 1]) if preds.shape[0] > 1 else None
    checks["n_words"] = res["n_words"]
    checks["embeddings"] = {k: list(v.shape) for k, v in res["embeddings"].items()}
    checks["embeddings_ok"] = set(res["embeddings"]) >= {"video", "audio"}
    checks["timing_s"] = {k: round(v, 1) for k, v in res["timing"].items()}
    print(json.dumps(checks, indent=1))
    if args.save:
        save_outputs(res, video.stem, resolve(tc["preds_dir"]), resolve(tc["embeddings_dir"]), text_stream)
        print("saved")
    ok = checks["n_vertices_ok"] and checks["finite"] and checks["embeddings_ok"]
    print("SMOKE TEST", "PASSED" if ok else "FAILED (see embeddings_ok / n_vertices_ok / finite)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
