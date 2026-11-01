# CLAUDE.md — orientation for coding agents

Read this first. It says what the project is for, what has been decided, what is done, what is not, and how to work here.

## Purpose (one paragraph)

We are testing whether **predicted brain responses from Meta's TRIBE v2** add predictive value for **short-video
engagement** beyond the standard video/audio/text embeddings. If they do, the company would build a product that
ranks ad variants before launch. If they do not, the product can still exist as a conventional engagement predictor
with a brain-based *explanation* layer, but the brain part would not be a technical moat. **Phase 1 (this repo) is
the go/no-go experiment.** Phase 2 (a FastAPI + React app) only happens if Phase 1 is positive.

The scientific framing matters: TRIBE v2 is computed *from* V-JEPA 2, Wav2Vec-BERT and Llama 3.2 embeddings, so its
output contains **no new information**. The only way it can help is as a compact, structured representation when
training data is small. The experiment is built to answer exactly that, with controls that would expose a false
positive (see README "Preregistered protocol"). A July 2026 paper (arXiv 2607.01400, N=48 YouTube videos) found no
engagement signal in TRIBE once a temporal-position control was applied. Expect a null; report it honestly if so.

## Decisions already made with the user (do not re-litigate)

| Topic | Decision | Why |
|---|---|---|
| Scope | **Simple POC, ~200 videos, trainable in one day.** Not 1,000, not the full 106k. | Budget/time; it is a go/no-go test |
| Compute | **Lightning AI free tier + card (30 one-time credits), L4 24 GB Studio** | TRIBE OOMs on a 16 GB T4 (tribev2 issue #28); Lightning credits are one-time since Sept 2026 |
| Target | **ECR only** (P(watch > 5 s), rank-normalised) | NAWP was never publicly released; official val/test labels are hidden |
| Split | Our own repeated 5-fold CV on the labelled train rows | Official splits are unusable without labels; a fixed 100-video test set cannot detect realistic effects |
| Learner | Ridge (RidgeCV/GCV) for everything; gradient boosting once as robustness | Small N; same learner for A/B/C is the point |
| Text stream | Llama-3.2-3B access was being requested on 2026-11-01 | `--no-text` exists for the smoke test only; the real run needs it |

## Status (updated 2026-11-01)

Done and verified on the Mac (CPU):
- Repo, config, tests (10 passing), README with the preregistered protocol.
- **Subset selected and downloaded**: 200 clips in `data/videos/` (606 MB, gitignored), listed in `results/manifest.csv`.
  Duration bins 126 / 50 / 24 (5–20 / 20–40 / 40–60 s); long clips are rare among live links.
- **Pipeline exercised end to end on fabricated TRIBE outputs** (`scripts/dev_make_synthetic.py` + `configs/synthetic.yaml`):
  steps 3→4→5 run in ~80 s in `--quick` mode; permutation sanity ≈ 0. Outputs in `results/_synthetic/` (gitignored, NOT real).
- One real number: the trivial baseline (duration, loudness, motion, cuts) reaches ≈0.34 Spearman on real ECR; duration–ECR Spearman is 0.27.

Not done (needs the GPU machine):
- `scripts/00_smoke_test.py` on one clip. Its job: confirm `preds.shape == (T, 20484)`, timing per clip, and that
  the cached input embeddings can be read back via `extractor(events, start, duration)`. **That read is the one
  unverified assumption in the code** (`features/tribe_extract.py::extract_embeddings`). If it fails, the fallback is
  to run the same HF models directly (model ids are in the README) and write the same `.npz` format.
- `scripts/02_extract_tribe.py` over the 200 clips, then `03 → 04 → 05` for the real `results/summary.md`.

## How to work here

- Everything is driven by `configs/default.yaml`. Do not hard-code paths, seeds or sizes in scripts.
- Pipeline order is the script number: `00` smoke test (GPU) → `01` subset → `02` extraction (GPU) → `03` features → `04` experiments → `05` report.
  All are resumable/idempotent; `02` skips ids that already have `data/preds/<id>.npy` + `.meta.json`.
- `data/` is gitignored and is the only place for caches/videos/features. `results/` is committed and must contain
  only real outputs (synthetic runs go to `results/_synthetic/`, which is ignored).
- Keep `features/temporal_stats.py` as the single summary function for both the brain parcels and the baseline PCA
  components. Fairness of A vs B depends on it.
- Fit anything data-dependent (PCA, scalers, alpha) **inside** the training draw; `tests/test_cv.py` guards this.
- Run `.venv/bin/python -m pytest -q` before committing. Local env: `uv venv -p 3.11 && uv pip install -e ".[dev]"`.
  The GPU extra (`.[gpu]`) cannot install on macOS; use `scripts/setup_gpu_env.sh` on Linux.
- Commit messages end with the co-author line used in the existing history.

## Gotchas learned the hard way

- **TRIBE offset**: the 5 s hemodynamic lag is applied on the fMRI side during training (`neuralset` shifts BOLD by −5 s),
  so `preds[t]` is already the response to stimulus second `t`. Index 0 = onset. Do not slice off the first 5 rows.
- **Vertex order** is left hemisphere (10242) then right (10242). The Yeo-7 labels in `features/parcellate.py` come from
  the Yeo lab's native fsaverage5 `.annot` files (nilearn's Yeo fetch uses FTP, which was blocked here).
- **tribev2 install**: pin `exca>=0.5.20,<0.5.26` (neuralset 0.0.2 breaks on newer exca). Transcription shells out to
  `uvx whisperx` (large-v3), so `uv` and `ffmpeg` must be on PATH; it writes `<video>.wav` and `<video>.tsv` next to
  each mp4 (so `data/videos/` grows ~2×). If WhisperX dominates per-clip time, pre-write the `.tsv` files in one batch.
- **SnapUGC links**: ~30 % of CDN links are alive; dead ones return a 29-byte 404 body. `01_build_subset.py` keeps
  sampling (seeded) until it has enough live clips. 544 rows have ECR exactly 0; they are kept.
- **Lightning**: keep `data/` under 10 GB (free storage cap), measure minutes/clip with `--limit 5` before the full
  run, and lower `data.n_target` rather than overrun credits if a clip takes > 3 min.
- Licences: TRIBE v2 is CC-BY-NC-4.0 (research only). A commercial Phase 2 needs a different brain model.

## If the result is null

Say so plainly in `results/summary.md`. The verdict logic in `scripts/05_report.py` already distinguishes
"brain helps", "helps only at small n", "random readout does as well" and "null". Do not add feature sets or
learners after seeing results; anything beyond the preregistered sets must be labelled exploratory.
