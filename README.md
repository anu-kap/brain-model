# brainengage: do predicted-brain features add engagement-prediction value?

Proof of concept testing whether **TRIBE v2** predicted fMRI responses improve prediction of
short-video engagement (SnapUGC **ECR**) beyond the *same input embeddings TRIBE is computed from*
(V-JEPA 2 video, Wav2Vec-BERT audio, Llama 3.2 text). Because the brain features are a deterministic
function of those inputs they carry no new information; they can only help as a compact, structured
representation at small training sizes. This repo tests exactly that, honestly, at N≈200.

Prior evidence: arXiv 2607.01400 (N=48 YouTube videos) found no replay-heatmap signal in TRIBE
outputs once a temporal-position control was applied. The controls below are the video-level analogue.

## Preregistered protocol (fixed before any extraction finished)

* **Target**: ECR (released, rank-normalised, uniform on [0,1]). NAWP was never released publicly.
* **Data**: 200 SnapUGC train clips (5–60 s), downloaded by per-row CDN link (≈30 % of links are alive),
  deduplicated (perceptual hash + title), balanced over duration bins 5–20 / 20–40 / 40–60 s as supply allows.
* **Feature sets** (all include the *simple* descriptors: duration, loudness, motion, cuts, transcript words):
  * `A` baseline: per modality PCA of TRIBE's cached 1 Hz embeddings, fit **inside each training draw**, then the
    *same* temporal statistics as the brain parcels (mean, peak, time-to-peak, slope, variance, first-3 s mean).
    Variants `A_pca8`, `A_pca32`, `A_raw`.
  * `C` brain: TRIBE predictions (20,484 fsaverage5 vertices, 1 Hz, index 0 = stimulus onset, lag already
    compensated) averaged into Yeo-7 networks + global field power → same temporal statistics (48 dims).
  * `B` = A ∪ C. `trivial` = simple only.
* **Learner**: `StandardScaler → RidgeCV` (GCV over a log alpha grid). Gradient boosting once at full N as robustness.
* **Evaluation**: 10 × repeated 5-fold CV; learning curve by subsampling n ∈ {50, 100, all} **within** each
  training fold (10 draws); metrics on pooled out-of-fold predictions.
* **Primary endpoint**: Δ Spearman = B_best − A_best at full N (and at n=100), where "best" is chosen by A's own score;
  95 % CI by bootstrap over videos on repeat-averaged OOF predictions. Everything else is secondary/exploratory.
* **Controls**: (1) `C_rand` random linear readout of the identical embeddings, same dims, same stats; (2) `C_mismatch`
  another same-duration-bin video's brain series; (3) duplicates removed before splitting; (4) label-permutation
  sanity run; (5) secondary analysis restricted to the first 5 s (the ECR window).
* **Power**: with N≈200 the minimum detectable Δ at 80 % power is ≈0.10. Expect absolute A ≈ 0.3–0.45 Spearman
  (the challenge winner reaches 0.71 with 100k videos and a 7B video-LLM).

## Layout

```
configs/default.yaml           all knobs (seed, N, bins, stats, PCA dims, CV protocol)
src/brainengage/
  data/select_subset.py        download-by-link, probe, dedupe, balance -> manifest
  features/tribe_extract.py    TRIBE v2 per video -> preds (T,20484) + cached input embeddings   [GPU]
  features/parcellate.py       Yeo-7 (native fsaverage5 annot) / HCP-MMP vertex->network maps
  features/temporal_stats.py   ONE summary function shared by A and C
  features/simple.py           PyAV descriptors (no ffmpeg binary needed)
  features/build.py            feature tables + controls
  modeling/cv.py               repeated K-fold, within-fold subsampling, per-draw PCA, ridge/HGB
  modeling/metrics.py          Spearman/Pearson, video bootstrap
  reporting/plots.py           learning curve, markdown tables
scripts/00..05_*.py            the pipeline, in order; dev_make_synthetic.py for CPU-only dry runs
tests/                         offset/indexing, parcel maps, split balance, CV leakage
results/                       committed outputs (tables, plot, summary.md)
data/                          gitignored caches (videos, tribe_cache, preds, embeddings, features)
```

## Getting the data on a fresh clone

`data/` is **gitignored**: a clone has no videos, no label CSV and no caches. What is committed is the *recipe* plus
`results/manifest.csv`, the exact 200 video ids used. On any new machine:

```bash
.venv/bin/python scripts/01_build_subset.py --from-manifest results/manifest.csv
```
This downloads `train_data.csv` (31 MB, Google Drive) and then exactly those 200 clips (~600 MB) by their CDN links,
and rewrites `data/manifest.parquet`. If some links have died since 2026-11-01 the script says which; the experiment
then runs on the survivors (report the count). Alternatively copy `data/videos/` + `data/manifest.parquet` from a
machine that has them (rsync/scp); the ids are the same either way.

Running the script *without* `--from-manifest` makes a fresh seeded selection, which is only identical if the same
links are still alive. Use that only when deliberately re-sampling or scaling N.

Other files that are re-created automatically: Yeo-7 atlas labels (`data/tribe_cache/atlas`, from GitHub),
TRIBE weights and HF models (`data/tribe_cache`, several GB on the GPU machine). The expensive outputs
(`data/preds`, `data/embeddings`, ≈1 GB for 200 clips) are produced only by step 02 on the GPU; copy them to
wherever steps 03–05 run.

## How to run

### Local (CPU, macOS ok): subset, features, experiments, report
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv -p 3.11 && uv pip install -e ".[dev]"
.venv/bin/python -m pytest -q
.venv/bin/python scripts/01_build_subset.py --from-manifest results/manifest.csv   # exact committed subset
# ... run step 2 on the GPU machine, copy data/preds + data/embeddings back (≈1 GB) ...
.venv/bin/python scripts/03_build_features.py
.venv/bin/python scripts/04_run_experiments.py           # --quick for a 2x3 dev run
.venv/bin/python scripts/05_report.py                    # -> results/summary.md, learning_curve.png
```
CPU-only dry run of steps 3–5 without a GPU (fabricated TRIBE outputs, pipeline test only):
```bash
.venv/bin/python scripts/dev_make_synthetic.py
.venv/bin/python scripts/03_build_features.py  # with tribe.preds_dir/embeddings_dir pointed at data/synthetic (see configs/synthetic.yaml)
```

### GPU (Lightning AI Studio, **L4 24 GB**; a 16 GB T4 OOMs, tribev2 issue #28)
```bash
git clone <this repo> && cd brain-model
bash scripts/setup_gpu_env.sh
source .venv/bin/activate && huggingface-cli login     # needs approved access to meta-llama/Llama-3.2-3B
python scripts/01_build_subset.py --from-manifest results/manifest.csv   # or rsync data/videos + manifest from the Mac
python scripts/00_smoke_test.py --video data/videos/<id>.mp4   # shapes, timing, embedding-cache check
#   add --no-text while Llama access is pending; add -u data.video_feature.image.batch_size=2 on OOM
python scripts/02_extract_tribe.py --limit 5             # measure minutes/clip, then
nohup python scripts/02_extract_tribe.py > data/extract.out 2>&1 &   # resumable; skips finished ids
python scripts/03_build_features.py && python scripts/04_run_experiments.py && python scripts/05_report.py
```
Budget guide: Lightning free tier (card added) = 30 one-time credits ≈ 30 L4-hours at $0.79/h. Keep `data/`
under 10 GB (free storage). If a clip takes > 3 min, lower `data.n_target` rather than overrun.

## Facts verified against sources (2026-11-01)
* TRIBE v2 output `(n_TR, 20484)` float32, 1 s TR, fsaverage5 left hemisphere (10242) then right. The 5 s
  hemodynamic offset is applied on the fMRI side in training (`neuralset` FmriExtractor shifts BOLD by −5 s), so
  `preds[t]` is the lag-compensated response to stimulus second `t`.
* Transcription runs `uvx whisperx` (large-v3) as a subprocess per clip and caches `<video>.tsv`; audio is
  written to `<video>.wav` next to the mp4. `audio_only=True` skips the text stream (used by `--no-text`).
* Extractor features (2 Hz, 3 layers each) are exca-cached in `data/tribe_cache` and readable after `predict`
  via `extractor(events, start, duration)`; `tribe_extract.extract_embeddings` pools them to 1 Hz.
* SnapUGC `train_data.csv`: 106,192 rows, `Id, Title, Description, Download_link, ECR`; ECR uniform (rank-normalised),
  544 exact zeros; 63 % empty titles. No NAWP, no durations, val/test unlabeled.
* Licences: TRIBE v2 code+weights CC-BY-NC-4.0 (research only). SnapUGC: no stated licence; treat as research only.

## Status (2026-11-01)

| Step | State |
|---|---|
| 01 subset | **Done.** 200 clips downloaded (548 live of 1,728 link attempts), bins 126/50/24, see `results/manifest.csv` |
| 00 smoke test | **Not run** (needs the L4 Studio). Verifies shapes, timing and the embedding-cache read |
| 02 TRIBE extraction | **Not run** (GPU) |
| 03–05 features / experiments / report | **Code verified** on fabricated TRIBE outputs (`dev_make_synthetic.py`); real run pending |
| tests | 10 passing |

Real results will appear in `results/summary.md` and `results/learning_curve.png` once step 5 runs on real
extractions. Anything under `results/_synthetic/` is a pipeline test, not a finding.

For agents: read `CLAUDE.md` first (purpose, decisions, gotchas, next steps).

## Company site (`docs/`)

`docs/index.html` is the BrainGage landing page, a single static file with no build step.
It deploys automatically: `.github/workflows/pages.yml` publishes `docs/` to GitHub Pages on every push to
`main` that touches `docs/` (Pages source is set to "GitHub Actions").
