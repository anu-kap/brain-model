"""Step 5: results/ -> results table (md), learning_curve.png, summary.md with a verdict."""
from __future__ import annotations

import argparse
import json

import pandas as pd

from brainengage.config import load_config, resolve
from brainengage.modeling.cv import FEATURE_SETS
from brainengage.reporting.plots import deltas_table_md, learning_curve, results_table_md


def verdict(deltas: pd.DataFrame, meta: dict) -> str:
    d = deltas.set_index(["comparison", "n"])
    def get(c, n):
        return d.loc[(c, n)] if (c, n) in d.index else None
    prim_all, prim_100 = get("primary_B_vs_A_best", "all"), get("primary_B_vs_A_best", "100")
    rand = get("C_vs_random_readout", "all")
    mis = get("C_vs_mismatched_brain", "all")
    brand = get("B_vs_B_random_readout", "all")
    lines = []
    if prim_all is None:
        return "Primary comparison not available (missing feature sets)."
    def fmt(r):
        return f"Δ = {r.point:+.3f}, 95% CI [{r.lo:+.3f}, {r.hi:+.3f}]"
    lines.append(f"- Primary endpoint, full N: B ({meta['best_B']}) − A ({meta['best_A']}): {fmt(prim_all)}.")
    if prim_100 is not None:
        lines.append(f"- Same at n=100 training videos: {fmt(prim_100)}.")
    pos_all = prim_all.lo > 0
    pos_100 = prim_100 is not None and prim_100.lo > 0
    if rand is not None:
        lines.append(f"- Brain vs random readout of the same inputs (C − C_rand): {fmt(rand)}.")
    if mis is not None:
        lines.append(f"- Brain vs mismatched-video brain (C − C_mismatch): {fmt(mis)}.")
    if brand is not None:
        lines.append(f"- B vs B with random readout instead of brain: {fmt(brand)}.")
    controls_ok = (rand is None or rand.lo > 0) and (brand is None or brand.lo > 0)
    if pos_all and controls_ok:
        v = "**Brain features add predictive value beyond the baseline at full N, and the gain survives the random-readout control.**"
    elif pos_100 and controls_ok:
        v = "**Brain features help only at small training sizes (n≈100), consistent with a compact-representation effect; no gain at full N.**"
    elif pos_all or pos_100:
        v = "**B beats A, but a random readout of the same inputs does as well: the gain is a compression/regularisation effect, not brain-specific information.**"
    else:
        v = "**Null result: adding TRIBE brain features does not improve ECR prediction over the input-embedding baseline at any training size tested.**"
    return v + "\n\n" + "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--title", default="Brain-feature engagement POC: results")
    args = ap.parse_args()
    cfg = load_config()
    out = resolve(args.results)
    summary = pd.read_csv(out / "summary_table.csv", dtype={"n": str})
    pooled = pd.read_csv(out / "cv_pooled_results.csv", dtype={"n": str})
    deltas = pd.read_csv(out / "deltas.csv", dtype={"n": str})
    folds = pd.read_csv(out / "cv_fold_results.csv", dtype={"n": str})
    meta = json.loads((out / "run_meta.json").read_text())
    n_train_by_n = folds.groupby("n").n_train.max().to_dict()
    n_rep, n_fold = int(folds.repeat.max()) + 1, int(folds.fold.max()) + 1
    learning_curve(pooled, meta["best_A"], meta["best_B"], n_train_by_n, out / "learning_curve.png")
    order = [s for s in FEATURE_SETS if s in set(summary.set)]
    md = [f"# {args.title}", "",
          f"N = {meta['n_videos']} videos, target = ECR (rank-normalised), learner = ridge (RidgeCV, GCV), "
          f"{n_rep}×{n_fold}-fold CV, pooled OOF metrics"
          + (" **(quick dev run)**" if meta.get("quick") else "") + ".", "",
          "## Verdict", "", verdict(deltas, meta), "",
          "## Results table (mean ± sd over repeats × draws of pooled OOF correlation)", "", results_table_md(summary, order), "",
          "## Paired differences (video bootstrap on repeat-averaged OOF predictions)", "", deltas_table_md(deltas), "",
          "## Learning curve", "", "![learning curve](learning_curve.png)", "",
          "## Sanity checks", "",
          f"- Label-permutation run (should be ≈ 0): {json.dumps({k: round(v, 3) for k, v in meta['permutation_sanity_spearman'].items()})}",
          f"- Feature sets skipped (inputs missing): {meta['sets_skipped']}",
          f"- Elapsed: {meta['elapsed_s']} s", ""]
    hgb = summary[summary.learner == "hgb"]
    if len(hgb):
        md += ["## Robustness: gradient boosting at full N", "", hgb[["set", "spearman_mean", "spearman_sd"]].round(3).to_markdown(index=False), ""]
    md += ["## Caveats", "",
           "- With N≈200 the minimum detectable Δ Spearman (80% power) is roughly 0.10; absolute scores are far below the 0.71 of the 100k-video challenge winner by design.",
           "- Only ~30% of SnapUGC CDN links are still alive; the subset is a survivor sample.",
           "- TRIBE v2 is average-subject, cortex-only (no striatum/accumbens), CC-BY-NC-4.0 (research use only).",
           "- Brain features are a deterministic function of the same embeddings used by Model A; any gain is a representation effect, not new information.", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
