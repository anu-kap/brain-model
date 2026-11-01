"""Step 4: repeated-CV experiments on cached feature tables (CPU). Writes results/*.csv|parquet.
    python scripts/04_run_experiments.py            # full protocol from configs/default.yaml
    python scripts/04_run_experiments.py --quick    # 2 repeats x 3 draws for development
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from brainengage.config import load_config, resolve
from brainengage.modeling.cv import FEATURE_SETS, FeatureStore, ensemble_oof, pooled_metrics, run_cv
from brainengage.modeling.metrics import bootstrap_ci, spearman

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("experiments")


A_VARIANTS = ["A_pca8", "A_pca32", "A_raw"]


def pick_best_variant(pooled: pd.DataFrame) -> tuple[str | None, str | None]:
    """k* = the A variant with the best mean pooled Spearman at n=all (ridge). B uses the SAME k*,
    so the primary comparison is matched and the brain side is never chosen on its own score."""
    sub = pooled[(pooled.set.isin(A_VARIANTS)) & (pooled.n == "all") & (pooled.learner == "ridge")]
    if not len(sub):
        return None, None
    best_a = sub.groupby("set").spearman.mean().idxmax()
    best_b = "B_" + best_a[2:]
    return best_a, (best_b if best_b in set(pooled.set) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--features", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--sets", nargs="*", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    store = FeatureStore(resolve(args.features or cfg["data"].get("features_dir", "data/features")), cfg)
    repeats, draws = (2, 3) if args.quick else (None, None)
    learners = ("ridge", "hgb") if cfg["modeling"].get("hgb_at_full_n", True) else ("ridge",)

    t = time.time()
    rows, oof = run_cv(store, cfg, sets=args.sets, repeats=repeats, draws=draws, learners=learners)
    log.info("CV done in %.0fs: %d fold rows", time.time() - t, len(rows))
    y_by_id = dict(zip(store.ids, store.y))
    pooled = pooled_metrics(oof, y_by_id)
    rows.to_csv(out / "cv_fold_results.csv", index=False)
    pooled.to_csv(out / "cv_pooled_results.csv", index=False)
    oof.to_parquet(out / "oof_predictions.parquet", index=False)

    summary = (pooled.groupby(["set", "learner", "n"]).agg(spearman_mean=("spearman", "mean"), spearman_sd=("spearman", "std"),
                                                           pearson_mean=("pearson", "mean"), pearson_sd=("pearson", "std"),
                                                           runs=("spearman", "size")).reset_index())
    nfeat = rows.groupby("set").n_features.max()
    summary["n_features"] = summary.set.map(nfeat)
    summary.to_csv(out / "summary_table.csv", index=False)

    # ---- paired deltas with video-bootstrap CIs on repeat-averaged OOF predictions ----
    ens = ensemble_oof(oof, store.ids)
    y = store.y
    bestA, bestB = pick_best_variant(pooled)
    pairs = []
    for n in [str(x) for x in cfg["modeling"]["learning_curve_n"]]:
        pairs += [("primary_B_vs_A_best", bestA, bestB, n), ("B_vs_A_pca8", "A_pca8", "B_pca8", n),
                  ("B_vs_A_pca32", "A_pca32", "B_pca32", n), ("B_vs_A_raw", "A_raw", "B_raw", n),
                  ("C_vs_trivial", "trivial", "C", n), ("C_vs_random_readout", "C_rand", "C", n),
                  ("C_vs_mismatched_brain", "C_mismatch", "C", n), ("B_vs_B_random_readout", "B_rand_pca8", "B_pca8", n),
                  ("B_vs_B_mismatched", "B_mismatch_pca8", "B_pca8", n), ("B_w5_vs_A_w5", "A_pca8_w5", "B_pca8_w5", n),
                  ("C_w5_vs_Crand_w5", "C_rand_w5", "C_w5", n), ("B_hcp_vs_A_pca8", "A_pca8", "B_hcp_pca8", n)]
    deltas = []
    for label, a, b, n in pairs:
        ka, kb = (a, "ridge", n), (b, "ridge", n)
        if a is None or b is None or ka not in ens or kb not in ens:
            continue
        ci = bootstrap_ci(y, {a: ens[ka], b: ens[kb]}, spearman, int(cfg["modeling"]["bootstrap_n"]), int(cfg["seed"]), pair=(a, b))
        deltas.append({"comparison": label, "baseline_set": a, "test_set": b, "n": n, **ci,
                       "spearman_a_ens": spearman(y, ens[ka]), "spearman_b_ens": spearman(y, ens[kb])})
    deltas = pd.DataFrame(deltas)
    deltas.to_csv(out / "deltas.csv", index=False)

    # ---- label-permutation sanity run (pipeline should give ~0) ----
    prow, poof = run_cv(store, cfg, sets=["A_pca8", "C"], repeats=1, draws=1, permute_y=True, learners=("ridge",))
    ppooled = pooled_metrics(poof, dict(zip(store.ids, np.random.default_rng(int(cfg["seed"]) + 999).permutation(store.y))))
    perm = ppooled[ppooled.n == "all"].groupby("set").spearman.mean().to_dict()

    meta = {"n_videos": len(store.ids), "best_A": bestA, "best_B": bestB, "quick": args.quick,
            "permutation_sanity_spearman": perm, "sets_run": sorted(pooled.set.unique().tolist()),
            "sets_skipped": [s for s in FEATURE_SETS if s not in set(pooled.set.unique())], "elapsed_s": round(time.time() - t)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(summary[summary.learner == "ridge"].pivot(index="set", columns="n", values="spearman_mean").round(3))
    print(deltas[["comparison", "n", "point", "lo", "hi"]].round(3).to_string(index=False))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
