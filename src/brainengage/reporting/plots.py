from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed categorical slots (validated palette, light surface): identity never cycles.
CURVE_SETS = [("A", "A: baseline inputs", "#2a78d6"), ("B", "B: baseline + brain", "#eb6834"),
              ("C", "C: brain (+simple)", "#1baf7a"), ("trivial", "Trivial (simple descriptors)", "#eda100"),
              ("C_rand", "Control: random readout", "#e87ba4"), ("C_mismatch", "Control: mismatched brain", "#008300")]


def learning_curve(pooled: pd.DataFrame, best_a: str, best_b: str, n_train_by_n: dict[str, int], path: Path) -> None:
    pooled = pooled[pooled.learner == "ridge"]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#e6e6e3", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    for key, label, color in CURVE_SETS:
        sname = {"A": best_a, "B": best_b}.get(key, key)
        if sname is None or sname not in set(pooled.set):
            continue
        g = pooled[pooled.set == sname].groupby("n").spearman
        xs, ms, los, his = [], [], [], []
        for n, vals in g:
            v = vals.to_numpy()
            xs.append(n_train_by_n[str(n)])
            ms.append(v.mean())
            lo, hi = np.percentile(v, [2.5, 97.5]) if len(v) > 2 else (v.mean(), v.mean())
            los.append(lo)
            his.append(hi)
        order = np.argsort(xs)
        xs, ms, los, his = (np.asarray(a)[order] for a in (xs, ms, los, his))
        ax.plot(xs, ms, "-o", color=color, lw=1.5, ms=6, mec="white", mew=1.0,
                label=f"{label} [{sname}]" if key in ("A", "B") else label)
        ax.fill_between(xs, los, his, color=color, alpha=0.10, lw=0)
        ax.annotate(f"{ms[-1]:.2f}", (xs[-1], ms[-1]), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color="#5f5e5a", va="center")
    ax.set_xscale("log")
    ax.set_xticks(sorted(set(n_train_by_n.values())))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("training videos per fold")
    ax.set_ylabel("pooled out-of-fold Spearman (ECR)")
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.tick_params(colors="#5f5e5a")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    ax.set_title("Pooled OOF Spearman vs training size (band = 2.5-97.5 pct over repeats x draws)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def results_table_md(summary: pd.DataFrame, sets_order: list[str]) -> str:
    s = summary[summary.learner == "ridge"].copy()
    ns = [n for n in ["50", "100", "all"] if n in set(s.n.astype(str))]
    s["n"] = s.n.astype(str)
    lines = ["| set | features | " + " | ".join(f"Spearman n={n}" for n in ns) + " | Pearson n=all |",
             "|---|---|" + "|".join("---" for _ in ns) + "|---|"]
    for name in sets_order:
        g = s[s.set == name]
        if g.empty:
            continue
        cells = []
        for n in ns:
            r = g[g.n == n]
            cells.append(f"{r.spearman_mean.iloc[0]:.3f} ± {r.spearman_sd.iloc[0]:.3f}" if len(r) else "")
        ra = g[g.n == "all"]
        pear = f"{ra.pearson_mean.iloc[0]:.3f}" if len(ra) else ""
        lines.append(f"| {name} | {int(g.n_features.iloc[0])} | " + " | ".join(cells) + f" | {pear} |")
    return "\n".join(lines)


def deltas_table_md(deltas: pd.DataFrame) -> str:
    lines = ["| comparison | n | Δ Spearman (test − baseline) | 95% CI | P(Δ≤0) |", "|---|---|---|---|---|"]
    for r in deltas.itertuples(index=False):
        lines.append(f"| {r.comparison} ({r.test_set} − {r.baseline_set}) | {r.n} | {r.point:+.3f} | [{r.lo:+.3f}, {r.hi:+.3f}] | {r.p_le_0:.2f} |")
    return "\n".join(lines)
