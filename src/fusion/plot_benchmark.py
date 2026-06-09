"""
Plot the SLM benchmark results produced by benchmark_slm.py.

Reads results/benchmark/results.csv and writes PNG charts to results/benchmark/plots/:
  overall_prf_by_slm.png      grouped P/R/F1 per SLM (headline)
  f1_by_type_by_slm.png       F1 per event type per SLM
  npred_vs_truth_by_slm.png   predictions vs ground-truth count (over-firing)
  runtime_by_slm.png          mean SLM runtime per match

    python src/fusion/plot_benchmark.py
    python src/fusion/plot_benchmark.py --results results/benchmark/results.csv
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")  # headless (Colab / no display)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import common
except ImportError:
    import common

PLOTS_DIR = "results/benchmark/plots"


def _num(df, cols):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def plot_overall_prf(overall):
    slms = list(overall["slm"])
    x = np.arange(len(slms))
    w = 0.25
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(slms)), 5))
    ax.bar(x - w, overall["precision"], w, label="Precision")
    ax.bar(x,     overall["recall"],    w, label="Recall")
    ax.bar(x + w, overall["f1"],        w, label="F1")
    ax.set_xticks(x); ax.set_xticklabels(slms, rotation=20, ha="right")
    ax.set_ylim(0, 1); ax.set_ylabel("score")
    ax.set_title("Model F — overall detection P/R/F1 by SLM")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "overall_prf_by_slm.png")


def plot_f1_by_type(results):
    df = results[results["scope"].isin(common.EVENT_TYPES)]
    if df.empty:
        return
    pivot = df.pivot_table(index="scope", columns="slm", values="f1", aggfunc="mean")
    pivot = pivot.reindex([t for t in common.EVENT_TYPES if t in pivot.index])
    types = list(pivot.index); slms = list(pivot.columns)
    x = np.arange(len(types)); w = 0.8 / max(1, len(slms))
    fig, ax = plt.subplots(figsize=(max(7, 1.8 * len(types)), 5))
    for i, slm in enumerate(slms):
        ax.bar(x + (i - (len(slms) - 1) / 2) * w, pivot[slm].values, w, label=slm)
    ax.set_xticks(x); ax.set_xticklabels(types)
    ax.set_ylim(0, 1); ax.set_ylabel("F1")
    ax.set_title("Model F — F1 per event type by SLM")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "f1_by_type_by_slm.png")


def plot_npred_vs_truth(overall):
    slms = list(overall["slm"])
    x = np.arange(len(slms)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(slms)), 5))
    ax.bar(x - w / 2, overall["n_pred"],  w, label="predicted")
    ax.bar(x + w / 2, overall["n_truth"], w, label="ground truth")
    ax.set_xticks(x); ax.set_xticklabels(slms, rotation=20, ha="right")
    ax.set_ylabel("count (summed over matches)")
    ax.set_title("Model F — predictions vs ground truth by SLM (over-firing)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "npred_vs_truth_by_slm.png")


def plot_runtime(overall):
    slms = list(overall["slm"])
    fig, ax = plt.subplots(figsize=(max(7, 1.6 * len(slms)), 5))
    x = np.arange(len(slms))
    ax.bar(x, overall["runtime_sec"])
    ax.set_xticks(x); ax.set_xticklabels(slms, rotation=20, ha="right")
    ax.set_ylabel("mean runtime per match (s)")
    ax.set_title("Model F — SLM runtime by model")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "runtime_by_slm.png")


def plot_confusion_matrices(conf_dir="results/benchmark/confusion"):
    """One heatmap per method from results/benchmark/confusion/<method>.csv."""
    if not os.path.isdir(conf_dir):
        return
    for fname in sorted(os.listdir(conf_dir)):
        if not fname.endswith(".csv"):
            continue
        method = fname[:-4]
        df = pd.read_csv(os.path.join(conf_dir, fname), index_col=0)
        mat = df.values.astype(float)
        fig, ax = plt.subplots(figsize=(1.3 * df.shape[1] + 2, 1.3 * df.shape[0] + 1.5))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(df.shape[1])); ax.set_xticklabels(df.columns, rotation=30, ha="right")
        ax.set_yticks(range(df.shape[0])); ax.set_yticklabels(df.index)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"Confusion — {method}")
        thr = mat.max() / 2 if mat.max() else 0.5
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                v = int(mat[i, j])
                ax.text(j, i, v, ha="center", va="center",
                        color="white" if mat[i, j] > thr else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        _save(fig, f"confusion_{method}.png")


def plot_a_vs_best_f(overall):
    """Headline bar: Model A vs the best-F SLM (by F1), if both present."""
    if "model_A" not in set(overall["slm"]):
        return
    f_rows = overall[overall["slm"] != "model_A"]
    if f_rows.empty:
        return
    best_f = f_rows.sort_values("f1", ascending=False).iloc[0]
    a = overall[overall["slm"] == "model_A"].iloc[0]
    labels = [f"Model A\n(keyword)", f"Model F\n({best_f['slm']})"]
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics)); w = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - w / 2, [a[m] for m in metrics], w, label=labels[0])
    ax.bar(x + w / 2, [best_f[m] for m in metrics], w, label=labels[1])
    ax.set_xticks(x); ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1); ax.set_title("Headline: Model A vs best Model F")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    _save(fig, "headline_A_vs_bestF.png")


def main():
    parser = argparse.ArgumentParser(description="Plot SLM benchmark results")
    parser.add_argument("--results", default="results/benchmark/results.csv")
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"no results at {args.results} — run benchmark_slm.py first")
        return
    results = _num(pd.read_csv(args.results),
                   ["precision", "recall", "f1", "tp", "fp", "fn",
                    "n_pred", "n_truth", "runtime_sec"])

    ov = results[results["scope"] == "overall"]
    if ov.empty:
        print("no 'overall' rows to plot")
        return
    overall = ov.groupby("slm", as_index=False).agg(
        precision=("precision", "mean"), recall=("recall", "mean"), f1=("f1", "mean"),
        n_pred=("n_pred", "sum"), n_truth=("n_truth", "sum"),
        runtime_sec=("runtime_sec", "mean")).sort_values("f1", ascending=False)

    plot_overall_prf(overall)
    plot_f1_by_type(results)
    plot_npred_vs_truth(overall)
    plot_runtime(overall)
    plot_a_vs_best_f(overall)
    plot_confusion_matrices()
    print(f"\nplots → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
