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
    print(f"\nplots → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
