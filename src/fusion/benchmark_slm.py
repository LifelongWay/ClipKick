"""
Benchmark several SLMs on the Model F highlight-detection task.

All SLMs read the SAME cached Granite transcript (Granite runs once per match), so
this is cheap after transcription. For each (model, match) it records F3 metrics +
runtime into a tidy table, and keeps each model's highlights so nothing is overwritten.

    python src/fusion/benchmark_slm.py                       # all default SLMs × all matches
    python src/fusion/benchmark_slm.py --models Qwen/Qwen2.5-1.5B-Instruct --matches <id>
    python src/fusion/benchmark_slm.py --models a,b,c        # custom set

Gemma-2-2b-it is gated: set HF_TOKEN (and accept its license) or it is recorded as skipped.
Outputs:
    results/benchmark/results.csv   (long: slm,match,scope,P,R,F1,tp,fp,fn,n_pred,n_truth,runtime_sec)
    results/benchmark/summary.csv   (per-SLM headline aggregates)
Then plot with:  python src/fusion/plot_benchmark.py
"""

import argparse
import os
import time

import pandas as pd

try:
    from . import common, evaluate, fuse_slm, timeline
except ImportError:
    import common, evaluate, fuse_slm, timeline

DEFAULT_MODELS = [
    # small → large, so cheap results land first and a big-model OOM costs least
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "google/gemma-2-2b-it",          # gated — needs HF_TOKEN + accepted license
    "Qwen/Qwen2.5-7B-Instruct",      # ~15 GB bf16 — use A100
    "google/gemma-2-9b-it",          # ~18 GB bf16, gated — use A100 + HF_TOKEN
    "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",  # 4-bit 32B (~20 GB) — needs bitsandbytes, use A100
]

OUT_DIR = "results/benchmark"


def _audio_for(match_id):
    p = os.path.join("data/raw/audio", match_id + ".mp3")
    return p if os.path.exists(p) else None


def _discover_matches():
    """Match ids from the audio-energy layer, the audio folder, or cached transcripts."""
    ids = set(timeline._all_match_ids())
    for d in ("data/raw/audio", fuse_slm.TRANSCRIPT_DIR):
        if os.path.isdir(d):
            ids.update(os.path.splitext(f)[0] for f in os.listdir(d)
                       if f.endswith((".mp3", ".wav", ".csv")))
    return sorted(ids)


def _rows_from_report(slm, match_id, report, runtime):
    """Flatten an evaluate() report dict into long-format rows (overall + per type)."""
    rows = []
    o = report["overall"]
    rows.append({"slm": slm, "match": match_id, "scope": "overall",
                 "precision": o["precision"], "recall": o["recall"], "f1": o["f1"],
                 "tp": o["tp"], "fp": o["fp"], "fn": o["fn"],
                 "n_pred": report["n_pred"], "n_truth": report["n_truth"],
                 "runtime_sec": round(runtime, 2)})
    for etype, m in report["per_type"].items():
        rows.append({"slm": slm, "match": match_id, "scope": etype,
                     "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
                     "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                     "n_pred": "", "n_truth": "", "runtime_sec": ""})
    return rows


def run(models, matches):
    os.makedirs(OUT_DIR, exist_ok=True)
    # Build every transcript once up-front (Granite), so the model loop is GPU-light.
    for mid in matches:
        audio = _audio_for(mid)
        if audio:
            fuse_slm.build_or_load_transcript(mid, audio)

    all_rows = []
    for model_id in models:
        tag = fuse_slm.model_tag(model_id)
        print(f"\n========== SLM: {model_id} ({tag}) ==========")
        try:
            slm = fuse_slm.SLMExtractor(model_id)
        except Exception as e:
            print(f"[SKIP] could not load {model_id}: {e}")
            all_rows.append({"slm": tag, "match": "*", "scope": "skipped",
                             "precision": "", "recall": "", "f1": "", "tp": "", "fp": "",
                             "fn": "", "n_pred": "", "n_truth": "", "runtime_sec": ""})
            continue

        for mid in matches:
            audio = _audio_for(mid)
            if not audio:
                print(f"[{mid}] no audio — skip")
                continue
            try:
                t0 = time.perf_counter()
                hl = fuse_slm.run_match(mid, audio, model_id=model_id, slm=slm, tag=tag)
                runtime = time.perf_counter() - t0
                report = evaluate.evaluate(hl, mid, verbose=True)
                all_rows.extend(_rows_from_report(tag, mid, report, runtime))
            except Exception as e:  # one match failing must not lose the whole benchmark
                print(f"[{mid}] {tag} FAILED: {type(e).__name__}: {e}")
                all_rows.append({"slm": tag, "match": mid, "scope": "failed",
                                 "precision": "", "recall": "", "f1": "", "tp": "", "fp": "",
                                 "fn": "", "n_pred": "", "n_truth": "", "runtime_sec": ""})
            # persist after every match so a later crash never wipes earlier results
            pd.DataFrame(all_rows).to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)

        try:
            slm.close()  # free GPU before the next model
        except Exception:
            pass

    results = pd.DataFrame(all_rows)
    results_path = os.path.join(OUT_DIR, "results.csv")
    results.to_csv(results_path, index=False)
    print(f"\nwrote {results_path}")

    _write_summary(results)
    return results


def _write_summary(results):
    """Per-SLM headline: overall P/R/F1 (micro over matches) + means."""
    ov = results[results["scope"] == "overall"].copy()
    if ov.empty:
        print("no successful runs to summarize")
        return
    for c in ["tp", "fp", "fn", "n_pred", "n_truth", "runtime_sec", "f1", "precision", "recall"]:
        ov[c] = pd.to_numeric(ov[c], errors="coerce")

    rows = []
    for slm, g in ov.groupby("slm"):
        tp, fp, fn = g["tp"].sum(), g["fp"].sum(), g["fn"].sum()
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        rows.append({"slm": slm,
                     "micro_precision": round(p, 4), "micro_recall": round(r, 4),
                     "micro_f1": round(f1, 4),
                     "macro_f1": round(g["f1"].mean(), 4),
                     "mean_n_pred": round(g["n_pred"].mean(), 1),
                     "mean_n_truth": round(g["n_truth"].mean(), 1),
                     "mean_runtime_sec": round(g["runtime_sec"].mean(), 1)})
    summary = pd.DataFrame(rows).sort_values("micro_f1", ascending=False)
    path = os.path.join(OUT_DIR, "summary.csv")
    summary.to_csv(path, index=False)
    print(f"wrote {path}\n")
    print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark SLMs on Model F")
    parser.add_argument("--models", default=None, help="comma-separated HF ids (default: built-in set)")
    parser.add_argument("--matches", default=None, help="comma-separated match ids (default: all)")
    args = parser.parse_args()

    models = args.models.split(",") if args.models else DEFAULT_MODELS
    matches = args.matches.split(",") if args.matches else _discover_matches()
    if not matches:
        print("no matches found (need data/raw/audio/<id>.mp3, a cached transcript, "
              "or results/audio/rms) — or pass --matches <id>")
        return
    print(f"Models: {models}\nMatches: {matches}")
    run(models, matches)


if __name__ == "__main__":
    main()
