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
    from . import common, evaluate, fuse_slm, fuse_decision, fuse_hybrid, timeline
except ImportError:
    import common, evaluate, fuse_slm, fuse_decision, fuse_hybrid, timeline

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
CONF_DIR = os.path.join(OUT_DIR, "confusion")


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


def _failed_row(method, mid, scope="failed"):
    return {"slm": method, "match": mid, "scope": scope, "precision": "", "recall": "",
            "f1": "", "tp": "", "fp": "", "fn": "", "n_pred": "", "n_truth": "", "runtime_sec": ""}


def _score_match(method, mid, hl, runtime, all_rows, conf_acc):
    """Evaluate one method's highlights on one match: P/R/F1 rows + confusion accumulation."""
    report = evaluate.evaluate(hl, mid, verbose=True)
    all_rows.extend(_rows_from_report(method, mid, report, runtime))
    conf = evaluate.confusion_counts(hl, common.load_truth(mid))
    conf_acc[method] = evaluate.add_confusion(conf_acc.get(method), conf)


def run(models, matches, with_model_a=False, with_model_g=False,
        g_model="Qwen/Qwen2.5-7B-Instruct"):
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CONF_DIR, exist_ok=True)
    # Build every transcript once up-front (Granite), so the model loop is GPU-light.
    for mid in matches:
        audio = _audio_for(mid)
        if audio:
            fuse_slm.build_or_load_transcript(mid, audio)

    all_rows, conf_acc = [], {}

    def persist():
        df = pd.DataFrame(all_rows)
        df.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
        _write_summary(df, verbose=False)   # keep summary.csv current even if interrupted

    # ── Model A (audio + speech keywords, no vision) — rule-based, no GPU model ──
    if with_model_a:
        method = "model_A"
        print(f"\n========== METHOD: {method} (audio + speech keywords, no vision) ==========")
        for mid in matches:
            try:
                t0 = time.perf_counter()
                hl = fuse_decision.run_match(mid, write=False)   # reads F2 features + candidates
                runtime = time.perf_counter() - t0
                common.write_highlights(mid, hl, suffix="__" + method)
                _score_match(method, mid, hl, runtime, all_rows, conf_acc)
            except Exception as e:
                print(f"[{mid}] {method} FAILED: {type(e).__name__}: {e} "
                      f"(did you run audio_layer + speech_layer --from-transcript + timeline?)")
                all_rows.append(_failed_row(method, mid))
            persist()
        if method in conf_acc:
            evaluate.save_confusion(conf_acc[method], os.path.join(CONF_DIR, method + ".csv"))

    # ── Model G — SLM-verified cascade (Model A candidates → SLM verifier) ──
    if with_model_g:
        method = "g-" + fuse_slm.model_tag(g_model)
        print(f"\n========== METHOD: {method} (Model G — verify Model A with {g_model}) ==========")
        try:
            verifier = fuse_slm.SLMExtractor(g_model)
        except Exception as e:
            print(f"[SKIP] could not load verifier {g_model}: {e}")
            all_rows.append(_failed_row(method, "*", scope="skipped"))
            persist()
        else:
            for mid in matches:
                audio = _audio_for(mid)
                if not audio:
                    print(f"[{mid}] no audio — skip")
                    continue
                try:
                    t0 = time.perf_counter()
                    hl = fuse_hybrid.run_match(mid, audio, slm=verifier, tag=method)
                    runtime = time.perf_counter() - t0
                    _score_match(method, mid, hl, runtime, all_rows, conf_acc)
                except Exception as e:
                    print(f"[{mid}] {method} FAILED: {type(e).__name__}: {e}")
                    all_rows.append(_failed_row(method, mid))
                persist()
            if method in conf_acc:
                evaluate.save_confusion(conf_acc[method], os.path.join(CONF_DIR, method + ".csv"))
            try:
                verifier.close()
            except Exception:
                pass

    # ── Model F across SLMs ──
    for model_id in models:
        tag = fuse_slm.model_tag(model_id)
        print(f"\n========== METHOD: {tag} (Model F / SLM) ==========")
        try:
            slm = fuse_slm.SLMExtractor(model_id)
        except Exception as e:
            print(f"[SKIP] could not load {model_id}: {e}")
            all_rows.append(_failed_row(tag, "*", scope="skipped"))
            persist()
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
                _score_match(tag, mid, hl, runtime, all_rows, conf_acc)
            except Exception as e:  # one match failing must not lose the whole benchmark
                print(f"[{mid}] {tag} FAILED: {type(e).__name__}: {e}")
                all_rows.append(_failed_row(tag, mid))
            persist()  # after every match so a later crash never wipes earlier results

        if tag in conf_acc:
            evaluate.save_confusion(conf_acc[tag], os.path.join(CONF_DIR, tag + ".csv"))
        try:
            slm.close()  # free GPU before the next model
        except Exception:
            pass

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(OUT_DIR, "results.csv"), index=False)
    print(f"\nwrote {os.path.join(OUT_DIR, 'results.csv')} + confusion/*.csv")
    _write_summary(results)
    return results


def _write_summary(results, verbose=True):
    """Per-method headline: overall P/R/F1 (micro over matches) + means."""
    if results.empty or "scope" not in results.columns:
        return
    ov = results[results["scope"] == "overall"].copy()
    if ov.empty:
        if verbose:
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
    if verbose:
        print(f"wrote {path}\n")
        print(summary.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Benchmark SLMs on Model F")
    parser.add_argument("--models", default=None, help="comma-separated HF ids (default: built-in set)")
    parser.add_argument("--matches", default=None, help="comma-separated match ids (default: all)")
    parser.add_argument("--with-model-a", action="store_true",
                        help="also benchmark Model A (needs audio_layer + speech_layer + timeline first)")
    parser.add_argument("--with-model-g", action="store_true",
                        help="also benchmark Model G (SLM-verified cascade over Model A candidates)")
    parser.add_argument("--g-model", default="Qwen/Qwen2.5-7B-Instruct",
                        help="verifier SLM for Model G")
    args = parser.parse_args()

    if args.models is None:
        models = DEFAULT_MODELS
    elif args.models.strip().lower() in ("", "none"):
        models = []                       # run no Model-F SLMs (e.g. only Model A and/or G)
    else:
        models = args.models.split(",")
    matches = args.matches.split(",") if args.matches else _discover_matches()
    if not matches:
        print("no matches found (need data/raw/audio/<id>.mp3, a cached transcript, "
              "or results/audio/rms) — or pass --matches <id>")
        return
    extra = (("model_A + " if args.with_model_a else "")
             + (f"model_G({args.g_model}) + " if args.with_model_g else ""))
    print(f"Methods: {extra}{models}\nMatches: {matches}")
    run(models, matches, with_model_a=args.with_model_a,
        with_model_g=args.with_model_g, g_model=args.g_model)


if __name__ == "__main__":
    main()
