# ClipKick — Commands Reference

How to run every model and the shared stages. Replace `<id>` with a match id, e.g.
`2018WC_Portugal_vs_Spain` (the audio file is `data/raw/audio/<id>.mp3`).

> **In Colab notebooks** prefix each line with `!`, and inject the Python variable with `{M}`,
> e.g. `M = "2018WC_Portugal_vs_Spain"` then `!python ... --match {M}`.

---

## 0. Setup

**Local (this repo's venv):** the audio layer runs on CPU. The speech/vision/SLM models need a GPU,
so run those on **Colab (A100 recommended)** or Apple Silicon.

```bash
pip install -r requirements-gpu.txt          # torch, transformers, bitsandbytes, moviepy, …
export HF_TOKEN=...                           # needed for gated models (gemma-2-*)
```
Colab equivalent: `import os; os.environ["HF_TOKEN"] = "..."`.

---

## 1. The two model families

| Family | Models | Self-contained? |
|---|---|---|
| **Cascade / fusion** | A (weighted), B (learned), E (two-pass) | A/B need shared stages; **E is self-contained** |
| **Speech→LLM** | F (SLM) + the SLM benchmark | **F is self-contained (ONE command)** |

"Self-contained" = the one command runs Granite (cached) itself and needs no separate stages.

---

## 2. Shared prerequisite stages (for Models A & B)

These produce the inputs A/B consume. Run once per match (cheap; cached where noted).

```bash
# Audio energy layer — processes ALL data/raw/audio/*.mp3 → results/audio/{events,rms}
python src/audio/audio_layer.py

# Speech layer (F1, Granite) — one match → results/speech/{events,score}
python src/speech/speech_layer.py --audio data/raw/audio/<id>.mp3 --mode sliding

# Vision layer (gated to audio windows) → results/vision/{events,windows}  [needs the video]
python src/video/process_video.py --input <video.mp4> --audio data/raw/audio/<id>.mp3 \
    --output out.mp4 --events results/audio/events/<id>.csv

# Timeline (F2) — fuses the three layers onto a 1s grid → results/fusion/features/<id>.csv
python src/fusion/timeline.py --match <id>
```

---

## 3. Ground truth (F4) — optional, regenerates annotations from Granite

```bash
python src/speech/build_ground_truth.py --match <id>        # overwrites data/annotations/<id>.json
python src/speech/build_ground_truth.py                     # all matches
python src/speech/build_ground_truth.py --match <id> --fp32 # if MPS fp16 misbehaves
```
> Backs up the original human labels to `data/annotations_original/` first. The 2018WC match already
> has hand-made 9-event annotations — don't overwrite those unless you want Granite's version.

---

## 4. Run each model

### Model A — weighted decision fusion
Needs stages in §2 first (audio + speech + vision + timeline).
```bash
python src/fusion/fuse_decision.py --match <id> --eval     # uses default weights, prints P/R/F1
python src/fusion/fuse_decision.py --tune                  # grid-search weights across matches
```

### Model B — learned stacked classifier (leave-one-match-out)
Needs §2 stages for **every** match (it trains on the others to predict each one).
```bash
python src/fusion/fuse_stacked.py                          # LOMO over all matches
python src/fusion/fuse_stacked.py --match <id>             # train on the rest, predict this one
```

### Model E — progressive two-pass (self-contained, drives Granite itself)
Needs only the audio layer (§2, line 1) for the peaks.
```bash
python src/audio/audio_layer.py                            # once (for the peaks)
python src/fusion/fuse_progressive.py --match <id> --audio data/raw/audio/<id>.mp3 --eval
```

### Model F — SLM (self-contained, ONE command) ★
Builds the Granite transcript on first run (cached), then runs the SLM. **No separate transcribe step.**
```bash
# basic
python src/fusion/fuse_slm.py --match <id> --audio data/raw/audio/<id>.mp3 --eval

# swap the SLM (reuses the cached transcript — NO Granite rerun)
python src/fusion/fuse_slm.py --match <id> --model Qwen/Qwen2.5-7B-Instruct --eval

# raise precision (drop low-confidence detections)
python src/fusion/fuse_slm.py --match <id> --min-confidence 0.6 --eval

# from a video (extracts mp3) and also export clips
python src/fusion/fuse_slm.py --match <id> --video <video.mp4> --clips --eval

# force a fresh transcription (ignore the cache)
python src/fusion/fuse_slm.py --match <id> --audio data/raw/audio/<id>.mp3 --no-cache

# ONLY build/cache the Granite transcript, no SLM stage (→ results/speech/transcript/<id>.csv)
python src/fusion/fuse_slm.py --match <id> --audio data/raw/audio/<id>.mp3 --transcript-only
```

---

## 5. Orchestrated pipeline (one command end-to-end)

`run_pipeline.py` runs the right stages for the chosen model. A/B run audio→speech→vision→timeline→fusion;
**E and F skip those and run self-contained.**
```bash
python run_pipeline.py --video <video.mp4> --audio data/raw/audio/<id>.mp3 --output out.mp4 --fusion A
python run_pipeline.py --video x --audio data/raw/audio/<id>.mp3 --output out.mp4 --fusion E
python run_pipeline.py --video x --audio data/raw/audio/<id>.mp3 --output out.mp4 --fusion F
python run_pipeline.py --video <video.mp4> --audio <...> --output out.mp4 --fusion A --skip-vision
```
Flags: `--fusion {A,B,E,F,none}` (default A), `--speech-mode {sliding,gated}`, `--skip-vision`.

---

## 6. SLM benchmark (compare many SLMs on Model F)

Transcribes once per match (cached), then runs each SLM and scores it.
```bash
python src/fusion/benchmark_slm.py --matches <id>          # all 7 default SLMs on one match
python src/fusion/benchmark_slm.py                         # default SLMs × all matches
python src/fusion/benchmark_slm.py --models "Qwen/Qwen2.5-7B-Instruct,google/gemma-2-9b-it" --matches <id>

python src/fusion/plot_benchmark.py                        # → results/benchmark/plots/*.png
```
Outputs: `results/benchmark/results.csv`, `summary.csv`, `plots/*.png`.
Default models: SmolLM2-1.7B, Qwen2.5-1.5B, Phi-3.5-mini, gemma-2-2b-it, Qwen2.5-7B, gemma-2-9b-it,
Qwen2.5-32B-Instruct-bnb-4bit. **Use A100**; set `HF_TOKEN` for the gemma models.

---

## 7. Evaluate / export (work on any model's highlights)

```bash
# Score a highlights CSV against ground truth (per-type + overall P/R/F1)
python src/fusion/evaluate.py --pred results/fusion/highlights/<id>.csv --match <id>

# Cut video clips from a highlights CSV (+ optional stitched reel)
python src/fusion/export_clips.py --match <id> --video <video.mp4> --reel
python src/fusion/export_clips.py --highlights results/fusion/highlights/<id>__<tag>.csv --video <video.mp4>
```

---

## 8. Where things land

| Output | Path |
|---|---|
| Audio events / RMS | `results/audio/{events,rms}/<id>.csv` |
| Speech layer (F1) | `results/speech/{events,score}/<id>.csv` |
| **Granite transcript (cached, Model F)** | `results/speech/transcript/<id>.csv` |
| Vision | `results/vision/{events,windows}/<id>.csv` |
| Timeline (F2) | `results/fusion/features/<id>.csv` |
| Highlights | `results/fusion/highlights/<id>.csv` (benchmark: `<id>__<tag>.csv` + `_detail.csv`) |
| Benchmark | `results/benchmark/{results,summary}.csv`, `plots/*.png` |
| Clips | `results/clips/<id>/*.mp4` |
| Ground truth | `data/annotations/<id>.json` |

---

## 9. Flag reference (every script)

### `run_pipeline.py` — end-to-end orchestrator
| Flag | What it's for |
|---|---|
| `--video` (req) | source match video (passed to the vision stage; use a dummy like `x` for E/F or with `--skip-vision`) |
| `--audio` (req) | match mp3; its basename is the match id |
| `--output` (req) | output annotated video path for the vision stage |
| `--fusion {A,B,E,F,none}` | which fusion model to run (default `A`; `none` = stages only) |
| `--speech-mode {sliding,gated}` | F1 mode: whole match (`sliding`) vs only audio-event windows (`gated`) |
| `--skip-vision` | skip the expensive vision stage (A/B without vision evidence) |

### `src/audio/audio_layer.py` — energy layer
*No flags* — processes every file in `data/raw/audio/`.

### `src/speech/build_ground_truth.py` — F4 (Granite → annotations)
| Flag | What it's for |
|---|---|
| `--match` | only this match (default: all) |
| `--audio-dir` | where the mp3s live (default `data/raw/audio`) |
| `--out-dir` | where annotations are written (default `data/annotations`) |
| `--fp32` | force float32 if MPS fp16 produces empty/NaN output |

### `src/speech/speech_layer.py` — F1 (speech signals for A/B)
| Flag | What it's for |
|---|---|
| `--audio` (req) | match mp3 to transcribe |
| `--mode {sliding,gated}` | whole match vs only audio-event windows |
| `--fp32` | force float32 (MPS safety) |

### `src/fusion/timeline.py` — F2 (shared feature grid)
| Flag | What it's for |
|---|---|
| `--match` | build the timeline for this match (default: all with audio RMS) |

### `src/fusion/fuse_decision.py` — Model A
| Flag | What it's for |
|---|---|
| `--match` | which match to score (default: all) |
| `--tune` | grid-search the audio/speech/vision weights + threshold against F3 |
| `--eval` | print per-type + overall P/R/F1 after producing highlights |

### `src/fusion/fuse_stacked.py` — Model B
| Flag | What it's for |
|---|---|
| `--match` | train on the other matches, predict this one (default: leave-one-match-out over all) |
| `--threshold` | min predicted probability to emit a highlight (precision/recall trade) |

### `src/fusion/fuse_progressive.py` — Model E
| Flag | What it's for |
|---|---|
| `--match` | which match (default: all) |
| `--audio` | match mp3 (defaults to `data/raw/audio/<match>.mp3`) — E transcribes it in two passes |
| `--eval` | print P/R/F1 after |

### `src/fusion/fuse_slm.py` — Model F ★
| Flag | What it's for |
|---|---|
| `--match` | which match (default: all) |
| `--audio` | match mp3 (defaults to `data/raw/audio/<match>.mp3`) |
| `--video` | use a video instead — extracts the mp3 from it first |
| `--model` | which SLM to use (default SmolLM2-1.7B-Instruct); **swapping reuses the cached transcript** |
| `--eval` | print P/R/F1 after |
| `--no-cache` | ignore the cached transcript and re-run Granite from scratch |
| `--clips` | after highlights, export video clips (requires `--video`) |
| `--min-confidence` | drop detections below this score — the **precision knob** (0 = keep all) |
| `--transcript-only` | build/cache the Granite transcript and exit — **no SLM stage** |

### `src/fusion/benchmark_slm.py` — SLM benchmark
| Flag | What it's for |
|---|---|
| `--models` | comma-separated HF model ids to compare (default: the built-in 7) |
| `--matches` | comma-separated match ids (default: all discoverable) |

### `src/fusion/plot_benchmark.py` — graphs
| Flag | What it's for |
|---|---|
| `--results` | benchmark results CSV to plot (default `results/benchmark/results.csv`) |

### `src/fusion/evaluate.py` — F3 scorer
| Flag | What it's for |
|---|---|
| `--pred` (req) | highlights CSV to score |
| `--match` | match id for the ground truth (defaults to the CSV's basename) |
| `--ann-dir` | annotations directory (default `data/annotations`) |

### `src/fusion/export_clips.py` — clip exporter
| Flag | What it's for |
|---|---|
| `--match` | locate `results/fusion/highlights/<match>.csv` automatically |
| `--highlights` | use this highlights CSV instead of `--match`'s default |
| `--video` (req) | source video the clips are cut from |
| `--out-dir` | where clips are written (default `results/clips/<match>/`) |
| `--reel` | also concatenate the clips into one highlight reel |
| `--sort {time,score}` | order clips by match time or by score |

---

## 10. Common notes

- **Granite runs once per match.** The transcript is cached at `results/speech/transcript/<id>.csv`;
  Models F and the benchmark reuse it (`using cached transcript …`). Only rerun Granite if that file
  is gone (e.g. Colab reset) — then any F command rebuilds it automatically.
- **`results/` is gitignored.** On Colab, copy what you want to keep to Drive, e.g.
  `cp -r results/speech/transcript results/benchmark /content/drive/MyDrive/clipkick_data/`.
- **GPU:** small SLMs / ≤9B fit L4; the 32B-4bit and 7B/9B in bf16 want **A100**. OOM on a model is
  recorded as `failed` and the benchmark continues.
- **Gated models** (`gemma-2-*`) need `HF_TOKEN` + accepted license, else they're `skipped`.
