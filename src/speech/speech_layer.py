"""
F1 — Speech layer (production wrapper around Granite).

Runs Granite Speech over a match and emits two aligned artifacts the fusion stage
consumes:
  * results/speech/events/<match>.csv  — typed keyword hits
        time_sec,end_sec,type,keyword,confidence,excitement,word_rate
  * results/speech/score/<match>.csv   — per-second continuous signal
        time_sec,excitement,word_rate,asr_conf

Two modes:
  * sliding (default) — whole match in overlapping windows (recall / weak-label mining)
  * gated            — only windows around audio events (cheap; the cascade default)

Reuses the model wrapper + lexicon from build_ground_truth.py (F4) so there is one
Granite code path. Run from project root:

    python src/speech/speech_layer.py --audio data/raw/audio/<match>.mp3
    python src/speech/speech_layer.py --audio <...> --mode gated
"""

import argparse
import os
import re
import warnings

import librosa
import numpy as np
import pandas as pd

# Same-directory import (script dir is on sys.path when run as `python src/speech/...`).
from build_ground_truth import (
    GraniteTranscriber, compile_lexicon, match_events, pick_device,
    LEXICON, MODEL_ID, TARGET_SR, WINDOW_SEC, HOP_SEC, SILENCE_THRESH,
)
import torch

warnings.filterwarnings("ignore")

EVENTS_DIR = "results/speech/events"
SCORE_DIR  = "results/speech/score"
AUDIO_EVENTS_DIR = "results/audio/events"

# Gated-mode window around each audio event (asymmetric: commentary lags the moment).
GATE_PRE  = 5
GATE_POST = 15

_ELONG = re.compile(r"(\w)\1\1+")  # a character repeated 3+ times → "goooal"


def excitement_score(text):
    """Cheap lexical excitement proxy from elongated words and exclamations (0..1)."""
    if not text:
        return 0.0
    n_elong = len(_ELONG.findall(text))
    n_excl = text.count("!")
    return float(min(1.0, 0.25 * n_elong + 0.1 * n_excl))


def sliding_windows(duration, window_sec=WINDOW_SEC, hop_sec=HOP_SEC):
    return [(s, s + window_sec) for s in range(0, int(duration) + 1, hop_sec)]


def gated_windows(match_id):
    path = os.path.join(AUDIO_EVENTS_DIR, match_id + ".csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return [(max(0.0, float(t) - GATE_PRE), float(t) + GATE_POST) for t in df["time_sec"]]


def run_speech(audio_path, transcriber, compiled, mode="sliding"):
    match_id = os.path.splitext(os.path.basename(audio_path))[0]
    print(f"[{match_id}] loading audio…")
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = len(y) / sr

    if mode == "gated":
        wins = gated_windows(match_id)
        if wins is None:
            print(f"[{match_id}] no audio events for gated mode — falling back to sliding")
            wins = sliding_windows(duration)
    else:
        wins = sliding_windows(duration)
    print(f"[{match_id}] {duration:.1f}s — {len(wins)} {mode} windows")

    n_frames = int(np.ceil(duration)) + 1
    exc = np.zeros(n_frames)
    wr = np.zeros(n_frames)
    conf_arr = np.zeros(n_frames)
    events = []

    for i, (w0, w1) in enumerate(wins):
        a, b = int(w0 * sr), int(w1 * sr)
        chunk = y[a:b]
        if len(chunk) < sr or np.max(np.abs(chunk)) < SILENCE_THRESH:
            continue

        text, conf = transcriber.transcribe(chunk)
        if not text:
            continue

        e = excitement_score(text)
        w_rate = len(text.split()) / max(1.0, (w1 - w0))

        lo, hi = int(w0), min(n_frames, int(w1))
        exc[lo:hi] = np.maximum(exc[lo:hi], e)
        wr[lo:hi] = np.maximum(wr[lo:hi], w_rate)
        conf_arr[lo:hi] = np.maximum(conf_arr[lo:hi], conf)

        for etype, kw in match_events(text.lower(), compiled).items():
            events.append({
                "time_sec":   round(w0, 3),
                "end_sec":    round(w1, 3),
                "type":       etype,
                "keyword":    kw,
                "confidence": conf,
                "excitement": round(e, 4),
                "word_rate":  round(w_rate, 3),
            })

        if (i + 1) % 20 == 0:
            print(f"  [{match_id}] window {i + 1}/{len(wins)} — {len(events)} hits")

    _write(match_id, events, exc, wr, conf_arr, n_frames)
    return events


def _write(match_id, events, exc, wr, conf_arr, n_frames):
    os.makedirs(EVENTS_DIR, exist_ok=True)
    os.makedirs(SCORE_DIR, exist_ok=True)

    pd.DataFrame(events, columns=[
        "time_sec", "end_sec", "type", "keyword", "confidence", "excitement", "word_rate"
    ]).to_csv(os.path.join(EVENTS_DIR, match_id + ".csv"), index=False)

    pd.DataFrame({
        "time_sec":  np.arange(n_frames, dtype=float),
        "excitement": exc,
        "word_rate":  wr,
        "asr_conf":   conf_arr,
    }).to_csv(os.path.join(SCORE_DIR, match_id + ".csv"), index=False)

    print(f"[{match_id}] {len(events)} speech events → {EVENTS_DIR}/{match_id}.csv")


def main():
    parser = argparse.ArgumentParser(description="F1 — Granite speech layer")
    parser.add_argument("--audio", required=True, help="path to a match audio file")
    parser.add_argument("--mode", choices=["sliding", "gated"], default="sliding")
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()

    device, dtype = pick_device()
    if args.fp32:
        dtype = torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    transcriber = GraniteTranscriber(MODEL_ID, device, dtype)
    compiled = compile_lexicon(LEXICON)
    run_speech(args.audio, transcriber, compiled, mode=args.mode)


if __name__ == "__main__":
    main()
