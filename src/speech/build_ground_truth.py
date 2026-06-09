"""
F4 — Granite-generated multi-event ground truth.

Slides a window over each full-match audio file, transcribes every window with
IBM Granite Speech 4.1-2b, keyword-spots multi-event highlights (goal / penalty /
card / save) against a multilingual lexicon, merges overlapping detections, and
writes them as the ground-truth annotation JSON for each match.

Design notes:
  * Chunk-level timestamps (no -plus variant): an event's time is the window it was
    heard in, so precision ≈ WINDOW_SEC. 50% overlap avoids missing boundary words.
  * Ground truth is regenerated ENTIRELY from Granite (the prior human labels are
    replaced). The originals are snapshotted to data/annotations_original/ first so
    they are never lost — they can serve as an independent reference holdout.
  * Device auto-detect (cuda/mps/cpu) so the SAME script runs unchanged on Colab
    (CUDA, fp16) and on a Mac M4 Max (MPS, fp16). MPS+fp16 occasionally yields empty
    output; we fall back to fp32 on MPS once if that happens.

Run from project root:
    python src/speech/build_ground_truth.py                 # all matches
    python src/speech/build_ground_truth.py --match 2018WC_Portugal_vs_Spain
    python src/speech/build_ground_truth.py --fp32          # force float32
"""

import argparse
import json
import os
import re
import shutil
import warnings

import librosa
import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
AUDIO_DIR      = "data/raw/audio"
ANN_DIR        = "data/annotations"
ANN_BACKUP_DIR = "data/annotations_original"

# ── Model / inference params ──────────────────────────────────────────────────
MODEL_ID       = "ibm-granite/granite-speech-4.1-2b"
TARGET_SR      = 16000
WINDOW_SEC     = 20
HOP_SEC        = 10          # 50% overlap
SILENCE_THRESH = 0.01
MAX_NEW_TOKENS = 256
PROMPT_TEXT    = "Transcribe the following speech into English text.<|audio|>"
FALLBACK_CONF  = 0.75        # used when token-probability confidence is unavailable

# ── Keyword lexicon ───────────────────────────────────────────────────────────
# event type -> surface forms (lowercased). Granite transcribes to English, so the
# English forms catch translated commentary; the Spanish/Portuguese forms are a
# safety net for untranslated tokens. Matched with word boundaries to avoid false
# fires like "goalkeeper" / "scorecard".
LEXICON = {
    "goal":    ["goal", "gol", "golazo", "goool", "goaal"],
    "penalty": ["penalty", "penalti", "penal", "pênalti", "spot kick"],
    "card":    ["yellow card", "red card", "straight red", "second yellow",
                "booked", "booking", "caution", "cautioned", "sent off", "sent-off",
                "dismissed", "into the book", "shown a card", "shows a card",
                "tarjeta", "tarjeta roja", "tarjeta amarilla", "expulsión", "expulsion"],
    "save":    ["save", "great save", "what a save", "parada", "paradón",
                "paradon", "off the post", "hits the bar", "crossbar",
                "denied", "so close"],
}


def compile_lexicon(lexicon):
    """type -> list of (surface_form, compiled word-boundary regex)."""
    compiled = {}
    for etype, words in lexicon.items():
        compiled[etype] = [
            (w, re.compile(r"\b" + re.escape(w) + r"\b", re.IGNORECASE))
            for w in words
        ]
    return compiled


def match_events(text, compiled):
    """Return {event_type: first_matching_keyword} for all types present in text."""
    found = {}
    for etype, patterns in compiled.items():
        for word, pat in patterns:
            if pat.search(text):
                found[etype] = word
                break
    return found


# ── Device ────────────────────────────────────────────────────────────────────
def pick_device():
    if torch.cuda.is_available():
        return "cuda", torch.float16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ── Granite wrapper ───────────────────────────────────────────────────────────
class GraniteTranscriber:
    """Loads Granite once and transcribes audio chunks, with an MPS fp32 fallback."""

    def __init__(self, model_id, device, dtype):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = self._load(dtype)
        self.prompt = self.processor.tokenizer.apply_chat_template(
            [{"role": "user", "content": PROMPT_TEXT}],
            tokenize=False, add_generation_prompt=True,
        )
        self._fp32_fallback_done = False

    def _load(self, dtype):
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)

    def transcribe(self, chunk):
        text, conf = self._run(chunk)
        # MPS + fp16 can occasionally produce empty / NaN generations; retry in fp32.
        if (not text and self.device == "mps" and self.dtype == torch.float16
                and not self._fp32_fallback_done):
            print("  [warn] empty output on MPS+fp16 — reloading model in fp32")
            self.dtype = torch.float32
            self.model = self._load(torch.float32)
            self._fp32_fallback_done = True
            text, conf = self._run(chunk)
        return text, conf

    def _run(self, chunk):
        inputs = self.processor(audio=chunk, text=self.prompt, return_tensors="pt")
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)
            if inputs[k].is_floating_point():
                inputs[k] = inputs[k].to(self.dtype)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                repetition_penalty=1.1,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        text = text.replace("Transcribe the following speech into English text.", "").strip()
        text = re.sub(r"\[.*?\]", "", text).strip()
        return text, self._confidence(out)

    def _confidence(self, out):
        """Mean per-token probability of the generated sequence (ASR certainty proxy)."""
        try:
            trans = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True
            )
            probs = trans[0].exp()
            probs = probs[torch.isfinite(probs)]
            if probs.numel() == 0:
                return FALLBACK_CONF
            c = float(probs.mean().item())
            return round(c, 4) if np.isfinite(c) else FALLBACK_CONF
        except Exception:
            return FALLBACK_CONF


# ── Event post-processing ─────────────────────────────────────────────────────
def merge_events(raw):
    """Merge same-type events whose windows overlap/touch; keep the max-confidence keyword."""
    by_type = {}
    for e in raw:
        by_type.setdefault(e["type"], []).append(e)

    merged = []
    for evs in by_type.values():
        evs.sort(key=lambda e: e["start"])
        cur = None
        for e in evs:
            if cur is None:
                cur = dict(e)
            elif e["start"] <= cur["end"]:          # overlap or adjacency
                cur["end"] = max(cur["end"], e["end"])
                if e["confidence"] > cur["confidence"]:
                    cur["confidence"] = e["confidence"]
                    cur["keyword"] = e["keyword"]
            else:
                merged.append(cur)
                cur = dict(e)
        if cur is not None:
            merged.append(cur)

    merged.sort(key=lambda e: e["start"])
    return merged


def snapshot_originals(match_id):
    """Copy the existing annotation to data/annotations_original/ once (never overwrite)."""
    src = os.path.join(ANN_DIR, match_id + ".json")
    dst = os.path.join(ANN_BACKUP_DIR, match_id + ".json")
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(ANN_BACKUP_DIR, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[{match_id}] snapshotted original human labels → {dst}")


def write_annotations(match_id, events, out_dir):
    snapshot_originals(match_id)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, match_id + ".json")
    with open(path, "w") as f:
        json.dump({"match_id": match_id, "events": events}, f, indent=2, ensure_ascii=False)
    print(f"[{match_id}] wrote {len(events)} events → {path}")


# ── Per-match driver ──────────────────────────────────────────────────────────
def process_match(path, transcriber, compiled, out_dir):
    match_id = os.path.splitext(os.path.basename(path))[0]
    print(f"[{match_id}] loading audio…")
    y, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    print(f"[{match_id}] {len(y) / sr:.1f}s @ {sr}Hz — sliding {WINDOW_SEC}s / {HOP_SEC}s")

    win = int(WINDOW_SEC * sr)
    hop = int(HOP_SEC * sr)
    raw_events = []

    start_idx = 0
    i = 0
    while start_idx < len(y):
        chunk = y[start_idx:start_idx + win]
        start_sec = start_idx / sr

        if len(chunk) >= sr and np.max(np.abs(chunk)) >= SILENCE_THRESH:
            text, conf = transcriber.transcribe(chunk)
            if text:
                for etype, keyword in match_events(text.lower(), compiled).items():
                    raw_events.append({
                        "start":      int(round(start_sec)),
                        "end":        int(round(start_sec + WINDOW_SEC)),
                        "type":       etype,
                        "keyword":    keyword,
                        "confidence": conf,
                        "source":     "granite",
                    })

        i += 1
        if i % 20 == 0:
            print(f"  [{match_id}] window {i} @ {start_sec:.0f}s — {len(raw_events)} raw hits")
        start_idx += hop

    events = merge_events(raw_events)
    print(f"[{match_id}] {len(raw_events)} raw → {len(events)} merged events")
    write_annotations(match_id, events, out_dir)
    return events


def main():
    parser = argparse.ArgumentParser(
        description="F4 — Granite-generated multi-event ground truth")
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--out-dir", default=ANN_DIR)
    parser.add_argument("--match", default=None,
                        help="only process this match id (filename without extension)")
    parser.add_argument("--fp32", action="store_true",
                        help="force float32 (use if MPS fp16 misbehaves)")
    args = parser.parse_args()

    device, dtype = pick_device()
    if args.fp32:
        dtype = torch.float32
    print(f"Device: {device} | dtype: {dtype}")

    transcriber = GraniteTranscriber(MODEL_ID, device, dtype)
    compiled = compile_lexicon(LEXICON)

    files = sorted(f for f in os.listdir(args.audio_dir) if f.endswith((".mp3", ".wav")))
    if args.match:
        files = [f for f in files if os.path.splitext(f)[0] == args.match]
        if not files:
            print(f"No audio file matching '{args.match}' in {args.audio_dir}")
            return

    for fname in files:
        process_match(os.path.join(args.audio_dir, fname), transcriber, compiled, args.out_dir)


if __name__ == "__main__":
    main()
