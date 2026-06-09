"""
Granite speech-to-text library (shared transcription code).

Hosts the Granite ASR wrapper, the multilingual keyword lexicon, and the device
auto-detect used across the project. Imported by:
  * src/speech/speech_layer.py        (F1 — speech signals for Models A/B)
  * src/fusion/fuse_slm.py            (Models F & G — whole-match transcript)
  * src/fusion/fuse_progressive.py    (Model E — two-pass transcription)

This module contains NO ground-truth generation — annotations in data/annotations/
are hand-curated from match reports and must not be machine-overwritten.

Design notes:
  * Chunk-level timestamps (no -plus variant): an event's time is the window it was
    heard in, so precision ≈ WINDOW_SEC. 50% overlap avoids missing boundary words.
  * Device auto-detect (cuda/mps/cpu) so the same code runs on Colab (CUDA, fp16)
    and on Apple Silicon (MPS, fp16). MPS+fp16 occasionally yields empty output; we
    fall back to fp32 on MPS once if that happens.
"""

import re
import warnings

import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

warnings.filterwarnings("ignore")

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
