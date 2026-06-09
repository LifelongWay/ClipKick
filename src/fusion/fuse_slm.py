"""
Model F — SLM-based highlight detection (Granite transcript → SmolLM2 reasoning).

Unlike Model E (keyword matching), F has a small language model READ the full
commentary transcript and reason about which moments are highlights + their type.
So it can catch events described without a trigger word ("rounds the keeper and
slots it home" → goal).

Two stages, with a transcript cache so swapping SLMs needs no GPU re-transcription:
  Stage 1 (Granite, expensive, once): whole-match transcript → results/speech/transcript/<m>.csv
  Stage 2 (SLM, repeatable):          chunked reasoning over the transcript → highlights

Run on Colab / Apple Silicon (needs a GPU):
    python src/fusion/fuse_slm.py --match <id> --audio data/raw/audio/<id>.mp3 --eval
    python src/fusion/fuse_slm.py --match <id> --video match.mp4 --clips --eval
    python src/fusion/fuse_slm.py --match <id> --model Qwen/Qwen2.5-1.5B-Instruct   # swap SLM
"""

import argparse
import json
import os
import re
import sys

try:
    from . import common, evaluate, timeline
except ImportError:
    import common, evaluate, timeline

DEFAULT_SLM = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
TRANSCRIPT_DIR = "results/speech/transcript"

CHUNK_SEGMENTS = 30      # transcript segments per SLM call (fits ~8k context with room)
CHUNK_OVERLAP  = 5       # overlap so a moment on a boundary isn't lost
PAD_PRE  = 6.0          # commentary lags play → generous pre-roll
PAD_POST = 2.0
BASE_SCORE = 0.75
AGREE_BONUS = 0.1
MAX_NEW_TOKENS = 256


def _add_speech_path():
    speech_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech"))
    if speech_dir not in sys.path:
        sys.path.insert(0, speech_dir)


def model_tag(model_id):
    """Short filesystem-safe tag from a HF model id, e.g. 'smollm2-1.7b-instruct'."""
    return model_id.rstrip("/").split("/")[-1].lower()


def pick_slm_device():
    """SLMs prefer bf16 (Phi-3.5 / Gemma-2 / Qwen are bf16-native; fp16 can NaN on Gemma)."""
    import torch
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.float16
    return "cpu", torch.float32


# ── Stage 1: transcript (cached) ──────────────────────────────────────────────
def build_or_load_transcript(match_id, audio_path, use_cache=True):
    """Return list of (start_sec, end_sec, text). Cache to disk; reuse if present."""
    import pandas as pd
    cache = os.path.join(TRANSCRIPT_DIR, match_id + ".csv")
    if use_cache and os.path.exists(cache):
        df = pd.read_csv(cache).fillna("")
        print(f"[{match_id}] using cached transcript ({len(df)} segments) → {cache}")
        return [(float(r.start_sec), float(r.end_sec), str(r.text)) for r in df.itertuples()]

    # No cache → run Granite over the whole match.
    import librosa
    _add_speech_path()
    from build_ground_truth import GraniteTranscriber, pick_device, MODEL_ID, TARGET_SR
    from speech_layer import sliding_windows, iter_window_transcripts

    print(f"[{match_id}] no transcript cache — transcribing with Granite…")
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    device, dtype = pick_device()
    transcriber = GraniteTranscriber(MODEL_ID, device, dtype)
    windows = sliding_windows(len(y) / sr)

    segments = []
    for i, (w0, w1, text, conf) in enumerate(iter_window_transcripts(y, sr, windows, transcriber)):
        segments.append((round(w0, 2), round(w1, 2), text))
        if (i + 1) % 25 == 0:
            print(f"  [{match_id}] {i + 1} segments transcribed…")

    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
    pd.DataFrame(segments, columns=["start_sec", "end_sec", "text"]).to_csv(cache, index=False)
    print(f"[{match_id}] transcript cached ({len(segments)} segments) → {cache}")
    return segments


# ── Stage 2: SLM ──────────────────────────────────────────────────────────────
class SLMExtractor:
    """Wraps an instruction-tuned causal LM that extracts highlights from commentary."""

    SYSTEM = ("You are a football match highlight detector. You read commentary lines, each "
              "prefixed with its time in seconds like [123]. Identify only genuine key moments: "
              "goal, penalty, card, save. Reply with ONLY a JSON array of objects "
              '{"time": <seconds int>, "type": "goal|penalty|card|save"} and nothing else. '
              "Use the [seconds] tag of the line where the moment happens. If none, reply [].")

    FEWSHOT_USER = ("[40] midfield battle continues\n"
                    "[60] he shoots and it's a brilliant save by the keeper\n"
                    "[95] long ball forward, cleared away\n"
                    "[120] GOAL! he smashes it into the top corner")
    FEWSHOT_ASSISTANT = '[{"time": 60, "type": "save"}, {"time": 120, "type": "goal"}]'

    def __init__(self, model_id, device=None, dtype=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if device is None:
            device, dtype = pick_slm_device()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        # Prefer the NATIVE transformers implementation (Phi-3.5's own remote code is
        # outdated and crashes on newer transformers: DynamicCache.seen_tokens). Only
        # fall back to remote code if a model genuinely isn't natively supported.
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(device)
        except (ValueError, KeyError):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=dtype, trust_remote_code=True).to(device)
        self.device = device

    def _render(self, chunk_text):
        """Apply the chat template; fall back to merging system into the first user
        message for templates that reject a 'system' role (e.g. Gemma-2)."""
        msgs_sys = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": self.FEWSHOT_USER},
            {"role": "assistant", "content": self.FEWSHOT_ASSISTANT},
            {"role": "user", "content": chunk_text},
        ]
        try:
            return self.tok.apply_chat_template(msgs_sys, tokenize=False, add_generation_prompt=True)
        except Exception:
            msgs_no_sys = [
                {"role": "user", "content": self.SYSTEM + "\n\n" + self.FEWSHOT_USER},
                {"role": "assistant", "content": self.FEWSHOT_ASSISTANT},
                {"role": "user", "content": chunk_text},
            ]
            return self.tok.apply_chat_template(msgs_no_sys, tokenize=False, add_generation_prompt=True)

    def extract(self, chunk_text):
        import torch
        prompt = self._render(chunk_text)
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        gen = out[0][inputs["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)

    def close(self):
        """Free GPU memory so the benchmark can load the next model."""
        import gc
        import torch
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Defensive parsing of SLM output ───────────────────────────────────────────
_TYPES = set(common.EVENT_TYPES)
_FALLBACK = re.compile(r"(\d+)\D{0,20}?(goal|penalty|card|save)", re.IGNORECASE)


def parse_slm_output(text):
    """Return list of {time:int, type:str}. Try JSON first, then regex fallback."""
    hits = []
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if m:
        try:
            for obj in json.loads(m.group(0)):
                t, ty = obj.get("time"), str(obj.get("type", "")).lower()
                if ty in _TYPES and t is not None:
                    hits.append({"time": int(float(t)), "type": ty})
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            pass
    if not hits:  # regex fallback on malformed output
        for t, ty in _FALLBACK.findall(text):
            ty = ty.lower()
            if ty in _TYPES:
                hits.append({"time": int(t), "type": ty})
    return hits


# ── Timing: snap to a real segment, drop hallucinated times ───────────────────
def snap_to_segment(t, segments, window_sec=20.0):
    """Nearest segment (start, end, text) to time t, or None if farther than a window
    (the SLM hallucinated a time with no matching transcribed audio)."""
    best, best_d = None, 1e9
    for seg in segments:
        center = (seg[0] + seg[1]) / 2.0
        d = abs(t - center)
        if d < best_d:
            best, best_d = seg, d
    if best is None or best_d > window_sec:
        return None
    return best


# ── Orchestration ─────────────────────────────────────────────────────────────
def _chunks(segments, size=CHUNK_SEGMENTS, overlap=CHUNK_OVERLAP):
    step = max(1, size - overlap)
    for i in range(0, len(segments), step):
        yield segments[i:i + size]


def run_match(match_id, audio_path, model_id=DEFAULT_SLM, slm=None,
              tag=None, use_cache=True, write=True):
    """Extract highlights for one match. If `slm` (a preloaded SLMExtractor) is given
    it is reused (benchmark loads each model once); otherwise it is built here.
    `tag` suffixes the output files so per-model results are not overwritten."""
    segments = build_or_load_transcript(match_id, audio_path, use_cache=use_cache)
    if not segments:
        print(f"[{match_id}] empty transcript — nothing to do")
        return []

    if slm is None:
        device, dtype = pick_slm_device()
        print(f"[{match_id}] SLM {model_id} on {device}")
        slm = SLMExtractor(model_id, device, dtype)
    suffix = ("__" + tag) if tag else ""

    # tally per (rounded) anchor time so overlapping chunks can agree → bonus
    agree, typed = {}, {}
    for chunk in _chunks(segments):
        chunk_text = "\n".join(f"[{int(s0)}] {txt}" for s0, s1, txt in chunk if txt.strip())
        if not chunk_text:
            continue
        for hit in parse_slm_output(slm.extract(chunk_text)):
            seg = snap_to_segment(hit["time"], segments)
            if seg is None:
                continue  # hallucinated time
            key = round((seg[0] + seg[1]) / 2.0)
            agree[key] = agree.get(key, 0) + 1
            typed[key] = (seg, hit["type"])

    highlights, detail = [], []
    for key, (seg, etype) in typed.items():
        s0, s1, text = seg
        score = min(1.0, BASE_SCORE + (AGREE_BONUS if agree[key] > 1 else 0.0))
        h = {"start": round(max(0.0, s0 - PAD_PRE), 2), "end": round(s1 + PAD_POST, 2),
             "event_type": etype, "score": round(score, 4)}
        highlights.append(h)
        detail.append({**h, "text": text})

    highlights = common.temporal_nms(highlights)
    print(f"[{match_id}] SLM produced {len(highlights)} highlights")
    if write:
        common.write_highlights(match_id, highlights, suffix=suffix)
        _write_detail(match_id, detail, suffix)
    return highlights


def _write_detail(match_id, detail, suffix):
    """Per-highlight commentary record: which transcript line triggered each pick."""
    import pandas as pd
    os.makedirs(common.HIGHLIGHTS_DIR, exist_ok=True)
    path = os.path.join(common.HIGHLIGHTS_DIR, match_id + suffix + "_detail.csv")
    detail = sorted(detail, key=lambda d: d["start"])
    pd.DataFrame(detail, columns=["start", "end", "event_type", "score", "text"]).to_csv(path, index=False)
    print(f"[{match_id}] detail (with commentary) → {path}")


def _extract_audio_from_video(video_path):
    try:
        from moviepy import VideoFileClip          # moviepy 2.x
    except ImportError:
        from moviepy.editor import VideoFileClip    # moviepy 1.x
    mp3 = os.path.splitext(video_path)[0] + ".mp3"
    if not os.path.exists(mp3):
        print(f"extracting audio → {mp3}")
        VideoFileClip(video_path).audio.write_audiofile(mp3, logger=None)
    return mp3


def main():
    parser = argparse.ArgumentParser(description="Model F — SLM highlight detection")
    parser.add_argument("--match", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--video", default=None, help="extract mp3 from this if --audio missing")
    parser.add_argument("--model", default=DEFAULT_SLM)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--clips", action="store_true", help="export video clips (needs --video)")
    args = parser.parse_args()

    ids = [args.match] if args.match else timeline._all_match_ids()
    for mid in ids:
        audio = args.audio
        if not audio and args.video:
            audio = _extract_audio_from_video(args.video)
        if not audio:
            audio = os.path.join("data/raw/audio", mid + ".mp3")
        if not os.path.exists(audio):
            print(f"[{mid}] no audio (looked for {audio}) — pass --audio or --video")
            continue

        hl = run_match(mid, audio, model_id=args.model, use_cache=not args.no_cache)
        if args.eval:
            evaluate.evaluate(hl, mid)
        if args.clips:
            if not args.video:
                print(f"[{mid}] --clips needs --video; skipping clip export")
            else:
                try:
                    from . import export_clips
                except ImportError:
                    import export_clips
                export_clips.export(mid, args.video)


if __name__ == "__main__":
    main()
