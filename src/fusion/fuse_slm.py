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
DEFAULT_CONF = 0.5      # used when the model omits confidence (or regex fallback)
AGREE_BONUS = 0.05     # small boost when the same moment is flagged in >1 chunk
SLM_NMS_GAP = 60.0     # same-type suppression window (collapses repeated mentions)
SLM_MIN_CONF = 0.0     # default keeps all (preserves recall); raise to trade for precision
MAX_NEW_TOKENS = 512   # headroom so a chunk with several events isn't truncated mid-JSON


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

    SYSTEM = (
        "You are a football match highlight detector. You read commentary lines, each prefixed "
        "with its time in seconds like [123]. Tag a moment ONLY if the commentary describes the "
        "event happening LIVE at that instant: a goal being scored, a penalty awarded, a card "
        "shown (the referee actually produces a yellow or red card now — a booking, caution, second "
        "yellow, or sending-off), or a notable save or a shot that hits the woodwork (the post or "
        "crossbar). Do NOT tag retrospective mentions, replays, "
        "statistics, build-up, or talk about earlier events (e.g. 'that was his second goal'). "
        "In particular do NOT tag card DISCUSSION: a player already 'on a yellow', one who 'escapes "
        "a booking', 'hasn't been booked', a foul that 'wasn't enough for a yellow', or recalling an "
        "earlier card — only the instant a card is actually shown. "
        "A goal counts even when stated plainly ('it's one-nil', 'in the back of the net', "
        "'he makes it two'), not only when shouted as 'GOAL!'. "
        "For each live event return its [seconds] tag as time, the type "
        "(goal|penalty|card|save), and a confidence 0.0-1.0: use ~0.9+ for a clear, unmistakable "
        "live event, ~0.6-0.8 for a likely-but-uncertain one (an appeal, a half-chance, a moment "
        "under review), and simply skip anything you are not reasonably sure happened live. "
        "Reply with ONLY a JSON array of "
        '{"time": <int>, "type": "goal|penalty|card|save", "confidence": <float>} and nothing '
        "else. If nothing is happening live, reply [].")

    # Generic, match-agnostic examples (no real team/player names) so the prompt
    # generalises to ANY match and there is no leakage into the benchmark.
    FEWSHOT_USER = (
        "[40] and that was a well-taken goal earlier in the half, you'll remember\n"     # retrospective goal
        "[60] he strikes it — and what a save, the keeper gets a strong hand to it!\n"   # live save (keeper)
        "[80] he gets it onto the bar — it rattles the crossbar and stays out!\n"        # live save (woodwork)
        "[95] the ball is floated into the box and headed clear\n"                       # nothing
        "[105] he's already on a yellow card, so a second here would be costly\n"        # card DISCUSSION → ignore
        "[120] GOAL! he fires it into the roof of the net\n"                             # live goal (loud)
        "[135] the keeper goes the wrong way and it's in, one-nil to the home side\n"    # live goal (subtle, no 'GOAL!')
        "[150] he's brought down in the box and the referee points to the spot, penalty\n"  # live penalty (clear)
        "[175] he was lucky not to be booked for that one earlier\n"                     # retrospective card
        "[200] the referee shows a yellow card, he's booked for the late challenge\n"    # live card (explicit + 'booked')
        "[230] penalty appeals as he goes down in the area, the referee is checking")    # uncertain penalty
    FEWSHOT_ASSISTANT = ('[{"time": 60, "type": "save", "confidence": 0.9}, '
                         '{"time": 80, "type": "save", "confidence": 0.88}, '
                         '{"time": 120, "type": "goal", "confidence": 0.96}, '
                         '{"time": 135, "type": "goal", "confidence": 0.85}, '
                         '{"time": 150, "type": "penalty", "confidence": 0.92}, '
                         '{"time": 200, "type": "card", "confidence": 0.9}, '
                         '{"time": 230, "type": "penalty", "confidence": 0.6}]')

    def __init__(self, model_id, device=None, dtype=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        if device is None:
            device, dtype = pick_slm_device()
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if any(k in model_id.lower() for k in ("bnb", "4bit", "8bit", "gptq", "awq")):
            # Pre-quantized (e.g. bitsandbytes 4-bit): quant config is baked into the
            # weights, device_map places it, and you must NOT call .to() on it.
            # Needs bitsandbytes + accelerate installed.
            self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
            self.device = next(self.model.parameters()).device  # for input-tensor placement
        else:
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
_OBJ = re.compile(r"\{[^{}]*\}")  # one flat JSON object (our schema has no nesting)
_FALLBACK = re.compile(r"(\d+)\D{0,20}?(goal|penalty|card|save)", re.IGNORECASE)


def _clamp_conf(c):
    try:
        return max(0.0, min(1.0, float(c)))
    except (TypeError, ValueError):
        return DEFAULT_CONF


def parse_slm_output(text):
    """Return list of {time:int, type:str, confidence:float}.

    Parses each ``{...}`` object independently rather than relying on the outer
    ``[...]`` brackets, so prose around the JSON, echoed input timestamps, multiple
    arrays, or a truncated array (token limit cut the closing ``]``) don't wipe out
    the per-event confidence. Falls back to a loose regex only if nothing parses."""
    hits = []
    for block in _OBJ.findall(text):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        ty = str(obj.get("type", "")).lower()
        t = obj.get("time")
        if ty not in _TYPES or t is None:
            continue
        try:
            ti = int(float(t))
        except (TypeError, ValueError):
            continue
        conf = _clamp_conf(obj.get("confidence")) if "confidence" in obj else DEFAULT_CONF
        hits.append({"time": ti, "type": ty, "confidence": conf})
    if not hits:  # last-ditch regex fallback on non-JSON output — no confidence available
        for t, ty in _FALLBACK.findall(text):
            ty = ty.lower()
            if ty in _TYPES:
                hits.append({"time": int(t), "type": ty, "confidence": DEFAULT_CONF})
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
def _dedupe_overlap(segments):
    """Drop overlapping transcript windows so the SLM doesn't read each commentary
    phrase twice (Granite slides 20s windows at a 10s hop → ~50% repeated text).
    Greedy non-overlapping selection keeps full temporal coverage."""
    out, last_end = [], -1.0
    for seg in segments:
        if seg[0] >= last_end:
            out.append(seg)
            last_end = seg[1]
    return out


def _chunks(segments, size=CHUNK_SEGMENTS, overlap=CHUNK_OVERLAP):
    step = max(1, size - overlap)
    for i in range(0, len(segments), step):
        yield segments[i:i + size]


def run_match(match_id, audio_path, model_id=DEFAULT_SLM, slm=None,
              tag=None, use_cache=True, write=True, min_confidence=SLM_MIN_CONF):
    """Extract highlights for one match. If `slm` (a preloaded SLMExtractor) is given
    it is reused (benchmark loads each model once); otherwise it is built here.
    `tag` suffixes the output files so per-model results are not overwritten.
    `min_confidence` drops low-confidence detections (precision knob)."""
    segments = build_or_load_transcript(match_id, audio_path, use_cache=use_cache)
    if not segments:
        print(f"[{match_id}] empty transcript — nothing to do")
        return []
    # Feed the SLM non-overlapping windows (no repeated commentary). Lines are tagged
    # with the window CENTER so it matches snap_to_segment's center-based lookup.
    slm_segments = _dedupe_overlap(segments)

    if slm is None:
        device, dtype = pick_slm_device()
        print(f"[{match_id}] SLM {model_id} on {device}")
        slm = SLMExtractor(model_id, device, dtype)
    suffix = ("__" + tag) if tag else ""

    # tally per (rounded) anchor time; keep the MAX confidence seen across chunks
    agree, typed = {}, {}
    for chunk in _chunks(slm_segments):
        chunk_text = "\n".join(f"[{int((s0 + s1) / 2)}] {txt}" for s0, s1, txt in chunk if txt.strip())
        if not chunk_text:
            continue
        for hit in parse_slm_output(slm.extract(chunk_text)):
            seg = snap_to_segment(hit["time"], slm_segments)
            if seg is None:
                continue  # hallucinated time
            key = round((seg[0] + seg[1]) / 2.0)
            agree[key] = agree.get(key, 0) + 1
            prev = typed.get(key)
            if prev is None or hit["confidence"] > prev[2]:
                typed[key] = (seg, hit["type"], hit["confidence"])

    highlights, detail = [], []
    for key, (seg, etype, conf) in typed.items():
        s0, s1, text = seg
        score = min(1.0, conf + (AGREE_BONUS if agree[key] > 1 else 0.0))
        if score < min_confidence:
            continue  # precision filter
        h = {"start": round(max(0.0, s0 - PAD_PRE), 2), "end": round(s1 + PAD_POST, 2),
             "event_type": etype, "score": round(score, 4)}
        highlights.append(h)
        detail.append({**h, "text": text})

    # same-type 60s suppression collapses repeated mentions of one event
    highlights = common.temporal_nms(highlights, min_gap=SLM_NMS_GAP, type_aware=True)
    # keep detail rows aligned with the survivors
    kept_keys = {(h["start"], h["event_type"]) for h in highlights}
    detail = [d for d in detail if (d["start"], d["event_type"]) in kept_keys]
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
    parser.add_argument("--min-confidence", type=float, default=SLM_MIN_CONF,
                        help="drop detections below this score (precision knob; 0 keeps all)")
    parser.add_argument("--transcript-only", action="store_true",
                        help="build/cache the Granite transcript and exit (no SLM stage)")
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

        if args.transcript_only:
            build_or_load_transcript(mid, audio, use_cache=not args.no_cache)
            continue

        hl = run_match(mid, audio, model_id=args.model, use_cache=not args.no_cache,
                       min_confidence=args.min_confidence)
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
