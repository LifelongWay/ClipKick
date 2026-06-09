"""
Model G — SLM-verified cascade (Model A proposes → SLM verifies & types).

Model A (audio energy peaks ∪ speech keyword hits) is a high-recall / low-precision
proposer. Model G feeds each of its candidates to an SLM that reads the surrounding
commentary and answers a focused question — "is a highlight actually happening here,
and which type?" — keeping only the verified ones, re-typed and confidence-scored.

This keeps Model A's recall while stripping its false positives (e.g. card-talk) and
fixing its weak typing. Verification (yes/no per candidate) is more reliable for small
models than Model F's open-ended extraction.

    python src/fusion/fuse_hybrid.py --match <id> --audio data/raw/audio/<id>.mp3 --eval
    python src/fusion/fuse_hybrid.py --match <id> --model unsloth/Qwen2.5-32B-Instruct-bnb-4bit
"""

import argparse
import json
import re

try:
    from . import common, evaluate, fuse_slm, timeline
except ImportError:
    import common, evaluate, fuse_slm, timeline

DEFAULT_VERIFIER = "Qwen/Qwen2.5-7B-Instruct"
CTX = 20.0          # seconds of commentary context on each side of a candidate
PAD_PRE = fuse_slm.PAD_PRE
PAD_POST = fuse_slm.PAD_POST
NMS_GAP = fuse_slm.SLM_NMS_GAP

_TYPES = set(common.EVENT_TYPES)

SYSTEM = (
    "You verify whether ONE specific moment in a football match is a live highlight. "
    "You are given the commentary around that moment. Decide if, right at this moment, one of "
    "these is actually happening: a goal being scored, a penalty awarded, a card actually shown "
    "(the referee produces a yellow or red), or a notable save / a shot off the woodwork. "
    "Answer false for mere discussion, build-up, replays, statistics, retrospective mentions, or a "
    "player merely being 'on a yellow' / 'escaping a booking'. "
    'Reply with ONLY JSON {"highlight": true|false, "type": "goal|penalty|card|save|none", '
    '"confidence": <0.0-1.0>} and nothing else.')

FEWSHOT = [
    ("Commentary around the moment:\n"
     "the cross comes in and he heads it down, GOAL, buried in the bottom corner!",
     '{"highlight": true, "type": "goal", "confidence": 0.96}'),
    ("Commentary around the moment:\n"
     "he's already on a yellow card so he'll have to be careful in this second half",
     '{"highlight": false, "type": "none", "confidence": 0.0}'),
]

_OBJ = re.compile(r"\{[^{}]*\}")


def parse_verify(text):
    """Parse the verifier's JSON → (keep: bool, type or None, confidence)."""
    for block in _OBJ.findall(text):
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            continue
        hl = obj.get("highlight")
        ty = str(obj.get("type", "")).lower()
        if isinstance(hl, str):
            hl = hl.strip().lower() == "true"
        if hl and ty in _TYPES:
            try:
                conf = max(0.0, min(1.0, float(obj.get("confidence", fuse_slm.DEFAULT_CONF))))
            except (TypeError, ValueError):
                conf = fuse_slm.DEFAULT_CONF
            return True, ty, conf
        if hl is not None:   # a well-formed negative
            return False, None, 0.0
    return False, None, 0.0


def context_for(segments, t, ctx=CTX):
    """Join transcript text within [t-ctx, t+ctx] → the snippet shown to the verifier."""
    lo, hi = t - ctx, t + ctx
    parts = [str(txt) for s0, s1, txt in segments if s1 >= lo and s0 <= hi and str(txt).strip()]
    return " ".join(parts).strip()


def verify_candidate(slm, snippet):
    msgs = [{"role": "system", "content": SYSTEM}]
    for u, a in FEWSHOT:
        msgs.append({"role": "user", "content": u})
        msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": "Commentary around the moment:\n" + snippet})
    return parse_verify(slm.chat(msgs, max_new_tokens=64))


def run_match(match_id, audio_path, model_id=DEFAULT_VERIFIER, slm=None,
              tag=None, use_cache=True, write=True, min_confidence=0.0):
    candidates = common.build_candidates(match_id)
    if not candidates:
        print(f"[{match_id}] no Model-A candidates — did audio_layer + speech_layer run?")
        return []
    segments = fuse_slm.build_or_load_transcript(match_id, audio_path, use_cache=use_cache)

    if slm is None:
        slm = fuse_slm.SLMExtractor(model_id)
    suffix = ("__" + tag) if tag else ""

    highlights, detail = [], []
    for c in candidates:
        t = c["time"]
        snippet = context_for(segments, t)
        if not snippet:
            continue
        keep, etype, conf = verify_candidate(slm, snippet)
        if not keep or conf < min_confidence:
            continue
        h = {"start": round(max(0.0, t - PAD_PRE), 2), "end": round(t + PAD_POST, 2),
             "event_type": etype, "score": round(conf, 4)}
        highlights.append(h)
        detail.append({**h, "text": snippet[:300]})

    highlights = common.temporal_nms(highlights, min_gap=NMS_GAP, type_aware=True)
    kept = {(h["start"], h["event_type"]) for h in highlights}
    detail = [d for d in detail if (d["start"], d["event_type"]) in kept]
    print(f"[{match_id}] Model G verified {len(highlights)}/{len(candidates)} candidates")
    if write:
        common.write_highlights(match_id, highlights, suffix=suffix)
        fuse_slm._write_detail(match_id, detail, suffix)
    return highlights


def _default_audio(match_id):
    import os
    p = os.path.join("data/raw/audio", match_id + ".mp3")
    return p if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser(description="Model G — SLM-verified cascade")
    parser.add_argument("--match", default=None)
    parser.add_argument("--audio", default=None)
    parser.add_argument("--model", default=DEFAULT_VERIFIER)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    ids = [args.match] if args.match else timeline._all_match_ids()
    for mid in ids:
        audio = args.audio or _default_audio(mid)
        if not audio:
            print(f"[{mid}] no audio — pass --audio")
            continue
        hl = run_match(mid, audio, model_id=args.model, min_confidence=args.min_confidence)
        if args.eval:
            evaluate.evaluate(hl, mid)


if __name__ == "__main__":
    main()
