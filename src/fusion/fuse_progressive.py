"""
Model E — Progressive two-pass recall-recovery (anytime cascade).

This GENUINELY stages the Granite compute (not a post-hoc split):

  Pass 1 (fast):  transcribe ONLY the windows around audio energy peaks → emit a
                  preliminary highlight reel immediately (results/.../<m>_preliminary.csv).
  Pass 2 (then):  transcribe the REST of the match (windows not already covered by
                  Pass 1) → recover keyword events Pass 1 missed → final reel.

So you see obvious goals quickly, then the slower full sweep fills in the quiet
events (cards/penalties/saves that produced no crowd roar). No vision, no F2.

Requires a GPU (drives Granite directly) — run on Colab / Apple Silicon:
    python src/fusion/fuse_progressive.py --match <id> --audio data/raw/audio/<id>.mp3 --eval
"""

import argparse
import os
import sys

try:
    from . import common, evaluate, timeline
except ImportError:
    import common, evaluate, timeline


def _add_speech_path():
    """Make src/speech importable regardless of the current working directory."""
    speech_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "speech")
    speech_dir = os.path.abspath(speech_dir)
    if speech_dir not in sys.path:
        sys.path.insert(0, speech_dir)


def _merge_windows(wins):
    """Merge overlapping/touching (w0, w1) windows so Pass 1 doesn't re-transcribe."""
    merged = []
    for w in sorted(wins):
        if merged and w[0] <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], w[1]))
        else:
            merged.append((w[0], w[1]))
    return merged


def _complement_windows(occupied, duration, window_sec, hop_sec):
    """Sliding windows over [0, duration] that do NOT overlap any Pass-1 window."""
    out = []
    s = 0
    while s < duration:
        w = (float(s), float(s + window_sec))
        if not any(w[0] < o[1] and o[0] < w[1] for o in occupied):
            out.append(w)
        s += hop_sec
    return out


PAD = 2.0  # small symmetric pad around the transcription window (seconds)


def _mk(ev, near):
    """Speech-event dict → highlight dict. Pass-1 events get a small agreement bonus.

    The clip IS the transcription window [time, end] (±PAD). Granite heard the
    keyword from exactly that audio, so the event is guaranteed to fall inside
    [start, end] — unlike clip_window(), which re-anchored on the window start and
    could end before the event. NMS is anchored on the window center for both passes.
    """
    start = max(0.0, ev["time"] - PAD)
    end = ev["end"] + PAD
    center = (ev["time"] + ev["end"]) / 2.0
    return {"start": round(start, 2), "end": round(end, 2),
            "event_type": ev["type"],
            "score": round(min(1.0, ev["confidence"] + (0.1 if near else 0.0)), 4),
            "_t": center}


def run_match(match_id, audio_path, write=True):
    # Heavy imports are lazy so the pure helpers above stay importable without a GPU.
    import librosa
    _add_speech_path()
    from build_ground_truth import (GraniteTranscriber, compile_lexicon, pick_device,
                                     LEXICON, MODEL_ID, TARGET_SR, WINDOW_SEC, HOP_SEC)
    from speech_layer import windows_to_events, GATE_PRE, GATE_POST

    audio_times = [a["time"] for a in common.load_audio_events(match_id)]
    print(f"[{match_id}] loading audio… ({len(audio_times)} audio peaks)")
    y, sr = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    duration = len(y) / sr

    device, dtype = pick_device()
    print(f"[{match_id}] Granite on {device} ({dtype})")
    transcriber = GraniteTranscriber(MODEL_ID, device, dtype)
    compiled = compile_lexicon(LEXICON)

    pass1_windows = _merge_windows(
        [(max(0.0, t - GATE_PRE), t + GATE_POST) for t in audio_times])
    pass2_windows = _complement_windows(pass1_windows, duration, WINDOW_SEC, HOP_SEC)

    # ── Pass 1: fast — only the audio-peak windows ──
    print(f"[{match_id}] Pass 1: transcribing {len(pass1_windows)} peak windows…")
    prelim_events = windows_to_events(y, sr, pass1_windows, transcriber, compiled)
    prelim = common.temporal_nms([_mk(e, near=True) for e in prelim_events], time_key="_t")
    if write:
        common.write_highlights(match_id, prelim, suffix="_preliminary")
    anchors = [h["_t"] for h in prelim]
    print(f"[{match_id}] Pass 1 done → {len(prelim)} preliminary highlights")

    # ── Pass 2: recover — the rest of the match ──
    print(f"[{match_id}] Pass 2: transcribing {len(pass2_windows)} remaining windows…")
    recov_events = windows_to_events(y, sr, pass2_windows, transcriber, compiled)
    recovered = []
    for it in sorted([_mk(e, near=False) for e in recov_events],
                     key=lambda x: x["score"], reverse=True):
        if all(abs(it["_t"] - a) >= common.NMS_GAP for a in anchors):
            recovered.append(it)
            anchors.append(it["_t"])
    print(f"[{match_id}] Pass 2 done → recovered {len(recovered)} missed events")

    final = sorted(prelim + recovered, key=lambda x: x["start"])
    if write:
        common.write_highlights(match_id, final)
    return final


def _default_audio(match_id):
    p = os.path.join("data/raw/audio", match_id + ".mp3")
    return p if os.path.exists(p) else None


def main():
    parser = argparse.ArgumentParser(description="Model E — progressive two-pass fusion")
    parser.add_argument("--match", default=None)
    parser.add_argument("--audio", default=None, help="match audio (defaults to data/raw/audio/<match>.mp3)")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    ids = [args.match] if args.match else timeline._all_match_ids()
    for mid in ids:
        audio = args.audio or _default_audio(mid)
        if not audio:
            print(f"[{mid}] no audio file — pass --audio")
            continue
        hl = run_match(mid, audio)
        if args.eval:
            evaluate.evaluate(hl, mid)


if __name__ == "__main__":
    main()
