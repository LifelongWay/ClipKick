"""
Model E — Progressive two-pass recall-recovery (anytime cascade).

Pass 1 (fast): keep Granite speech events that sit near an audio energy peak →
a quick preliminary highlight set.
Pass 2 (complete): from the whole-match Granite events, recover any keyword event
Pass 1 did not already cover → append it.

No training (rule/keyword based). Pure post-processing over the F1 speech events +
audio events, so it runs without a GPU once those exist. Emits the same highlights
CSV as A/B, graded by F3.

    python src/fusion/fuse_progressive.py --match <id>
"""

import argparse

try:
    from . import common, evaluate, timeline
except ImportError:
    import common, evaluate, timeline

GATE = 15.0  # a speech event within GATE seconds of an audio peak counts as "Pass 1"


def _mk(ev, near):
    start, end = common.clip_window(ev["time"])
    return {"start": round(start, 2), "end": round(end, 2),
            "event_type": ev["type"],
            "score": round(min(1.0, ev["confidence"] + (0.1 if near else 0.0)), 4),
            "_t": ev["time"]}


def run_match(match_id, write=True):
    audio_times = [a["time"] for a in common.load_audio_events(match_id)]
    speech = common.load_speech_events(match_id)

    def near_audio(t):
        return any(abs(t - at) <= GATE for at in audio_times)

    pass1 = [_mk(ev, True) for ev in speech if near_audio(ev["time"])]
    pass2 = [_mk(ev, False) for ev in speech if not near_audio(ev["time"])]

    kept = common.temporal_nms(pass1, time_key="_t")
    anchors = [k["_t"] for k in kept]

    recovered = []
    for it in sorted(pass2, key=lambda x: x["score"], reverse=True):
        if all(abs(it["_t"] - a) >= common.NMS_GAP for a in anchors):
            recovered.append(it)
            anchors.append(it["_t"])

    final = kept + recovered
    final.sort(key=lambda x: x["start"])
    for f in final:
        f.pop("_t", None)

    print(f"[{match_id}] pass1={len(kept)} recovered={len(recovered)} total={len(final)}")
    if write:
        common.write_highlights(match_id, final)
    return final


def main():
    parser = argparse.ArgumentParser(description="Model E — progressive two-pass fusion")
    parser.add_argument("--match", default=None)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    ids = [args.match] if args.match else timeline._all_match_ids()
    for mid in ids:
        hl = run_match(mid)
        if args.eval:
            evaluate.evaluate(hl, mid)


if __name__ == "__main__":
    main()
