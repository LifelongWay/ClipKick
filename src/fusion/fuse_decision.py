"""
Model A — Decision-level / weighted cascade (no training).

For each candidate (audio ∪ speech union), combine normalized audio energy, speech
evidence, and vision confirmation into one weighted score; threshold + temporal NMS;
type the event from the Granite keyword. Weights are hand-set and can be grid-searched
against F3.

    python src/fusion/fuse_decision.py --match <id>
    python src/fusion/fuse_decision.py --tune          # grid-search weights+threshold
"""

import argparse
import itertools

try:
    from . import common, evaluate, timeline
except ImportError:
    import common, evaluate, timeline

DEFAULT_W = (0.2, 0.5, 0.3)   # (audio, speech, vision)
DEFAULT_T = 0.45


def score_candidate(c, features, w):
    wa, ws, wv = w
    audio = min(1.0, max(0.0, (c["audio_rise"] - 1.0) / 3.0))

    row = common.feature_row(features, c["time"])
    excitement = float(row.get("excitement", 0.0) or 0.0)
    speech = min(1.0, c["speech_conf"] if c["speech_type"] else excitement)

    vision_ok = float(row.get("vision_ok", 0) or 0) >= 0.5
    if vision_ok:
        vision = float(row.get("vision_goal", 0) or 0) * float(row.get("ball_in_goal", 0) or 0)
        s = (wa * audio + ws * speech + wv * vision) / (wa + ws + wv)
    else:
        denom = wa + ws
        s = (wa * audio + ws * speech) / denom if denom else 0.0

    # Type from the speech keyword if present; otherwise an audio-energy peak with no
    # keyword is, in football, almost always a goal (crowd roar) → default "goal"
    # (a valid type, so it can be scored and never produces an untypeable event).
    etype = c["speech_type"] or "goal"
    return s, etype


def run_match(match_id, w=DEFAULT_W, threshold=DEFAULT_T, write=True):
    features = common.load_features(match_id)
    highlights = []
    for c in common.build_candidates(match_id):
        s, etype = score_candidate(c, features, w)
        if s >= threshold:
            start, end = common.clip_window(c["time"])
            highlights.append({"start": round(start, 2), "end": round(end, 2),
                               "event_type": etype, "score": round(s, 4)})
    # Same-type suppression so a real penalty + goal seconds apart both survive
    # (cross-type NMS would wrongly drop one).
    highlights = common.temporal_nms(highlights, type_aware=True)
    if write:
        common.write_highlights(match_id, highlights)
    return highlights


def tune(match_ids):
    """Small grid search; maximize summed overall-F1 across matches."""
    grid_w = [0.1, 0.2, 0.3, 0.5]
    grid_t = [0.3, 0.4, 0.5, 0.6]
    best = (-1.0, DEFAULT_W, DEFAULT_T)
    for wa, ws, wv in itertools.product(grid_w, repeat=3):
        if wa + ws + wv == 0:
            continue
        for t in grid_t:
            total = 0.0
            for mid in match_ids:
                hl = run_match(mid, w=(wa, ws, wv), threshold=t, write=False)
                rep = evaluate.evaluate(hl, mid, verbose=False)
                total += rep["overall"]["f1"]
            if total > best[0]:
                best = (total, (wa, ws, wv), t)
    print(f"Best: weights={best[1]} threshold={best[2]} (summed overall-F1={best[0]:.3f})")
    return best[1], best[2]


def main():
    parser = argparse.ArgumentParser(description="Model A — weighted decision fusion")
    parser.add_argument("--match", default=None)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--eval", action="store_true", help="print F3 metrics after")
    args = parser.parse_args()

    ids = [args.match] if args.match else timeline._all_match_ids()

    if args.tune:
        w, t = tune(ids)
    else:
        w, t = DEFAULT_W, DEFAULT_T

    for mid in ids:
        hl = run_match(mid, w=w, threshold=t)
        if args.eval or args.tune:
            evaluate.evaluate(hl, mid)


if __name__ == "__main__":
    main()
