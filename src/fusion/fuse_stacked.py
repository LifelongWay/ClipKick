"""
Model B — Stacked late fusion (learned combiner).

Builds a fixed feature vector per candidate (audio ∪ speech union), labels it from
the Granite ground truth (F4), and trains a gradient-boosted classifier to output a
calibrated event probability + type. HistGradientBoosting handles missing vision
natively (sparse layer), so the availability mask is just another feature.

Evaluation uses leave-one-match-out so candidates from one event never leak across
the train/test boundary. With only a few matches this is data-limited — treat the
numbers as indicative.

    python src/fusion/fuse_stacked.py                 # leave-one-match-out + F3
    python src/fusion/fuse_stacked.py --match <id>    # train on the rest, predict this one
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

try:
    from . import common, evaluate, timeline
except ImportError:
    import common, evaluate, timeline

FEATURE_NAMES = [
    "audio_rise", "audio_peak", "speech_conf", "has_speech", "n_sources",
    "excitement", "word_rate", "asr_conf",
    "kw_goal", "kw_penalty", "kw_card", "kw_save",
    "vision_goal", "ball_in_goal", "vision_ok",
]
LABEL_TOL = 25.0
DEFAULT_T = 0.40


def candidate_features(c, features):
    row = common.feature_row(features, c["time"])
    g = lambda k: float(row.get(k, 0) or 0)
    return {
        "audio_rise": c["audio_rise"], "audio_peak": c["audio_peak"],
        "speech_conf": c["speech_conf"], "has_speech": 1.0 if c["speech_type"] else 0.0,
        "n_sources": float(len(c["sources"])),
        "excitement": g("excitement"), "word_rate": g("word_rate"), "asr_conf": g("asr_conf"),
        "kw_goal": g("kw_goal"), "kw_penalty": g("kw_penalty"),
        "kw_card": g("kw_card"), "kw_save": g("kw_save"),
        "vision_goal": g("vision_goal"), "ball_in_goal": g("ball_in_goal"), "vision_ok": g("vision_ok"),
    }


def candidate_label(c, truth):
    best, best_dt = "none", 1e9
    for g in truth:
        dt = abs(c["time"] - g["start"])
        if dt <= LABEL_TOL and dt < best_dt:
            best, best_dt = g["type"], dt
    return best


def build_dataset(match_ids):
    rows, labels = [], []
    for mid in match_ids:
        features = common.load_features(mid)
        truth = common.load_truth(mid)
        for c in common.build_candidates(mid):
            rows.append(candidate_features(c, features))
            labels.append(candidate_label(c, truth))
    X = pd.DataFrame(rows, columns=FEATURE_NAMES) if rows else pd.DataFrame(columns=FEATURE_NAMES)
    return X, np.array(labels)


def train_model(X, y):
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3)
    clf.fit(X, y)
    return clf


def predict_highlights(clf, match_id, threshold):
    features = common.load_features(match_id)
    cands = common.build_candidates(match_id)
    if not cands:
        return []
    X = pd.DataFrame([candidate_features(c, features) for c in cands], columns=FEATURE_NAMES)
    proba = clf.predict_proba(X)
    classes = list(clf.classes_)
    none_idx = classes.index("none") if "none" in classes else None

    highlights = []
    for c, p in zip(cands, proba):
        best_type, best_p = None, 0.0
        for cls, pp in zip(classes, p):
            if cls == "none":
                continue
            if pp > best_p:
                best_type, best_p = cls, pp
        none_p = p[none_idx] if none_idx is not None else 0.0
        if best_type and best_p >= threshold and best_p >= none_p:
            start, end = common.clip_window(c["time"])
            highlights.append({"start": round(start, 2), "end": round(end, 2),
                               "event_type": best_type, "score": round(float(best_p), 4)})
    return common.temporal_nms(highlights)


def leave_one_match_out(match_ids, threshold):
    for held in match_ids:
        train_ids = [m for m in match_ids if m != held]
        X, y = build_dataset(train_ids)
        if len(set(y)) < 2:
            print(f"[{held}] training set has <2 classes — skip")
            continue
        clf = train_model(X, y)
        hl = predict_highlights(clf, held, threshold)
        common.write_highlights(held, hl)
        evaluate.evaluate(hl, held)


def main():
    parser = argparse.ArgumentParser(description="Model B — stacked classifier fusion")
    parser.add_argument("--match", default=None, help="train on the other matches, predict this one")
    parser.add_argument("--threshold", type=float, default=DEFAULT_T)
    args = parser.parse_args()

    ids = timeline._all_match_ids()
    truth_ids = [m for m in ids if common.load_truth(m)]

    if args.match:
        train_ids = [m for m in truth_ids if m != args.match]
        X, y = build_dataset(train_ids)
        if len(set(y)) < 2:
            print("Not enough labeled variety to train.")
            return
        clf = train_model(X, y)
        hl = predict_highlights(clf, args.match, args.threshold)
        common.write_highlights(args.match, hl)
        evaluate.evaluate(hl, args.match)
    else:
        leave_one_match_out(truth_ids, args.threshold)


if __name__ == "__main__":
    main()
