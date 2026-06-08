"""
Shared fusion utilities used by F2/F3 and all fusion models (A/B/E).

Defines the on-disk data contracts, candidate building (union of audio + speech
proposers), temporal non-max suppression, and asymmetric tolerance matching of
predictions against ground truth.

Coordinate convention: everything is in seconds. Highlight predictions carry
`start`, `end`, `event_type`, `score`. Ground-truth events carry `start`, `end`,
`type` (see data/annotations/<match>.json).
"""

import json
import os

import pandas as pd

# ── Data-contract directories ─────────────────────────────────────────────────
AUDIO_EVENTS_DIR   = "results/audio/events"      # time_sec,peak_rms_n,rise,dwell_sec
AUDIO_RMS_DIR      = "results/audio/rms"          # time_sec,rms,rms_n,rise,elevated
SPEECH_EVENTS_DIR  = "results/speech/events"      # time_sec,end_sec,type,keyword,confidence,excitement,word_rate
SPEECH_SCORE_DIR   = "results/speech/score"       # time_sec,excitement,word_rate,asr_conf  (1s grid)
VISION_EVENTS_DIR  = "results/vision/events"       # time_sec,type,confidence
VISION_WINDOWS_DIR = "results/vision/windows"      # start_sec,end_sec
FEATURES_DIR       = "results/fusion/features"      # F2 grid
HIGHLIGHTS_DIR     = "results/fusion/highlights"    # start,end,event_type,score
ANN_DIR            = "data/annotations"             # ground-truth JSON

EVENT_TYPES = ["goal", "penalty", "card", "save"]
GRID_DT = 1.0  # F2 timeline resolution (seconds)

# Highlight clip padding around a candidate's anchor time.
CLIP_PRE  = 8.0
CLIP_POST = 12.0

# Matching tolerance: a predicted clip [start,end] matches a truth event when the
# event time falls inside the clip, expanded by this slack on each side (commentary
# /roar lag can push the clip a little late or early relative to the on-pitch moment).
MATCH_PRE  = 12.0
MATCH_POST = 15.0

# Default temporal-NMS gap between two kept highlights.
NMS_GAP = 20.0


# ── Loaders ───────────────────────────────────────────────────────────────────
def _read_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None


def load_audio_events(match_id):
    df = _read_csv(os.path.join(AUDIO_EVENTS_DIR, match_id + ".csv"))
    if df is None or df.empty:
        return []
    return [{"time": float(r.time_sec), "rise": float(r.rise),
             "peak": float(r.peak_rms_n), "dwell": float(r.dwell_sec)}
            for r in df.itertuples()]


def load_speech_events(match_id):
    df = _read_csv(os.path.join(SPEECH_EVENTS_DIR, match_id + ".csv"))
    if df is None or df.empty:
        return []
    out = []
    for r in df.itertuples():
        out.append({
            "time":       float(r.time_sec),
            "end":        float(getattr(r, "end_sec", r.time_sec)),
            "type":       str(r.type),
            "keyword":    str(getattr(r, "keyword", "")),
            "confidence": float(getattr(r, "confidence", 0.0)),
            "excitement": float(getattr(r, "excitement", 0.0)),
            "word_rate":  float(getattr(r, "word_rate", 0.0)),
        })
    return out


def load_truth(match_id, ann_dir=ANN_DIR):
    path = os.path.join(ann_dir, match_id + ".json")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    return [{"start": float(e["start"]),
             "end":   float(e.get("end", e["start"])),
             "type":  str(e.get("type", "event"))}
            for e in data.get("events", [])]


def load_features(match_id):
    """F2 timeline as a DataFrame indexed by integer second (or None)."""
    df = _read_csv(os.path.join(FEATURES_DIR, match_id + ".csv"))
    if df is None:
        return None
    df = df.copy()
    df["sec"] = df["time_sec"].round().astype(int)
    return df.set_index("sec")


def feature_row(features, t):
    """Nearest-second row from the F2 frame (dict), or empty dict."""
    if features is None or len(features) == 0:
        return {}
    sec = int(round(t))
    if sec in features.index:
        return features.loc[sec].to_dict()
    nearest = features.index[(features.index - sec).to_series().abs().values.argmin()]
    return features.loc[nearest].to_dict()


# ── Candidate building ────────────────────────────────────────────────────────
def build_candidates(match_id, merge_gap=15.0):
    """
    Union of two proposers (audio energy peaks ∪ Granite speech keyword hits),
    merged when within `merge_gap` seconds. Each candidate carries the strongest
    audio evidence and the most-confident speech type seen in its cluster.
    """
    items = []
    for e in load_audio_events(match_id):
        items.append({"time": e["time"], "source": "audio",
                      "audio_rise": e["rise"], "audio_peak": e["peak"],
                      "speech_type": None, "speech_conf": 0.0})
    for e in load_speech_events(match_id):
        items.append({"time": e["time"], "source": "speech",
                      "audio_rise": 0.0, "audio_peak": 0.0,
                      "speech_type": e["type"], "speech_conf": e["confidence"]})

    items.sort(key=lambda x: x["time"])
    merged = []
    for it in items:
        if merged and it["time"] - merged[-1]["time"] <= merge_gap:
            m = merged[-1]
            m["sources"].add(it["source"])
            m["audio_rise"] = max(m["audio_rise"], it["audio_rise"])
            m["audio_peak"] = max(m["audio_peak"], it["audio_peak"])
            if it["speech_type"] and it["speech_conf"] >= m["speech_conf"]:
                m["speech_type"] = it["speech_type"]
                m["speech_conf"] = it["speech_conf"]
        else:
            merged.append({"time": it["time"], "sources": {it["source"]},
                           "audio_rise": it["audio_rise"], "audio_peak": it["audio_peak"],
                           "speech_type": it["speech_type"], "speech_conf": it["speech_conf"]})
    return merged


def clip_window(t):
    """Asymmetric clip [start, end] around an anchor time."""
    return max(0.0, t - CLIP_PRE), t + CLIP_POST


# ── Post-processing + scoring ─────────────────────────────────────────────────
def temporal_nms(events, min_gap=NMS_GAP, score_key="score", time_key="start"):
    """Greedy NMS in time: keep highest-scoring events, drop neighbours within min_gap."""
    if not events:
        return []
    kept = []
    for e in sorted(events, key=lambda x: x[score_key], reverse=True):
        if all(abs(e[time_key] - k[time_key]) >= min_gap for k in kept):
            kept.append(e)
    kept.sort(key=lambda e: e[time_key])
    return kept


def match_predictions(preds, truth, type_aware=True, pre=MATCH_PRE, post=MATCH_POST):
    """
    Greedy one-to-one matching. A prediction matches a truth event when the event
    time falls inside the predicted clip [start, end] expanded by (pre, post).
    Returns (tp, fp, fn). `preds` use 'start'/'end'/'event_type'; `truth` use 'start'/'type'.
    """
    used = [False] * len(truth)
    tp = 0
    for p in sorted(preds, key=lambda x: x["start"]):
        lo = p["start"] - pre
        hi = p.get("end", p["start"]) + post
        for i, g in enumerate(truth):
            if used[i]:
                continue
            if type_aware and p.get("event_type") != g["type"]:
                continue
            if lo <= g["start"] <= hi:
                used[i] = True
                tp += 1
                break
    fp = len(preds) - tp
    fn = used.count(False)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def write_highlights(match_id, highlights, out_dir=HIGHLIGHTS_DIR, suffix=""):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, match_id + suffix + ".csv")
    pd.DataFrame(highlights, columns=["start", "end", "event_type", "score"]).to_csv(path, index=False)
    print(f"[{match_id}] wrote {len(highlights)} highlights → {path}")
    return path
