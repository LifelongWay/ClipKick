"""
F2 — Shared multimodal timeline.

Resamples the three layers onto one 1-second grid and writes a single feature
matrix per match (results/fusion/features/<match>.csv) with per-modality
availability masks. This is the one object every fusion model consumes.

Columns:
    time_sec
    rms_n, rise, elevated                 (audio energy; dense)
    excitement, word_rate, asr_conf       (speech continuous; medium)
    kw_goal, kw_penalty, kw_card, kw_save (speech keyword flags)
    vision_goal, ball_in_goal             (vision; sparse)
    audio_ok, speech_ok, vision_ok        (availability masks)

Robust to missing speech/vision: the corresponding columns are zero and the
mask is 0, so downstream models can tell "absent" from "saw nothing".

Run from project root:
    python src/fusion/timeline.py                 # all matches with audio RMS
    python src/fusion/timeline.py --match <id>
"""

import argparse
import os

import numpy as np
import pandas as pd

try:
    from . import common
except ImportError:  # run as a script
    import common


def _interp(grid, x, y, default=0.0):
    if x is None or len(x) == 0:
        return np.full_like(grid, default, dtype=float)
    return np.interp(grid, x, y)


def build_timeline(match_id):
    rms = common._read_csv(os.path.join(common.AUDIO_RMS_DIR, match_id + ".csv"))
    if rms is None or rms.empty:
        print(f"[{match_id}] no audio RMS — skip")
        return None

    duration = float(rms["time_sec"].max())
    grid = np.arange(0.0, duration + common.GRID_DT, common.GRID_DT)
    df = pd.DataFrame({"time_sec": grid})

    # ── Audio (dense) ──
    df["rms_n"] = _interp(grid, rms["time_sec"].values, rms["rms_n"].values)
    df["rise"] = _interp(grid, rms["time_sec"].values, rms["rise"].values, default=1.0)
    df["elevated"] = (_interp(grid, rms["time_sec"].values, rms["elevated"].values) >= 0.5).astype(int)
    df["audio_ok"] = 1

    # ── Speech continuous (per-second score file) ──
    score = common._read_csv(os.path.join(common.SPEECH_SCORE_DIR, match_id + ".csv"))
    for col in ("excitement", "word_rate", "asr_conf"):
        df[col] = 0.0
    if score is not None and not score.empty:
        s = score.set_index(score["time_sec"].round().astype(int))
        idx = df["time_sec"].round().astype(int).values
        for col in ("excitement", "word_rate", "asr_conf"):
            if col in s.columns:
                df[col] = [float(s[col].get(i, 0.0)) for i in idx]
        df["speech_ok"] = 1
    else:
        df["speech_ok"] = 0

    # ── Speech keyword flags (from events) ──
    for t in common.EVENT_TYPES:
        df["kw_" + t] = 0
    for ev in common.load_speech_events(match_id):
        lo, hi = int(round(ev["time"])), int(round(ev["end"]))
        mask = (df["time_sec"] >= lo) & (df["time_sec"] <= hi)
        df.loc[mask, "kw_" + ev["type"]] = 1

    # ── Vision (sparse) ──
    df["vision_goal"] = 0.0
    df["ball_in_goal"] = 0.0
    df["vision_ok"] = 0
    vw = common._read_csv(os.path.join(common.VISION_WINDOWS_DIR, match_id + ".csv"))
    if vw is not None and not vw.empty:
        for r in vw.itertuples():
            mask = (df["time_sec"] >= r.start_sec) & (df["time_sec"] <= r.end_sec)
            df.loc[mask, "vision_ok"] = 1
    ve = common._read_csv(os.path.join(common.VISION_EVENTS_DIR, match_id + ".csv"))
    if ve is not None and not ve.empty:
        for r in ve.itertuples():
            mask = (df["time_sec"] - float(r.time_sec)).abs() <= common.GRID_DT
            df.loc[mask, "vision_goal"] = 1.0
            df.loc[mask, "ball_in_goal"] = float(getattr(r, "confidence", 1.0))

    os.makedirs(common.FEATURES_DIR, exist_ok=True)
    out = os.path.join(common.FEATURES_DIR, match_id + ".csv")
    df.to_csv(out, index=False)
    print(f"[{match_id}] timeline {len(df)} frames → {out} "
          f"(speech_ok={int(df['speech_ok'].iloc[0])}, "
          f"vision frames={int((df['vision_ok'] == 1).sum())})")
    return df


def _all_match_ids():
    if not os.path.isdir(common.AUDIO_RMS_DIR):
        return []
    return sorted(os.path.splitext(f)[0] for f in os.listdir(common.AUDIO_RMS_DIR)
                  if f.endswith(".csv"))


def main():
    parser = argparse.ArgumentParser(description="F2 — shared multimodal timeline")
    parser.add_argument("--match", default=None)
    args = parser.parse_args()

    ids = [args.match] if args.match else _all_match_ids()
    for mid in ids:
        build_timeline(mid)


if __name__ == "__main__":
    main()
