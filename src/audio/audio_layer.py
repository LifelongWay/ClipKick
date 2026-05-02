"""
Full-match loudness-based event detector.

Pipeline:
    load (sr=8000, mono)  →  300–4000 Hz bandpass  →  framed RMS
                          →  60s rolling-median baseline
                          →  rise = rms / baseline
                          →  elevated = rise ≥ RISE_RATIO
                          →  contiguous runs ≥ MIN_DWELL_SEC, separated by MIN_GAP_SEC

Outputs per match:
    results/audio/rms/<match_id>.csv      — per-frame curve
    results/audio/events/<match_id>.csv   — detected event timestamps

Run from project root:
    python src/audio/audio_layer.py
"""

import os
import warnings

import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
AUDIO_DIR     = "data/raw/audio"
RMS_OUT_DIR   = "results/audio/rms"
EVENTS_OUT_DIR = "results/audio/events"

# ── Feature params ────────────────────────────────────────────────────────────
SR           = 8000
FRAME_LENGTH = 8192   # 1.024 s per frame
HOP_LENGTH   = 4096   # 0.512 s hop → ~2 frames/s
DT           = HOP_LENGTH / SR

# ── Detection params (from notebooks/energy_rms/sustained_detection) ──────────
BASELINE_SEC  = 60.0
RISE_RATIO    = 1.15
MIN_DWELL_SEC = 8.0
MIN_GAP_SEC   = 45.0

# ── Bandpass ──────────────────────────────────────────────────────────────────
BAND_LOW  = 300
BAND_HIGH = 4000


def load_audio(path, sr=SR):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


def bandpass(y, sr, low=BAND_LOW, high=BAND_HIGH):
    nyq = sr / 2
    high = min(high, nyq - 50)
    sos = butter(4, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, y)


def compute_rms(y, sr=SR, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH):
    y_f = bandpass(y, sr)
    rms = librosa.feature.rms(y=y_f, frame_length=frame_length,
                              hop_length=hop_length)[0]
    return rms


def rolling_median(arr, window_frames):
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - window_frames)
        out[i] = np.median(arr[lo : i + 1])
    return out


def detect_events(rms, dt=DT,
                  rise_ratio=RISE_RATIO,
                  min_dwell_sec=MIN_DWELL_SEC,
                  min_gap_sec=MIN_GAP_SEC,
                  baseline_sec=BASELINE_SEC):
    """
    Returns per-frame arrays (rms_n, rise, elevated) and a list of event dicts:
        {time_sec, peak_rms_n, rise, dwell_sec}
    """
    times = np.arange(len(rms)) * dt
    rms_n = rms / (np.max(rms) + 1e-8)

    baseline_frames = max(1, int(baseline_sec / dt))
    baseline = rolling_median(rms, baseline_frames)
    rise     = rms / (baseline + 1e-8)

    elevated = rise >= rise_ratio

    min_dwell_frames = int(min_dwell_sec / dt)
    min_gap_frames   = int(min_gap_sec   / dt)

    events = []
    last_end = -min_gap_frames

    i = 0
    while i < len(elevated):
        if elevated[i]:
            j = i
            while j < len(elevated) and elevated[j]:
                j += 1
            run_len = j - i
            if run_len >= min_dwell_frames:
                peak_offset = int(np.argmax(rms_n[i:j]))
                peak_frame  = i + peak_offset
                if peak_frame - last_end >= min_gap_frames:
                    events.append({
                        "time_sec":   float(times[peak_frame]),
                        "peak_rms_n": float(rms_n[peak_frame]),
                        "rise":       float(rise[peak_frame]),
                        "dwell_sec":  float(run_len * dt),
                    })
                    last_end = j
            i = j
        else:
            i += 1

    return times, rms_n, rise, elevated, events


def process_file(input_path, rms_dir=RMS_OUT_DIR, events_dir=EVENTS_OUT_DIR):
    os.makedirs(rms_dir,    exist_ok=True)
    os.makedirs(events_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(input_path))[0]
    print(f"[{base}] loading…")

    y, sr = load_audio(input_path)
    print(f"[{base}] loaded {len(y)/sr:.1f}s at sr={sr}")

    rms = compute_rms(y, sr=sr)
    times, rms_n, rise, elevated, events = detect_events(rms)

    rms_path = os.path.join(rms_dir, base + ".csv")
    pd.DataFrame({
        "time_sec": times,
        "rms":      rms,
        "rms_n":    rms_n,
        "rise":     rise,
        "elevated": elevated.astype(int),
    }).to_csv(rms_path, index=False)

    events_path = os.path.join(events_dir, base + ".csv")
    pd.DataFrame(events,
                 columns=["time_sec", "peak_rms_n", "rise", "dwell_sec"]
                 ).to_csv(events_path, index=False)

    print(f"[{base}] {len(events)} events  →  {events_path}")
    print(f"[{base}] curve              →  {rms_path}")


def main():
    os.makedirs(RMS_OUT_DIR,    exist_ok=True)
    os.makedirs(EVENTS_OUT_DIR, exist_ok=True)

    for fname in sorted(os.listdir(AUDIO_DIR)):
        if fname.endswith((".mp3", ".wav")):
            process_file(os.path.join(AUDIO_DIR, fname))


if __name__ == "__main__":
    main()
