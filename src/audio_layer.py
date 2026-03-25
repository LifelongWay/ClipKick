import os
import librosa
import numpy as np
import pandas as pd


# -------------------------
# Load audio
# -------------------------
def load_audio(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr


# -------------------------
# Feature extraction
# -------------------------
def extract_rms(y):
    rms = librosa.feature.rms(y=y)[0]  # shape: (n_frames,)
    return rms


# -------------------------
# Normalize signal (0–1)
# -------------------------
def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-8)


# -------------------------
# Compute excitement score
# -------------------------
def compute_audio_score(rms):
    rms_norm = normalize(rms)

    # Simple heuristic: use RMS directly
    score = rms_norm

    return score


# -------------------------
# Save results
# -------------------------
def save_scores(times, rms, score, output_path):
    df = pd.DataFrame({
        "time_sec": times,
        "rms": rms,
        "audio_score": score
    })

    df.to_csv(output_path, index=False)


# -------------------------
# Process single file
# -------------------------
def process_file(input_path, output_dir):
    print(f"Processing: {input_path}")

    # Load
    y, sr = load_audio(input_path)

    # Features
    rms = extract_rms(y)

    # Time axis
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)

    # Score
    score = compute_audio_score(rms)

    # Output filename
    base = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, base + ".csv")

    # Save
    save_scores(times, rms, score, output_path)

    print(f"Saved → {output_path}")


# -------------------------
# Main runner
# -------------------------
def main():
    input_dir = "data/audio"
    output_dir = "data/audio_scores"

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        if file.endswith(".mp3") or file.endswith(".wav"):
            input_path = os.path.join(input_dir, file)
            process_file(input_path, output_dir)


if __name__ == "__main__":
    main()
