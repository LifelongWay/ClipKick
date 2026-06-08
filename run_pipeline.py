"""
End-to-end ClipKick pipeline:

    audio energy  →  speech (Granite, F1)  →  vision (gated)  →  timeline (F2)  →  fusion

Run from project root:
    python run_pipeline.py --video match.mp4 --audio data/raw/audio/<match>.mp3 --output out.mp4
    python run_pipeline.py --video ... --audio ... --output ... --fusion E --skip-vision
"""

import argparse
import os
import subprocess

FUSION_SCRIPT = {
    "A": "src/fusion/fuse_decision.py",
    "B": "src/fusion/fuse_stacked.py",
    "E": "src/fusion/fuse_progressive.py",
}


def run(cmd):
    print("›", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fusion", choices=["A", "B", "E", "none"], default="A")
    parser.add_argument("--speech-mode", choices=["sliding", "gated"], default="sliding")
    parser.add_argument("--skip-vision", action="store_true")
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.audio))[0]
    events_csv = f"results/audio/events/{base}.csv"

    # 1. Audio energy layer (whole match) — proposes candidate windows.
    run(["python", "src/audio/audio_layer.py"])

    # Model E is self-contained: it drives Granite itself in two passes and uses
    # neither the F1 speech CSV nor the F2 timeline nor vision. Skip those stages.
    if args.fusion == "E":
        run(["python", FUSION_SCRIPT["E"], "--match", base, "--audio", args.audio])
        print(f"Highlights → results/fusion/highlights/{base}.csv "
              f"(+ {base}_preliminary.csv from Pass 1)")
        return

    # 2. Speech layer (F1) — Granite keyword/excitement signals (for A/B).
    run(["python", "src/speech/speech_layer.py", "--audio", args.audio, "--mode", args.speech_mode])

    # 3. Vision (gated to audio windows) — optional/expensive.
    if not args.skip_vision:
        run(["python", "src/video/process_video.py",
             "--input", args.video, "--audio", args.audio,
             "--output", args.output, "--events", events_csv])

    # 4. Shared timeline (F2).
    run(["python", "src/fusion/timeline.py", "--match", base])

    # 5. Fusion model → highlights.
    if args.fusion != "none":
        run(["python", FUSION_SCRIPT[args.fusion], "--match", base])
        print(f"Highlights → results/fusion/highlights/{base}.csv")


if __name__ == "__main__":
    main()
