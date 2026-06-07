import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base_name = os.path.splitext(os.path.basename(args.audio))[0]
    events_csv = f"results/audio/events/{base_name}.csv"

    subprocess.run(["python", "src/audio/audio_layer.py"], check=True)

    subprocess.run([
        "python", "src/video/process_video.py",
        "--input", args.video,
        "--audio", args.audio,
        "--output", args.output,
        "--events", events_csv
    ], check=True)

if __name__ == "__main__":
    main()