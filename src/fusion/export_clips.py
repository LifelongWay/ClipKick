"""
Shared clip exporter — model-agnostic.

Turns ANY model's highlights CSV (start,end,event_type,score) into actual video
clips cut from the match mp4, plus an optional stitched highlight reel. Reusable by
Models A/B/E/F.

    python src/fusion/export_clips.py --match <id> --video data/raw/video/<id>.mp4
    python src/fusion/export_clips.py --match <id> --video <...> --reel --sort score
    python src/fusion/export_clips.py --highlights results/fusion/highlights/<id>.csv --video <...>
"""

import argparse
import os

import pandas as pd

try:
    from . import common
except ImportError:
    import common

CLIPS_DIR = "results/clips"


def _moviepy():
    """Import moviepy across 1.x (moviepy.editor) and 2.x (top-level)."""
    try:
        from moviepy import VideoFileClip, concatenate_videoclips        # 2.x
    except ImportError:
        from moviepy.editor import VideoFileClip, concatenate_videoclips  # 1.x
    return VideoFileClip, concatenate_videoclips


def _subclip(clip, a, b):
    """`.subclipped` on moviepy 2.x, `.subclip` on 1.x."""
    fn = getattr(clip, "subclipped", None) or clip.subclip
    return fn(a, b)


def _load(highlights_csv):
    if not os.path.exists(highlights_csv):
        print(f"no highlights CSV at {highlights_csv}")
        return []
    df = pd.read_csv(highlights_csv)
    rows = [{"start": float(r.start), "end": float(getattr(r, "end", r.start)),
             "event_type": str(r.event_type), "score": float(getattr(r, "score", 0.0))}
            for r in df.itertuples()]
    return rows


def export(match_id, video_path, highlights_csv=None, out_dir=None, reel=False, sort="time"):
    """Cut one mp4 per highlight from `video_path`; optionally stitch a reel."""
    # Cheap checks first, before importing the heavy media lib.
    if highlights_csv is None:
        highlights_csv = os.path.join(common.HIGHLIGHTS_DIR, match_id + ".csv")
    rows = _load(highlights_csv)
    if not rows:
        print(f"[{match_id}] no highlights to export")
        return []
    if not os.path.exists(video_path):
        print(f"[{match_id}] video not found: {video_path}")
        return []

    VideoFileClip, concatenate_videoclips = _moviepy()

    rows.sort(key=lambda r: -r["score"] if sort == "score" else r["start"])

    out_dir = out_dir or os.path.join(CLIPS_DIR, match_id)
    os.makedirs(out_dir, exist_ok=True)

    source = VideoFileClip(video_path)
    duration = source.duration
    written, subclips = [], []
    for i, r in enumerate(rows):
        start = max(0.0, r["start"])
        end = min(duration, r["end"])
        if end <= start:
            print(f"[{match_id}] skip clip {i} — empty/out-of-range [{r['start']}, {r['end']}]")
            continue
        clip = _subclip(source, start, end)
        name = f"{i:02d}_{r['event_type']}_{int(start)}s.mp4"
        path = os.path.join(out_dir, name)
        clip.write_videofile(path, codec="libx264", audio_codec="aac", logger=None)
        written.append(path)
        subclips.append(clip)
        print(f"[{match_id}] clip → {path}")

    if reel and subclips:
        reel_path = os.path.join(out_dir, "reel.mp4")
        concatenate_videoclips(subclips).write_videofile(
            reel_path, codec="libx264", audio_codec="aac", logger=None)
        print(f"[{match_id}] reel → {reel_path}")
        written.append(reel_path)

    source.close()
    print(f"[{match_id}] exported {len(written)} file(s) → {out_dir}")
    return written


def main():
    parser = argparse.ArgumentParser(description="Export highlight clips from a match video")
    parser.add_argument("--match", default=None)
    parser.add_argument("--highlights", default=None, help="highlights CSV (defaults to the match's)")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--reel", action="store_true", help="also stitch one concatenated reel")
    parser.add_argument("--sort", choices=["time", "score"], default="time")
    args = parser.parse_args()

    match_id = args.match or (
        os.path.splitext(os.path.basename(args.highlights))[0] if args.highlights else None)
    if not match_id:
        print("pass --match or --highlights")
        return
    export(match_id, args.video, highlights_csv=args.highlights,
           out_dir=args.out_dir, reel=args.reel, sort=args.sort)


if __name__ == "__main__":
    main()
