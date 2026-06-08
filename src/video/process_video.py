import cv2
import numpy as np
import time
import os
import math
import pandas as pd
import argparse
import torch
from collections import deque
from transformers import pipeline
from PIL import Image
from ultralytics import YOLO
from tqdm import tqdm


def pick_device():
    """cuda on Colab, mps on Apple Silicon, cpu fallback — same script everywhere."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ParticleFilter:
    def __init__(self, num_particles, width, height):
        self.num_particles = num_particles
        self.width = width
        self.height = height
        self.particles = np.zeros((num_particles, 4))
        self.weights = np.ones(num_particles) / num_particles

    def initialize(self, x, y):
        self.particles[:, 0] = x + np.random.normal(0, 10, self.num_particles)
        self.particles[:, 1] = y + np.random.normal(0, 10, self.num_particles)
        self.particles[:, 2] = np.random.normal(0, 5, self.num_particles)
        self.particles[:, 3] = np.random.normal(0, 5, self.num_particles)

    def predict(self):
        self.particles[:, 0] += self.particles[:, 2] + np.random.normal(0, 10, self.num_particles)
        self.particles[:, 1] += self.particles[:, 3] + np.random.normal(0, 10, self.num_particles)
        self.particles[:, 2] *= 0.95
        self.particles[:, 3] *= 0.95

    def update(self, measurement):
        if measurement is None: return
        mx, my = measurement[0], measurement[1]
        distances = np.sqrt((self.particles[:, 0] - mx) ** 2 + (self.particles[:, 1] - my) ** 2)
        self.weights = np.exp(-distances ** 2 / (2 * 15 ** 2)) + 1e-300
        self.weights /= self.weights.sum()
        indices = np.random.choice(np.arange(self.num_particles), size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]

    def get_state(self):
        return np.mean(self.particles, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args()

    device = pick_device()
    print(f"🚀 Initializing Dual Vision System on device: {device}")

    # 1. Goal Structure Spotter: Grounding DINO (Fires rarely / when tracker drops)
    dino_model = pipeline("zero-shot-object-detection", model="IDEA-Research/grounding-dino-base", device=device)
    dino_labels = ["white metal goalpost", "mesh netting"]

    # 2. High-Speed Ball Tracker: Native YOLO (Fires every frame)
    yolo_model = YOLO("yolo11m.pt")

    cap = cv2.VideoCapture(args.input)
    w_vid = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_vid = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_vid, h_vid))

    windows = []
    if os.path.exists(args.events):
        events_df = pd.read_csv(args.events)
        for index, row in events_df.iterrows():
            t = row['time_sec']
            start_f = max(0, int((t - 10) * fps))
            min_end_f = min(total_frames, int((t + 5) * fps))
            max_end_f = min(total_frames, int((t + 15) * fps))
            windows.append([start_f, min_end_f, max_end_f])

    merged_windows = []
    if windows:
        windows.sort(key=lambda x: x[0])
        current_w = windows[0]
        for w in windows[1:]:
            if w[0] <= current_w[2]:
                current_w[1] = max(current_w[1], w[1])
                current_w[2] = max(current_w[2], w[2])
            else:
                merged_windows.append(current_w)
                current_w = w
        merged_windows.append(current_w)
    else:
        merged_windows = [[0, total_frames, total_frames]]

    # --- Video Optimization Scaling for DINO ---
    DINO_W = 640
    DINO_H = int(640 * h_vid / w_vid)
    scale_x = w_vid / DINO_W
    scale_y = h_vid / DINO_H

    total_window_frames = sum(max_e - start for start, _, max_e in merged_windows)
    pbar = tqdm(total=total_window_frames, desc="Processing Video Windows", unit="frame")

    goal_tracker = None
    goal_tracking_active = False
    MISS_COOLDOWN = fps * 3
    cooldown_frames_left = 0
    LOWER_GREEN = np.array([35, 40, 40])
    UPPER_GREEN = np.array([85, 255, 255])

    # Ball tracking state tools
    pf = ParticleFilter(2000, w_vid, h_vid)
    ball_initialized = False
    consecutive_rejections = 0
    ball_w, ball_h = 15, 15
    ball_history = deque(maxlen=10)
    confirmed_goal_frames = 0
    prev_hist = None

    start_time = time.time()
    frame_count = 0
    window_idx = 0
    frames_processed = 0
    actual_processed_windows = []
    vision_goal_events = []

    while cap.isOpened() and window_idx < len(merged_windows):
        target_start, min_target_end, max_target_end = merged_windows[window_idx]

        if frame_count < target_start:
            while frame_count < target_start:
                ret = cap.grab()
                if not ret: break
                frame_count += 1

            goal_tracking_active = False
            ball_initialized = False
            cooldown_frames_left = 0
            ball_history.clear()
            prev_hist = None

        ret, frame = cap.read()
        if not ret:
            if window_idx < len(merged_windows):
                actual_processed_windows.append([target_start, frame_count])
            break

        frame_count += 1
        frames_processed += 1
        pbar.update(1)

        # Broadcast Camera Cut Detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        curr_hist = cv2.calcHist([hsv_frame], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)

        is_camera_cut = False
        if prev_hist is not None:
            if cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL) < 0.6:
                is_camera_cut = True
                goal_tracking_active = False
                ball_initialized = False
                cooldown_frames_left = MISS_COOLDOWN

        prev_hist = curr_hist

        if frame_count > min_target_end:
            if is_camera_cut or frame_count >= max_target_end:
                actual_processed_windows.append([target_start, frame_count])
                window_idx += 1
                continue

        # =======================================================
        # PHASE 1: HIGH-SPEED BALL TRACKING (YOLOv11 - EVERY FRAME)
        # =======================================================
        if ball_initialized:
            pf.predict()

        # Run optimized YOLOv11 inference directly on the MPS device back-end
        yolo_results = yolo_model.predict(frame, imgsz=640, conf=0.1, classes=[32], device=device, verbose=False)

        best_ball_det = None
        if len(yolo_results[0].boxes) > 0:
            # Grab top confident soccer ball tracking box
            box = yolo_results[0].boxes[0]
            best_ball_det = box.xywh[0].tolist()  # [center_x, center_y, width, height]

        # Particle filter gating logic
        if best_ball_det and ball_initialized:
            current_state = pf.get_state()
            dist = np.sqrt((current_state[0] - best_ball_det[0]) ** 2 + (current_state[1] - best_ball_det[1]) ** 2)
            if dist > 100:  # Max allowable jump threshold
                consecutive_rejections += 1
                if consecutive_rejections > 10:
                    pf.initialize(best_ball_det[0], best_ball_det[1])
                    consecutive_rejections = 0
                else:
                    best_ball_det = None
            else:
                consecutive_rejections = 0

        if best_ball_det:
            if not ball_initialized:
                pf.initialize(best_ball_det[0], best_ball_det[1])
                ball_initialized = True
            else:
                pf.update([best_ball_det[0], best_ball_det[1]])
            ball_w, ball_h = best_ball_det[2], best_ball_det[3]

        ball_state = pf.get_state() if ball_initialized else None
        if ball_state is not None:
            ball_history.append((int(ball_state[0]), int(ball_state[1]), ball_w))

        # =======================================================
        # PHASE 2: STATIC GOAL DETECTION & TRACKING (DINO + CSRT)
        # =======================================================
        current_goal_bbox = None

        if cooldown_frames_left > 0:
            cooldown_frames_left -= 1
        elif frame_count == target_start or not goal_tracking_active:
            # DINO ONLY fires if we don't have an active tracking lock on the target structure
            small_frame = cv2.resize(frame, (DINO_W, DINO_H))
            frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            dino_results = dino_model(image=pil_image, candidate_labels=dino_labels)

            valid_results = []
            for r in dino_results:
                if r['score'] < 0.30: continue
                box = r['box']
                xmin = int(box['xmin'] * scale_x)
                ymin = int(box['ymin'] * scale_y)
                xmax = int(box['xmax'] * scale_x)
                ymax = int(box['ymax'] * scale_y)
                box_w, box_h = xmax - xmin, ymax - ymin

                if box_w < 80 and box_h < 80: continue
                if (box_w * box_h) > (w_vid * h_vid * 0.40):
                    y1, y2 = max(0, ymin), min(h_vid, ymax)
                    x1, x2 = max(0, xmin), min(w_vid, xmax)
                    if y2 <= y1 or x2 <= x1: continue
                    crop = frame[y1:y2, x1:x2]
                    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                    green_mask = cv2.inRange(hsv_crop, LOWER_GREEN, UPPER_GREEN)
                    grass_ratio = cv2.countNonZero(green_mask) / (box_w * box_h) if (box_w * box_h) > 0 else 1.0
                    if grass_ratio > 0.80: continue

                r['scaled_bbox'] = (xmin, ymin, box_w, box_h)
                valid_results.append(r)

            best_match = None
            goalposts = [r for r in valid_results if r['label'] == 'white metal goalpost']
            if goalposts:
                best_match = max(goalposts, key=lambda x: x['score'])
            elif not best_match:
                netting = [r for r in valid_results if r['label'] == 'mesh netting']
                if netting: best_match = max(netting, key=lambda x: x['score'])

            if best_match:
                dino_bbox = best_match['scaled_bbox']
                try:
                    # Preferred: Modern OpenCV 4.x Contrib CSRT syntax
                    goal_tracker = cv2.TrackerCSRT.create()
                except AttributeError:
                    try:
                        # Fallback 1: Legacy namespace if using an older wheel build
                        goal_tracker = cv2.legacy.TrackerCSRT_create()
                    except AttributeError:
                        try:
                            # Fallback 2: Stable Core MIL tracker (available everywhere in 4.x)
                            goal_tracker = cv2.TrackerMIL.create()
                        except AttributeError:
                            # Fallback 3: Old-style Core MIL creation method
                            goal_tracker = cv2.TrackerMIL_create()

                goal_tracker.init(frame, dino_bbox)
                goal_tracking_active = True
                current_goal_bbox = dino_bbox
            else:
                goal_tracking_active = False
                cooldown_frames_left = MISS_COOLDOWN

        elif goal_tracking_active:
            # If tracking is active, update using ultra-fast native C++ CSRT tracker (No deep math)
            success, tracked_bbox = goal_tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in tracked_bbox]
                margin = 15
                if x < margin or y < margin or (x + w) > (w_vid - margin) or (y + h) > (
                        h_vid - margin) or w < 60 or h < 60:
                    goal_tracking_active = False
                    cooldown_frames_left = MISS_COOLDOWN
                else:
                    current_goal_bbox = tracked_bbox
            else:
                goal_tracking_active = False

        # =======================================================
        # PHASE 3: INTERSECTION / FUSION PHYSICS
        # =======================================================
        if current_goal_bbox and len(ball_history) >= 5:
            gx, gy, gw, gh = [int(v) for v in current_goal_bbox]
            curr_bx, curr_by, _ = ball_history[-1]
            old_bx, old_by, _ = ball_history[-5]

            if (gx < curr_bx < (gx + gw)) and (gy < curr_by < (gy + gh)):
                recent_velocity = math.dist((curr_bx, curr_by), (old_bx, old_by))
                moving_up = curr_by < old_by
                near_crossbar = (curr_by - gy) < (gh * 0.2)

                if recent_velocity < 15 and not (moving_up and near_crossbar) and len(ball_history) == 10:
                    ancient_bx, ancient_by, _ = ball_history[0]
                    if math.dist((old_bx, old_by), (ancient_bx, ancient_by)) > 25:
                        confirmed_goal_frames = 45
                        vision_goal_events.append({"time_sec": round(frame_count / fps, 2), "confidence": 0.8})
                        ball_history.clear()

        # Visual Annotations
        if current_goal_bbox:
            x, y, w, h = [int(v) for v in current_goal_bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
            cv2.putText(frame, "Goal Track (CSRT)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if ball_state is not None:
            bx, by = int(ball_state[0]), int(ball_state[1])
            cv2.circle(frame, (bx, by), int(ball_w / 2), (0, 0, 255), -1)
            cv2.circle(frame, (bx, by), int(ball_w / 2) + 10, (0, 255, 0), 2)

        if confirmed_goal_frames > 0:
            cv2.putText(frame, "POTENTIAL GOAL DETECTED", (w_vid // 2 - 300, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5,
                        (0, 255, 0), 4)
            confirmed_goal_frames -= 1

        out.write(frame)

    pbar.close()
    cap.release()
    out.release()
    total_time = time.time() - start_time

    if total_time > 0:
        print(
            f"Video processing complete. Frames: {frames_processed}, Time: {total_time:.2f}s, FPS: {frames_processed / total_time:.2f}")
    else:
        print("Video processing complete.")

    vbase = os.path.splitext(os.path.basename(args.audio))[0]
    os.makedirs("results/vision/events", exist_ok=True)
    os.makedirs("results/vision/windows", exist_ok=True)
    pd.DataFrame(
        [{"time_sec": e["time_sec"], "type": "goal", "confidence": e["confidence"]} for e in vision_goal_events],
        columns=["time_sec", "type", "confidence"],
    ).to_csv(f"results/vision/events/{vbase}.csv", index=False)
    pd.DataFrame(
        [{"start_sec": round(s / fps, 2), "end_sec": round(e / fps, 2)} for s, e in actual_processed_windows],
        columns=["start_sec", "end_sec"],
    ).to_csv(f"results/vision/windows/{vbase}.csv", index=False)
    print(f"Vision: {len(vision_goal_events)} goal events, {len(actual_processed_windows)} windows → results/vision/")

    try:
        from moviepy import VideoFileClip, AudioFileClip, concatenate_audioclips

        video_clip = VideoFileClip(args.output)
        original_audio = AudioFileClip(args.audio)

        audio_clips = []
        for w in actual_processed_windows:
            start_sec = w[0] / fps
            end_sec = w[1] / fps
            if end_sec > start_sec:
                audio_clips.append(original_audio.subclipped(start_sec, end_sec))

        if audio_clips:
            final_audio = concatenate_audioclips(audio_clips)
            final_video = video_clip.with_audio(final_audio)
            final_output_path = args.output.replace(".mp4", "_with_audio.mp4")
            final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac", logger=None)

        video_clip.close()
        original_audio.close()
    except ImportError:
        pass


if __name__ == "__main__":
    main()