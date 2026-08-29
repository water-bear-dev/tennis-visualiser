import math
import os
import cv2
from ultralytics import YOLO
from src.config import (
    MODEL_CANDIDATES,
    BALL_CONF_THRESHOLD,
    PERSON_CONF_THRESHOLD,
    COCO_BALL_CLASS_ID,
    COCO_PERSON_CLASS_ID,
    MAX_MISSING_FRAMES,
    MAX_BALL_SPEED_PIXELS
)
from src.utils.roi_utils import is_inside_roi
from src.utils.scene_utils import detect_scene_cut
from src.trackers.player_tracker import PlayerTracker


def get_model_path() -> str:
    """Finds the first available custom model or falls back to yolov8n.pt."""
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            print(f"Loading local weights: '{candidate}'")
            return candidate
    print(f"No custom weights found. Using default '{MODEL_CANDIDATES[-1]}'")
    return MODEL_CANDIDATES[-1]


def load_detector() -> YOLO:
    """Initializes and returns the YOLO model."""
    model_path = get_model_path()
    return YOLO(model_path)


def calculate_distance(p1, p2) -> float:
    """Calculates Euclidean distance between two 2D points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def extract_detections(cap: cv2.VideoCapture, model: YOLO, roi_polygon_pixels) -> list:
    """
    Pass 1: Runs YOLOv8 inference across all video frames with velocity filtering,
    strict ROI enforcement, scene cut detection, and persistent 2-player tracking.
    """
    print("\n--- Pass 1: Extracting Detections with 2-Player Tracking & Safety Filters ---")
    frame_detections = []
    frame_idx = 0
    raw_ball_detections_count = 0
    scene_cuts_count = 0

    player_tracker = PlayerTracker()
    prev_hist = None
    last_ball_pos = None
    last_ball_frame = -1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_height, frame_width = frame.shape[:2]

        # 1. Scene Cut Detection
        is_cut, curr_hist = detect_scene_cut(prev_hist, frame)
        prev_hist = curr_hist
        if is_cut:
            scene_cuts_count += 1
            last_ball_pos = None
            last_ball_frame = -1
            player_tracker.reset()

        # 2. Run YOLO Inference
        results = model.predict(frame, conf=min(BALL_CONF_THRESHOLD, PERSON_CONF_THRESHOLD), verbose=False)

        candidate_balls = []
        raw_detected_players = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Ball detection candidate
                if cls_id == COCO_BALL_CLASS_ID and conf >= BALL_CONF_THRESHOLD:
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    # Strict ROI check on candidate center
                    if is_inside_roi((center_x, center_y), roi_polygon_pixels):
                        candidate_balls.append({
                            'pos': (center_x, center_y),
                            'conf': conf
                        })

                # Player detection candidate
                elif cls_id == COCO_PERSON_CLASS_ID and conf >= PERSON_CONF_THRESHOLD:
                    feet_pos = ((x1 + x2) / 2.0, y2)
                    if is_inside_roi(feet_pos, roi_polygon_pixels):
                        raw_detected_players.append((int(x1), int(y1), int(x2), int(y2), conf))

        # 3. Persistent 2-Player Assignment
        tracked_players = player_tracker.track_players(raw_detected_players, frame_height)

        # 4. Velocity / Physical Limit Filtering for Ball
        selected_ball = None
        if candidate_balls:
            if last_ball_pos is not None:
                dt = frame_idx - last_ball_frame
                if dt <= MAX_MISSING_FRAMES:
                    max_allowed_dist = dt * MAX_BALL_SPEED_PIXELS
                    # Filter candidates within physical speed limit
                    valid_candidates = [
                        c for c in candidate_balls 
                        if calculate_distance(c['pos'], last_ball_pos) <= max_allowed_dist
                    ]
                    if valid_candidates:
                        # Choose closest to last position among valid candidates
                        valid_candidates.sort(key=lambda c: calculate_distance(c['pos'], last_ball_pos))
                        selected_ball = valid_candidates[0]['pos']
                else:
                    # Gap too large: reset track and start fresh track with highest confidence candidate
                    candidate_balls.sort(key=lambda c: c['conf'], reverse=True)
                    selected_ball = candidate_balls[0]['pos']
            else:
                # No active track: start new track with highest confidence candidate
                candidate_balls.sort(key=lambda c: c['conf'], reverse=True)
                selected_ball = candidate_balls[0]['pos']

        if selected_ball is not None:
            raw_ball_detections_count += 1
            last_ball_pos = selected_ball
            last_ball_frame = frame_idx

        frame_detections.append({
            'ball': selected_ball,
            'players': tracked_players,
            'scene_cut': is_cut
        })

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"Pass 1: Analyzed {frame_idx} frames... (Balls: {raw_ball_detections_count}, Scene Cuts: {scene_cuts_count})")

    print(f"Pass 1 Complete: Total Frames={frame_idx}, Raw Balls={raw_ball_detections_count}, Scene Cuts={scene_cuts_count}")
    return frame_detections
