import math
import os
import cv2
from ultralytics import YOLO
from src.config import (
    MODEL_CANDIDATES,
    PERSON_CONF_THRESHOLD,
    COCO_PERSON_CLASS_ID
)
from src.utils.roi_utils import is_inside_roi
from src.utils.scene_utils import detect_scene_cut
from src.trackers.player_tracker import PlayerTracker
from src.detectors.ball_detector import TennisBallDetector


def get_model_path() -> str:
    """Finds the first available custom model or falls back to yolov8n.pt."""
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            print(f"Loading player model weights: '{candidate}'")
            return candidate
    return 'yolov8n.pt'


def load_detector() -> YOLO:
    """Initializes and returns the YOLO player detector model."""
    model_path = get_model_path()
    return YOLO(model_path)


def extract_detections(cap: cv2.VideoCapture, model: YOLO, roi_polygon_pixels) -> list:
    """
    Pass 1: Runs parallel extraction with specialized high-res TennisBallDetector
    and PlayerTracker across all video frames.
    """
    print("\n--- Pass 1: Extracting Detections with High-Res Ball Engine & Player Re-ID ---")
    frame_detections = []
    frame_idx = 0
    raw_ball_detections_count = 0
    scene_cuts_count = 0

    player_tracker = PlayerTracker()
    ball_detector = TennisBallDetector(model=model)
    prev_hist = None

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
            player_tracker.reset()
            ball_detector.reset()

        # 2. Player Inference (YOLO standard scale)
        player_results = model.predict(frame, conf=PERSON_CONF_THRESHOLD, verbose=False)
        raw_detected_players = []

        if player_results and len(player_results) > 0:
            boxes = player_results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                if cls_id == COCO_PERSON_CLASS_ID and conf >= PERSON_CONF_THRESHOLD:
                    feet_pos = ((x1 + x2) / 2.0, y2)
                    if is_inside_roi(feet_pos, roi_polygon_pixels):
                        raw_detected_players.append((int(x1), int(y1), int(x2), int(y2), conf))

        # Persistent 2-Player Assignment
        tracked_players = player_tracker.track_players(raw_detected_players, frame_height)

        # 3. High-Resolution Specialized Ball Detection (imgsz=1280 + strict court ROI)
        selected_ball = ball_detector.detect_ball(frame, frame_idx, roi_polygon_pixels)
        if selected_ball is not None:
            raw_ball_detections_count += 1

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
