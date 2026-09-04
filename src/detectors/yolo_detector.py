"""
src/detectors/yolo_detector.py
Pass 1 Video Inference Orchestrator.
Processes raw broadcast frames, performs person detection and persistent 2-player Re-ID,
invokes the high-resolution TennisBallDetector, and handles scene cut tracking resets.
"""

import math
import os
import cv2
import numpy as np
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
    """
    Finds the first available custom model checkpoint or falls back to yolov8n.pt.

    Returns:
        str: Absolute or relative file path to the preferred YOLO checkpoint.
    """
    for candidate in MODEL_CANDIDATES:
        if os.path.exists(candidate):
            print(f"Loading player model weights: '{candidate}'")
            return candidate
    return 'yolov8n.pt'


def load_detector() -> YOLO:
    """
    Initializes and returns the primary YOLO model instance for person and object detection.

    Returns:
        YOLO: Instantiated YOLO model.
    """
    model_path = get_model_path()
    return YOLO(model_path)


def extract_detections(cap: cv2.VideoCapture, model: YOLO, roi_polygon_pixels: np.ndarray) -> list[dict]:
    """
    Pass 1: Runs parallel feature extraction across all video frames using:
    - High-resolution TennisBallDetector (1280px)
    - Persistent PlayerTracker with court net partitioning
    - HSV histogram scene cut detection

    Args:
        cap (cv2.VideoCapture): Open OpenCV video capture stream.
        model (YOLO): Loaded YOLO model for player detection.
        roi_polygon_pixels (np.ndarray): Polygon vertices for court boundary spatial filtering.

    Returns:
        list[dict]: Sequential list of per-frame detection dictionaries containing:
            - 'ball' (tuple[float, float] | None): (x, y) ball coordinates.
            - 'players' (dict): Tracked player bounding boxes keyed by player ID (1 or 2).
            - 'scene_cut' (bool): True if this frame was flagged as an abrupt scene cut.
    """
    print("\n--- Pass 1: Extracting Detections with High-Res Ball Engine & Player Re-ID ---")
    frame_detections = []
    frame_idx = 0
    raw_ball_detections_count = 0
    scene_cuts_count = 0

    # Initialize tracking instances
    player_tracker = PlayerTracker()
    ball_detector = TennisBallDetector(model=model)
    prev_hist = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_height, frame_width = frame.shape[:2]

        # ----------------------------------------------------------------------
        # 1. Scene Cut & Camera Transition Detection
        # ----------------------------------------------------------------------
        is_cut, curr_hist = detect_scene_cut(prev_hist, frame)
        prev_hist = curr_hist
        if is_cut:
            scene_cuts_count += 1
            # Reset player and ball tracking buffers on camera transitions
            player_tracker.reset()
            ball_detector.reset()

        # ----------------------------------------------------------------------
        # 2. Player Inference (YOLO standard scale)
        # ----------------------------------------------------------------------
        player_results = model.predict(frame, conf=PERSON_CONF_THRESHOLD, verbose=False)
        raw_detected_players = []

        if player_results and len(player_results) > 0:
            boxes = player_results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Filter for person class with strict confidence
                if cls_id == COCO_PERSON_CLASS_ID and conf >= PERSON_CONF_THRESHOLD:
                    # Feet contact position (bottom-center of bounding box)
                    feet_pos = ((x1 + x2) / 2.0, y2)
                    if is_inside_roi(feet_pos, roi_polygon_pixels):
                        raw_detected_players.append((int(x1), int(y1), int(x2), int(y2), conf))

        # Assign persistent player IDs (Player 1 = Near Court, Player 2 = Far Court)
        tracked_players = player_tracker.track_players(raw_detected_players, frame_height)

        # ----------------------------------------------------------------------
        # 3. High-Resolution Specialized Ball Detection (imgsz=1280 + strict court ROI)
        # ----------------------------------------------------------------------
        selected_ball = ball_detector.detect_ball(frame, frame_idx, roi_polygon_pixels)
        if selected_ball is not None:
            raw_ball_detections_count += 1

        # Store aggregated per-frame state for Pass 2 processing
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
