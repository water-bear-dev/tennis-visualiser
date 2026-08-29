import os
import cv2
from ultralytics import YOLO
from src.config import (
    MODEL_CANDIDATES,
    BALL_CONF_THRESHOLD,
    PERSON_CONF_THRESHOLD,
    COCO_BALL_CLASS_ID,
    COCO_PERSON_CLASS_ID
)
from src.utils.roi_utils import is_inside_roi


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


def extract_detections(cap: cv2.VideoCapture, model: YOLO, roi_polygon_pixels) -> list:
    """
    Pass 1: Runs YOLOv8 inference across all video frames to extract player and ball positions.
    """
    print("\n--- Pass 1: Extracting Detections with Tuned Confidence & ROI ---")
    frame_detections = []
    frame_idx = 0
    raw_ball_detections_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Run inference targeting both person and sports ball with low baseline conf
        results = model.predict(frame, conf=min(BALL_CONF_THRESHOLD, PERSON_CONF_THRESHOLD), verbose=False)
        
        detected_ball = None
        best_ball_conf = 0.0
        detected_players = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Ball detection
                if cls_id == COCO_BALL_CLASS_ID and conf >= BALL_CONF_THRESHOLD:
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0

                    # Apply ROI filtering on ball center
                    if is_inside_roi((center_x, center_y), roi_polygon_pixels):
                        # Pick highest confidence ball candidate in case of duplicates
                        if conf > best_ball_conf:
                            best_ball_conf = conf
                            detected_ball = (center_x, center_y)

                # Player detection
                elif cls_id == COCO_PERSON_CLASS_ID and conf >= PERSON_CONF_THRESHOLD:
                    # Filter players based on bottom-center feet position
                    feet_pos = ((x1 + x2) / 2.0, y2)
                    if is_inside_roi(feet_pos, roi_polygon_pixels):
                        detected_players.append((int(x1), int(y1), int(x2), int(y2), conf))

        if detected_ball is not None:
            raw_ball_detections_count += 1

        frame_detections.append({
            'ball': detected_ball,
            'players': detected_players
        })

        frame_idx += 1
        if frame_idx % 60 == 0:
            print(f"Pass 1: Analyzed {frame_idx} frames... (Raw ball detected in {raw_ball_detections_count} frames)")

    print(f"Pass 1 Complete: Total Frames={frame_idx}, Raw Ball Detections={raw_ball_detections_count}")
    return frame_detections
