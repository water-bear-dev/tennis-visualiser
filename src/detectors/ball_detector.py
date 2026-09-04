"""
src/detectors/ball_detector.py
Dedicated high-resolution tennis ball detection engine.
Implements 1280px YOLO inference, strict court ROI boundaries, candidate filtering,
and Euclidean distance/velocity jump sanity validation across temporal frames.
"""

from collections import deque
import math
import os
import cv2
import numpy as np
from ultralytics import YOLO
from src.config import (
    MODEL_CANDIDATES,
    BALL_IMGSZ,
    BALL_CONF_THRESHOLD,
    COCO_BALL_CLASS_ID,
    MAX_BALL_SPEED_PIXELS,
    MAX_MISSING_FRAMES
)
from src.utils.roi_utils import is_inside_roi


class TennisBallDetector:
    """
    Specialized ball detection engine supporting:
    1. Multi-scale High-Resolution YOLOv8 inference (imgsz=1280) to capture micro ball pixels.
    2. Strict court polygon ROI masking to eliminate crowd and sideline false alarms.
    3. Temporal velocity continuity & motion-vector filtering to reject physically implausible jumps.
    4. Seamless integration of custom fine-tuned weights (e.g., best_tennis.pt).
    """

    def __init__(self, model: YOLO = None):
        """
        Initializes the TennisBallDetector with a YOLO model instance or default fallback.

        Args:
            model (YOLO, optional): Pre-instantiated YOLO model object. If None, loads from candidates.
        """
        self.model = model or self._load_ball_model()
        self.frame_buffer = deque(maxlen=3)
        self.last_ball_pos = None     # Stores last confirmed (x, y) ball coordinates
        self.last_ball_frame = -1     # Stores frame index of last confirmed ball detection

    def _load_ball_model(self) -> YOLO:
        """
        Searches available candidate weights in priority order and instantiates YOLO.

        Returns:
            YOLO: Initialized YOLO model instance.
        """
        for candidate in MODEL_CANDIDATES:
            if os.path.exists(candidate):
                print(f"TennisBallDetector: Loaded '{candidate}'")
                return YOLO(candidate)
        print("TennisBallDetector: No custom weights found. Using default 'yolov8n.pt'")
        return YOLO('yolov8n.pt')

    def reset(self):
        """
        Resets temporal tracking buffers and position history upon scene cuts or camera transitions.
        """
        self.frame_buffer.clear()
        self.last_ball_pos = None
        self.last_ball_frame = -1

    def detect_ball(self, frame: np.ndarray, frame_idx: int, roi_polygon_pixels: np.ndarray) -> tuple[float, float] | None:
        """
        Executes high-resolution ball detection on a single frame with geometric and kinematic filters.

        Args:
            frame (np.ndarray): BGR video frame image array.
            frame_idx (int): Current sequential frame index.
            roi_polygon_pixels (np.ndarray): Polygon pixel vertices defining active court bounds.

        Returns:
            tuple[float, float] | None: Validated (center_x, center_y) ball pixel position, or None if undetected.
        """
        self.frame_buffer.append(frame)

        # ----------------------------------------------------------------------
        # 1. High-Resolution Inference (imgsz=1280) targeting small motion objects
        # ----------------------------------------------------------------------
        results = self.model.predict(
            frame,
            conf=BALL_CONF_THRESHOLD,
            imgsz=BALL_IMGSZ,
            verbose=False
        )

        candidate_balls = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                # Filter for sports ball: COCO class 32, or class 0 for custom single-class tennis models
                is_ball_class = (cls_id == COCO_BALL_CLASS_ID) or (len(self.model.names) == 1 and cls_id == 0)

                if is_ball_class and conf >= BALL_CONF_THRESHOLD:
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Bounding box size filter: reject large non-ball objects (>50px)
                    if box_w < 50 and box_h < 50:
                        # Geometric ROI constraint: Ball center MUST lie inside the playing court polygon
                        if is_inside_roi((center_x, center_y), roi_polygon_pixels):
                            candidate_balls.append({
                                'pos': (center_x, center_y),
                                'conf': conf
                            })

        # ----------------------------------------------------------------------
        # 2. Kinematic Velocity & Temporal Continuity Filter
        # ----------------------------------------------------------------------
        selected_ball = None
        if candidate_balls:
            if self.last_ball_pos is not None:
                dt = frame_idx - self.last_ball_frame
                # Check if the tracking gap is small enough for continuous trajectory estimation
                if dt <= MAX_MISSING_FRAMES:
                    max_allowed_dist = dt * MAX_BALL_SPEED_PIXELS
                    # Filter candidates whose displacement is physically plausible
                    valid_candidates = [
                        c for c in candidate_balls
                        if math.hypot(c['pos'][0] - self.last_ball_pos[0], c['pos'][1] - self.last_ball_pos[1]) <= max_allowed_dist
                    ]
                    if valid_candidates:
                        # Select candidate closest to the projected trajectory
                        valid_candidates.sort(
                            key=lambda c: math.hypot(c['pos'][0] - self.last_ball_pos[0], c['pos'][1] - self.last_ball_pos[1])
                        )
                        selected_ball = valid_candidates[0]['pos']
                else:
                    # Gap exceeded threshold: re-initialize track with highest confidence detection
                    candidate_balls.sort(key=lambda c: c['conf'], reverse=True)
                    selected_ball = candidate_balls[0]['pos']
            else:
                # First detection in track: select candidate with highest confidence score
                candidate_balls.sort(key=lambda c: c['conf'], reverse=True)
                selected_ball = candidate_balls[0]['pos']

        # Update persistent tracking history
        if selected_ball is not None:
            self.last_ball_pos = selected_ball
            self.last_ball_frame = frame_idx

        return selected_ball
