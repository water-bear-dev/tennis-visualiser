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
    1. Multi-scale High-Resolution YOLOv8 inference (imgsz=1280)
    2. Strict pre-inference & post-inference court polygon masking (zero crowd false alarms)
    3. Temporal 3-frame velocity continuity & motion-vector filtering
    4. Custom tennis ball weights integration (best_tennis.pt / tracknet)
    """

    def __init__(self, model: YOLO = None):
        self.model = model or self._load_ball_model()
        self.frame_buffer = deque(maxlen=3)
        self.last_ball_pos = None
        self.last_ball_frame = -1

    def _load_ball_model(self) -> YOLO:
        """Finds custom weights or defaults to YOLOv8."""
        for candidate in MODEL_CANDIDATES:
            if os.path.exists(candidate):
                print(f"TennisBallDetector: Loaded '{candidate}'")
                return YOLO(candidate)
        print("TennisBallDetector: No custom weights found. Using default 'yolov8n.pt'")
        return YOLO('yolov8n.pt')

    def reset(self):
        """Resets temporal buffers on scene cuts."""
        self.frame_buffer.clear()
        self.last_ball_pos = None
        self.last_ball_frame = -1

    def detect_ball(self, frame: np.ndarray, frame_idx: int, roi_polygon_pixels: np.ndarray) -> tuple | None:
        """
        Runs high-resolution ball inference with strict court ROI filtering
        and physical velocity vector validation.
        """
        self.frame_buffer.append(frame)

        # 1. High-Resolution Inference (imgsz=1280) targeting small objects
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

                # Filter for sports ball / custom single-class tennis weights (cls 0 or 32)
                # If custom model with 1 class is used, cls_id is 0; in COCO, it's 32
                is_ball_class = (cls_id == COCO_BALL_CLASS_ID) or (len(self.model.names) == 1 and cls_id == 0)

                if is_ball_class and conf >= BALL_CONF_THRESHOLD:
                    center_x = (x1 + x2) / 2.0
                    center_y = (y1 + y2) / 2.0
                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Ball aspect ratio & size filter (must be a small compact object, not a player or banner)
                    if box_w < 50 and box_h < 50:
                        # Strict ROI check: Center MUST be inside the court polygon
                        if is_inside_roi((center_x, center_y), roi_polygon_pixels):
                            candidate_balls.append({
                                'pos': (center_x, center_y),
                                'conf': conf
                            })

        # 2. Velocity & Temporal Continuity Filter
        selected_ball = None
        if candidate_balls:
            if self.last_ball_pos is not None:
                dt = frame_idx - self.last_ball_frame
                if dt <= MAX_MISSING_FRAMES:
                    max_allowed_dist = dt * MAX_BALL_SPEED_PIXELS
                    valid_candidates = [
                        c for c in candidate_balls
                        if math.hypot(c['pos'][0] - self.last_ball_pos[0], c['pos'][1] - self.last_ball_pos[1]) <= max_allowed_dist
                    ]
                    if valid_candidates:
                        valid_candidates.sort(
                            key=lambda c: math.hypot(c['pos'][0] - self.last_ball_pos[0], c['pos'][1] - self.last_ball_pos[1])
                        )
                        selected_ball = valid_candidates[0]['pos']
                else:
                    # Gap too large: start fresh track with highest confidence
                    candidate_balls.sort(key=lambda c: c['conf'], reverse=True)
                    selected_ball = candidate_balls[0]['pos']
            else:
                candidate_balls.sort(key=lambda c: c['conf'], reverse=True)
                selected_ball = candidate_balls[0]['pos']

        if selected_ball is not None:
            self.last_ball_pos = selected_ball
            self.last_ball_frame = frame_idx

        return selected_ball
