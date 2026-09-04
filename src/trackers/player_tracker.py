"""
src/trackers/player_tracker.py
Persistent 2-Player Tracking & Re-Identification Engine.
Spatially partitions court players across the net line into Player 1 (near court) and Player 2 (far court),
filtering out audience, ball boys, line judges, and bench personnel.
"""

import math
import numpy as np


class PlayerTracker:
    """
    Tracks and isolates the two active tennis players across video frames.
    Assigns persistent IDs: 'player_1' (near/bottom court) and 'player_2' (far/top court)
    using geometric net-line partitioning and temporal Euclidean nearest-neighbor tracking.
    """

    def __init__(self, net_y_ratio: float = 0.50, max_track_dist: float = 140.0):
        """
        Initializes the PlayerTracker.

        Args:
            net_y_ratio (float): Normalized Y-coordinate (0.0 - 1.0) dividing far court (top) and near court (bottom).
            max_track_dist (float): Maximum pixel distance for associating player across adjacent frames.
        """
        self.net_y_ratio = net_y_ratio
        self.max_track_dist = max_track_dist
        self.prev_p1_box = None  # Previous frame bounding box for Player 1 (near court)
        self.prev_p2_box = None  # Previous frame bounding box for Player 2 (far court)

    def _get_box_center(self, box: tuple) -> tuple[float, float]:
        """Calculates center (cx, cy) of a bounding box."""
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _get_box_feet(self, box: tuple) -> tuple[float, float]:
        """Calculates bottom-center feet contact position (cx, y2) of a bounding box."""
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2.0, float(y2))

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculates Euclidean distance between two 2D points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def _compute_iou(self, box1: tuple, box2: tuple) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes."""
        x1_a, y1_a, x2_a, y2_a = box1[:4]
        x1_b, y1_b, x2_b, y2_b = box2[:4]

        xi1 = max(x1_a, x1_b)
        yi1 = max(y1_a, y1_b)
        xi2 = min(x2_a, x2_b)
        yi2 = min(y2_a, y2_b)

        inter_w = max(0.0, xi2 - xi1)
        inter_h = max(0.0, yi2 - yi1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, x2_a - x1_a) * max(0.0, y2_a - y1_a)
        area_b = max(0.0, x2_b - x1_b) * max(0.0, y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0.0:
            return 0.0
        return inter_area / union_area

    def _match_best_candidate(self, candidates: list[tuple], prev_box: tuple | None) -> tuple | None:
        """
        Selects the single best player candidate from a court half using IoU and centroid proximity.
        """
        if not candidates:
            return None

        if prev_box is None:
            # First frame or re-initialization: choose candidate with highest detection confidence
            candidates.sort(key=lambda b: b[4] if len(b) > 4 else 0.0, reverse=True)
            return candidates[0]

        prev_center = self._get_box_center(prev_box)
        best_candidate = None
        best_cost = float('inf')

        for cand in candidates:
            cand_center = self._get_box_center(cand)
            dist = self._distance(cand_center, prev_center)
            iou = self._compute_iou(cand, prev_box)
            conf = cand[4] if len(cand) > 4 else 0.5

            # Fused association cost: lower is better
            cost = dist - (iou * 80.0) - (conf * 15.0)

            if cost < best_cost:
                best_cost = cost
                best_candidate = cand

        # Fallback verification: if best match is within allowable association distance
        if best_candidate is not None:
            dist = self._distance(self._get_box_center(best_candidate), prev_center)
            if dist <= self.max_track_dist * 2.0:
                return best_candidate

        # If all candidates are beyond tracking jump threshold, fallback to highest confidence
        candidates.sort(key=lambda b: b[4] if len(b) > 4 else 0.0, reverse=True)
        return candidates[0]

    def track_players(self, detected_players: list[tuple], frame_height: int) -> dict[str, tuple | None]:
        """
        Filters raw detected people into the two primary active tennis players.
        Enforces strict Court-Half Spatial Partitioning and single player per half.

        Args:
            detected_players (list[tuple]): List of detected candidate boxes (x1, y1, x2, y2, conf).
            frame_height (int): Pixel height of the video frame.

        Returns:
            dict: Dictionary with keys 'player_1' and 'player_2', mapping to (x1, y1, x2, y2, conf) or None.
        """
        if not detected_players:
            return {'player_1': None, 'player_2': None}

        # Calculate absolute pixel horizon for the tennis net
        net_y_pixel = frame_height * self.net_y_ratio

        # Strictly separate candidate detections across the net horizon
        top_candidates = []     # Far court (Player 2 only: Y_feet <= net_y_pixel)
        bottom_candidates = []  # Near court (Player 1 only: Y_feet > net_y_pixel)

        for p in detected_players:
            _, feet_y = self._get_box_feet(p)
            if feet_y <= net_y_pixel:
                top_candidates.append(p)
            else:
                bottom_candidates.append(p)

        # ----------------------------------------------------------------------
        # Select Player 1 (Strictly Near / Bottom Court: Y > Y_net)
        # ----------------------------------------------------------------------
        p1 = self._match_best_candidate(bottom_candidates, self.prev_p1_box)
        if p1 is not None:
            self.prev_p1_box = p1

        # ----------------------------------------------------------------------
        # Select Player 2 (Strictly Far / Top Court: Y <= Y_net)
        # ----------------------------------------------------------------------
        p2 = self._match_best_candidate(top_candidates, self.prev_p2_box)
        if p2 is not None:
            self.prev_p2_box = p2

        return {
            'player_1': p1,
            'player_2': p2
        }

    def reset(self):
        """
        Resets tracking state on scene cuts or camera transitions to prevent cross-scene identity drift.
        """
        self.prev_p1_box = None
        self.prev_p2_box = None
