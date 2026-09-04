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

    def __init__(self, net_y_ratio: float = 0.50, max_track_dist: float = 120.0):
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
        return ((x1 + x2) / 2.0, y2)

    def _distance(self, p1: tuple[float, float], p2: tuple[float, float]) -> float:
        """Calculates Euclidean distance between two 2D points."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def track_players(self, detected_players: list[tuple], frame_height: int) -> dict[str, tuple | None]:
        """
        Filters raw detected people into the two primary active tennis players.

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

        # Separate player candidate detections across the net horizon
        top_candidates = []
        bottom_candidates = []

        for p in detected_players:
            feet_x, feet_y = self._get_box_feet(p)
            if feet_y < net_y_pixel:
                top_candidates.append(p)     # Far court candidate
            else:
                bottom_candidates.append(p)  # Near court candidate

        # ----------------------------------------------------------------------
        # Select Player 1 (Near / Bottom Court)
        # ----------------------------------------------------------------------
        p1 = None
        if bottom_candidates:
            if self.prev_p1_box is not None:
                prev_center = self._get_box_center(self.prev_p1_box)
                # Sort candidates by distance to previous position
                bottom_candidates.sort(key=lambda b: self._distance(self._get_box_center(b), prev_center))
                # Choose closest candidate if within tracking threshold; otherwise fallback to highest confidence
                if self._distance(self._get_box_center(bottom_candidates[0]), prev_center) <= self.max_track_dist * 2:
                    p1 = bottom_candidates[0]
                else:
                    bottom_candidates.sort(key=lambda b: b[4], reverse=True)
                    p1 = bottom_candidates[0]
            else:
                # First frame: select detection with highest confidence score
                bottom_candidates.sort(key=lambda b: b[4], reverse=True)
                p1 = bottom_candidates[0]
            self.prev_p1_box = p1

        # ----------------------------------------------------------------------
        # Select Player 2 (Far / Top Court)
        # ----------------------------------------------------------------------
        p2 = None
        if top_candidates:
            if self.prev_p2_box is not None:
                prev_center = self._get_box_center(self.prev_p2_box)
                # Sort candidates by distance to previous position
                top_candidates.sort(key=lambda b: self._distance(self._get_box_center(b), prev_center))
                if self._distance(self._get_box_center(top_candidates[0]), prev_center) <= self.max_track_dist * 2:
                    p2 = top_candidates[0]
                else:
                    top_candidates.sort(key=lambda b: b[4], reverse=True)
                    p2 = top_candidates[0]
            else:
                # First frame: select detection with highest confidence score
                top_candidates.sort(key=lambda b: b[4], reverse=True)
                p2 = top_candidates[0]
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
