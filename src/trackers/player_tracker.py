import math
import numpy as np


class PlayerTracker:
    """
    Tracks and isolates the two active tennis players (near court vs far court)
    across video frames, assigning persistent IDs: 'Player 1' (bottom) and 'Player 2' (top).
    """

    def __init__(self, net_y_ratio: float = 0.50, max_track_dist: float = 120.0):
        """
        :param net_y_ratio: Normalized Y-coordinate dividing far court (top) and near court (bottom).
        :param max_track_dist: Maximum pixel distance for associating player across adjacent frames.
        """
        self.net_y_ratio = net_y_ratio
        self.max_track_dist = max_track_dist
        self.prev_p1_box = None  # Near court player (bottom)
        self.prev_p2_box = None  # Far court player (top)

    def _get_box_center(self, box):
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def _get_box_feet(self, box):
        x1, y1, x2, y2 = box[:4]
        return ((x1 + x2) / 2.0, y2)

    def _distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def track_players(self, detected_players: list, frame_height: int) -> dict:
        """
        Filters raw detected people into the two active players.
        Returns dict: {'player_1': (x1, y1, x2, y2, conf) or None, 'player_2': (x1, y1, x2, y2, conf) or None}
        """
        if not detected_players:
            return {'player_1': None, 'player_2': None}

        net_y_pixel = frame_height * self.net_y_ratio

        # Separate candidates into far court (Y < net) and near court (Y >= net)
        top_candidates = []
        bottom_candidates = []

        for p in detected_players:
            feet_x, feet_y = self._get_box_feet(p)
            if feet_y < net_y_pixel:
                top_candidates.append(p)
            else:
                bottom_candidates.append(p)

        # Select Player 1 (Near / Bottom Court)
        p1 = None
        if bottom_candidates:
            if self.prev_p1_box is not None:
                prev_center = self._get_box_center(self.prev_p1_box)
                bottom_candidates.sort(key=lambda b: self._distance(self._get_box_center(b), prev_center))
                # Choose closest if within reasonable distance, else highest confidence
                if self._distance(self._get_box_center(bottom_candidates[0]), prev_center) <= self.max_track_dist * 2:
                    p1 = bottom_candidates[0]
                else:
                    bottom_candidates.sort(key=lambda b: b[4], reverse=True)
                    p1 = bottom_candidates[0]
            else:
                # Pick highest confidence / lowest Y (closest to bottom)
                bottom_candidates.sort(key=lambda b: b[4], reverse=True)
                p1 = bottom_candidates[0]
            self.prev_p1_box = p1

        # Select Player 2 (Far / Top Court)
        p2 = None
        if top_candidates:
            if self.prev_p2_box is not None:
                prev_center = self._get_box_center(self.prev_p2_box)
                top_candidates.sort(key=lambda b: self._distance(self._get_box_center(b), prev_center))
                if self._distance(self._get_box_center(top_candidates[0]), prev_center) <= self.max_track_dist * 2:
                    p2 = top_candidates[0]
                else:
                    top_candidates.sort(key=lambda b: b[4], reverse=True)
                    p2 = top_candidates[0]
            else:
                top_candidates.sort(key=lambda b: b[4], reverse=True)
                p2 = top_candidates[0]
            self.prev_p2_box = p2

        return {
            'player_1': p1,
            'player_2': p2
        }

    def reset(self):
        """Resets tracking state on scene cuts."""
        self.prev_p1_box = None
        self.prev_p2_box = None
