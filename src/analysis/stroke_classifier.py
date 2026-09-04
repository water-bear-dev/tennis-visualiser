"""
src/analysis/stroke_classifier.py
Stroke Biomechanics Classification Module.
Classifies tennis strokes into SERVE, FOREHAND, BACKHAND, or VOLLEY based on
court spatial positioning, net proximity, lateral impact delta, and rally state.
"""

import math
from src.mini_court.mini_court import MiniCourt


class StrokeClassifier:
    """
    Classifies tennis shot types:
    - SERVE
    - FOREHAND
    - BACKHAND
    - VOLLEY / SMASH
    Based on court spatial positioning, net proximity, and ball-to-player impact orientation.
    """

    def __init__(self, mini_court: MiniCourt):
        """
        Initializes the stroke classifier.

        Args:
            mini_court (MiniCourt): MiniCourt homography instance for metric projections.
        """
        self.mini_court = mini_court

    def classify_stroke(self, hitter: str, ball_meter: tuple[float, float], player_box: tuple, 
                        is_start_of_rally: bool, ball_speed_kmh: float) -> str:
        """
        Classifies stroke type given player and ball coordinates at the impact frame.

        Args:
            hitter (str): "Player 1" (near court) or "Player 2" (far court).
            ball_meter (tuple): Metric (X, Y in meters) coordinates of ball impact.
            player_box (tuple): Player bounding box (x1, y1, x2, y2, conf).
            is_start_of_rally (bool): True if this is the opening shot of a rally.
            ball_speed_kmh (float): Estimated ball exit velocity in km/h.

        Returns:
            str: Classified stroke type ("SERVE", "VOLLEY", "FOREHAND", "BACKHAND", or "SHOT").
        """
        if ball_meter is None or player_box is None:
            return "SHOT"

        x1, y1, x2, y2, conf = player_box
        player_cx_pixel = (x1 + x2) / 2.0
        
        x_m, y_m = ball_meter
        net_y = MiniCourt.COURT_LENGTH_M / 2.0  # Net line at Y = 11.885m

        # ----------------------------------------------------------------------
        # 1. Check for Serve (First shot of a rally initiated near the baseline)
        # ----------------------------------------------------------------------
        if is_start_of_rally:
            # Player 2 baseline (Y <= 6.0m) or Player 1 baseline (Y >= 18.0m)
            if (hitter == "Player 2" and y_m <= 6.0) or (hitter == "Player 1" and y_m >= 18.0):
                if ball_speed_kmh >= 90.0:
                    return "SERVE"

        # ----------------------------------------------------------------------
        # 2. Check for Volley / Net Attack (Impact within transition / forecourt zone)
        # ----------------------------------------------------------------------
        # Net area: Y between 8.5m and 15.2m (approx 3.3m around the net line)
        if 8.5 <= y_m <= 15.2:
            return "VOLLEY"

        # ----------------------------------------------------------------------
        # 3. Groundstrokes: Forehand vs. Backhand (Assuming right-handed default)
        # ----------------------------------------------------------------------
        # Lateral offset analysis:
        # For Player 1 (facing away from camera / near court):
        #   - Ball to the right of player center -> Forehand
        #   - Ball to the left of player center -> Backhand
        # For Player 2 (facing towards camera / far court):
        #   - Ball to the right of player in camera frame is player's left side -> Backhand
        #   - Ball to the left of player in camera frame is player's right side -> Forehand
        player_metric = self.mini_court.project_point_to_meters(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
        if player_metric:
            delta_x = x_m - player_metric[0]
        else:
            delta_x = 0.0

        if hitter == "Player 1":
            return "FOREHAND" if delta_x >= 0 else "BACKHAND"
        else:
            return "FOREHAND" if delta_x <= 0 else "BACKHAND"
