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
        self.mini_court = mini_court

    def classify_stroke(self, hitter: str, ball_meter: tuple, player_box: tuple, 
                        is_start_of_rally: bool, ball_speed_kmh: float) -> str:
        """
        Classifies stroke type given player and ball coordinates at impact frame.
        """
        if ball_meter is None or player_box is None:
            return "SHOT"

        x1, y1, x2, y2, conf = player_box
        player_cx_pixel = (x1 + x2) / 2.0
        
        x_m, y_m = ball_meter
        net_y = MiniCourt.COURT_LENGTH_M / 2.0  # 11.885m

        # 1. Check for Serve (First shot of a rally hit near baseline with high speed)
        if is_start_of_rally:
            if (hitter == "Player 2" and y_m <= 6.0) or (hitter == "Player 1" and y_m >= 18.0):
                if ball_speed_kmh >= 90.0:
                    return "SERVE"

        # 2. Check for Volley / Net Attack (Hit close to the net)
        # Net area: Y between 8.5m and 15.2m
        if 8.5 <= y_m <= 15.2:
            return "VOLLEY"

        # 3. Groundstrokes: Forehand vs. Backhand (Assuming right-handed default)
        # In camera view:
        # For Player 1 (facing away from camera / near court):
        # Ball to the right of player center -> Forehand
        # Ball to the left of player center -> Backhand
        # For Player 2 (facing towards camera / far court):
        # Ball to the right of player center in frame is player's left side -> Backhand
        # Ball to the left of player center in frame is player's right side -> Forehand
        ball_canvas = self.mini_court.project_point_to_meters(((x1 + x2)/2.0, (y1 + y2)/2.0))
        if ball_canvas:
            delta_x = x_m - ball_canvas[0]
        else:
            delta_x = 0.0

        if hitter == "Player 1":
            return "FOREHAND" if delta_x >= 0 else "BACKHAND"
        else:
            return "FOREHAND" if delta_x <= 0 else "BACKHAND"
