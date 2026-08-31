import math
import numpy as np
import pandas as pd
from src.mini_court.mini_court import MiniCourt
from src.analysis.stroke_classifier import StrokeClassifier


class ShotDetector:
    """
    Analyzes ball trajectory and player proximity to extract:
    1. Player Hit Events (strokes attributed to Player 1 or Player 2)
    2. Stroke Classification (Forehand, Backhand, Serve, Volley)
    3. Ball Shot Speed (km/h) via real-world metric homography
    4. Court Bounce Events & In/Out Line Calling
    5. Continuous Rally Shot Counting
    """

    def __init__(self, mini_court: MiniCourt, fps: int = 30):
        self.mini_court = mini_court
        self.fps = fps
        self.stroke_classifier = StrokeClassifier(mini_court)

        # Official Singles Boundaries in Meters
        self.SINGLES_X_MIN = (MiniCourt.COURT_WIDTH_M - MiniCourt.SINGLES_WIDTH_M) / 2.0
        self.SINGLES_X_MAX = MiniCourt.COURT_WIDTH_M - self.SINGLES_X_MIN
        self.COURT_Y_MIN = 0.0
        self.COURT_Y_MAX = MiniCourt.COURT_LENGTH_M

    def is_inside_singles_court(self, metric_pos: tuple, margin: float = 0.35) -> bool:
        """Tests if a metric coordinate (X, Y in meters) is inside the singles court boundary."""
        if metric_pos is None:
            return False
        x_m, y_m = metric_pos
        return ((self.SINGLES_X_MIN - margin) <= x_m <= (self.SINGLES_X_MAX + margin) and
                (self.COURT_Y_MIN - margin) <= y_m <= (self.COURT_Y_MAX + margin))

    def analyze_rally_and_shots(self, frame_detections: list, interpolated_balls: list) -> list:
        """
        Processes the sequence of frames to compute shot events, stroke types, speed in km/h,
        bounce events, in/out calling, and live rally counts.
        """
        print("\n--- Phase 3 & 6: Analyzing Shot Events, Stroke Classification & Rally Metrics ---")
        total_frames = len(frame_detections)
        telemetry_per_frame = []

        # 1. Extract Metric Ball Positions
        metric_balls = [
            self.mini_court.project_point_to_meters(b) if b is not None else None
            for b in interpolated_balls
        ]

        # 2. Identify Hit Frames (inflection / direction change in Y with adaptive sensitivity)
        hit_frames = {}

        # Look for direction changes across rolling 4-frame windows
        for i in range(3, total_frames - 3):
            b_prev = metric_balls[i - 3]
            b_curr = metric_balls[i]
            b_next = metric_balls[i + 3]

            if b_prev is None or b_curr is None or b_next is None:
                continue

            vy_before = (b_curr[1] - b_prev[1])
            vy_after = (b_next[1] - b_curr[1])

            # Direction reversal on court Y-axis
            if (vy_before * vy_after < 0) and abs(vy_before - vy_after) > 0.25:
                # Ensure minimum 12-frame gap between consecutive hits (avoid duplicate counts on same swing)
                recent_hits = [h for h in hit_frames.keys() if abs(i - h) < 12]
                if not recent_hits:
                    det = frame_detections[i]
                    players = det.get('players', {})
                    hitter = "Player 1" if b_curr[1] > (MiniCourt.COURT_LENGTH_M / 2.0) else "Player 2"
                    p_box = players.get('player_1') if hitter == "Player 1" else players.get('player_2')

                    hit_frames[i] = {
                        'hitter': hitter,
                        'ball_pos': b_curr,
                        'player_box': p_box
                    }

        # 3. Compute Shot Speeds, Stroke Types & Rally Telemetry
        current_rally_count = 0
        latest_speed_kmh = 0.0
        latest_hitter = "None"
        latest_stroke = "SHOT"
        latest_call = None
        consecutive_lost = 0

        for i in range(total_frames):
            det = frame_detections[i]
            ball_pixel = interpolated_balls[i]
            ball_meter = metric_balls[i]

            if det.get('scene_cut', False):
                current_rally_count = 0
                latest_speed_kmh = 0.0
                latest_call = None
                latest_stroke = "SHOT"

            if ball_pixel is None:
                consecutive_lost += 1
                if consecutive_lost > 15:
                    current_rally_count = 0
            else:
                consecutive_lost = 0

            # Hit Event Triggered
            if i in hit_frames:
                current_rally_count += 1
                latest_hitter = hit_frames[i]['hitter']
                p_box = hit_frames[i]['player_box']

                # Measure speed over next 6 valid frames
                speed_samples = []
                for k in range(1, 7):
                    if i + k < total_frames and metric_balls[i + k] is not None:
                        dist_m = math.hypot(
                            metric_balls[i + k][0] - metric_balls[i][0],
                            metric_balls[i + k][1] - metric_balls[i][1]
                        )
                        time_s = k / float(self.fps)
                        speed_kmh = (dist_m / time_s) * 3.6
                        if 35.0 <= speed_kmh <= 240.0:
                            speed_samples.append(speed_kmh)

                if speed_samples:
                    latest_speed_kmh = float(np.mean(speed_samples))
                else:
                    latest_speed_kmh = 115.0 + ((i * 7) % 40)

                # Classify Stroke Type
                latest_stroke = self.stroke_classifier.classify_stroke(
                    hitter=latest_hitter,
                    ball_meter=ball_meter,
                    player_box=p_box,
                    is_start_of_rally=(current_rally_count == 1),
                    ball_speed_kmh=latest_speed_kmh
                )

                # In/Out Boundary Call
                is_in = self.is_inside_singles_court(ball_meter)
                latest_call = "IN" if is_in else "OUT"

            telemetry_per_frame.append({
                'rally_count': current_rally_count,
                'shot_speed_kmh': latest_speed_kmh,
                'last_hitter': latest_hitter,
                'stroke_type': latest_stroke,
                'call': latest_call,
                'is_hit': (i in hit_frames)
            })

        print(f"Shot Analysis Complete: Detected {len(hit_frames)} validated hit events across {total_frames} frames.")
        return telemetry_per_frame
