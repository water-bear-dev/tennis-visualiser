"""
src/analysis/player_analytics.py
Player Kinetics and Spatial Movement Analytics Engine.
Calculates rolling instantaneous running speeds (km/h), cumulative running distance (meters),
and renders high-resolution 2D Gaussian density court heatmaps for tactical court positioning analysis.
"""

import math
import cv2
import numpy as np
from src.mini_court.mini_court import MiniCourt
from src.config import HEATMAP_P1_PATH, HEATMAP_P2_PATH


class PlayerAnalytics:
    """
    Computes kinetic player metrics:
    1. Instantaneous running speed (km/h) across rolling frame windows
    2. Cumulative distance covered (meters) across points
    3. 2D spatial court density heatmaps saved to data/analysis/
    """

    def __init__(self, mini_court: MiniCourt, fps: int = 30, speed_window: int = 5):
        """
        Initializes the player analytics engine.

        Args:
            mini_court (MiniCourt): Homography engine for projecting pixel feet to metric court coordinates.
            fps (int): Video frame rate.
            speed_window (int): Rolling frame window size for robust speed differentiation (default 5 frames).
        """
        self.mini_court = mini_court
        self.fps = fps
        self.speed_window = speed_window

    def analyze_player_kinetics(self, frame_detections: list[dict]) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """
        Processes frame detections to compute per-frame player speed and cumulative distance,
        and generates 2D court density heatmaps for Player 1 and Player 2.

        Args:
            frame_detections (list[dict]): Sequential list of frame detection records from Pass 1.

        Returns:
            tuple[list, np.ndarray, np.ndarray]: A tuple containing:
                - kinetics_per_frame (list[dict]): Per-frame player kinetics records.
                - p1_heatmap (np.ndarray): Rendered heatmap image for Player 1.
                - p2_heatmap (np.ndarray): Rendered heatmap image for Player 2.
        """
        print("\n--- Phase 4: Calculating Player Kinetics (Speed, Distance, Heatmaps) ---")
        total_frames = len(frame_detections)

        # ----------------------------------------------------------------------
        # 1. Project Player Feet Coordinates to Metric Meters & Radar Canvas
        # ----------------------------------------------------------------------
        p1_metric_coords = []
        p2_metric_coords = []
        p1_canvas_coords = []
        p2_canvas_coords = []

        for det in frame_detections:
            players = det.get('players', {})
            p1 = players.get('player_1')
            p2 = players.get('player_2')

            # Player 1 (Near court): bottom-center of bounding box
            if p1 is not None:
                feet_p1 = ((p1[0] + p1[2]) / 2.0, p1[3])
                p1_metric_coords.append(self.mini_court.project_point_to_meters(feet_p1))
                p1_canvas_coords.append(self.mini_court.project_point(feet_p1))
            else:
                p1_metric_coords.append(None)
                p1_canvas_coords.append(None)

            # Player 2 (Far court): bottom-center of bounding box
            if p2 is not None:
                feet_p2 = ((p2[0] + p2[2]) / 2.0, p2[3])
                p2_metric_coords.append(self.mini_court.project_point_to_meters(feet_p2))
                p2_canvas_coords.append(self.mini_court.project_point(feet_p2))
            else:
                p2_metric_coords.append(None)
                p2_canvas_coords.append(None)

        # ----------------------------------------------------------------------
        # 2. Compute Instantaneous Running Speed & Cumulative Distance (Meters)
        # ----------------------------------------------------------------------
        kinetics_per_frame = []
        cum_dist_p1 = 0.0
        cum_dist_p2 = 0.0

        for i in range(total_frames):
            det = frame_detections[i]
            is_cut = det.get('scene_cut', False)

            speed_p1 = 0.0
            speed_p2 = 0.0

            # ---------------------------
            # Player 1 Speed & Distance
            # ---------------------------
            if i >= self.speed_window and not is_cut:
                p_prev = p1_metric_coords[i - self.speed_window]
                p_curr = p1_metric_coords[i]
                if p_prev is not None and p_curr is not None:
                    d_m = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                    time_s = self.speed_window / float(self.fps)
                    calc_speed = (d_m / time_s) * 3.6  # Convert m/s to km/h
                    # Apply athletic sanity boundary: 0.5 km/h to 32 km/h sprint peak
                    if 0.5 <= calc_speed <= 32.0:
                        speed_p1 = calc_speed

            if i > 0 and not is_cut:
                prev_1 = p1_metric_coords[i - 1]
                curr_1 = p1_metric_coords[i]
                if prev_1 is not None and curr_1 is not None:
                    step_d1 = math.hypot(curr_1[0] - prev_1[0], curr_1[1] - prev_1[1])
                    # Filter frame jitter anomalies (>1.5m in single frame)
                    if step_d1 < 1.5:
                        cum_dist_p1 += step_d1

            # ---------------------------
            # Player 2 Speed & Distance
            # ---------------------------
            if i >= self.speed_window and not is_cut:
                p_prev = p2_metric_coords[i - self.speed_window]
                p_curr = p2_metric_coords[i]
                if p_prev is not None and p_curr is not None:
                    d_m = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
                    time_s = self.speed_window / float(self.fps)
                    calc_speed = (d_m / time_s) * 3.6
                    if 0.5 <= calc_speed <= 32.0:
                        speed_p2 = calc_speed

            if i > 0 and not is_cut:
                prev_2 = p2_metric_coords[i - 1]
                curr_2 = p2_metric_coords[i]
                if prev_2 is not None and curr_2 is not None:
                    step_d2 = math.hypot(curr_2[0] - prev_2[0], curr_2[1] - prev_2[1])
                    if step_d2 < 1.5:
                        cum_dist_p2 += step_d2

            kinetics_per_frame.append({
                'p1_speed_kmh': speed_p1,
                'p1_dist_m': cum_dist_p1,
                'p2_speed_kmh': speed_p2,
                'p2_dist_m': cum_dist_p2
            })

        # ----------------------------------------------------------------------
        # 3. Generate 2D Positional Gaussian Heatmaps
        # ----------------------------------------------------------------------
        p1_heatmap = self._generate_heatmap(p1_canvas_coords, "Player 1 Heatmap (Near Court)")
        p2_heatmap = self._generate_heatmap(p2_canvas_coords, "Player 2 Heatmap (Far Court)")

        # Export generated heatmap PNGs
        cv2.imwrite(HEATMAP_P1_PATH, p1_heatmap)
        cv2.imwrite(HEATMAP_P2_PATH, p2_heatmap)
        print(f"Kinetics Complete: Player 1 ran {cum_dist_p1:.1f}m | Player 2 ran {cum_dist_p2:.1f}m")
        print(f"Exported: '{HEATMAP_P1_PATH}' and '{HEATMAP_P2_PATH}'")

        return kinetics_per_frame, p1_heatmap, p2_heatmap

    def _generate_heatmap(self, canvas_coords: list[tuple | None], title: str) -> np.ndarray:
        """
        Renders a 2D Gaussian density heatmap overlaid on the mini-court template.

        Args:
            canvas_coords (list): Sequence of 2D mini-court canvas pixel coordinates.
            title (str): Overlay heading title for the exported heatmap image.

        Returns:
            np.ndarray: Blended BGR heatmap image array.
        """
        w = self.mini_court.canvas_width
        h = self.mini_court.canvas_height

        density = np.zeros((h, w), dtype=np.float32)

        # Accumulate visit frequency per canvas pixel
        for pt in canvas_coords:
            if pt is not None:
                x, y = pt
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(density, (x, y), 15, 1.0, -1)

        # Apply Gaussian blur kernel to model continuous movement distribution
        density = cv2.GaussianBlur(density, (31, 31), 0)
        max_val = np.max(density)
        if max_val > 0:
            density = density / max_val

        density_uint8 = (density * 255).astype(np.uint8)
        color_heatmap = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)

        # Render court geometry background
        base_canvas = np.full((h, w, 3), (35, 30, 25), dtype=np.uint8)
        self.mini_court.draw_court_lines(base_canvas)

        # Alpha blend heat intensity over the base court diagram
        mask = (density > 0.05).astype(np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask])
        blended = np.where(mask_3ch > 0, cv2.addWeighted(color_heatmap, 0.65, base_canvas, 0.35, 0), base_canvas)

        cv2.putText(blended, title, (14, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
        return blended
