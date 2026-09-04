"""
src/mini_court/mini_court.py
2D Mini-Court Radar and Perspective Homography Engine.
Computes perspective transformation matrices mapping broadcast video camera pixels
to top-down 2D canvas coordinates and official ITF metric meter measurements (23.77m x 10.97m).
"""

import cv2
import numpy as np


class MiniCourt:
    """
    Renders a 2D top-down mini-court radar overlay and computes homography matrices
    to project video pixel coordinates of players and ball onto the 2D mini-court
    and real-world metric dimensions (in meters).
    """

    # Official ITF Court Dimensions in meters: 23.77m long, 10.97m wide (doubles), 8.23m (singles)
    COURT_LENGTH_M = 23.77
    COURT_WIDTH_M = 10.97
    SINGLES_WIDTH_M = 8.23
    SERVICE_LINE_DIST_M = 6.40

    def __init__(self, canvas_width: int = 220, canvas_height: int = 420, margin: int = 16):
        """
        Initializes the MiniCourt engine.

        Args:
            canvas_width (int): Pixel width of the rendered mini-court radar HUD card.
            canvas_height (int): Pixel height of the rendered mini-court radar HUD card.
            margin (int): Boundary padding around the painted court lines.
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.margin = margin

        # Usable court drawing dimensions within the padded radar canvas
        self.court_w = self.canvas_width - 2 * self.margin
        self.court_h = self.canvas_height - 2 * self.margin

        # Mini-court metric keypoints in canvas pixels
        self.mini_keypoints = self._generate_mini_court_keypoints()
        self.homography_matrix = None          # Video Pixels -> Radar Canvas Pixels
        self.metric_homography_matrix = None   # Video Pixels -> Real-World Meters

    def _generate_mini_court_keypoints(self) -> np.ndarray:
        """
        Generates the 14 standard keypoint coordinates on the 2D radar canvas.

        Returns:
            np.ndarray: Array of shape (14, 2) in canvas pixel space.
        """
        m_x = self.margin
        m_y = self.margin
        cw = self.court_w
        ch = self.court_h

        # Doubles alley width ratio: (10.97 - 8.23) / 2 / 10.97 ~= 0.125
        alley_ratio = 0.125
        s_left = m_x + alley_ratio * cw
        s_right = m_x + (1.0 - alley_ratio) * cw
        center_x = m_x + cw / 2.0

        # Baseline to service line: (23.77/2 - 6.4) / 23.77 ~= 0.2307
        srv_top_y = m_y + (5.485 / 23.77) * ch
        net_y = m_y + ch / 2.0
        srv_bot_y = m_y + (1.0 - (5.485 / 23.77)) * ch

        return np.array([
            [m_x, m_y],                # 0: Top-left double
            [s_left, m_y],             # 1: Top-left single
            [s_right, m_y],            # 2: Top-right single
            [m_x + cw, m_y],           # 3: Top-right double
            [s_left, srv_top_y],       # 4: Top service left
            [center_x, srv_top_y],     # 5: Top service center (T)
            [s_right, srv_top_y],      # 6: Top service right
            [m_x, net_y],              # 7: Net left
            [center_x, net_y],         # 8: Net center
            [m_x + cw, net_y],         # 9: Net right
            [s_left, srv_bot_y],       # 10: Bottom service left
            [center_x, srv_bot_y],     # 11: Bottom service center (T)
            [s_right, srv_bot_y],      # 12: Bottom service right
            [m_x, m_y + ch],           # 13: Bottom-left double
        ], dtype=np.float32)

    def _generate_real_world_metric_keypoints(self) -> np.ndarray:
        """
        Generates official ITF keypoint coordinates in metric meters:
        Origin (0,0) is top-left doubles corner. Court spans [0, 10.97] in X and [0, 23.77] in Y.

        Returns:
            np.ndarray: Array of shape (14, 2) in meter units.
        """
        W = self.COURT_WIDTH_M
        L = self.COURT_LENGTH_M
        s_alley = (W - self.SINGLES_WIDTH_M) / 2.0
        s_right = W - s_alley
        c_x = W / 2.0
        srv_top = 5.485
        net_y = L / 2.0
        srv_bot = L - 5.485

        return np.array([
            [0.0, 0.0],          # 0: Top-left double
            [s_alley, 0.0],      # 1: Top-left single
            [s_right, 0.0],      # 2: Top-right single
            [W, 0.0],            # 3: Top-right double
            [s_alley, srv_top],  # 4: Top service left
            [c_x, srv_top],      # 5: Top service center (T)
            [s_right, srv_top],  # 6: Top service right
            [0.0, net_y],        # 7: Net left
            [c_x, net_y],        # 8: Net center
            [W, net_y],          # 9: Net right
            [s_alley, srv_bot],  # 10: Bottom service left
            [c_x, srv_bot],      # 11: Bottom service center (T)
            [s_right, srv_bot],  # 12: Bottom service right
            [0.0, L],            # 13: Bottom-left double
        ], dtype=np.float32)

    def compute_homography(self, video_keypoints: np.ndarray):
        """
        Computes homography transformation matrices for both canvas pixels and real-world meters using RANSAC.

        Args:
            video_keypoints (np.ndarray): 14 court keypoints in camera pixel coordinates.
        """
        src_pts = video_keypoints[:14]
        
        # 1. Video pixels -> 2D Mini-Court Canvas Pixels
        dst_canvas = self.mini_keypoints[:14]
        H_canvas, _ = cv2.findHomography(src_pts, dst_canvas, cv2.RANSAC, 5.0)
        self.homography_matrix = H_canvas

        # 2. Video pixels -> Real-World Metric Coordinates (Meters)
        dst_metric = self._generate_real_world_metric_keypoints()[:14]
        H_metric, _ = cv2.findHomography(src_pts, dst_metric, cv2.RANSAC, 5.0)
        self.metric_homography_matrix = H_metric

    def project_point(self, point: tuple[float, float]) -> tuple[int, int] | None:
        """
        Projects a (x, y) video pixel coordinate onto the 2D mini-court radar canvas.

        Args:
            point (tuple): Camera (x, y) coordinate.

        Returns:
            tuple[int, int] | None: Transformed (x', y') coordinate on the mini-court canvas, or None.
        """
        if self.homography_matrix is None or point is None:
            return None
        pt_in = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        pt_out = cv2.perspectiveTransform(pt_in, self.homography_matrix)
        x_out, y_out = pt_out[0][0]

        if 0 <= x_out <= self.canvas_width and 0 <= y_out <= self.canvas_height:
            return (int(round(x_out)), int(round(y_out)))
        return None

    def project_point_to_meters(self, point: tuple[float, float]) -> tuple[float, float] | None:
        """
        Projects a (x, y) video pixel coordinate into real-world court meters (X, Y).

        Args:
            point (tuple): Camera (x, y) coordinate.

        Returns:
            tuple[float, float] | None: Transformed coordinate in meters [0-10.97m, 0-23.77m].
        """
        if self.metric_homography_matrix is None or point is None:
            return None
        pt_in = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
        pt_out = cv2.perspectiveTransform(pt_in, self.metric_homography_matrix)
        x_m, y_m = pt_out[0][0]
        return (float(x_m), float(y_m))

    def draw_court_lines(self, canvas: np.ndarray):
        """
        Draws 2D tennis court boundary lines, service boxes, and net on the canvas.

        Args:
            canvas (np.ndarray): 3-channel BGR radar canvas image.
        """
        color = (255, 255, 255)
        m_x = self.margin
        m_y = self.margin
        cw = self.court_w
        ch = self.court_h

        # 1. Outer Court Rectangle
        cv2.rectangle(canvas, (m_x, m_y), (m_x + cw, m_y + ch), color, 2)

        # 2. Singles Sidelines
        kps = self.mini_keypoints
        cv2.line(canvas, (int(kps[1][0]), m_y), (int(kps[1][0]), m_y + ch), color, 1)
        cv2.line(canvas, (int(kps[2][0]), m_y), (int(kps[2][0]), m_y + ch), color, 1)

        # 3. Service Lines
        cv2.line(canvas, (int(kps[4][0]), int(kps[4][1])), (int(kps[6][0]), int(kps[6][1])), color, 1)
        cv2.line(canvas, (int(kps[10][0]), int(kps[10][1])), (int(kps[12][0]), int(kps[12][1])), color, 1)

        # 4. Center Service Line
        cv2.line(canvas, (int(kps[5][0]), int(kps[5][1])), (int(kps[11][0]), int(kps[11][1])), color, 1)

        # 5. Net Line (Highlighted in yellow)
        cv2.line(canvas, (m_x - 4, int(kps[7][1])), (m_x + cw + 4, int(kps[9][1])), (0, 255, 255), 2)

    def render_radar(self, p1_pos: tuple = None, p2_pos: tuple = None, ball_pos: tuple = None, ball_trajectory: list = None) -> np.ndarray:
        """
        Renders the complete 2D radar image with player markers, ball dot, and motion trails.

        Args:
            p1_pos (tuple, optional): Feet coordinate of Player 1.
            p2_pos (tuple, optional): Feet coordinate of Player 2.
            ball_pos (tuple, optional): Current ball coordinate.
            ball_trajectory (list, optional): Historical list of ball coordinates for trail drawing.

        Returns:
            np.ndarray: Rendered BGR image array of the 2D radar.
        """
        canvas = np.full((self.canvas_height, self.canvas_width, 3), (35, 30, 25), dtype=np.uint8)
        cv2.rectangle(canvas, (self.margin, self.margin), 
                      (self.canvas_width - self.margin, self.canvas_height - self.margin), 
                      (70, 50, 30), -1)

        self.draw_court_lines(canvas)

        # 1. Draw Ball Trajectory on Mini-Court
        if ball_trajectory:
            mini_trail = [self.project_point(p) for p in ball_trajectory if p is not None]
            mini_trail = [p for p in mini_trail if p is not None]
            for i in range(1, len(mini_trail)):
                cv2.line(canvas, mini_trail[i - 1], mini_trail[i], (0, 255, 255), 2, lineType=cv2.LINE_AA)

        # 2. Draw Ball Marker
        if ball_pos is not None:
            mini_ball = self.project_point(ball_pos)
            if mini_ball:
                cv2.circle(canvas, mini_ball, 5, (0, 255, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, mini_ball, 8, (0, 200, 255), 1, lineType=cv2.LINE_AA)

        # 3. Draw Players (P1: Orange/Gold, P2: Blue/Cyan)
        if p1_pos is not None:
            mini_p1 = self.project_point(p1_pos)
            if mini_p1:
                cv2.circle(canvas, mini_p1, 7, (255, 160, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, mini_p1, 10, (255, 200, 50), 2, lineType=cv2.LINE_AA)
                cv2.putText(canvas, "P1", (mini_p1[0] + 10, mini_p1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        if p2_pos is not None:
            mini_p2 = self.project_point(p2_pos)
            if mini_p2:
                cv2.circle(canvas, mini_p2, 7, (0, 140, 255), -1, lineType=cv2.LINE_AA)
                cv2.circle(canvas, mini_p2, 10, (50, 180, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(canvas, "P2", (mini_p2[0] + 10, mini_p2[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(canvas, "2D MINI-COURT", (self.margin, self.margin - 6), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, lineType=cv2.LINE_AA)

        return canvas
