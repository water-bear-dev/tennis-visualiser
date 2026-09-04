"""
src/visualizers/video_annotator.py
Pass 2 Video Rendering and Broadcast Compositor.
Renders player bounding boxes with identity badges, high-resolution ball tracking markers,
decaying alpha motion trails, 2D top-down mini-court radar overlays, and telemetry HUD cards.
"""

from collections import deque
import cv2
import numpy as np
from src.config import (
    OUTPUT_PATH,
    ENABLE_ROI_FILTER,
    TRAJECTORY_MAX_POINTS,
    MAX_MISSING_FRAMES
)
from src.mini_court.mini_court import MiniCourt


def draw_player_box(frame: np.ndarray, player_data: tuple | None, label: str, color: tuple):
    """
    Draws a styled bounding box and text badge for a tracked tennis player.

    Args:
        frame (np.ndarray): Video frame image array.
        player_data (tuple, optional): (x1, y1, x2, y2, conf) tuple or None.
        label (str): Player label (e.g. "Player 1" or "Player 2").
        color (tuple): BGR color tuple for the bounding box.
    """
    if player_data is None:
        return
    x1, y1, x2, y2, conf = player_data
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, lineType=cv2.LINE_AA)
    
    # Render label badge above the player bounding box
    text = f"{label} ({conf:.2f})"
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    badge_y1 = max(y1 - th - 10, 5)
    badge_y2 = badge_y1 + th + 6
    cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 10, badge_y2), color, -1)
    cv2.putText(frame, text, (x1 + 5, badge_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, lineType=cv2.LINE_AA)


def draw_broadcast_hud(frame: np.ndarray, telemetry: dict, kinetics: dict, offset_x: int = 30, offset_y: int = 30):
    """
    Renders an expanded broadcast-grade telemetry HUD card in the top-left corner
    with match stats, ball speed, stroke type classification, and player physical exertion metrics.

    Args:
        frame (np.ndarray): Target video frame.
        telemetry (dict): Current frame shot telemetry (rally count, stroke type, speed, line call).
        kinetics (dict): Current frame player kinetics (sprint speeds, cumulative distance).
        offset_x (int): Horizontal pixel offset from left edge.
        offset_y (int): Vertical pixel offset from top edge.
    """
    card_w = 360
    card_h = 165
    x1 = offset_x
    y1 = offset_y
    x2 = x1 + card_w
    y2 = y1 + card_h

    # Semi-transparent dark glassmorphism card background
    sub_img = frame[y1:y2, x1:x2]
    dark_card = np.full(sub_img.shape, (25, 22, 18), dtype=np.uint8)
    blended = cv2.addWeighted(dark_card, 0.88, sub_img, 0.12, 0)
    frame[y1:y2, x1:x2] = blended

    cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), 1, lineType=cv2.LINE_AA)
    cv2.line(frame, (x1, y1), (x2, y1), (0, 255, 255), 3, lineType=cv2.LINE_AA)

    # 1. Title Header
    cv2.putText(frame, "AI MATCH & KINETICS HUD", (x1 + 14, y1 + 22), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, lineType=cv2.LINE_AA)

    # 2. In/Out Call Badge (if recent)
    call = telemetry.get('call')
    if call:
        call_color = (0, 220, 0) if call == "IN" else (0, 0, 240)
        cv2.rectangle(frame, (x2 - 58, y1 + 10), (x2 - 14, y1 + 34), call_color, -1)
        cv2.putText(frame, call, (x2 - 50, y1 + 27), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, lineType=cv2.LINE_AA)

    # 3. Live Rally Counter Badge
    rally_count = telemetry.get('rally_count', 0)
    rally_text = f"RALLY: {rally_count} SHOTS"
    cv2.putText(frame, rally_text, (x1 + 14, y1 + 52), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    # 4. Ball Speed, Stroke Type & Player Attribution
    speed = telemetry.get('shot_speed_kmh', 0.0)
    hitter = telemetry.get('last_hitter', 'None')
    stroke = telemetry.get('stroke_type', 'SHOT')
    if speed > 0:
        speed_text = f"{stroke}: {speed:.0f} km/h ({hitter})"
        cv2.putText(frame, speed_text, (x1 + 14, y1 + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, lineType=cv2.LINE_AA)
    else:
        cv2.putText(frame, "Ball Speed: -- km/h", (x1 + 14, y1 + 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, lineType=cv2.LINE_AA)

    # Divider Line
    cv2.line(frame, (x1 + 14, y1 + 94), (x2 - 14, y1 + 94), (55, 55, 55), 1)

    # 5. Player Kinetic Exertion (Sprint Speeds & Cumulative Distance)
    p1_spd = kinetics.get('p1_speed_kmh', 0.0)
    p1_dist = kinetics.get('p1_dist_m', 0.0)
    p1_text = f"P1: {p1_spd:.1f} km/h | Dist: {p1_dist:.1f}m"
    cv2.putText(frame, p1_text, (x1 + 14, y1 + 119), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 180, 50), 1, lineType=cv2.LINE_AA)

    p2_spd = kinetics.get('p2_speed_kmh', 0.0)
    p2_dist = kinetics.get('p2_dist_m', 0.0)
    p2_text = f"P2: {p2_spd:.1f} km/h | Dist: {p2_dist:.1f}m"
    cv2.putText(frame, p2_text, (x1 + 14, y1 + 146), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (50, 180, 255), 1, lineType=cv2.LINE_AA)


def overlay_radar_on_frame(frame: np.ndarray, radar_img: np.ndarray, offset_x: int = 30, offset_y: int = 30):
    """
    Overlays the 2D Mini-Court radar with a border on the top-right corner.

    Args:
        frame (np.ndarray): Primary video frame.
        radar_img (np.ndarray): Rendered 2D mini-court radar canvas.
        offset_x (int): Margin offset from right edge.
        offset_y (int): Margin offset from top edge.
    """
    rh, rw = radar_img.shape[:2]
    fh, fw = frame.shape[:2]

    x1 = fw - rw - offset_x
    y1 = offset_y
    x2 = x1 + rw
    y2 = y1 + rh

    if x1 < 0 or y1 < 0 or x2 > fw or y2 > fh:
        return

    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (20, 20, 20), 2)
    roi = frame[y1:y2, x1:x2]
    blended = cv2.addWeighted(radar_img, 0.92, roi, 0.08, 0)
    frame[y1:y2, x1:x2] = blended


def render_annotated_video(cap: cv2.VideoCapture, frame_detections: list[dict], interpolated_balls: list[tuple | None], 
                           roi_polygon_pixels: np.ndarray, mini_court: MiniCourt, telemetry_per_frame: list[dict],
                           kinetics_per_frame: list[dict], width: int, height: int, fps: int,
                           output_path: str = OUTPUT_PATH):
    """
    Pass 2: Renders court ROI polygon, persistent player boxes, ball marker,
    dynamic alpha motion trails, 2D Mini-Court radar, and the broadcast kinetics telemetry HUD.

    Args:
        cap (cv2.VideoCapture): Open video capture stream.
        frame_detections (list[dict]): Pass 1 frame detection records.
        interpolated_balls (list): Smoothed ball pixel coordinates.
        roi_polygon_pixels (np.ndarray): Court boundary polygon vertices.
        mini_court (MiniCourt): MiniCourt homography and radar engine.
        telemetry_per_frame (list[dict]): Shot speeds, rally counts, and line calls.
        kinetics_per_frame (list[dict]): Player running speeds and cumulative meters.
        width (int): Video frame width.
        height (int): Video frame height.
        fps (int): Video frame rate.
        output_path (str): Destination file path for rendered annotated video (default OUTPUT_PATH).
    """
    print(f"\n--- Pass 2: Rendering Annotations to '{output_path}' ---")
    
    # Rewind video capture to frame 0 for Pass 2 rendering
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trajectory = deque(maxlen=TRAJECTORY_MAX_POINTS)
    frame_idx = 0
    missing_counter = 0

    COLOR_P1 = (255, 160, 0)   # Blue/Cyan
    COLOR_P2 = (0, 140, 255)   # Deep Orange

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(frame_detections):
            break

        detection_data = frame_detections[frame_idx]
        telemetry = telemetry_per_frame[frame_idx] if frame_idx < len(telemetry_per_frame) else {}
        kinetics = kinetics_per_frame[frame_idx] if frame_idx < len(kinetics_per_frame) else {}

        # ----------------------------------------------------------------------
        # 1. Reset trajectory trail on scene cuts
        # ----------------------------------------------------------------------
        if detection_data.get('scene_cut', False):
            trajectory.clear()
            missing_counter = 0

        # ----------------------------------------------------------------------
        # 2. Draw subtle court ROI polygon overlay
        # ----------------------------------------------------------------------
        if ENABLE_ROI_FILTER and roi_polygon_pixels is not None:
            cv2.polylines(frame, [roi_polygon_pixels], isClosed=True, color=(100, 255, 100), thickness=1, lineType=cv2.LINE_AA)

        # ----------------------------------------------------------------------
        # 3. Draw Persistent Players (Player 1 & Player 2)
        # ----------------------------------------------------------------------
        players = detection_data.get('players', {})
        p1_data = players.get('player_1')
        p2_data = players.get('player_2')
        draw_player_box(frame, p1_data, "Player 1", COLOR_P1)
        draw_player_box(frame, p2_data, "Player 2", COLOR_P2)

        # ----------------------------------------------------------------------
        # 4. Draw Ball & Decaying Alpha Motion Trail
        # ----------------------------------------------------------------------
        ball_pos = interpolated_balls[frame_idx]
        if ball_pos is not None:
            missing_counter = 0
            bx, by = int(round(ball_pos[0])), int(round(ball_pos[1]))
            trajectory.append((bx, by))

            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                alpha = float(i) / len(trajectory)
                thickness = max(1, int(3 * alpha))
                color = (0, int(255 * alpha), int(255 * (1 - 0.5 * alpha)))
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness, lineType=cv2.LINE_AA)

            is_interpolated = detection_data['ball'] is None
            ball_color = (0, 165, 255) if is_interpolated else (0, 255, 255)
            cv2.circle(frame, (bx, by), 7, ball_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            
            ball_label = "Ball (est)" if is_interpolated else "Ball"
            cv2.putText(frame, ball_label, (bx + 10, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ball_color, 1)
        else:
            missing_counter += 1
            if missing_counter >= MAX_MISSING_FRAMES:
                trajectory.clear()

        # ----------------------------------------------------------------------
        # 5. Render 2D Mini-Court Radar Overlay
        # ----------------------------------------------------------------------
        p1_feet = ((p1_data[0] + p1_data[2]) / 2.0, p1_data[3]) if p1_data else None
        p2_feet = ((p2_data[0] + p2_data[2]) / 2.0, p2_data[3]) if p2_data else None

        radar_img = mini_court.render_radar(
            p1_pos=p1_feet,
            p2_pos=p2_feet,
            ball_pos=ball_pos,
            ball_trajectory=list(trajectory)
        )
        overlay_radar_on_frame(frame, radar_img, offset_x=30, offset_y=30)

        # ----------------------------------------------------------------------
        # 6. Render Broadcast Telemetry HUD Card
        # ----------------------------------------------------------------------
        draw_broadcast_hud(frame, telemetry, kinetics, offset_x=30, offset_y=30)

        # Write rendered frame to output video
        out.write(frame)
        frame_idx += 1

        if frame_idx % 60 == 0:
            print(f"Pass 2: Rendered {frame_idx}/{len(frame_detections)} frames...")

    out.release()
    print(f"\nFinished! Output successfully saved to: '{OUTPUT_PATH}'")
