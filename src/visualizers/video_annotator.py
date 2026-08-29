from collections import deque
import cv2
from src.config import (
    OUTPUT_PATH,
    ENABLE_ROI_FILTER,
    TRAJECTORY_MAX_POINTS,
    MAX_MISSING_FRAMES
)


def render_annotated_video(cap: cv2.VideoCapture, frame_detections: list, interpolated_balls: list, 
                           roi_polygon_pixels, width: int, height: int, fps: int):
    """
    Pass 2: Re-reads the video and renders court ROI, player bounding boxes, ball markers, and trajectory trails.
    Clears tracking queues on scene cuts and large tracking gaps.
    """
    print("\n--- Pass 2: Rendering Annotations & Ball Trajectory Trail ---")
    
    # Reset video to the beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    trajectory = deque(maxlen=TRAJECTORY_MAX_POINTS)
    frame_idx = 0
    missing_counter = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success or frame_idx >= len(frame_detections):
            break

        detection_data = frame_detections[frame_idx]

        # 1. Reset trajectory on Scene Cut
        if detection_data.get('scene_cut', False):
            trajectory.clear()
            missing_counter = 0

        # 2. Draw Subtle Court ROI overlay
        if ENABLE_ROI_FILTER and roi_polygon_pixels is not None:
            cv2.polylines(frame, [roi_polygon_pixels], isClosed=True, color=(100, 255, 100), thickness=1, lineType=cv2.LINE_AA)

        # 3. Draw Players
        for x1, y1, x2, y2, conf in detection_data['players']:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
            label = f"Player {conf:.2f}"
            cv2.putText(frame, label, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 120, 0), 2)

        # 4. Draw Ball & Trajectory
        ball_pos = interpolated_balls[frame_idx]
        if ball_pos is not None:
            missing_counter = 0
            bx, by = int(round(ball_pos[0])), int(round(ball_pos[1]))
            trajectory.append((bx, by))

            # Draw ball trailing trajectory
            for i in range(1, len(trajectory)):
                if trajectory[i - 1] is None or trajectory[i] is None:
                    continue
                alpha = float(i) / len(trajectory)
                thickness = max(1, int(3 * alpha))
                color = (0, int(255 * alpha), int(255 * (1 - 0.5 * alpha)))  # Fade from cyan/yellow
                cv2.line(frame, trajectory[i - 1], trajectory[i], color, thickness, lineType=cv2.LINE_AA)

            # Draw Ball Marker (outer circle + solid center dot)
            is_interpolated = detection_data['ball'] is None
            ball_color = (0, 165, 255) if is_interpolated else (0, 255, 255)  # Orange if interpolated, yellow if raw
            cv2.circle(frame, (bx, by), 7, ball_color, 2, lineType=cv2.LINE_AA)
            cv2.circle(frame, (bx, by), 3, (0, 255, 0), -1, lineType=cv2.LINE_AA)
            
            ball_label = "Ball (est)" if is_interpolated else "Ball"
            cv2.putText(frame, ball_label, (bx + 10, by - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, ball_color, 1)
        else:
            missing_counter += 1
            if missing_counter >= MAX_MISSING_FRAMES:
                trajectory.clear()

        # Write annotated frame
        out.write(frame)
        frame_idx += 1

        if frame_idx % 60 == 0:
            print(f"Pass 2: Rendered {frame_idx}/{len(frame_detections)} frames...")

    out.release()
    print(f"\nFinished! Output successfully saved to: '{OUTPUT_PATH}'")
