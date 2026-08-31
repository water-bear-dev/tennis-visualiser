import cv2
from src.config import VIDEO_PATH
from src.detectors.yolo_detector import load_detector, extract_detections
from src.trackers.ball_interpolator import interpolate_ball_positions
from src.visualizers.video_annotator import render_annotated_video
from src.utils.roi_utils import get_roi_polygon_pixels
from src.court_detector.court_line_detector import CourtLineDetector
from src.mini_court.mini_court import MiniCourt
from src.analysis.shot_detector import ShotDetector
from src.analysis.player_analytics import PlayerAnalytics


def main():
    # 1. Initialize detector model
    model = load_detector()

    # 2. Open input video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video '{VIDEO_PATH}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Court Keypoints & Geometry Setup (Phase 2)
    court_detector = CourtLineDetector()
    success, sample_frame = cap.read()
    if not success:
        print("Error: Could not read sample frame from video.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Computing Court Geometry & Keypoints...")
    court_keypoints = court_detector.detect_keypoints(sample_frame, width, height)
    
    # Initialize 2D Mini-Court and calculate Homography
    mini_court = MiniCourt(canvas_width=220, canvas_height=420, margin=16)
    mini_court.compute_homography(court_keypoints)
    print("Homography Matrix successfully computed!")

    # 4. Calculate pixel-level ROI bounds
    roi_polygon_pixels = get_roi_polygon_pixels(width, height)

    # 5. Pipeline Execution
    # Pass 1: Extract detections & persistent player tracks
    frame_detections = extract_detections(cap, model, roi_polygon_pixels)
    
    # Tracker: Interpolate missing ball frames with safety guardrails
    interpolated_balls = interpolate_ball_positions(frame_detections)

    # Phase 3 Analytics: Shot events, speed (km/h), bounce/calls & rally counters
    shot_detector = ShotDetector(mini_court=mini_court, fps=fps)
    telemetry_per_frame = shot_detector.analyze_rally_and_shots(frame_detections, interpolated_balls)
    
    # Phase 4 Analytics: Player running speed, cumulative distance covered, & 2D heatmaps
    player_analytics = PlayerAnalytics(mini_court=mini_court, fps=fps)
    kinetics_per_frame, p1_heatmap, p2_heatmap = player_analytics.analyze_player_kinetics(frame_detections)

    # Pass 2: Render visualizations, trajectory trails, 2D Mini-Court radar & Expanded Kinetics HUD
    render_annotated_video(
        cap=cap,
        frame_detections=frame_detections,
        interpolated_balls=interpolated_balls,
        roi_polygon_pixels=roi_polygon_pixels,
        mini_court=mini_court,
        telemetry_per_frame=telemetry_per_frame,
        kinetics_per_frame=kinetics_per_frame,
        width=width,
        height=height,
        fps=fps
    )

    # 6. Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
