"""
main.py
Single-Match Tennis Analytics Pipeline Entrypoint.
Coordinates Pass 1 object detection, court keypoint extraction, homography computation,
ball interpolation, biomechanical stroke analysis, player kinetics, report generation,
and Pass 2 broadcast video composition.
"""

import os
import shutil
import cv2
from src.config import (
    VIDEO_PATH,
    OUTPUT_PATH,
    ensure_directories
)
from src.detectors.yolo_detector import load_detector, extract_detections
from src.trackers.ball_interpolator import interpolate_ball_positions
from src.visualizers.video_annotator import render_annotated_video
from src.utils.roi_utils import get_roi_polygon_pixels
from src.court_detector.court_line_detector import CourtLineDetector
from src.mini_court.mini_court import MiniCourt
from src.analysis.shot_detector import ShotDetector
from src.analysis.player_analytics import PlayerAnalytics
from src.analysis.report_generator import MatchReportGenerator


def resolve_video_input() -> str:
    """
    Resolves the input match video path from data/inputs/input.mp4 or root input.mp4 fallback.

    Returns:
        str: Absolute or relative path to the verified input video file.
    """
    if os.path.exists(VIDEO_PATH):
        return VIDEO_PATH
    if os.path.exists('input.mp4'):
        # Auto-copy root input.mp4 to data/inputs/ for organized folder structure
        shutil.copy('input.mp4', VIDEO_PATH)
        return VIDEO_PATH
    return VIDEO_PATH


def main():
    """
    Main pipeline controller orchestrating:
    - Step 0: Directory structure verification
    - Step 1: Model loading & Video stream initialization
    - Step 2: Court keypoint detection & Homography matrix computation
    - Step 3: Pass 1 feature extraction & bounded trajectory interpolation
    - Step 4: Biomechanical shot detection, speeds, and player kinetics
    - Step 5: Report generation (JSON & standalone HTML)
    - Step 6: Pass 2 visual rendering & video export
    """
    # 0. Ensure dedicated directory structure exists
    ensure_directories()
    video_input_file = resolve_video_input()

    # 1. Initialize detector model
    model = load_detector()

    # 2. Open input video
    cap = cv2.VideoCapture(video_input_file)
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_input_file}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Court Keypoints & Geometry Setup
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
    # Pass 1: Extract detections with specialized high-res ball detector & persistent player tracks
    frame_detections = extract_detections(cap, model, roi_polygon_pixels)
    
    # Tracker: Interpolate missing ball frames with safety guardrails
    interpolated_balls = interpolate_ball_positions(frame_detections)

    # Phase 3 & 6 Analytics: Shot events, stroke types, speed (km/h), bounce/calls & rally counters
    shot_detector = ShotDetector(mini_court=mini_court, fps=fps)
    telemetry_per_frame = shot_detector.analyze_rally_and_shots(frame_detections, interpolated_balls)
    
    # Phase 4 Analytics: Player running speed, cumulative distance covered, & 2D heatmaps (saved to data/analysis/)
    player_analytics = PlayerAnalytics(mini_court=mini_court, fps=fps)
    kinetics_per_frame, p1_heatmap, p2_heatmap = player_analytics.analyze_player_kinetics(frame_detections)

    # Phase 5 Delivery: Generate JSON match summary & standalone HTML report in data/analysis/
    report_generator = MatchReportGenerator(fps=fps)
    report_generator.generate_report(frame_detections, telemetry_per_frame, kinetics_per_frame)

    # Pass 2: Render visualizations, trajectory trails, 2D Mini-Court radar & Expanded Kinetics HUD to data/outputs/
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
