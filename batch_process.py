import glob
import json
import os
import cv2
from src.config import (
    INPUT_DIR,
    OUTPUT_DIR,
    ANALYSIS_DIR,
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


def process_single_video(video_path: str, model) -> dict:
    """Processes a single tennis match video through the full analytics pipeline."""
    vid_stem = os.path.splitext(os.path.basename(video_path))[0]
    out_video_path = os.path.join(OUTPUT_DIR, f"{vid_stem}_annotated.mp4")
    summary_json_path = os.path.join(ANALYSIS_DIR, f"{vid_stem}_summary.json")

    print(f"\n=======================================================")
    print(f"🎬 Processing Match: {video_path}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return {}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    court_detector = CourtLineDetector()
    success, sample_frame = cap.read()
    if not success:
        print(f"Error: Could not read sample frame from {video_path}")
        return {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    court_keypoints = court_detector.detect_keypoints(sample_frame, width, height)
    mini_court = MiniCourt(canvas_width=220, canvas_height=420, margin=16)
    mini_court.compute_homography(court_keypoints)

    roi_polygon_pixels = get_roi_polygon_pixels(width, height)

    # Pass 1: Detection Extraction
    frame_detections = extract_detections(cap, model, roi_polygon_pixels)
    interpolated_balls = interpolate_ball_positions(frame_detections)

    # Analytics
    shot_detector = ShotDetector(mini_court=mini_court, fps=fps)
    telemetry_per_frame = shot_detector.analyze_rally_and_shots(frame_detections, interpolated_balls)

    player_analytics = PlayerAnalytics(mini_court=mini_court, fps=fps)
    kinetics_per_frame, p1_hm, p2_hm = player_analytics.analyze_player_kinetics(frame_detections)

    report_generator = MatchReportGenerator(fps=fps)
    report_generator.generate_report(frame_detections, telemetry_per_frame, kinetics_per_frame)

    # Move/save summary under match-specific name
    if os.path.exists(os.path.join(ANALYSIS_DIR, 'match_summary.json')):
        with open(os.path.join(ANALYSIS_DIR, 'match_summary.json'), 'r') as f:
            match_stats = json.load(f)
        with open(summary_json_path, 'w') as f:
            json.dump(match_stats, f, indent=4)
    else:
        match_stats = {}

    cap.release()
    cv2.destroyAllWindows()
    return match_stats


def run_batch():
    """Processes all videos in data/inputs/ and produces an aggregated tournament summary."""
    ensure_directories()
    video_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.mp4')) + glob.glob('*.mp4'))
    video_files = list(dict.fromkeys(video_files))

    if not video_files:
        print(f"No video files found in '{INPUT_DIR}' or root directory.")
        return

    print(f"\n🎾 Starting Batch Match Processor on {len(video_files)} video(s)...")
    model = load_detector()
    tournament_records = []

    for vid in video_files:
        stats = process_single_video(vid, model)
        if stats:
            stats['video_file'] = os.path.basename(vid)
            tournament_records.append(stats)

    # Compile Tournament Summary
    tourney_path = os.path.join(ANALYSIS_DIR, 'tournament_summary.json')
    with open(tourney_path, 'w') as f:
        json.dump({
            "total_matches_processed": len(tournament_records),
            "matches": tournament_records
        }, f, indent=4)

    print(f"\n=======================================================")
    print(f"🏆 Batch Processing Complete! Processed {len(tournament_records)} matches.")
    print(f"Tournament Summary saved to: '{tourney_path}'")
    print(f"=======================================================\n")


if __name__ == '__main__':
    run_batch()
