import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.trackers.player_tracker import PlayerTracker
from src.analysis.shot_detector import ShotDetector
from src.mini_court.mini_court import MiniCourt


def test_player_tracker():
    print("\n--- Testing PlayerTracker ---")
    tracker = PlayerTracker(net_y_ratio=0.50, max_track_dist=140.0)
    frame_h = 1080

    # Frame 1: 1 top player, 1 bottom player, 1 referee/crowd
    detections_f1 = [
        (400, 200, 460, 350, 0.90),  # Top player (Player 2: feet_y=350 <= 540)
        (500, 700, 580, 950, 0.88),  # Bottom player (Player 1: feet_y=950 > 540)
        (100, 800, 150, 900, 0.45),  # Crowd near bottom (lower conf)
    ]
    tracked_f1 = tracker.track_players(detections_f1, frame_h)
    assert tracked_f1['player_1'] == detections_f1[1], f"P1 mismatch: {tracked_f1['player_1']}"
    assert tracked_f1['player_2'] == detections_f1[0], f"P2 mismatch: {tracked_f1['player_2']}"
    print("✓ Frame 1: Correct partitioning and confidence selection.")

    # Frame 2: Slight movement
    detections_f2 = [
        (405, 205, 465, 355, 0.91),  # Player 2 moved slightly
        (508, 705, 588, 955, 0.89),  # Player 1 moved slightly
        (900, 200, 950, 300, 0.70),  # Top spectator far right
    ]
    tracked_f2 = tracker.track_players(detections_f2, frame_h)
    assert tracked_f2['player_1'] == detections_f2[1], f"P1 tracking mismatch: {tracked_f2['player_1']}"
    assert tracked_f2['player_2'] == detections_f2[0], f"P2 tracking mismatch: {tracked_f2['player_2']}"
    print("✓ Frame 2: Correct centroid/IoU temporal association.")

    # Frame 3: Single person on court (e.g. P2 occluded)
    detections_f3 = [
        (515, 710, 595, 960, 0.89),  # Only Player 1 detected
    ]
    tracked_f3 = tracker.track_players(detections_f3, frame_h)
    assert tracked_f3['player_1'] == detections_f3[0]
    assert tracked_f3['player_2'] is None, "Player 2 should be None when no top candidates exist."
    print("✓ Frame 3: Single player per half enforced (no duplicate assignment).")


def test_shot_detector():
    print("\n--- Testing ShotDetector ---")
    from src.court_detector.court_line_detector import CourtLineDetector
    court_detector = CourtLineDetector()
    dummy_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    court_kp = court_detector.detect_keypoints(dummy_frame, 1920, 1080)

    mini_court = MiniCourt(canvas_width=220, canvas_height=420, margin=16)
    mini_court.compute_homography(court_kp)
    detector = ShotDetector(mini_court=mini_court, fps=30)

    total_frames = 200
    frame_detections = []
    interpolated_balls = []

    # Simulate 4 rally hits between P1 (y_px=950) and P2 (y_px=360)
    # Hits at frames: 20 (P1), 60 (P2), 100 (P1), 140 (P2)
    # Net is at y_px = 561
    for f in range(total_frames):
        # Players centered around x=960
        p1_box = (910, 800, 1010, 950, 0.90)  # Near court (feet at 960, 950)
        p2_box = (910, 220, 1010, 360, 0.90)  # Far court (feet at 960, 360)
        frame_detections.append({
            'players': {'player_1': p1_box, 'player_2': p2_box},
            'scene_cut': False
        })

        # Ball trajectory in pixels
        if 0 <= f < 20:
            # Ball moving towards P1 (y increasing from 560 to 940)
            y_px = 560 + (f / 20.0) * 380
        elif 20 <= f < 60:
            # P1 hits at f=20, ball moving towards P2 (y decreasing from 940 to 360)
            t = (f - 20) / 40.0
            y_px = 940 - t * 580
        elif 60 <= f < 100:
            # P2 hits at f=60, ball moving towards P1 (y increasing from 360 to 940)
            t = (f - 60) / 40.0
            y_px = 360 + t * 580
        elif 100 <= f < 140:
            # P1 hits at f=100, ball moving towards P2 (y decreasing from 940 to 360)
            t = (f - 100) / 40.0
            y_px = 940 - t * 580
        elif 140 <= f < 180:
            # P2 hits at f=140, ball moving towards P1 (y increasing from 360 to 940)
            t = (f - 140) / 40.0
            y_px = 360 + t * 580
        else:
            y_px = 940

        # Add simulated micro-jitter (+- 2 pixels)
        jitter = ((f % 3) - 1) * 2.0
        interpolated_balls.append((960.0 + jitter, y_px + jitter))

    telemetry = detector.analyze_rally_and_shots(frame_detections, interpolated_balls)
    final_rally_count = telemetry[-1]['rally_count']
    hits = [i for i, t in enumerate(telemetry) if t['is_hit']]

    print(f"Total Hit Events Detected: {len(hits)} at frames {hits}")
    print(f"Final Rally Count: {final_rally_count}")
    assert len(hits) == 4, f"Expected 4 validated hits, got {len(hits)}: {hits}"
    assert final_rally_count == 4, f"Expected final rally count 4, got {final_rally_count}"
    print("✓ ShotDetector State Machine: Exactly 4 hits registered with alternating net crossing and jitter rejection!")


if __name__ == '__main__':
    test_player_tracker()
    test_shot_detector()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")
