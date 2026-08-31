import os

# Dedicated Directories
INPUT_DIR = os.path.join('data', 'inputs')
OUTPUT_DIR = os.path.join('data', 'outputs')
ANALYSIS_DIR = os.path.join('data', 'analysis')

# File Paths
DEFAULT_VIDEO_NAME = 'input.mp4'
DEFAULT_OUTPUT_NAME = 'output.mp4'

VIDEO_PATH = os.path.join(INPUT_DIR, DEFAULT_VIDEO_NAME)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, DEFAULT_OUTPUT_NAME)

MATCH_SUMMARY_PATH = os.path.join(ANALYSIS_DIR, 'match_summary.json')
MATCH_REPORT_PATH = os.path.join(ANALYSIS_DIR, 'match_report.html')
HEATMAP_P1_PATH = os.path.join(ANALYSIS_DIR, 'heatmap_player_1.png')
HEATMAP_P2_PATH = os.path.join(ANALYSIS_DIR, 'heatmap_player_2.png')

# Candidate weights to check in order (custom weights first, fallback to standard YOLOv8)
MODEL_CANDIDATES = ['best_tennis.pt', 'tennis_ball_detector.pt', 'yolov8x.pt', 'yolov8m.pt', 'yolov8n.pt']

# COCO Class IDs
COCO_PERSON_CLASS_ID = 0
COCO_BALL_CLASS_ID = 32

# High-Resolution & Ball Inference Parameters
BALL_IMGSZ = 1280               # High resolution to prevent sub-pixel downsampling of small motion-blurred ball
BALL_CONF_THRESHOLD = 0.08      # Highly sensitive ball confidence threshold
PERSON_CONF_THRESHOLD = 0.40    # Filter out background crowd & staff false positives

# Spatial Filtering / ROI Settings
ENABLE_ROI_FILTER = True
COURT_ROI_NORMALIZED = [
    (0.08, 0.12),  # Top-left of court zone
    (0.92, 0.12),  # Top-right of court zone
    (0.98, 0.98),  # Bottom-right
    (0.02, 0.98),  # Bottom-left
]

# Tracking Safety Checks & Limits
MAX_MISSING_FRAMES = 8          # Max consecutive frames without detection before resetting track
MAX_BALL_SPEED_PIXELS = 220     # Max plausible displacement in pixels between adjacent frames
SCENE_CUT_THRESHOLD = 0.60      # Color histogram correlation threshold below which a scene cut is triggered

# Visualization Settings
TRAJECTORY_MAX_POINTS = 25      # Number of historical points to display for the ball trail


def ensure_directories():
    """Ensures all dedicated data and analysis directories exist."""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
