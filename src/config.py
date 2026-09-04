"""
src/config.py
Configuration module containing all tunable thresholds, file paths, model candidate
hierarchies, spatial ROI boundaries, and tracking guardrails for the tennis analytics pipeline.
"""

import os

# ==============================================================================
# Dedicated Project Directories
# ==============================================================================
INPUT_DIR = os.path.join('data', 'inputs')       # Directory storing raw input videos
OUTPUT_DIR = os.path.join('data', 'outputs')     # Directory storing rendered annotated videos
ANALYSIS_DIR = os.path.join('data', 'analysis')  # Directory storing reports, JSONs, and heatmaps

# ==============================================================================
# Default File Paths
# ==============================================================================
DEFAULT_VIDEO_NAME = 'input.mp4'
DEFAULT_OUTPUT_NAME = 'output.mp4'

VIDEO_PATH = os.path.join(INPUT_DIR, DEFAULT_VIDEO_NAME)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, DEFAULT_OUTPUT_NAME)

MATCH_SUMMARY_PATH = os.path.join(ANALYSIS_DIR, 'match_summary.json')
MATCH_REPORT_PATH = os.path.join(ANALYSIS_DIR, 'match_report.html')
HEATMAP_P1_PATH = os.path.join(ANALYSIS_DIR, 'heatmap_player_1.png')
HEATMAP_P2_PATH = os.path.join(ANALYSIS_DIR, 'heatmap_player_2.png')

# ==============================================================================
# Model Checkpoint Hierarchy (Checked sequentially in priority order)
# ==============================================================================
MODEL_CANDIDATES = [
    'best_tennis.pt',             # 1. Custom fine-tuned tennis ball detector (Highest priority)
    'tennis_ball_detector.pt',    # 2. Pre-trained custom ball checkpoint
    'yolov8x.pt',                 # 3. High-capacity general COCO model
    'yolov8m.pt',                 # 4. Medium-capacity general COCO model
    'yolov8n.pt'                  # 5. Lightweight nano fallback
]

# ==============================================================================
# COCO Dataset Class Identifiers
# ==============================================================================
COCO_PERSON_CLASS_ID = 0  # Class 0: 'person'
COCO_BALL_CLASS_ID = 32   # Class 32: 'sports ball'

# ==============================================================================
# High-Resolution Inference & Detection Thresholds
# ==============================================================================
BALL_IMGSZ = 1280               # High resolution (1280px) to prevent sub-pixel downsampling of micro ball
BALL_CONF_THRESHOLD = 0.08      # Sensitive confidence floor to catch fast motion-blurred balls
PERSON_CONF_THRESHOLD = 0.40    # High threshold to filter out background audience, line judges, and ball boys

# ==============================================================================
# Spatial Filtering & Court Region of Interest (ROI)
# ==============================================================================
ENABLE_ROI_FILTER = True
COURT_ROI_NORMALIZED = [
    (0.08, 0.12),  # Top-left of active court zone (normalized x, y)
    (0.92, 0.12),  # Top-right of active court zone
    (0.98, 0.98),  # Bottom-right
    (0.02, 0.98),  # Bottom-left
]

# ==============================================================================
# Tracking Safety Guardrails & Physical Limits
# ==============================================================================
MAX_MISSING_FRAMES = 8          # Maximum consecutive frames without detection before resetting ball track
MAX_BALL_SPEED_PIXELS = 220     # Plausible max displacement in pixels between adjacent frames (jump filter)
SCENE_CUT_THRESHOLD = 0.60      # Color histogram correlation threshold below which a scene cut is triggered

# ==============================================================================
# Visualization Settings
# ==============================================================================
TRAJECTORY_MAX_POINTS = 25      # Number of historical trajectory points to display for the ball trail


def ensure_directories():
    """
    Creates necessary input, output, and analysis folders if they do not already exist.
    """
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
