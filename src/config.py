# Candidate weights to check in order (custom weights first, fallback to standard YOLOv8)
MODEL_CANDIDATES = ['best_tennis.pt', 'tennis_ball.pt', 'yolov8n.pt']

VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output.mp4'

# COCO Class IDs
COCO_PERSON_CLASS_ID = 0
COCO_BALL_CLASS_ID = 32

# Confidence thresholds
BALL_CONF_THRESHOLD = 0.12     # Lower threshold to capture fast/motion-blurred tennis ball
PERSON_CONF_THRESHOLD = 0.40   # Higher threshold to filter out low-confidence background false positives

# Spatial Filtering / ROI Settings
ENABLE_ROI_FILTER = True
COURT_ROI_NORMALIZED = [
    (0.08, 0.12),  # Top-left of court zone
    (0.92, 0.12),  # Top-right of court zone
    (0.98, 0.98),  # Bottom-right
    (0.02, 0.98),  # Bottom-left
]

# Tracking Safety Checks & Limits
MAX_MISSING_FRAMES = 7          # Max consecutive frames without detection before resetting track
MAX_BALL_SPEED_PIXELS = 180     # Max plausible displacement in pixels between adjacent frames
SCENE_CUT_THRESHOLD = 0.60      # Color histogram correlation threshold below which a scene cut is triggered

# Visualization Settings
TRAJECTORY_MAX_POINTS = 25      # Number of historical points to display for the ball trail
