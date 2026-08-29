# src configuration and parameters
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

# Interpolation & Trajectory Settings
MAX_INTERPOLATION_GAP = 15     # Max consecutive missing frames to fill with linear interpolation
TRAJECTORY_MAX_POINTS = 25     # Number of historical points to display for the ball trail
