import cv2
import numpy as np
from src.config import SCENE_CUT_THRESHOLD


def compute_frame_histogram(frame: np.ndarray) -> np.ndarray:
    """Computes a 2D HSV (Hue, Saturation) color histogram for a frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Calculate 2D histogram for H (30 bins) and S (32 bins)
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def detect_scene_cut(prev_hist: np.ndarray, curr_frame: np.ndarray, threshold: float = SCENE_CUT_THRESHOLD) -> tuple[bool, np.ndarray]:
    """
    Compares the current frame's histogram with the previous frame.
    Returns (is_scene_cut, current_hist).
    """
    curr_hist = compute_frame_histogram(curr_frame)
    if prev_hist is None:
        return False, curr_hist

    # Correlation comparison: 1.0 is exact match, lower values indicate large visual difference / cut
    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    is_cut = bool(correlation < threshold)
    return is_cut, curr_hist
