"""
src/utils/scene_utils.py
Scene cut and camera transition detector using 2D HSV color histogram correlation.
Prevents trajectory jump errors and reset tracking state across video cuts or broadcast replays.
"""

import cv2
import numpy as np
from src.config import SCENE_CUT_THRESHOLD


def compute_frame_histogram(frame: np.ndarray) -> np.ndarray:
    """
    Computes a normalized 2D HSV (Hue, Saturation) color histogram for a video frame.

    Args:
        frame (np.ndarray): Input BGR video frame image array.

    Returns:
        np.ndarray: Normalized 2D histogram matrix (30 Hue bins x 32 Saturation bins).
    """
    # Convert BGR frame to HSV color space to decouple illumination from chromaticity
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate 2D histogram for H (30 bins: [0, 180]) and S (32 bins: [0, 256])
    hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    
    # Normalize histogram to scale range [0, 1] for stable statistical comparison
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def detect_scene_cut(prev_hist: np.ndarray, curr_frame: np.ndarray, threshold: float = SCENE_CUT_THRESHOLD) -> tuple[bool, np.ndarray]:
    """
    Compares the current frame's histogram with the previous frame to detect abrupt scene transitions.

    Args:
        prev_hist (np.ndarray): Normalized 2D HSV histogram from the preceding frame.
        curr_frame (np.ndarray): Current BGR video frame image array.
        threshold (float): Correlation threshold below which a transition is flagged (default 0.60).

    Returns:
        tuple[bool, np.ndarray]: A tuple containing:
            - is_scene_cut (bool): True if an abrupt camera transition or cut occurred.
            - current_hist (np.ndarray): The computed histogram for the current frame.
    """
    curr_hist = compute_frame_histogram(curr_frame)
    if prev_hist is None:
        # First frame of the video sequence - no scene cut possible
        return False, curr_hist

    # Correlation metric: 1.0 indicates identical color distribution, lower values indicate visual cuts
    correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
    is_cut = bool(correlation < threshold)
    return is_cut, curr_hist
