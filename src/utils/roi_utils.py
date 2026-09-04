"""
src/utils/roi_utils.py
Spatial geometry utility functions for Region of Interest (ROI) filtering.
Prevents false positive detections from spectators, line judges, and background noise.
"""

import cv2
import numpy as np
from src.config import ENABLE_ROI_FILTER, COURT_ROI_NORMALIZED


def get_roi_polygon_pixels(width: int, height: int) -> np.ndarray:
    """
    Converts normalized ROI coordinates (0.0 - 1.0) into absolute frame pixel coordinates.

    Args:
        width (int): Frame width in pixels.
        height (int): Frame height in pixels.

    Returns:
        np.ndarray: Array of integer pixel coordinates shape (N, 2) defining the polygon vertices.
    """
    return np.array([
        [int(x * width), int(y * height)] for x, y in COURT_ROI_NORMALIZED
    ], dtype=np.int32)


def is_inside_roi(point: tuple[float, float] | list[float], roi_polygon_pixels: np.ndarray) -> bool:
    """
    Evaluates whether an (x, y) point falls inside the designated court polygon using OpenCV pointPolygonTest.

    Args:
        point (tuple or list): Coordinate (x, y) of the candidate object centroid.
        roi_polygon_pixels (np.ndarray): Pixel coordinate vertices of the ROI polygon.

    Returns:
        bool: True if the point is within or on the polygon boundary (or if filtering is disabled), False otherwise.
    """
    # If spatial filtering is disabled or no polygon is provided, permit all detections
    if not ENABLE_ROI_FILTER or roi_polygon_pixels is None:
        return True
    
    # pointPolygonTest returns >0 (inside), 0 (on edge), <0 (outside)
    return cv2.pointPolygonTest(roi_polygon_pixels, (float(point[0]), float(point[1])), False) >= 0
