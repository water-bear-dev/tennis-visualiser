import cv2
import numpy as np
from src.config import ENABLE_ROI_FILTER, COURT_ROI_NORMALIZED


def get_roi_polygon_pixels(width: int, height: int) -> np.ndarray:
    """Converts normalized ROI coordinates into pixel coordinates."""
    return np.array([
        [int(x * width), int(y * height)] for x, y in COURT_ROI_NORMALIZED
    ], dtype=np.int32)


def is_inside_roi(point, roi_polygon_pixels: np.ndarray) -> bool:
    """Checks if a point (x, y) falls inside the polygon defined by pixel coordinates."""
    if not ENABLE_ROI_FILTER or roi_polygon_pixels is None:
        return True
    return cv2.pointPolygonTest(roi_polygon_pixels, (float(point[0]), float(point[1])), False) >= 0
