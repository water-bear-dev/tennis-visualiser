"""
src/trackers/ball_interpolator.py
Time-series trajectory interpolation engine for the tennis ball.
Performs bounded piecewise linear interpolation over segmented video scenes,
preventing cross-scene cut bleeding while bridging short camera occlusions.
"""

import numpy as np
import pandas as pd
from src.config import MAX_MISSING_FRAMES


def interpolate_segment(segment_detections: list[dict]) -> list[tuple[float, float] | None]:
    """
    Interpolates ball coordinates within a single continuous scene segment.
    Gaps larger than MAX_MISSING_FRAMES are strictly left as None to prevent trajectory distortion.

    Args:
        segment_detections (list[dict]): List of frame detection records belonging to a continuous scene.

    Returns:
        list[tuple[float, float] | None]: Interpolated list of (x, y) coordinates for each frame in the segment.
    """
    if not segment_detections:
        return []

    # Convert ball detection positions into a tabular numerical format for pandas
    data = []
    for f in segment_detections:
        if f['ball'] is not None:
            data.append({'x': f['ball'][0], 'y': f['ball'][1]})
        else:
            data.append({'x': np.nan, 'y': np.nan})

    df = pd.DataFrame(data)

    # Perform linear interpolation bounded by MAX_MISSING_FRAMES (default 8 frames)
    # limit_area='inside' ensures we do not extrapolate beyond the first or last valid detection
    df_interpolated = df.interpolate(method='linear', limit=MAX_MISSING_FRAMES, limit_area='inside')

    interpolated_balls = []
    for idx, row in df_interpolated.iterrows():
        if pd.notna(row['x']) and pd.notna(row['y']):
            interpolated_balls.append((float(row['x']), float(row['y'])))
        else:
            interpolated_balls.append(None)

    return interpolated_balls


def interpolate_ball_positions(frame_detections: list[dict]) -> list[tuple[float, float] | None]:
    """
    Interpolates missing ball coordinates across all video frames while respecting scene cut boundaries.

    Args:
        frame_detections (list[dict]): Full list of detection dictionaries from Pass 1.

    Returns:
        list[tuple[float, float] | None]: Full sequence of smoothed/interpolated ball coordinates.
    """
    print("\n--- Interpolating Missing Ball Coordinates with Safety Limits ---")
    if not frame_detections:
        return []

    # --------------------------------------------------------------------------
    # 1. Segment frames by detected scene cuts to avoid cross-cut interpolation
    # --------------------------------------------------------------------------
    segments = []
    current_segment = []

    for f in frame_detections:
        if f.get('scene_cut', False) and current_segment:
            segments.append(current_segment)
            current_segment = []
        current_segment.append(f)

    if current_segment:
        segments.append(current_segment)

    # --------------------------------------------------------------------------
    # 2. Interpolate each segment independently
    # --------------------------------------------------------------------------
    all_interpolated = []
    total_filled = 0
    raw_count = sum(1 for f in frame_detections if f['ball'] is not None)

    for seg in segments:
        seg_interpolated = interpolate_segment(seg)
        all_interpolated.extend(seg_interpolated)

    # Calculate statistics for telemetry reporting
    for idx, f in enumerate(frame_detections):
        if f['ball'] is None and all_interpolated[idx] is not None:
            total_filled += 1

    total_valid = sum(1 for b in all_interpolated if b is not None)
    print(f"Interpolation Complete across {len(segments)} scene segments:")
    print(f"  - Raw detected balls: {raw_count}")
    print(f"  - Interpolated frames filled (within {MAX_MISSING_FRAMES} max gap): {total_filled}")
    print(f"  - Total valid tracked ball frames: {total_valid}/{len(frame_detections)}")

    return all_interpolated
