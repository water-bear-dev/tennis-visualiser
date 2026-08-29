import numpy as np
import pandas as pd
from src.config import MAX_MISSING_FRAMES


def interpolate_segment(segment_detections: list) -> list:
    """
    Interpolates ball coordinates within a single continuous scene segment.
    Gaps larger than MAX_MISSING_FRAMES are strictly left as None.
    """
    if not segment_detections:
        return []

    data = []
    for f in segment_detections:
        if f['ball'] is not None:
            data.append({'x': f['ball'][0], 'y': f['ball'][1]})
        else:
            data.append({'x': np.nan, 'y': np.nan})

    df = pd.DataFrame(data)

    # Linear interpolation restricted by MAX_MISSING_FRAMES limit
    df_interpolated = df.interpolate(method='linear', limit=MAX_MISSING_FRAMES, limit_area='inside')

    interpolated_balls = []
    for idx, row in df_interpolated.iterrows():
        if pd.notna(row['x']) and pd.notna(row['y']):
            interpolated_balls.append((float(row['x']), float(row['y'])))
        else:
            interpolated_balls.append(None)

    return interpolated_balls


def interpolate_ball_positions(frame_detections: list) -> list:
    """
    Interpolates missing ball coordinates across frames, respecting scene cut boundaries
    and max missing frame limits.
    """
    print("\n--- Interpolating Missing Ball Coordinates with Safety Limits ---")
    if not frame_detections:
        return []

    # 1. Segment frames by scene cuts
    segments = []
    current_segment = []

    for f in frame_detections:
        if f.get('scene_cut', False) and current_segment:
            segments.append(current_segment)
            current_segment = []
        current_segment.append(f)

    if current_segment:
        segments.append(current_segment)

    # 2. Interpolate each segment independently
    all_interpolated = []
    total_filled = 0
    raw_count = sum(1 for f in frame_detections if f['ball'] is not None)

    for seg in segments:
        seg_interpolated = interpolate_segment(seg)
        all_interpolated.extend(seg_interpolated)

    for idx, f in enumerate(frame_detections):
        if f['ball'] is None and all_interpolated[idx] is not None:
            total_filled += 1

    total_valid = sum(1 for b in all_interpolated if b is not None)
    print(f"Interpolation Complete across {len(segments)} scene segments:")
    print(f"  - Raw detected balls: {raw_count}")
    print(f"  - Interpolated frames filled (within {MAX_MISSING_FRAMES} max gap): {total_filled}")
    print(f"  - Total valid tracked ball frames: {total_valid}/{len(frame_detections)}")

    return all_interpolated
