import numpy as np
import pandas as pd
from src.config import MAX_INTERPOLATION_GAP


def interpolate_ball_positions(frame_detections: list) -> list:
    """
    Interpolates missing ball coordinates across frames using Pandas linear interpolation.
    """
    print("\n--- Interpolating Missing Ball Coordinates ---")
    data = []
    for f in frame_detections:
        if f['ball'] is not None:
            data.append({'x': f['ball'][0], 'y': f['ball'][1]})
        else:
            data.append({'x': np.nan, 'y': np.nan})

    df = pd.DataFrame(data)
    
    # Linear interpolation with gap limit
    df_interpolated = df.interpolate(method='linear', limit=MAX_INTERPOLATION_GAP, limit_direction='both')

    interpolated_balls = []
    filled_count = 0
    for idx, row in df_interpolated.iterrows():
        if pd.notna(row['x']) and pd.notna(row['y']):
            interpolated_balls.append((float(row['x']), float(row['y'])))
            if pd.isna(df.loc[idx, 'x']):
                filled_count += 1
        else:
            interpolated_balls.append(None)

    total_valid = sum(1 for b in interpolated_balls if b is not None)
    print(f"Interpolation Complete: Filled {filled_count} missing frames. Total frames with ball={total_valid}/{len(frame_detections)}")
    return interpolated_balls
