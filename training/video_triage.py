"""
training/video_triage.py
Smart VLM Video Triage and Active Rally Segmentation.
Scans raw broadcast tennis footage and uses local Moondream vision model to classify
frames into active live rallies vs. non-play segments (crowd, breaks, close-ups, replays).
"""

import argparse
import base64
import json
import os
import sys
import cv2
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import INPUT_DIR

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_VISION_MODEL = "moondream"


def encode_image_to_base64(image_mat) -> str:
    """
    Encodes OpenCV image numpy array to base64 jpeg string.

    Args:
        image_mat (np.ndarray): Video frame image array.

    Returns:
        str: Base64 UTF-8 encoded string.
    """
    success, buffer = cv2.imencode('.jpg', image_mat)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')


def classify_scene_frame(frame, model: str = DEFAULT_VISION_MODEL) -> str:
    """
    Classifies video frame using local VLM into:
    - 'RALLY' (Full court view with active tennis play)
    - 'REPLAY_OR_BREAK' (Close-up, crowd, player walking, chair umpire)
    """
    b64_str = encode_image_to_base64(frame)
    if not b64_str:
        return "RALLY"

    prompt = "Is this image showing a full-court tennis match view during active play? Answer with only 'RALLY' or 'OTHER'."
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [b64_str],
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=20)
        if response.status_code == 200:
            text = response.json().get("response", "").strip().upper()
            return "RALLY" if "RALLY" in text else "OTHER"
        return "RALLY"
    except Exception:
        return "RALLY"


def triage_video(video_path: str, sample_interval_sec: float = 2.0, model: str = DEFAULT_VISION_MODEL) -> dict:
    """
    Scans video every N seconds and builds an index of active rally timestamps.
    """
    print(f"\n=======================================================")
    print(f"🎬 VLM Smart Video Triage: {video_path}")
    print(f"=======================================================")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step_frames = int(fps * sample_interval_sec)

    rally_segments = []
    current_frame = 0
    sampled_count = 0
    rally_count = 0

    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_sec = round(current_frame / fps, 2)
        scene_type = classify_scene_frame(frame, model=model)
        sampled_count += 1

        if scene_type == "RALLY":
            rally_count += 1
            rally_segments.append(timestamp_sec)

        current_frame += step_frames

    cap.release()

    rally_pct = (rally_count / sampled_count * 100) if sampled_count > 0 else 0
    print(f"Triage Result: {rally_count}/{sampled_count} sample points ({rally_pct:.1f}%) identified as Live Rally.")

    triage_info = {
        "video_file": os.path.basename(video_path),
        "total_sampled_points": sampled_count,
        "live_rally_points": rally_count,
        "live_rally_percentage": round(rally_pct, 1),
        "rally_timestamps_sec": rally_segments
    }
    return triage_info


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Smart VLM Video Triage & Rally Segmentation")
    parser.add_argument('--video', type=str, default=os.path.join(INPUT_DIR, 'input_grass.mp4'), help="Path to input video")
    parser.add_argument('--model', type=str, default=DEFAULT_VISION_MODEL, help="Local VLM model name")
    args = parser.parse_args()

    triage_video(args.video, model=args.model)
