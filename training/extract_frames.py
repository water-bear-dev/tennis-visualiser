"""
training/extract_frames.py
Multi-Video Frame Harvester.
Scans data/inputs/ for match video footage and extracts sampled training frames
at configurable sampling frequencies into the training/dataset/images/ repository.
"""

import argparse
import glob
import os
import sys
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import INPUT_DIR


def extract_frames_from_videos(input_dir: str = INPUT_DIR, output_dir: str = 'training/dataset/images', 
                               sample_rate: int = 5, max_frames_per_video: int = 400):
    """
    Extracts sampled frames from all .mp4 videos in the input directory.

    Args:
        input_dir (str): Folder containing raw .mp4 match videos.
        output_dir (str): Target folder to write extracted .jpg images.
        sample_rate (int): Extract every Nth frame (e.g. 5 = ~6 frames/sec from 30fps video).
        max_frames_per_video (int): Maximum frames to harvest per video file to maintain class balance.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(input_dir, '*.mp4')) + glob.glob('*.mp4'))
    video_files = list(dict.fromkeys(video_files))  # Deduplicate matching files

    if not video_files:
        print(f"No .mp4 video files found in '{input_dir}' or project root.")
        return

    print(f"\n--- Multi-Video Frame Harvester: Found {len(video_files)} video(s) ---")
    total_extracted = 0

    for vid_path in video_files:
        vid_stem = os.path.splitext(os.path.basename(vid_path))[0]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Could not open: {vid_path}")
            continue

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing '{vid_path}': {total_vid_frames} frames @ {fps}fps...")

        frame_idx = 0
        extracted_for_vid = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or extracted_for_vid >= max_frames_per_video:
                break

            if frame_idx % sample_rate == 0:
                frame_filename = f"{vid_stem}_f{frame_idx:06d}.jpg"
                out_path = os.path.join(output_dir, frame_filename)
                cv2.imwrite(out_path, frame)
                extracted_for_vid += 1
                total_extracted += 1

            frame_idx += 1

        cap.release()
        print(f"  -> Extracted {extracted_for_vid} frames from '{vid_stem}'")

    print(f"\nHarvester Complete: Total of {total_extracted} frames saved to '{output_dir}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract frames from multiple tennis videos for model training.")
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR, help="Directory containing raw .mp4 videos")
    parser.add_argument('--output_dir', type=str, default='training/dataset/images', help="Output directory for extracted frames")
    parser.add_argument('--sample_rate', type=int, default=5, help="Extract every N-th frame")
    parser.add_argument('--max_frames', type=int, default=400, help="Max frames to extract per video")
    args = parser.parse_args()

    extract_frames_from_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sample_rate=args.sample_rate,
        max_frames_per_video=args.max_frames
    )
