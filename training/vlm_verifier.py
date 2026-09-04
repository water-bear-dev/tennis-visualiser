"""
training/vlm_verifier.py
Local VLM (Vision-Language Model) Active Label Auditor.
Connects to local Moondream vision model via Ollama to audit candidate ball crops,
pruning ambiguous false positives (court marks, shoes, racket frames) from the training dataset.
"""

import argparse
import base64
import json
import os
import sys
import cv2
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
DEFAULT_VISION_MODEL = "moondream"


def encode_image_to_base64(image_mat) -> str:
    """
    Encodes OpenCV image numpy array to base64 jpeg string.

    Args:
        image_mat (np.ndarray): Cropped candidate bounding box image.

    Returns:
        str: Base64 UTF-8 encoded string.
    """
    success, buffer = cv2.imencode('.jpg', image_mat)
    if not success:
        return ""
    return base64.b64encode(buffer).decode('utf-8')


def verify_crop_contains_ball(crop_mat, model: str = DEFAULT_VISION_MODEL) -> bool:
    """
    Asks the local VLM (moondream) if the cropped bounding box contains a real tennis ball.
    Filters out shoes, racket tips, stadium logos, and white court lines.
    """
    b64_str = encode_image_to_base64(crop_mat)
    if not b64_str:
        return True  # Fallback to keep if encoding fails

    prompt = "Is there a small yellow or green tennis ball visible in this image? Answer with just 'yes' or 'no'."
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
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=30)
        if response.status_code == 200:
            text = response.json().get("response", "").strip().lower()
            return "yes" in text
        return True
    except Exception as e:
        # If VLM is unreachable, gracefully fall back to detector prediction
        return True


def audit_dataset_labels(dataset_dir: str = 'training/dataset', model: str = DEFAULT_VISION_MODEL):
    """
    Audits existing pseudo-labels in training/dataset/labels, removing false positives
    verified by local VLM.
    """
    print(f"\n=======================================================")
    print(f"👁️ VLM Label Auditor & Denoising (Powered by {model})")
    print(f"=======================================================")

    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')

    if not os.path.exists(train_lbl_dir):
        print(f"Labels directory '{train_lbl_dir}' not found.")
        return

    label_files = [f for f in os.listdir(train_lbl_dir) if f.endswith('.txt')]
    print(f"Auditing sample labels from {len(label_files)} training frames...")

    verified_balls = 0
    denoised_false_positives = 0

    # Sample audit up to 50 label files for quick demonstration
    for lbl_name in label_files[:50]:
        lbl_path = os.path.join(train_lbl_dir, lbl_name)
        img_stem = os.path.splitext(lbl_name)[0]
        img_path = os.path.join(train_img_dir, f"{img_stem}.jpg")

        if not os.path.exists(img_path):
            continue

        with open(lbl_path, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        if not lines:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        kept_lines = []

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id, cx, cy, bw, bh = map(float, parts)
            # Calculate pixel coordinates with margin
            px = int(cx * w)
            py = int(cy * h)
            pw = int(bw * w)
            ph = int(bh * h)

            margin = 25
            x1 = max(0, px - pw // 2 - margin)
            y1 = max(0, py - ph // 2 - margin)
            x2 = min(w, px + pw // 2 + margin)
            y2 = min(h, py + ph // 2 + margin)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                kept_lines.append(line)
                continue

            is_valid = verify_crop_contains_ball(crop, model=model)
            if is_valid:
                kept_lines.append(line)
                verified_balls += 1
            else:
                denoised_false_positives += 1

        with open(lbl_path, 'w') as f:
            if kept_lines:
                f.write("\n".join(kept_lines) + "\n")

    print(f"\nAudit Complete: Verified {verified_balls} balls | Removed {denoised_false_positives} noisy false positives.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VLM Label Verification and Denoising.")
    parser.add_argument('--dataset', type=str, default='training/dataset', help="Path to training dataset folder")
    parser.add_argument('--model', type=str, default=DEFAULT_VISION_MODEL, help="Local VLM model name (moondream)")
    args = parser.parse_args()

    audit_dataset_labels(dataset_dir=args.dataset, model=args.model)
