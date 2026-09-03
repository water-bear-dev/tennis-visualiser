import glob
import os
import random
import shutil
import sys
import cv2
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import (
    MODEL_CANDIDATES,
    BALL_IMGSZ,
    BALL_CONF_THRESHOLD,
    COCO_BALL_CLASS_ID,
    COURT_ROI_NORMALIZED
)
from src.utils.roi_utils import get_roi_polygon_pixels, is_inside_roi


def auto_label_dataset(images_dir: str = 'training/dataset/images', 
                       dataset_dir: str = 'training/dataset',
                       val_split: float = 0.20):
    """
    Runs high-confidence inference on extracted images, generates YOLO format labels,
    splits data into train/val sets, and writes data.yaml.
    """
    image_files = sorted(glob.glob(os.path.join(images_dir, '*.jpg')) + glob.glob(os.path.join(images_dir, '*.png')))
    if not image_files:
        print(f"No image files found in '{images_dir}'. Run extract_frames.py first.")
        return

    print(f"\n--- Semi-Supervised Auto-Labeler: Processing {len(image_files)} images ---")
    
    # Load detector
    model_path = 'yolov8n.pt'
    for c in MODEL_CANDIDATES:
        if os.path.exists(c):
            model_path = c
            break
    model = YOLO(model_path)
    print(f"Using baseline model for pseudo-labeling: '{model_path}'")

    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')
    train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        os.makedirs(d, exist_ok=True)

    labeled_count = 0

    for img_path in image_files:
        img_name = os.path.basename(img_path)
        stem = os.path.splitext(img_name)[0]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        roi_polygon = get_roi_polygon_pixels(w, h)

        # Run inference
        results = model.predict(img, conf=BALL_CONF_THRESHOLD, imgsz=BALL_IMGSZ, verbose=False)
        labels = []

        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy

                is_ball = (cls_id == COCO_BALL_CLASS_ID) or (len(model.names) == 1 and cls_id == 0)
                if is_ball:
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    bw = x2 - x1
                    bh = y2 - y1

                    if bw < 45 and bh < 45 and is_inside_roi((cx, cy), roi_polygon):
                        # Convert to normalized YOLO format: class_id cx cy w h
                        norm_cx = cx / w
                        norm_cy = cy / h
                        norm_w = bw / w
                        norm_h = bh / h
                        labels.append(f"0 {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")

        # Decide train vs val split
        is_val = (random.random() < val_split)
        target_img_dir = val_img_dir if is_val else train_img_dir
        target_lbl_dir = val_lbl_dir if is_val else train_lbl_dir

        # Copy image
        shutil.copy(img_path, os.path.join(target_img_dir, img_name))

        # Write label file (even if empty to teach background)
        lbl_file = os.path.join(target_lbl_dir, f"{stem}.txt")
        with open(lbl_file, 'w') as f:
            if labels:
                f.write("\n".join(labels) + "\n")
                labeled_count += 1

    # Generate data.yaml
    yaml_content = f"""path: {os.path.abspath(dataset_dir)}
train: images/train
val: images/val

names:
  0: tennis_ball
"""
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\nAuto-Labeling Complete: {labeled_count}/{len(image_files)} frames contained ball detections.")
    print(f"Dataset generated at '{dataset_dir}' with config: '{yaml_path}'")


if __name__ == '__main__':
    auto_label_dataset()
