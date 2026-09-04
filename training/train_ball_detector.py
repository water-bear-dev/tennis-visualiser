"""
training/train_ball_detector.py
Custom YOLOv8 Tennis Ball Model Training & MLOps Fine-Tuning CLI.
Fine-tunes YOLOv8 backbones on custom multi-video datasets and automatically
deploys the top-performing checkpoint (runs/.../best.pt) directly to 'best_tennis.pt'.
"""

import argparse
import os
import shutil
import sys
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import BALL_IMGSZ


def train_ball_detector(data_yaml: str = 'training/dataset/data.yaml', 
                        epochs: int = 50, 
                        batch_size: int = 8, 
                        base_model: str = 'yolov8n.pt',
                        imgsz: int = BALL_IMGSZ):
    """
    Trains a custom YOLOv8 model for high-resolution tennis ball detection
    and automatically deploys the best checkpoint as 'best_tennis.pt'.

    Args:
        data_yaml (str): Path to data.yaml dataset definition.
        epochs (int): Number of training iterations.
        batch_size (int): Training batch size.
        base_model (str): Base checkpoint to fine-tune from (e.g. yolov8n.pt).
        imgsz (int): Resolution for training inference.
    """
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset config '{data_yaml}' not found.")
        print("Please run 'python training/extract_frames.py' and 'python training/auto_label.py' first.")
        return

    print(f"\n=======================================================")
    print(f"🎾 Initiating Custom Tennis Ball Detector Training")
    print(f"=======================================================")
    print(f"Data YAML:   {data_yaml}")
    print(f"Base Model:  {base_model}")
    print(f"Epochs:      {epochs}")
    print(f"Batch Size:  {batch_size}")
    print(f"Resolution:  {imgsz}x{imgsz}")
    print(f"=======================================================\n")

    model = YOLO(base_model)
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        name='tennis_ball_detector',
        save=True,
        plots=True,
        exist_ok=True
    )

    # Check for exported best weights
    best_weights = os.path.join('runs', 'detect', 'tennis_ball_detector', 'weights', 'best.pt')
    if os.path.exists(best_weights):
        target_path = 'best_tennis.pt'
        shutil.copy(best_weights, target_path)
        print(f"\nTraining Complete! Best model automatically deployed to '{target_path}'.")
        print("The visualizer pipeline will now automatically use your custom fine-tuned weights!")
    else:
        print(f"\nTraining finished. Check 'runs/detect/tennis_ball_detector' for metrics.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for tennis ball detection on custom multi-video dataset.")
    parser.add_argument('--data', type=str, default='training/dataset/data.yaml', help="Path to data.yaml")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch', type=int, default=8, help="Batch size")
    parser.add_argument('--base', type=str, default='yolov8n.pt', help="Base model checkpoint")
    parser.add_argument('--imgsz', type=int, default=BALL_IMGSZ, help="Image resolution for training")
    args = parser.parse_args()

    train_ball_detector(
        data_yaml=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        base_model=args.base,
        imgsz=args.imgsz
    )
