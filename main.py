import cv2
from src.config import VIDEO_PATH
from src.detectors.yolo_detector import load_detector, extract_detections
from src.trackers.ball_interpolator import interpolate_ball_positions
from src.visualizers.video_annotator import render_annotated_video
from src.utils.roi_utils import get_roi_polygon_pixels


def main():
    # 1. Initialize detector model
    model = load_detector()

    # 2. Open input video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video '{VIDEO_PATH}'")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 3. Calculate pixel-level ROI bounds
    roi_polygon_pixels = get_roi_polygon_pixels(width, height)

    # 4. Pipeline Execution
    # Pass 1: Extract detections
    frame_detections = extract_detections(cap, model, roi_polygon_pixels)
    
    # Tracker: Interpolate missing ball frames
    interpolated_balls = interpolate_ball_positions(frame_detections)
    
    # Pass 2: Render visualizations & trajectory trail
    render_annotated_video(cap, frame_detections, interpolated_balls, roi_polygon_pixels, width, height, fps)

    # 5. Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
