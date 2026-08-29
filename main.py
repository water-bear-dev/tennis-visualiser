import cv2
from ultralytics import YOLO

def main():
    # Load the YOLOv8 model (this will automatically download yolov8n.pt if not present)
    model = YOLO('yolov8n.pt')

    # Open the video file
    video_path = 'input.mp4'
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Get video properties for output
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    print(f"Processing video {video_path}...")
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        
        if success:
            # Run YOLOv8 inference on the frame
            # stream=True is recommended for continuous processing
            results = model(frame, verbose=False)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Write the frame into the file 'output.mp4'
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...")
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close all frames
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Finished processing. Output saved to output.mp4")

if __name__ == '__main__':
    main()
