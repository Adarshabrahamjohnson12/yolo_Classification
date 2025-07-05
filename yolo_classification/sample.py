# Import required libraries
import tensorflow as tf
import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO(r'C:\Users\adars\Desktop\tist\yolo_classification\best.pt')
# Process webcam frames
import cv2
from ultralytics import YOLO  # make sure you have this installed

# Load YOLO model
model = YOLO('yolov8n.pt')  # or your custom model like 'best.pt'

# Open webcam (0 is usually the default laptop camera)
cap = cv2.VideoCapture(0)

# Get webcam video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # fallback to 30 if 0

# Define the video writer to save output (optional)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('webcam_output.mp4', fourcc, fps, (width, height))

# Process webcam frames
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert BGR to RGB for YOLO
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO prediction
        results = model.predict(rgb_frame, verbose=False)

        # Draw results
        annotated_frame = results[0].plot()

        # Convert RGB back to BGR for OpenCV
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Save output frame
        out.write(bgr_frame)

        # Optional: display
        cv2.imshow('YOLOv8 Webcam', bgr_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()
