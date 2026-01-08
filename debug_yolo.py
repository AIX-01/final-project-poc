from ultralytics import YOLO
import cv2
import numpy as np

print("Imported YOLO")
try:
    model = YOLO('yolov8n-pose.pt')
    print("Model loaded")
    # Create a dummy image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(img)
    print("Inference successful")
except Exception as e:
    print(f"Error: {e}")
