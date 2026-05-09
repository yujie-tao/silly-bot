import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

# 1. Callback function to process results
def print_result(result: vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    if result.hand_landmarks:
        print(f'Hand landmarker result: {result.hand_landmarks[0][0]}') # Prints thumb tip

# 2. Configure MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM, # Crucial for camera
    result_callback=print_result)
detector = vision.HandLandmarker.create_from_options(options)

# 3. Webcam Input
cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success: break

    # Convert to MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    
    # 4. Detect Async
    detector.detect_async(mp_image, time.time_ns() // 1_000_000)

    cv2.imshow('MediaPipe Hand Tracking', image)
    if cv2.waitKey(1) & 0xFF == 27: # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
