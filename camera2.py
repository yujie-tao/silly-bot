import cv2
import time
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Global variable to store the latest result from the callback
latest_result = None

# 1. Update the callback to store the result
def save_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result

# Define the path to your downloaded model
model_path = 'pose_landmarker.task' # Make sure this path is correct

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=save_result # Use the updated callback
)

# 2. Open the webcam using OpenCV
cap = cv2.VideoCapture(0)

# Create the landmarker instance
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the frame from BGR (OpenCV default) to RGB (MediaPipe requirement)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert the RGB frame to a MediaPipe Image object
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Calculate the timestamp in milliseconds
        frame_timestamp_ms = int(time.time() * 1000)

        # 3. Send the image to the landmarker
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # 4. Draw the landmarks if we have a result
        if latest_result and latest_result.pose_landmarks:
            for pose_landmarks in latest_result.pose_landmarks:
                # Convert the tasks API landmark format to the protobuf format required by drawing_utils
                pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                pose_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) 
                    for landmark in pose_landmarks
                ])
                
                # Draw the landmarks on the original BGR frame
                mp_drawing.draw_landmarks(
                    frame,
                    pose_landmarks_proto,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing_styles.get_default_pose_landmarks_style()
                )

        # 5. Display the frame
        cv2.imshow('MediaPipe Pose Live Stream', frame)

        # Press 'ESC' to exit
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()