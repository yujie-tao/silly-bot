import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def get_nose_position(face_landmarks):
    """Extracts the nose position from MediaPipe face landmarks"""
    if face_landmarks:
        # landmark 1 is nose tip
        nose = face_landmarks.landmark[1]
        return nose.x, nose.y, nose.z
    return None

def main():
    # For webcam input:
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue

        # pass image by reference (not writable) for performance
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # get nose position
        nose_pos = get_nose_position(results.face_landmarks)
        if nose_pos:
            print(f"Nose position: {nose_pos}")

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # draw dot on teh nose
        if nose_pos:
            h, w, _ = image.shape
            cx, cy = int(nose_pos[0] * w), int(nose_pos[1] * h)
            cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)

        # flip image horizontally for selfie view
        cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()

if __name__ == '__main__':
    main()