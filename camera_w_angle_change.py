import cv2
import numpy as np
import mediapipe as mp
from scipy.optimize import least_squares

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


X = 1.0
Y = 1.5
L = 5.0
b = L
c = L

def solve_theta_from_xy(X, Y, L, b, c, n_starts=25):
    """
    Solve for theta1, theta2 from X, Y using the equations:

        P = (-L sin(theta1) + L cos(B),
              L cos(theta1) + L sin(B))

        P = ( L sin(theta2) - L sin(C),
              L cos(theta2) + L sin(C))

    with principal/smallest B and C from acos, so sin(B), sin(C) >= 0.
    """
    
    def residual(theta):
        theta1, theta2 = theta

        # Distance a
        a = L * np.sqrt(
            (np.sin(theta1) - np.sin(theta2))**2
            + (np.cos(theta1) - np.cos(theta2))**2
        )

        # Avoid division by zero
        if a < 1e-12:
            return np.array([1e6, 1e6, 1e6, 1e6])

        cosB = (a**2 + c**2 - b**2) / (2 * a * c)
        cosC = (a**2 + b**2 - c**2) / (2 * a * b)

        # Invalid triangle geometry
        if abs(cosB) > 1 or abs(cosC) > 1:
            return np.array([1e6, 1e6, 1e6, 1e6])

        # Principal smallest angles: B,C in [0, pi]
        sinB = np.sqrt(1 - cosB**2)
        sinC = np.sqrt(1 - cosC**2)

        X1 = -L * np.sin(theta1) + L * cosB
        Y1 =  L * np.cos(theta1) + L * sinB

        X2 =  L * np.sin(theta2) - L * sinC
        Y2 =  L * np.cos(theta2) + L * sinC

        return np.array([
            X1 - X,
            Y1 - Y,
            X2 - X,
            Y2 - Y
        ])

    best = None

    # Multistart search over possible theta1/theta2 values
    guesses = np.linspace(-np.pi, np.pi, n_starts)

    for g1 in guesses:
        for g2 in guesses:
            sol = least_squares(
                residual,
                x0=np.array([g1, g2]),
                bounds=([-np.pi, -np.pi], [np.pi, np.pi]),
                xtol=1e-12,
                ftol=1e-12,
                gtol=1e-12,
                max_nfev=5000
            )

            err = np.linalg.norm(residual(sol.x))

            if best is None or err < best["error"]:
                best = {
                    "theta1": np.degrees(sol.x[0]),
                    "theta2": np.degrees(sol.x[1]),
                    "error": err,
                    "success": sol.success
                }

    return best

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
            theta_result = solve_theta_from_xy(nose_pos[0], nose_pos[1], L, b, c)
            print(f"Nose position: {nose_pos}")
            print(theta_result)

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
