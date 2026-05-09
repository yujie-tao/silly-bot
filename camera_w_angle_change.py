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

def solve_theta_from_xy(X, Y, L, b=None, c=None):
    """
    Closed-form inverse kinematics for the 5-bar linkage.

    Geometry (matches visualization.py, +y down):
        P1 = (0, 0)            left motor anchor
        P2 = (L, 0)            right motor anchor
        P5 = (X, Y)            end effector
        |P1-P3| = |P3-P5| = L  left arm + left rod
        |P2-P4| = |P4-P5| = L  right arm + right rod

    theta1 is measured at P1 from +y toward -x (left arm swings outward).
    theta2 is measured at P2 from +y toward +x (right arm swings outward).

    Each arm is an independent 2-link IK problem, solved with the law of
    cosines. The branch choices below force the elbows outward (left arm
    points left, right arm points right) to match compute_joints().

    Returns dict with theta1, theta2, error, success — or success=False
    if (X, Y) is unreachable.
    """
    b = L if b is None else b
    c = L if c is None else c

    fail = {"theta1": np.nan, "theta2": np.nan,
            "error": np.inf, "success": False}

    # Left arm: triangle P1-P3-P5 with sides L (arm), b (rod), d1 (base)
    d1 = np.hypot(X, Y)
    if d1 < 1e-12 or d1 > L + b or d1 < abs(L - b):
        return fail
    cos_a1 = (L * L + d1 * d1 - b * b) / (2 * L * d1)
    alpha1 = np.arccos(np.clip(cos_a1, -1.0, 1.0))
    phi1 = np.arctan2(Y, X)
    P3 = L * np.array([np.cos(phi1 + alpha1), np.sin(phi1 + alpha1)])
    theta1 = np.arctan2(-P3[0], P3[1])

    # Right arm: triangle P2-P4-P5 with sides L (arm), c (rod), d2 (base)
    d2 = np.hypot(X - L, Y)
    if d2 < 1e-12 or d2 > L + c or d2 < abs(L - c):
        return fail
    cos_a2 = (L * L + d2 * d2 - c * c) / (2 * L * d2)
    alpha2 = np.arccos(np.clip(cos_a2, -1.0, 1.0))
    phi2 = np.arctan2(Y, X - L)
    P4 = np.array([L, 0.0]) + L * np.array(
        [np.cos(phi2 - alpha2), np.sin(phi2 - alpha2)]
    )
    theta2 = np.arctan2(P4[0] - L, P4[1])

    # Closure check: rods from P3, P4 to (X, Y) must have lengths b, c.
    err = max(abs(np.hypot(X - P3[0], Y - P3[1]) - b),
              abs(np.hypot(X - P4[0], Y - P4[1]) - c))

    return {
        "theta1": float(theta1),
        "theta2": float(theta2),
        "error": float(err),
        "success": True,
    }
def solve_theta_from_xy(X, Y, L, b=None, c=None):
    """
    Closed-form inverse kinematics for the 5-bar linkage.

    Geometry (matches visualization.py, +y down):
        P1 = (0, 0)            left motor anchor
        P2 = (L, 0)            right motor anchor
        P5 = (X, Y)            end effector
        |P1-P3| = |P3-P5| = L  left arm + left rod
        |P2-P4| = |P4-P5| = L  right arm + right rod

    theta1 is measured at P1 from +y toward -x (left arm swings outward).
    theta2 is measured at P2 from +y toward +x (right arm swings outward).

    Each arm is an independent 2-link IK problem, solved with the law of
    cosines. The branch choices below force the elbows outward (left arm
    points left, right arm points right) to match compute_joints().

    Returns dict with theta1, theta2, error, success — or success=False
    if (X, Y) is unreachable.
    """
    b = L if b is None else b
    c = L if c is None else c

    fail = {"theta1": np.nan, "theta2": np.nan,
            "error": np.inf, "success": False}

    # Left arm: triangle P1-P3-P5 with sides L (arm), b (rod), d1 (base)
    d1 = np.hypot(X, Y)
    if d1 < 1e-12 or d1 > L + b or d1 < abs(L - b):
        return fail
    cos_a1 = (L * L + d1 * d1 - b * b) / (2 * L * d1)
    alpha1 = np.arccos(np.clip(cos_a1, -1.0, 1.0))
    phi1 = np.arctan2(Y, X)
    P3 = L * np.array([np.cos(phi1 + alpha1), np.sin(phi1 + alpha1)])
    theta1 = np.arctan2(-P3[0], P3[1])

    # Right arm: triangle P2-P4-P5 with sides L (arm), c (rod), d2 (base)
    d2 = np.hypot(X - L, Y)
    if d2 < 1e-12 or d2 > L + c or d2 < abs(L - c):
        return fail
    cos_a2 = (L * L + d2 * d2 - c * c) / (2 * L * d2)
    alpha2 = np.arccos(np.clip(cos_a2, -1.0, 1.0))
    phi2 = np.arctan2(Y, X - L)
    P4 = np.array([L, 0.0]) + L * np.array(
        [np.cos(phi2 - alpha2), np.sin(phi2 - alpha2)]
    )
    theta2 = np.arctan2(P4[0] - L, P4[1])

    # Closure check: rods from P3, P4 to (X, Y) must have lengths b, c.
    err = max(abs(np.hypot(X - P3[0], Y - P3[1]) - b),
              abs(np.hypot(X - P4[0], Y - P4[1]) - c))

    return {
        "theta1": np.degrees(float(theta1)),
        "theta2": np.degrees(float(theta2)),
        "error": float(err),
        "success": True,
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
