import cv2
import numpy as np
import mediapipe as mp
from scipy.optimize import least_squares
import serial
import time
import serial.tools.list_ports

# --- Serial Setup ---
arduino_port = '/dev/cu.usbmodem11301' # Change to your actual port if needed
baud_rate = 9600

print(f"Connecting to {arduino_port}...")
try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    time.sleep(2) # Wait for the Arduino to reboot
    print("Connected to Arduino!")
except serial.SerialException as e:
    print(f"Serial communication error: {e}")
    exit()

def send_angles(angle1, angle2):
    """Formats the angles as 'angle1,angle2\n' and sends them."""
    # Ensure angles are within safe servo limits (0 to 180)
    angle1 = angle1 + 90
    angle2 = angle2 + 90
    angle1 = max(0, min(180, int(angle1)))
    angle2 = max(0, min(180, int(angle2)))
    
    
    command = f"{angle1},{angle2}\n"
    # ser.write(command.encode('utf-8'))
    print(f"Sent: {command.strip()}") # Uncomment to debug sent values

# --- Camera & Math Setup ---
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
        "theta1": np.degrees(float(theta1)),
        "theta2": np.degrees(float(theta2)),
        "error": float(err),
        "success": True,
    }
    
    return best

def get_nose_position(face_landmarks):
    if face_landmarks:
        nose = face_landmarks.landmark[1]
        return nose.x, nose.y, nose.z
    return None

def main():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        try:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)

                nose_pos = get_nose_position(results.face_landmarks)
                if nose_pos:
                    theta_result = solve_theta_from_xy(nose_pos[0], nose_pos[1], L, b, c)
                    
                    # --- NEW: Pass the results to the Arduino ---
                    if theta_result and theta_result["success"]:
                        print(theta_result["theta1"], theta_result["theta2"])
                        
                        # theta_result is a dictionary, so we access it by key rather than index
                        send_angles(theta_result["theta1"], theta_result["theta2"])

                # --- NEW: Read any debug messages the Arduino sends back ---
                while ser.in_waiting > 0:
                    response = ser.readline().decode('utf-8', errors='ignore').strip()
                    print(nose_pos[0], nose_pos[1])
                    print(f"Arduino says: {response}")

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if nose_pos:
                    h, w, _ = image.shape
                    cx, cy = int(nose_pos[0] * w), int(nose_pos[1] * h)
                    cv2.circle(image, (cx, cy), 8, (0, 0, 255), -1)

                cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    break
        
        finally:
            # Safely close the serial port when exiting (e.g., pressing ESC or Ctrl+C)
            ser.close()
            print("Serial connection closed.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()