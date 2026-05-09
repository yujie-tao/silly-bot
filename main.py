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
    angle1 = max(0, min(180, int(angle1)))
    angle2 = max(0, min(180, int(angle2)))
    
    command = f"{angle1},{angle2}\n"
    ser.write(command.encode('utf-8'))
    # print(f"Sent: {command.strip()}") # Uncomment to debug sent values

# --- Camera & Math Setup ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

X = 1.0
Y = 1.5
L = 5.0
b = L
c = L

def solve_theta_from_xy(X, Y, L, b, c, n_starts=25):
    # [Unchanged from your original code]
    def residual(theta):
        theta1, theta2 = theta
        a = L * np.sqrt((np.sin(theta1) - np.sin(theta2))**2 + (np.cos(theta1) - np.cos(theta2))**2)
        if a < 1e-12: return np.array([1e6, 1e6, 1e6, 1e6])
        cosB = (a**2 + c**2 - b**2) / (2 * a * c)
        cosC = (a**2 + b**2 - c**2) / (2 * a * b)
        if abs(cosB) > 1 or abs(cosC) > 1: return np.array([1e6, 1e6, 1e6, 1e6])
        sinB = np.sqrt(1 - cosB**2)
        sinC = np.sqrt(1 - cosC**2)
        X1 = -L * np.sin(theta1) + L * cosB
        Y1 =  L * np.cos(theta1) + L * sinB
        X2 =  L * np.sin(theta2) - L * sinC
        Y2 =  L * np.cos(theta2) + L * sinC
        return np.array([X1 - X, Y1 - Y, X2 - X, Y2 - Y])

    best = None
    guesses = np.linspace(-np.pi, np.pi, n_starts)
    for g1 in guesses:
        for g2 in guesses:
            sol = least_squares(residual, x0=np.array([g1, g2]), bounds=([-np.pi, -np.pi], [np.pi, np.pi]),
                                xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=5000)
            err = np.linalg.norm(residual(sol.x))
            if best is None or err < best["error"]:
                best = {"theta1": np.degrees(sol.x[0]), "theta2": np.degrees(sol.x[1]), 
                        "error": err, "success": sol.success}
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