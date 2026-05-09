import serial
import time
import numpy as np
import serial.tools.list_ports

ports = serial.tools.list_ports.comports()

print("Available ports:")
for port, desc, hwid in sorted(ports):
    print(f"- {port}: {desc}")

# Set up the serial connection (Change 'COM3' to your actual port)
arduino_port = '/dev/cu.usbmodem1101' #'/dev/cu.usbmodem31301'
baud_rate = 9600

print(f"Connecting to {arduino_port}...")
ser = serial.Serial(arduino_port, baud_rate, timeout=1)

# Wait for the Arduino to reboot after opening the serial port
time.sleep(2) 
print("Connected! Type angles in the terminal.")

L = 5.0
b = L
c = L

def send_angles(angle1, angle2):
    """Formats the angles as 'angle1,angle2\n' and sends them."""
    # Ensure angles are within safe servo limits (0 to 180)

    angle1 = 180-(angle1 + 90)
    angle2 = angle2 + 90

    angle1 = max(0, min(180, int(angle1)))
    angle2 = max(0, min(180, int(angle2)))
    
    # Create the string format: "90,135\n"
    command = f"{angle1},{angle2}\n"
    print(command)
    
    # Send it encoded as bytes
    ser.write(command.encode('utf-8'))
    # print(f"Sent: {command.strip()}")

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

try:

    t = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    cx, cy = 5 / 2, 7.5     # centre: (2.5, 7.0)
    r = 1
    xs = cx + r * np.cos(t)
    ys = cy + r * np.sin(t)

    for i in range(len(xs)):
        print(xs[i], ys[i])
        theta_result = solve_theta_from_xy(xs[i], ys[i], L, b, c)
        send_angles(theta_result["theta1"], theta_result["theta2"])
        i = i +1
        time.sleep(0.1)
    
    # while True:
    #     # Get input from the user
    #     user_input = input("Enter two angles separated by a space (e.g., '90 45') or 'q' to quit: ")
        
    #     if user_input.lower() == 'q':
    #         break
            
    #     try:
    #         # Split the input into two separate numbers
    #         val1, val2 = user_input.split()
    #         send_angles(val1, val2)
            
    #         # Wait a tiny bit for the Arduino to process and reply
    #         time.sleep(0.1)
            
    #         # Read any debug messages the Arduino sends back
    #         while ser.in_waiting > 0:
    #             response = ser.readline().decode('utf-8').strip()
    #             print(f"Arduino says: {response}")
                
    #     except ValueError:
    #         print("Invalid input. Please enter two numbers separated by a space.")

except serial.SerialException as e:
    print(f"Serial communication error: {e}")

finally:
    ser.close()
    print("Connection closed.")