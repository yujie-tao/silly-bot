# import numpy as np
# from scipy.optimize import fsolve

# def solve_kinematics(target_x, target_y, L, b, c):
#     """
#     Solves for theta1 and theta2 given X, Y and linkage constants.
#     """
    
#     def equations(vars):
#         t1, t2 = vars
        
#         # 1. Calculate 'a' using the distance formula (Euclidean distance)
#         # a = sqrt((L*sin(t1) - L*sin(t2))^2 + (L*cos(t1) - L*cos(t2))^2)
#         dx = L*np.sin(t1) - L*np.sin(t2)
#         dy = L*np.cos(t1) - L*np.cos(t2)
#         a = np.sqrt(dx**2 + dy**2)
        
#         # 2. Calculate B and C using Law of Cosines
#         # Smallest angles are handled by np.arccos (returns 0 to pi)
#         # Note: Added epsilon to avoid division by zero
#         cos_B = (-b**2 + (a**2 + c**2)) / (2 * a * c + 1e-9)
#         cos_C = (-c**2 + (a**2 + b**2)) / (2 * a * b + 1e-9)
        
#         # Clip values to [-1, 1] to prevent NaN from precision errors
#         B = np.arccos(np.clip(cos_B, -1, 1))
#         C = np.arccos(np.clip(cos_C, -1, 1))
        
#         # 3. Define the two expressions for P(x, y)
#         # We want: Target - Formula = 0
#         eq1_x = -L*np.sin(t1) + L*np.cos(B) - target_x
#         eq1_y =  L*np.cos(t1) + L*np.sin(B) - target_y
        
#         # You have two definitions for P. In a valid system, they must coincide.
#         # We solve primarily for the first definition.
#         return [eq1_x, eq1_y]

#     # Initial guess for theta1, theta2 (in radians)
#     initial_guess = [0.5, 0.5]
    
#     solution = fsolve(equations, initial_guess)
#     return solution

# # Example Usage:
# L, b, c = 10, 5, 5  # Example lengths
# target_x, target_y = 2.5, 12.0

# theta1, theta2 = solve_kinematics(target_x, target_y, L, b, c)

# print(f"Theta 1: {np.degrees(theta1):.2f} degrees")
# print(f"Theta 2: {np.degrees(theta2):.2f} degrees")


import numpy as np
import serial

PORT_TEENSY, BAUD_TEENSY = "COM5", 9600
SERIAL_TIMEOUT = 1


def open_serial_port(port=PORT_TEENSY, baud=BAUD_TEENSY):
    return serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)


def send_thetas(ser, theta1, theta2):
    theta1_degrees = int(round(np.degrees(theta1)))
    theta2_degrees = int(round(np.degrees(theta2)))
    message = f"a{theta1_degrees},b{theta2_degrees}\n"
    ser.write(message.encode("utf-8"))
    ser.flush()


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


X = 1.0
Y = 1.5
L = 5.0
b = L
c = L


if __name__ == "__main__":
    while True:
        sol = solve_theta_from_xy(X, Y, L, b, c)
        if not sol["success"]:
            raise RuntimeError(f"({X}, {Y}) is unreachable for L={L}")
        with open_serial_port() as ser:
            send_thetas(ser, sol["theta1"], sol["theta2"])
            print("sent theta1/theta2 to Arduino")