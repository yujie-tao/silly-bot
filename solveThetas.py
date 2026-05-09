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
from scipy.optimize import least_squares
import serial

PORT_TEENSY, BAUD_TEENSY = "COM3", 115200
SERIAL_TIMEOUT = 1


def open_serial_port(port=PORT_TEENSY, baud=BAUD_TEENSY):
    return serial.Serial(port, baud, timeout=SERIAL_TIMEOUT)


def send_thetas(ser, theta1, theta2):
    theta1_degrees = int(round(np.degrees(theta1)))
    theta2_degrees = int(round(np.degrees(theta2)))
    message = f"a{theta1_degrees},b{theta2_degrees}\n"
    ser.write(message.encode("utf-8"))
    ser.flush()


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
                    "theta1": sol.x[0],
                    "theta2": sol.x[1],
                    "error": err,
                    "success": sol.success
                }

    return best


X = 1.0
Y = 1.5
L = 5.0
b = L
c = L

sol = solve_theta_from_xy(X, Y, L, b, c)

print("theta1 =", sol["theta1"])
print("theta2 =", sol["theta2"])
print("error  =", sol["error"])


if __name__ == "__main__":
    if sol is None:
        raise RuntimeError("No theta solution found")

    with open_serial_port() as ser:
        send_thetas(ser, sol["theta1"], sol["theta2"])
        print("sent theta1/theta2 to Arduino")