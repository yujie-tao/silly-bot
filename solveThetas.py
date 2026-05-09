import numpy as np
from scipy.optimize import fsolve

def solve_kinematics(target_x, target_y, L, b, c):
    """
    Solves for theta1 and theta2 given X, Y and linkage constants.
    """
    
    def equations(vars):
        t1, t2 = vars
        
        # 1. Calculate 'a' using the distance formula (Euclidean distance)
        # a = sqrt((L*sin(t1) - L*sin(t2))^2 + (L*cos(t1) - L*cos(t2))^2)
        dx = L*np.sin(t1) - L*np.sin(t2)
        dy = L*np.cos(t1) - L*np.cos(t2)
        a = np.sqrt(dx**2 + dy**2)
        
        # 2. Calculate B and C using Law of Cosines
        # Smallest angles are handled by np.arccos (returns 0 to pi)
        # Note: Added epsilon to avoid division by zero
        cos_B = (-b**2 + (a**2 + c**2)) / (2 * a * c + 1e-9)
        cos_C = (-c**2 + (a**2 + b**2)) / (2 * a * b + 1e-9)
        
        # Clip values to [-1, 1] to prevent NaN from precision errors
        B = np.arccos(np.clip(cos_B, -1, 1))
        C = np.arccos(np.clip(cos_C, -1, 1))
        
        # 3. Define the two expressions for P(x, y)
        # We want: Target - Formula = 0
        eq1_x = -L*np.sin(t1) + L*np.cos(B) - target_x
        eq1_y =  L*np.cos(t1) + L*np.sin(B) - target_y
        
        # You have two definitions for P. In a valid system, they must coincide.
        # We solve primarily for the first definition.
        return [eq1_x, eq1_y]

    # Initial guess for theta1, theta2 (in radians)
    initial_guess = [0.5, 0.5]
    
    solution = fsolve(equations, initial_guess)
    return solution

# Example Usage:
L, b, c = 10, 5, 5  # Example lengths
target_x, target_y = 2.5, 12.0

theta1, theta2 = solve_kinematics(target_x, target_y, L, b, c)

print(f"Theta 1: {np.degrees(theta1):.2f} degrees")
print(f"Theta 2: {np.degrees(theta2):.2f} degrees")