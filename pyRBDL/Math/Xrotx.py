import numpy as np
import math

def Xrotx(theta: float)->np.ndarray:
    """Xrotx  spatial coordinate transform for X-axis rotation).
    Xrotx(theta) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, where coordinate frame B is
    rotated by an angle theta (radians) relative to frame A about their
    common X axis.

    Args:
        theta (float): radians angle value

    Returns:
        np.ndarray: float (6, 6)
    """    
    c = math.cos(theta)
    s = math.sin(theta)
    X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, c, s, 0.0, 0.0, 0.0],
         [0.0, -s, c, 0.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
         [0.0, 0.0, 0.0, 0.0, c, s],
         [0.0, 0.0, 0.0, 0.0, -s, c]])
    return X

if __name__ == "__main__":
    print(Xrotx(math.pi))