import numpy as np
import math

def Xrotz(theta: float)->np.ndarray:
    """Xrotz  spatial coordinate transform of Z-axis rotation.
    Xrotz(theta) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, where coordinate frame B is
    rotated by an angle theta (radians) relative to frame A about their
    common Z axis.

    Args:
        theta (float): radians angle value

    Returns:
        np.ndarray: float (6, 6)
    """    
    c = math.cos(theta)
    s = math.sin(theta)
    X = np.array([[c, s, 0.0, 0.0, 0.0, 0.0],
                  [-s, c, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, c, s, 0.0],
                  [0.0, 0.0, 0.0, -s, c, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    return X

if __name__ == "__main__":
    print(Xrotz(math.pi))