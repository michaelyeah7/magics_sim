import numpy as np
import math

def Xroty(theta: float)->np.ndarray:
    """Xroty  spatial coordinate transform for Y-axis rotation. 
    Xroty(theta) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, where coordinate frame B is
    rotated by an angle theta (radians) relative to frame A about their
    common Y axis.

    Args:
        theta (float): radians angle value

    Returns:
        np.ndarray: float (6, 6)
    """    
    c = math.cos(theta)
    s = math.sin(theta)
    X = np.array([[c, 0.0, -s, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [s, 0.0, c, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, c, 0.0, -s],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, s, 0.0, c]])
    return X

if __name__ == "__main__":
    print(Xroty(math.pi))