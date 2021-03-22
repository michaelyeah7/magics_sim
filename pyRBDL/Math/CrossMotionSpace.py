import numpy as np

def CrossMotionSpace(v: np.ndarray) -> np.ndarray:
    """ CrossMotionSpace spatial motion cross-product operator.
    CrossMotionSpace(v) calculates the 6x6 matrix such that the expression CrossMotionSpace(v)*m is the 
    cross product of the spatial motion vectors v and m.

    Args:
        v (np.ndarray): float (6, 1) or (1, 6) or (6,)

    Returns:
        np.ndarray: float (6, 6)
    """
    v = v.flatten()
    vcross = np.array([[0.0, -v[2], v[1], 0.0, 0.0, 0.0],
                       [v[2], 0.0, -v[0], 0.0, 0.0, 0.0],
                       [-v[1], v[0], 0.0, 0.0, 0.0, 0.0],
                       [0.0, -v[5], v[4], 0.0, -v[2], v[1]],
                       [v[5], 0.0, -v[3], v[2], 0.0, -v[0]],
                       [-v[4], v[3], 0.0, -v[1], v[0], 0.0]])
    return vcross


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,])
    print(CrossMotionSpace(a))


