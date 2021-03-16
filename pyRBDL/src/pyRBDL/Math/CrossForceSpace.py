import numpy as np
from pyRBDL.Math.CrossMotionSpace import CrossMotionSpace

def CrossForceSpace(v: np.ndarray)->np.ndarray:
    """CrossForceSpac spatial force cross-product operator.
    CrossForceSpac(v) calculates the 6x6 matrix such that the expression CrossForceSpac(v)*f is the
    cross product of the spatial motion vector v with the spatial force vector f.

    Args:
        v (np.ndarray): float (6, 1) or (1, 6) or (6,)

    Returns:
        np.ndarray: float (6, 6)
    """    
    vcross = -CrossMotionSpace(v).transpose()
    return vcross


if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0,])
    print(CrossForceSpace(a))

