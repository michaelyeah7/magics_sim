import numpy as np
from numpy.lib.shape_base import column_stack
from pyRBDL.Math.CrossMatrix import CrossMatrix

def SpatialTransform(E: np.ndarray, r:np.ndarray)->np.ndarray:
    """Xtrans  spatial coordinate transform for the translation  and rotation of origin.
    SpatialTransform(E, r) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, in which frame B is translated by
    an amount r (3D vector) relative to frame A and then rotated by E (3D * 3D rotation matrix) relative to the current frame.

    Args:
        E (np.ndarray): float (3, 3)
        r (np.ndarray): float (3, 1) or (1, 3) or (3,)

    Returns:
        np.ndarray: float (6, 6)
    """    
    r = r.reshape(*(3, 1))
    X_T = np.vstack([np.hstack([E, np.zeros((3, 3))]) , np.hstack([np.matmul(-E, CrossMatrix(r)), E])])
    return np.asfarray(X_T)


if __name__ == "__main__":
    E = np.random.randn(*(3, 3))
    r = np.random.randn(*(3,))
    print(SpatialTransform(E, r))
    