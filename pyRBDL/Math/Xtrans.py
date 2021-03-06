import numpy as np

def Xtrans(r: np.ndarray)->np.ndarray:
    """Xtrans  spatial coordinate transform for the translation of origin.
    Xtrans(r) calculates the coordinate transform matrix from A to B
    coordinates for spatial motion vectors, in which frame B is translated by
    an amount r (3D vector) relative to frame A.

    Args:
        r (np.ndarray): float (3, 1) or (1, 3) or (3,)

    Returns:
        np.ndarray: float (6, 6)
    """    
    r = r.flatten()
    X = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
         [0.0, r[2], -r[1], 1.0, 0.0, 0.0],
         [-r[2], 0.0, r[0], 0.0,  1.0,  0.0],
         [r[1], -r[0], 0.0, 0.0, 0.0, 1.0]])
    return X

if __name__ == "__main__":
    a = np.array([1.0, 2.0, 3.0])
    print(Xtrans(a))

