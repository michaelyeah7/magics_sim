import numpy as np

def CrossMatrix(v: np.ndarray) -> np.ndarray:
    """[summary]

    Args:
        v (np.ndarray): float (3, 1) or (1, 3) or (3,)

    Returns:
        np.ndarray: float (3, 3)
    """    
    v = v.flatten()
    CroMat = np.array(
        [[0.0, -v[2], v[1]],
         [v[2], 0.0, -v[0]],
         [-v[1], v[0], 0.0]])
    return CroMat

if __name__ == "__main__":
    vector = np.array([1.0, 2.0, 3.0])
    print(CrossMatrix(vector))

