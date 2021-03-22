import numpy as np

def TransformToPosition(X: np.ndarray)->np.ndarray:
    """TransformToPosition extracts traslation vector (3D)
    from general sptial trasnformation (6D*6D) matrix.

    Args:
        X (np.ndarray): float (6, 6)

    Returns:
        np.ndarray: float (3, 1)
    """    
    E = X[0:3, 0:3]
    rx = -np.matmul(E.transpose(),X[3:6, 0:3])
    r = np.array([-rx[1, 2], rx[0, 2], -rx[0, 1]]).reshape(3, 1)
    return r

if __name__ == "__main__":
    print(TransformToPosition(np.random.randn(*(6, 6))))