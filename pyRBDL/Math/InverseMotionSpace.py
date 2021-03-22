import numpy as np

def InverseMotionSpace(X: np.ndarray)->np.ndarray:
    """[summary]

    Args:
        X (np.ndarray): float (6, 6)

    Returns:
        np.ndarray: float (6, 6)
    """    
    E = X[0:3, 0:3]
    r = X[3:6, 0:3]
    Xinv = np.asfarray(np.vstack([np.hstack([E.transpose(), np.zeros((3, 3))]),
    np.hstack([r.transpose(), E.transpose()])]))
    return Xinv

if __name__ == "__main__":
    X = np.random.randn(*(6, 6))
    print(InverseMotionSpace(X))