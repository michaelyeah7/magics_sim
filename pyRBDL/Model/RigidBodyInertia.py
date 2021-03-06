import numpy as np

def RigidBodyInertia(m: float, c: np.ndarray, I: np.ndarray)->np.ndarray:
    """RigidBodyInertia  spatial rigid-body inertia from mass about rotating point, CoM and rotational inertia.
    RigidBodyInertia(m,c,I) calculates the spatial inertia matrix of a rigid body from its 
    mass, centre of mass (3D vector) and rotational inertia (3x3 matrix) about its centre of mass.

    Args:
        m (float): mass
        c (np.ndarray): float (3, 1) or (1, 3) or (3,) translation for center of mass
        I (np.ndarray): rotational inertia about its center of mass

    Returns:
        np.ndarray: float (6, 6) spatial inertia of rigid-body
    """

    c = c.flatten()
    C = np.array(
        [[0.0, -c[2], c[1]],
         [c[2], 0.0, -c[0]],
         [-c[1], c[0], 0.0]])
    rbi = np.asfarray(np.vstack(
        [np.hstack([I + m * np.matmul(C, C.transpose()), m * C]),
         np.hstack([m * C.transpose(), m * np.eye(3)])]))

    return rbi

if __name__ == "__main__":
    m = 1.0
    c = np.array([1.0, 0.0, 0.0])
    I = np.random.randn(*(3, 3))
    print(RigidBodyInertia(m, c, I))