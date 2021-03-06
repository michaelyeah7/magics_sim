from pyRBDL.Math.Xrotx import Xrotx
from pyRBDL.Math.Xroty import Xroty
from pyRBDL.Math.Xrotz import Xrotz
from pyRBDL.Math.Xtrans import Xtrans
from typing import Tuple
import numpy as np

def JointModel(jtype: int, jaxis: str, q: float)->Tuple[np.ndarray, np.ndarray]:
    """JointModel  Calculate joint transform and motion subspace.
    Xj, S = JointModel(pitch, q) calculates the joint transform and motion subspace 
    matrices for a revolute (pitch==0), prismatic (pitch==inf).
    For revolute joints, q is the joint angle.
    For prismatic joints, q is the linear displacement.


    Args:
        jtype (int):  0 for revolute joint, 1 for prismatic joint
        jaxis (str): 'x' or 'y' or 'z' axis 
        q (float): joint angle or linear displacement

    Returns:
        Tuple[np.ndarray, np.ndarray]: spatial transform and motion subspace
    """
    Xj = np.array([])
    S = np.array([])
    if jtype == 0:
        # revolute joint
        if jaxis == 'x':
            Xj = Xrotx(q)
            S = np.array([1, 0, 0, 0, 0, 0]).reshape(6, 1)
        if jaxis == 'y':
            Xj = Xroty(q)
            S = np.array([0, 1, 0, 0, 0, 0]).reshape(6, 1)
        if jaxis == 'z':
            Xj = Xrotz(q)
            S = np.array([0, 0, 1, 0, 0, 0]).reshape(6, 1)
    if jtype == 1:
        # prismatic joint
        if jaxis == 'x':
            Xj = Xtrans(np.array([q, 0, 0]))
            S = np.array([0, 0, 0, 1, 0, 0]).reshape(6, 1)
        if jaxis == 'y':
            Xj = Xtrans(np.array([0, q, 0]))
            S = np.array([0, 0, 0, 0, 1, 0]).reshape(6, 1)
        if jaxis == 'z':
            Xj = Xtrans(np.array([0, 0, q]))
            S = np.array([0, 0, 0, 0, 0, 1]).reshape(6, 1)
    
    return (Xj, S)

