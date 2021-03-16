from typing import Pattern
import numpy as np
from pyRBDL.Model.JointModel import JointModel
from pyRBDL.Math.Xtrans import Xtrans
from pyRBDL.Kinematics.TransformToPosition import TransformToPosition

def CalcBodyToBaseCoordinates(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype'].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)
    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
    except:
        Xtree = model['Xtree']
    X0 = [] 

    for i in range(body_id):
        XJ, _ = JointModel(jtype[i], jaxis[i], q[i])
        Xup = np.matmul(XJ, Xtree[i])
        if parent[i] == 0:
            X0.append(Xup)
        else:
            X0.append(np.matmul(Xup, X0[parent[i] - 1]))
    
    XT_point = Xtrans(point_pos)
    X0_point =  np.matmul(XT_point, X0[body_id - 1])
    pos = TransformToPosition(X0_point)

    return pos
