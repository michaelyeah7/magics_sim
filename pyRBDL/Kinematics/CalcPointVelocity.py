import numpy as np
from pyRBDL.Model.JointModel import JointModel
from pyRBDL.Math.Xtrans import Xtrans

def CalcPointVelocity(model: dict, q: np.ndarray, qdot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype'].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)
    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
    except:
        Xtree = model['Xtree']

    X0 = []
    Xup = []

    S = []
    v = []

    for i in range(body_id):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = S[i] * qdot[i]
        Xup = np.matmul(XJ, Xtree[i])
        if parent[i] == 0:
            v.append(vJ)
            X0.append(Xup)
        else:
            v.append(np.matmul(Xup, v[parent[i] - 1]) + vJ)
            X0.append(np.matmul(Xup, X0[parent[i] - 1]))
    
    XT_point = Xtrans(point_pos)
    X0_point = np.matmul(XT_point,  X0[body_id-1])
    vel_spatial = np.matmul(XT_point, v[body_id-1])
    vel = np.matmul(X0_point[0:3,0:3].transpose(), vel_spatial[3:6])

    return vel