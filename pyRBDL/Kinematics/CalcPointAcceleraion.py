import numpy as np
from numpy.lib.function_base import append
from pyRBDL.Math.CrossMotionSpace import CrossMotionSpace
from pyRBDL.Math.Xtrans import Xtrans
from pyRBDL.Model.JointModel import JointModel

def CalcPointAcceleration(model: dict, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    qddot = qddot.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype'].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)
    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
    except:
        Xtree = model['Xtree']

    Xup = []
    v = []
    avp = []
    X0 = []

    for i in range(body_id):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        vJ = Si * qdot[i]
        Xup.append(np.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            avp.append(np.matmul(Xup[i], np.zeros((6, 1))))
            X0.append(Xup[i])
        else:
            v.append(np.matmul(Xup[i], v[parent[i]-1]) + vJ)
            avp.append(np.matmul(Xup[i], avp[parent[i] - 1]) + Si * qddot[i] + np.matmul(CrossMotionSpace(v[i]), vJ))
            X0.append(np.matmul(Xup[i], X0[parent[i]-1]))

    E_point = X0[body_id-1][0:3,0:3]
    XT_point = Xtrans(point_pos)
    vel_p = np.matmul(XT_point, v[body_id-1])
    avp_p = np.matmul(XT_point, avp[body_id-1])
    
    acc = np.matmul(E_point.transpose(), avp_p[3:6]) + \
        np.cross(np.matmul(E_point.transpose(), vel_p[0:3]).squeeze(), np.matmul(E_point.transpose(), vel_p[3:6]).squeeze()).reshape(3, 1)

    return acc