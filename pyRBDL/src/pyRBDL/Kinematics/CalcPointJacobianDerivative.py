import numpy as np
from pyRBDL.Model.JointModel import JointModel
from pyRBDL.Math.Xtrans import Xtrans
from pyRBDL.Math.CrossMotionSpace import CrossMotionSpace
from pyRBDL.Math.InverseMotionSpace import InverseMotionSpace

def CalcPointJacobianDerivative(model: dict, q: np.ndarray, qdot: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
    qdot = qdot.flatten()
    point_pos = point_pos.flatten()
    jtype = model['jtype'].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)
    NB = int(model["NB"])
    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
    except:
        Xtree = model['Xtree']

    S = []
    Xup = []
    X0 = []
    v = []

    for i in range(body_id):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        vJ = S[i] * qdot[i]
        Xup.append(np.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            v.append(vJ)
            X0.append(Xup[i])
        else:
            v.append(np.add(np.matmul(Xup[i], v[parent[i]-1]), vJ))
            X0.append(np.matmul(Xup[i], X0[parent[i]-1]))

    XT_point = Xtrans(point_pos)
    X0_point = np.matmul(XT_point, X0[body_id-1])
    v_point = np.matmul(XT_point, v[body_id-1])

    BJ = np.zeros((6, NB))
    dBJ = np.zeros((6, NB))
    id_p = id =  body_id - 1
    Xe = np.zeros((NB, 6, 6))

    while id_p != -1:
        if id_p == body_id - 1:
            Xe[id_p,...] = np.matmul(XT_point, Xup[id_p])
            BJ[:,[id_p,]] = np.matmul(XT_point, S[id_p])
            dBJ[:,[id_p,]] = np.matmul(np.matmul(CrossMotionSpace(np.matmul(XT_point, v[id_p]) - v_point), XT_point), S[id_p])
        else:
            Xe[id_p,...] = np.matmul(Xe[id, ...], Xup[id_p])
            BJ[:,[id_p,]] = np.matmul(Xe[id, ...], S[id_p])
            dBJ[:,[id_p,]] = np.matmul(np.matmul(CrossMotionSpace(np.matmul(Xe[id, ...], v[id_p]) - v_point), Xe[id,...]), S[id_p])        
        id = id_p
        id_p = parent[id] - 1
    X0 = InverseMotionSpace(X0_point)
    E0 = np.asfarray(np.vstack([np.hstack([X0[0:3,0:3], np.zeros((3, 3))]), np.hstack([np.zeros((3, 3)), X0[0:3, 0:3]])]))
    dE0 = np.matmul(CrossMotionSpace(np.matmul(X0,v_point)), E0)
    E0 = E0[0:3, 0:3]
    dE0 = dE0[0:3,0:3]
    JDot = np.matmul(np.matmul(dE0, np.hstack([np.zeros((3,3)), np.eye(3)])), BJ) \
        + np.matmul(np.matmul(E0, np.hstack([np.zeros((3,3)), np.eye(3)])), dBJ)


    return JDot
