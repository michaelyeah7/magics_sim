import numpy as np
from numpy.matrixlib.defmatrix import matrix
from pyRBDL.Model.JointModel import JointModel
from pyRBDL.Math.Xtrans import Xtrans
from pyRBDL.Math.InverseMotionSpace import InverseMotionSpace

def CalcPointJacobian(model: dict, q: np.ndarray, body_id: int, point_pos: np.ndarray)->np.ndarray:
    q = q.flatten()
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

    for i in range(body_id):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        Xup.append(np.matmul(XJ, Xtree[i]))
        if parent[i] == 0:
            X0.append(Xup[i])
        else:
            X0.append(np.matmul(Xup[i], X0[parent[i]-1]))

    XT_point = Xtrans(point_pos)
    X0_point = np.matmul(XT_point, X0[body_id-1])

    j_p = body_id - 1
    BJ = np.zeros((6, NB))
    while j_p != -1:
        Xe = np.matmul(X0_point, InverseMotionSpace(X0[j_p]))
        BJ[:, [j_p, ]] = np.matmul(Xe, S[j_p])
        j_p = parent[j_p] - 1

    E0 = X0_point[0:3, 0:3].transpose()
    J = np.matmul(np.matmul(E0, np.hstack([ np.zeros((3, 3)), np.eye(3)])), BJ)
    return J
