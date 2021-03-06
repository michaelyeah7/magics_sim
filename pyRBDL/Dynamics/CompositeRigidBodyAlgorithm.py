import numpy as np
from pyRBDL.Model.JointModel import JointModel

def CompositeRigidBodyAlgorithm(model: dict, q: np.ndarray)->np.ndarray:
    q = q.flatten()
    NB = int(model["NB"])
    H = np.zeros((NB, NB))
    jtype = model["jtype"].flatten()
    jaxis = model['jaxis']
    parent = model['parent'].flatten().astype(int)
    try:
        Xtree = np.squeeze(model['Xtree'], axis=0)
        IC = np.squeeze(model['I'], axis=0).copy()
    except:
        Xtree = model['Xtree']
        IC = model['I'].copy()

    S = []
    Xup = []

    for i in range(NB):
        XJ, Si = JointModel(jtype[i], jaxis[i], q[i])
        S.append(Si)
        Xup.append(np.matmul(XJ, Xtree[i]))

    for j in range(NB-1, -1, -1):
        if parent[j] != 0:
            IC[parent[j] - 1] = IC[parent[j] - 1] + np.matmul(np.matmul(Xup[j].transpose(), IC[j]), Xup[j])


    for i in range(NB):
        fh = np.matmul(IC[i],  S[i])
        H[i, i] = np.matmul(S[i].transpose(), fh)
        j = i
        while parent[j] > 0:
            fh = np.matmul(Xup[j].transpose(), fh)
            j = parent[j] - 1
            H[i, j] = np.matmul(S[j].transpose(), fh)
            H[j, i] = H[i, j]

    return H

    



