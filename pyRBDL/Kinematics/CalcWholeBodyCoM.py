import numpy as np
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates


def CalcWholeBodyCoM(model: dict, q: np.ndarray)-> np.ndarray:
    """CalcWholeBodyCoM - Calculate whole body's CoM position in world frame

    Args:
        model (dict): dictionary of model specification
        q (np.ndarray): an array of joint position

    Returns:
        np.ndarray: float (3, 3)
    """  
    q = q.flatten()
    idcomplot = model["idcomplot"].flatten().astype(int)
    try:
        CoM = np.squeeze(model["CoM"], axis=0)
        Mass = np.squeeze(model["Mass"], axis=0)
    except:
        CoM = model["CoM"]
        Mass = model["Mass"]
    
    
    num = np.max(idcomplot.shape)
    CoM_list = []
    Clink = np.zeros((3,1))
    for i in range(num):
        Clink = CalcBodyToBaseCoordinates(model, q, idcomplot[i], CoM[i])
        CoM_list.append(Clink)
    
    C = np.zeros((3, 1))
    M = 0

    for i in range(num):
        C = C + np.multiply(CoM_list[i], Mass[i])
        M = M + Mass[i]

    Pcom = np.asfarray(np.divide(C, M))
    
    return Pcom

