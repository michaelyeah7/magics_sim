from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Tools.CalcInertiaCuboid import ClacInertiaCuboid
from pyRBDL.Tools.PlotInertiaCuboid import PlotInertiaCuboid

def PlotCoMInertia(model: dict, q: np.ndarray, ax: Axes3D):


    try:
  
        idcomplot = np.squeeze(model["idcomplot"], axis=0)
        Inertia = np.squeeze(model["Inertia"], axis=0)
        idcomplot = idcomplot.astype(int)
        CoM = np.squeeze(model["CoM"], axis=0)
        Mass = np.squeeze(model["Mass"], axis=0)

    except:
        idlinkplot = model["idlinkplot"]
        idcomplot = model["idcomplot"]
        idcontact = model["idcontact"]
        CoM = model["CoM"]
        Inertia = model["Inertia"]
        Mass = model["Mass"]
        linkplot = model["linkplot"]
        contactpoint = model["contactpoint"]

    num = np.max(idcomplot.shape)
  

    pos_com = []
    for i in range(num):
        pos_com.append(CalcBodyToBaseCoordinates(model, q, idcomplot[i], CoM[i]))
        
    pos_com = np.asfarray(np.concatenate(pos_com, axis=1))
    ax.scatter(pos_com[0,:], pos_com[1, :], pos_com[2, :], marker="*")

    # print(num)
    for i in range(num):
        lxyz  = ClacInertiaCuboid(np.diag(Inertia[i]), Mass[i])
        PlotInertiaCuboid(pos_com[:, i:i+1], lxyz, ax)

    return ax