from typing import Optional

from numpy.core.numeric import flatnonzero
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np


def PlotContactForce(model: dict, q: np.ndarray, fc: Optional[np.ndarray] , fcqp: Optional[np.ndarray], fcpd: Optional[np.ndarray], fwho: str, ax: Axes3D):
    if fc is None and fcqp is None and fcpd is None: 
        print("No contact force to plot!")
        return
    
    fplot = None
    if fwho == "fc":
        fplot = fc.flatten()
    if fwho == "fcqp":
        fplot = fcqp.flatten()
    if fwho == "fcpd":
        fplot = fcpd.flatten()
    if fplot is None:
        print("No contact force available!")
        return


    NC = int(model["NC"])

    try: 
        idcontact = np.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = np.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    fshrink = 100
    nc = max(idcontact.shape)
    pos_contact_list = []
    fplot = fplot.reshape((3, nc), order="F")
    for i in range(nc):
        pos_contact_list.append(CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i]))

    pos_contact = np.asfarray(np.concatenate(pos_contact_list, axis=1))
    # print(fplot)
    ax.quiver(pos_contact[0, :], pos_contact[1, :], pos_contact[2, :], fplot[0, :]/100.0, fplot[1, :]/100.0, fplot[2,:]/100.0)



    
    


    