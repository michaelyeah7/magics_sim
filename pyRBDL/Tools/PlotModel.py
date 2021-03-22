from matplotlib.pyplot import axis
from mpl_toolkits.mplot3d.axes3d import Axes3D
from numpy.testing._private.utils import import_nose
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Tools.PlotLink import PlotLink

def PlotModel(model: dict, q: np.ndarray, ax: Axes3D):
    try:
        idlinkplot = np.squeeze(model["idlinkplot"], axis=0).astype(int)
        linkplot = np.squeeze(model["linkplot"], axis=0)
        idcontact = np.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = np.squeeze(model["contactpoint"], axis=0)
    except:
        idlinkplot = model["idlinkplot"]
        linkplot = model["linkplot"]
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    pos_o = []
    pos_e = []

    num = np.max(idlinkplot.shape)
    for i in range(num):
        pos_o.append(CalcBodyToBaseCoordinates(model, q, idlinkplot[i], np.zeros((3,1))))
        pos_e.append(CalcBodyToBaseCoordinates(model, q, idlinkplot[i], linkplot[i]))

    pos_o = np.concatenate(pos_o, axis=1)
    pos_e = np.concatenate(pos_e, axis=1)

    nc = np.max(idcontact.shape)

    pos_contact = []
    for i in range(nc):
        pos_contact.append(CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i]))
    pos_contact = np.concatenate(pos_contact, axis=1)
    
    ax = PlotLink(pos_o, pos_e, 6, pos_contact, ax)
    return ax

