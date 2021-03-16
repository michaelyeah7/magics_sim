from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Kinematics.CalcWholeBodyCoM import CalcWholeBodyCoM
import numpy as np
import jax.numpy as jnp
import math
from jaxRBDL.Utils.ModelWrapper import ModelWrapper
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from oct2py import octave

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
OCTAVE_PATH = os.path.join(os.path.dirname(CURRENT_PATH), "octave")
MRBDL_PATH = os.path.join(OCTAVE_PATH, "mRBDL")
MATH_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Math")
MODEL_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Model")
TOOLS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Tools")
KINEMATICS_PATH = os.path.join(OCTAVE_PATH, "mRBDL", "Kinematics")
IPPHYSICIALPARAMS_PATH = os.path.join(OCTAVE_PATH, "ipPhysicalParams") 
IPTOOLS_PATH = os.path.join(OCTAVE_PATH, "ipTools")


octave.addpath(MRBDL_PATH)
octave.addpath(MATH_PATH)
octave.addpath(MODEL_PATH)
octave.addpath(TOOLS_PATH)
octave.addpath(KINEMATICS_PATH)
octave.addpath(IPPHYSICIALPARAMS_PATH)
octave.addpath(IPTOOLS_PATH)
octave.addpath(OCTAVE_PATH)


def plot_robot(plot_points):
    xline,yline,zline = [],[],[]
    for _p in plot_points:
        xline.append(_p[0][0]),yline.append(_p[1][0]),zline.append(_p[2][0])
    fig = plt.figure()
    ax =  Axes3D(fig)
    ax.plot(np.asarray(xline[:]), np.asarray(yline[:]), np.asarray(zline[:]), 'green',linewidth=7.5)
    plt.show()


#load urdf
from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
model  = UrdfWrapper("/root/RBDL/urdf/arm.urdf").model
model["jtype"] = np.asarray(model["jtype"])
model["parent"] = np.asarray(model["parent"])
urdf_path = model['urdf_path']
jname = model['jname']
print(model['NB'])
print(len(model['Xtree']))
print(model['jaxis'])
print(model['jname'])


#pybullet render
from jaxRBDL.Utils.PybulletRender import PybulletRender
pyrb = PybulletRender(urdf_path,jname)

#pyrbdl calculation
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates


#q should be between (pi,-pi)
q = 2*  np.array([  0.0, 0.0, math.pi/6, math.pi/6, -math.pi/3, 0.0,0.0])

for k in range(10):
    q = np.array([ 0.0, 0.0, np.random.uniform(-math.pi/2,math.pi/2),  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
        np.random.uniform(-math.pi/2,math.pi/2),0.0])
    # q = np.array([1.0,1.0])
    print(q)
    output_points = []
    for i in range(1, int(model['NB']) + 1):
        input = (model, q, i, np.zeros((3,)))
        py_output = CalcBodyToBaseCoordinates(*input)
        output_points.append(py_output)
        print(py_output)
    pyrb.step(q)
    plot_robot(output_points)
