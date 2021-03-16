import os
import numpy as np
import math
from pyRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from pyRBDL.Dynamics.InverseDynamics import InverseDynamics

from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def plot_robot(plot_points):
    xline,yline,zline = [],[],[]
    for _p in plot_points:
        xline.append(_p[0][0]),yline.append(_p[1][0]),zline.append(_p[2][0])
    fig = plt.figure()
    ax =  Axes3D(fig)
    ax.plot(np.asarray(xline[:]), np.asarray(yline[:]), np.asarray(zline[:]), 'green',linewidth=7.5)
    plt.show()

from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
def plot_dynamics(model,q):
    output_points = []
    for i in range(1, int(model['NB']) + 1):
        input = (model, q, i, np.zeros((3,)))
        py_output = CalcBodyToBaseCoordinates(*input)
        output_points.append(py_output)
        print(py_output)
    plot_robot(output_points)


class TestDynamics():
    def __init__(self):
        model = dict()
        self.model  = UrdfWrapper("/root/RBDL/urdf/cartpole.urdf").model
        self.model["jtype"] = np.asarray(self.model["jtype"])
        self.model["parent"] = np.asarray(self.model["parent"])
        self.jnum = self.model['NB']
        self.q = np.array([0.0,  0.4765])
        self.qdot = np.array([1.0, 1.0])
        self.qddot = np.array([1.0, 1.0])
        self.tau = np.array([10.0, 1.0])

    def test_CompositeRigidBodyAlgorithm(self):
        input = (self.model, self.q * np.random.randn(*(self.jnum, )))
        py_output = CompositeRigidBodyAlgorithm(*input)

    def test_ForwardDynamics(self):
        q =  self.q * np.random.randn(*(self.jnum, ))
        qdot =  self.qdot * np.random.randn(*(self.jnum, ))
        tau = self.tau * np.random.randn(*(self.jnum, ))
        input = (self.model, q, qdot, tau)
        qddot_hat = ForwardDynamics(*input).flatten() #qddot
        print("applied torque",tau,"qddot_hat",qddot_hat)

        q_hat,qdot_hat = self.calculate_q(dt=1.0,q=q,qdot=qdot,qddot=qddot_hat)
        plot_dynamics(self.model,q_hat)

        return q_hat,qdot_hat,tau

    def test_InverseDynamics(self):
        q = self.q * np.random.randn(*(self.jnum, ))
        qdot = self.qdot * np.random.randn(*(self.jnum, ))
        qddot = self.qddot * np.random.randn(*(self.jnum, ))
        input = (self.model, q, qdot, qddot)
        py_output = InverseDynamics(*input)
        print(py_output)

    def calculate_q(self,dt,q,qdot,qddot):
        qdot = qdot + qddot * dt
        q = q + qdot * dt
        return q,qdot

    

if __name__ == "__main__":
    rb_dyn = TestDynamics()
    q_hat,qdot_hat,torque = rb_dyn.test_ForwardDynamics()
    print("rbdl pos",q_hat)

    urdf_path = rb_dyn.model['urdf_path']
    jname = rb_dyn.model['jname']
    grav = rb_dyn.model['a_grav'][-1]   
    from jaxRBDL.Utils.PybulletRender import PybulletRender
    pyrb = PybulletRender(urdf_path,jname,grav=grav,dt=1.0)
    pyrb.step_torque(torque)
    pyrb.get_joints()
    ##TODO step only once
    print("apllied torque:",torque,grav)
    print(pyrb.get_joints())
    while(True):
        continue
        # pyrb.step_torque([0.0,0.0])
        # pyrb.step_debuger()
        # print(pyrb.jointIds)
