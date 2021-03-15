from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
import time
import numpy as np
import pybullet as p

tau = 0.1
# model = UrdfWrapper("urdf/laikago_toes.urdf").model
# model = UrdfWrapper("urdf/arm.urdf").model
# model = UrdfWrapper("urdf/quadrupedal.urdf").model
# print("model",model)
# osim = ObdlSim(model,dt=tau,vis=True)
# q = np.array([0,0,0])
# while True:
#     osim.step_theta(q)
#     time.sleep(100)

p.connect(p.GUI)
# test_robot = p.loadURDF("urdf/arm.urdf",[0,0,0])
while True:
    time.sleep(100)
