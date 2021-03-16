from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
import time
import numpy as np
import pybullet as p

tau = 0.1
model = UrdfWrapper("urdf/laikago/laikago.urdf").model
# model = UrdfWrapper("urdf/arm.urdf").model
# model = UrdfWrapper("urdf/quadrupedal.urdf").model
# print("model",model)

# osim = ObdlSim(model,dt=tau,vis=True)
# q = np.array([0,0,0])

# while True:
#     # osim.step_theta(q)
#     time.sleep(100)

p.connect(p.GUI)
# test_robot = p.loadURDF("urdf/quadrupedal.urdf",[0,0,0])
test_robot = p.loadURDF("urdf/laikago/laikago.urdf",[0,0,.5],[0,0.5,0.5,0])
while True:
    time.sleep(100)


# model  = UrdfWrapper("/root/RBDL/urdf/laikago/laikago.urdf").model
# model  = UrdfWrapper("/root/RBDL/urdf/arm.urdf").model



# model["jtype"] = np.asarray(model["jtype"])
# model["parent"] = np.asarray(model["parent"])

# contact_foot = [13,10,7,4]

# print("jaxis:",model['jaxis'])
# print("jtype:",model['jtype'])

# rder = ObdlRender(model)
# time.sleep(1)
# q = [0.0] * 14
# q[1] = 1.57
# rder.step_render(q)

# num_b = model['jtype'].shape[0] + 1
# for i in range(num_b):
#     rder.p.setTimeStep(0.1)
#     rder.p.stepSimulation()
#     info = rder.p.getContactPoints(i)
#     # info = rder.p.getContactPoints(0,i)
#     print(info)

# while(True):
#     time.sleep(0.5)
