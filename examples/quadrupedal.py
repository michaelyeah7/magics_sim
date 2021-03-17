from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
import time
import numpy as np
import pybullet as p
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics

def arm_test():
    dt = 0.001
    model = UrdfWrapper("urdf/arm.urdf").model
    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])
    rder = ObdlRender(model)
    time.sleep(1)
    q = np.array([0.0] * 7)
    q[3] = 1.57
    qdot = np.array([0.0] * 7)
    torque = np.array([0.0] * 7)
    torque[5] = 0.001

    for i in range(1000):
        q[3] = 1.57
        input = (model, q, qdot, torque)
        accelerations = ForwardDynamics(*input)
        

        # print("shape acc",accelerations.shape())
        accelerations = accelerations.flatten()
        print("accelerations",accelerations)

        #step one forward
        
        for j in range(2,7):
            q[j] = q[j] + dt * qdot[j]
            qdot[j] = qdot[j] + dt * accelerations[j]
        # print("qdot",qdot)
        print("q",q)

        rder.step_render(q)

def laikago_trajectory():

    model = UrdfWrapper("urdf/laikago/laikago.urdf").model

    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])
    rder = ObdlRender(model)
    with open("examples/data1.txt","r") as filestream:
        for line in filestream:
            currentline = line.split(",")
            frame = currentline[0]
            t = currentline[1]
            joints=[float(x) for x in currentline[2:14] ]
            joints.insert(0,1.57)
            joints.insert(0,0)

            rder.step_render(joints)        

            # time.sleep(1./500.)
    print("loop is over")


def laikago_dynamics_test():
    tau = 0.1
    model = UrdfWrapper("urdf/laikago/laikago.urdf").model

    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])
    # model = UrdfWrapper("urdf/quadrupedal.urdf").model
    # print("model",model)

    # osim = ObdlSim(model,dt=tau,vis=True)
    # while True:
    #     time.sleep(100)

    # p.connect(p.GUI)
    # test_robot = p.loadURDF("urdf/quadrupedal.urdf",[0,0,0])
    # test_robot = p.loadURDF("urdf/laikago/laikago.urdf",[0,0,.5],[0,0.5,0.5,0])
    # while True:
    #     time.sleep(100)


    # contact_foot = [13,10,7,4]

    # print("jaxis:",model['jaxis'])
    # print("jtype:",model['jtype'])

    rder = ObdlRender(model)
    time.sleep(1)
    q = np.array([0.0] * 14)
    qdot = np.array([0.0] * 14)
    torque = np.array([0.0] * 14)
    q[1] = 1.57
    rder.step_render(q)
    dt = 0.001
    # num_b = model['jtype'].shape[0] + 1
    # for i in range(num_b):
    #     rder.p.setTimeStep(0.1)
    #     rder.p.stepSimulation()
    #     info = rder.p.getContactPoints(i)
    #     # info = rder.p.getContactPoints(0,i)
    #     print(info)

    for i in range(1000):

        input = (model, q, qdot, torque)
        accelerations = ForwardDynamics(*input)
        

        # print("shape acc",accelerations.shape())
        accelerations = accelerations.flatten()
        # print("accelerations",accelerations)

        #step one forward
        
        for j in range(2,14):
            q[j] = q[j] + dt * qdot[j]
            qdot[j] = qdot[j] + dt * accelerations[j]
        
        print("q",q)
        print("qdot",qdot)

        rder.step_render(q)


def target_pos():
    model = UrdfWrapper("urdf/laikago/laikago.urdf").model

    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])
    rder = ObdlRender(model)
    q = np.array([0.0] * 14)
    q[1] = 1.57
    q[6] = 0.9
    q[12] = 0.9

    

    for i in range(1000):
        rder.step_render(q)
        time.sleep(10)



if __name__ == "__main__":
    # arm_test()
    laikago_dynamics_test()
    # laikago_trajectory()
#     target_pos()

