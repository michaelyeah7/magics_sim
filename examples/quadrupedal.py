from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
import time
import numpy as np
import pybullet as p
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
import jax
import jax.numpy as jnp

def arm_test():
    dt = 0.001
    model = UrdfWrapper("urdf/two_link_arm.urdf").model
    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])
    rder = ObdlRender(model)
    time.sleep(1)
    q = np.array([0.0] * 4)
    q[3] = 100
    qdot = np.array([0.0] * 7)
    torque = np.array([0.0] * 7)
    torque[5] = 0.001
    while True:
        rder.step_render(q)

    # for i in range(1000):
    #     q[3] = 1.57
    #     input = (model, q, qdot, torque)
    #     accelerations = ForwardDynamics(*input)
        

    #     # print("shape acc",accelerations.shape())
    #     accelerations = accelerations.flatten()
    #     print("accelerations",accelerations)

    #     #step one forward
        
    #     for j in range(2,7):
    #         q[j] = q[j] + dt * qdot[j]
    #         qdot[j] = qdot[j] + dt * accelerations[j]
    #     # print("qdot",qdot)
    #     print("q",q)

    #     rder.step_render(q)

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


def cartpole_dynamics(state, action):
    tau = 0.01
    model = UrdfWrapper("urdf/cartpole_add_base.urdf").model


    x, x_dot, theta, theta_dot = state
    force = action[0] * 100

    q = jnp.array([0,0,x,theta])
    qdot = jnp.array([0,0,x_dot,theta_dot])
    torque = jnp.array([0,0,force,0.])
    # print("q",q)
    # print("qdot",qdot)
    # print("force",force)
    input = (model, q, qdot, torque)
    accelerations = ForwardDynamics(*input)
    # print("accelerations",accelerations)
    xacc = accelerations[2][0]
    thetaacc = accelerations[3][0]

    x_dot = x_dot + tau * xacc
    x = x + tau * x_dot
    
    theta_dot = theta_dot + tau * thetaacc
    theta = theta + tau * theta_dot
    
    # reward = jnp.exp(x**2) + (100*theta)**2 + theta_dot**2 
    reward = jnp.exp(x**2) + (100*theta)**2  
    return reward

def _dynamics(state, action):
    tau = 0.01
    model = UrdfWrapper("urdf/arm.urdf").model
    model["jtype"] = np.asarray(model["jtype"])
    model["parent"] = np.asarray(model["parent"])
    q, qdot = state
    torque = action/100
    input = (model, q, qdot, torque)
    #ForwardDynamics return shape(NB, 1) array
    qddot = ForwardDynamics(*input)
    qddot = qddot.flatten()
    qdot = qdot + tau * qddot
    q = q + tau * qdot
    

    target = jnp.array([0,0,0,0,1.57,0,0])
    reward = jnp.linalg.norm(jnp.square(q - target))
    # reward = jnp.exp(q[0]**2)
    return reward

def test_gradient():
    # #for cartpole
    # f_grad = jax.grad(cartpole_dynamics,argnums=1)
    # state = jnp.array([1,1,1,1])
    # action = jnp.ones(2)
    # grads = f_grad(state, action)

    #for arm
    f_grad = jax.grad(_dynamics,argnums=1)
    state = jnp.array([jnp.ones(7),jnp.ones(7)])
    action = jnp.ones(7)
    grads = f_grad(state, action)

    print('grads',grads)



def contact_example():
    model = UrdfWrapper("urdf/laikago/laikagolow.urdf").model
    osim = ObdlSim(model,dt=2e-3,vis=True)
    NL = len(model['parent'])
    print("contact Id:",model["idcontact"])
    #roate
    q = np.array([0.0]*NL)
    q[1] = 1.57
    osim.step_theta(q)
    time.sleep(2.0)
    #apply force
    q = np.array([0.0]*NL)
    q[2]=10.0
    #osim.step_toruqe(q)
    #contact
    i = 0
    while(True):    
        print("loop",i)
        i+=1
        osim.step_contact(q)
        # time.sleep(0.1)

if __name__ == "__main__":
    # arm_test()
    # laikago_dynamics_test()
    # laikago_trajectory()
#     target_pos()
    # test_gradient()
    contact_example()

