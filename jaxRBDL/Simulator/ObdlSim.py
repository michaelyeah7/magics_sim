from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
from pyRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from pyRBDL.Dynamics.InverseDynamics import InverseDynamics

import numpy as np
import math
class ObdlSim():
    def __init__(self,model,dt,vis=False):
        self.model = model
        #for pyjbdl
        self.model["jtype"] = np.asarray(self.model["jtype"])
        self.model["parent"] = np.asarray(self.model["parent"])

        #render
        self.visual = vis
        if self.visual:
            self.render = ObdlRender(model)
        self.dt = dt
        self.jnum = self.model['NB']

        #current state
        self.q = np.zeros((self.jnum,))
        self.qdot = np.zeros((self.jnum,))
        self.qddot = np.zeros((self.jnum,))
        self.tau = np.zeros((self.jnum,))
    
    def step_toruqe(self,tau):
        """
        use pyrbdl to calcaulate next state and render
        """
        q = self.q.copy()
        qdot = self.qdot.copy()

        input = (self.model, q, qdot, tau)
        qddot_hat = ForwardDynamics(*input).flatten() 
        # print("applied torque",tau,"qddot_hat",qddot_hat)
        q_hat,qdot_hat = self.calculate_q(dt=self.dt,q=q,qdot=qdot,qddot=qddot_hat)

        #calac contact here to get JDotDot
        # CalcContactJdotQdot
        
        if(self.visual):
            self.render.step_render(q_hat)

        self.q = q_hat
        self.qdot =  np.zeros((self.jnum,))#TODO qdot_hat or zeros
        self.qddot = np.zeros((self.jnum,)) #TODO qddot_hat or zeros
        return

    def calculate_q(self,dt,q,qdot,qddot):
        qdot = qdot + qddot * dt
        q = q + qdot * dt
        return q,qdot
    
    def step_theta(self,q):
        """
        TODO:we are suppoed to use inverse dynamics to make it move here 
        """
        if(self.visual):
            self.render.step_render(q)
        
        self.q = np.array(q)


if __name__ == "__main__":
    model = UrdfWrapper("/root/RBDL/urdf/arm.urdf").model
    osim = ObdlSim(model,dt=0.1,vis=True)

    import time
    while(True):    
        q = np.array([ 0.0, 0.0,0.0,  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
            np.random.uniform(-math.pi/2,math.pi/2),0.0])
        osim.step_theta(q)
        time.sleep(3)




    