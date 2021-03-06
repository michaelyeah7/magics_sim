from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
from jaxRBDL.Utils.UrdfReader import URDF
import numpy as np
import math

if __name__ == "__main__":
    model = UrdfWrapper("/root/RBDL/urdf/cartpole.urdf").model
    osim = ObdlSim(model,dt=0.1,vis=True)

    import time
    # while(True):    
    #     q = np.array([ 0.0, 0.0,0.5,  np.random.uniform(-math.pi/2,math.pi/2), np.random.uniform(-math.pi/2,math.pi/2), \
    #         np.random.uniform(-math.pi/2,math.pi/2),0.0])
    #     osim.step_theta(q)
    #     time.sleep(3)

    while(True):    
        # q = np.array([ 0.0, 0.0,0.0, 0.02,0.02,0.0,0.0]) 
        q = np.array([ 0.0, 0.0, 10.0, 0.0,0.02,0.0,0.0]) 
        osim.step_toruqe(q)
        time.sleep(0.2)