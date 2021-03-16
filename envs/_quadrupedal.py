from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from envs.core import Env
class Quadrupedal(Env):
    """
    Description:
        A quadrupedal robot(UNITREE) environment. The robot has totally 14 joints,
        the previous two are one virtual joint and a base to chasis joint.
    """   
    def __init__(self, reward_fn=None, seed=0):
        self.dt = 0.02

        self.model = UrdfWrapper("urdf/laikago/laikago.urdf").model
        self.osim = ObdlSim(self.model,dt=self.dt,vis=True)


        def _dynamics(state, action):
            #state is a list with length 14

            q = state
            q = jnp.zeros(14)
            qdot = jnp.zeros(14)
            torque = jnp.zeros(14)
            torque[3] = 0.5
            # print("q",q)
            # print("qdot",qdot)
            # print("force",force)
            input = (self.model, q, qdot, torque)
            accelerations = ForwardDynamics(*input)
            # print("accelerations",accelerations)
            xacc = accelerations[2][0]
            thetaacc = accelerations[3][0]

            #step one forward
            x = x + self.dt * x_dot
            x_dot = x_dot + self.dt * xacc
            theta = theta + self.dt * theta_dot
            theta_dot = theta_dot + self.dt * thetaacc
            

        self.dynamics = _dynamics