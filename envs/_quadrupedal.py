from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from envs.core import Env
import gym


class Qaudrupedal(Env):
    """
    Description:
        A quadrupedal robot(UNITREE) environment. The robot has totally 14 joints,
        the previous two are one virtual joint and a base to chasis joint.
    """   
    def __init__(self, reward_fn=None, seed=0):
        self.dt = 0.02
        self.target = jnp.zeros(14)
        self.target[1] = 1.57

        #front and rear left leg lift up
        self.target[6] = 0.9
        self.target[12] = 0.9

        self.model = UrdfWrapper("urdf/laikago/laikago.urdf").model
        # self.osim = ObdlSim(self.model,dt=self.dt,vis=True)
        self.rder = ObdlRender(model)


        def _dynamics(state, action):
            
            q, qdot = state
            torque = action

            # q = state
            # q = jnp.zeros(14)
            # qdot = jnp.zeros(14)
            # torque = jnp.zeros(14)
            # torque[3] = 0.5
            # print("q",q)
            # print("qdot",qdot)
            # print("force",force)
            input = (self.model, q, qdot, torque)
            qddot = ForwardDynamics(*input)
            # print("accelerations",accelerations)

            #step one forward
            # for j in range(2,14):
            #     q[j] = q[j] + dt * qdot[j]
            #     qdot[j] = qdot[j] + dt * accelerations[j]
            for i in range(2,14):
                q = jax.ops.index_add(q, i, self.dt * qdot[i]) 
                qdot = jax.ops.index_add(qdot, i, self.dt * qddot[i][0])

            return jnp.array([q, qdot])
            

        self.dynamics = _dynamics

    def reset(self):
        q = jax.random.uniform(
            self.random.get_key(), shape=(14,), minval=-0.05, maxval=0.05
        )
        qdot = jax.random.uniform(
            self.random.get_key(), shape=(14,), minval=-0.05, maxval=0.05
        )
        self.state = jnp.array([q,qdot])
        return self.state

    def step(self, state, action):
        self.state = self.dynamics(state, action)
        q, qdot = self.state

        done = False
        # if (len(q[q>self.theta_threshold_radians]) >0):
        #     print("q in done",q)
        #     done = True

        # reward = 1 - done
        reward = self.reward_func(self.state)

        return self.state, reward, done, {}


    def reward_func(self,state):
        # # x, x_dot, theta, theta_dot = state
        # reward = state[0]**2 + (state[1])**2 + 100*state[2]**2 + state[3]**2 
        # # reward = jnp.exp(state[0])-1 + state[2]**2 + state[3]**2 
        q, qdot = self.state
        reward = jnp.sum(jnp.square(q - self.target))

        return reward


    def osim_render(self):
        q, _ = self.state
        # print("q for render",q)
        rder.step_render(q)