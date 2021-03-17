from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from envs.core import Env
import gym
import jax
import jax.numpy as jnp


class Qaudrupedal(Env):
    """
    Description:
        A quadrupedal robot(UNITREE) environment. The robot has totally 14 joints,
        the previous two are one virtual joint and a base to chasis joint.
    """   
    def __init__(self, reward_fn=None, seed=0):
        self.dt = 0.02
        self.q_threshold = 1.0
        self.qdot_threshold = 0.2
        self.target = jnp.zeros(14)
        self.target = jax.ops.index_update(self.target, 1, 1.57)

        #joint number 6,9 front and rear left leg lift up
        self.target = jax.ops.index_update(self.target, 6, 0.9)
        self.target = jax.ops.index_update(self.target, 9, 0.9)

        self.qdot_target = jnp.zeros(14)

        model = UrdfWrapper("urdf/laikago/laikago.urdf").model
        model["jtype"] = jnp.asarray(model["jtype"])
        model["parent"] = jnp.asarray(model["parent"])

        self.model = model
        # self.osim = ObdlSim(self.model,dt=self.dt,vis=True)
        self.rder = ObdlRender(self.model)


        def _dynamics(state, action):
            
            q, qdot = state
            torque = action/10


            # print("q",q)
            # print("qdot",qdot)
            # print("torque",torque)
            input = (self.model, q, qdot, torque)
            qddot = ForwardDynamics(*input)
            qddot = qddot.flatten()
            # print("qddot",qddot)

            #step one forward
            # for j in range(2,14):
            #     q[j] = q[j] + dt * qdot[j]
            #     qdot[j] = qdot[j] +  dt * accelerations[j]
            for i in range(2,14):
                q = jax.ops.index_add(q, i, self.dt * qdot[i]) 
                qdot = jax.ops.index_add(qdot, i, self.dt * qddot[i])

            return jnp.array([q, qdot])
            

        self.dynamics = _dynamics

    def reset(self):
        # q = jax.random.uniform(
        #     self.random.get_key(), shape=(14,), minval=-0.05, maxval=0.05
        # )
        q = jnp.zeros(14)
        q = jax.ops.index_update(q, 1, 1.57)

        # qdot = jax.random.uniform(
        #     self.random.get_key(), shape=(14,), minval=-0.05, maxval=0.05
        # )
        qdot = jnp.zeros(14)
        self.state = jnp.array([q,qdot])
        return self.state

    def step(self, state, action):
        self.state = self.dynamics(state, action)
        q, qdot = self.state

        done = False
        # if (len(q[q>self.theta_threshold_radians]) >0):
        #     print("q in done",q)
        #     done = True

        # if (len(qdot[qdot>self.qdot_threshold]) >0):
        # if (len(q[q>self.q_threshold]) >0):  
        #     # print("q in done",q)
        #     done = True

        # reward = 1 - done
        reward = self.reward_func(self.state)

        return self.state, reward, done, {}


    def reward_func(self,state):
        # # x, x_dot, theta, theta_dot = state
        # reward = state[0]**2 + (state[1])**2 + 100*state[2]**2 + state[3]**2 
        # # reward = jnp.exp(state[0])-1 + state[2]**2 + state[3]**2 
        q, qdot = state
        # reward = jnp.log(jnp.sum(jnp.square(q - self.target))) + jnp.log(jnp.sum(jnp.square(qdot - self.qdot_target)))
        reward = jnp.log((q[6]-0.9)**2) + jnp.log((q[9]-0.9)**2) 

        return reward


    def osim_render(self):
        q, _ = self.state
        # print("q for render",q)
        self.rder.step_render(q)