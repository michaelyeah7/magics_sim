# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math

import gym
import jax
import jax.numpy as jnp
# from jax.ops import index_add
import numpy as np

# from deluca.envs.core import Env
# from deluca.utils import Random
from envs.core import Env
from utils import Random
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from jaxRBDL.Utils.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from jaxRBDL.Simulator.ObdlRender import ObdlRender
from jaxRBDL.Simulator.ObdlSim import ObdlSim
import os
import pybullet as p
from numpy import sin, cos

class Arm_rbdl(Env):
    """
    Description:
        A 7 link arm robot contains 6 joints. The first base_link to arm_link_0 fixed joint 
        will be interpreted as prismatic joint (rbdl index 1) by rbdl. The remaining 5 joints are revolute
        joints (rbdl index 0).

    State:
        array of two jnp array       
        7-element array: Angles of 7 joints (the first one is a virtual prismatic joint transform from world to base).
        7-element array: Angle velocity of 7 joints.

    Actions:
        7-element List: forces applied on 7 joints.

    Reward:
        Square difference between current state and target. 

    Starting State:
        All zeros.

    Episode Termination:
        At least one angle is larger than 45 degrees.
    """

    def __init__(self, reward_fn=None, seed=0):
        self.tau = 0.1  # seconds between state updates
        self.kinematics_integrator = "euler"
        self.viewer = None
        self.target = jnp.array([0,0,0,0,0,0,0])
        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold_radians = math.pi / 2
        # self.x_threshold = 2.4

        self.random = Random(seed)

        self.model = UrdfWrapper("urdf/arm.urdf").model
        # self.model = UrdfWrapper("urdf/two_link_arm.urdf").model
        self.osim = ObdlSim(self.model,dt=self.tau,vis=True)
        
        self.reset()

        # @jax.jit
        def _dynamics(state, action):
            q, qdot = state
            torque = action/100 
            #calculate xacc & thetaacc using jaxRBDL
            # q = jnp.array(state[0])
            # qdot = jnp.array(state[1])
            # torque = jnp.array(action)
            print("q",q)
            print("qdot",qdot)
            print("torque",torque)
            input = (self.model, q, qdot, torque)
            #ForwardDynamics return shape(NB, 1) array
            qddot = ForwardDynamics(*input)
            qddot = jnp.clip(qddot,0,0.5)
            print("qddot",qddot)
            # print("xacc",xacc)
            # print("thetaacc",thetaacc)

            # _q_0 = q[0] + self.tau * qdot[0]
            # _qdot_0 = qdot[0] + self.tau * qddot[0][0]

            # q = jnp.ones((7,)) * _q_0
            # qdot = jnp.ones((7,)) * _qdot_0

            # _q = jnp.zeros((7,))
            # _qdot = jnp.zeros((7,))
            for i in range(2,len(q)):
                q = jax.ops.index_add(q, i, self.tau * qdot[i]) 
                qdot = jax.ops.index_add(qdot, i, self.tau * qddot[i][0])
            # q[0] = 0
            # qdot[0] = 0
            jax.ops.index_update(q, 0, 0.)
            # print("q[0]".q[0])
            jax.ops.index_update(qdot, 0, 0.)

            # jax.ops.index_add(q, 0, 6.)
            # _q[0] = q[0] + self.tau * qdot[0]
            # _qdot[0] = qdot[0] + self.tau * qddot[0][0] 
            # _q[1] = q[1] + self.tau * qdot[1]
            # _qdot[1] = qdot[1] + self.tau * qddot[1][0]  
            # _q[2] = q[2] + self.tau * qdot[2]
            # _qdot[2] = qdot[2] + self.tau * qddot[2][0] 
            # _q[3] = q[3] + self.tau * qdot[3]
            # _qdot[3] = qdot[3] + self.tau * qddot[3][0] 
            # _q[4] = q[4] + self.tau * qdot[4]
            # _qdot[4] = qdot[4] + self.tau * qddot[4][0] 
            # _q[5] = q[5] + self.tau * qdot[5]
            # _qdot[5] = qdot[5] + self.tau * qddot[5][0] 
            # _q[6] = q[6] + self.tau * qdot[6]
            # _qdot[6] = qdot[6] + self.tau * qddot[6][0] 

            # q = _q
            # qdot = _qdot

            return jnp.array([q, qdot])
        
        self.dynamics = _dynamics

    def reset(self):
        q = jax.random.uniform(
            self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        )
        qdot = jax.random.uniform(
            self.random.get_key(), shape=(7,), minval=-0.05, maxval=0.05
        )
        self.state = jnp.array([q,qdot])
        return self.state

    def step(self, state, action):
        self.state = self.dynamics(state, action)
        q, qdot = self.state

        # done = jax.lax.cond(
        #     (jnp.abs(x) > jnp.abs(self.x_threshold))
        #     + (jnp.abs(theta) > jnp.abs(self.theta_threshold_radians)),
        #     lambda done: True,
        #     lambda done: False,
        #     None,
        # )
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
        print("q for render",q)
        self.osim.step_theta(q)