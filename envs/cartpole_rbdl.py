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
import numpy as np

# from utils import Random
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
# from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from Simulator.UrdfWrapper import UrdfWrapper
# from jaxRBDL.Utils.UrdfWrapper_guo import UrdfWrapper
from Simulator.ObdlRender import ObdlRender
from Simulator.ObdlSim import ObdlSim
import os
import pybullet as p
from numpy import sin, cos

class Cartpole_rbdl():
    def __init__(self, render_flag=False, reward_fn=None, seed=0):

        # self.gravity = 9.8
        # self.masscart = 1.0
        # self.masspole = 0.1
        # self.total_mass = self.masspole + self.masscart
        # self.length = 0.5  # actually half the pole's length
        # self.polemass_length = self.masspole * self.length
        # self.force_mag = 10.0
        # self.tau = 0.02  # seconds between state updates
        # self.kinematics_integrator = "euler"
        # self.viewer = None
        # self.render_flag = render_flag

        self.gravity = 9.82
        self.masscart = 0.5
        self.masspole = 0.5
        self.total_mass = self.masspole + self.masscart
        self.length = 0.6  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.01 # seconds between state updates
        self.kinematics_integrator = "euler"
        self.viewer = None
        self.render_flag = render_flag



        self.g = 9.82  # gravity
        self.m_c = 0.5  # cart mass
        self.m_p = 0.5  # pendulum mass
        self.total_m = (self.m_p + self.m_c)
        self.l = 0.6  # pole's length
        # self.l = 1.2  # pole's length
        self.m_p_l = (self.m_p * self.l)
        self.force_mag = 10.0
        self.dt = 0.01  # seconds between state updates
        self.b = 0.1  # friction coefficient



        # Angle at which to fail the episode
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        # self.x_threshold = 2.4
        self.x_threshold = 6.0

        # self.random = Random(seed)

        self.model = UrdfWrapper("urdf/cartpole_add_base.urdf").model
        # self.osim = ObdlSim(self.model,dt=self.tau,vis=True)
        
        #three dynamic options "RBDL" "Original" "PDP" "DDPG"
        self.dynamics_option = "DDPG"
        # self.model["NB"] = self.model["NB"] + 1 

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        # self.action_space = jnp.array([0, 1])
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,))
        # TODO: no longer use gym.spaces
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.state_size, self.action_size = 4, 1
        self.observation_size = self.state_size

        # self.d_reward_d_x_prime = jax.grad(self.reward_func)
        # self.d_x_prime_d_a = jax.jacfwd(self.dynamics,argnums=1)
        self.past_reward = 0
        self.reset()


        #simulator
        # CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
        # p.connect(p.GUI)
        # p.setAdditionalSearchPath(CURRENT_PATH) 
        # self.test_robot = p.loadURDF("urdf/cartpole.urdf",[0,0,1])

        # for j in range (p.getNumJoints(self.test_robot)):
        #     info = p.getJointInfo(self.test_robot,j)
        #     print("joint info", info)

        # @jax.jit
        def _dynamics(state, action):
            # print("state",state)
            # print("action",action)
            x, x_dot, theta, theta_dot = state

            # force = jax.lax.cond(action == 1, lambda x: x, lambda x: -x, self.force_mag)
            # force = (action - 0.5) * 2 * 10
            # force = np.clip(action[1] * 100,-10,10)
            # print("action",action)

            #works for original
            # force = np.clip(action[0] * 100,-10,10)
            #works for PDP and Original
            # force = action[0] * 10
            #works for RBDL
            # force = action[0] * 100
            #works for hybrid env
            force = action[0]
            # force = action[0] * 100
            # print("fr",action)
            # print("force",force)
            if (self.dynamics_option == "Original"):
                costheta = jnp.cos(theta)
                sintheta = jnp.sin(theta)

                # cartpole dynamics
                temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
                thetaacc_manually = (self.gravity * sintheta - costheta * temp) / (
                    self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
                )
                xacc_manually = temp - self.polemass_length * thetaacc_manually * costheta / self.total_mass
                # print("xacc_manually",xacc_manually)            
                # print("thetaacc_manually",thetaacc_manually)
                xacc = xacc_manually
                thetaacc = thetaacc_manually

            if (self.dynamics_option == "PDP"):
                #calculate xacc & thetaacc using PDP
                # x = 0.04653214
                dx = x_dot
                q = theta + jnp.pi
                dq = theta_dot
                U = force

                #mass of cart and pole
                mp = 0.1
                mc = 1.0
                l = 0.5

                g=9.81

                ddx = (U + mp * jnp.sin(q) * (l * dq * dq + g * jnp.cos(q))) / (
                        mc + mp * jnp.sin(q) * jnp.sin(q))  # acceleration of x
                ddq = (-U * jnp.cos(q) - mp * l * dq * dq * jnp.sin(q) * jnp.cos(q) - (
                        mc + mp) * g * jnp.sin(
                    q)) / (
                                l * mc + l * mp * jnp.sin(q) * jnp.sin(q))  # acceleration of theta
                xacc = ddx
                thetaacc = ddq

            if (self.dynamics_option == "RBDL"):
                #calculate xacc & thetaacc using jaxRBDL
                force = force/100
                q = jnp.array([0,0,x,theta])
                qdot = jnp.array([0,0,x_dot,theta_dot])
                torque = jnp.array([0,0,force,0.])
                # print("q",q)
                # print("qdot",qdot)
                # print("force",force)
                input = (self.model, q, qdot, torque)
                accelerations = ForwardDynamics(*input)
                # print("accelerations",accelerations)
                # xacc = accelerations[2][0]
                # thetaacc = accelerations[3][0]
                xacc = accelerations[2]
                thetaacc = accelerations[3]           
                # print("xacc",xacc)
                # print("thetaacc",thetaacc)


            if (self.dynamics_option == "DDPG"):
                action = action
                s = jnp.sin(theta)
                c = jnp.cos(theta)

                xdot_update_ddpg = (-2 * self.m_p_l * (
                        theta_dot ** 2) * s + 3 * self.m_p * self.g * s * c + 4 * action - 4 * self.b * x_dot) / (
                                    4 * self.total_m - 3 * self.m_p * c ** 2)
                thetadot_update_ddpg = (-3 * self.m_p_l * (theta_dot ** 2) * s * c + 6 * self.total_m * self.g * s + 6 * (
                        action - self.b * x_dot) * c) / (4 * self.l * self.total_m - 3 * self.m_p_l * c ** 2)

                # print(colored("xdot_update_ddpg",'green'),xdot_update_ddpg)
                # print(colored("thetadot_update_ddpg",'green'),thetadot_update_ddpg)
                xacc = xdot_update_ddpg[0]
                thetaacc = thetadot_update_ddpg[0]




            if self.kinematics_integrator == "euler":
                x = x + self.tau * x_dot
                x_dot = x_dot + self.tau * xacc
                theta = theta + self.tau * theta_dot
                theta_dot = theta_dot + self.tau * thetaacc
            else:  # semi-implicit euler
                x_dot = x_dot + self.tau * xacc
                x = x + self.tau * x_dot
                theta_dot = theta_dot + self.tau * thetaacc
                theta = theta + self.tau * theta_dot

            return jnp.array([x, x_dot, theta, theta_dot])
        
        self.dynamics = _dynamics

    def reset(self):
        # self.state = jax.random.uniform(
        #     self.random.get_key(), shape=(4,), minval=-0.05, maxval=0.05
        # )
        # self.state = jnp.array(list(np.random.uniform(-0.05,0.05,4)))
        # self.state = jnp.array([0.,0.,jnp.pi,0.])
        self.state = jnp.array([0.,0.,0.1,0.])
        return self.state

    def step(self, state, action):
        # print("start step")
        # print("action is",action)
        self.state = self.dynamics(state, action)
        x, x_dot, theta, theta_dot = self.state
        # print("x",x)
        # print("type",type(x))

        # done = jax.lax.cond(
        #     (jnp.abs(x) > jnp.abs(self.x_threshold))
        #     + (jnp.abs(theta) > jnp.abs(self.theta_threshold_radians)),
        #     lambda done: True,
        #     lambda done: False,
        #     None,
        # )
        done = jax.lax.cond(
            (jnp.abs(x) > jnp.abs(self.x_threshold)),
            lambda done: True,
            lambda done: False,
            None,
        )
        # done = False

        # reward = 1 - done
        reward = self.reward_func(self.state,action)

        return self.state, reward, done, {}


    def reward_func(self,state,action):
        x, x_dot, theta, theta_dot = state
        # reward = state[0]**2 + (state[1])**2 + 100*state[2]**2 + state[3]**2 
        # reward = jnp.exp(state[0])-1 + state[2]**2 + state[3]**2 
        # costs = jnp.exp(state[0]**2) + (100*state[2])**2 + state[3]**2 
        # normalized_theta = state[2] % jnp.pi
        # costs = (state[0]**2) + 100 * (state[2]**2) + state[3]**2 
        # costs = 0.1 * (state[0]**2) + 0.6 * (state[2]**2) + 0.1 * (state[1]**2) + 0.1 * (state[3]**2) 
        # reward = -costs 

        # reward function as described in dissertation of Deisenroth with A=1
        A = 1
        invT = A * jnp.array([[1, self.l, 0], [self.l, self.l ** 2, 0], [0, 0, self.l ** 2]])
        j = jnp.array([x, jnp.sin(theta), jnp.cos(theta)])
        j_target = np.array([0.0, 0.0, 1.0])

        reward = jnp.matmul((j - j_target), invT)
        reward = jnp.matmul(reward, (j - j_target))
        reward = -(1 - jnp.exp(-0.5 * reward))


        return reward


    def original_render(self, mode="human"):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        carty = 100  # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)
            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
            axleoffset = cartheight / 4.0
            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0.8, 0.6, 0.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.5, 0.5, 0.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None:
            return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
        pole.v = [(l, b), (l, t), (r, t), (r, b)]

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=mode == "rgb_array")


    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        screen_width = 600
        screen_height = 600

        world_width = 5
        scale = screen_width / world_width
        carty = screen_height / 2
        polewidth = 6.0
        polelen = scale * self.l
        cartwidth = 40.0
        cartheight = 20.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2

            cart = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            cart.set_color(1, 0, 0)
            self.viewer.add_geom(cart)

            l, r, t, b = -polewidth / 2, polewidth / 2, polelen - polewidth / 2, -polewidth / 2
            pole = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            pole.set_color(0, 0, 1)
            self.poletrans = rendering.Transform(translation=(0, 0))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)

            self.axle = rendering.make_circle(polewidth / 2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(0.1, 1, 1)
            self.viewer.add_geom(self.axle)

            self.pole_bob = rendering.make_circle(polewidth / 2)
            self.pole_bob_trans = rendering.Transform()
            self.pole_bob.add_attr(self.pole_bob_trans)
            self.pole_bob.add_attr(self.poletrans)
            self.pole_bob.add_attr(self.carttrans)
            self.pole_bob.set_color(0, 0, 0)
            self.viewer.add_geom(self.pole_bob)

            self.wheel_l = rendering.make_circle(cartheight / 4)
            self.wheel_r = rendering.make_circle(cartheight / 4)
            self.wheeltrans_l = rendering.Transform(translation=(-cartwidth / 2, -cartheight / 2))
            self.wheeltrans_r = rendering.Transform(translation=(cartwidth / 2, -cartheight / 2))
            self.wheel_l.add_attr(self.wheeltrans_l)
            self.wheel_l.add_attr(self.carttrans)
            self.wheel_r.add_attr(self.wheeltrans_r)
            self.wheel_r.add_attr(self.carttrans)
            self.wheel_l.set_color(0, 0, 0)
            self.wheel_r.set_color(0, 0, 0)
            self.viewer.add_geom(self.wheel_l)
            self.viewer.add_geom(self.wheel_r)

            self.track = rendering.Line(
                (screen_width / 2 - self.x_threshold * scale, carty - cartheight / 2 - cartheight / 4),
                (screen_width / 2 + self.x_threshold * scale, carty - cartheight / 2 - cartheight / 4))
            self.track.set_color(0, 0, 0)
            self.viewer.add_geom(self.track)

        if self.state is None: return None

        x = self.state
        cartx = x[0] * scale + screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(x[2])
        self.pole_bob_trans.set_translation(-self.l * np.sin(x[2]), self.l * np.cos(x[2]))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')



    def osim_render(self):
        # # q=[0,self.state[0],self.state[1]]
        # for j in range(2):
        #     p.setJointMotorControl2(self.test_robot,j,p.POSITION_CONTROL,self.state[j],force = 1)
        # p.stepSimulation()

        # x, x_dot, theta, theta_dot = state
        # print("self.state",self.state)
        q = [0,0,self.state[0],self.state[2]]
        self.osim.step_theta(q)

class Cartpole_Hybrid():
    def __init__(self, model_lr = 1e-2, seed=0):
        #init one layer parameters
        import numpy.random as npr
        rng=npr.RandomState(0)
        self.model_lr = model_lr
        #TODO add sigma
        # self.model_params = [rng.randn(4, 4),rng.randn(4)]
        self.model_params = [(rng.randn(4, 4),rng.randn(4))]
        # self.model_params = [(rng.randn(4, 32),rng.randn(32)),(rng.randn(32, 4),rng.randn(4))]
        self.model = UrdfWrapper("urdf/cartpole_add_base.urdf").model 
        self.model_losses = []
        self.tau = 0.02
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4


    def reset(self):
        # self.random = Random(seed)
        # self.state = jax.random.uniform(
        #     self.random.get_key(), shape=(4,), minval=-0.05, maxval=0.05
        # )
        # self.state = jax.random.uniform(
        #     self.random.get_key(), shape=(4,), minval=-0.209, maxval=0.209
        # )
        # self.state = jnp.array(list(np.random.uniform(-0.209,0.209,4)))
        self.state = jnp.array([0.,0.,3.14,0.])
        # print("init state", self.state)
        return self.state

    def forward(self, state, action, model_params):
        x, x_dot, theta, theta_dot = state
        force = action[0] * 10
        #works for RBDL
        # force = action[0] * 100
        
        # q = jnp.array([0,0,x,theta])
        # qdot = jnp.array([0,0,x_dot,theta_dot])
        # torque = jnp.array([0,0,force,0.])

        # input = (self.model, q, qdot, torque)
        # accelerations = ForwardDynamics(*input)
        # # print("accelerations",accelerations)
        # xacc = accelerations[2][0]
        # thetaacc = accelerations[3][0]

        #calculate xacc & thetaacc using PDP
        # x = 0.04653214
        dx = x_dot
        q = theta
        dq = theta_dot
        U = force

        #mass of cart and pole
        mp = 0.1
        mc = 1.0
        l = 0.5

        g=9.81

        ddx = (U + mp * jnp.sin(q) * (l * dq * dq + g * jnp.cos(q))) / (
                mc + mp * jnp.sin(q) * jnp.sin(q))  # acceleration of x
        ddq = (-U * jnp.cos(q) - mp * l * dq * dq * jnp.sin(q) * jnp.cos(q) - (
                mc + mp) * g * jnp.sin(
            q)) / (
                        l * mc + l * mp * jnp.sin(q) * jnp.sin(q))  # acceleration of theta
        xacc = ddx
        thetaacc = ddq

        #Integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        next_state = jnp.array([x, x_dot, theta, theta_dot])

        # w, b = model_params
        # outputs = jnp.dot(next_state, w) + b
        # dist = jax.nn.elu(outputs)
        activations = next_state
        for w,b in model_params[:-1]:
            outputs = jnp.dot(activations, w) + b
            activations = jax.nn.elu(outputs)
        final_w, final_b = model_params[-1]
        dist = jnp.dot(activations, final_w) + final_b
        return dist

    def step(self, state, action):
        # print("start step")
        # print("action is",action)
        self.state = self.forward(state, action,self.model_params)
        x, x_dot, theta, theta_dot = self.state
        # print("x",x)
        # print("type",type(x))

        done = jax.lax.cond(
            (jnp.abs(x) > jnp.abs(self.x_threshold))
            + (jnp.abs(theta) > jnp.abs(self.theta_threshold_radians)),
            lambda done: True,
            lambda done: False,
            None,
        )

        # reward = 1 - done
        reward = self.reward_func(self.state)

        return self.state, reward, done, {}


    def reward_func(self,state):
        # x, x_dot, theta, theta_dot = state
        # reward = state[0]**2 + (state[1])**2 + 100*state[2]**2 + state[3]**2 
        # reward = jnp.exp(state[0])-1 + state[2]**2 + state[3]**2 
        # reward = jnp.exp(state[0]**2) + (100*state[2])**2 + state[3]**2 
        costs = 0.1 * (state[0]**2) + 0.6 * (state[2]**2) + 0.1 * (state[1]**2) + 0.1 * (state[3]**2)
        reward = -costs
        return reward
