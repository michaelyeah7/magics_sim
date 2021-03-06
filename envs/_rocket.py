import math

import gym
import jax
import jax.numpy as jnp
import numpy as np

from envs.core import Env
from utils import Random
from numpy.linalg import inv

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import scipy.integrate as integrate
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Circle, PathPatch

class Rocket():
    """
    Observation:
        r: position of CoM
        v: velocity of CoM
        q: quartenion denoting the attitude of rocket body frame with respect to the inertial frame.
        w: angular velocity of rocket expressed in the rocket body frame.

    Action:
        u[Tx,Ty,Tz]: thrust force vector acting on the gimbal point 
                      of the engine (situated at the tail of the rocket) and is expressed in the body frame.
    """

    def __init__(self):
        self.g = 10
        self.mass = 1.0
        self.length = 1.0
        self.rg_record = []
        self.rh_record = []


        def _dynamics(state, action):
            r, v, q, w = state

            #forward dynamics
            d_r = v

            C_B_I = self.dir_cosine(q)
            C_I_B = jnp.transpose(C_B_I)
            d_v = 1/self.mass * jnp.matmul(C_I_B, action) + jnp.array([-self.g,0,0]) 

            d_q = 1 / 2 * jnp.matmul(self.omega(w), q)

            r_T_B = jnp.array([-self.length / 2, 0, 0])
            J_B = jnp.diag(jnp.array([0.5,1,1]))
            d_w = jnp.matmul(inv(J_B),
                    jnp.matmul(self.skew(r_T_B), action) -
                    jnp.matmul(jnp.matmul(self.skew(w), J_B), w))

            #next state
            dt = 0.1
            next_r = r+d_r*dt
            next_v = v+d_v*dt
            next_q = q+d_q*dt
            next_w = w+d_w*dt

            # positions of tip and tail for plotting
            # position of gimbal point (rocket tail)
            # print("jnp.matmul(C_I_B, r_T_B)",jnp.matmul(C_I_B, r_T_B))
            rg = r + jnp.matmul(C_I_B, r_T_B)
            self.rg_record.append(rg)

            # position of rocket tip
            rh = r - jnp.matmul(C_I_B, r_T_B)
            self.rh_record.append(rh)

            next_state = [next_r,next_v,next_q,next_w]
            return next_state
        
        self.dynamics = _dynamics

    def reset(self,init_state):
        self.state = init_state
        self.rg_record = []
        self.rh_record = []

    def step(self,state,action):
        next_state = self.dynamics(state,action)
        self.state = next_state

        z_threshold = 15.0
        r, v, q, w = next_state

        done = jax.lax.cond(
            (jnp.abs(r[2]) > jnp.abs(z_threshold))
            ,
            lambda done: True,
            lambda done: False,
            None,
        )
        reward = self.reward_func(next_state)


        return reward, next_state, done

    def reward_func(self,state):
        r, v, q, w = state
        cost_r = jnp.dot(r,r)
        cost_v = jnp.dot(v,v)
        cost_w = jnp.dot(w,w)

        # tilt angle upward direction of rocket should be close to upward of earth
        C_I_B = jnp.transpose(self.dir_cosine(q))
        nx = np.array([1., 0., 0.])
        ny = np.array([0., 1., 0.])
        nz = np.array([0., 0., 1.])
        proj_ny = jnp.dot(ny, jnp.matmul(C_I_B, nx))
        proj_nz = jnp.dot(nz, jnp.matmul(C_I_B, nx))
        cost_tilt = proj_ny ** 2 + proj_nz ** 2

        cost = 10*cost_r + cost_v + cost_w + 50*cost_tilt

        return cost


    def dir_cosine(self, q):
        C_B_I = jnp.array([
            [1 - 2 * (q[2] ** 2 + q[3] ** 2), 2 * (q[1] * q[2] + q[0] * q[3]), 2 * (q[1] * q[3] - q[0] * q[2])],
            [2 * (q[1] * q[2] - q[0] * q[3]), 1 - 2 * (q[1] ** 2 + q[3] ** 2), 2 * (q[2] * q[3] + q[0] * q[1])],
            [2 * (q[1] * q[3] + q[0] * q[2]), 2 * (q[2] * q[3] - q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)]
        ])
        return C_B_I

    def omega(self, w):
        omeg = jnp.array([
            [0, -w[0], -w[1], -w[2]],
            [w[0], 0, w[2], -w[1]],
            [w[1], -w[2], 0, w[0]],
            [w[2], w[1], -w[0], 0]
        ])
        return omeg

    def skew(self, v):
        v_cross = jnp.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        return v_cross

    # converter to quaternion from (angle, direction)
    def toQuaternion(self, angle, dir):
        if type(dir) == list:
            dir = np.array(dir)
        dir = dir / np.linalg.norm(dir)
        quat = np.zeros(4)
        quat[0] = math.cos(angle / 2)
        quat[1:] = math.sin(angle / 2) * dir
        return quat.tolist()

    def play_animation(self):
        title='Rocket Powered Landing'
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('East (m)')
        ax.set_ylabel('North (m)')
        ax.set_zlabel('Upward (m)')
        ax.set_zlim(0, 10)
        ax.set_ylim(-8, 8)
        ax.set_xlim(-8, 8)
        ax.set_title(title, pad=20, fontsize=15)

        # target landing point
        p = Circle((0, 0), 3, color='g', alpha=0.3)
        ax.add_patch(p)
        art3d.pathpatch_2d_to_3d(p, z=0, zdir="z")


        xg, yg, zg = self.rg_record[0]
        xh, yh, zh = self.rh_record[0]
        line_rocket, = ax.plot([yg, yh], [zg, zh], [xg, xh], linewidth=5, color='black')

        # time label
        # time_template = 'time = %.1fs'
        # time_text = ax.text2D(0.66, 0.55, "time", transform=ax.transAxes)

        def update_traj(num):
            # time_text.set_text(time_template % (num * dt))
            t=num
            # rocket
            xg, yg, zg = self.rg_record[t]
            xh, yh, zh = self.rh_record[t]
            line_rocket.set_data([yg, yh], [zg, zh])
            line_rocket.set_3d_properties([xg, xh])

            return line_rocket

        ani = animation.FuncAnimation(fig, update_traj, len(self.rg_record), interval=100, blit=False)
        plt.show()