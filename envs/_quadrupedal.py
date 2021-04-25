from Simulator.UrdfWrapper import UrdfWrapper
from Simulator.ObdlRender import ObdlRender
from Simulator.ObdlSim import ObdlSim
from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics, ForwardDynamicsCore
from envs.core import Env
import gym
import jax
import jax.numpy as jnp
from jax.api import jit

from jaxRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from jaxRBDL.Dynamics.InverseDynamics import InverseDynamics
from jax.numpy.linalg import inv
from jaxRBDL.Contact.DetectContact import DetectContact
from jaxRBDL.Contact.CalcContactForceDirect import CalcContactForceDirect
from jaxRBDL.Contact.SolveContactLCP import SolveContactLCP
# from jaxRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from jaxRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates


class Qaudrupedal(Env):
    """
    Description:
        A quadrupedal robot(UNITREE) environment. The robot has totally 14 joints,
        the previous two are one virtual joint and a base to chasis joint.
    """   
    def __init__(self, reward_fn=None, seed=0):
        self.dt = 2e-03
        self.q_threshold = 1.0
        self.qdot_threshold = 1000
        self.target = jnp.zeros(14)
        self.target = jax.ops.index_update(self.target, 1, 1.57)

        #joint number 6,9 front and rear left leg lift up
        # self.target = jax.ops.index_update(self.target, 6, 0.9)
        # self.target = jax.ops.index_update(self.target, 9, 0.9)

        self.qdot_target = jnp.zeros(14)

        model = UrdfWrapper("urdf/laikago/laikagolow.urdf").model
        model["jtype"] = jnp.asarray(model["jtype"])
        model["parent"] = jnp.asarray(model["parent"])

        self.model = model
        # self.osim = ObdlSim(self.model,dt=self.dt,vis=True)
        self.rder = ObdlRender(self.model)

        # @jit
        def _dynamics(state, action):
            
            q, qdot = state
            torque = action/20


            # # print("q",q)
            # # print("qdot",qdot)
            # # print("torque",torque)
            # input = (self.model, q, qdot, torque)
            # qddot = ForwardDynamics(*input)
            # qddot = qddot.flatten()
            # # print("qddot",qddot)

            # #step one forward
            # # for j in range(2,14):
            # #     q[j] = q[j] + dt * qdot[j]
            # #     qdot[j] = qdot[j] +  dt * accelerations[j]
            # for i in range(2,14):
            #     q = jax.ops.index_add(q, i, self.dt * qdot[i]) 
            #     qdot = jax.ops.index_add(qdot, i, self.dt * qddot[i])


            _X = jnp.hstack((q,qdot))
            _model = self.model
            _model['tau'] = torque 
            _model['ST'] = jnp.zeros((3,)) # useless

            NB = int(_model["NB"])
            NC = int(_model["NC"])
            ST = _model["ST"]

            # Get q qdot tau
            q = _X[0:NB]
            qdot = _X[NB: 2 * NB]
            tau = _model["tau"]


            # Calcualte H C 
            model["H"] = CompositeRigidBodyAlgorithm(model, q)
            model["C"] = InverseDynamics(model, q, qdot, jnp.zeros((NB, 1)))
            model["Hinv"] = inv(model["H"])


            contact_cond = dict()
            contact_cond["contact_pos_lb"] = jnp.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
            contact_cond["contact_pos_ub"] = jnp.array([0.0001, 0.0001, 0.0001]).reshape(-1, 1)
            contact_cond["contact_vel_lb"] = jnp.array([-0.05, -0.05, -0.05]).reshape(-1, 1)
            contact_cond["contact_vel_ub"] = jnp.array([0.01, 0.01, 0.01]).reshape(-1, 1)
            
            #forward dynamics
            T = self.dt
            contact_force = dict()

            model["contact_cond"] = contact_cond

            # Calculate contact force in joint space
            # flag_contact = DetectContact(model, q, qdot, contact_cond)
            flag_contact_tuple = DetectContact(model, q, qdot)
            flag_contact_list = []
            flag_contact_list.append(flag_contact_tuple)
            print("flag_contact_list",flag_contact_list)
            flag_contact = jnp.array(flag_contact_list).flatten()
            print("flag_contact",flag_contact)
            # print("In Dynamics!!!")
            # print(flag_contact)
            if jnp.sum(flag_contact) !=0: 
                # lambda, fqp, fpd] = SolveContactLCP(q, qdot, tau, flag_contact);
                # lam, fqp, fc, fcqp, fcpd = CalcContactForceDirect(_model, q, qdot, tau, flag_contact, 2)
                lam, fqp, fc, fcqp, fcpd = SolveContactLCP(_model, q, qdot, tau, flag_contact,0.1)
                contact_force["fc"] = fc
                contact_force["fcqp"] = fcqp
                contact_force["fcpd"] = fcpd
            else:
                # print("No Conatact")
                lam = jnp.zeros((NB, 1))
                contact_force["fc"] = jnp.zeros((3*NC, 1))
                contact_force["fcqp"] = jnp.zeros((3*NC, 1))
                contact_force["fcpd"] = jnp.zeros((3*NC, 1))


            # Forward dynamics
            Tau = tau + lam
            qddot = ForwardDynamics(model, q, qdot, Tau).flatten()

            qdot_hat = qdot + qddot * T
            q_hat = q + qdot * T

            return jnp.array([q_hat, qdot_hat])
            

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

        reward = self.reward_func(self.state)

        if (len(qdot[qdot>self.qdot_threshold]) >0):
        # if (len(q[q>self.q_threshold]) >0):  
            # print("q in done",q)
            done = True
            reward += 10

        return self.state, reward, done, {}


    def reward_func(self,state):
        # # x, x_dot, theta, theta_dot = state
        # reward = state[0]**2 + (state[1])**2 + 100*state[2]**2 + state[3]**2 
        # # reward = jnp.exp(state[0])-1 + state[2]**2 + state[3]**2 
        q, qdot = state
        # reward = jnp.log(jnp.sum(jnp.square(q - self.target))) + jnp.log(jnp.sum(jnp.square(qdot - self.qdot_target)))
        # reward = jnp.log(jnp.sum(jnp.square(q - self.target))) 
        reward = jnp.sum(jnp.square(q - self.target)) 
        # reward = jnp.log((q[6]-0.9)**2) + jnp.log((q[9]-0.9)**2) 

        return reward


    def osim_render(self):
        q, _ = self.state
        # print("q for render",q)
        self.rder.step_render(q)