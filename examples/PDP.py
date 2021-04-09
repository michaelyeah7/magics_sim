from casadi import *
import numpy
from scipy import interpolate
import math
import time


class CartPole():
    def __init__(self, project_name='cart-pole-system'):
        self.project_name = project_name

    def initDyn(self, mc=None, mp=None, l=None):
        # set the global parameters
        g = 10

        # declare system parameters
        parameter = []
        if mc is None:
            self.mc = SX.sym('mc')
            parameter += [self.mc]
        else:
            self.mc = mc

        if mp is None:
            self.mp = SX.sym('mp')
            parameter += [self.mp]
        else:
            self.mp = mp
        if l is None:
            self.l = SX.sym('l')
            parameter += [self.l]
        else:
            self.l = l
        self.dyn_auxvar = vcat(parameter)

        # Declare system variables
        self.x, self.q, self.dx, self.dq = SX.sym('x'), SX.sym('q'), SX.sym('dx'), SX.sym('dq')
        self.X = vertcat(self.x, self.q, self.dx, self.dq)
        self.U = SX.sym('u')
        ddx = (self.U + self.mp * sin(self.q) * (self.l * self.dq * self.dq + g * cos(self.q))) / (
                self.mc + self.mp * sin(self.q) * sin(self.q))  # acceleration of x
        ddq = (-self.U * cos(self.q) - self.mp * self.l * self.dq * self.dq * sin(self.q) * cos(self.q) - (
                self.mc + self.mp) * g * sin(
            self.q)) / (
                      self.l * self.mc + self.l * self.mp * sin(self.q) * sin(self.q))  # acceleration of theta
        self.f = vertcat(self.dx, self.dq, ddx, ddq)  # continuous dynamics

    def initCost(self, wx=None, wq=None, wdx=None, wdq=None, wu=0.001):
        # declare system parameters
        parameter = []
        if wx is None:
            self.wx = SX.sym('wx')
            parameter += [self.wx]
        else:
            self.wx = wx

        if wq is None:
            self.wq = SX.sym('wq')
            parameter += [self.wq]
        else:
            self.wq = wq
        if wdx is None:
            self.wdx = SX.sym('wdx')
            parameter += [self.wdx]
        else:
            self.wdx = wdx

        if wdq is None:
            self.wdq = SX.sym('wdq')
            parameter += [self.wdq]
        else:
            self.wdq = wdq
        self.cost_auxvar = vcat(parameter)

        X_goal = [0.0, math.pi, 0.0, 0.0]

        self.path_cost = self.wx * (self.x - X_goal[0]) ** 2 + self.wq * (self.q - X_goal[1]) ** 2 + self.wdx * (
                self.dx - X_goal[2]) ** 2 + self.wdq * (
                                 self.dq - X_goal[3]) ** 2 + wu * (self.U * self.U)
        self.final_cost = self.wx * (self.x - X_goal[0]) ** 2 + self.wq * (self.q - X_goal[1]) ** 2 + self.wdx * (
                self.dx - X_goal[2]) ** 2 + self.wdq * (
                                  self.dq - X_goal[3]) ** 2  # final cost    

'''
# =============================================================================================================
# This class is used to solve the motion planning or optimal control problems:
# The standard form of the system dynamics is
# x_k+1= f（x_k, u_k)
# The standard form of the cost function is
# J = sum_0^(T-1) path_cost +final cost,
# where
# path_cost = c(x, u)
# final_cost= h(x)

# Note that most of the notations used below are consistent with the notations defined in the PDP paper
# Particularly the learning mode 3 of the PDP framework

# An important feature of this mode is that we have implemented in the code the warping techniques:
# please see all the below methods beginning with 'warped_****'
# The advantage of using the warping technique is that it is more robust and converge much faster,
# However, the disadvantage of this that the control trajectory are dicretized too much
# Users can first use the warped PDP to get the coarse solutions, and then use the PDP to refine it.
'''


class ControlPlanning:

    def __init__(self, project_name="planner"):
        self.project_name = project_name

    def setStateVariable(self, state, state_lb=[], state_ub=[]):
        self.state = state
        self.n_state = self.state.numel()
        if len(state_lb) == self.n_state:
            self.state_lb = state_lb
        else:
            self.state_lb = self.n_state * [-1e20]

        if len(state_ub) == self.n_state:
            self.state_ub = state_ub
        else:
            self.state_ub = self.n_state * [1e20]

    def setControlVariable(self, control, control_lb=[], control_ub=[]):
        self.control = control
        self.n_control = self.control.numel()

        if len(control_lb) == self.n_control:
            self.control_lb = control_lb
        else:
            self.control_lb = self.n_control * [-1e20]

        if len(control_ub) == self.n_control:
            self.control_ub = control_ub
        else:
            self.control_ub = self.n_control * [1e20]

    def setDyn(self, ode):
        self.dyn = ode
        self.dyn_fn = casadi.Function('dynFun', [self.state, self.control], [self.dyn])

        # Differentiate system dynamics
        self.dfx = jacobian(self.dyn, self.state)
        self.dfx_fn = casadi.Function('dfx', [self.state, self.control], [self.dfx])
        self.dfu = jacobian(self.dyn, self.control)
        self.dfu_fn = casadi.Function('dfu', [self.state, self.control], [self.dfu])

    def setPathCost(self, path_cost):
        self.path_cost = path_cost
        self.path_cost_fn = casadi.Function('pathCost', [self.state, self.control], [self.path_cost])

        # Differentiate the objective (cost) functions: the notations used here are the consistent with the notations
        # defined in the PDP paper
        # This is in fact belonging to diffPMP, but we just add it here
        self.dcx_fn = casadi.Function('dcx', [self.state, self.control], [jacobian(self.path_cost, self.state)])
        self.dcu_fn = casadi.Function('dcx', [self.state, self.control], [jacobian(self.path_cost, self.control)])

    def setFinalCost(self, final_cost):
        self.final_cost = final_cost
        self.final_cost_fn = casadi.Function('finalCost', [self.state], [self.final_cost])

        # Differentiate the final cost function
        self.dhx_fn = casadi.Function('dhx', [self.state], [jacobian(self.final_cost, self.state)])

    def setPolyControl(self, pivots):
        # Use the Lagrange polynomial to represent the control (input) function: u_t=u(t,auxvar).
        # Note that here we still use auxvar to denote the unknown parameter

        # time variable
        self.t = SX.sym('t')

        # pivots of time steps for the Lagrange polynomial
        poly_control = 0
        piovts_control = []
        for i in range(len(pivots)):
            Ui = SX.sym('U_' + str(i), self.n_control)
            piovts_control += [Ui]
            bi = 1
            for j in range(len(pivots)):
                if j != i:
                    bi = bi * (self.t - pivots[j]) / (pivots[i] - pivots[j])
            poly_control = poly_control + bi * Ui
        self.auxvar = vcat(piovts_control)
        self.n_auxvar = self.auxvar.numel()
        self.policy_fn = casadi.Function('policy_fn', [self.t, self.state, self.auxvar], [poly_control])

        # Differentiate control policy function
        dpolicy_dx = casadi.jacobian(poly_control, self.state)
        self.dpolicy_dx_fn = casadi.Function('dpolicy_dx', [self.t, self.state, self.auxvar], [dpolicy_dx])
        dpolicy_de = casadi.jacobian(poly_control, self.auxvar)
        self.dpolicy_de_fn = casadi.Function('dpolicy_de', [self.t, self.state, self.auxvar], [dpolicy_de])

    def setNeuralPolicy(self,hidden_layers):
        # Use neural network to represent the policy function: u_t=u(t,x,auxvar).
        # Note that here we use auxvar to denote the parameter of the neural policy
        layers=hidden_layers+[self.n_control]

        # time variable
        self.t = SX.sym('t')

        # construct the neural policy with the argument inputs to specify the hidden layers of the neural policy
        a=self.state
        auxvar=[]
        Ak = SX.sym('Ak', layers[0], self.n_state)  # weights matrix
        bk = SX.sym('bk', layers[0])  # bias vector
        auxvar += [Ak.reshape((-1, 1))]
        auxvar += [bk]
        a=mtimes(Ak, a) + bk
        for i in range(len(layers)-1):
            a=tanh(a)
            Ak = SX.sym('Ak', layers[i+1],layers[i] )  # weights matrix
            bk = SX.sym('bk', layers[i+1])  # bias vector
            auxvar += [Ak.reshape((-1, 1))]
            auxvar += [bk]
            a = mtimes(Ak, a) + bk
        self.auxvar=vcat(auxvar)
        self.n_auxvar = self.auxvar.numel()
        neural_policy=a
        self.policy_fn = casadi.Function('policy_fn', [self.t, self.state, self.auxvar], [neural_policy])

        # Differentiate control policy function
        dpolicy_dx = casadi.jacobian(neural_policy, self.state)
        self.dpolicy_dx_fn = casadi.Function('dpolicy_dx', [self.t, self.state, self.auxvar], [dpolicy_dx])
        dpolicy_de = casadi.jacobian(neural_policy, self.auxvar)
        self.dpolicy_de_fn = casadi.Function('dpolicy_de', [self.t, self.state, self.auxvar], [dpolicy_de])

    # The following are to solve PDP control and planning with polynomial control policy

    def integrateSys(self, ini_state, horizon, auxvar_value):
        assert hasattr(self, 'dyn_fn'), "Set the dynamics first!"
        assert hasattr(self, 'policy_fn'), "Set the control policy first, you may use [setPolicy_polyControl] "

        if type(ini_state) == list:
            ini_state = numpy.array(ini_state)

        # do the system integration
        control_traj = numpy.zeros((horizon, self.n_control))
        state_traj = numpy.zeros((horizon + 1, self.n_state))
        state_traj[0, :] = ini_state
        cost = 0
        for t in range(horizon):
            curr_x = state_traj[t, :]
            curr_u = self.policy_fn(t, curr_x, auxvar_value).full().flatten()
            state_traj[t + 1, :] = self.dyn_fn(curr_x, curr_u).full().flatten()
            control_traj[t, :] = curr_u
            cost += self.path_cost_fn(curr_x, curr_u).full()
        cost += self.final_cost_fn(state_traj[-1, :]).full()

        traj_sol = {'state_traj': state_traj,
                    'control_traj': control_traj,
                    'cost': cost.item()}
        return traj_sol

    def getAuxSys(self, state_traj, control_traj, auxvar_value):
        # check the pre-requisite conditions
        assert hasattr(self, 'dfx_fn'), "Set the dynamics equation first!"
        assert hasattr(self, 'dpolicy_de_fn'), "Set the policy first, you may want to use method [setPolicy_]"
        assert hasattr(self, 'dpolicy_dx_fn'), "Set the policy first, you may want to use method [setPolicy_]"

        # Initialize the coefficient matrices of the auxiliary control system: note that all the notations used here are
        # consistent with the notations defined in the PDP paper.
        dynF, dynG = [], []
        dUx, dUe = [], []
        for t in range(numpy.size(control_traj, 0)):
            curr_x = state_traj[t, :]
            curr_u = control_traj[t, :]
            start_time = time.time()
            dynF += [self.dfx_fn(curr_x, curr_u).full()]
            print("jacobian takes %s seconds ---" % (time.time() - start_time))
            dynG += [self.dfu_fn(curr_x, curr_u).full()]
            dUx += [self.dpolicy_dx_fn(t, curr_x, auxvar_value).full()]
            dUe += [self.dpolicy_de_fn(t, curr_x, auxvar_value).full()]

        auxSys = {"dynF": dynF,
                  "dynG": dynG,
                  "dUx": dUx,
                  "dUe": dUe}

        return auxSys

    def integrateAuxSys(self, dynF, dynG, dUx, dUe, ini_condition):

        # pre-requisite check
        if type(dynF) != list or type(dynG) != list or type(dUx) != list or type(dUe) != list:
            assert False, "The input dynF, dynE, dUx, and dUe should be list of numpy.array!"
        if len(dynG) != len(dynF) or len(dUe) != len(dUx) or len(dUe) != len(dynG):
            assert False, "The length of dynF, dynE, dUx, and dUe should be the same"
        if type(ini_condition) is not numpy.ndarray:
            assert False, "The initial condition should be numpy.array"

        horizon = len(dynF)
        state_traj = [ini_condition]
        control_traj = []
        for t in range(horizon):
            F_t = dynF[t]
            G_t = dynG[t]
            Ux_t = dUx[t]
            Ue_t = dUe[t]
            X_t = state_traj[t]
            U_t = numpy.matmul(Ux_t, X_t) + Ue_t
            state_traj += [numpy.matmul(F_t, X_t) + numpy.matmul(G_t, U_t)]
            control_traj += [U_t]

        aux_sol = {'state_traj': state_traj,
                   'control_traj': control_traj}
        return aux_sol

    def init_step(self, horizon, n_poly=5):
        # set the control polynomial policy
        pivots = numpy.linspace(0, horizon, n_poly + 1)
        self.setPolyControl(pivots)

    def init_step_neural_policy(self, hidden_layers=None):
        if hidden_layers is None:
            hidden_layers=[self.n_state]
        self.setNeuralPolicy(hidden_layers)

    def step(self, ini_state, horizon, auxvar_value):

        assert hasattr(self, 'policy_fn'), 'please set the control policy by running the init_step method first!'

        # generate the system trajectory using the current policy
        sol = self.integrateSys(ini_state=ini_state, horizon=horizon, auxvar_value=auxvar_value)
        state_traj = sol['state_traj']
        control_traj = sol['control_traj']
        loss = sol['cost']

        #  establish the auxiliary control system
        aux_sys = self.getAuxSys(state_traj=state_traj, control_traj=control_traj, auxvar_value=auxvar_value)
        # solve the auxiliary control system
        aux_sol = self.integrateAuxSys(dynF=aux_sys['dynF'], dynG=aux_sys['dynG'],
                                       dUx=aux_sys['dUx'], dUe=aux_sys['dUe'],
                                       ini_condition=numpy.zeros((self.n_state, self.n_auxvar)))
        dxdauxvar_traj = aux_sol['state_traj']
        dudauxvar_traj = aux_sol['control_traj']

        # Evaluate the current loss and the gradients
        dauxvar = numpy.zeros(self.n_auxvar)
        for t in range(horizon):
            # chain rule
            dauxvar += (numpy.matmul(self.dcx_fn(state_traj[t, :], control_traj[t, :]).full(), dxdauxvar_traj[t]) +
                        numpy.matmul(self.dcu_fn(state_traj[t, :], control_traj[t, :]).full(),
                                     dudauxvar_traj[t])).flatten()
        dauxvar += numpy.matmul(self.dhx_fn(state_traj[-1, :]).full(), dxdauxvar_traj[-1]).flatten()

        return loss, dauxvar


if __name__ == "__main__":

    # --------------------------- load environment ----------------------------------------
    cartpole = CartPole()
    mc, mp, l = 0.1, 0.1, 1
    cartpole.initDyn(mc=mc, mp=mp, l=l)
    wx, wq, wdx, wdq, wu = 0.1, 0.6, 0.1, 0.1, 0.3
    cartpole.initCost(wx=wx, wq=wq, wdx=wdx, wdq=wdq, wu=wu)

    # --------------------------- create PDP Control/Planning object ----------------------------------------
    dt = 0.05
    horizon = 25
    ini_state = [0, 0, 0, 0]
    oc = ControlPlanning()
    print("cartpole.X",type(cartpole.X))
    oc.setStateVariable(cartpole.X)
    oc.setControlVariable(cartpole.U)
    dyn = cartpole.X + dt * cartpole.f
    print("dyn",dyn)
    oc.setDyn(dyn)
    oc.setPathCost(cartpole.path_cost)
    oc.setFinalCost(cartpole.final_cost)


    # --------------------------- do the system control and planning ----------------------------------------
    for j in range(5): # trial loop
        # learning rate
        lr = 1e-4
        loss_trace, parameter_trace = [], []
        oc.init_step_neural_policy(hidden_layers=[oc.n_state,oc.n_state])
        initial_parameter = numpy.random.randn(oc.n_auxvar)
        current_parameter = initial_parameter
        max_iter = 5000
        start_time = time.time()
        for k in range(int(max_iter)):
            # one iteration of PDP
            loss, dp = oc.step(ini_state, horizon, current_parameter)
            # update
            current_parameter -= lr * dp
            loss_trace += [loss]
            parameter_trace += [current_parameter]
            # print
            if k % 100 == 0:
                print('trial:', j ,'Iter:', k, 'loss:', loss)



