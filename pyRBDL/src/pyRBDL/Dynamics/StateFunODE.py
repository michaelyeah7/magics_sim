import numpy as np
from oct2py import octave
from pyRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from pyRBDL.Dynamics.InverseDynamics import InverseDynamics
from numpy.linalg import inv

def DynamicsFun(model: dict, X: np.ndarray, tau: np.ndarray):
    
    NB = int(model["NB"])
    try: 
        ST = np.squeeze(model["ST"], axis=0)
    except:
        ST = model["ST"]



    # Get q qdot tau
    q = X[0:NB, 0]
    qdot = X[NB: 2 * NB, 0]
    tau = model["tau"]


    # Calcualte H C 
    model["H"] = CompositeRigidBodyAlgorithm(model, q)
    model["C"] = InverseDynamics(model, q, qdot, np.zeros((NB, 1)))
    model["Hinv"] = inv(model["H"])

    # Calculate contact force in joint space
    flag_contact = DetectContact(q, qdot)
    if sum(flag_contact)~=0 
        % [lambda, fqp, fpd] = SolveContactLCP(q, qdot, tau, flag_contact);
        [lambda, fqp] = CalcContactForceDirect(q, qdot, tau, flag_contact);
    else
        lambda = zeros(model.NB, 1);
    end

    % Forward dynamics
    Tau = tau + lambda;
    qddot = ForwardDynamics(model, q, qdot, Tau)

    % Return Xdot
    Xdot = [qdot; qddot];



def StateFunODE(model: dict, xk: np.ndarray, uk: np.ndarray, T: float) -> np.ndarray:

    NB = int(model["NB"])
    
    try: 
        ST = np.squeeze(model["ST"], axis=0)
    except:
        ST = model["ST"]

    # Get q qdot tau
    q = xk[0: NB, 0]
    qdot = xk[NB: 2*NB, 0]
    tau = np.matmul(ST, uk)
    model["tau"] = tau

    # Calculate state vector by ODE
    t0 = 0
    tf = T
    tspan = [t0, tf]

