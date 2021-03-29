import re
import numpy as np
from numpy.core.defchararray import array
from numpy.core.numeric import flatnonzero
from pyRBDL.Dynamics.CompositeRigidBodyAlgorithm import CompositeRigidBodyAlgorithm
from pyRBDL.Dynamics.InverseDynamics import InverseDynamics
from numpy.linalg import inv
from pyRBDL.Contact.DetectContact import DetectContact
from pyRBDL.Contact.CalcContactForceDirect import CalcContactForceDirect
from pyRBDL.Dynamics.ForwardDynamics import ForwardDynamics
from pyRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from pyRBDL.Contact.ImpulsiveDynamics import ImpulsiveDynamics

def solver_ode(X: np.ndarray, model: dict, flag_contact: np.array,T:float):
    """
    param:
        X: current q, qdot
        model: add tau, ST
    return:
        Xdot: next qdot, qddot
    """
    from scipy.integrate import solve_ivp
    
    status = -1
    # Calculate state vector by ODE
    t0 = 0
    tf = T
    tspan = (t0, tf)
    
    NB=model['NB']
    q,qdot = np.array(X[0:NB]), np.array(X[NB:2*NB])
    x0 = np.asfarray(np.hstack([q, qdot]))
    
    contact_force = dict()

    print("flag contact",flag_contact,model['contactpoint'])

    while status != 0:
        # ODE calculate 
        sol = solve_ivp(dynamics_fun, tspan, x0, args=(model, flag_contact,contact_force),\
                            method='RK45', rtol=1e-3, atol=1e-4)
        status = sol.status
        if(status == -1):
            print("error message:",sol.message)
        assert status != -1, "Integration Failed!!!"

        if status == 0:
            print("The solver successfully reached the end of tspan.")
            pass

        if status == 1:
            print("A termination event occurred")
            t_events = sol.t_events
            te_idx = t_events.index(min(t_events))
            te = float(t_events[te_idx])
            xe = sol.y_events[te_idx].flatten()

            # Get q qdot
            print("xe",xe)
            q = xe[0:NB]
            qdot = xe[NB:2* NB]

            # Detect contact TODO
            flag_contact = np.array(flag_contact)#DetectContact(model, q, qdot, contact_cond)

            # Impact dynamics TODO
            qdot_impulse = ImpulsiveDynamics(model, q, qdot, flag_contact, nf=2);  
            qdot_impulse = qdot_impulse.flatten()

            # Update initial state
            x0 = np.hstack([q, qdot_impulse])
            tspan = (te, tf)

    xk = sol.y[:, -1]
    # print("solved xk",xk[0:NB])
    return xk, contact_force

def dynamics_fun(t: float, X: np.ndarray, model: dict, flag_contact: np.array, contact_force: dict)->np.ndarray:
    """
    param:
        X: current q, qdot
        model: add tau, ST
    return:
        Xdot: next qdot, qddot
    """

    NB = int(model["NB"])
    NC = int(model["NC"])

    # Get q qdot tau
    q = X[0:NB]
    qdot = X[NB: 2 * NB]
    tau = model["tau"]

    # Calcualte H C 
    model["H"] = CompositeRigidBodyAlgorithm(model, q)
    model["C"] = InverseDynamics(model, q, qdot, np.zeros((NB, 1)))
    model["Hinv"] = inv(model["H"])

    # Calculate contact force in joint space
    if np.sum(flag_contact) !=0: 
        lam, fqp, fc, fcqp, fcpd = CalcContactForceDirect(model, q, qdot, tau, flag_contact, 2)
        contact_force["fc"] = fc
        contact_force["fcqp"] = fcqp
        contact_force["fcpd"] = fcpd
    else:
        lam = np.zeros((NB, 1))
        contact_force["fc"] = np.zeros((3*NC, 1))
        contact_force["fcqp"] = np.zeros((3*NC, 1))
        contact_force["fcpd"] = np.zeros((3*NC, 1))


    # Forward dynamics
    Tau = tau + lam
    qddot = ForwardDynamics(model, q, qdot, Tau).flatten()

    # Return Xdot
    Xdot = np.asfarray(np.hstack([qdot, qddot]))

    # print("Xdot in dynamics",Xdot)
    return Xdot