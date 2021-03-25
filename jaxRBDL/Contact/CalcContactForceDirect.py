from typing import Tuple
import numpy as np
from jaxRBDL.Contact.CalcContactJacobian import CalcContactJacobian
from jaxRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdot
from jaxRBDL.Contact.CalcContactForcePD import CalcContactForcePD
from jaxRBDL.Contact.GetContactForce import GetContactForce
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres
import jax.numpy as jnp
import jax

def CheckContactForce(model: dict, flag_contact: jnp.ndarray, fqp: jnp.ndarray, nf: int):
    NC = int(model["NC"])
    flag_contact = flag_contact.flatten()

    flag_recalc = 0
    flag_newcontact = flag_contact

    k = 0
    for i in range(NC):
        if flag_contact[i] != 0:
            if fqp[k*nf+nf-1, 0] < 0:
                # flag_newcontact[i] = 0
                flag_newcontact = jax.ops.index_update(flag_newcontact,i,0)
                flag_recalc = 1
                break
            k = k+1

    return flag_newcontact, flag_recalc

def CalcContactForceDirect(model: dict, q: jnp.ndarray, qdot: jnp.ndarray, tau: jnp.ndarray, flag_contact: jnp.ndarray, nf: int):
    NB = int(model["NB"])
    NC = int(model["NC"])
        
    flag_recalc = 1
    fqp = jnp.empty((0, 1))
    flcp = jnp.empty((0, 1))  
    while flag_recalc:
        if jnp.sum(flag_contact)==0:
            fqp = jnp.zeros((NC*nf, 1))
            flcp = jnp.zeros((NB, 1))
            fc = jnp.zeros((3*NC,))
            fcqp = jnp.zeros((3*NC,))
            fcpd = jnp.zeros((3*NC,))
            break

        # Calculate contact force
        Jc = CalcContactJacobian(model, q, flag_contact, nf)
        JcdotQdot = CalcContactJdotQdot(model, q, qdot, flag_contact, nf)

        M = jnp.matmul(jnp.matmul(Jc, model["Hinv"]), jnp.transpose(Jc))
        tau = tau.reshape(-1, 1)
        d0 = jnp.matmul(jnp.matmul(Jc, model["Hinv"]), tau - model["C"])
        d = jnp.add(d0, JcdotQdot )
        
        #TODO M may be sigular for nf=3 
        fqp = -jnp.linalg.solve(M,d)
     

        # Check whether the Fz is positive
        flag_contact, flag_recalc = CheckContactForce(model, flag_contact, fqp, nf)
        if flag_recalc == 0:
            flcp = jnp.matmul(jnp.transpose(Jc), fqp)
        
        contact_force_kp = jnp.array([10000.0, 10000.0, 10000.0])
        contact_force_kd = jnp.array([1000.0, 1000.0, 1000.0])

        # Calculate contact force from PD controller
        fpd = CalcContactForcePD(model, q, qdot, flag_contact, contact_force_kp, contact_force_kd, nf)
        fc, fcqp, fcpd = GetContactForce(model, fqp, fpd, flag_contact, nf)  


    return flcp, fqp, fc, fcqp, fcpd