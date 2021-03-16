from typing import Tuple
import numpy as np
from pyRBDL.Contact.CalcContactJacobian import CalcContactJacobian
from pyRBDL.Contact.CalcContactJdotQdot import CalcContactJdotQdot
import scipy.sparse.linalg as spla
from scipy.sparse.linalg import gmres

def CheckContactForce(model: dict, flag_contact: np.ndarray, fqp: np.ndarray, nf: int):
    NC = int(model["NC"])
    flag_contact = flag_contact.flatten()

    flag_recalc = 0
    flag_newcontact = flag_contact

    k = 0
    for i in range(NC):
        if flag_contact[i] != 0:
            if fqp[k*nf+nf-1, 0] < 0:
                flag_newcontact[i] = 0
                flag_recalc = 1
                break
            k = k+1

    return flag_newcontact, flag_recalc

def CalcContactForceDirect(model: dict, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, flag_contact: np.ndarray, nf: int)-> Tuple[np.ndarray, np.ndarray]:
    NB = int(model["NB"])
    NC = int(model["NC"])
        
    flag_recalc = 1
    fqp = np.empty((0, 1))
    flcp = np.empty((0, 1))  
    while flag_recalc:
        if np.sum(flag_contact)==0:
            fqp = np.zeros((NC*nf, 1))
            flcp = np.zeros((NB, 1))  
            break

        # Calculate contact force
        Jc = CalcContactJacobian(model, q, flag_contact, nf)
        JcdotQdot = CalcContactJdotQdot(model, q, qdot, flag_contact, nf)

        M = np.matmul(np.matmul(Jc, model["Hinv"]), np.transpose(Jc))
        d = np.add(np.matmul(np.matmul(Jc, model["Hinv"]), tau - model["C"]), JcdotQdot )
        
        #TODO M may be sigular for nf=3 
        fqp = -np.linalg.solve(M,d)
     

        # Check whether the Fz is positive
        flag_contact, flag_recalc = CheckContactForce(model, flag_contact, fqp, nf)
        if flag_recalc == 0:
            flcp = np.matmul(np.transpose(Jc), fqp)   

    return flcp, fqp