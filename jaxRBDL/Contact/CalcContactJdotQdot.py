import numpy as np
from jaxRBDL.Kinematics.CalcPointAcceleraion import CalcPointAcceleration
import jax.numpy as jnp

def CalcContactJdotQdot(model: dict, q: jnp.ndarray, qdot: jnp.ndarray, flag_contact: jnp.ndarray, nf: int=3)->np.ndarray:
    NC = int(model["NC"])
    NB = int(model["NB"])
    q = q.flatten()
    qdot = qdot.flatten()
    flag_contact = flag_contact.flatten()

    try: 
        idcontact = jnp.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = jnp.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    
    JdotQdot = []
    for i in range(NC):
        JdotQdoti = jnp.empty((0, 1))
        if flag_contact[i] != 0.0:
            JdQd = CalcPointAcceleration(model, q, qdot, jnp.zeros((NB, 1)), idcontact[i], contactpoint[i])
            if nf == 2:
                JdotQdoti = JdQd[[0, 2], :] # only x\z direction
            elif nf == 3:
                JdotQdoti = JdQd
   

        JdotQdot.append(JdotQdoti)

    JdotQdot = jnp.asarray(jnp.concatenate(JdotQdot, axis=0))

    return JdotQdot
                
