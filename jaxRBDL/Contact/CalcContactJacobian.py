import numpy as np
from jaxRBDL.Kinematics.CalcPointJacobian import CalcPointJacobian
import jax.numpy as jnp

def CalcContactJacobian(model: dict, q: jnp.ndarray, flag_contact: jnp.ndarray, nf: int=3)->jnp.ndarray:
    NC = int(model["NC"])
    NB = int(model["NB"])
    q = q.flatten()
    flag_contact = flag_contact.flatten()

    try: 
        idcontact = jnp.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = jnp.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    Jc = []
    for i in range(NC):
        Jci = jnp.empty((0, NB))
        if flag_contact[i] != 0.0:
            # Calculate Jacobian
            J = CalcPointJacobian(model, q, idcontact[i], contactpoint[i])

            # Make Jacobian full rank according to contact model
            if nf == 2:
                Jci = J[[0, 2], :] # only x\z direction
            elif nf == 3:
                Jci = J          
        Jc.append(Jci)

    Jc = jnp.asarray(jnp.concatenate(Jc, axis=0))
    return Jc
