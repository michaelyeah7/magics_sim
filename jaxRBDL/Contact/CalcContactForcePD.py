import numpy as np
from jaxRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from jaxRBDL.Kinematics.CalcPointVelocity import CalcPointVelocity
import jax.numpy as jnp
import jax


def CalcContactForcePD(model: dict, q: jnp.ndarray, qdot: jnp.ndarray, 
                       flag_contact: jnp.ndarray, contact_force_kp: jnp.ndarray,
                       contact_force_kd: jnp.ndarray, nf: int=3)->jnp.ndarray:
    
    NC = int(model["NC"])
    NB = int(model["NB"])
    q = q.flatten()
    qdot = qdot.flatten()
    flag_contact = flag_contact.flatten()
    contact_force_kp = contact_force_kp.flatten()
    contact_force_kd = contact_force_kd.flatten()

    try: 
        idcontact = jnp.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = jnp.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]



    if jnp.all(flag_contact==0):
        fpd = jnp.zeros((NC*nf, 1))

    else:
        endpos = jnp.zeros((3, NC))
        endvel = jnp.zeros((3, NC))
        fpd = []

        for i in range(NC):
            if flag_contact[i] != 0:
                # Calcualte pos and vel of foot endpoint
                # endpos[:, i:i+1] = CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i])
                # endvel[:, i:i+1] = CalcPointVelocity(model, q, qdot, idcontact[i], contactpoint[i])

                pos = CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i])
                vel = CalcPointVelocity(model, q, qdot, idcontact[i], contactpoint[i])
                endpos = jax.ops.index_update(endpos,jax.ops.index[:,i:i+1],pos)
                endvel = jax.ops.index_update(endvel,jax.ops.index[:,i:i+1],vel)
                
                # Calculate contact force by PD controller

                if nf==2:
                    fpdi = jnp.zeros((2, 1))
                    # fpdi[0, 0] = -contact_force_kp[1]*endvel[0, i]
                    # fpdi[1, 0] = -contact_force_kp[2]*endpos[2, i] - contact_force_kd[2] * min(endvel[2, i], 0.0)
                    fpdi = jax.ops.index_update(fpdi,jax.ops.index[0,0],-contact_force_kp[1]*endvel[0, i])
                    fpdi = jax.ops.index_update(fpdi,jax.ops.index[1,0],-contact_force_kp[2]*endpos[2, i] - contact_force_kd[2] * min(endvel[2, i], 0.0))
                elif nf==3:
                    fpdi = jnp.zeros((3, 1))
                    # fpdi[0, 0] = -contact_force_kp[0] * endvel[0, i]
                    # fpdi[1, 0] = -contact_force_kp[1] * endvel[1, i]
                    # fpdi[2, 0] = -contact_force_kp[2] * min(endpos[2, i], 0.0) - contact_force_kd[2] * min(endvel[2, i], 0.0)
                    fpdi = jax.ops.index_update(fpdi,jax.ops.index[0,0],-contact_force_kp[0]*endvel[0, i])
                    fpdi = jax.ops.index_update(fpdi,jax.ops.index[1,0],-contact_force_kp[1]*endvel[1, i])
                    fpdi = jax.ops.index_update(fpdi,jax.ops.index[2,0],-contact_force_kp[2] * min(endpos[2, i], 0.0) - contact_force_kd[2] * min(endvel[2, i], 0.0))
                else:
                    fpdi = jnp.empty((0, 1))
             
                fpd.append(fpdi)
        fpd = jnp.asarray(jnp.concatenate(fpd, axis=0))

    return fpd
    
