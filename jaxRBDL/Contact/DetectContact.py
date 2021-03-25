import numpy as np
from jaxRBDL.Kinematics.CalcBodyToBaseCoordinates import CalcBodyToBaseCoordinates
from jaxRBDL.Kinematics.CalcPointVelocity import CalcPointVelocity
import jax.numpy as jnp

def DeterminContactType(pos: jnp.ndarray, vel: jnp.ndarray, contact_cond: dict)->int:
    pos = pos.flatten()
    vel = vel.flatten()
    contact_pos_lb = contact_cond["contact_pos_lb"].flatten()
    contact_vel_lb = contact_cond["contact_vel_lb"].flatten()
    contact_vel_ub = contact_cond["contact_vel_ub"].flatten()
    # print("------------")
    # print(pos)
    # print(vel)


    if pos[2] < contact_pos_lb[2]:
        if vel[2] < contact_vel_lb[2]:
            contact_type = 2 # impact    
        elif vel[2] > contact_vel_ub[2]:
            contact_type = 0 # uncontact
        else:
            contact_type = 1 # contact 
    else:
        contact_type = 0  # uncontact
    
    # if(contact_type !=0):
    #     print("------------")
    #     print(pos)
    #     print(vel)

    return contact_type

def  DetectContact(model: dict, q: jnp.ndarray, qdot: jnp.ndarray, contact_cond: dict)->jnp.ndarray:
    NC = int(model["NC"])

    try: 
        idcontact = jnp.squeeze(model["idcontact"], axis=0).astype(int)
        contactpoint = jnp.squeeze(model["contactpoint"], axis=0)
    except:
        idcontact = model["idcontact"]
        contactpoint = model["contactpoint"]

    flag_contact = jnp.zeros((NC, 1))


    flag_contact_list = []

    for i in range(NC):
        # Calcualte pos and vel of foot endpoint, column vector
        endpos_item = CalcBodyToBaseCoordinates(model, q, idcontact[i], contactpoint[i])
        endvel_item = CalcPointVelocity(model, q, qdot, idcontact[i], contactpoint[i])

        # Detect contact
        flag_contact_list.append(DeterminContactType(endpos_item, endvel_item, contact_cond))

    flag_contact = jnp.asarray(flag_contact_list).reshape((-1, 1))

    return flag_contact
    
