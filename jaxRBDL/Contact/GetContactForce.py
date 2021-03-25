import numpy as np
import jax.numpy as jnp
import jax

def GetContactForce(model: dict, fqp: jnp.ndarray, fpd: jnp.ndarray, flag_contact: jnp.ndarray, nf:int):
    fqp = fqp.flatten()
    fpd = fpd.flatten()

    NC = int(model["NC"])
    fc = jnp.zeros((3*NC,))
    fcqp = jnp.zeros((3*NC,))
    fcpd = jnp.zeros((3*NC,))
    k = 0
    for i in range(NC):
        if flag_contact[i]!=0:
            if nf==2: # Only x/z direction
                # fc[3*i:3*i+3] = jnp.array([fqp[k*nf] + fpd[k*nf], 0.0, fqp[k*nf+nf-1] + fpd[k*nf+nf-1]])
                # fcqp[3*i:3*i+3] = jnp.array([fqp[k*nf], 0.0, fqp[k*nf+nf-1]])
                # fcpd[3*i:3*i+3] = jnp.array([fpd[k*nf], 0.0, fpd[k*nf+nf-1]])
                fc = jax.ops.index_update(fc,jax.ops.index[3*i:3*i+3],jnp.array([fqp[k*nf] + fpd[k*nf], 0.0, fqp[k*nf+nf-1] + fpd[k*nf+nf-1]]))
                fcqp = jax.ops.index_update(fcqp,jax.ops.index[3*i:3*i+3],jnp.array([fqp[k*nf], 0.0, fqp[k*nf+nf-1]]))
                fcpd = jax.ops.index_update(fcpd,jax.ops.index[3*i:3*i+3],jnp.array([fpd[k*nf], 0.0, fpd[k*nf+nf-1]]))
            else: 
                # fc[3*i:3*i+3] = fqp[k*nf:k*nf+nf] + fpd[k*nf:k*nf+nf] 
                # fcqp[3*i:3*i+3] = fqp[k*nf:k*nf+nf]
                # fcpd[3*i:3*i+3] = fpd[k*nf:k*nf+nf]
                fc = jax.ops.index_update(fc,jax.ops.index[3*i:3*i+3],fqp[k*nf:k*nf+nf] + fpd[k*nf:k*nf+nf])
                fcqp = jax.ops.index_update(fcqp,jax.ops.index[3*i:3*i+3],fqp[k*nf:k*nf+nf])
                fcpd = jax.ops.index_update(fcpd,jax.ops.index[3*i:3*i+3],fpd[k*nf:k*nf+nf])

            k = k+1

    fc = fc.reshape(-1, 1)
    fcqp = fcqp.reshape(-1, 1)
    fcpd = fcpd.reshape(-1, 1)
    return fc, fcqp, fcpd

