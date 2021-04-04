from casadi import *
import numpy
from scipy import interpolate
import math
import time
import jax.numpy as jnp
from jax import jacfwd
import jax
from jax.api import jit

def casadi_jacobian():
    # Declare system variables
    g = 10.0
    mc, mp, l = 0.1, 0.1, 1
    x, q, dx, dq = SX.sym('x'), SX.sym('q'), SX.sym('dx'), SX.sym('dq')
    X = vertcat(x, q, dx, dq)
    U = SX.sym('u')
    ddx = (U + mp * sin(q) * (l * dq * dq + g * cos(q))) / (
            mc + mp * sin(q) * sin(q))  # acceleration of x
    ddq = (-U * cos(q) - mp * l * dq * dq * sin(q) * cos(q) - (
            mc + mp) * g * sin(
        q)) / (
                    l * mc + l * mp * sin(q) * sin(q))  # acceleration of theta
    f = vertcat(dx, dq, ddx, ddq)  # continuous dynamics

    dt = 0.05
    dyn = X + dt * f

    dfx = jacobian(dyn, X)
    dfx_fn = casadi.Function('dfx', [X,U], [dfx])

    _x = vertcat(1,1,1,1)
    _u = 1
    start_time = time.time()
    dynF = dfx_fn(_x,_u).full()
    print("casadi jacobian takes %s seconds ---" % (time.time() - start_time))
    # print("casadi jacobian",dynF)

@jit
def dynamics(X,U):
    from jax.numpy import sin, cos
    # start_time = time.time()    
    g = 10.0
    mc, mp, l = 0.1, 0.1, 1
    # x, q, dx, dq = SX.sym('x'), SX.sym('q'), SX.sym('dx'), SX.sym('dq')
    # X = vertcat(x, q, dx, dq)
    x, q, dx, dq = X
    # print("x",x)
    # U = SX.sym('u')
    # print("array takes %s seconds ---" % (time.time() - start_time)) 

    ddx = (U + mp * sin(q) * (l * dq * dq + g * cos(q))) / (
            mc + mp * sin(q) * sin(q))  # acceleration of x
    ddq = (-U * cos(q) - mp * l * dq * dq * sin(q) * cos(q) - (
            mc + mp) * g * sin(
        q)) / (
                    l * mc + l * mp * sin(q) * sin(q))  # acceleration of theta
    # f = vertcat(dx, dq, ddx, ddq)  # continuous dynamics
    # print("dx",dx)
    # print("ddx",ddx)
    # dx = jnp.array([dx])
    # dq = jnp.array([dq])
    # ddx = ddx[0]
    # ddq = ddq[0]
       
    f = jnp.array([dx, dq, ddx, ddq])
    dt = 0.05

    next_X = X + dt * f    

    return next_X 

@jit
def jax_jacobian():
    import jax
    from jax.numpy import sin, cos           
    # fwdfunc = jax.jacfwd(dynamics,argnums=0)
    fwdfunc = jax.jit(jax.jacfwd(dynamics,argnums=0))
    _X = jnp.array([1., 1., 1., 1.])
    _U = 1.0
    start_time = time.time()
    jac = fwdfunc(_X,_U)
    print("jax jacobian takes %s seconds ---" % (time.time() - start_time))
    # print("jac",jac)


if __name__ == "__main__":
    casadi_jacobian()
    jax_jacobian()