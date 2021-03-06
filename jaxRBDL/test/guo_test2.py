import numpy as np
import jax.numpy as jnp
from jaxRBDL.Math.SpatialTransform import SpatialTransform
import sys
import os

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
raw_E = np.loadtxt(CURRENT_PATH+"/E.txt")
raw_r = np.loadtxt(CURRENT_PATH+"/r.txt")
E = []
r = []
X_tree = []
for i in range(7):
    E.append(jnp.array([raw_E[i*3],raw_E[i*3+1],raw_E[i*3+2]]))
    r.append(jnp.array([raw_r[i*3],raw_r[i*3+1],raw_r[i*3+2]]))
    X_tree.append(SpatialTransform(E[i], r[i]))

print(X_tree[1])
