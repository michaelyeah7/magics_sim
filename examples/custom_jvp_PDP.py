import jax.numpy as jnp
from jax import custom_jvp

@custom_jvp
def fa(x, y):
  return jnp.sin(x) * y

@fa.defjvp
def f_jvp(primals, tangents):
  x, y = primals
  x_dot, y_dot = tangents
  primal_out = fa(x, y)
  tangent_out = 2 * jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
  return primal_out, tangent_out

from jax import jvp, grad

print(fa(2., 3.))
y, y_dot = jvp(fa, (2., 3.), (1., 0.))
print(y)
print(y_dot)
print(grad(fa,argnums=0)(2., 3.))