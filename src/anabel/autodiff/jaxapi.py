import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, grad




def potential_grad(func):
    pass


def stiffness_matrix(f, nf, mode='fwd'):
    F = lambda x, **params: f(x,**params)
    if mode == 'fwd':
        f_new = lambda x, **params: jnp.squeeze(jax.jacfwd(f)(x,**params))[:nf,:nf]
        return jax.jit(f_new)


