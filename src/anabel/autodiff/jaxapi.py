import jax
import jax.numpy as jnp
from jax import jacfwd, jacrev, grad




def potential_grad(func):
    pass


def stiffness_matrix(f, nf, mode='fwd',jit=False):
    F = lambda x, **params: f(x,**params)
    if mode == 'fwd':
        Df = lambda x, **params: jnp.squeeze(jax.jacfwd(f)(x,**params))[:nf,:nf]
        if jit: Df = jax.jit(Df)
        return Df


