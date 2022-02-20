from typing import Callable, Union, Sequence
import jax
import jax.numpy as jnp
from jax import grad


def jacfwd(fun: Callable, outnums: Union[int, Sequence[int]] = None,
            argnums: Union[int,Sequence[int]]=0, *,
            holomorphic: bool = False, squeeze=True, nf=None) -> Callable:
    if outnums is None:
        if nf is None:
            jac = jax.jacfwd(fun, argnums, holomorphic)
        else:
            jac = jax.jacfwd(lambda *args,**kwds: fun(*args,**kwds)[:nf,:nf], argnums, holomorphic)
    else:
        jac = jax.jacfwd(lambda *args,**kwds: fun(*args,**kwds)[outnums],argnums,holomorphic)
    
    if squeeze:
        return lambda *args, **kwds: jnp.squeeze(jac(*args,**kwds))
    else:
        return jac

def jacrev(fun: Callable, outnums: Union[int, Sequence[int]] = None,
            argnums: Union[int,Sequence[int]]=0, *,
            holomorphic: bool = False, squeeze=True) -> Callable:
    if outnums is None:
        jac = jax.jacrev(fun, argnums, holomorphic)
    else:
        jac = jax.jacrev(lambda *args,**kwds: fun(*args,**kwds)[outnums],argnums,holomorphic)
    if squeeze:
        return lambda *args, **kwds: jnp.squeeze(jac(*args,**kwds))
    else:
        return jac

def stiffness_matrix(f, nf, mode='fwd',jit=False):
    F = lambda x, **params: f(x,**params)
    if mode == 'fwd':
        Df = lambda x, **params: jnp.squeeze(jax.jacfwd(f)(x,**params))[:nf,:nf]
        if jit: Df = jax.jit(Df)
        return Df


