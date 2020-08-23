from functools import partial

import jax
import jax.numpy as jnp


def LETruss(jit=False,**kwds):
    def f(u, x0, E, A, q0=0.0, params={} ):
        DX = x0[1,0] - x0[0,0]
        DY = x0[1,1] - x0[0,1]
        L = jnp.linalg.norm([DX,DY])

        v = (u[2]-u[0])*DX/L + (u[3]-u[1])*DY/L + ((u[2]-u[0])**2 + (u[3]-u[1])**2)/(2*L)
        dv = jnp.array([[-DX/L - (u[2]-u[0])/L],
                        [-DY/L - (u[3]-u[1])/L], 
                        [ DX/L + (u[2]-u[0])/L],
                        [ DY/L + (u[3]-u[1])/L]])
        dv = dU@dv
        e = v/L 
        q = E*A*e + q0
        return (dv*q, None)

    def Df(u, x0, E, A, q0=0.0):
        pass

    Df = jax.jit(partial(Df,**kwds))
    f = partial(f,**kwds)
    if jit: f = jax.jit(f)
    params = {}

    Space = object()
    Space.f = f 
    Space.Df = Df 
    Space.maps = (params, [4])
    
    return Space

