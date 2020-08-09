from functools import partial

from anabel.models.utils import number_dof_basic

try:
    import jax.numpy as jnp
    import jax
except:
    import numpy as jnp

def basic(el: dict, mesh: dict, bn: dict,jit=False, **kwds):
    ndf = max(len( con) for  con in mesh.values())
    nr  = sum(sum(boun) for boun in   bn.values())
    nodes = { node for con in mesh.values() for node in con[1] }
    nn = len( nodes )
    DOF = number_dof_basic( mesh, bn )
    nf = ndf * nn - nr
    
    el_sign = {  tag: 1.0**(n[1][0] > n[1][1]) for tag, n in mesh.items() }
    el_DOF  = { elem:[DOF[node] for node in mesh[elem][1]] for elem in sorted(mesh) }

    Be = jnp.concatenate([
         jnp.array([[1.0 if j==i else 0.0 for j in range(1,nf+1)] for i in sum(el_dofs , []) ]).T
         for tag, el_dofs in el_DOF.items() ], axis=1)
    
    params = {}

    def f(dx, x0, params):
        F = jnp.concatenate([
                el[con[0]](
                    u = jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                    x0 = jnp.array([ x0[node]  for node in  sorted(con[1]) ]),
                    **params[el_tag] )
                for el_tag, con in sorted(mesh.items( )) ] ,axis=0)

        return Be @ F 


    if 'x0' in kwds: f = partial(f, x0=kwds['x0'])

    if jit: f = jax.jit(f)
    
    # params = {'params': {elem: el[con[0]].params  for elem, con in mesh.items()}}
    
    return f

