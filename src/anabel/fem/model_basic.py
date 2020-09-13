from functools import partial

from anabel.fem.utils import number_dof_basic

try:
    import jax.numpy as jnp
    import jax
except:
    import numpy as jnp



def basic_no_1(el: dict, mesh: dict, bn: dict, jit=False, verbose=False,debug=False,**kwds):
    ndf = max(len( con) for  con in mesh.values())
    nr  = sum(sum(boun) for boun in   bn.values())
    nodes = { node for con in mesh.values() for node in con[1] }
    nn = len( nodes )
    DOF = number_dof_basic( mesh, bn ,verbose=verbose)
    nf = ndf * nn - nr
    if verbose:
        print(f'nf: {nf}\nnr: {nr}')
        print(f'DOFs: {DOF}')
    
    el_sign = {  tag: 1.0**(n[1][0] > n[1][1]) for tag, n in mesh.items() }
    el_DOF  = { elem:[DOF[node] for node in mesh[elem][1]] for elem in sorted(mesh) }

    Be = jnp.concatenate([
         jnp.array([[1.0 if j==i else 0.0 for j in range(1,nf+1)] for i in sum(el_dofs , []) ]).T
         for tag, el_dofs in el_DOF.items() ], axis=1)
    
    params = {}

    x_e = lambda xyz, con: jnp.array(
        [xyz[node] for node in sorted(con[1],key=lambda k: int(k))]
    )
    u_e = lambda dx, el_tag: jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten()



    def f(dx, xyz, params):
        F = jnp.concatenate([
                el[con[0]](
                    u=u_e(dx, el_tag),
                    xyz=x_e(xyz, con),
                    **params[el_tag] )
                    for el_tag, con in sorted(mesh.items()) ],
                axis=0)
        return Be @ F 


    if 'xyz' in kwds: f = partial(f, xyz=kwds['xyz'])
    if jit: f = jax.jit(f)
    return f

