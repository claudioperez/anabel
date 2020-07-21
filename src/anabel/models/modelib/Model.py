import inspect
from collections import namedtuple, defaultdict
from functools import partial

from jax import numpy as jnp
import jax
arry = jnp.array

class Func:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)

def partial_state_model(ElemSpace: dict, mesh: dict, bn: dict,jit=False, **kwds):
    """
    ### Elastic Elements

    `f.maps: ( params => response )`

    ### Inelastic Elements

    Inelastic elements are identified by a `maps` property with a size greater than three.

    `f.maps: ( state => params => response )`
    """
    ndf = max(len(con) for con in mesh.values())
    nr = sum(sum(boun) for boun in bn.values())
    nodes = {node for con in mesh.values() for node in con[1]}
    nn = len(nodes)
    DOF = numDof_b(mesh,bn)
    nf = ndf * nn - nr
    
    el_DOF = {elem: [DOF[node] for node in mesh[elem][1]] for elem in mesh}

    Be = jnp.concatenate([
        jnp.array([[1.0  if j == i else 0.0 for j in range(1,nf+1)] for i in sum(el_dofs,[])]).T 
        for el_dofs in sorted(el_DOF).values()], axis=1)
    
    params = {}
    def fe(dx, el_tag, params):
        return el[mesh[el_tag][0]].f( jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                    x0 = arry([ kwds['x0'][node] for node in mesh[el_tag][1] ]),
                    **params[el_tag])
    
    if 'x0' in kwds:
        x0 = kwds['x0']

        def f(dx, params):
            F = jnp.concatenate([
                    el[con[0]].f(jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                        x0 = arry([ x0[node] for node in con[1] ]),
                        **params[el_tag] )[0]
                    
                    for el_tag, con in mesh.items()] ,axis=0)
            
            f = Be @ F
            return f , None

        def Df(dx, kwds):
            pass
    else:
        def f(dx, x0, params):

            F = jnp.concatenate([
                    el[con[0]].f( 
                        u = jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                        x0 = arry([ x0[node] for node in con[1] ]),
                        **params[el_tag] ) [ 0 ]
                    
                    for el_tag, con in mesh.items()] ,axis=0)

            f = Be @ F

            return f , None

        def Df(dx, x0, args):
            pass

    # state = State({elem: el[con[0]].state for elem, con in mesh.items()})
    if jit: f = jax.jit(f)
    # params = {'params': {elem: {'state': el[con[0]].params.get('state')}  for elem, con in mesh.items()}}

    params = {'params': {elem: el[con[0]].params  for elem, con in mesh.items()}}

    return Func(f=f, Df=Df, maps=(params, (nf,1)), clr=locals())
    
"""
 name: Model
  args:
  - name: ndm
    datatype: int
    description: Dimension of structural model
  - name: nn
    datatype: int
    description: Number of nodes in structural model
  - name: ne
    datatype: int
    description: Number of elements
  - name: nf
    datatype: int
    description: Number of free degrees of freedom
  - name: nt
    datatype: scalar
    description: Total number of degrees of freedom
  - name: ndf
    datatype: int Array
    shape: (1 x ne)
    description: Number of element DOFs/node
  - name: nen
    datatype: int Array
    shape: (1 x ne)
    description: Number of end nodes/element
  - name: XYZ
    datatype: float array
    shape: (ndm x nn)
    description: Node coordinates (nodes are stored column-wise)
  - name: BOUN
    datatype: scalar Array
    shape: (ndfmax x nn)
    description: Boundary conditions (nodes stored column-wise)
  - name: CON
"""



def model(el: dict, mesh: dict, bn: dict,jit=False, **kwds):
    ndf = max(len(con) for con in mesh.values())
    nr = sum(sum(boun) for boun in bn.values())
    nodes = {node for con in mesh.values() for node in con[1]}
    nn = len(nodes)
    DOF = numDof_b(mesh,bn)
    nf = ndf * nn - nr
    
    el_DOF = {elem: [DOF[node] for node in mesh[elem][1]] for elem in mesh}

    Be = jnp.concatenate([
        jnp.array([[1.0  if j == i else 0.0 for j in range(1,nf+1)] for i in sum(el_dofs,[])]).T 
        for el_dofs in sorted(el_DOF).values()], axis=1)
    
    params = {}
    def fe(dx, el_tag, params):
        return el[mesh[el_tag][0]].f( jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                    x0 = arry([ kwds['x0'][node] for node in mesh[el_tag][1] ]),
                    **params[el_tag])
    
    if 'x0' in kwds:
        x0 = kwds['x0']

        def f(dx, params):
            # print(params)
            # F = []
            # for el_tag, con in mesh.items():
            #     u = jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten()
            #     print(kwds[el_tag])
            #     print(con[0])
            #     f_elem = el[con[0]].f( u = u, x0 = arry([ x0[node] for node in con[1] ]), **kwds[el_tag] )
            #     print(f_elem)
            #     F = jnp.concatenate([F,f_elem],axis=0)
            F = jnp.concatenate([
                    el[con[0]].f(jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                        x0 = arry([ x0[node] for node in con[1] ]),
                        **params[el_tag] )[0]
                    
                    for el_tag, con in mesh.items()] ,axis=0)
            
            f = Be @ F
            return f , None

        def Df(dx, kwds):
            pass
    else:
        def f(dx, x0, params):

            F = jnp.concatenate([
                    el[con[0]].f( 
                        u = jnp.take(dx, jnp.array(el_DOF[el_tag], dtype='int32')-1).flatten(),
                        x0 = arry([ x0[node] for node in con[1] ]),
                        **params[el_tag] ) [ 0 ]
                    
                    for el_tag, con in mesh.items()] ,axis=0)

            f = Be @ F

            return f , None

        def Df(dx, x0, args):
            pass

    # state = State({elem: el[con[0]].state for elem, con in mesh.items()})
    if jit: f = jax.jit(f)
    # params = {'params': {elem: {'state': el[con[0]].params.get('state')}  for elem, con in mesh.items()}}

    params = {'params': {elem: el[con[0]].params  for elem, con in mesh.items()}}
    
    return Elem(f, Df, params, locals())


def numDof_b(mesh, bn, debug=False):
    """Basic dof numbering"""
    ndf = max(len(con) for con in mesh.values())
    nr = sum(sum(boun) for boun in bn.values())
    nodes = {node for con in mesh.values() for node in con[1]}
    nn = len(nodes)

    crxns = ndf*nn - nr + 1

    df = 1
    temp = {}

    for node in sorted(nodes):
        DOFs = []
        try:
            for rxn in bn[node]:
                if not rxn:
                    DOFs.append(df)
                    df += 1
                else:
                    DOFs.append(crxns)
                    crxns += 1
        except KeyError:
            df -= 1
            DOFs = [df := df + 1 for _ in range(ndf)]
            df += 1

        temp[node] = DOFs
    print(locals())
    return temp
    
def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

def get_initial_state()
