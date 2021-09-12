# Claudio Perez
# anabel
"""
"""
# Standard library
import gc
import copy
import inspect
import functools
from inspect import signature
from typing import Callable, List

import jax
import numpy as np
import scipy.sparse

from anon import diff
from anabel.template import get_unspecified_parameters, template
from anabel.abstract.element import MeshCell

try:
    import anabel.backend as anp
except:
    anp = np


def _is_sequence(obj):
    return hasattr(type(obj), '__iter__')


@template(dim="shape",main="assm")
def assemble_integral(self, elem=None, verbose=False, **kwds)->Callable:
    """
    Parameters
    ----------
    elem: `f(u,xyz) -> R^[ndf*nen]`

    Returns
    -------
    f: `f(U, (xi, dV)) -> R^[nf]`
        quad(elem(U[el], X[el])*dV

    """
    ndf = self.ndf
    nr = self.nr
    nn = self.nn
    ne = len(self.elems)
    nf = self.nf
    nt = self.nt
    nr = self.nr

    nodes = set(self.dnodes.keys())
    shape = ((nf,1),(nf,1))
    params = self.params
    # Assign model degrees of freedom
    self.DOF = DOF = self.dofs
    el_DOF  = self.element_dofs
    Be_map = self.dof_element_map()
    
    _unpack_coords = lambda xyz: f"[{','.join(str(x) for x in xyz)}]"
    local_scope = locals()
    # TODO: coords are not being used
    exec(
    f"def collect_coords(*coords):\n"
     "  return anp.array([\n"
    f"""    {f",".join(f"[{','.join(_unpack_coords(node.xyz) for node in el.nodes) }]" for el in self.elems )}\n"""
     "  ])",
     dict(anp=anp),local_scope,
    )
    collect_coords = local_scope["collect_coords"]

    if elem is None:
        elem = self.elems[0].compose()
    
    param_map = {
        tag: get_unspecified_parameters(elem) for tag,el in enumerate(self.elems)
    }

    
    # Create DOF - element mapping
    if verbose: print("Creating Be_map")
    state = {
        ... : [elem.origin[2] for tag in range(ne)]
    }
    param_arg = {0: {}}

    _p0 = anp.zeros((nf,1))

    if verbose: print("Collecting coordinates.")
    xyz = eval(f"""{{ { ','.join(f'"{tag}": [{",".join(str(x) for x in node.xyz)}]' for tag, node in self.dnodes.items() )} }}""")

    def map_dofs(U,dofs):
        print(U,dofs)
        return anp.take(U.flatten(), anp.asarray(dofs, dtype='int32')-1)[:,None]

    vresp = jax.vmap(elem,(0,None,None,0,None,None))
    vdisp = jax.vmap(map_dofs, (None,0))
    Z = anp.zeros((nr,1))
    def assm(u=_p0, p=_p0, state=None, xyz=None, points=None, weights=None, params=param_arg):
        U = anp.concatenate([u,Z],axis=0)
        coords = collect_coords(None)
        u_el = anp.take(U,el_DOF,axis=0)
        responses = vresp(u_el,None,None,coords,points,weights)

        F = anp.array([
            sum(responses[el][i] for el,i in dof)
            for dof in Be_map[:nf]
        ])
        return F

    # create a vectorized function. TODO: This currently doesnt handle
    # arbitrary parameterizations which were handled by keyword args.
    elem_jac = jax.vmap(diff.jacx(elem),(None,None,None,0,None,None))
    vrow = jax.vmap(lambda eJ, eij: sum(eJ[tuple(zip(*(eij)))]), (None,1))
    vjac = jax.vmap(vrow, (None,0))
    #z = anp.zeros((nt))
    def sparse_jac(u,p,state,xyz=None,points=None,weights=None,params=param_arg):
        coords = collect_coords(None)
        el_jacs = elem_jac(
            [],[],[],coords, points, weights
        )
        print("\tState determination complete")
        K = scipy.sparse.lil_matrix((nt,nt))
        for tag, el_dofs in enumerate(el_DOF):
            for j, dof_j in enumerate(el_dofs):
                if j < nf:
                    for i, dof_i in enumerate(el_dofs):
                        if i < nf: K[dof_i,dof_j] += el_jacs[tag,i,j]
        print("\tMatrix assembly complete")

        del el_jacs
        gc.collect()
        return K[:nf,:nf].tocsr()
    assm.sparse_jac = sparse_jac
    return locals()


    @template(dim="shape",main="assm")
    def assemble_linear(self, elem=None,verbose=False,**kwds)->Callable:
        """
        `elem(None,xyz) -> R^[ndf*nen]`

        Returns
        -------
        f: `f(U, (xi, dV)) -> R^[nf]`
            quad(elem(None, X[el])*dV
        
        """
        ndf = self.ndf
        nr = self.nr
        nn = self.nn
        ne = len(self.elems)
        nf = self.nf
        nr = self.nr
        nodes = set(self.dnodes.keys())
        shape = ((nf,1),(nf,1))
        params = self.params
        if elem is None:
            elem = self.elems[0].compose()
        
        _unpack_coords = lambda xyz: f"[{','.join(str(x) for x in xyz)}]"
        local_scope = locals()
        # TODO: coords are not being used
        exec(
        f"def collect_coords(*coords):\n"
         "  return anp.array([\n"
        f"""    {f",".join(f"[{','.join(_unpack_coords(node.xyz) for node in el.nodes) }]" for el in self.elems )}\n"""
         "  ])",
         dict(anp=anp),local_scope,
        )
        collect_coords = local_scope["collect_coords"]
        
        model_map = {
            el.tag: elem for el in self.elems
        }
        param_map = {
            tag: anon.dual.get_unspecified_parameters(elem) for tag,el in enumerate(self.elems)
        }

        self.DOF = DOF = self.dofs
        el_DOF  = self.element_dofs
        #el_DOF  = anp.concatenate([ elem.dofs for tag,elem in enumerate(self.elems) ],axis=-1).T
        Be_map = self.dof_element_map()
        if self.verbose: print("Element-DOF map complete.")

        state = {
            ... : [elem.origin[2] for tag in range(ne)]
        }
        param_arg = {0: {}}

        _p0 = anp.zeros((nf,1))
        xyz = eval(f"""{{ { ','.join(f'"{tag}": [{",".join(str(x) for x in node.xyz)}]' for tag, node in self.dnodes.items() )} }}""")
        
        vresp = jax.vmap(elem,(0,None,None,0,None))
        vdisp = jax.vmap( 
                lambda U,dofs:  anp.take(U.flatten(), anp.asarray(dofs, dtype='int32')-1)[:,None], (None,0)
        )
        Z = anp.zeros((nr,1))
        def assm(u=_p0,p=_p0,state=None, xyz=None, params=param_arg):
            U = anp.concatenate([u,Z],axis=0) 
            coords = collect_coords(None)
            u_el = vdisp(U,el_DOF)
            responses = vresp(u_el,None,None,coords,params[0]["xi"])
            print(responses)
            F = anp.array([
                sum(responses[el][i] for el,i in dof)
                for dof in Be_map[:nf]
            ])
            return None,  F, {}

        #----------------------------------------------------------------
        if verbose: print("Constructing jacobian map.")
        DOF_el_jac = [ [ [] for j in range(nr+nf)] for i in range(nr+nf)]
        for tag,el_dofs in enumerate(el_DOF): 
            for j, dof_j in enumerate(el_dofs):
                for i, dof_i in enumerate(el_dofs):
                    DOF_el_jac[dof_i][dof_j].append([tag,i,j])
        if verbose: print("Jacobian map complete.")
        #----------------------------------------------------------------
        # create a vectorized function. TODO: This currently doesnt handle
        # arbitrary parameterizations which were handled by keyword args.
        elem_jac = jax.vmap(diff.jacx(elem),(None,None,None,0,None))
        vrow = jax.vmap(lambda eJ,eij: sum(eJ[tuple(zip(*(eij)))]), (None,1)) 
        vjac = jax.vmap(vrow,(None,0))

        def jac_x(u,p,state,xyz=xyz,params=param_arg):
            coords = collect_coords(None)
            U = anp.concatenate([u,anp.zeros((nr,1))],axis=0)
            el_jacs = elem_jac(
                    [],[],[],
                    coords,
                    params[0]["xi"]
            )
            print("State determination complete")
            K = anp.zeros((nf,nf))
            for tag, el_dofs in enumerate(el_DOF): 
                for j, dof_j in enumerate(el_dofs):
                    for i, dof_i in enumerate(el_dofs):
                        K = jax.ops.index_add(K, jax.ops.index[dof_i,dof_j], el_jacs[tag,i,j])
            return K

        def jacx(u,p,state,xyz=xyz,params=param_arg):
            coords = collect_coords(None)
            U = anp.concatenate([u,anp.zeros((nr,1))],axis=0)
            el_jacs = elem_jac(
                    [],[],[],
                    coords,
                    params[0]["xi"]
            )
            jac = anp.array([
                [
                    sum([el_jacs[tag][i,j]
                        for tag, i, j  in DOF_el_jac[dof_i][dof_j]
                    ]) for dof_j in range(nf)
                ] for dof_i in range(nf)
            ])
            return jac
        return locals()


