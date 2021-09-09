# Claudio Perez
# anabel
"""
# MeshGroup

Core mesh model building classes.
"""
# Standard library
import gc
import copy
import inspect
import functools
from inspect import signature
from functools import partial
from typing import Callable, List, Union

import jax
import meshio
import numpy as np
import scipy.sparse
from mpl_toolkits.mplot3d import Axes3D

from anon import diff
from anabel.template import get_unspecified_parameters, template
from anabel.abstract.element import MeshCell

try:
    import anon.atom as anp
except:
    anp = np

__all__ = ["MeshGroup"]

def _is_sequence(obj):
    return hasattr(type(obj), '__iter__')

class Boundary:
    def __init__(self, essential=[], natural=[]):
        self.essential = essential
        self.natural = natural
        pass

class Assembler:
    """Base class for assembler objects.

    An assembler is typically characterized by collections of
    nodes, elements and parameters. The purpose of the assembler
    is to provide a convenient interface for interacting with 
    and managing these entities.
    """
    def __init__(self, ndm:int, ndf:int,verbose=0):
        """
        Parameters
        -----------
        ndm: int
            number of model dimensions
        ndf: int
            number of degrees of freedom (dofs) at each node

        """
        self.ndf: int = ndf
        self.ndm: int = ndm
        self.DOF: list = None
        self._numberer  = None
        self.dtype='float64'
        self.verbose = verbose

        # model inventory lists
        self.elems: list = []
        self.nodes: list = []
        self.paths: dict = {}

        self.bound: list = []
        self.cells: list = []

        # model inventory dictionaries
        self.delems: dict = {}
        self.dnodes: dict = {}
        self.params = {}
        self.expressions = {}
        
        # Define DOF list indexing 
        if ndm == 1:
            self.prob_type = '1d'
            self.dof_names: dict = {'x': 0}  # Degrees of freedom at each node

        if ndf == 1 and ndm == 2:
            self.prob_type = ''
            self.dof_names: dict = { 'x': 0, 'y': 1} # Degrees of freedom

        if ndf == 2:
            self.prob_type = '2d-truss'
            self.dof_names: dict = { 'x': 0, 'y': 1} # Degrees of freedom
        elif ndm == 2 and ndf ==3:
            self.prob_type = '2d-frame'
            self.dof_names: dict = { 'x': 0, 'y': 1, 'rz':2}
        elif ndm == 3 and ndf ==3:
            self.prob_type = '3d-truss'
            self.dof_names: dict = { 'x': 0, 'y': 1, 'z':2}
        elif ndm == 3 and ndf ==6:
            self.prob_type = '3d-frame'
            self.dof_names: dict = { 'x': 0, 'y': 1, 'z':2, 'rx':3, 'ry':4, 'rz':5}
    
    @property
    def nn(self) -> int:
        """Number of nodes in model"""
        return len(self.nodes)

    @property
    def ne(self) -> int:
        """Number of elements in model"""
        return len(self.elems)

    @property
    def nf(self) -> int:
        "Number of free degrees of freedom."
        return  self.nt - self.nr


    def cycle(self, tag, 
        origin, 
        max_val, 
        steps:int, 
        quarter_cycles:int,
        dof:Union[int,str]=None, 
        parameter=None
    ):
        """Create a parameter path that is managed by the model.

        created 2021-05-16
        """
        if isinstance(origin,(float,int)):
            assert (dof is not None), "Scalar value must be assigned to a DOF."
        if tag not in self.paths:
            self.paths[tag] = {}

        delta = (max_val - origin)/steps
        self.paths[tag].update({dof: anp.array([
            anp.linspace(
                    origin if not i%2 else max_val, max_val - delta if not i%2 else origin + delta, steps -1
            ) * (1 if (not i%4) or (not (i-1)%4) else -1)
                for i in range(quarter_cycles)
            ]).flatten() 
        })

    def build_load_path(self, tag):
        dofs = self.dofs
        path = self.paths[tag]
        # start as regular numpy array to support index assignment
        load = np.zeros((self.nf,len(list(path.values())[0])))
        for dof in path:
            load[dof,:] = path[dof]
        return anp.stack([l[:,None] for l in load.T])

         


    
    def param(self,*param_names,shape=0,dtype=float,default=None):
        """Create a parameter that is managed by the model.

        created 2021-03-31
        """
        param_kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
        if shape != 0:
            self.params.update({
                param: anp.empty(shape=shape,dtype=dtype) for param in param_names
        })
        else:
            self.params.update({
                name: inspect.Parameter(name,param_kind,default=default) for name in param_names
            })
        if len(param_names) > 1:
            return tuple(inspect.Parameter(name,param_kind) for name in param_names)
        else:
            return inspect.Parameter(param_names[0],param_kind)

    def expr(self, expression:Callable, *parameters):
        n = len(self.expressions)
        name = f"_expr_{n}"
        param = self.param(name)#, default=expression)
        self.expressions.update({name: {"params": list(map(lambda x: x.name,parameters)), "expression": expression}})
        return param


class UniformAssembler(Assembler):
    """Abstract base class for model where every element is of the same arity.
    """

    
    @property
    def nt(self)->int:
        return self.ndf * len(self.nodes)

    @property
    def dofs(self):
        """Plain DOF numbering scheme.

        2021-05-07
        """
        if self._numberer is None:
            free_dofs = iter(range(self.nf))
            fixed_dofs = iter(range(self.nf, self.nt))
            return [
                [
                    next(fixed_dofs) if fixity else next(free_dofs)
                    for fixity in node 
                ] for node in self.fixities
            ]

    @property
    def element_dofs(self):
        return anp.concatenate([elem.dofs for tag,elem in enumerate(self.elems)],axis=-1).T

    def dof_element_map(self):
        el_DOF = self.element_dofs
        if self.verbose: print("Constructing element-DOF map.")
        Be_map = [[] for i in range(self.nt)]
        for el,dofs in enumerate(el_DOF):
            for i,dof in enumerate(dofs): 
                Be_map[dof].append((el,i))
        return Be_map


class MeshGroup(UniformAssembler):
    """Homogeneous 2D mesh group.

    """
    def __init__(self, *args, ndm=2, ndf=1,  mesh=None, **kwds):
        super().__init__(ndm=ndm, ndf=ndf)
        self.mesh = mesh
        self.nodes = [
            Node(self, str(i), self.ndf, point) 
            for i,point in enumerate(mesh.points)
        ]
        self.elems = [
            MeshCell(f"elem-{i}", len(cell), ndf, ndm, [self.nodes[j] for j in cell])
            for i,cell in enumerate(mesh.cells[0][1])
        ]
        self.fixities = mesh.point_data["fixities"] \
                if "fixities" in mesh.point_data else [[0] for i in range(self.nn)]

    @classmethod
    def read(cls, filename:str, *args, **kwds):
        """Create a class instance by reading in a mesh file.

        This function should work with any mesh format that is supported
        in the external [`meshio`](https://github.com/nschloe/meshio) Python library.

        Parameters
        ----------
        filetype: str
            In addition to those supported in `meshio`, the following formats
            are supported:
            `m228`:
                Simple text file; see docstrings in source code of `_read_m228`.
        """
        if "file_type" in kwds:
            mesh = meshio.read(filename,*args,**kwds)
        elif "m228" in args:
            mesh = _read_m228(filename,**kwds)
        return cls(mesh=mesh)

    def write(self,filename:str, **kwds):
        """Export mesh using `meshio`.
        """
        meshio.write(self.mesh,filename)

    @property
    def nr(self):
        """Return number of fixed degrees of freedom"""
        return np.count_nonzero(self.fixities)

    @property
    def dofs(self):
        """Plain DOF numbering scheme.

        Returns
        -------
        dofs: Sequence (`nn`, `ndf`)
            A sequence with shape `nn` by `ndf` where:
            [`ndf`]
            [`nn`]

        2021-05-07
        """
        if self._numberer is None:
            free_dofs = iter(range(self.nf))
            fixed_dofs = iter(range(self.nf, self.nt))
            return [
                [
                    next(fixed_dofs) if fixity else next(free_dofs)
                    for fixity in node
                ] for node in self.fixities
            ]

    def plot(self,values=None,func=None,scale=1.0,interact=False,savefig:str=None,**kwds):
        """
        Parameters
        ----------
        u: Union[ Callable, Sequence ]
            Values to plot over domain.

        savefig: str
            File path to save image to.


        Plot mesh using `pyvista` interface to VTK.

        Pure numpy is used for generality.

        Claudio Perez
        """
        from matplotlib import cm
        if values is None:
            point_values = np.asarray([[func(node.xyz)] for node in self.nodes])
        elif callable(values):
            point_values = np.array([values(node.xyz) for node in self.nodes])
            print(point_values)
        else:
            # TODO: appends zeros at the end under the assumption that solution
            # vanished on Dirichlet boundary. This should be changed.
            dof_values = anp.concatenate([values, anp.zeros((self.nr,1))],axis=0)
            point_values = anp.take(dof_values, self.dofs)

        point_values *= scale
        show_edges = True if self.ne < 10_000 else False
        self.mesh.point_data["u"] = point_values

        import pyvista as pv
        pv.start_xvfb(wait=0.05)
        pv.set_plot_theme("document")
        mesh = pv.utilities.from_meshio(self.mesh)
        mesh = mesh.warp_by_scalar("u", factor=0.5)
        mesh.set_active_scalars("u")
        if not pv.OFF_SCREEN:
            if interact:
                plotter_itk = pv.PlotterITK()
                plotter_itk.add_mesh(mesh)
                plotter_itk.show(True)
            else:
                plotter = pv.Plotter(notebook=True)
                plotter.add_mesh(mesh,
                   show_edges=show_edges,
                   cmap=cm.get_cmap("RdYlBu_r"),
                   lighting=False,
                   **kwds)
                if self.nn < 1000:
                    plotter.add_mesh(
                       pv.PolyData(mesh.points), color='red',
                       point_size=5, render_points_as_spheres=True)
                if savefig:
                    plotter.show(screenshot=savefig)
                else:
                    plotter.show()

    def norm(self,u,h,quad):
        du = diff.jacx(u)
        U = self.compose(quad.points,quad.weights)()
        inner = None
        return

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
        #el_DOF  = [ elem.dofs for tag,elem in enumerate(self.elems) ]
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
            tag: get_unspecified_parameters(elem) for tag,el in enumerate(self.elems)
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
            #print(el_jacs.shape)
            #print(vrow(el_jacs,DOF_el_jac[0]))
            #jac = vjac(el_jacs,DOF_el_jac)
            jac = anp.array([
                [
                    sum([el_jacs[tag][i,j]
                        for tag, i, j  in DOF_el_jac[dof_i][dof_j]
                    ]) for dof_j in range(nf)
                ] for dof_i in range(nf)
            ])
            return jac
        return locals()

    def compose(self,elem=None,verbose=False,solver=None):
        """
        Parameters
        ----------
        elem: Callable
            local function to be integrated over.

        solver: str
            Either of the following.
            "sparse": use `scipy.sparse.linalg.spsolve`
            "cg": Conjugate gradient using `jax.scipy.sparse.linalg.cg`
            `None`: default to `anabel.backend.linalg.solve`

        2021-05-07
        """
        if solver is None:
            f = self.compose_quad(elem=elem,verbose=verbose)
            jac = diff.jacx(f)
            solver = anp.linalg.solve
            stiff = jax.vmap(lambda loc,weight: jac(loc)*weight,(0,0))
            force = jax.vmap(lambda loc,weight: f(loc)*weight, (0,0))
            def U(points, weights):
                A = sum(stiff(points,weights))
                print(A)
                b = sum(force(points,weights))
                print(b)
                return solver(A, b)
            #def F(points, weights):
            #    return solver(
            #        sum(jac(xi)*w for xi,w in zip(points,weights)),
            #        sum(f(xi)*w for xi,w in zip(points,weights))
            #    )
        elif solver == "2":
            F = self.assemble_integral(elem=elem)
            jac = diff.jacx(lambda *args: F(*args)+b)
            solver = jax.scipy.sparse.linalg.cg
            z = anp.zeros((self.nf,1))
            def U(points, weights):
                b = F(z,None,None,None,points,weights)
                A = jac(z,None,None,None,points,weights)
                return solver(A, b)

        elif solver == "sparse":
            F = self.assemble_integral(elem=elem)
            jac = F.sparse_jac
            solver = scipy.sparse.linalg.spsolve
            z = anp.zeros((self.nf,1))
            def U(points, weights):
                b = F(z,None,None,None,points,weights)
                print("source vector assembled")
                A = jac([],None,None,None,points,weights)
                print("stiffness matrix assembled")
                return anp.atleast_2d(solver(A, -b)).T

        elif solver == "pos":
            F = self.assemble_integral(elem=elem)
            solver = jax.scipy.linalg.solve
            z = anp.zeros((self.nf,1))
            def U(points, weights):
                b = F(z,None,None,None,points,weights)
                A = lambda U: jax.jvp(lambda x: F(x,None,None,None,points,weights),(z,),(U,))[1]
                #b = F(z,None,None,None,points,weights)[1]
                #return solver(lambda U: F(U,None,None,None,points,weights)[1]-b, z, x0=z)[0]
                return solver(A, -b, tol=1e-12, atol=0.0, x0=z)[0]

        elif solver == "cg":
            F = self.assemble_integral(elem=elem)
            solver = jax.scipy.sparse.linalg.cg
            z = anp.zeros((self.nf,1))
            def U(points, weights):
                b = F(z,None,None,None,points,weights)
                A = lambda U: jax.jvp(lambda x: F(x,None,None,None,points,weights),(z,),(U,))[1]
                return solver(A, -b, tol=1e-12, atol=0.0, x0=z)[0]
            
        return U


    def compose_quad(self,f=None,jit=True,verbose=False,**kwds):
        if f is None:
            f = self.assemble_linear(jit=jit,**kwds,verbose=verbose,_expose_closure=True)
        f_jacx = diff.jacx(f)
        cnl = ',\n'
        u0 = anp.zeros((self.nf,1))
        model_map = f.closure["model_map"]
        param_map = f.closure["param_map"]
        params = self.params
        elem = f.closure["elem"]
        local_scope = locals()
        #------------------------------------------------------------------
        def _unpack(el):
            params = get_unspecified_parameters(el,recurse=True)
            ls = ",".join(
                f"'{p.name}': {p.default.name}"
                    for p in params.values()
                      if isinstance(p,inspect.Parameter) and p.default
                )
            if "params" in params and params["params"]:
                subparams = "'params': {" + ",".join(
                        f"'{k}': {v.default.name}" for k,v in params["params"].items() if v.default
                ) + "}"
                if ls:
                    return  ",".join((ls, subparams))
                else:
                    return subparams
            else:
                return ls
        exec(
            f"def collect_params({','.join(p for p in self.params )}):\n"
             "  return {'params': {\n"
            f"""    { f'0: {{ {_unpack(elem)} }}' }\n"""
             "  }}",
             local_scope
        )
        collect_params = local_scope["collect_params"]

        exec(
            f"def resp({','.join(p for p in self.params)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params)});\n"""
            f"   return f(f.origin[0], f.origin[1], f.origin[2], None, params['params'])[1]\n"

            f"def jacx({','.join(p for p in self.params)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params)})\n"""
            f"   return f_jacx(f.origin[0], f.origin[1], f.origin[2], None, params['params'])",
            local_scope
        )
        main = local_scope["resp"]
        main.jacx = local_scope["jacx"]
        main.collect_params = collect_params
        main.origin = tuple(anp.zeros(p.shape) if hasattr(p,"shape") else 0.0 for p in self.params)
        return main






##<****************************Model objects************************************
# Model objects/classes
# These should be created using the methods above
#******************************************************************************
class Node():
    """Model node object"""
    def __init__(self, model, tag: str, ndf, xyz, mass=None):
        if mass is None: mass=0.0

        self.xyz = np.array([xi for xi in xyz if xi is not None])

        self.tag = tag
        self.xyz0 = self.xyz # coordinates in base configuration (unstrained, not necessarily unstressed).
        self.xyzi = self.xyz # coordinates in reference configuration.  

        self.x0: float = xyz[0] # x-coordinate in base configuration (unstrained, not necessarily unstressed).  
        self.y0: float = xyz[1] # y-coordinate in base configuration (unstrained, not necessarily unstressed).  
        # z-coordinate in base configuration (unstrained, not necessarily unstressed).  
        self.z0: float = xyz[2] if len(xyz) > 2 else None

        
        self.x: float = xyz[0]
        self.y: float = xyz[1]
        self.z: float = xyz[2] if len(xyz) > 2 else None

        
        self.rxns = [0]*ndf
        self.model = model
        self.mass = mass
        self.elems = []

        self.p = {dof:0.0 for dof in model.dof_names}
        
    def __repr__(self):
        return 'nd-{}'.format(self.tag)

    def p_vector(self):
        return np.array(list(self.p.values()))

        
    @property
    def dofs(self):
        """Nodal DOF array"""
        # if self.model.DOF == None: self.model.numDOF()
        idx = self.model.nodes.index(self)
        return np.asarray(self.model.DOF[idx],dtype=int)


class Rxn():
    def __init__(self, node, dirn):
        self.node = node
        self.dirn = dirn

    def __repr__(self):
        return 'rxn-{}'.format(self.dirn)


class Hinge():
    def __init__(self, elem, node):
        self.elem = elem
        self.node = node



def parabolicArch(length, height):
    pass

def spacetruss(ns, Ro, Ri, H):
    """Macro for generating 3D space truss"""

    alpha = { i -96:chr(i) for i in range(ord("a"), ord("a") + 26)}
    model = Model(3,3)
    m1 = model.material('default', 1.0)
    s1 = model.xsection('default', 1.0, 1.0)
    
    # specify node coordinates for support points
    # angle for supports
    phi = np.arange(ns)/ns*2*np.pi

    # coordinates for support points
    X  = np.cos(phi)*Ro
    Y  = np.sin(phi)*Ro
    # generate support points with height Z of 0
    for i in range(ns):
        model.node(str(i+1), X[i], Y[i], 0.0)
    
    # angles for upper ring (offset by pi/ns degrees from supports)
    phi = phi+np.pi/ns

    # coordinates for upper ring
    X   = np.append(X, np.cos(phi)*Ri)
    Y   = np.append(Y, np.sin(phi)*Ri)

    # generate coordinates for upper ring with height H
    for i in np.arange(ns, 2*ns):
        model.node(str(i+1), X[i], Y[i], H)

    for i, j, k in zip(np.arange(ns), np.arange(0, ns), np.arange(ns, 2*ns)):
        model.truss3d(alpha[i+1], model.nodes[j], model.nodes[k], m1, s1)

    model.truss3d(alpha[ns+1], model.nodes[0], model.nodes[2*ns-1], m1, s1)

    for i, j, k in zip(np.arange(ns+1, 2*ns), np.arange(1, ns), np.arange(ns, 2*ns-1)):
        model.truss3d(alpha[i+1], model.nodes[j], model.nodes[k], m1, s1)

    for i, j, k in zip(np.arange(2*ns, 3*ns-1), np.arange(ns, 2*ns-1), np.arange(ns+1, 2*ns)):
        model.truss3d(alpha[i+1], model.nodes[j], model.nodes[k], m1, s1)

    model.truss3d(alpha[3*ns], model.nodes[ns], model.nodes[2*ns-1], m1, s1)


    # boundary conditions
    for node in model.nodes[0:ns]:
        model.fix(node, ['x', 'y', 'z'])

    model.numDOF()

    return model



def _read_m228(filename:str,cell="triangle",ndf=1)->meshio.Mesh:
    """
    Read a mesh file that adheres to the following spec:

    ```
    N = number of nodes, NE = number of elements)

    N  NE      <-- first line
    x y flag   <-- flag=0: interior, 1 bdry
    x y flag
    ...
    x y flag
    a b c d e f   <-- quadratic elements have six nodes
    a b c d e f
    ...
    a b c d e f


    in the triangle, the numbers a b c d e f correspond to nodes

    c
    f e
    a d b

    (This is the FEAP convention).

    meshes arranged so that if an edge is curved, it is edge b e c.
    (so the interior vertex of the triangle is first in the list)

    Node and element indices start at 1.
    ```
    """
    with open(filename,"r") as f:
        num_nodes, num_cells = map(int, f.readline().split(" "))
    nodes = np.loadtxt(filename, skiprows=1, max_rows=num_nodes)
    connectivity = np.loadtxt(filename, skiprows=1+num_nodes, max_rows=num_cells, dtype=int) - 1
    cells = [
        (cell, connectivity)
    ]
    points = nodes[:,:2]
    fixities = np.concatenate([nodes[:,2][:,None]]*ndf,axis=1)
    return meshio.Mesh(points, cells, point_data={"fixities": fixities})
