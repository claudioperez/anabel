# Claudio Perez
# anabel
"""
# Assemblers(`2`)

Core model building classes.
"""
import gc
import copy
import inspect
import functools
from inspect import signature
from functools import partial
from typing import Callable, List

import jax
import meshio
import numpy as np
import scipy.sparse
from mpl_toolkits.mplot3d import Axes3D

from anon import diff
from anabel.template import get_unspecified_parameters, template
from anabel.elements import *
try:
    import anon.atom as anp
except:
    anp = np

__all__ = ["Assembler", "MeshGroup", "Model", "rModel", "SkeletalModel"]

def _is_sequence(obj):
    return hasattr(type(obj), '__iter__')

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

        self.bound: list = []
        self.cells: list = []

        # model inventory dictionaries
        self.delems: dict = {}
        self.dnodes: dict = {}
        self.params = {}
        
        # Define DOF list indexing 
        if ndm == 1:
            self.prob_type = '1d'
            self.ddof: dict = {'x': 0}  # Degrees of freedom at each node

        if ndf == 1 and ndm == 2:
            self.prob_type = ''
            self.ddof: dict = { 'x': 0, 'y': 1} # Degrees of freedom

        if ndf == 2:
            self.prob_type = '2d-truss'
            self.ddof: dict = { 'x': 0, 'y': 1} # Degrees of freedom
        elif ndm == 2 and ndf ==3:
            self.prob_type = '2d-frame'
            self.ddof: dict = { 'x': 0, 'y': 1, 'rz':2}
        elif ndm == 3 and ndf ==3:
            self.prob_type = '3d-truss'
            self.ddof: dict = { 'x': 0, 'y': 1, 'z':2}
        elif ndm == 3 and ndf ==6:
            self.prob_type = '3d-frame'
            self.ddof: dict = { 'x': 0, 'y': 1, 'z':2, 'rx':3, 'ry':4, 'rz':5}
        
    @property
    def nn(self) -> int:
        """Number of nodes in model"""
        return len(self.nodes)

    @property
    def ne(self) -> int:
        """Number of elements in model"""
        return len(self.nodes)

    @property
    def nf(self) -> int:
        "Number of free degrees of freedom."
        return  self.nt - self.nr
    
    @property
    def nt(self)->int:
        return self.ndf * len(self.nodes)

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



class MeshGroup(Assembler):
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
        in the external `meshio` Python library.

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

    def plot(self,values=None,func=None,interact=False,savefig:str=None,**kwds):
        """
        Plot mesh using `pyvista` interface to VTK.
        """
        from matplotlib import cm
        if values is None:
            point_values = [[func(node.xyz)] for node in self.nodes]
        elif callable(values):
            point_values = [[values(node.xyz)] for node in self.nodes]
            print(point_values)
        else:
            # TODO: appends zeros at the end under the assumption that solution
            # vanished on Dirichlet boundary. This should be changed.
            dof_values = anp.concatenate([values, anp.zeros((self.nr,1))],axis=0)
            point_values = anp.take(dof_values, self.dofs)

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
                   show_edges=True,
                   cmap=cm.RdYlBu,
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
        nr = self.nr
        nodes = set(self.dnodes.keys())
        shape = ((nf,1),(nf,1))
        params = self.params
        # Assign model degrees of freedom
        self.DOF = DOF = self.dofs
        
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
            tag: anon.dual.get_unspecified_parameters(elem) for tag,el in enumerate(self.elems)
        }
        #el_DOF  = [ elem.dofs for tag,elem in enumerate(self.elems) ]
        el_DOF  = self.element_dofs
        Be_map = self.dof_element_map()

        
        # Create DOF - element mapping
        if verbose: print("Creating Be_map")
        #Be_map = tuple(
        #    [(el,i) for el in range(ne) for i, el_dof in enumerate(el_DOF[el]) if dof==el_dof]
        #    for dof in range(nf)
        #)
        state = {
            ... : [elem.origin[2] for tag in range(ne)]
        }
        param_arg = {0: {}}

        _p0 = anp.zeros((nf,1))
        
        if verbose: print("Collecting coordinates.")
        xyz = eval(f"""{{ { ','.join(f'"{tag}": [{",".join(str(x) for x in node.xyz)}]' for tag, node in self.dnodes.items() )} }}""")
         
        #def assm(u, p=None, state=None, points=None, weights=None, params=param_arg):
        #    U = anp.concatenate([u,anp.zeros((nr,1))],axis=0)
        #    coords = collect_coords(None)
        #    return  sum(
        #        elem(
        #            anp.take(U, anp.asarray(dofs, dtype='int32')-1),
        #            None,None,
        #            xyz = coords[tag],
        #            xi = loc,
        #            **params[0]
        #        )[1] * weight
        #        for tag,dofs in enumerate(el_DOF) for loc,weight in zip(points,weights)
        #    )
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
            #return None,  F, {}
            return F

        # create a vectorized function. TODO: This currently doesnt handle
        # arbitrary parameterizations which were handled by keyword args.
        elem_jac = jax.vmap(diff.jacx(elem),(None,None,None,0,None,None))
        vrow = jax.vmap(lambda eJ, eij: sum(eJ[tuple(zip(*(eij)))]), (None,1))
        vjac = jax.vmap(vrow, (None,0))
        z = anp.zeros((self.nt))
        def sparse_jac(u,p,state,xyz=None,points=None,weights=None,params=param_arg):
            coords = collect_coords(None)
            el_jacs = elem_jac(
                [],[],[],coords, points, weights
            )
            print("State determination complete")
            K = scipy.sparse.lil_matrix((nf,nf))
            for tag, el_dofs in enumerate(el_DOF):
                for j, dof_j in enumerate(el_dofs):
                    for i, dof_i in enumerate(el_dofs):
                        K[dof_i,dof_j] += el_jacs[tag,i,j]

            del el_jacs
            gc.collect()
            return K.to_csr()
        assm.sparse_jac = sparse_jac
        return locals()


    @template(dim="shape",main="assm")
    def assemble_linear(self, elem=None,verbose=False,**kwds)->Callable:
        """
        elem(None,xyz) -> R^[ndf*nen]


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
        TODO: try using jax.scipy.sparse.linalg.cg

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
                return solver(A, -b)

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


class Model(Assembler):
    def __init__(self, ndm:int, ndf:int):
        """Basic structural model class

        Parameters
        -----------
        ndm: int
            number of model dimensions
        ndf: int
            number of degrees of freedom (dofs) at each node

        """
        super().__init__(ndm=ndm, ndf=ndf)
        self.ndf: int = ndf
        self.ndm: int = ndm
        #self.DOF: list = None
        self.dtype='float32'


        self.clean()
    
    def clean(self,keep=None):
        if keep is None:
            keep = []

        self.DOF = None

        # model inventory lists
        self.elems: list = []
        self.nodes: list = []
        if "params" not in keep:
            self.params: dict = {}
            """2021-03-31"""

        self.rxns:  list = []
        self.hinges: list = []
        self.iforces: list = []
        self.states: list = []
        self.redundants: list = []

        # model inventory dictionaries
        self.delems: dict = {}
        self.dnodes: dict = {}
        self.dxsecs: dict = {}
        self.dhinges: dict = {}
        self.dstates: dict = {}
        self.dxsecs: dict = {}
        self.materials: dict = {}
        self.xsecs: dict = {}
        self.dredundants: dict = {}
        # Initialize default material/section properties
        self.material('default', 1.0)
        self.xsection('default', 1.0, 1.0)



    def compose(self,resp="d",jit=True,verbose=False,**kwds):
        return self.compose_param(jit_force=jit,verbose=verbose,**kwds)

    def compose_param(self,jit_force=True,verbose=False,**kwds):
        f = self.compose_displ(jit_force=jit_force,**kwds,verbose=verbose,_expose_closure=True)
        cnl = ',\n'
        u0 = anp.zeros((self.nf,1))
        model_map = f.closure["model_map"]
        param_map = f.closure["param_map"]
        params = self.params
        parameterized_loads = {
            dof: p.name if isinstance(p,inspect.Parameter) else p for node in self.nodes for dof,p in zip(node.dofs,node.p.values())
        }
        #params.update({
        #    p.name: p for el in param_map.values() for p in el.values()
        #})
        local_scope = locals()
        #-------------------
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
            f"""    {cnl.join(f'"{tag}": {{ {_unpack(el)} }}' for tag,el in model_map.items()) }\n"""
             "  }}",
             local_scope
        )
        collect_params = local_scope["collect_params"]

        #-------------------
        #local_scope = locals()
        exec(
        f"def collect_loads({','.join(p for p in parameterized_loads.values() if isinstance(p,str) )}):\n"
         "  return anp.array([\n"
        f"""    {','.join("["+p.name+"]" if isinstance(p,inspect.Parameter) else f'[{p}]' for node in self.nodes for dof,fixity,p in zip(node.dofs,node.rxns,node.p.values()) if not fixity ) }\n"""
         "  ])",
         dict(anp=anp),local_scope,
        )
        collect_loads = local_scope["collect_loads"]

        #-------------------
        exec(
            f"def displ({','.join(p for p in self.params)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params)})\n"""
            f"   return f(collect_loads({','.join(p for p in parameterized_loads.values() if isinstance(p,str) )}), f.origin[1], f.origin[2], params)[1]",
            local_scope
        )
        main = local_scope["displ"]
        origin = tuple(anp.zeros(p.shape) if hasattr(p,"shape") else 0.0 for p in self.params)

        # Evaluate once with zeros to JIT-compile
        return main
        #return collect_params
    
    def compose_force(self,jit=True,**kwds):
        f = self.assemble_force(jit=jit,**kwds,_expose_closure=True)
        f_jacx = diff.jacx(f)
        cnl = ',\n'
        u0 = anp.zeros((self.nf,1))
        model_map = f.closure["model_map"]
        param_map = f.closure["param_map"]
        params = self.params
        local_scope = locals()
        #-------------------
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
            f"""    {cnl.join(f'"{tag}": {{ {_unpack(el)} }}' for tag,el in model_map.items()) }\n"""
             "  }}",
             local_scope
        )
        collect_params = local_scope["collect_params"]

        #-------------------
        #exec(
        #    f"def collect_loads({','.join(p for p in parameterized_loads.values() if isinstance(p,str) )}):\n"
        #     "  return anp.array([\n"
        #    f"""    {','.join("["+p.name+"]" if isinstance(p,inspect.Parameter) else f'[{p}]' for node in self.nodes for dof,fixity,p in zip(node.dofs,node.rxns,node.p.values()) if not fixity ) }\n"""
        #     "  ])",
        # dict(anp=anp),local_scope,
        #)
        #collect_loads = local_scope["collect_loads"]

        #-------------------
        exec(
            f"def force({','.join(p for p in self.params)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params)})\n"""
            f"   return f(f.origin[0], f.origin[1], f.origin[2], None, params['params'])[1]\n"

            f"def jacx({','.join(p for p in self.params)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params)})\n"""
            f"   return f_jacx(f.origin[0], f.origin[1], f.origin[2], None, params['params'])",
            local_scope
        )
        main = local_scope["force"]
        main.jacx = local_scope["jacx"]
        main.collect_params = collect_params
        main.origin = tuple(anp.zeros(p.shape) if hasattr(p,"shape") else 0.0 for p in self.params)
        return main
    

    @template(dim="shape")
    def compose_displ(self, solver=None, solver_opts={}, elem=None, jit_force=True, **kwds):
        """
        dynamically creates functions `collect_loads` and `collect_coord`.
        """
        if solver is None:
            import elle.numeric
            solver = elle.numeric.inverse.inv_no1
        if not solver_opts:
            solver_opts = {
                    "tol": 1e-10,
                    "maxiter": 10
            }
        f = self.assemble_force(_jit=jit_force,**kwds,_expose_closure=True)
        model_map = f.closure["model_map"]
        state = {...:f.origin[2]}
        #state = f.origin[2]
        nf = self.nf
        shape = ((nf,1),(nf,1))
        u0 = anp.zeros((self.nf,1))

        #local_scope = locals()

        jacx = diff.jacx(f)
        #jacx = f.jacx
        model_map = f.closure["model_map"]
        param_map = f.closure["param_map"]
        main = solver(f,jacx=jacx,**solver_opts)
        # def main(p, u, state=state, params={}):
        #     return g()

        return locals()

    @template(dim="shape",main="force")
    def assemble_force(self, elem=None,**kwds)->Callable:
        """A simple force composer for skeletal structures."""
        ndf = self.ndf
        nr = self.nr
        nn = self.nn
        nf = self.nf
        nr = self.nr
        nodes = set(self.dnodes.keys())
        shape = ((nf,1),(nf,1))
        #shape = nf
        
        _unpack_coords = lambda xyz: f"[{','.join(str(x) for x in xyz)}]"
        local_scope = locals()
        exec(
        f"def collect_coords(*coords):\n"
         "  return {\n"
        f"""    {f",".join(f"'{el.tag}': anp.array([{','.join(_unpack_coords(node.xyz) for node in el.nodes) }])" for el in self.delems.values() ) }\n"""
         "  }",
         dict(anp=anp),local_scope,
        )
        collect_coords = local_scope["collect_coords"]

        if self.DOF is None:
            self.numDOF()

        if elem is not None:
            """When an `elem` argument is passed, use it for all
            elements in the model"""
            model_map = {
                el.tag: elem for el in self.delems.values()
            }
        else:
            model_map = {
                tag: el.compose(L=el.L) for tag,el in self.delems.items()
            }
        param_map = {
            tag: anon.dual.get_unspecified_parameters(el) for tag,el in model_map.items()
        }
        param_arg = {
            tag: {} for tag,el in model_map.items()
        }

        params = self.params

        DOF = {node_tag: dofs for node_tag, dofs in zip(self.dnodes.keys(),self.DOF)}

        el_DOF  = { elem.tag: elem.dofs for elem in self.delems.values() }

        Be = anp.concatenate([
             anp.array([[1.0 if j==i else 0.0 for j in range(1,nf+1)] for i in el.dofs ]).T
             for el in self.delems.values()], axis=1)

        state = {
            ... : {
                tag: m.origin[2] for tag, m, in model_map.items()
            }
        }
        _p0 = anp.zeros((nf,1))
        xyz = eval(f"""{{ { ','.join(f'"{tag}": [{",".join(str(x) for x in node.xyz)}]' for tag, node in self.dnodes.items() )} }}""")
        def force(u,p=_p0,state=state, xyz=None, params=param_arg):
            U = anp.concatenate([u,anp.zeros((nr,1))],axis=0)
            coords = collect_coords(xyz)
            responses = [
                el(
                    anp.take(U, anp.array(el_DOF[tag], dtype='int32')-1)[:,None],
                    None,
                    xyz = coords[tag],
                    state = state[...][tag],
                    **params[tag]
                )
                for tag,el in model_map.items()
            ]
            F = anp.concatenate([r[1] for r in responses],axis=0)
            state = {
                ...: {
                    tag: r[2] for tag,r in zip(model_map.keys(),responses)
                }
            }
            return u, (Be @ F), state

        eljac_map = {tag: diff.jacx(el) for tag,el in model_map.items()}
        DOF_el_jac = {
            dof_i : {
                dof_j: [
                    (tag, i, j)
                    for tag,el in self.delems.items()
                        for j, _dof_j in enumerate(el.dofs) if _dof_j == dof_j
                            for i, _dof_i in enumerate(el.dofs) if _dof_i == dof_i
               ]
               for dof_j in range(1,nf+1)
            }
            for dof_i in range(1,nf+1)
        }

        def jacx(u,p,state,xyz=None,params=param_arg):
            coords = collect_coords(xyz)
            U = anp.concatenate([u,anp.zeros((nr,1))],axis=0)
            el_jacs = {
                tag: jac(
                    anp.take(U, anp.array(el_DOF[tag], dtype='int32')-1)[:,None],
                    None,
                    xyz = coords[tag],
                    state = state[...][tag],
                    **params[tag]
                ).squeeze()
              for tag,jac in eljac_map.items()
            }
            jac = anp.array([
                [
                    sum(el_jacs[tag][i,j]
                        for tag, i, j  in DOF_el_jac[dof_i][dof_j]
                    ) for dof_j in range(1,nf+1)
                ] for dof_i in range(1,nf+1)
            ])
            return jac
        return locals()


    @property
    def rel(self):
        return [rel for elem in self.elems for rel in elem.rel.values()]


    @property
    def nr(self) -> int:
        """number of constrained dofs in model"""
        return len(self.rxns)

    @property
    def ne(self) -> int:
        """number of elements in model"""
        return len(self.elems)

    @property
    def nQ(self):
        f = 0
        for elem in self.elems:
            f += len(elem.basic_forces)
        return f

    @property
    def nq(self):
        """Number of basic element forces"""
        f = []
        for elem in self.elems:
            f.append(sum(1 for x in elem.q))
        return f

    @property
    def nv(self):
        """Number of basic deformation variables"""
        lst = []
        for elem in self.elems:
            lst.append(sum([1 for x in elem.v]))
        return lst

    @property
    def nf(self) -> int:
        """Number of free model degrees of freedom"""
        x = self.nt - self.nr
        return x

    @property
    def nt(self) -> int:
        """Total number of model degrees of freedom."""
        return self.ndf*self.nn

    @property
    def rdofs(self):
        """Sequence of restrained dofs in model"""
        DOF = self.DOF
        return []

    @property
    def NOS(self) -> int:
        """Degree of static indeterminacy"""
        nf = self.nf
        nq = sum(self.nq)
        return nq - nf

    @property
    def basic_forces(self)->np.ndarray:
        return np.array([q for elem in self.elems for q in elem.basic_forces ])

    @property
    def rdnt_forces(self)->np.ndarray:
        cforces = self.cforces
        return np.array([q  for q in cforces if q.redundant ])

    @property
    def cforces(self)->np.ndarray:
        return np.array([q for elem in self.elems for q in elem.basic_forces if not q.rel])

    @property
    def eforces(self):
        """Array of elastic element forces"""
        return np.array([q for elem in self.elems for q in elem.basic_forces if (q.plastic_event is None)])

    @property
    def idx_c(self):
        cforces = self.cforces
        forces = self.basic_forces
        idx_c = np.where(np.isin(forces,cforces))[0]
        return idx_c

    @property
    def idx_e(self):
        """Indices of elastic basic (not plastic) forces"""
        cforces = self.cforces
        eforces = self.eforces
        idx_e = np.where(np.isin(cforces,eforces))[0]
        return idx_e

    @property
    def idx_f(self):
        return np.arange(0,self.nf)

    @property
    def idx_i(self):
        rdts = self.rdnt_forces
        #rdts = self.redundants
        forces = self.basic_forces
        idx_i = np.where(np.logical_not(np.isin(forces, rdts)))[0]
        return idx_i

    @property
    def idx_x(self):
        rdts = self.rdnt_forces
        forces = self.basic_forces
        idx_x = np.where(np.isin(forces, rdts))[0]
        return idx_x

    def node(self, tag: str, x: float, y=None, z=None, mass: float=None):
        """Add a new emme.Node object to the model

        Parameters
        ----------
        x, y, z: float
            Node coordinates.
        """
        newNode = Node(self, tag, self.ndf, [x, y, z], mass)
        self.nodes.append(newNode)
        self.dnodes.update({newNode.tag : newNode})
        return newNode

    def displ(self,val):
        pass

    def load(self,obj,*args,pattern=None,**kwds):
        """
        Apply a load to a model object

        Claudio Perez 2021-04-01
        """
        if isinstance(obj,str):
            obj = self.dnodes[obj]
        if isinstance(obj,Node):
            return self.load_node(obj,*args,**kwds)

    def load_node(self,node,load,**kwds):
        """
        Claudio Perez 2021-04-01
        """
        if _is_sequence(load):
            pass
        else:
            assert "dof" in kwds
            node.p[kwds["dof"]] = load
        pass

    def state(self, method="Linear"):
        if self.DOF is None:
            self.numDOF()
        newState = State(self, method)

        ElemTypes = {type(elem) for elem in self.elems}
        StateVars = {key for elem in ElemTypes for key in elem.stateVars.keys() }

        stateDict = {var : {elem.tag : copy.deepcopy(elem.stateVars[var]) for elem in self.elems if var in elem.stateVars.keys()} for var in StateVars}
        self.states.append(stateDict)
        return stateDict

    def numDOF(self):
        crxns = self.ndf*len(self.nodes) - len(self.rxns)+1
        df = 1
        temp = []
        for node in self.nodes:
            DOFs = []
            for rxn in node.rxns:
                if not(rxn):
                    DOFs.append(df)
                    df += 1
                else:
                    DOFs.append(crxns)
                    crxns += 1
            temp.append(DOFs)
        self.DOF = temp
        return self.DOF

    def update(self,U_vector):
        for node in self.nodes:
            delta = [0.,0.]
            for i,dof in enumerate(node.dofs[0:2]):
                if not node.rxns[i]:
                    try: delta[i] = U_vector[U_vector.row_data.index(str(dof))]
                    except: pass

            node.xi= node.x
            node.x = delta[0] + node.xi
            node.yi= node.y
            node.y = delta[1] + node.yi
        pass

    def fix(self, node, dirn=["x","y","rz"]): # for dirn enter string (e.g. "x", 'y', 'rz')
        """Define a fixed boundary condition at specified degrees of freedom of the supplied node

        Parameters
        ----------
        node: emme.Node

        dirn: Sequence[String]

        """
        if isinstance(node,str):
            node = self.dnodes[node]
        if isinstance(dirn,list):
            rxns = []
            for df in dirn:
                newRxn = Rxn(node, df)
                self.rxns.append(newRxn)
                rxns.append(newRxn)
                node.rxns[self.ddof[df]] = 1
            return rxns
        else:
            newRxn = Rxn(node, dirn)
            self.rxns.append(newRxn)
            node.rxns[self.ddof[dirn]] = 1
            return newRxn

    def boun(self, node, ones):
        if isinstance(node,str):
            node = self.dnodes[node]
        for i, dof in enumerate(self.ddof):
            if ones[i]:
                self.fix(node, dof)

    def pin(self, *nodes):
        """
        Create a pinned reaction by fixing all translational degrees
        of freedom at the specified nodes.

        Parameters
        ----------
        node: emme.Node

        """
        for node in nodes:
            if isinstance(node, str):
                node = self.dnodes[node]
            self.fix(node, ['x', 'y'])
            if self.ndm == 3:
                self.fix(node, 'z')

    def roller(self, node):
        """
        Create a roller reaction at specified node

        """
        return self.fix(node, 'y')


 # Other
    def material(self, tag: str, E: float):
        newMat = Material(tag, E)
        self.materials[tag]=newMat
        return newMat

    def xsection(self, tag: str, A: float, I: float):
        newXSect = XSect(tag, A, I)
        self.xsecs[tag] = newXSect
        return newXSect

 # Elements
    def elem(self,elem,nodes,tag):
        if isinstance(nodes[0],str):
            nodes = [self.dnodes[node_tag] for node_tag in nodes]
        element = Element(elem.shape[0][0],self.ndm,nodes=nodes,elem=elem)
        element.tag = tag
        self.elems.append(element)
        self.delems.update({tag:element})
        return element


    def add_element(self, element):
        """Add a general element to model

        Parameters
        ----------
        element : emme.elements.Element

        """

        self.elems.append(element)
        self.delems.update({element.tag:element})

        for node in element.nodes:
            node.elems.append(element)

        return element

    def add_elements(self, elements):
        """Add a general element to model

        Parameters
        ----------
        element : emme.elements.Element

        """
        for element in elements:
            self.elems.append(element)
            self.delems.update({element.tag:element})
            for node in element.nodes:
                node.elems.append(element)

        return elements


    def beam(self, tag: str, iNode, jNode, mat=None, sec=None, Qpl=None,**kwds):
        """Create and add a beam object to model

        Parameters
        ----------
        tag : str
            string used for identifying object
        iNode : emme.Node or str
            node object at element i-end
        jNode : emme.Node or str
            node object at element j-end
        mat : emme.Material

        sec : emme.Section


        """

        if mat is None:
            E = kwds.pop("E") if "E" in kwds else self.materials["default"].E
        else:
            E = mat.E

        if sec is None:
            A = kwds.pop("A") if "A" in kwds else self.xsecs["default"].A
            I = kwds.pop("I") if "I" in kwds else self.xsecs["default"].I
        else:
            A = sec.A
            I = sec.I

        if isinstance(iNode,str):
            iNode = self.dnodes[iNode]
        if isinstance(jNode,str):
            jNode = self.dnodes[jNode]
        
        if isinstance(I,int): I = float(I)
        if isinstance(A,int): A = float(A)
        
        newElem = Beam(tag, iNode, jNode, E, A, I, **kwds)
        self.elems.append(newElem)
        self.delems.update({newElem.tag:newElem})
        # self.connect([iNode, jNode], "Beam") # considering deprecation
        iNode.elems.append(newElem)
        jNode.elems.append(newElem)

        if Qpl is not None:
            newElem.Qpl = np.zeros((3,2))
            newElem.Np = [Qpl[0], Qpl[0]]
            newElem.Mp = [[Qpl[1], Qpl[1]],[Qpl[2], Qpl[2]]]
            for i, key in enumerate(newElem.Qp['+']):
                newElem.Qp['+'][key] = newElem.Qp['-'][key] = Qpl[i] # consider depraction of elem.Qp in favor of elem.Qpl
                newElem.Qp['+'][key] = newElem.Qp['-'][key] = Qpl[i]
                newElem.Qpl[i,:] = Qpl[i] # <- consider shifting to this format for storing plastic capacities
        return newElem


    def girder(self, nodes, mats=None, xsecs=None, story=None):
        tags=[chr(i) for i in range(ord("a"), ord("a") + 26)]

        if mats is None: mats = [self.materials['default']]*(len(nodes)-1)
        if xsecs is None: xsecs = [self.xsecs['default']]*(len(nodes)-1)
        newElems = []

        for i, nd in enumerate(nodes[0:len(nodes)-1]):
            iNode = nd
            jNode = nodes[i+1]
            if story is None: tag = tags[i]
            else: tag = tags[story]+str(len(newElems)+1)
            newElem = Beam(tag, iNode, jNode, mats[i].E, xsecs[i].A, xsecs[i].I)
            self.elems.append(newElem)
            self.delems.update({newElem.tag:newElem})
            iNode.elems.append(newElem)
            jNode.elems.append(newElem)
            newElems.append(newElem)
        return newElems

    def frame(self, bays, stories, column_mat=None, column_sec=None, 
                                   girder_mat=None, girder_sec=None):
        """Macro for generating rectangular building frames
        
        Parameters
        ---------------------
        bays: tuple
            tuple containing bay width, and number of bays
        stories: tuple
            tuple
        column_mat: 
        
        """
        o = {'x':0.0, 'y': 0.0}
        w = bays[1]     # Bay width
        nb = bays[0]    # Number of bays
        h = stories[1]  # Story height
        ns = stories[0] # Number of stories

        if girder_mat==None: girder_mat = self.materials['default']
        if girder_sec==None: girder_sec = self.xsecs['default']
        if column_mat==None: column_mat = self.materials['default']
        if column_sec==None: column_sec = self.xsecs['default']

        self.snodes = []
        ntag = 1 # Counter for node tags
        ctag = 1 # Counter for column tags
        for s in range(ns+1):
            snodes = []
            for b in range(nb+1):
                snodes.append(self.node(str(ntag), o['x']+b*w, o['y']+s*h))
                ntag += 1
                if not s == 0:
                    self.beam('cl' + str(ctag), self.snodes[s-1][b], snodes[b], column_mat, column_sec)
                    ctag += 1
            if not s == 0: self.girder(snodes, [girder_mat]*nb, [girder_sec]*nb, s-1)
            self.snodes.append(snodes)

    def truss(self, tag: str, iNode, jNode, elem=None, mat=None, xsec=None, Qpl=None,A=None,E=None):
        if mat is None: mat = self.materials['default']
        if E is None: E = mat.E
        # cross section
        if xsec is None: xsec = self.xsecs['default']
        if A is None: A = xsec.A
        if isinstance(iNode,str):
            iNode = self.dnodes[iNode]
        if isinstance(jNode,str):
            jNode = self.dnodes[jNode]

        newElem = Truss(tag, iNode, jNode, E, A, elem=elem)
        self.delems.update({newElem.tag:newElem})
        self.elems.append(newElem)
        iNode.elems.append(newElem)
        jNode.elems.append(newElem)

        if Qpl is not None:
            newElem.Np = [Qpl[0], Qpl[0]]
            newElem.Qp['+']['1'] = newElem.Qp['-']['1'] = Qpl[0]
        return newElem
    
    # def tensor_truss(self, tag: str, iNode, jNode, mat=None, xsec=None, Qpl=None,A=None,E=None):
    #     if mat is None: mat = self.materials['default']
    #     if E is None: E = mat.E
    #     # cross section
    #     if xsec is None: xsec = self.xsecs['default']
    #     if A is None: A = xsec.A

    #     newElem = TensorTruss(tag, iNode, jNode, E, A)
    #     self.delems.update({newElem.tag:newElem})
    #     self.elems.append(newElem)
    #     iNode.elems.append(newElem)
    #     jNode.elems.append(newElem)

    #     if Qpl is not None:
    #         newElem.Np = [Qpl[0], Qpl[0]]
    #         newElem.Qp['+']['1'] = newElem.Qp['-']['1'] = Qpl[0]
    #     return newElem

    def taprod(self, tag: str, iNode, jNode, mat=None, xsec=None, Qpl=None,A=None,E=None):
        """Construct a tapered rod element with variable E and A values."""
        if mat is None: mat = self.materials['default']
        if E is None: E = mat.E
        # cross section
        if xsec is None: xsec = self.xsecs['default']
        if A is None: A = xsec.A
        
        newElem = TaperedTruss(tag, iNode, jNode, E, A)
        self.delems.update({newElem.tag:newElem})
        self.elems.append(newElem)
        iNode.elems.append(newElem)
        jNode.elems.append(newElem)

        if Qpl is not None:
            newElem.Np = [Qpl[0], Qpl[0]]
            newElem.Qp['+']['1'] = newElem.Qp['-']['1'] = Qpl[0]
        return newElem
    
    def truss3d(self, tag: str, iNode, jNode, mat=None, xsec=None):
        """Add an emme.Truss3d object to model
        
        Parameters
        ---------

        """
        if mat is None: mat = self.materials['default']
        if xsec is None: xsec = self.xsecs['default']
        newElem = Truss3D(tag, iNode, jNode, mat, xsec)
        self.elems.append(newElem)
        self.delems.update({newElem.tag:newElem})
        iNode.elems.append(newElem)
        jNode.elems.append(newElem)
        return newElem

    def hinge(self, elem, node): # pin a beam end.
        newHinge = Hinge(elem, node)
        self.hinges.append(newHinge)
        if node == elem.nodes[0]:
            elem.rel['2'] = True
            elem.q.pop('2')
            elem.basic_forces[1].rel = True
        elif node == elem.nodes[1]:
            elem.rel['3'] = True
            elem.q.pop('3')
            elem.basic_forces[2].rel = True
        else: print("Error: element {} is not bound to node {}.".format(elem, node))

        return newHinge

    def redundant(self, elem: object, nature):
        """
        nature:
        """

        newq = IntForce(elem, nature, nature)
        elem.red[nature] = True
        self.redundants.append(newq)

    def update(self, U):
        for node in self.nodes:
            delta = [0.,0.]
            for i, dof in enumerate(node.dofs[0:2]):
                if not node.rxns[i]:
                    try:
                        delta[i] = U[U.row_data.index(str(dof))]
                    except: pass
                node.x += delta[0]
                node.y += delta[1]

    def load_state(self, state):
        for elem in state["elems"]:
            if "q" in state["elems"][elem]:
                for i, q in enumerate(self.delems[elem].q):
                    self.delems[elem].q[q] = state["elems"][elem]["q"][i]


class rModel(Model):
    def __init__(self, ndm, ndf):
        super().__init__(ndm=2, ndf=3)
        self.material('default', 1.0)
        self.xsection('default', 1.0, 1.0)
    
    def isortho(self, elem):
        if (abs(elem.cs) == 1.0) or (abs(elem.sn) == 1.0):
            return True
        else:
            return False

    def numdofs(self):
        current_rxn = 1
        current_dof = 1
        rxn_ixs = []
        DOFs = [[0, 0, 0] for node in self.nodes]
        for i, node in enumerate(self.nodes):
            # x-dof
            dirn = 0
            if not(node.rxns[dirn]): # node is free
                if not(DOFs[i][dirn]): # node unassigned
                    for elem in node.elems:
                        if abs(elem.cs) == 1.0: # x-dof coupled to far end
                            if elem.nodes[0] == node:
                                far_node = self.nodes.index(elem.nodes[1])
                            if elem.nodes[1] == node:
                                far_node = self.nodes.index(elem.nodes[0])

                            if not(DOFs[far_node][dirn]): # Far node dof unassigned
                                if not(self.nodes[far_node].rxns[dirn]): # Far node is free
                                    DOFs[far_node][dirn] = current_dof
                                    DOFs[i][dirn] = current_dof
                                    current_dof += 1
                                else: # Far node is fixed
                                    DOFs[far_node][dirn] = current_rxn
                                    DOFs[i][dirn] = current_rxn
                                    current_rxn += 1
                                    rxn_ixs.append( (i,dirn) )
                                    rxn_ixs.append( (far_node,dirn) )
                            else: # Far node dof already assigned
                                DOFs[i][dirn] = DOFs[far_node][dirn]
                                if self.nodes[far_node].rxns[dirn]: # Far node is fixed
                                    rxn_ixs.append( (i,dirn) )
                        elif all([abs(elem.cs) != 1.0 for elem in node.elems]): # x-dof free/uncoupled
                            if not(DOFs[i][dirn]):
                                DOFs[i][dirn] = current_dof 
                                current_dof += 1
            else: # node is fixed
                if not(DOFs[i][dirn]): # node is unassigned
                    DOFs[i][dirn] = current_rxn 
                    current_rxn += 1
                    rxn_ixs.append( (i,dirn) )

            # y-dof
            dirn = 1
            if not(node.rxns[dirn]):
                if not(DOFs[i][dirn]):
                    for elem in node.elems:
                        if abs(elem.sn) == 1.0:
                            # get far node index
                            if elem.nodes[0] == node:
                                far_node = self.nodes.index(elem.nodes[1])
                            if elem.nodes[1] == node:
                                far_node = self.nodes.index(elem.nodes[0])
                            
                            if not(DOFs[far_node][dirn]):
                                if not(self.nodes[far_node].rxns[dirn]):
                                    DOFs[far_node][dirn] = current_dof
                                    DOFs[i][dirn] = current_dof
                                    current_dof += 1
                                else:
                                    DOFs[far_node][dirn] = current_rxn
                                    DOFs[i][dirn] = current_rxn
                                    current_rxn += 1
                                    rxn_ixs.append( (i,dirn) )
                                    rxn_ixs.append( (far_node,dirn) )
                            else: 
                                DOFs[i][dirn] = DOFs[far_node][dirn]
                                if self.nodes[far_node].rxns[dirn]:
                                    rxn_ixs.append( (i,dirn) )
                        elif all([abs(elem.sn) != 1.0 for elem in node.elems]):
                            if not(DOFs[i][dirn]):
                                DOFs[i][dirn] = current_dof
                                current_dof += 1
            else:
                if not(DOFs[i][dirn]):
                    DOFs[i][dirn] = current_rxn
                    current_rxn += 1
                    rxn_ixs.append( (i,dirn) )
            
          # rz-dof
            dirn = 2
            if not(node.rxns[2]):
                DOFs[i][dirn] = current_dof 
                current_dof += 1
            else:
                DOFs[i][dirn] = current_rxn 
                current_rxn += 1
                rxn_ixs.append( (i,dirn) )

        for ids in rxn_ixs:
            DOFs[ids[0]][ids[1]] += current_dof - 1

        return DOFs
         
    def _dof_num_2(self):
        
        pass

    def numDOF(self):
        crxns = self.ndf*self.nn - len(self.rxns)+1
        df = 1
        temp = []
        for node in self.nodes:
            DOFs = []
            for rxn in node.rxns:
                if not(rxn):
                    DOFs.append(df)
                    df += 1
                else:
                    DOFs.append(crxns)
                    crxns += 1
            temp.append(DOFs)
        self.DOF = temp
        return self.DOF

    @property 
    def triv_forces(self):
        """list of trivial axial forces"""
        lst = []
        for elem in self.elems:
            if len(elem.basic_forces) > 1:
                if elem.dofs[0]==elem.dofs[3] or elem.dofs[1]==elem.dofs[4]:
                    lst.append(elem.basic_forces[0])
        return np.array(lst)

    # @property
    # def basic_forces(self):
    #     # bmax = self.TrAx_forces
    #     forces = np.array([q for elem in self.elems for q in elem.basic_forces if not q.rel])
    #     return forces
    
    @property
    def cforces(self):
        triv = self.triv_forces
        arry = np.array([q for elem in self.elems for q in elem.basic_forces if (
            (q.plastic_event is None) and (
            not q in triv) and (
            not q.rel))])
        return arry

    @property
    def nr(self):
        return len(self.rxns)

    # @property
    # def nq(self):
    #     f = []
    #     for elem in self.elems:
    #         f.append(sum([1 for q in elem.basic_forces if not q.rel and (not q in self.triv_forces)]))
    #     return f
    
    @property
    def nv(self):
        """Returns number of element deformations in model"""
        lst = []
        for elem in self.elems:
            lst.append(sum([1 for x in elem.v]))
        return lst


    
   # @property
   # def fdof(self): 
   #     """Return list of free dofs"""
   #     pass

    @property
    def nt(self):
        nt = max([max(dof) for dof in self.DOF])
        return nt
    
   # @property
   # def nm(self):
   #     """No. of kinematic mechanisms, or the no. of dimensions spanned by the 
   #     m independent inextensional mechanisms.
   #     
   #     """
   #     # A = A_matrix(mdl)
   #     pass

    @property
    def NOS(self):
        nf = self.nf
        nq = sum(self.nq)
        return nq - nf



##<****************************Abstract objects*********************************
class Material():
    def __init__(self, tag, E, nu=None):
        self.tag = tag
        self.E: float = E
        self.elastic_modulus: float = E
        self.poisson_ratio = nu

class XSect():
    def __init__(self, tag, A, I):
        self.tag = tag
        self.A: float = A
        self.I: float = I

#def  UnitProperties(): # Legacy; consider removing and adding unit materials to model by default
#    return (Material('unit', 1.0), XSect('unit', 1.0, 1.0))

class State():
    """STATE is a data structure with information about the current state of the structure in fields

    """
    def __init__(self, model, method="Linear"):
        self.model = model
        self.num = len(model.states)
        self.data = {"Q": [[qi for qi in elem.q.values()] for elem in model.elems],
                     "P": {str(dof):0 for dof in [item for sublist in model.DOF for item in sublist]},
                     "DOF": 'model.numDOF(model)'
                    }
        self.method = method


    def eload(self, elem, mag, dirn='y'):
        if type(elem) is str:
            elem = self.model.delems[elem]
        if not(type(mag) is list):
            if dirn=='y':
                mag = [0.0, mag]
            elif dirn=='x':
                mag = [mag, 0.0]
        elem.w[self.num] = mag


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

        # Attributes for nonlinear analysis
        # self.xi: float = x # x-coordinate in reference configuration.  
        # self.yi: float = y # y-coordinate in reference configuration.  
        # self.zi: float = z # z-coordinate in reference configuration.  
        
        self.x: float = xyz[0]
        self.y: float = xyz[1]
        self.z: float = xyz[2] if len(xyz) > 2 else None

        
        self.rxns = [0]*ndf
        self.model = model
        self.mass = mass
        self.elems = []

        self.p = {dof:0.0 for dof in model.ddof}
        
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

SkeletalModel = Model


class Domain(Model):
    """Deprecated. Use Model instead."""
    pass


#def number_dof_plain(conn, boun, verbose=False):
#    """Basic dof numbering"""
#    ndf = max(len(con) for con in mesh.values())
#    nr = sum(sum(boun) for boun in bn.values())
#    nodes = {node for con in mesh.values() for node in con[1]}
#    nn = len(nodes)
#
#    crxns = ndf*nn - nr + 1
#
#    df = 1
#    temp = {}
#    try:
#        sorted_nodes = sorted(nodes, key=lambda k: int(k))
#    except:
#        sorted_nodes = sorted(nodes)
#    for node in sorted_nodes:
#        DOFs = []
#        try:
#            for rxn in bn[node]:
#                if not rxn:
#                    DOFs.append(df)
#                    df += 1
#                else:
#                    DOFs.append(crxns)
#                    crxns += 1
#        except KeyError:
#            df -= 1
#            DOFs = [df := df + 1 for _ in range(ndf)]
#            df += 1
#
#        temp[node] = DOFs
#    return temp
