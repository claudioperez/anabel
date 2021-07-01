# Claudio Perez
# anabel
"""
# DomainBuilders(`2`)

Core model building classes.
"""
# Standard library
import inspect
import functools
from inspect import signature
from typing import Callable, List, Union

import jax
import meshio
import numpy as np
import scipy.sparse
from mpl_toolkits.mplot3d import Axes3D

#from anon import diff
from anabel.template import get_unspecified_parameters, template


try:
    import anon.atom as anp
except:
    anp = np

__all__ = ["MeshGroup", "Model", "rModel", "SkeletalModel"]

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

class DomainBuilder:
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
        return len(self.elems)

    @property
    def nf(self) -> int:
        "Number of free degrees of freedom."
        return  self.nt - self.nr

    def dump(self, writer):
        return writer(self).dump()



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


class UniformDomainBuilder(DomainBuilder):
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


