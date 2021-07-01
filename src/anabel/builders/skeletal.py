# Claudio Perez
# anabel
"""
# Assemblers(`2`)

Core model building classes.
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

from anabel.abstract import DomainBuilder, Material, FrameSection, Node, Element
from anabel.template import get_unspecified_parameters, template
from anabel.elements import *

try:
    import anon.atom as anp
except:
    anp = np

__all__ = ["MeshGroup", "Model", "rModel", "SkeletalModel"]

def _is_sequence(obj):
    return hasattr(type(obj), '__iter__')

class SkeletalModel(DomainBuilder):
    def __init__(self, ndm:int, ndf:int, units="metric"):
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
        self.dtype    ='float32'
        self._units   = units

        self.clean()

    @property
    def units(self):
        import elle.units
        return elle.units.UnitHandler(self._units)

    def dump_opensees(self, **kwds):
        ndm, ndf = self.ndm, self.ndf
        cmds  = f"# Create ModelBuilder (with {ndm} dimensions and {ndf} DOF/node)"
        cmds += f"\nmodel BasicBuilder -ndm {ndm} -ndf {ndf}" 
        rxns = "\n".join([f"{r.dump_opensees()}" for r in self.rxns])
        return "\n\n".join([cmds, super().dump_opensees(**kwds), rxns])
    
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
                    for fixity in node.rxns 
                ] for node in self.nodes
            ]



    def compose(self,resp="d",jit=True,verbose=False,**kwds):
        return self.compose_param(jit_force=jit,verbose=verbose,**kwds)

    def compose_param(self,f=None,jit_force=True,verbose=False,**kwds):
        if f is None:
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
        expressions = self.expressions
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
                            f"'{k}': {v.default.name}" if "expr" not in v.default.name 
                            else f"'{k}': expressions['{v.default.name}']['expression']({','.join(expressions[v.default.name]['params'])})" 
                    for k,v in params["params"].items() if v.default
                ) + "}"
                if ls:
                    return  ",".join((ls, subparams))
                else:
                    return subparams

            else:
                return ls
        exec(
            f"def collect_params({','.join(p for p in self.params if 'expr' not in p )}):\n"
             "  return {'params': {\n"
            f"""    {cnl.join(f'"{tag}": {{ {_unpack(el)} }}' for tag,el in model_map.items()) }\n"""
             "  }}",
             local_scope
        )
        collect_params = local_scope["collect_params"]

        #-------------------
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
            f"def displ({','.join(p for p in self.params if 'expr' not in p)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params if 'expr' not in p)})\n"""
            f"   return f(collect_loads({','.join(p for p in parameterized_loads.values() if isinstance(p,str) )}), f.origin[1], f.origin[2], params)[1]",
            local_scope
        )
        main = local_scope["displ"]
        origin = tuple(anp.zeros(p.shape) if hasattr(p,"shape") else 0.0 for p in self.params)

        # Evaluate once with zeros to JIT-compile
        return main
    
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
        if "steps" in kwds:
            solver_opts.update({"steps": kwds["steps"]})
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

        def jac_x(u,p,state,xyz=None,params=param_arg):
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

    def double_node(self, tag: str, *coords, mass : float = None):
        self.node(".double." + tag, *coords, mass=mass)
        return self.node(tag, *coords, mass=mass)
 
    def node(self, tag: str, x: float, y=None, z=None, mass: float=None):
        """Add a new emme.Node object to the model

        Parameters
        ----------
        x, y, z: float
            Node coordinates.
        """
        newNode = Node(self, tag, self.ndf, [x, y, z], mass)
        self.nodes.append(newNode)
        self.dnodes.update({tag: newNode})
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
        """Define a fixed boundary condition at specified 
        degrees of freedom of the supplied node

        Parameters
        ----------
        node: anabel.Node

        dirn: Sequence[String]

        """
        if isinstance(node,str):
            node = self.dnodes[node]

        if isinstance(dirn,list):
            rxns = []
            for df in dirn:
                newRxn = Rxn(node, df, self.ddof[df])
                self.rxns.append(newRxn)
                rxns.append(newRxn)
                node.rxns[self.ddof[df]] = 1
            return rxns
        else:
            newRxn = Rxn(node, dirn, self.ddof[df])
            self.rxns.append(newRxn)
            node.rxns[self.ddof[dirn]] = 1
            return newRxn

    def boun(self, node, ones: list):
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
        node: anabel.Node
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
        newXSect = FrameSection(tag=tag, area=A, I=I)
        self.xsecs[tag] = newXSect
        return newXSect

 # Elements
    def elem(self, elem, nodes, tag):
        if isinstance(tag, (list, tuple)):
            nodes, tag = tag, nodes

        if isinstance(nodes[0], str):
            nodes = [self.dnodes[node_tag] for node_tag in nodes]


        if isinstance(elem, Element):
            ndf = elem.ndf
            element = copy.copy(elem)
            element.nodes = nodes

        else:
            ndf = elem.shape[0][0]
            element = Element(ndf, self.ndm, nodes=nodes, elem=elem)
        
        element.domain = self
        self.elems.append(element)
        self.delems.update({tag: element})
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


