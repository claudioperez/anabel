# Claudio Perez
# ema
"""Core model building classes.
"""
import inspect
from inspect import signature
from functools import partial
from typing import Callable, List
import copy

import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from anon.dual import get_unspecified_parameters
from emme.elements import *
try:
    import anon.atom as anp
except:
    anp = np

__all__ = ["Assembler", "Model", "rModel", "SkeletalModel"]

def _is_sequence(obj):
    return hasattr(type(obj), '__iter__')


class Assembler:
    def __init__(self, ndm:int, ndf:int):
        """Basic assembler class

        An assembler is typically characterized by collections of
        nodes, elements and parameters.

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
        self.dtype='float32'

        # model inventory lists
        self.elems: list = []
        self.nodes: list = []

        # model inventory dictionaries
        self.delems: dict = {}
        self.dnodes: dict = {}




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
        #super().__init__(ndm, ndf)
        self.ndf: int = ndf
        self.ndm: int = ndm
        self.DOF: list = None
        self.dtype='float32'

        # Define DOF list indexing 
        if ndm == 1:
            self.prob_type = '1d'
            self.ddof: dict = {'x': 0}  # Degrees of freedom at each node

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

        # model inventory lists
        self.elems: list = []
        self.nodes: list = []
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


    def compose(self,resp="d",jit=True,**kwds):
        return self.compose_param(jit_force=jit,**kwds)

    def compose_param(self,jit_force=True,**kwds):
        f = self.compose_displ(jit_force=jit_force,**kwds,_expose_closure=True)
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
            f"def main({','.join(p for p in self.params)}): \n"
          f"""   params = collect_params({','.join(p for p in self.params)})\n"""
            f"   return f(collect_loads({','.join(p for p in parameterized_loads.values() if isinstance(p,str) )}), f.origin[1], f.origin[2], params)[1]",
            local_scope
        )
        main = local_scope["main"]
        return main
        #return collect_params

    @anon.dual.generator(dim="shape")
    def compose_displ(self, solver=None, solver_opts={}, elem=None, jit_force=True, **kwds):
        """
        Dynamically creates functions `collect_loads` and `collect_coord`.
        """
        if solver is None:
            import elle.numeric
            solver = elle.numeric.inverse.inv_no1
        f = self.compose_force(_jit=jit_force,**kwds,_expose_closure=True)
        model_map = f.closure["model_map"]

        state = {...: f.origin[2]}
        nf = self.nf
        #shape = self.nf
        #shape = (self.nf,self.nf)
        shape = ((nf,1),(nf,1))
        u0 = anp.zeros((self.nf,1))

        #local_scope = locals()


        model_map = f.closure["model_map"]
        param_map = f.closure["param_map"]
        main = solver(f,**solver_opts)
        # def main(p, u, state=state, params={}):
        #     return g()

        return locals()

    @anon.dual.generator(dim="shape")
    def compose_force(self, elem=None,**kwds)->Callable:
        """A simple force composer for skeletal truss structures."""
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
        #-------------------------
        #exec(
        #f"def collect_displ({','.join(p for p in parameterized_loads.values() if isinstance(p,str) )}):\n"
        # "  return anp.array([\n"
        #f"""    {','.join("["+u.name+"]" if isinstance(u,inspect.Parameter) else f'[0.0]' for node in self.nodes for dof,fixity,p in zip(node.dofs,node.rxns,node.p.values()) if not fixity ) }\n"""
        # "  ])",
        # dict(anp=anp),local_scope, 
        #)
        #collect_displ = local_scope["collect_displ"]


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

        #params.update({
        #    p.name: p for el in param_map.values() for p in el.values()
        #})

        DOF = {node_tag: dofs for node_tag, dofs in zip(self.dnodes.keys(),self.DOF)}

        #el_sign = {  tag: 1.0**(n[1][0] > n[1][1]) for tag, n in mesh.items() }
        el_DOF  = { elem.tag: elem.dofs for elem in self.delems.values() }

        Be = anp.concatenate([
             anp.array([[1.0 if j==i else 0.0 for j in range(1,nf+1)] for i in el.dofs ]).T
             for el in self.delems.values()], axis=1)

        state = {
            ... : {
                tag: m.origin[2] for tag, m, in model_map.items()
            }
        }
        xyz = eval(f"""{{ { ','.join(f'"{tag}": [{",".join(str(x) for x in node.xyz)}]' for tag, node in self.dnodes.items() )} }}""")
        def main(u,p,state, xyz=None, params=param_arg):
            U = anp.concatenate([u,anp.zeros((nr,1))],axis=0)
            coords = collect_coords(xyz)
            F = anp.concatenate([
                el(
                    anp.take(U, anp.array(el_DOF[tag], dtype='int32')-1)[:,None],
                    None,
                    xyz = coords[tag],
                    state = state[...][tag],
                    **params[tag]
                )[1]
                for tag,el in model_map.items()] ,axis=0)
            return u, (Be @ F), state

        return locals()


    @property
    def rel(self):
        return [rel for elem in self.elems for rel in elem.rel.values()]

    @property
    def nn(self) -> int:
        """Number of nodes in model"""
        return len(self.nodes)

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
            f.append(sum([1 for x in elem.q]))
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

    def load(self,obj,*args,pattern=None,**kwds):
        """
        Apply a load to a model object
        
        Claudio Perez 2021-04-01
        """
        if isinstance(obj,Node):
            print(obj)
        elif isinstance(obj,str):
            return self.load_node(self.dnodes[obj],*args,**kwds)

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

    def param(self,*param_names,shape=0,dtype=float,default=None):
        """Add a parameter to a model.
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
        """Add a beam object to model

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
            E = kwds["E"] if "E" in kwds else self.materials["default"].E
        else:
            E = mat.E

        if sec is None:
            A = kwds["A"] if "A" in kwds else self.xsecs["default"].A
            I = kwds["I"] if "I" in kwds else self.xsecs["default"].I
        else:
            A = sec.A
            I = sec.I

        if isinstance(iNode,str):
            iNode = self.dnodes[iNode]
        if isinstance(jNode,str):
            jNode = self.dnodes[jNode]

        
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
        self.z0: float = xyz[2] # z-coordinate in base configuration (unstrained, not necessarily unstressed).  

        # Attributes for nonlinear analysis
        # self.xi: float = x # x-coordinate in reference configuration.  
        # self.yi: float = y # y-coordinate in reference configuration.  
        # self.zi: float = z # z-coordinate in reference configuration.  
        
        self.x: float = xyz[0]
        self.y: float = xyz[1]
        self.z: float = xyz[2]

        
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
        return np.array(self.model.DOF[idx],dtype=int)


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

# ##>*****************************************************************************

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

class SkeletalModel(Model):
    pass

class Domain(Model):
    """Deprecated. Use Model instead."""
    pass

#----------------------------------------------------------------------
#
#----------------------------------------------------------------------


