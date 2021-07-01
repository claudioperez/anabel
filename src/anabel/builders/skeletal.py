# Claudio Perez
# anabel
"""
# Skeletal Model Builder

This page contains the `SkeletalModel` class. This class
provides a high level API for constructing skeletal structural
models which can be exported to an effecient backend implementation.

"""
# Standard library
import copy
import inspect
import functools
from inspect import signature
from functools import partial
from typing import Callable, List, Union

import jax
import numpy as np
import scipy.sparse
from mpl_toolkits.mplot3d import Axes3D

from anabel.abstract import DomainBuilder, Material, FrameSection, Node, Element
from anabel.constraints import SP_Constraint as Rxn
from anabel.template import get_unspecified_parameters, template
#from anabel.elements import *

try:
    import anon.atom as anp
except:
    anp = np

__all__ = ["SkeletalModel"]

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
        self.rxns:    list = []
        self.hinges:  list = []
        self.iforces: list = []
        self.states:  list = []
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



    def fix(self, node, *dirn): # for dirn enter string (e.g. "x", 'y', 'rz')
        """Define a fixed boundary condition at specified 
        degrees of freedom of the supplied node

        Parameters
        ----------
        node: anabel.Node

        dirn: Union[Sequence[String], String]
        
        ### Example

        ```py
        # Fix all dofs at node named "abut"
        model.fix("abut")

        # Create a pinned reaction at node 2
        model.fix(2, "y")

        # Create a node and impose a roller reaction
        a = model.node("a", 0.0, 0.0)
        model.fix(a, "y", "x")

        # Fix the rotational dof in a node a 3-dof model
        model.fix("n1", 0, 0, 1)
        ```
        """
        if not dirn:
            dirn = ["x","y","rz"]

        if isinstance(node,str):
            node = self.dnodes[node]

        rxns = []
        for dfs in dirn:
            if isinstance(dfs,list):
                for df in dirn:
                    if isinstance(df, str):
                        dof_name = self.ddof[df]
                    newRxn = Rxn(node, df, self.ddof[df])
                    self.rxns.append(newRxn)
                    rxns.append(newRxn)
                    node.rxns[self.ddof[df]] = 1
            else:
                newRxn = Rxn(node, dfs, self.ddof[dfs])
                self.rxns.append(newRxn)
                rxns.append(newRxn)
                node.rxns[self.ddof[dfs]] = 1
        if len(rxns) > 1:
            return rxns
        else:
            return rxns[0]

    def _fix_str_flags(self, node, flags):
        pass

    def _fix_int_flags(self, node, flags):
        pass

    def boun(self, node, flags: list):
        """
        Impose single-point constraints at the specified node. This 
        function is provided to give a familiar interface for users
        of platforms like FEAP and FEDEAS.

        Parameters
        ----------
        node: Union[anabel.abstract.Node, anabel.abstract.TagType]
            Node identifier

        flags: List[int]

        ### Example

        ```py
        model.boun(1, [0, 0, 1])
        ```

        """
        if isinstance(node,str):
            node = self.dnodes[node]

        for i, dof in enumerate(self.ddof):
            if flags[i]:
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
        """Add an arbitrary element to the structural model.

        Parameters
        ==========
        elem: Union[object, str]

        nodes: List[Node]

        """
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
        
        element._domain = self
        element._name = tag

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

    def hinge(self, elem, node):
        """
        Add a hinge to the structural model.
        """
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
        Identify an element unknown as redundant.
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


