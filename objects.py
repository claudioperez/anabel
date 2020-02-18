# UTF-8
# Claudio Perez
# ema


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
from ema.elements import *
import sympy as sp
import openseespy as ops


class Model:
    def __init__(self, ndm, ndf):
        """Basic model object
        
        Parameters
        -----------
        ndm:
            number of model dimensions
        ndf:
            number of degrees of freedom (dofs) at each node

        """
        self.ndf: int = ndf
        self.ndm: int = ndm
        self.DOF: list = None

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
        self.rxns:  list = []
        self.hinges: list = []
        self.iforces: list = []
        self.loads: list = []
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

    @property
    def rel(self):
        return [rel for elem in self.elems for rel in elem.rel.values()]

    @property
    def nn(self) -> int:
        """return number of nodes in model"""
        return len(self.nodes)
    
    @property
    def nr(self) -> int:
        """return number of constrained dofs in model"""
        return len(self.rxns)
    
    @property
    def ne(self) -> int:
        """return number of elements in model"""
        return len(self.elems)

    @property
    def nQ(self):
        f = 0
        for elem in self.elems:
            f += len(elem.basic_forces)
        return f

    @property
    def nq(self):
        f = []
        for elem in self.elems:
            f.append(sum([1 for x in elem.q]))
        return f
    
    @property
    def nv(self):
        lst = []
        for elem in self.elems:
            lst.append(sum([1 for x in elem.v]))
        return lst

    @property
    def nf(self) -> int:
        x = self.nt - self.nr
        return x

    @property
    def nt(self) -> int:
        return self.ndf*self.nn

    @property
    def rdofs(self): 
        """Return list of restrained dofs"""
        DOF = self.DOF

        return []

    @property
    def NOS(self) -> int:
        nf = self.nf
        nq = sum(self.nq)
        return nq - nf

    @property
    def basic_forces(self):
        return np.array([q for elem in self.elems for q in elem.basic_forces ])

    @property
    def rdnt_forces(self):
        cforces = self.cforces
        return np.array([q  for q in cforces if q.redundant ])
    

    @property
    def cforces(self):
        return np.array([q for elem in self.elems for q in elem.basic_forces if not q.rel])
    
    @property 
    def eforces(self):
        """Return array of elastic element forces"""
        return np.array([q for elem in self.elems for q in elem.basic_forces if (q.plastic_event is None)])

    @property
    def idx_c(self):
        cforces = self.cforces
        forces = self.basic_forces
        idx_c = np.where(np.isin(forces,cforces))[0]
        return idx_c

    @property
    def idx_e(self):
        """return indices of elastic basic (not plastic) forces"""
        cforces = self.cforces
        eforces = self.eforces
        idx_e = np.where(np.isin(cforces,eforces))[0]
        return idx_e

    @property
    def idxx_f(self):
        return None
    
    @property
    def idx_i(self):
        rdts = self.rdnt_forces
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
        newNode = Node(self, tag, self.ndf, [x, y, z], mass)
        self.nodes.append(newNode)
        self.dnodes.update({newNode.tag : newNode})
        return newNode
    
    def state(self, method="Linear"):
        if self.DOF==None: self.numDOF()
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

    def fix(self, node, dirn): # for dirn enter string (e.g. "x", 'y', 'rz')
        if type(dirn) is list:
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
        for i, dof in enumerate(self.ddof):
            if ones[i]:
                self.fix(node, dof)

    def pin(self, node):
        """creates pinned reaction at specified node

        """
        self.fix(node, ['x', 'y'])
        if self.ndm == 3:
            self.fix(node, 'z')
        return

    def roller(self, node):
        """creates roller reaction at specified node

        """
        self.fix(node, 'y')
        return

    # def connect(self, connodes, elemType):
    #     tempNodes = []
    #     for point in connodes:
    #         tempNodes.append(self.nodes.index(point))
    #     # self.CON.append(tempNodes)
    #     return 0

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
        ---------
        element : obj

        """

        self.elems.append(element)
        self.delems.update({element.tag:element})

        for node in element.nodes:
            node.elems.append(element)
        
        return element


    def beam(self, tag: str, iNode, jNode, mat=None, sec=None, Qpl=None):
        """Add a 2D linear Euler-Bernouli beam object to model
        
        Parameters
        ---------
        tag : str
            string used for identifying object
        iNode : ema.Node
            node object at element i-end
        jNode : ema.Node
            node object at element j-end
        mat : ema.Material 

        sec : ema.Section
        

        """

        if mat is None:
            mat = self.materials['default']
        if sec is None:
            sec = self.xsecs['default']
        newElem = Beam(tag, iNode, jNode, mat.E, sec.A, sec.I)
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

    def truss(self, tag: str, iNode, jNode, mat=None, xsec=None, Qpl=None,A=None,E=None):
        if mat is None: mat = self.materials['default']
        if E is None: E = mat.E
        # cross section
        if xsec is None: xsec = self.xsecs['default']
        if A is None: A = xsec.A

        newElem = Truss(tag, iNode, jNode, E, A)
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
        """Add an ema.Truss3d object to model
        
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
    
    
    def redundant(self, elem, nature):
        newq = IntForce(elem, nature)
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


    
    @property
    def fdof(self): 
        """Return list of free dofs"""
        pass

    @property
    def nt(self):
        nt = max([max(dof) for dof in self.DOF])
        return nt
    
    @property
    def nm(self):
        """No. of kinematic mechanisms, or the no. of dimensions spanned by the 
        m independent inextensional mechanisms.
        
        """
        # A = A_matrix(mdl)
        pass

    @property
    def NOS(self):
        nf = self.nf
        nq = sum(self.nq)
        return nq - nf


class DofSpace:

    def __init__ ( self, Model ):
        self.ndf = Model.ndf
        self.nn = Model.nn
        # self.dofTypes = elements.getDofTypes()
        self.dofs     = np.array( range( self.nn * self.ndf ) ).reshape( ( self.nn, self.ndf ) )
        self.model = Model

        #Create the ID map
        # self.IDmap = itemList()
        self.IDmap = {}

        for ind,node in enumerate(Model.nodes):
            if node.tag in self.IDmap:
                raise RuntimeError( 'ID ' + str(node.tag) + ' already exists in ' + type(self.IDmap).__name__ )
            self.IDmap[node.tag] = ind

        self.constrainedDofs = []
        self.constrainedVals = []
        self.constrainedFac  = 1.0

    def __str__ ( self ):
        return str(self.model.dofs)

    def __len__ ( self ):
        return self.model.nt

    def setConstrainFactor( self , fac ):
        self.constrainedFac = fac
    
    def readFromFile( self, fname ):
        pass

    def constrain ( self, nodeID, dofTypes , val = 0. ):
        if not nodeID in self.model.nodes:
            raise RuntimeError('Node ID ' + str(nodeID) + ' does not exist')

        ind = self.IDmap.get( nodeID )

        if isinstance( dofTypes, str ):
            dofTypes = [dofTypes]

        #Check if the dofTypes exist
        for dofType in dofTypes:
            if dofType not in self.dofTypes:
                raise RuntimeError('DOF type "' + dofType + '" does not exist')
            
        for dofType in dofTypes:    
            self.constrainedDofs.append( self.dofs[ind,self.dofTypes.index(dofType)] )
            self.constrainedVals.append( val )

    def getForType ( self, nodeIDs, dofType ):
        return self.dofs[self.IDmap.get( nodeIDs ),self.dofTypes.index(dofType)]

    def get ( self, nodeIDs ):
        return self.dofs[self.IDmap.get(nodeIDs)].flatten()

    def getConstraintsMatrix ( self ):

        n_constrained = len( self.constrainedDofs )
        n             = len( self )

        C = np.zeros( (n,n-n_constrained) )
        j = 0
        
        for i in range(n):
        
            if i in self.constrainedDofs:
                continue

            C[i,j] = 1.
            j+=1

        return C

    def solve ( self, A, b ):

        if len(A.shape) == 2:
            C = self.getConstraintsMatrix()

            a = np.ALLOW_THREADSzeros(len(self))
            a[self.constrainedDofs] = self.constrainedFac * array(self.constrainedVals)

            A_constrained = np.dot( dot( C.transpose(), A ), C )
            b_constrained = np.dot( C.transpose(), b + np.dot( A , -a ) )

            x_constrained = scipy.linalg.solve( A_constrained, b_constrained )

            x = np.dot( C, x_constrained )

            x[self.constrainedDofs] = self.constrainedFac * np.array(self.constrainedVals)
            
        elif len(A.shape) == 1:
            x = b / A

            x[self.constrainedDofs] = self.constrainedFac * array(self.constrainedVals)
        
        return x
    
    def eigensolve( self, A , B ):

        C = self.getConstraintsMatrix()

        A_constrained = ( C.T @ A ) @ C 
        B_constrained = ( C.T @ B ) @ C 

        #x_constrained = scipy.linalg.solve( A_constrained, b_constrained )

        return A,B

    def norm ( self, r ):

        C = self.getConstraintsMatrix()
        
        return scipy.linalg.norm( C.T @ r )

##>*****************************************************************************

##<****************************Abstract objects*********************************
class Material():
    def __init__(self, tag, E):
        self.tag = tag
        self.E: float = E

class XSect():
    def __init__(self, tag, A, I):
        self.tag = tag
        self.A: float = A
        self.I: float = I

def  UnitProperties(): # Legacy; consider removing and adding unit materials to model by default
    return (Material('unit', 1.0), XSect('unit', 1.0, 1.0))

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


    
    # def nLoad(self, node, dirn, Quant):
    #     """chargement"""
    #     nodes = self.model.nodes
    #     nodeNum = nodes.index(node)
    #     dof = self.model.DOF[nodeNum][self.model.ddof[dirn]]
    #     self.data['P'][str(dof)] = Quant
    #     return sp.symbols('P^'+node.tag+"_"+dirn)

    def eload(self, elem, mag, dirn='y'):
        if type(elem) is str:
            elem = self.model.delems[elem]    
        if not(type(mag) is list):
            if dirn=='y':
                mag = [0.0, mag]
            elif dirn=='x':
                mag = [mag, 0.0]
        elem.w[self.num] = mag

##>*****************************************************************************

##<****************************Model objects************************************
# Model objects/classes
# These should be created using the methods above
#******************************************************************************
class Node():
    def __init__(self, model, tag: str, ndf, xyz, mass=None):
        if mass is None: mass=0.0

        self.xyz = np.array([xi for xi in xyz if xi is not None])
        
        self.tag = tag
        self.xyz0 = self.xyz # coordinates in base configuration (unstrained, not necessarily unstressed).
        self.xyzi = self.xyz # coordinates in reference configuration.  

        # self.x0: float = x # x-coordinate in base configuration (unstrained, not necessarily unstressed).  
        # self.y0: float = y # y-coordinate in base configuration (unstrained, not necessarily unstressed).  
        # self.z0: float = z # z-coordinate in base configuration (unstrained, not necessarily unstressed).  
        
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
        # if self.model.DOF == None: self.model.numDOF()
        idx = self.model.nodes.index(self)
        return self.model.DOF[idx]




class Rxn():
    def __init__(self, node, dirn):
        self.node = node
        self.dirn = dirn

    def __repr__(self):
        return 'rxn-{}'.format(self.dirn)
    
    @property
    def dof(self):
        model = self.node.model
        idof = self.dirn


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

class Domain(Model):
    """Deprecated. Use Model instead."""
    pass