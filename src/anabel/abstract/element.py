"""Element library
"""
import inspect
from typing import Union
from functools import partial
from abc import abstractmethod

from anabel.abstract.component import ModelComponent
import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
import scipy.integrate

import anon
try:
    import anon.atom as anp
except:
    anp = np

try:
    from emme.matrices import Structural_Matrix, Structural_Vector
except:
    from anabel.matrices import Structural_Matrix, Structural_Vector

class IntForce:
    def __init__(self, elem:object, number:int, nature:str=None):
        self.number:int = number
        self.nature:str = nature
        self.elem = elem
        self.rel = False
        self.redundant = False
        self.plastic_event = None

    @property
    def tag(self)->str:
        return self.elem.tag + '_' + str(self.number)

    def __str__(self):
        return self.tag

    def __repr__(self):
        return self.tag


class BasicLink():
    """Class implementing general geometric element methods"""

    def __init__(self, ndf, ndm, nodes):
        self.nodes = nodes
        self.ndf: int = ndf
        self.ndm: int = ndm
        self.nen = len(nodes)

    @property
    def loc(self)->anp.ndarray:
        return anp.array([node.xyz for node in self.nodes])

    @property
    def dofs(self):
        """

        This function is very slow.
        """
        eid = np.array([],dtype=int)
        for node in self.nodes:
            eid = np.append(eid, node.dofs[0:self.ndf])
        #untested alternative:
        #return np.array([node.dofs[0:self.ndf] for node in self.nodes])
        return eid

    @property
    def L(self):
        xyzi = self.nodes[0].xyz
        xyzj = self.nodes[1].xyz
        L = np.linalg.norm(xyzi-xyzj)
        return L

    @property
    def L0(self):
        xyzi = self.nodes[0].xyz0
        xyzj = self.nodes[1].xyz0
        L = np.linalg.norm(xyzi-xyzj)
        return L

    @property
    def Li(self):
        n1 = self.nodes[0]
        n2 = self.nodes[1]
        xyzi = np.array([n1.xi, n1.yi, n1.zi])
        xyzj = np.array([n2.xi, n2.yi, n2.zi])
        L = np.linalg.norm(xyzi-xyzj)
        return L


    @property
    def Dx(self)->float:
        return self.nodes[1].x - self.nodes[0].x

    @property
    def Dy(self)->float:
        return self.nodes[1].y - self.nodes[0].y

    @property
    def Dz(self)->float:
        return self.nodes[1].z - self.nodes[0].z

    @property
    def sn(self)->float:
        """directional sine"""
        L = self.L
        sn = (self.nodes[1].y - self.nodes[0].y)/L
        return sn

    @property
    def cs(self):
        """directional cosine"""
        L = self.L
        cs = (self.nodes[1].x - self.nodes[0].x)/L
        return cs

    @property
    def cz(self):
        L = self.L
        cz = (self.nodes[1].z - self.nodes[0].z)/L
        return cz

    def Rx_matrix(self):
        """Rotation about x

        """
        cs = self.cs
        sn = self.sn
        Rx = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0,  cs, -sn, 0.0, 0.0, 0.0],
            [0.0,  sn,  cs, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0,  cs, -sn],
            [0.0, 0.0, 0.0, 0.0,  sn,  cs],
        ])

        return 0

    def Ry_matrix(self):
        "Rotation about z"
        cs = self.cs
        sn = self.sn
        Ry = np.array([
            [ cs, 0.0,  sn],
            [0.0, 1.0, 0.0],
            [-sn, 0.0,  cs],
        ])

        return 0

    def Rz_matrix(self)->np.ndarray:
        "Rotation about z"
        cs = self.cs
        sn = self.sn
        Rz = np.array([
            [ cs, -sn, 0.0, 0.0, 0.0, 0.0],
            [ sn,  cs, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0,  cs, -sn, 0.0],
            [0.0, 0.0, 0.0,  sn,  cs, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        return Rz


class Element(BasicLink,ModelComponent):
    """Element parent class"""
    _tag: Union[int,None]

    def __init__(self,  ndf, ndm, force_dict=None, nodes=None, elem=None, resp=None, proto=None, name=None, tag=None, transformation=None):
        super().__init__(ndf, ndm, nodes)
        self._transformation = transformation
        self._resp = resp
        self.elem = elem
        if name is None and tag is not None:
            name = tag

        self._name = name
        self._tag = name if isinstance(name, int) else None

        self.history = {}
        self.current = {}
        if force_dict is None:
            force_dict = {str(i+1): 0 for i in range(ndf)}

        nq = len(force_dict)
        self.rel = {str(i): False for i in range(1, nq+1)}
        self.red = {str(i): False for i in range(1, nq+1)}
        self.q0  = {str(i): 0.  for i in range(1, nq+1)}
        self.v0  = {str(i): 0.  for i in range(1, nq+1)}
        self.e0  = {str(i): 0.  for i in range(1, nq+1)}
        self.Qp  = {'+': {str(i): 0.  for i in range(1, nq+1)}, '-':{str(i): 0.  for i in range(1, nq+1)}}

        self.basic_forces = np.array([IntForce(self, i, nature=str(i)) for i in range(1, nq+1)])
        self.basic_deformations = self.basic_forces
    @property
    def nn(self): return len(self.nodes)

    @property
    def tag(self):
        if self._tag is None:
            self._tag = self.domain.elems.index(self)
        return self._tag

    @property
    def transform(self):
        pass

    @property
    def resp(self):
        return self._resp

    @resp.setter
    def resp(self, value):
        self._resp = value

    @property
    def force_keys(self):
        return [self.tag+'_'+key for key in self.rel]

    def compose(self, **model_params):
        """
        created 2021-03-31
        """
        if self.elem is None:
            # create a default linear element
            ke = anp.array(self.ke_matrix())
            def f(x,y=None,state={},params={},**kwds):
                return None,ke@x,state
            stiff = lambda x,y,state={},params={},**kwds: ke
            return anon.dual.wrap(f,dim=(self.ndf,1),jacx=stiff)

        else:
            elem = self.elem
            #parameters = inspect.signature(elem).parameters
            #for name,param in model_params.items():
            #    parameters[name].replace(default=param)

            return elem



    def v0_vector(self):
        return np.array([0]*self.ndf*self.nn)

    def pw_vector(self):
        if all(self.w.values())==0.0:
            return np.array([0.0]*self.ndf*self.nn)

class PolyRod(Element):
    nv  = 1
    nn  = 2
    ndm = 1
    ndf = 1 # number of dofs at each node
    force_dict = {'1':0}

    def __init__(self, name, nodes, E, A):
        super().__init__(self.ndf, self.ndm, self.force_dict, nodes, name=name)
        self.E:float = E
        """Young's modulus of elasticity"""
        self.A:float = A
        """cross-sectional areal"""
        self.q = {'1':0}
        self.v = {'1':0}
        self.w = {'1':0.0}

        if isinstance(self.E,float):
            self.E = Polynomial([self.E])

        if isinstance(self.A,float):
            self.A = Polynomial([self.A])
    
    def N(self):
        L = self.L
        N1 = Polynomial([1,-1/L])
        N2 = Polynomial([0, 1/L])
        return np.array([[N1],[N2]])
    
    def B(self):
        N = self.N()
        return np.array([[Polynomial.deriv(Ni,1) for Ni in row] for row in N])

    def k_matrix(self):
        E = self.E
        A = self.A

        L = self.L
        B = self.B()
        k = Structural_Matrix([
            [quad(E*A*(B@B.T)[i,j],0,L)[0] for j in range(2)] for i in range(2)
        ])

        # Metadata
        k.tag = 'k'
        k.row_data = k.column_data = ['u_'+str(int(dof)) for dof in self.dofs]
        return k

    def ke_matrix(self):
        return self.k_matrix()

    def pw_vector(self):
        L = self.L
        pw = self.w['1']
        N = self.N()
        if isinstance(pw,np.polynomial.Polynomial) or isinstance(pw,float):
            # p1 = -quad(N[0]*pw,0,L)[0]
            # p2 = -quad(N[1]*pw,0,L)[0]
            p = np.array([[-quad(Ni*pw,0,L)[0] for Ni in row] for row in N])
        else:
            print('Unsupported load vector pw')

        return p

    ## Post Processing
    def localize(self,U_vector):
        dofs = [int(dof) for dof in self.dofs]
        return np.array([U_vector.get(str(dof)) for dof in dofs])
    
    def displx(self,U_vector):
        u = self.localize(U_vector)
        N = self.N()
        return N[0,0]*u[0] + N[1,0]*u[1]
        
    def strainx(self,U_vector):
        dofs = [int(dof) for dof in self.dofs]
        u = np.array([U_vector.get(str(dof)) for dof in dofs])
        B = self.B()
        return B[0,0]*u[0] + B[1,0]*u[1]

    def iforcex(self,U_vector):
        """Resisting force vector"""
        dofs = [int(dof) for dof in self.dofs]
        u = np.array([U_vector.get(str(dof)) for dof in dofs])
        B = self.B()
        P = self.E*self.A*(B[0,0]*u[0] + B[1,0]*u[1])
        return P

class MeshCell(Element):
    def __init__(self, name, nn, ndf, ndm, nodes):
        node_map = {}
        super().__init__(ndf, ndm, nodes=nodes, name=name)
    
    @property
    def dofs(self):
        """"""
        return np.asarray([node.dofs[0:self.ndf] for node in self.nodes])


    def ke_matrix(self, *args, **kwds):
        return anp.eye(self.ndm*self.nn)

class Truss(Element):
    nv = 1
    nn = 2
    nq = 1
    ndm = 2
    ndf = 2 #: number of dofs at each node
    force_dict = {'1':0}
    Qpl = np.zeros((2,nq))

    def __init__(self, tag, iNode, jNode, E=None, A=None, geo='lin',properties=None,**kwds):
        super().__init__(self.ndf, self.ndm, self.force_dict, [iNode, jNode], tag=tag, **kwds)
        if isinstance(properties,dict):
            E, A = properties['E'], properties['A']
        self.type = '2D truss'
        self.E = E
        self.A = A
        self.geo=geo
        self.Q = np.array([0.0])

        self.q = {'1':0}
        self.v = {'1':0}
        
        self.w = {'1':0.0}

    def __repr__(self):
        return 'truss-{}'.format(self.tag)
    
    def N(self):
        L = self.L
        N1 = Polynomial([1,-1/L])
        N2 = Polynomial([0, 1/L])
        return np.array([N1,N2])
    
    def B(self):
        L = self.L
        B1 = self.N()[0].deriv(1)
        B2 = self.N()[1].deriv(1)
        return np.array([B1,B2])

    def v0_vector(self):
        EA = self.E*self.A
        L = self.L
        e0 = self.e0
        q0 = self.q0
        w = self.w
        v0 =  np.array([e0['1']*L])
        v0 += [q0['1']*L/EA]
        v0 += [w['1']*L*L/(2*EA)]
        return v0
        
    def q0_vector(self):
        EA = self.E*self.A
        e0 = self.e0['1']
        q0 = self.q0['1']
        q0 = q0 - EA*e0
        return [q0]

    def pw_vector(self):
        L = self.L
        pw = self.w['1']
        N = self.N()
        if isinstance(pw,np.polynomial.Polynomial) or isinstance(pw,float):
            p1 = -quad(N[0]*pw,0,L)[0]
            p2 = -quad(N[1]*pw,0,L)[0]
        else:
            print('Unsupported load vector pw')

        return np.array([p1,0,p2,0])

    def ke_matrix(self):
        ag = self.ag()
        k = self.k_matrix()
        return ag.T@(k*ag)

    def kg_matrix(self, N):
        """return element local stiffness matrix"""
        E = self.E
        A = self.A
        L = self.L
        k = Structural_Matrix([N/L])
        # Metadata
        k.tag = 'kg'
        k.row_data = ['q_'+ key for key in self.q0.keys()]
        k.column_data = ['v_' + key for key in self.e0.keys()]
        k.c_cidx = k.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return k

    def k_matrix(self): 
        """return element local stiffness matrix"""

        E = self.E
        A = self.A
        L = self.L
        k = Structural_Matrix([E*A/L])
        # Metadata
        k.tag = 'k'
        k.row_data = ['q_'+ key for key in self.q0.keys()]
        k.column_data = ['v_' + key for key in self.e0.keys()]
        k.c_cidx = k.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return k

    def bg_matrix(self, **kwds):
        """return element static matrix, $\\mathbf{b}_g$"""

        cs = self.cs
        sn = self.sn
        bg = np.array([[-cs],
                       [-sn],
                       [ cs],
                       [ sn]])
        return bg

    def f_matrix(self, Roption=True):
        """return element flexibility matrix, $\\mathbf{f}$"""
        
        A = self.A
        E = self.E
        L = self.L
        f = Structural_Matrix([L/(E*A)])

        if Roption:
            pass

        # Metadata
        f.tag = 'f'
        f.column_data = ['q_' + key for key in self.q0.keys()]
        f.row_data = ['v_'+ key for key in self.e0.keys()]
        f.c_cidx = f.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]

        return f
    
    def ag(self): 
        cs = self.cs
        sn = self.sn
        ag = np.array([[-cs,-sn , cs, sn],])
        return ag
    

    def GLstrain(self):
        Li = self.Li
        L0 = self.L0
        E_GL = (Li**2 - L0**2) / (2*L0**2)
        return E_GL
    
    def iGLstrain(self):
        """incremental Green-Lagrange strain"""
        L  = self.L
        Li = self.Li
        E_GL = (L**2 - Li**2) / (2*Li**2)
        return E_GL

class TaperedTruss(Truss):
    def k_matrix(self):
        if isinstance(self.E,float):
            E = Polynomial([self.E])
        else:
            E = self.E
        if isinstance(self.A,float):
            A = Polynomial([self.A])
        else:
            A = self.A
        A = self.A 
        L = self.L
        B = self.B()
        # ke = np.zeros((self.ndf,self.ndf))
        # for i in range(self.ndf):
        #     for j in range(self.ndf):
        #         f = lambda x: B[i](x)*E(x)*A(x)*B[j](x)
        #         ke[i,j] = quad(f,0,L)[0]

        f = lambda x: B[0](x)*E(x)*A(x)*B[0](x)
        k = Structural_Matrix([quad(f,0,L)[0]])
        
        # Metadata
        k.tag = 'k'
        k.row_data = ['q_'+ key for key in self.q0.keys()]
        k.column_data = ['v_' + key for key in self.e0.keys()]
        k.c_cidx = k.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return k
    
    def q0_vector(self):
        # EA = self.E*self.A()
        # e0 = self.e0['1']
        # q0 = self.q0['1']
        # q0 = q0 - EA*e0
        return [0]
    
    def v0_vector(self):
        return [0]

class Truss3D(Element):
    ndf = 3
    ndm = 3
    force_dict = {'1':0}
    def __init__(self, tag, iNode, jNode, A, E):
        super().__init__(self.ndf, self.ndm, self.force_dict, [iNode,jNode], tag=tag)
        self.type = '2D truss'
        self.A = A
        self.E = E

        self.q = {'1':0}
        self.v = {'1':0}
    
    def __repr__(self):
        return 'tr-{}'.format(self.tag)
           
    def bg_matrix(self): 
        """return element static matrix, bg - pp. 57"""
        cs = self.cs
        sn = self.sn
        cz = self.cz
        bg = np.array([[-cs],
                       [-sn],
                       [-cz],
                       [ cs],
                       [ sn],
                       [ cz]])
        return bg

