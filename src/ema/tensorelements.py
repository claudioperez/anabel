import numpy as np
# import sympy as sp
from abc import abstractmethod
from numpy.polynomial import Polynomial
from scipy.integrate import quad
import tensorflow as tf

# from ema.utilities import Structural_Matrix, Structural_Vector
from ema.matrices import Structural_Matrix, Structural_Vector
from ema.elements import IntForce
dtype = 'float32'
int_dtype='int32'
tf.keras.backend.set_floatx(dtype)


class TensorLink:
    """Class implementing general geometric element methods"""

    def __init__(self, ndf, ndm, nodes):
        self.nodes = nodes
        self.ndf: int = ndf
        self.ndm: int = ndm
        self.nen = len(nodes)
        
    @property
    def dofs(self):
        return tf.reshape(tf.constant([node.dofs[0:self.ndf] for node in self.nodes], dtype=int_dtype),[self.ndf*self.nen])
    
    @property
    def L(self):
        xyzi = tf.constant(self.nodes[0].xyz,dtype=dtype)
        xyzj = tf.constant(self.nodes[1].xyz,dtype=dtype)
        L = tf.linalg.norm(xyzi-xyzj)
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
        xyzi = tf.Variable([n1.xi, n1.yi, n1.zi],dtype=dtype)
        xyzj = tf.Variable([n2.xi, n2.yi, n2.zi],dtype=dtype)
        L = np.linalg.norm(xyzi-xyzj)
        return L


    @property
    def Dx(self):
        return self.nodes[1].x - self.nodes[0].x
    
    @property
    def Dy(self):
        return self.nodes[1].y - self.nodes[0].y

    @property
    def Dz(self):
        return self.nodes[1].z - self.nodes[0].z

    @property
    def sn(self):
        L = self.L
        sn = (self.nodes[1].y - self.nodes[0].y)/L
        return sn

    @property
    def cs(self):
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
        
        https://en.wikipedia.org/wiki/Rotation_matrix
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

    def Rz_matrix(self):
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

class TensorElement(TensorLink):
    """Element parent class"""
   
    def __init__(self,  ndf, ndm, force_dict, nodes):
        super().__init__(  ndf, ndm, nodes)

        self.history = {}
        self.current = {}

        nq = len(force_dict)
        self.rel = {str(i): False for i in range(1, nq+1)}
        self.red = {str(i): False for i in range(1, nq+1)}
        self.q0  = {str(i): 0.  for i in range(1, nq+1)}
        self.v0  = {str(i): 0.  for i in range(1, nq+1)}
        self.e0  = {str(i): 0.  for i in range(1, nq+1)}
        self.Qp  = {'+': {str(i): 0.  for i in range(1, nq+1)}, '-':{str(i): 0.  for i in range(1, nq+1)}}
        
        self.basic_forces = np.array([IntForce(self, i) for i in range(1, nq+1)])
        self.basic_deformations = self.basic_forces

    @property
    def force_keys(self):
        return [self.tag+'_'+key for key in self.rel]
    
    
    def v0_vector(self):
        return np.array([0]*self.ndf*self.nn)
        
    def pw_vector(self):
        if all(self.w.values())==0.0:
            return np.array([0.0]*self.ndf*self.nn)
    
class TensorRod(TensorElement):
    nv  = 1
    nn  = 2
    ndm = 1
    ndf = 1 # number of dofs at each node
    force_dict = {'1':0}

    def __init__(self,tag, nodes, E, A):
        super().__init__(self.ndf, self.ndm, self.force_dict, nodes)
        self.tag = tag
        self.E = E 
        self.A = A
        self.q = {'1':0}
        self.v = {'1':0}
        self.w = {'1':0.0}

        if isinstance(self.E,float):
            self.E = tf.tensor([self.E])

        if isinstance(self.A,float):
            self.A = tf.tensor([self.A])
    
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

        def F(i,j):
            pol = (B@B.T)[i,j]
            return lambda x: pol(x)*E*A

        k = Structural_Matrix([
            [quad(F(i,j),0,L)[0] for j in range(2)] for i in range(2)
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

class TensorTruss(TensorElement,tf.keras.layers.Layer):
    nv : int = 1
    nn : int = 2
    nq : int = 1
    ndm: int = 2
    ndf: int = 2 # number of dofs at each node
    force_dict = {'1':0.0}
    Qpl = tf.zeros((2,nq))

    def __init__(self, tag, iNode, jNode, E=None, A=None, properties=None,geom='lin'):
        super().__init__(self.ndf, self.ndm, self.force_dict, [iNode, jNode])
        
        if isinstance(properties,dict):
            E, A = properties['E'], properties['A']

        self.type = '2D truss'
        self.tag = tag
        self.geom = geom
        self.E = tf.constant( E , dtype=dtype)
        self.A = tf.constant( A , dtype=dtype)
        self.Q = tf.constant(0.0, dtype=dtype)

        self.q = {'1':tf.constant(0.0,dtype=dtype)} # 
        self.v = {'1':tf.constant(0.0,dtype=dtype)} # 

        self.s0 = tf.constant(0.0, dtype=dtype) 
        self.e0 = tf.constant(0.0, dtype=dtype) 
        
        self.w = {'1': 0.0}

    def __repr__(self):
        return 'truss-{}'.format(self.tag)
    
    
    def v0_vector(self):
        EA = self.E*self.A
        L  = self.L
        e0 = self.e0
        q0 = self.q0
        w  = self.w
        v0 =  np.array([e0['1']*L])
        v0 += [q0['1'] * L/EA]
        v0 += [ w['1'] * L*L/(2*EA)]
        return v0
        
    def q0_vector(self):
        EA = self.E*self.A
        e0 = self.e0['1']
        q0 = self.q0['1']
        q0 = q0 - EA*e0
        return [q0]
    
    def ke_matrix(self,U):
        ag = self.ag(U)
        k = self.k_matrix( U )
        return ag.T@( k*ag )

    def k_matrix(self, U ): 
        """return element local stiffness Matrix"""

        E = self.E               
        A = self.A               
        L = self.L               
        k = tf.math.multiply(E,A)
        k = tf.math.divide(k,L)
        return k

    def ag(self,u): 
        cs = self.cs
        sn = self.sn
        ag = np.array([[-cs],[-sn] , [cs], [sn]]).T
        return ag

    def v_layer(self,xyz,U):
        L = tf.linalg.norm(xyz[...,1,:]-xyz[...,0,:])
        DX = xyz[...,1,0] - xyz[...,0,0]
        DY = xyz[...,1,1] - xyz[...,0,1]

        v = (U[...,2]-U[...,0])*DX/L+ (U[...,1]-U[...,3])*DY/L + ((U[...,2]-U[...,0])**2 + (U[...,1]-U[...,3])**2)/(2*L)
        return v
    
    def e_layer(self,xyz,U):
        L = tf.linalg.norm(xyz[...,1,:]-xyz[...,0,:])
        DX = xyz[...,1,0] - xyz[...,0,0]
        DY = xyz[...,1,1] - xyz[...,0,1]

        v = (U[...,2]-U[...,0])*DX/L + (U[...,1]-U[...,3])*DY/L + ((U[...,2]-U[...,0])**2 + (U[...,1]-U[...,3])**2)/(2*L)
        return v/L
    
    def dv_layer(self,xyz,U,dU):
        # U = tf.gather(U,dofs)
        L = tf.linalg.norm(xyz[...,1,:] - xyz[...,0,:])
        DX = tf.math.subtract( xyz[...,1,0], xyz[...,0,0])
        DY = tf.math.subtract( xyz[...,1,1], xyz[...,0,1])
        # DX = xyz[1,0] - xyz[0,0]
        # DY = xyz[1,1] - xyz[0,1]

        dv  = tf.concat([[-DX/L - (U[...,2]-U[...,0])/L], 
                        [ DY/L + (U[...,1]-U[...,3])/L],  
                        [ DX/L + (U[...,2]-U[...,0])/L], 
                        [-DY/L - (U[...,1]-U[...,3])/L]],axis=0)
        print('dv',dv)
        return dU@dv

    # @tf.function
    def q_layer(self,ElemData,e):
        E,A,L,s0 = ElemData[...,0],ElemData[...,1],ElemData[...,2],ElemData[...,3]
        
        # E,A,L,s0 = ElemData
        print(E,A,L,s0)
        return E*A*e + s0

    def pr_layer(self,q,dv):
        return tf.math.multiply(dv,q)

class TensorTruss2(TensorElement,tf.keras.layers.Layer):
    nv : int = 1
    nn : int = 2
    nq : int = 1
    ndm: int = 2
    ndf: int = 2 # number of dofs at each node
    force_dict = {'1':0.0}
    Qpl = tf.zeros((2,nq))

    def __init__(self, tag, iNode, jNode, E=None, A=None, properties=None,geom='lin'):
        super().__init__(self.ndf, self.ndm, self.force_dict, [iNode, jNode])
        
        if isinstance(properties,dict):
            E, A = properties['E'], properties['A']

        self.type = '2D truss'
        self.tag = tag
        self.geom = geom
        self.E = tf.constant( E , dtype=dtype)
        self.A = tf.constant( A , dtype=dtype)
        self.Q = tf.constant(0.0, dtype=dtype)

        self.q = {'1':tf.constant(0.0,dtype=dtype)} # 
        self.v = {'1':tf.constant(0.0,dtype=dtype)} # 

        self.s0 = tf.constant(0.0, dtype=dtype) 
        self.e0 = tf.constant(0.0, dtype=dtype) 
        
        self.w = {'1': 0.0}

    def __repr__(self):
        return 'truss-{}'.format(self.tag)
    
    # @property
    # def dofs(self):
    #     return tf.Variable([node.dofs[0:self.ndf] for node in self.nodes],dtype=int_dtype)

    def v0_vector(self):
        EA = self.E*self.A
        L  = self.L
        e0 = self.e0
        q0 = self.q0
        w  = self.w
        v0 =  np.array([e0['1']*L])
        v0 += [q0['1'] * L/EA]
        v0 += [ w['1'] * L*L/(2*EA)]
        return v0
        
    def q0_vector(self):
        EA = self.E*self.A
        e0 = self.e0['1']
        q0 = self.q0['1']
        q0 = q0 - EA*e0
        return [q0]
    
    def ke_matrix(self,U):
        ag = self.ag(U)
        k = self.k_matrix( U )
        return ag.T@( k*ag )

    def k_matrix(self, U ): 
        """return element local stiffness Matrix"""

        E = self.E               
        A = self.A               
        L = self.L               
        k = tf.math.multiply(E,A)
        k = tf.math.divide(k,L)
        return k

    def ag(self,u): 
        cs = self.cs
        sn = self.sn
        ag = np.array([[-cs],[-sn] , [cs], [sn]]).T
        return ag

    # @tf.function
    def v_layer(self,xyz,U):
        L = tf.linalg.norm(xyz[1,:]-xyz[0,:])
        DX = xyz[1,0] - xyz[0,0]
        DY = xyz[1,1] - xyz[0,1]

        v = (U[2]-U[0])*DX/L+ (U[1]-U[3])*DY/L + ((U[2]-U[0])**2 + (U[1]-U[3])**2)/(2*L)
        return v

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1,        ], dtype=dtype),
        tf.TensorSpec(shape=[ 2, 2], dtype=dtype),
        tf.TensorSpec(shape=[ 4   ], dtype=dtype)])
    def e_layer(self,xyz,U):
        print(U)
        L = tf.linalg.norm(xyz[1,:]-xyz[0,:])
        print('xyz:',xyz[0,0])
        DX = tf.math.subtract(xyz[1,0], xyz[0,0])
        DY = tf.math.subtract(xyz[1,1], xyz[0,1])
        print("dx:",DX)

        v = (U[2]-U[0])*DX/L+ (U[1]-U[3])*DY/L + ((U[2]-U[0])**2 + (U[1]-U[3])**2)/(2*L)
        return v/L

    # @tf.function(input_signature=[
    #     tf.TensorSpec(shape=[1,        ], dtype=dtype),
    #     tf.TensorSpec(shape=[None, 2, 2], dtype=dtype),
    #     tf.TensorSpec(shape=[None, 4   ], dtype=dtype),
    #     tf.TensorSpec(shape=[None, 4, 4], dtype=dtype)])
    def dv_layer(self,xyz,U,dU):
        # U = tf.gather(U,dofs)
        L = tf.linalg.norm(xyz[1,:] - xyz[0,:])
        DX = tf.math.subtract( xyz[1,0], xyz[0,0])
        DY = tf.math.subtract( xyz[1,1], xyz[0,1])
        dv = tf.stack([[-DX/L - (U[2]-U[0])/L], 
                       [ DY/L + (U[1]-U[3])/L],  
                       [ DX/L + (U[2]-U[0])/L], 
                       [-DY/L - (U[1]-U[3])/L]])
        return dU@dv

    # @tf.function
    def q_layer(self,ElemData,e):
        E,A,L,s0 = ElemData[0],ElemData[1],ElemData[2],ElemData[3]
        # E,A,L,s0 = ElemData
        return E*A*e + s0

    # @tf.function
    def pr_layer(self,q,dv):
        return tf.math.multiply(q,dv)

class TensorBeam(TensorElement): 
    """linear 2D Euler-Bernouli frame element"""
    
    nv = 3
    nn = 2
    nq = 3
    ndm = 2
    ndf = 3
    force_dict = {'1':0, '2':0, '3': 0}
    Qpl = np.zeros((2,nq))
    
    def __init__(self, tag, iNode, jNode, E=None, A=None, I=None, properties=None):
        super().__init__(self.ndf, self.ndm, self.force_dict, [iNode,jNode])
        if isinstance(properties,dict):
            E, A, I = properties['E'], properties['A'], properties['I']

        self.type = '2D beam'
        self.tag = tag
        self.E = tf.constant(E,dtype=dtype)
        self.A = tf.constant(A,dtype=dtype)
        self.I = tf.constant(I,dtype=dtype)
        self.q = {'1':tf.constant(0,dtype=dtype), '2':tf.constant(0,dtype=dtype), '3': tf.constant(0,dtype=dtype)}     # basic element force
        self.v = {'1':tf.constant(0,dtype=dtype), '2':tf.constant(0,dtype=dtype), '3': tf.constant(0,dtype=dtype)}
        self.w = {'x':tf.constant(0,dtype=dtype), 'y':tf.constant(0,dtype=dtype)}             #':  "uniform element loads in x and y",

    def __repr__(self):
        return 'el-{}'.format(self.tag) 

    def Elastic_curve(self, x, end_rotations, scale=10, global_coord=False):
        n = len(x)
        L = self.L
        N1 = 1-3*(x/L)**2+2*(x/L)**3
        N2 = L*(x/L-2*(x/L)**2+(x/L)**3)
        N3 = 3*(x/L)**2-2*(x/L)**3
        N4 = L*((x/L)**3-(x/L)**2)
        vi = end_rotations[0]
        vj = end_rotations[1]
        y = np.array(vi*N2+vj*N4)*scale
        xy = np.concatenate(([x],[y]))
        if global_coord:
            x0 = self.nodes[0].x
            y0 = self.nodes[0].y
            Rz = self.Rz_matrix()[0:2,0:2]
            xy = Rz@xy + [[x0]*n,[y0]*n]
        return xy

    @property
    def enq(self): 
        """element number of forces, considering deprecation"""
        return [1 for x in self.q]

    @property
    def dofs(self):
        return tf.Variable([node.dofs[0:self.ndf] for node in self.nodes],dtype=int_dtype)

    def ag(self):
        cs = self.cs
        sn = self.sn
        L = self.L
        ag = np.array([[ -cs ,  -sn , 0,  cs ,   sn , 0],
                       [-sn/L,  cs/L, 1, sn/L, -cs/L, 0],
                       [-sn/L,  cs/L, 0, sn/L, -cs/L, 1]])
        
        if self.dofs[0] == self.dofs[3] or self.dofs[1] == self.dofs[4]:
            ag[0,:] = [0.0]*ag.shape[1]
        return ag

    def ah(self):
        MR = [1 if x else 0 for x in self.rel.values()]
        ah = np.array([[1-MR[0],          0          ,             0        ],
                       [   0   ,       1-MR[1]       ,  -0.5*(1-MR[2])*MR[1]],
                       [   0   , -0.5*(1-MR[1])*MR[2],          1-MR[2]     ]])
        return ah

    def k_matrix(self): 
        """return element local stiffness Matrix"""
        E = self.E
        A = self.A
        I = self.I
        L = self.L
        EI = E*I
        ah = self.ah()
        k = np.array([[E*A/L,    0   ,   0   ],
                      [  0  , 4*EI/L , 2*EI/L],
                      [  0  , 2*EI/L , 4*EI/L]])
        k = ah.T @ k @ ah

        # Assemble matrix metadata
        k = Structural_Matrix(k)
        k.tag = 'k'
        k.row_data = ['q_'+ key for key in self.q0.keys()]
        k.column_data = ['v_' + key for key in self.v0.keys()]
        k.c_cidx = k.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return k

    def f_matrix(self, Roption=False):
        """Flexibility matrix of an element.

        """
        EA = self.E*self.A
        EI = self.E*self.I
        L = self.L
        f = Structural_Matrix([
            [L/EA,     0    ,      0   ],
            [  0 ,  L/(3*EI), -L/(6*EI)],
            [  0 , -L/(6*EI),  L/(3*EI)]])
        ide = set(range(3))

        if Roption:  
            if self.rel["2"]:
                f[:,1] = [0.0]*f.shape[0]
                f[1,:] = [0.0]*f.shape[1]

            if self.rel["3"]:
                f[:,2] = [0.0]*f.shape[0]
                f[2,:] = [0.0]*f.shape[1]

        # Define matrix metadata
        f.tag = 'f'
        f.column_data = ['q_'+ key for key in self.q0.keys()]
        f.row_data = ['v_' + key for key in self.v0.keys()]
        f.c_cidx = f.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return f

    def ke_matrix(self): 
        """return element global stiffness Matrix"""

        k  = self.k_matrix()
        ag = self.ag()
        
        ke = ag.T @ k @ ag
        ke = Structural_Matrix(ke) 
        ke.row_data = ke.column_data = ['u_'+str(int(dof)) for dof in self.dofs]
        return ke

    def bg_matrix(self, Roption=False): 
        """return element global static matrix, bg"""

        cs = self.cs
        sn = self.sn
        L  = self.L
        #                x      ri      rj   Global
        bg = np.array([[-cs, -sn/L,  -sn/L],  # x
                       [-sn,  cs/L,   cs/L],  # y
                       [0.0,   1.0,    0.0],  # rz
                       [ cs,  sn/L,   sn/L],
                       [ sn, -cs/L,  -cs/L],
                       [0.0,   0.0,    1.0]])
        if Roption:
            if self.rel['2']:
                bg[:,1] = [0.0]*bg.shape[0]

            if self.rel['3']:
                bg[:,2] = [0.0]*bg.shape[0]

        if self.dofs[0] == self.dofs[3] or self.dofs[1] == self.dofs[4]:
            bg[:,0] = [0.0]*bg.shape[0]
            # bg = np.delete(bg, 0, axis=1)

        bg = Structural_Matrix(bg)
        bg.tag = 'b'
        bg.column_data = ['x', 'ri', 'rj']
        bg.row_data = ['x', 'y', 'rz']
        bg.c_cidx = bg.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return bg
    
    def m_matrix(self):
        m = np.array([
            [140, 0.0, 0.0,  70.0, 0.0, 0.0],
            [0.0, ],
            [0.0, ],
            [70., ],
            [0.0, ],
            [0.0, ],
        ])
        return m

    def v0_vector(self):
        EA = self.E*self.A
        EI = self.E*self.I
        L = self.L
        e0 = self.e0
        w = self.w
        v0 =  np.array([e0['1']*L, -e0['2']*L/2, e0['2']*L/2])
        v0 += np.array([w['x']*L*L/(2*EA), w['y']*L**3/(24*EI), -w['y']*L**3/(24*EI)])
        v0 = Structural_Matrix(v0)
        v0.tag = 'v'
        v0.column_data = ['v_0']
        v0.row_data = ['axial', 'i-rotation', 'j-rotation']
        v0.c_cidx = False
        v0.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]
        return v0 

    def q0_vector(self):
        L = self.L
        E = self.E 
        A = self.A 
        I = self.I
        e0= self.e0 
        w = self.w
        q0 =  np.array([-e0['1']*E*A, +e0['2']*E*I, -e0['2']*E*I])
        if self.rel['2']:
            q0[1] *= 0
            q0[1] *= 1.5 
        if self.rel['3']:
            q0[2] *= 0
            q0[2] *= 1.5 

        q0[1] += -w['y']*L**2/12*((not self.rel['2']) and (not self.rel['3'])) - w['y']*L**2/8*(self.rel['3'])
        q0[2] += +w['y']*L**2/12*((not self.rel['2']) and (not self.rel['3'])) + w['y']*L**2/8*(self.rel['2'])

        # Metadata
        q0 = Structural_Matrix(q0)
        q0.tag = 'q'
        q0.column_data = ['q_0']
        q0.row_data = ['axial', 'M_i', 'M_j']
        q0.c_cidx = False
        q0.c_ridx = [int(key)-1 for key in self.rel.keys() if self.rel[key]]

        return q0 

    def κ(self, k, state=None, form=None):
        """Defines element curvature and calculates the resulting end deformations.

        """
        if form ==None: form = 'uniform'
        if state==None: state = -1
        L = self.L
        table = {
            'uniform': [-0.5*k*L, 0.5*k*L],
            'ilinear': [-k/3*L, k/6*L],
            'jlinear': [-k/6*L, k/2*L],
            'parabolic': [-k/3*L, k/3*L],
        }
        self.nodes[0].model.states[state]['v0'][self.tag]['2'] = table[form][0]
        self.nodes[0].model.states[state]['v0'][self.tag]["3"]= table[form][1]
        return

    def k_layer(self,):
        pass


class TensorIsoQuad(TensorElement):    
    def __init__(self, tag, iNode, jNode, E=None, A=None, properties=None,geom='lin'):
        super().__init__(self.ndf, self.ndm, self.force_dict, [iNode, jNode])
        
        if isinstance(properties,dict):
            E, A = properties['E'], properties['A']


##>*****************************************************************************
