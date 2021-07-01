"""The analysis module provides basic analysis functions that 
operate on `emme.Model` objects, or arrays represention various
structural matrices.
"""


from emme.matrices import *
# from emme.utilities.fedeas.syntax import *
import numpy as np  

import scipy.linalg
from scipy import optimize
from scipy.integrate import solve_ivp



class EqResponse:
    def __init__(self,ground_motion):
        self.ground_motion = ground_motion
    
    def RHA(self, period, damping=1.0, R=1.0):
        g=386.4
        ground_motion = elcentro
        time = ground_motion[:,0]
        t_span = (time[0], time[-1])
        u0 = (0.0, 0.0)

        def EOM(t, y):
            i = np.argmin(np.abs(time - t))
            D, dDdt = y
            dydt = [dDdt, -ground_motion[i,1]*g-omega**2*D-2*zeta*omega*dDdt]
            return dydt

        rtol = atol = 1e-8
        sol = solve_ivp(EOM, y0=u0, t_span=t_span, rtol=rtol, atol=atol)
        D = sol.y[0,:]
        t = sol.t
        return t, D


def ElcentroRHA(zeta, omega):
    g=386.4
    ground_motion = elcentro
    time = ground_motion[:,0]
    t_span = (time[0], time[-1])
    u0 = (0.0, 0.0)

    def EOM(t, y):
        i = np.argmin(np.abs(time - t))
        D, dDdt = y
        dydt = [dDdt, -ground_motion[i,1]*g-omega**2*D-2*zeta*omega*dDdt]
        return dydt

    rtol=atol=1e-8
    sol = solve_ivp(EOM, y0=u0, t_span=t_span, rtol=rtol, atol=atol)
    D = sol.y[0,:]
    t = sol.t
    return t, D


def characterize(model):
    # Bo = B_matrix(model).o.del_zeros()
    Bo = B_matrix(model).o
    r = Bo.rank 
    nc = len(Bo[0])
    nr = len(Bo.T[0])
    m = nr - r
    s = nc - r 
    # print("m = {}".format(m))
    # print("s = {}".format(s))
    return (m, s)
    
def kStaticCondensation(K, dofs=None, idxs=None):
    """Return the static condensation of a :class:`Stiffness_matrix` object based on specified dofs.
    
    Parameters
    --------------

    K_matrix: np.ndarray or emme.Structural_Matrix
        Array object with rows and columns ordered according to dof numbering, with no skipped dofs.
    

    Theory
    --------------

    The Guyan reduction can be expressed as a *change of basis* which produces 
    a low-dimensional representation of the original space, represented by the 
    master degrees of freedom.

    The linear transformation that maps the reduced space onto the full space 
    is expressed as:

    .. math::

       \\begin{Bmatrix}
       \mathbf{d}_m \\
       \mathbf{d}_s
       \end{Bmatrix} =
       \\begin{bmatrix}
         \mathbf{I} \\\
       - \mathbf{K}_{ss}^{-1}\mathbf{K}_{sm}
       \end{bmatrix}
       \\begin{Bmatrix}
       \mathbf{d}_m
       \end{Bmatrix}
       =
       \\begin{Bmatrix}
       \mathbf{T}_G
       \end{Bmatrix}
       \\begin{Bmatrix}
       \mathbf{d}_m
       \end{Bmatrix}
    
    where :math:`\mathbf{T}_G` represents the Guyan reduction [[transformation matrix]].
    Thus, the reduced problem is represented as:

    .. math::

       \mathbf{K}_G\mathbf{d}_m = \mathbf{f}_m


    In the above equation, :math:`\mathbf{K}_G` represents the reduced system of linear 
    equations that's obtained by applying the Guyan reduction transformation on the full 
    system, which is expressed as:

    .. math::

       \mathbf{K}_G = \mathbf{T}_G^T \mathbf{K} \mathbf{T}_G


    GUYAN, J., Reduction of stiffness and mass matrices, R. , 
    AIAA Journal 3 380--380 (1965) https://doi.org/10.2514/3.2874

    """

    K0 = K
    K = np.zeros(np.shape(K0))
    num_dofs = len(K0.T)
    dof_index = np.array(range(num_dofs))

    if idxs is None:
        idx_tt = [df-1 for df in dofs]
    else:
        idx_tt = idxs
    
    SlaveDofs = np.setdiff1d(dof_index, idx_tt)
    SlaveDofs = np.sort(SlaveDofs)
    # index = np.delete(index, SlaveDofs)
    n00 = len(SlaveDofs)

    ntt = len(idx_tt)

    Kss = np.zeros((n00, n00))
    for ii in range(n00):
        for ij in range(n00):
            K[ntt+ii, ntt+ij] = K0[SlaveDofs[ii], SlaveDofs[ij]]
            Kss[ii,ij] = K0[SlaveDofs[ii], SlaveDofs[ij]]

    Ksm = np.zeros((n00, ntt))
    for ii in range(n00):
        for ij in range(ntt):
            K[ntt+ii, ij] = K[ij, ntt+ii] = K0[SlaveDofs[ii], idx_tt[ij]]
            Ksm[ii, ij] =  K0[SlaveDofs[ii], idx_tt[ij]]
    
    for ii in range(ntt):
        for ij in range(ntt):
            K[ii, ij] = K0[idx_tt[ii], idx_tt[ij]]
    
    # print('Kss: \n',np.around(Kss,3), '\nKsm: \n',np.around(Ksm,3))
    # print("\nK: \n", np.around(K,3))

    P = (-1.*np.linalg.inv(Kss)) @ Ksm
    
    T = np.concatenate((np.eye(ntt),P), axis=0)
    assert np.allclose(K, K.T)
    Ktt = (T.T@(K@T)).view(Stiffness_matrix)



    # Metadata
    if isinstance(K_matrix, Structural_Matrix):
        Ktt.model = K_matrix.model
    row_data = column_data = [str(dof+1) for dof in idx_tt]
    Ktt.row_data    = row_data   
    Ktt.column_data = column_data

    return Ktt


def StaticCondensation(K_matrix, M_matrix=None, dofs=None, idxs=None):
    """Return the static condensation of a K_matrix object based on 
    a `Mass_matrix` object, or  specified dofs. 

    Parameters
    --------------
    K_matrix: arry
        Array object with rows and columns ordered according to dof numbering, with no skipped dofs.
    
    """

    if (dofs is None and idxs is None) and (Mass_matrix is not None):
        # This method may require emme.Stiffness_matrix and emme.Mass_matrix objects as 
        # opposed to general arrays.
        Mf = M_matrix.f
        Kf = K_matrix.f
        nf = len(Kf[0])
        free_dofs = [str(dof+1) for dof in range(nf)]
        cols = [idx for idx in range(len(Kf))]
        cols_00 = np.where(~Mf.any(axis=0))[0] # return indices of all-zero columns in mass matrix
        cols_tt = np.setdiff1d(cols, cols_00)
        ntt = cols_tt[-1]
        if len(cols_tt)-1 == cols_tt[-1]:
            ntt = cols_tt[-1] + 1
            nf = len(Kf[0,:])
            Ktt = Kf[0:ntt,0:ntt] - Kf[ntt:nf+1,0:ntt].T @ Kf[ntt:nf+1,ntt:nf+1].inv @ Kf[ntt:nf+1,0:ntt]
            Mtt = Mf[0:ntt, 0:ntt]
        Mtt.model = Mf.model
        Mtt.row_data=Mf.row_data[0:ntt]
        Mtt.column_data=Mf.column_data[0:ntt]
        Ktt.model = Kf.model
        Ktt.row_data=Kf.row_data[0:ntt]
        Ktt.column_data = Kf.column_data[0:ntt]
    else: 
        K = K_matrix
        M = M_matrix

        num_dofs = len(K.T)
        dof_index = np.array(range(num_dofs))

        if idxs is None:
            idx_tt = [df-1 for df in dofs]
        else:
            idx_tt = idxs
        
        SlaveDofs = np.setdiff1d(dof_index, idx_tt)
        SlaveDofs = np.sort(SlaveDofs)
        # index(SlaveDofs) = [];

        print("slave dof indices: {}".format(SlaveDofs))
        print("kept dof indices: {}".format(idx_tt))

        # index = np.delete(index, SlaveDofs)
        n00 = len(SlaveDofs)
        mx_00 = max(SlaveDofs)+1

        ntt = len(idx_tt)
        mx_tt = max(idx_tt)+1

        Kss = np.zeros((n00, n00))
        for ii in range(n00):
            for ij in range(n00):
                Kss[ii,ij] = K[SlaveDofs[ii],SlaveDofs[ij]]

        Ksm = np.zeros((n00, ntt))
        for ii in range(n00):
            for ij in range(ntt):
                Ksm[ii,ij] =  K[SlaveDofs[ii],idx_tt[ij]]

        P = - np.linalg.inv(Kss) @ Ksm
        
        T = np.concatenate((np.eye(ntt),P), axis=0)

        Mtt = (T.T@M@T).view(Mass_matrix)
        Ktt = (T.T@K@T).view(Stiffness_matrix)

        # Metadata
        if isinstance(M_matrix, Structural_Matrix) and isinstance(K_matrix, Structural_Matrix):
            Mtt.model = M_matrix.model
            Ktt.model = K_matrix.model
        row_data = column_data = [str(dof) for dof in idx_tt]
        Mtt.row_data    = row_data    
        Mtt.column_data = column_data 
        Ktt.row_data    = row_data   
        Ktt.column_data = column_data
    return Ktt, Mtt


def ModalAnalysis(Model, dofs=None, norm='mass'):
    """Takes a model object and returns its natural frequencies and modes.
    
    Parameters
    ----------------
    Model: emme.Model

    
    Returns
    ----------------
    w: array_like
        array of eigenvalues
    phi: array_like
        array of eigenmodes
        
    """

    # Shapes = np.array([])
    K = K_matrix(Model)
    M = Mass_matrix(Model)
    Ktt, Mtt = StaticCondensation(K,M)
    freq, shapes = scipy.linalg.eig(Ktt, Mtt)

    # Normalize
    if norm=='mass':
        for i, shape in enumerate(shapes.T):
            factor = shape.T@Mtt@shape
            shapes[:,i] = shape/np.sqrt(factor)
    if norm=='last':
        for i, shape in enumerate(shapes.T):
            shapes[:,i] = shape/shape[-1]

    return freq, shapes



def SolveDispl(Model):
    """Solve elements in a Model object for free DOF displacements
    
    Parameters
    -------------
    Model: emme.Model
        Model object with predefined geometry and element/nodal loads."""

    # Create model static matrix
    # B = B_matrix(Model)

    # Create model kinematic matrix
    # A = A_matrix(Model)

    # Create nodal force vector
    # P = P_vector(B)
    P = P_vector(Model)
    # Q0 = Q0_vector(Model)

    # Create initial element force vector
    # P0f = B.f @ Q0
    P0f = P0_vector(Model).f

    Pw = Pw_vector(Model)

    # Form structure stiffness matrix
    K = K_matrix(Model)

    # Find global free displacements
    Uf = K.f.inv@ (P.f - (Pw.f + P0f))
    Uf.model = Model
    # U = U_vector(Model, Uf)
    return Uf

def MemberDeformations(Model, U_vector):
    A = A_matrix(Model)
    return 0

def setupPlasticAnalysis_wLBT(model):
    """This function sets up a lower bound plastic hinge problem 
    as a linear programming problem to minimize a linear 
    objective function subject to linear equality and inequality 
    constraints.

    Parameters
    -------------
    model: emme.Model
        Model object with predefined element plastic capacities.
    
    Returns
    --------------
    c, A_ub, b_ub, A_eq, b_eq, bounds, Q_row_data
    
    Analysis Assumptions
    ---------------

    - resistance capacities are constant, no interaction
    - linear-elastic material response
    - Linear structural kinematics (small displacements)
    - Linear element kinematics (small deformations)
    - (INCOMPLETE)


    Theory
    ----------------
    Linear programming solves problems of the following form:

    .. math::
         
       \min_x \ & c^T x \\

       \mbox{such that} \ & A_{ub} x \leq b_{ub}, \\
       & A_{eq} x = b_{eq}, \\
       & l \leq x \leq u ,
        

    where
    :math:`x` is a vector of decision variables; 
    :math:`c`, :math:`b_{ub}`, :math:`b_{eq}`, :math:`l`, and :math:`u` are vectors; and
    :math:`A_{ub}` and :math:`A_{eq}` are matrices.

    """

    nf = model.nf
    B = B_matrix(model)
    Bf = B.f
    Bc = Bf[:,model.idx_c]
    # Pf = P_vector(B).f
    Pf = P_vector(model).f
    nq = len(Bc.T)

    # Define c
    q0 = np.zeros(nq)
    c = np.concatenate(([1], q0))
    bounds = [(None, None)]*(nq+1)
    bounds[0] = (None, None)

    # Define A_eq
    A_eq = np.zeros((nf, 1+nq))
    A_eq[:,0] = Pf
    A_eq[:,1:] = -Bc
    A_eq


    # Define b_eq
    Pcf = np.zeros(nf)
    b_eq = -Pcf

    # define b_ub
    
    # Qp_pos = [elem.Qp['+'][key] for elem in model.elems for key in elem.Qp['+']]
    # Qp_neg = [elem.Qp['-'][key] for elem in model.elems for key in elem.Qp['-']]

    # Q_row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.Qp['+']]

    # del_idx = np.where(~Bf.any(axis=0))[0]
    # Qp_pos = np.delete(Qp_pos, del_idx)
    # Qp_neg = np.delete(Qp_neg, del_idx)
    # Q_row_data = np.delete(Q_row_data, del_idx)
    Qpl = Qpl_vector(model)[model.idx_c,:]
    Q_row_data = Qpl.row_data
    # b_ub = np.concatenate((Qp_pos, Qp_neg))
    b_ub = np.concatenate((Qpl[:,0], Qpl[:,1]))

    # define A_ub
    A_ubr = np.concatenate((np.zeros((nq,1)), np.eye(nq)), axis=1)
    A_ub = np.concatenate((A_ubr, -A_ubr), axis=0)

    return (c, A_ub, b_ub, A_eq, b_eq, bounds, Q_row_data)


def PlasticAnalysis_wLBT(model, verbose=False):

    c, A_ub, b_ub, A_eq, b_eq, bounds, Q_row_data = setupPlasticAnalysis_wLBT(model)

    result =  optimize.linprog(c = -c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds = bounds, method='simplex')
    if verbose: print(result)

    lambdac = result['x'][0]
    Q = result['x'][1:]
    Q = iForce_vector(Q, model, Q_row_data)

    return lambdac, Q

    




