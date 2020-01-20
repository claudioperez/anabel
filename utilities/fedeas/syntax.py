import numpy as np
import scipy.linalg
from ema.matvecs import Static_matrix, Kinematic_matrix, Flexibility_matrix, Stiffness_matrix
from ema.matvecs import Deformation_vector, Displacement_vector
from ema.utilities import Structural_Matrix

from ema.analysis2 import Pw_vector, P_vector, P0_vector
# New matrix objects:
from ema.matvecs import nStatic_matrix, iForce_vector, nDeformation_vector, nDisplacement_vector, Diag_matrix


# def Localize(U_vector, P_vector, model=None):
#     if model is None: model = U_vector.model
#     A =  A_matrix(model)
#     Q0 = Q0_vector(model)
#     Ks = Ks_matrix(model)

#     V = A.f @ U_vector.f
#     Q  = Ks@V + Q0
#     return V, Q

# def B_matrix(model, matrix=None, rng=None):
#     """Returns a Static_matrix object"""
#     return Static_matrix(model, matrix, rng)

# def A_matrix(Domain,matrix=None):
#     """Returns a Kinematic_matrix object"""
#     return Kinematic_matrix(Domain,matrix)

# def F_matrix(Domain):
#     """Returns a Flexibility_matrix object"""    
#     return Flexibility_matrix(Domain, Roption=None)

# def K_matrix(Model):
#     """Returns a Stiffness_matrix object"""
#     return Stiffness_matrix(Model, Roption=None)

# def V_vector(A_matrix, vector=None):
#     """Returns a Deformation_vector object"""    
#     return Deformation_vector(A_matrix, vector)

# def V0_vector(model):
#     """Returns a Deformation_vector object"""   
#     arry = np.concatenate([elem.v0_vector() for elem in model.elems]) 
#     row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.v.keys()] 
#     return nDeformation_vector(arry, model, row_data)

# def U_vector(model, vector=None):
#     """Returns a Displacement_vector object"""   
#     U = np.zeros(model.nt)
#     row_data = [str(dof) for dof in range(1,model.nt+1)]
#     U = nDisplacement_vector(U, model, row_data)
#     if vector is not None:
#             for key in vector.row_data:
#                 if key in U.row_data:
#                     U.set_item(key, vector.rows([key]))
#     return U

# def Q_vector(model, vector=None):
#     """Returns a Deformation_vector object"""   
    
#     arry = np.zeros((sum(model.nq),1))
#     row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.rel.keys()]
#     Q = iForce_vector(arry, model, row_data)
#     if vector is not None:
#             for key in vector.row_data:
#                 if key in Q.row_data:
#                     Q.set_item(key, vector.rows([key]))
#     return Q

# def Q0_vector(model):
#     """Returns a vector of initial element forces"""   
#     arry = np.concatenate([elem.q0_vector() for elem in model.elems])
#     row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.q.keys()] 
#     return iForce_vector(arry, model, row_data)

# def Qpl_vector(model):
#     """Returns a vector of element plastic capacities""" 
#     B = B_matrix(model)
#     Bf = B.f
#     Qp_pos = [elem.Qp['+'][key] for elem in model.elems for key in elem.Qp['+']]
#     Qp_neg = [elem.Qp['-'][key] for elem in model.elems for key in elem.Qp['-']]
#     row_data = [elem.tag+'_'+key for elem in model.elems for key in elem.Qp['-']]

#     del_idx = np.where(~Bf.any(axis=0))[0]
#     Qp_pos = np.delete(Qp_pos, del_idx)
#     Qp_neg = np.delete(Qp_neg, del_idx)
#     row_data = np.delete(row_data, del_idx)
#     column_data = ['Q_{pl}^+', 'Q_{pl}^-']

#     Qpl = nStatic_matrix(np.array([Qp_pos, Qp_neg]).T, model, (row_data, column_data))
#     return Qpl


# def Fs_matrix(model):
#     """Returns a Flexibility_matrix object"""  
#     f  = np.array([elem.f_matrix() for elem in model.elems])
#     Fs = scipy.linalg.block_diag(*f) 
#     Fs = Diag_matrix(Fs, model)
#     Fs.basic_forces = np.array([q for elem in model.elems for q in elem.basic_forces]) 
#     return Fs

# def Ks_matrix(model):
#     """Returns a Flexibility_matrix object"""  
#     k  = np.array([elem.k_matrix() for elem in model.elems])
#     Ks = scipy.linalg.block_diag(*k)  
#     Ks = Diag_matrix(Ks, model)
#     Ks.basic_forces = np.array([q for elem in model.elems for q in elem.basic_forces])
#     return Ks

