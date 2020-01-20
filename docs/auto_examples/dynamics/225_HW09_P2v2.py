"""
Modal analysis - 1r
===================

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg'

mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems
w = 24
h = 12

mdl.frame((1,w), (3,h))

for node in n.values():
    mdl.fix(node, 'rz')
    
n['4'].mass = 1
n['6'].mass = 1
n['8'].mass = 0.5
mdl.fix(n['1'], ['x', 'y'])
mdl.fix(n['2'], ['x', 'y'])

mdl.DOF = [[10, 11, 12], [13, 14, 15], 
           [ 1, 11,  4], [ 1, 14,  5], 
           [ 2, 11,  6], [ 2, 14,  7], 
           [ 3, 11,  8], [ 3, 14,  9]]


######################################################################
# Part a)
# -------
# 

K = em.K_matrix(mdl)
M = em.Mass_matrix(mdl)
K.f*h**3

M.f


######################################################################
# Part b)
# -------
# 

shapes = em.analysis.ModalAnalysis(mdl)

shapes

U = em.U_vector(em.A_matrix(mdl))

colors = ['b', 'y','pink']
fig, ax = plt.subplots(1,3)
for i, shape in enumerate(shapes[1].T):
    U[0:3] = shape
    em.plot_modes(mdl, U, ax[i], color=colors[i], label=str(i+1))
plt.legend();


######################################################################
# Part c) Verify orthogonality
# ----------------------------
# 

phis = np.around(shapes[1], 7)
phis

print(phis[:,0] @ M.f @ phis[:,1])

print(phis[:,0] @ M.f @ phis[:,2])

print(phis[:,1] @ M.f @ phis[:,2])


######################################################################
# Part d) Normalize modes.
# ------------------------
# 


######################################################################
# Each mode shape is normalized so that when multiplied by the problem
# variable, :math:`\frac{1}{\sqrt{m}}`, they will generate a modal mass
# matrix, :math:`M_n`, equal to the identity matrix.
# 

phi_n = np.zeros((3,3))
phi_n[:,0] = phis[:,0]/((phis[:,0] @ M.f @ phis[:,0]))**(1/2)
print(phi_n[:,0])
phi_n[:,0].T @ M.f @ phi_n[:,0]

phi_n[:,1] = phis[:,1]/((phis[:,1] @ M.f @ phis[:,1]))**(1/2)
print(phi_n[:,1])
phi_n[:,1].T @ M.f @ phi_n[:,1]

phi_n[:,2] = phis[:,2]/((phis[:,2] @ M.f @ phis[:,2]))**(1/2)
print(phi_n[:,2])
phi_n[:,2].T @ M.f @ phi_n[:,2]

Id = np.around(phi_n.T @ M.f @ phi_n,7)
Id