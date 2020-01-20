"""
Hyperstatic Truss - Compatibility
=================================

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg'

mdl = em.Model(2,2)
n = mdl.dnodes
e = mdl.delems

mdl.node('1', 0.0, 0.0)
mdl.node('2', 8.0, 0.0)
mdl.node('3', 4.0, 3.0)
mdl.node('4', 4.0, 6.0)

mdl.truss('a', n['1'], n['3'])
mdl.truss('b', n['2'], n['3'])
mdl.truss('c', n['1'], n['4'])
mdl.truss('d', n['3'], n['4'])
mdl.truss('e', n['2'], n['4'])

mdl.fix(n['1'], ['x', 'y'])
mdl.fix(n['2'], ['x', 'y'])

mdl.numDOF()

fig1, ax1 = plt.subplots(1,1)
em.plot_structure(mdl, ax1)


######################################################################
# Part 1
# ------
# 
# Static-Kinematic Matrix Equivalence
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# .. math:: V = A_f U_f
# 

A = em.A_matrix(mdl)
B = em.B_matrix(mdl)


######################################################################
# .. math:: P_f = B_f Q
# 

B.f

B.f.T - A.f


######################################################################
# Part 2
# ------
# 


######################################################################
# Member d length
# ~~~~~~~~~~~~~~~
# 
# The kinematic matrix, :math:`A_f`, is given below:
# 

A.f


######################################################################
# And the corresponding deformation vector is:
# 

V = em.V_vector(A)
V.set_item('b_1', 0.1)
V.set_item('c_1', 0.2)
V


######################################################################
# The free dof displacement vector, :math:`U_f`, is then computed as
# follows:
# 
# .. math::
# 
# 
#    U_f = A_f^{-1}V_\epsilon
# 

Ve = V[[0,1,2,4]]
Ae = A.f[[0,1,2,4],:]
U = Ae.inv@Ve
U.disp


######################################################################
# Finally the fully deformation vector is computed from :math:`V=A_fU_f`,
# which gives the necessary deformation of element d. 
# 

Veh = A.f@U
Veh.disp


######################################################################
# Element d must therefore elongated by **0.037**.
# 


######################################################################
# Satisfy Compatibility
# ~~~~~~~~~~~~~~~~~~~~~
# 


######################################################################
# The matrix :math:`\bar{B}_x` is computed as follows:
# 

mdl.redundant(e['d'], '1')
B.barx


######################################################################
# This is multiplied by the deformation vector as follows:
# 

residual = B.barx.T@Veh
print(residual)
if residual < 10e-9:
    print("Compatibility is satisfied")


B.f.ker/-0.56694671

B.ker/-0.56694671

