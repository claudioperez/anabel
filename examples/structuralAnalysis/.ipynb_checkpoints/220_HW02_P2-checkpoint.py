"""
3D Isostatic Truss
==================

"""

import numpy as np
import sympy as sp
import ema as em
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

model = em.spacetruss(3, 20.0, 10.0, 5.0)
n = model.dnodes
e = model.delems

fig = plt.figure()
ax = fig.add_subplot(1,1,1, projection='3d')
em.plot_structure3d(model, ax);


######################################################################
# Set up the static :math:`B_f` matrix.
# -------------------------------------
# 

for elem in model.elems:
    Dx = elem.nodes[1].x-elem.nodes[0].x
    Dy = elem.nodes[1].y-elem.nodes[0].y
    Dz = elem.nodes[1].z-elem.nodes[0].z
    print("{}: DXYZ = ({:0.2f}, {:0.2f}, {:0.2f}), L = {:0.2f}".format(elem.tag, Dx, Dy, Dz, elem.L))

B = em.B_matrix(model)
B.f


######################################################################
# Determine the basic element forces for the given loading.
# ---------------------------------------------------------
# 

model.nodes[3].p['z'] = -40
model.nodes[4].p['z'] = -40
model.nodes[5].p['z'] = -40

P = em.P_vector(B)
P.f

# Bf = sp.Matrix(B[:9,:9])
Q = B.bari@P.f
Q

sp.Matrix(np.around(np.linalg.inv(B[:9,:9]),3))


######################################################################
# Support reactions, Global equilibrium
# =====================================
# 

R = B.d@Q
R


######################################################################
# Explanation of upper ring zeros.
# ================================
# 
# It can be said that the forces in the upper ring are zero because the
# appied force vector lies entirely in the column space of a linearly
# independent submatrix of :math:`B_f` where the columns pertaining to the
# upper ring element forces are ommitted. Similarly, it can be stated that
# the applied force vector lies within the nullspace of the
# :math:`B_f^{-1}` submatrix which pertains to the top ring elements. This
# means that :math:`P_f` is orthogonal to the row vectors of
# :math:`B_f^{-1}` corresponding to the basic forces in the top ring.
# 