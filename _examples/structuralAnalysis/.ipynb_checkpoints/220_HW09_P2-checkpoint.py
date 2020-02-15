"""
Hyperstatic Truss - Displacement method
=======================================

(220_HW09_P2)

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
# %config InlineBackend.figure_format = 'svg'

dm = em.Model(2,2)
n = dm.dnodes
e = dm.delems

A1 = 10000
Ac = 20000
I = 1
dm.xsection('default', A1, I)
csec = dm.xsection('section-c', Ac, I)

dm.node('1', 0.0, 0.0)
dm.node('2', 8.0, 0.0)
dm.node('3', 16., 0.0)
dm.node('4', 0.0, 6.0)
dm.node('5', 8.0, 6.0)

dm.truss('a', n['4'], n['5'])
dm.truss('b', n['1'], n['5'])
dm.truss('c', n['2'], n['5'], xsec=csec)
dm.truss('d', n['3'], n['5'], xsec=csec)

dm.pin(n['1'])
dm.pin(n['4'])
dm.pin(n['2'])
dm.pin(n['3'])

dm.numDOF()

# em.utilities.export.FEDEAS(dm)

fig, ax = plt.subplots()
em.plot_structure(dm, ax)

# identify redundants
dm.redundant(e['b'], '1')
dm.redundant(e['d'], '1')

B = em.B_matrix(dm)
K = em.K_matrix(dm)
P = em.P_vector(B)


######################################################################
# Part A: Nodal loading.
# ----------------------
# 


######################################################################
# Displacements
# ~~~~~~~~~~~~~
# 

K.f

P.set_item('1', 50)
P.set_item('2', 30)

Uf = K.f.inv@P.f
Uf


######################################################################
# Element basic forces
# ~~~~~~~~~~~~~~~~~~~~
# 

B.f.T

V = B.f.T@Uf
V

Q = K.s@V
Q


######################################################################
# Part B: Thermal loading
# -----------------------
# 


######################################################################
# Displacements
# ~~~~~~~~~~~~~
# 

e['c'].e0['1'] = 100*2e-5

A = em.A_matrix(dm)
V0 = em.V_vector(A).o
Q0 = -K.s@V0
Q0

P0f = B.f@Q0
P0f

Uf = K.f.inv@(-P0f)
Uf_func = em.analysis.SolveDispl(dm)
Uf

Uf_func


######################################################################
# Element Basic Forces
# ~~~~~~~~~~~~~~~~~~~~
# 

V = A.f@Uf
V

Q = K.s@V + Q0
Q