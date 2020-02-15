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

mdl = em.Model(2,2)
n = mdl.dnodes
e = mdl.delems

A1 = 10000
Ac = 20000
I = 1
mdl.xsection('default', A1, I)
csec = mdl.xsection('section-c', Ac, I)

mdl.node('1', 0.0, 0.0)
mdl.node('2', 8.0, 0.0)
mdl.node('3', 16., 0.0)
mdl.node('4', 0.0, 6.0)
mdl.node('5', 8.0, 6.0)

mdl.truss('a', n['4'], n['5'])
mdl.truss('b', n['1'], n['5'])
mdl.truss('c', n['2'], n['5'], xsec=csec)
mdl.truss('d', n['3'], n['5'], xsec=csec)

mdl.pin(n['1'])
mdl.pin(n['4'])
mdl.pin(n['2'])
mdl.pin(n['3'])

mdl.numDOF()

# em.utilities.export.FEDEAS(mdl)

fig, ax = plt.subplots()
em.plot_structure(mdl, ax)


######################################################################
# Part A: Nodal loading.
# ----------------------
# 


######################################################################
# Displacements
# ~~~~~~~~~~~~~
# 

n['5'].p['x'] = 50
n['5'].p['y'] = 30

result = em.analysis.SolveDispl(mdl)
V, Q = em.Localize(*result)
Q


np.array([result, em.P_vector(mdl)])


######################################################################
# Part B: Thermal loading
# -----------------------
# 


######################################################################
# Displacements
# ~~~~~~~~~~~~~
# 

n['5'].p['x'] = 0
n['5'].p['y'] = 0
e['c'].e0['1'] = 100*2e-5

P0f = B.f@Q0
P0f

Uf = K.f.inv@(-P0f)
Uf_func = em.analysis.SolveDispl(mdl)
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