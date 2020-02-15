"""
Problem 25
==========

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.xsection('default', 1e8, 40000)
xt = mdl.xsection('truss', 10e3, 1)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  8.0, 0.0)
mdl.node('3',  8.0, 6.0)
mdl.node('4', 16.0, 6.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])
mdl.truss('d', n['2'], n['4'], xsec=xt)

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['4'], ['y'])

mdl.numDOF()

fig, ax = plt.subplots()
em.plot_structure(mdl, ax)

e['a'].e0['2'] = 5e-3
e['b'].e0['2'] = 5e-3
e['c'].e0['2'] = 5e-3
e['d'].q0['1'] = 30
mdl.redundant(e['a'], '2')
mdl.redundant(e['c'], '2')
em.utilities.export.FEDEAS(mdl)

B = em.B_matrix(mdl)
P = em.P_vector(B)
ai = 263.4356
ci = 272.8436
Qx = np.array([ai, ci]).view(em.P_vector)

B.barx@Qx


######################################################################
# Solution
# --------
# 

Uf = em.analysis.SolveDispl(mdl)
Uf

A = em.A_matrix(mdl)
Ks = em.K_matrix(mdl).s
Q0 = em.Q_vector(B).o
V = A.f@Uf
Q = Ks@V+Q0
Q

