"""
Problem 32
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

mdl.node('1',  0.0, 0.0)
mdl.node('2',  8.0, 0.0)
mdl.node('3',  8.0, 6.0)
mdl.node('4', 16.0, 6.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])
mdl.truss('d', n['2'], n['4'])

mdl.hinge(e['a'], n['2'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['4'], ['y'])

mdl.numDOF()

fig, ax = plt.subplots()
em.plot_structure(mdl, ax)

e['a'].e0['2'] = 1e-3
e['b'].e0['2'] = 1e-3
e['c'].e0['2'] = -1e-3
e['d'].e0['1'] = -4.8e-3/10

V0 = em.V0_vector(mdl)
A = em.A_matrix(mdl)
V0

# A.c.inv@V0.c


######################################################################
# Displacement Method
# -------------------
# 

Uf = em.analysis.SolveDispl(mdl)
Uf

em.plot_U(mdl, Uf, ax, scale=50)