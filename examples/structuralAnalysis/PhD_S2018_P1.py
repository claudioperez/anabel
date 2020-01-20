"""
Ph.D Spring 2018 Problem 1
==========================

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.xsection('default', 1e10, 50000)
xt = mdl.xsection('truss', 50e3, 1)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  6.0, 8.0)
mdl.node('3', 12.0, 8.0)
mdl.node('4', 12.0, 0.0)

mdl.beam('a', n['2'], n['3'])
mdl.beam('b', n['3'], n['4'])
mdl.truss('c', n['1'], n['2'], xsec=xt)

# mdl.hinge(e['a'], n['2'])

mdl.fix(n['1'], ['x','y', 'rz'])
# mdl.fix(n['2'], ['rz'])
mdl.fix(n['4'], ['x','y', 'rz'])

mdl.numDOF()


######################################################################
# The loading is next defined, as follows:
# 

# Define loading
n['2'].p['x']  = -50
e['c'].q0['1'] =  50.183
# em.utilities.export.FEDEAS(mdl)

fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)
U_disp = em.analysis.SolveDispl(mdl) 
em.plot_U(mdl, U_disp, ax, scale=100)

mdl.redundant(e['c'], '1')
B = em.B_matrix(mdl)
B.barx


######################################################################
# Part 3: Find member forces
# --------------------------
# 

B = em.B_matrix(mdl)
A = em.A_matrix(mdl)
V = A.f@U_disp
V

Ks = em.K_matrix(mdl).s
Q0 = em.Q_vector(B).o
Q = Ks@V + Q0
Q