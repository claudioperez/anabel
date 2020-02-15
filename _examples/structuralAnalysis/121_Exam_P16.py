"""
Problem 16
==========

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

#Remove
mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.xsection('default', 1e6, 50000)
xt = mdl.xsection('truss', 50e3, 1)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  6.0, 8.0)
mdl.node('3', 12.0, 8.0)
mdl.node('4', 12.0, 0.0)

mdl.beam('a', n['2'], n['3'])
mdl.beam('b', n['3'], n['4'])
mdl.truss('c', n['1'], n['2'], xsec=xt)

mdl.hinge(e['a'], n['2'])
mdl.hinge(e['b'], n['4'])

mdl.fix(n['1'], ['x','y', 'rz'])
mdl.fix(n['2'], ['rz'])
mdl.fix(n['4'], ['x','y', 'rz'])

e['a'].w['y'] = -10
n['2'].p['x'] = - 5
mdl.DOF = [[4, 5, 6], [1, 2, 7], [1, 8, 3], [9, 8, 10]]


######################################################################
# Part 1: Degree of indeterminacy
# -------------------------------
# 

em.analysis.characterize(mdl)

fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)
U_disp = em.analysis.SolveDispl(mdl) 
em.plot_U(mdl, U_disp, ax, scale=100)


######################################################################
# Part 3: Find member forces
# --------------------------
# 

B = em.B_matrix(mdl)
P = em.P_vector(B)
Q = B.bari@(P.f-P.wf)
Q


######################################################################
# Part 4: Find node translations
# ------------------------------
# 

Fs = em.Fs_matrix(mdl)
A = em.A_matrix(mdl)
V = em.V_vector(A)

Q0 = em.Q_vector(B).o
Q0

P0 = B.f@Q0 + Pwf
P0