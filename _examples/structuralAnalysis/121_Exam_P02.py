"""
Problem 2
=========

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

#Remove
mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.node('1',  0.0, 0.0)
mdl.node('2',  8.0, 0.0)
mdl.node('3', 16.0, 0.0)
mdl.node('4',  8.0, 6.0)
mdl.node('5', 16.0, 6.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['4'])
mdl.beam('c', n['4'], n['5'])
mdl.truss('d', n['1'], n['4'])
mdl.truss('e', n['3'], n['4'])

mdl.fix(n['1'], ['x','y', 'rz'])
mdl.fix(n['3'], ['x','y', 'rz'])
mdl.fix(n['5'], ['y'])

e['a'].w['y'] = -10
e['b'].w['y'] = -10
mdl.numDOF()


######################################################################
# Part 1: Degree of indeterminacy
# -------------------------------
# 

em.analysis.characterize(mdl)

fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)
# U_disp = em.analysis.SolveDispl(mdl) 
# em.plot_U(mdl, U_disp, ax, scale=100)

