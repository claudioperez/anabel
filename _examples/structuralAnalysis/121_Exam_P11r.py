"""
Problem 11r
===========

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.node('1',  0.0, 0.0)
mdl.node('2', -8.0, 6.0)
mdl.node('3',  0.0, 6.0)
mdl.node('4', -8.0,12.0)
mdl.node('5',  0.0,12.0)

mdl.beam('a', n['1'], n['3'])
mdl.beam('b', n['3'], n['5'])
mdl.beam('c', n['2'], n['3'])
mdl.beam('d', n['4'], n['5'])
mdl.truss('e', n['2'], n['5'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.roller(n['2'])
mdl.roller(n['4'])

mdl.DOF = mdl.numdofs()

n['3'].p['x'] =  15
n['5'].p['x'] =  10
e['c'].w['y'] = -10
e['d'].w['y'] = -10

Uf = em.analysis.SolveDispl(mdl)

fig, ax = plt.subplots(1,1)
em.plot_U(mdl, Uf, ax, scale=0.001)

Uf