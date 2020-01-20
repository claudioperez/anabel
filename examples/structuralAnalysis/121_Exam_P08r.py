"""
Problem 8
=========

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.material('default', 1000)
mdl.xsection('default', 10000000000, 100)
xt = mdl.xsection('truss', 100, 1)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  8.0, 0.0)
mdl.node('3', 16.0, 0.0)
mdl.node('4',  8.0, 6.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['2'], n['4'])
mdl.truss('d', n['1'], n['4'], xsec=xt)

mdl.hinge(e['c'], n['2'])

mdl.pin(n['1'])
mdl.roller(n['3'])

mdl.numDOF()

n['4'].p['x'] = 10
e['b'].w['y'] = -5

Uf = em.analysis.SolveDispl(mdl)

fig, ax = plt.subplots(1,1)
em.plot_U(mdl, Uf, ax)

Uf

