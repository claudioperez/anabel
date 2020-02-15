"""
Problem 4
=========

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
mdl.node('3', 16.0, 0.0)
mdl.node('4',  8.0, 6.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['2'], n['4'])
mdl.truss('d', n['3'], n['4'])

# Define plastic capacity
e['a'].Qp['+']['1'] = e['a'].Qp['-']['1'] = 500
e['a'].Qp['+']['2'] = e['a'].Qp['-']['2'] = 150.001
e['a'].Qp['+']['3'] = e['a'].Qp['-']['3'] = 150.001
 
e['c'].Qp['+']['1'] = e['c'].Qp['-']['1'] = 500
e['c'].Qp['+']['2'] = e['c'].Qp['-']['2'] = 120
e['c'].Qp['+']['3'] = e['c'].Qp['-']['3'] = 120
 
e['b'].Qp['+']['1'] = e['b'].Qp['-']['1'] = 500
e['b'].Qp['+']['2'] = e['b'].Qp['-']['2'] = 150
e['b'].Qp['+']['3'] = e['b'].Qp['-']['3'] = 150

e['d'].Qp['+']['1'] = e['d'].Qp['-']['1'] =  20

mdl.roller(n['1'])
mdl.fix(n['3'], ['x', 'y', 'rz'])

mdl.numDOF()

n['4'].p['x'] = 10
n['2'].p['y'] = -30

Uf = em.analysis.SolveDispl(mdl)

fig, ax = plt.subplots(1,1)
em.plot_U(mdl, Uf, ax, scale=1e-3)

Uf

lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
Q

lambdac



