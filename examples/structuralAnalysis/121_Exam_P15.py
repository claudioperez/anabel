"""
Problem 15
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
mdl.node('2', -8.0, 6.0)
mdl.node('3', -4.0, 6.0)
mdl.node('4',  0.0, 6.0)
mdl.node('5',  0.0,12.0)

mdl.beam('a', n['1'], n['4'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])
mdl.beam('d', n['4'], n['5'])
mdl.truss('e', n['2'], n['5'])


mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['2'], ['y'])

mdl.numDOF()

fig, ax = plt.subplots()
em.plot_structure(mdl, ax, labeled=True)

n['3'].p['y'] = -80
n['4'].p['x'] = -20
n['5'].p['x'] = -40

e['a'].Qp['+']['1'] = e['a'].Qp['-']['1'] = 80
e['a'].Qp['+']['2'] = e['a'].Qp['-']['2'] = 600
e['a'].Qp['+']['3'] = e['a'].Qp['-']['3'] = 600

e['d'].Qp['+']['1'] = e['d'].Qp['-']['1'] = 80
e['d'].Qp['+']['2'] = e['d'].Qp['-']['2'] = 600
e['d'].Qp['+']['3'] = e['d'].Qp['-']['3'] = 600

e['b'].Qp['+']['1'] = e['b'].Qp['-']['1'] = 80
e['b'].Qp['+']['2'] = e['b'].Qp['-']['2'] = 500
e['b'].Qp['+']['3'] = e['b'].Qp['-']['3'] = 500

e['c'].Qp['+']['1'] = e['c'].Qp['-']['1'] = 80
e['c'].Qp['+']['2'] = e['c'].Qp['-']['2'] = 500
e['c'].Qp['+']['3'] = e['c'].Qp['-']['3'] = 500

e['e'].Qp['+']['1'] = e['e'].Qp['-']['1'] = 50

lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
lambdac

Q


######################################################################
# Mechanism Analysis
# ------------------
# 

mdl.redundant(e['a'], '2')
mdl.redundant(e['b'], '3')
mdl.redundant(e['e'], '1')

A = em.A_matrix(mdl)
Udot = A.i.ker / A.i.ker[0]
Udot

A.f@Udot

A.f



