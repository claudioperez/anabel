"""
Lower Bound - 2
===============

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

mdl.node('1', 0.0, 0.0)
mdl.node('2', 8.0, 0.0)
mdl.node('3', 8.0, 6.0)
mdl.node('4', 16., 6.0)
mdl.node('5', 16., -4.)

# elements
mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])
mdl.beam('d', n['4'], n['5'])
mdl.truss('e', n['2'], n['4'])

# redundants
mdl.redundant(e['a'], '2')
mdl.redundant(e['c'], '2')
mdl.redundant(e['d'], '3')
mdl.redundant(e['e'], '1')

# Fixities
mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['5'], ['x', 'y', 'rz'])

# Loading
n['3'].p['y'] = -30
n['3'].p['x'] =  50

# Define plastic capacity
e['a'].Qp['+']['1'] = e['a'].Qp['-']['1'] = 1000
e['a'].Qp['+']['2'] = e['a'].Qp['-']['2'] = 120
e['a'].Qp['+']['3'] = e['a'].Qp['-']['3'] = 120
 
e['c'].Qp['+']['1'] = e['c'].Qp['-']['1'] = 1000
e['c'].Qp['+']['2'] = e['c'].Qp['-']['2'] = 120
e['c'].Qp['+']['3'] = e['c'].Qp['-']['3'] = 120
 
e['b'].Qp['+']['1'] = e['b'].Qp['-']['1'] = 1000
e['b'].Qp['+']['2'] = e['b'].Qp['-']['2'] = 150
e['b'].Qp['+']['3'] = e['b'].Qp['-']['3'] = 150

e['d'].Qp['+']['1'] = e['d'].Qp['-']['1'] = 1000
e['d'].Qp['+']['2'] = e['d'].Qp['-']['2'] = 180
e['d'].Qp['+']['3'] = e['d'].Qp['-']['3'] = 180

e['e'].Qp['+']['1'] = e['e'].Qp['-']['1'] =  30

mdl.numDOF()
em.analysis.characterize(mdl)

#Remove
fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)

B = em.B_matrix(mdl)
A = em.A_matrix(mdl)
Qpl = em.Qpl_vector(mdl)
P = em.P_vector(B)
B

A

Qpr = B.bari@P.f


em.analysis.PlasticAnalysis_wLBT(mdl)