"""
Upper Bound Theorem 2
=====================

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg' # used to make plots look nicerbb

mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.node('1',  0.0, 0.0)
mdl.node('2',  6.0, 0.0)
mdl.node('3',  12.0, 0.0)
mdl.node('4', 0.0, 8.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.truss('c', n['2'], n['4'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['3'], ['y'])
mdl.fix(n['4'], ['x','y', 'rz'])

mdl.numDOF()

# Define plastic capacity
e['a'].Qp['+']['1'] = e['a'].Qp['-']['1'] = 500
e['a'].Qp['+']['2'] = e['a'].Qp['-']['2'] = 150
e['a'].Qp['+']['3'] = e['a'].Qp['-']['3'] = 150
e['c'].Qp['+']['1'] = e['c'].Qp['-']['1'] =  20
e['b'].Qp['+']['1'] = e['b'].Qp['-']['1'] = 500
e['b'].Qp['+']['2'] = e['b'].Qp['-']['2'] = 150
e['b'].Qp['+']['3'] = e['b'].Qp['-']['3'] = 150

em.analysis.characterize(mdl)

fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)

A = em.A_matrix(mdl)
A.f


######################################################################
# Vertical mechanism
# ~~~~~~~~~~~~~~~~~~
# 

n['2'].p['y'] = -50
n['3'].p['x'] =  0.
lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
Q

lambdac

B = em.B_matrix(mdl)
P = em.P_vector(B)
P.f

A.f@[0, 1,0, 0, 0]

[0, 1,0, 0, 0]@P.f

Qpl = em.Qpl_vector(mdl)[:,0]
Qpl

np.abs(A.f@[0, 1,0, 0, 0])@np.abs(Q)

