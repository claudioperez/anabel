"""
Upper Bound Theorem 1
=====================

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg' # used to make plots look nicerbb

#Remove
mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.node('1',  0.0, 0.0)
mdl.node('2',  6.0, 0.0)
mdl.node('3',  6.0, 4.0)
mdl.node('4', 12.0, 4.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['4'], ['y'])

mdl.numDOF()

# Define plastic capacity
e['a'].Qp['+']['1'] = e['a'].Qp['-']['1'] = 500
e['a'].Qp['+']['2'] = e['a'].Qp['-']['2'] = 200
e['a'].Qp['+']['3'] = e['a'].Qp['-']['3'] = 200
e['c'].Qp['+']['1'] = e['c'].Qp['-']['1'] = 500
e['c'].Qp['+']['2'] = e['c'].Qp['-']['2'] = 200
e['c'].Qp['+']['3'] = e['c'].Qp['-']['3'] = 200
e['b'].Qp['+']['1'] = e['b'].Qp['-']['1'] = 500
e['b'].Qp['+']['2'] = e['b'].Qp['-']['2'] = 240
e['b'].Qp['+']['3'] = e['b'].Qp['-']['3'] = 240

em.analysis.characterize(mdl)

fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)

A = em.A_matrix(mdl)
A.f


######################################################################
# Vertical mechanism
# ~~~~~~~~~~~~~~~~~~
# 

n['3'].p['y'] = -30
n['3'].p['x'] =  0.
lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
Q

lambdac


######################################################################
# Horizontal mechanism
# ~~~~~~~~~~~~~~~~~~~~
# 

n['3'].p['y'] = 0
n['3'].p['x'] = 30
lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
Q

lambdac


######################################################################
# Combined mechanism
# ~~~~~~~~~~~~~~~~~~
# 

n['3'].p['y'] = -30
n['3'].p['x'] = 30
lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
Q

lambdac