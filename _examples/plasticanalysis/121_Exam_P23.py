"""
33. Plastic Analysis
====================

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems

mdl.xsection('default', 1e8, 40000)
xt = mdl.xsection('truss', 10e3, 1)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  6.0, 0.0)
mdl.node('3',  6.0, 4.0)
mdl.node('4', 12.0, 4.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.beam('c', n['3'], n['4'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.roller(n['4'])

mdl.numDOF()

# Define plastic capacity
Q_axial = 500
e['a'].Qp['+']['1'] = e['a'].Qp['-']['1'] = Q_axial
e['a'].Qp['+']['2'] = e['a'].Qp['-']['2'] = 200
e['a'].Qp['+']['3'] = e['a'].Qp['-']['3'] = 200
e['c'].Qp['+']['1'] = e['c'].Qp['-']['1'] = Q_axial
e['c'].Qp['+']['2'] = e['c'].Qp['-']['2'] = 200
e['c'].Qp['+']['3'] = e['c'].Qp['-']['3'] = 200
e['b'].Qp['+']['1'] = e['b'].Qp['-']['1'] = Q_axial
e['b'].Qp['+']['2'] = e['b'].Qp['-']['2'] = 240
e['b'].Qp['+']['3'] = e['b'].Qp['-']['3'] = 240

n['3'].p['y'] =-30
n['3'].p['x'] = 30
lambdac, Q = em.analysis.PlasticAnalysis_wLBT(mdl)
lambdac

Q

