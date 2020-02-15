"""
2r - Event to Event
===================

(220_HW12_P2r)

"""

import ema as em
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

mdl = em.rModel(2,3)
n = mdl.dnodes
e = mdl.delems
mdl.material('default', E=1000)
mdl.xsection('default', 1e6, 50)
xt = mdl.xsection('truss', 10, 1)


mdl.node('1', 0.0, 0.0)
mdl.node('2', 8.0, 0.0)
mdl.node('3', 8.0, 6.0)
mdl.node('4', 16., 6.0)
mdl.node('5', 16., -4.)

# elements
mdl.beam('a', n['1'], n['2'], Qpl=[1e6, 120,120])
mdl.beam('b', n['2'], n['3'], Qpl=[1e6, 120,120])
mdl.beam('c', n['3'], n['4'], Qpl=[1e6, 120,120])
mdl.beam('d', n['4'], n['5'], Qpl=[1e6, 180,180])
mdl.truss('e', n['2'], n['4'], xsec=xt, Qpl=[30])

# Fixities
mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['5'], ['x', 'y', 'rz'])

# Loading
n['3'].p['y'] = -30
n['3'].p['x'] =  50

mdl.DOF = mdl.numdofs()

fig, ax = plt.subplots(1,1)
em.plot_structure(mdl, ax)

ee = em.Event2Event(mdl)
ee.run()
ee.Q[-2]

ee.U