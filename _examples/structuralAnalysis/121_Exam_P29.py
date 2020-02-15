"""
Spring
======

(121_Exam_P29)

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg' # used to make plots look nicerbb

mdl = em.Model(2,3)
n = mdl.dnodes
e = mdl.delems
mdl.material('default', E=1000)

mdl.node('1',  0.0, 0.0)
mdl.node('2',  8.0, 0.0)
mdl.node('3', 16.0, 0.0)
mdl.node('4',  0.0, 6.0)
mdl.node('5', 16.0,-1.0)

mdl.beam('a', n['1'], n['2'])
mdl.beam('b', n['2'], n['3'])
mdl.truss('c', n['2'], n['4'])
mdl.truss('d', n['3'], n['5'])

mdl.fix(n['1'], ['x', 'y', 'rz'])
mdl.fix(n['4'], ['x','y', 'rz'])
mdl.fix(n['5'], ['x','y', 'rz'])

mdl.numDOF()
em.analysis.characterize(mdl)

e['a'].I = 100
e['b'].I = 100
e['a'].A = 1e6
e['b'].A = 1e6
e['c'].A = 20
e['d'].A = 5

n['2'].p['y'] = -50
Uf = em.analysis.SolveDispl(mdl)
Uf

assert abs(Uf[1] - -0.015303) < 10e-3
assert abs(Uf[4] - -0.0024012) < 10e-3

fig, ax = plt.subplots(1,1)
# %matplotlib qt
em.plot_U(mdl, Uf, ax, scale=100)

A = em.A_matrix(mdl)
A.f

B = em.B_matrix(mdl)
P = em.P_vector(B)
P.f