"""
Example 12.3
============

(Page 335 of CE-220 Fall 2019 course reader)

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
# %config InlineBackend.figure_format = 'svg'

mdl = em.Model(2,3)

n1 = mdl.node('1',  0.0,  0.0)
n2 = mdl.node('2',  0.0,  5.0)
n3 = mdl.node('3',  4.0,  5.0)
n4 = mdl.node('4',  8.0,  5.0)
n5 = mdl.node('5',  8.0,  0.0)

a = mdl.beam('a', n1,  n2, Qpl = [1e5, 150, 150])
b = mdl.beam('b', n2,  n3, Qpl = [1e5, 120, 120])
c = mdl.beam('c', n3,  n4, Qpl = [1e5, 120, 120])
d = mdl.beam('d', n4,  n5, Qpl = [1e5, 150, 150])

mdl.fix(n1, ['x', 'y','rz'])
mdl.fix(n5, ['x', 'y','rz'])

n2.p['x'] =  30
n3.p['y'] = -50

mdl.numDOF()
em.analysis.characterize(mdl)

fig, ax = plt.subplots()
em.plot_structure(mdl, ax)

A = em.A_matrix(mdl)
A.c

var = em.analysis.setupPlasticAnalysis_wLBT(mdl)

em.analysis.PlasticAnalysis_wLBT(mdl)

