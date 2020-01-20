"""
Problem 10
==========

"""

import ema as em
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

p2 = em.Model(2,3)

n1 = mdl.node('1', 0.0, 0.0)
n2 = mdl.node('2', 6.0, 0.0)
n3 = mdl.node('3', 6.0, 8.0)
n4 = mdl.node('4', 12.0, 8.0)
n5 = mdl.node('5', 12.0, 0.0)

a = mdl.beam('a', n1, n2)
b = mdl.beam('b', n2, n3)
c = mdl.beam('c', n3, n4)
d = mdl.truss('d', n3, n5)

mdl.fix(n1, ['x', 'y', 'rz'])
mdl.fix(n5, ['x', 'y', 'rz'])
mdl.fix(n4, 'y')


fig, ax= plt.subplots(1,1)
em.plot_structure(mdl, ax)

p2.numDOF()

# em.utilities.export.FEDEAS(p2)

