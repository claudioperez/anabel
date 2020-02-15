"""
Problem 3
=========

"""

import numpy as np
import scipy.linalg 

k = 2.
b = 20.*12
m = 100./386
K = np.array([
    [6*k, 0.0, 0.0],
    [0.0, 6*k,-b*k],
    [0.0,-b*k, 3*k*b**2]
])
M = np.array([
    [  m, 0.0, 0.0],
    [0.0,   m, 0.0],
    [0.0, 0.0, m*b**2/6]
])
freq, shapes = scipy.linalg.eig(K,M)

freq

omegas = np.sqrt(freq)
omegas

np.around(shapes,5)