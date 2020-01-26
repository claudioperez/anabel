"""
Example 10.3
============

(Page 267 of CE-220 Fall 2019 course reader)

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
# %config InlineBackend.figure_format = 'svg'

dm = em.rModel(2,3)
A = 100000
I = 60
dm.xsection('default',A,I)
dm.material('default', 1000)

n1 = dm.node('1',  0.0,  2.0)
n2 = dm.node('2',  0.0, 10.0)
n3 = dm.node('3', 10.0, 10.0)
n4 = dm.node('4', 20.0, 10.0)
n5 = dm.node('5', 20.0,  0.0)

a = dm.beam('a', n1,  n2)
b = dm.beam('b', n2,  n3)
c = dm.beam('c', n3,  n4)
d = dm.beam('d', n4,  n5)

dm.hinge(a, n1)

a.w['y'] = -5
b.w['y'] = -10

dm.fix(n1, ['x', 'y', 'rz'])
dm.fix(n5, ['x', 'y', 'rz'])

# dm.DOF = [[6, 7, 8],[1, 7, 2], [1, 3, 4], [1, 9, 5], [10, 9, 11]]
dm.DOF = dm.numdofs()

fig, ax = plt.subplots()
em.plot_structure(dm, ax)


######################################################################
# Part 1
# ------
# 
# Determine the basic forces in all elements.
# 

# B = em.B_matrix(dm)
# Pw = em.Pw_vector(dm)
# Pw.f

Q0 = em.Q0_vector(dm)
Q0

assert abs(Q0.get('a_3') - -40.000000000) < 10e-8
assert abs(Q0.get('b_2') -  83.333333333) < 10e-8
assert abs(Q0.get('b_3') - -83.333333333) < 10e-8

P0 = B.f@Q0 + P.wf
P0

assert abs(P0.get('1') - -25) < 10e-8
assert abs(P0.get('2') -  43.33333333333) < 10e-8
assert abs(P0.get('3') -  50) < 10e-8
assert abs(P0.get('4') - -83.33333333333) < 10e-8

K = em.K_matrix(dm)
K.f

Uf = em.analysis.SolveDispl(dm)
Uf

Qi = B.bari@(P.f - P.wf)

Qi.df

Q = em.Q_vector(B,Qi)


######################################################################
# Part 2
# ------
# 
# Determine the horizontal and vertical translation at node 3.
# 

A = em.A_matrix(dm).add_cols(['1', '5']).add_cols(['3', '6']).remove(['7']).remove(['8']).remove('frame-axial')
V = em.V_vector(A)
A.c.df

F = em.F_matrix(dm)
F.s

V = F.s@Q
V

Uf = A.c.inv@V.c
Uf.df





