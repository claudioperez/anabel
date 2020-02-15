"""
Isostatic Truss
===============

The script ``Hw2P1.py`` was written in Python to define the model
geometry in terms of a python class named ``mdl``. Methods were then
written to act on this class and carry out the requested procedures.

"""

import ema as em #V0.1
import numpy as np
import sympy as sp
import numpy.linalg as LA
import matplotlib.pyplot as plt
# %config InlineBackend.figure_format = 'svg'

exec(open("scripts//Hw2P1.py").read())
nf = mdl.nf

fig, ax = plt.subplots(1,1)
em.plotting.plot_structure(mdl, ax)


######################################################################
# Systematically number all DOFs
# ------------------------------
# 

for i,node in enumerate(mdl.nodes):
    print("node {} DOFs : {}".format(node.tag, mdl.DOF[i]))


######################################################################
# Construct equilibrium equation, :math:`P_f = B_f Q`
# ---------------------------------------------------
# 

B = em.B_matrix(mdl)
P = em.P_vector(B)
P

B.f


######################################################################
# Solve for basic element force vector, :math:`Q`
# -----------------------------------------------
# 

B1 = B.bari
B1

Q = B1@P.f
Q


######################################################################
# Find support reactions, :math:`R`
# ---------------------------------
# 

R = B.d@Q
R


######################################################################
# Check global equilibrium
# ------------------------
# 

# sum horizontal forces:
R[0]+R[2] + P.get('13')

# sum vertical forces:
R[1]+R[3] + P.get('6')