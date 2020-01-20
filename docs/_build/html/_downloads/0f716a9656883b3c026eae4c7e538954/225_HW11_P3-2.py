"""
MDF Earthquake Response - 3
===========================

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod
tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim
veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea
commodo consequat. Duis aute irure dolor in reprehenderit in voluptate
velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint
occaecat cupidatat non proident, sunt in culpa qui officia deserunt
mollit anim id est laborum.

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg' # used to make plots look nicer
from ema_examples.dynamics import P09_07
from ema.utilities.ipyutils import disp_sbs 

ft = 12
h = 12*ft
m = 80/386.4
E = 29000
I = 1000
EI = E*I
RHA = False

N = I = 3
mdl = P09_07(h = h, m = m, EI = EI)
# fig, ax = plt.subplots()
# em.plot_structure(mdl, ax)

m, k = em.Mass_matrix(mdl), em.K_matrix(mdl)
k, m = em.analysis.StaticCondensation(k, m)
disp_sbs(m.df, k.df)

freq2, Phi = em.analysis.ModalAnalysis(mdl, norm='last')
Phi

omega = np.array([np.sqrt(np.real(freq)) for freq in freq2])
omega

M = Phi.T@m@Phi
K = Phi.T@k@Phi


######################################################################
# Modal expansion of earthquake forces
# ------------------------------------
# 
# Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi
# ut aliquip ex ea commodo consequat. Duis aute irure dolor in
# reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
# pariatur.
# 
# .. math:: \mathrm{p}_{\mathrm{eff}}(t)=-\mathrm{m} \iota \ddot{u}_{g}(t)
# 
# .. math:: \mathbf{m} \iota=\sum_{n=1}^{N} \mathbf{s}_{n}=\sum_{n=1}^{N} \Gamma_{n} \mathbf{m} \phi_{n}
# 
# Excepteur sint occaecat cupidatat non proident, sunt in culpa qui
# officia deserunt mollit anim id est laborum.
# 

I = N = 3
iota = np.ones(I)
L = np.array([sum(Phi.T[n,i]*sum(m[i,j]*iota[j] for j in range(I)) for i in range(I)) for n in range(N)])
L = Phi.T@m@iota
L

gamma = np.array([L[n]/M[n,n]  for n in range(N)])
gamma

s = np.array([gamma[n]*(m@Phi.T[n]) for n in range(N)]).T
s


######################################################################
# a) Determine :math:`A_n` and :math:`D_n`
# ----------------------------------------
# 

# Values read from response spectrum:
D = np.array([0.877, 0.10, 0.04]) # inches
D

# if RHA:
D = []
u = []
for i, w in enumerate(omega):
    zeta = 0.05
    t, d = em.analysis.ElcentroRHA(zeta, w)
    D.append(max(d))
    u.append([t,d])
print(D)


######################################################################
# Plot modes:
# 

fig2, ax2 = plt.subplots()
em.plot_structure(mdl, ax2)
for i in range(3):
    plt.plot(10*u[i][0],200+300*u[i][1], linewidth=0.5)
plt.show()

A = np.array([D[n]*omega[n]**2 for n in range(N)])
A


######################################################################
# b) Modal response quantities
# ----------------------------
# 


######################################################################
# Floor displacements
# ~~~~~~~~~~~~~~~~~~~
# 

Un = np.array([[gamma[n]*Phi[i,n]*D[n] for n in range(N)]for i in range(I)])
Un


######################################################################
# Story shears
# ~~~~~~~~~~~~
# 

Vin = np.array([[sum(s[j,n]*A[n] for j in range(i, I)) for n in range(N)] for i in range(I)])
Vin


######################################################################
# Floor and base moments
# ~~~~~~~~~~~~~~~~~~~~~~
# 

M_base = np.array([sum(s[i,n]*h*(i+1)*A[n]  for i in range(I)) for n in range(N)])
M_base # kip-inch

H = [h*(i+1) for i in range(I)]
H

M_floor = np.array([[sum((H[j]-h*(i+1))*s[j,n]*A[n] for j in range(i,N)) for n in range(N)] for i in range(I)])
M_floor # kip-inch


######################################################################
# c) Peak modal response combination
# ----------------------------------
# 
# For well-seperated modal frequencies, the SRSS method is employed.
# 

def ro(rno):
    return np.sqrt(sum(rn**2 for rn in rno))


######################################################################
# Floor displacements
# ~~~~~~~~~~~~~~~~~~~
# 

ro(Un.T)


######################################################################
# Story shears
# ~~~~~~~~~~~~
# 

ro(Vin.T)


######################################################################
# Floor and base overturning moments
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 

ro(M_base)

ro(M_floor)