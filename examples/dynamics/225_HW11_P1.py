"""
Problem 1
=========

"""

import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg' # used to make plots look nicer
from ema_examples.dynamics import P09_07
from ema.utilities.ipyutils import disp_sbs 

h = 1
m = 1
EI = 1/24

mdl = P09_07(h = h, m = m, EI = EI)
fig, ax = plt.subplots()
em.plot_structure(mdl, ax)

m, k = em.Mass_matrix(mdl), em.K_matrix(mdl)
k, m = em.analysis.StaticCondensation(k, m)
disp_sbs(m.df, k.df)

freq2, Phi = em.analysis.ModalAnalysis(mdl, norm='last')
Phi

omegan = np.array([np.sqrt(np.real(freq)) for freq in freq2])
omegan

M = Phi.T@m@Phi
K = Phi.T@k@Phi
M


######################################################################
# a) Modal expansion of earthquake forces
# ---------------------------------------
# 
# .. math:: \mathrm{p}_{\mathrm{eff}}(t)=-\mathrm{m} \iota \ddot{u}_{g}(t)
# 
# .. math:: \mathbf{m} \iota=\sum_{n=1}^{N} \mathbf{s}_{n}=\sum_{n=1}^{N} \Gamma_{n} \mathbf{m} \phi_{n}
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

# Remove
assert abs(s[0,0] - 6.22008468e-01) <1e-7
assert abs(s[1,0] - 1.07735027e+00) <1e-7
assert abs(s[0,1] - 3.33333333e-01) <1e-7


######################################################################
# Floor displacement response in terms of :math:`D_n(t)`
# ------------------------------------------------------
# 

Un = [[gamma[n]*Phi[i,n] for n in range(N)]for i in range(I)]
Un


######################################################################
# Story shear response in terms of :math:`A_n(t)`
# -----------------------------------------------
# 

Vin = np.array([sum(s[j] for j in range(i, I)) for i in range(I)])
Vin

assert abs(Vin[0,0] - 2.32136721) <1e-7
assert abs(Vin[1,0] - 1.69935874) <1e-7
assert abs(Vin[0,1] - 0.16666667) <1e-7


######################################################################
# d) Base overturning moment in terms of :math:`A_n(t)`
# -----------------------------------------------------
# 

Mbn = np.array([sum(s[i,n]*h*(i+1) for i in range(I)) for n in range(N)])
Mbn

assert abs(Mbn[0] -  4.64273441) <1e-7
assert abs(Mbn[1] - -0.16666666) <1e-7
assert abs(Mbn[2] -  0.02393225) <1e-7


######################################################################
# e) Effective modal mass and heights.
# ------------------------------------
# 

M_eff = np.array([gamma[n]*L[n] for n in range(N)])
M_eff

assert abs(M_eff[0] - 2.32136721) < 1e-7
assert abs(M_eff[1] - 0.16666667) < 1e-7
assert abs(M_eff[2] - 0.01196613) < 1e-7

L_theta = np.array([sum(m[i,i]*h*(i+1)*Phi[i,n] for i in range(I)) for n in range(N)])

h_eff = np.array([L_theta[n]/L[n] for n in range(N)])
h_eff

assert abs(h_eff[0] -  2.) < 1e-7
assert abs(h_eff[1] - -1.) < 1e-7
assert abs(h_eff[2] -  2.) < 1e-7