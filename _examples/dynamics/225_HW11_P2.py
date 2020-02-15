"""
Problem 2
=========

(see Ex 9.6)

"""

# Remove
import ema as em
import matplotlib.pyplot as plt
import numpy as np
# %config InlineBackend.figure_format = 'svg' # used to make plots look nicer
from ema_examples.dynamics import E09_06
from ema.utilities.ipyutils import disp_sbs 
from scipy.linalg import eig

L = 1
mass = 1
EI = 1/12

mdl = E09_06(L=L, m = mass, EI = EI)
fig, ax = plt.subplots()
em.plot_structure(mdl, ax)
mdl.DOF

m, K = em.Mass_matrix(mdl), em.K_matrix(mdl)
m[0,0] = 3.*mass
m[1,1] = 1.*mass
m[2:,2:] = 0*mass
disp_sbs(m.f.df, K.f.df)

# k, m = em.analysis.StaticCondensation(k.f, m.f, idxs=[3,4])
k = em.analysis.kStaticCondensation(K.f, idxs=[3,4])
# disp_sbs(k.df)
k

m = m[0:2,0:2]
freq2, Phi = eig(m, k)
Phi/0.57293852

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

I = N = 2
iota = np.array([0, 1])
Ln = np.array([sum(Phi.T[n,i]*sum(m[i,j]*iota[j] for j in range(I)) for i in range(I)) for n in range(N)])
Ln = Phi.T@m@iota
Ln

gamma = np.array([Ln[n]/M[n,n] for n in range(N)])
gamma

sn = [gamma[n]*(m@Phi[:,n]) for n in range(N)]
sn