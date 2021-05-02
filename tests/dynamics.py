import numpy as np
import emme as em


L = 1
k = np.array([
    [12, -12, -3*L, -3*L],
    [-12, 24,  3*L,   0 ],
    [-3*L, 3*L, L**2, L**2/2],
    [-3*L, 0.0, L**2/2, 2*L**2]])*8

K = em.analysis.kStaticCondensation(k, idxs = [0,1])* 7/48

assert abs(K[0,0] - 2) < 1e-7
assert abs(K[1,0] - -5) < 1e-7
assert abs(K[1,1] - 16) < 1e-7
assert abs(K[0,1] - -5) < 1e-7