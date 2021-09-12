import numpy as np
import scipy.linalg


def rotation(xyz, vert=(0,0,-1)):
    """


    ```
    1           2
    |__0  ->    |__ 0
    /2          /-1
    ```
    """
    DX = xyz[1] - xyz[0]
    R = np.zeros([len(DX)]*2)
    R[0,:] = DX / np.linalg.norm(DX)
    R[1,:] = np.asarray(vert)/np.linalg.norm(vert)
    R[2,:] = np.cross(R[0,:], R[1,:])
    return R


def _elastic_curve(x, u, L, EI=1e6, scale=1.0): #-> float[n]
    """
    u: float[2][2]

    u[0][1]
     /           
    #----------------#
    ^
     
    """
    vi = u[0]
    vj = u[1]
    xi = x/L
    N1 = 1-3*xi**2+2*xi**3
    N2 = L*(xi-2*xi**2+xi**3)
    N3 = 3*xi**2-2*xi**3
    N4 = L*(xi**3-xi**2)
    y = np.array(vi*N2+vj*N4)*scale
    return y.flatten()

#plan_rot = lambda u: u[[4,10]]
elev_rot = lambda u: u[[1,2]]
plan_rot = lambda u: u[[3,4]]


def local_deformations(u,L):
    xi, yi, zi, si, ei, pi = range(6)
    xj, yj, zj, sj, ej, pj = range(6,12)
    chord_elev = (u[zj]-u[zi]) / L
    chord_plan = (u[yj]-u[yi]) / L
    return np.array([
        [u[xj] - u[xi]], # xi
        [u[ei] - chord_elev],  # vi_elev
        [u[ej] - chord_elev],  # vj_elev

        [u[pi] - chord_plan],
        [u[pj] - chord_plan],
        [u[sj] - u[si]],
    ])

def displaced_chord(xyz, U, x=None, glob=False, scale=1.0):
    L = np.linalg.norm(xyz[1] - xyz[0])
    R = np.eye(3) #rotation(xyz)
    u = scipy.linalg.block_diag(*[R]*4)@U
    if not x: 
        x = np.linspace(0, L, 20)
    #dX = np.linspace(u[:3],  u[~5:~2], len(x)) * scale
    dX = np.stack([
        x + np.linspace(u[0], u[~5], len(x)) * scale,
        *np.linspace(u[1:3],  u[~4:~2], len(x)).T * scale
    ]).T
    if glob:
        return R.T@dX.T + xyz[0][None,:].T
    else:
        return dX.T

def displaced_profile(xyz, U, x=None, glob=False, scale=1.0, EI=None):
    L = np.linalg.norm(xyz[1] - xyz[0])
    #R = rotation(xyz)
    R = np.eye(3) #rotation(xyz)
    u = scipy.linalg.block_diag(*[R]*4)@U
    v = local_deformations(u,L)
    if not x: 
        x = np.linspace(0, L, 20)

    #dX = np.linspace(u[:3],  u[~5:~2], len(x)) * scale
    c = np.stack([
        x,
        _elastic_curve(x, plan_rot(v), L, scale=scale, EI=EI),
        _elastic_curve(x, elev_rot(v), L, scale=scale, EI=EI),
    ]) #+dX.T
    if glob:
        ret = R.T@c + xyz[0][None,:].T
        return None
    else:
        return c

