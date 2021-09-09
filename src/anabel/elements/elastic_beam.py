"""Element library
"""
import inspect
from typing import Union
from functools import partial
from abc import abstractmethod

import numpy as np
import scipy.linalg
import scipy.integrate
from numpy.polynomial import Polynomial
from scipy.integrate import quad

from anabel.abstract import SkeletalElement


try:
    import anabel.backend as anp
except:
    anp = np

try:
    from emme.matrices import Structural_Matrix, Structural_Vector
except:
    from anabel.matrices import Structural_Matrix, Structural_Vector


def _unpack(props,obj,fallback=None,default=lambda x: "$"+x):
    return map(lambda x: obj[x] if x in obj else getattr(fallback, x, default(x)), props)

def rotation_matrix(xyz):
    DX = xyz[1] - xyz[0]
    dx, dy, dz = DX / np.linalg.norm(DX)
    return np.array([
            [ dx, -dy, 0.0],
            [ dy,  dx, 0.0],
            [0.0, 0.0, 1.0]])
    #return  np.array([
    #    [dx*dz,  dx**2*dz-dy**2,  dx**2*dy+dy*dz],
    #    [dy*dz,  dx*dy*dz+dx*dy,  dx*dy**2-dx*dz],
    #    [ -dx,          dz**2,         dz*dy    ]])

def _elastic_curve(x, u, L, EI=1e6, scale=1.0): #-> float[n]
    """
    u: float[2][2]

    u[0][1]
     /           
    #----------------#
    ^ u[0][0]

    """
    xi = x/L
    N1 = 1-3*xi**2+2*xi**3
    N2 = L*(xi-2*xi**2+xi**3)
    N3 = 3*xi**2-2*xi**3
    N4 = L*(xi**3-xi**2)
    vi = u[0]
    vj = u[1]
    print(f"v: {vi}, {vj}")
    y = np.array(vi*N2+vj*N4)*scale
    # xy = np.concatenate(([x],[y]))
    return y.flatten()

#plan_rot = lambda u: u[[4,10]]
elev_rot = lambda u: u[[1,2]]
plan_rot = lambda u: u[[3,4]]


def local_deformations(u,L):
    xi, yi, zi, si, pi, ei = range(6)
    xj, yj, zj, sj, pj, ej = range(6,12)
    chord_elev = (u[yj]-u[yi]) / L
    chord_plan = (u[zj]-u[zi]) / L
    return np.array([
        [u[xj] - u[xi]], # xi
        [u[ei] - chord_elev],  # vi_plan
        [u[ej] - chord_elev],  # vj_plan

        [u[pi] + chord_plan],
        [u[pj] + chord_plan],
        [u[sj] - u[si]],
    ])


def displaced_profile(xyz, U, x=None, glob=False, scale=1.0, EI=None):
    L = np.linalg.norm(xyz[1] - xyz[0])
    print(L, L/12)
    R = rotation_matrix(xyz)
    u = scipy.linalg.block_diag(*[R.T]*4)@U
    print(f"u_glo_x: {U[0]}, {U[6]}") 
    print(f"u_loc_y: {u[1]}, {u[7]}") 
    u = U
    v = local_deformations(u,L)
    if not x: 
        x = np.linspace(0, L, 20) 
    c = np.stack([
        x,
        _elastic_curve(x, elev_rot(v), L, scale=scale, EI=EI),
       -_elastic_curve(x, plan_rot(v), L, scale=scale, EI=EI),
    ])
    if glob:
        #X0 = np.array(list(zip(*[xyz[0]]*len(x))))
        #print(u[:3],u[~5:~2])
        dX = np.linspace(u[:3],  u[~5:~2], len(x)) * scale
        #print(f"d: {R}")
        return R@(c + dX.T) + xyz[0][None,:].T
    else:
        return c


class _PlaceHolderNode:
    def __init__(self, tag):
        self.tag = tag


class ElasticBeam(SkeletalElement): 
    """Linear 2D/3D Euler-Bernouli frame element
    ```           
                  ^
    ______________|___ 
              |   |   |
              |   +---|->
    __________|_______|

    ```
    """

    def __init__(self,
            ndm,
            nodes=None,
            section=None,
            material=None,
            tag=None,
            units=None,
            properties=None,
            transform=None,
            massDens = None,
            consistent_mass = False,
            **kwds
    ):
        self.ndf = {2: 3, 3: 12}[ndm]
        self.nv = {2: 3, 3: 6}[ndm]
        #self.nn = 2
        self.nq = self.nv
        self.force_dict = {'1':0, '2':0, '3': 0}
        self.Qpl = np.zeros((2,self.nq))

        if nodes is None:
            nodes = [_PlaceHolderNode("$iNode"), _PlaceHolderNode("$jNode")]

        super().__init__(self.ndf, ndm, self.force_dict, nodes, tag=tag, **kwds)
        if units:
            self._units = units
        elif section and section.units:
            self._units = section.units
        elif material and material.units:
            self._units = material.units
        self._material = material
        self._mass = massDens

        if section:
            E, G, A, Iz, Iy, J = _unpack(["E", "G", "area","Ix","Iy","J"], kwds, section)

        if self._material:
            E = self._material.elastic_modulus
            G = self._material.shear_modulus
            self._mass = self._material.mass

        self.E:  float = E
        self.A:  float = A
        self.G:  float = G
        self.J:  float = J
        self.Iz: float = Iz
        self.Iy: float = Iy
        self._consistent_mass_flag = consistent_mass
        self._transform =transform
        self._section = section
        self._material = material

    @property
    def section(self):
        return self._section


    def dump_opensees(self, **kwds):
        transfTag = self._transform.tag if self._transform else "$transfTag"
        nodes = " ".join(f"{n.tag}" for n in self.nodes)
        props = " ".join(
                f"{getattr(self,p,'$'+p):4.4}" \
                    for p in ["A", "E", "G", "J", "Iy", "Iz"]
        )
        flags = ""
        if self._mass:
            mass = self._mass * self.A
            flags += f" -mass {mass}"
        flags += " -cMass" if self._consistent_mass_flag else ""
        return f"element elasticBeamColumn {self.tag} {nodes} {props} {transfTag} {flags}"


