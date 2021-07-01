"""Element library
"""
import inspect
from typing import Union
from functools import partial
from abc import abstractmethod

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
import scipy.integrate

from anabel.abstract import Element


try:
    import anabel.backend as anp
except:
    anp = np

try:
    from emme.matrices import Structural_Matrix, Structural_Vector
except:
    from anabel.matrices import Structural_Matrix, Structural_Vector


def _unpack(props,obj,fallback=None,default=None):
    return map(lambda x: obj[x] if x in obj else getattr(fallback, x, default), props)


class ElasticBeam(Element): 
    """Linear 2D/3D Euler-Bernouli frame element
    ```           
                  ^
    ______________|___ 
              |   |   |
              |   +---|->
    __________|_______|

    ```
    """
    nv = 3
    nn = 2
    nq = 3
    ndm = 2
    force_dict = {'1':0, '2':0, '3': 0}
    Qpl = np.zeros((2,nq))
    opensees_command = "element elasticBeamColumn {tag} {nodes:10}"
    section_properties = {
        "A":  "area",
        "Iy": "Iy",
        "Iz": "Ix"
    }

    def __init__(self,
            ndm,
            nodes=None,
            section=None,
            material=None,
            tag=None,
            properties=None,
            **kwds
    ):
        self.ndf = {2: 3, 3: 6}[ndm]

        if nodes is None:
            nodes = [None]*2

        super().__init__(self.ndf, self.ndm, self.force_dict, nodes, tag=tag, **kwds)

        if section:
            E, A, Iz, Iy, J = _unpack(["E","area","Ix","Iy","J"], kwds, section, default=None)

        self.E:  float = E
        self.A:  float = A
        self.G:  float = None
        self.J:  float = J
        self.Iz: float = Iz
        self.Iy: float = Iy

    def dump_opensees(self, transfTag=None, **kwds):
        nodes = " ".join(f"{n.tag:5}" for n in self.nodes)
        props = " ".join(
                f"{getattr(self,p)}" \
                    for p in ["A", "E", "G", "J", "Iy", "Iz"]
        )
        return f"element elasticBeamColumn {self.tag} {nodes} {props} {transfTag}"


