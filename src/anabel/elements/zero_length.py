"""Element library
"""
from typing import Union
from functools import partial
from abc import abstractmethod

import numpy as np
from numpy.polynomial import Polynomial
from scipy.integrate import quad
import scipy.integrate

from anabel.abstract import Element, ModelComponent

try:
    import anabel.backend as anp
except:
    anp = np

try:
    from emme.matrices import Structural_Matrix, Structural_Vector
except:
    from anabel.matrices import Structural_Matrix, Structural_Vector



class ZeroLength(Element): 
    """
    """
    nn = 2
    def __init__(self,
            tag=None,
            nodes=None,
            mat=None,
            properties=None,

            **kwds
    ):
        self._tag = tag
        self.ndf = len(mat)

        self._materials = self.materials = mat
        self.nodes = nodes



    def dump_opensees(self, **kwds):
        nodes = " ".join(f"{n.tag:5}" for n in self.nodes)
        mats = "-mat " + " ".join(str(m.tag) for m in self._materials.values())
        dirs = "-dir " + " ".join(f"${d}" for d in self.materials.keys())
        return f"element zeroLength {self.tag} {nodes} {mats} {dirs} "


