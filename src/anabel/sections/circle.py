from anabel.abstract import FrameSection

from .patches import CircularPatch

from anabel.abstract import ModelComponent
from anabel.utilities import take_keywords

class Circle(CircularPatch, FrameSection, ModelComponent):
    def __init__(self, **kwds):
        FrameSection.__init__(self, **take_keywords(kwds, "units","domain","parent"))
        CircularPatch.__init__(self, (0.0, 0.0), **kwds)

    def dump_opensees(self, **kwds)->str:
        if self.tag: 
            tag = self.tag
        else:
            tag = "$secTag"

        if self.material:
            matTag = self.material.tag
        else:
            matTag = "$matTag"
        numSubDivY, numSubDivZ = self.no_divs
        x,y = 0, 0
        cmd = f"section fiberSec {tag} {{\n"
        cmd += f"\tpatch circ {matTag} {numSubDivY} {numSubDivZ} {x}   {y}  0   {self.radius:8.8}  0    360\n}}"
        return cmd

