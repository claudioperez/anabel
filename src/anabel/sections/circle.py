from anabel.abstract import FrameSection

from .patches import CircularPatch

from anabel.abstract import ModelComponent
from anabel.utilities import take_keywords

class Circle(CircularPatch, FrameSection, ModelComponent):
    def __init__(self, **kwds):
        FrameSection.__init__(self, **take_keywords(kwds, "units","domain","parent"))
        CircularPatch.__init__(self, (0.0, 0.0), **kwds)

    def rein(self, typ, **kwds):
        pass

