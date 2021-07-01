import abc

from anabel.utilities import take_keywords
from anabel.abstract import ModelComponent

class FrameSection(ModelComponent):
    children = ["Material", "SectionFiber"]

    def __init__(self, material=None, **kwds):
        ModelComponent.__init__(self, **kwds)
        self._material = material
        self._materials = []

    def keys(self):
        return self.properties

    def __getitem__(self, key):
        return getattr(self, key)

    def index(self, item):
        pass

class SectionFiber:
    def __init__(self, **kwds):
        Section.__init__(self, **kwds)

class SectionLayer:
    def __init__(self, **kwds):
        Section.__init__(self, **kwds)

class SectionPatch(FrameSection):
    def __init__(self, **kwds):
        FrameSection.__init__(self, **kwds)

    @property
    def tag(self):
        if super().tag is None and self.parent:
            return self.parent.patches.index(self)
        else:
            return super().tag

    def rein(self, locations, material=None):
        if isinstance(locations, (float, int)):
            pass
        else:
            pass

    @property
    @abc.abstractmethod
    def Ix(self): pass

    @property
    @abc.abstractmethod
    def Iy(self): pass

    @property
    def J(self):
        return self.Ix + self.Iy

class VerticalSection:
    pass


