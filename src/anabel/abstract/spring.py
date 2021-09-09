#from .component import ModelComponent
from .material import Material

class Spring(Material):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
    pass


