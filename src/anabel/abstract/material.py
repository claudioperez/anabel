from .component import ModelComponent

class Material(ModelComponent):
    def __init__(self, tag=None,  nu=None, weight=None, mass=None, units=None):
        self._tag = tag
        #self.E: float = E
        #self.elastic_modulus: float = E
        self.poisson_ratio = nu
        self._mass = mass
        self._weight = weight
        self._units = units

    @property
    def dump_opensees(self, **kwds):
        if type(self) == Material:
            return ""
        else:
            super().dump_opensees(**kwds)

    @property
    def mass(self):
        if not self._mass and self._weight:
            return self._weight / self.units.gravity


