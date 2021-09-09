from anabel.abstract import Spring

class ElasticSpring(Spring):
    def __init__(self, E, G=None, **kwds):
        super().__init__(**kwds)
        self.E = self.elastic_modulus = E
        self.G = self.shear_modulus = G

    def dump_opensees(self, *args, **kwds):
        E = self.elastic_modulus if isinstance(self.elastic_modulus, float) \
            else f"${self.elastic_modulus}"
        return f"uniaxialMaterial Elastic {self.tag} {E}\n"

