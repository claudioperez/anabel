
class Material():
    def __init__(self, tag, E, nu=None):
        self.tag = tag
        self.E: float = E
        self.elastic_modulus: float = E
        self.poisson_ratio = nu


