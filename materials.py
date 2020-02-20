import numpy as np 

from ema.objects import Material


class Material1D(Material):
    pass 

class Material2D:
    def __init__(self,E,nu,G=None):
        self.elastic_modulus = self.E = E 
        self.poisson_ratio = self.nu = nu 
        if G is None:
            self.shear_modulus = self.G = E/2/(nu+1)
        else:
            self.shear_modulus = self.G = G
    pass 

class AxisyMat(Material2D):
    pass 

class PlaneStrainMat(Material2D):
    pass 

class PlaneStressMat(Material2D):
    pass