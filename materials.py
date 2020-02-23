import numpy as np 

from ema.objects import Material


class Material1D(Material):
    pass 

class Material2D:
    def __init__(self,E,nu,G=None, eps0=None, sig0=None, Case=None, irs=None):
        self.elastic_modulus = self.E = E

        if G is None:
            self.shear_modulus = self.G = E/2/(nu+1)
        else:
            self.shear_modulus = self.G = G
        if nu is None:    self.nu = 0.0
        if eps0 is None:  self.eps0 = np.zeros((4,1))
        if sig0 is None:  self.sig0 = np.zeros((4,1))
        if Case is None:  self.Case = 'stress'
        if ir is None:
            self.irs = np.array([1, 2, 4])
            self.ics = 3
        else:
            self.ics = setdiff(1:4,MatData.irs)
            self.eps0 = eps0 # initial strain tensor in 4x1 array form in the order 11, 22, 33, 12
            self.sig0 = sig0 # initial stress tensor in 4x1 array form in the order 11, 22, 33, 12
            self.Case = Case # 'stress' or 'strain'
            self.irs  = irs  # stress or strain components to be retained depending on Case
        pass 

class AxisyMat(Material2D):
    pass 

class PlaneStrainMat(Material2D):
    pass 

class PlaneStressMat(Material2D):
    pass