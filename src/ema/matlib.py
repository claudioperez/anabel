import numpy as np


def GMP(fy,E,Eh,r):
    # @jax.jit
    def sig(eps):
        xi = eps/(fy/E)
        b = Eh/E
        return fy*(xi*b+(1-b)*xi/(1+abs(xi)**r)**(1/r))

    # @jax.jit
    def Et(eps):
        xi = eps/(fy/E)
        b = Eh/E
        return E*(b+(1-b)/(1+abs(xi)**r)**(1+1/r))
    
    return sig, Et

def BiLinOld(epsy=None,E=None,fy=None,Eh=0.0,r=None,activ=None, MatData=True):
    if epsy is None: epsy = fy/E 
    if fy is None: fy = epsy*E
    epspmax = 1.0

    # @jax.jit
    def sig(eps):
        return min(eps,epsy)*E + max(0.0, min(epspmax,eps-epsy))*Eh

    # @jax.jit
    def Et(eps):
        return (E,Eh)[int((eps>=epsy))]

    if MatData:
        MatData = {'sig':sig,'Et':Et,'epsy':epsy,'fy':fy}
        return sig, Et, MatData

    return sig, Et

def BiLin(epsy=None,E=None,fy=None,Eh=0.0,r=None,activ=None, MatData=True):
    if epsy is None: epsy = fy/E 
    if fy is None: fy = epsy*E
    epspmax = 1.0

    def sig(eps):
        ex = np.clip(eps,-epsy,epsy)
        sgn = np.sign(ex)
        exp = np.clip((eps-ex)*sgn, -1.0,  1.0)*sgn
        return ex*E + exp*Eh

    def Et(eps):
        return (E,Eh)[int(abs(eps)>=epsy)]

    if MatData:
        MatData = {'sig':sig,'Et':Et,'epsy':epsy,'fy':fy}
        return sig, Et, MatData

    return sig, Et

def Mander88(e_c, f_c, f_cc, f_emin, confined=True):
    """Returns an array of values along the stress-strain curve for a 
    specified concrete section according to the relation proposed by 
    Mander et al. in 1988.

    :param f_c: Specified 28-day concrete compression strength
    :type f_c: float
    :param f_cc: Confined concrete compressive strength
    :type f_cc: float
    :param f_emin: Lesser of the effective confinement stresses
    :type f_emin: float

    :return: a list of strain values, and a list of stress values
    :rtype: : list
    """

    C = 0.85
    epsilon_0 = 0.002
    E_c = aci.e19_2_2_1a(C*f_c)
    
    if confined:
        epsilon_cc = epsilon_0*(1+5*(f_cc/(C*f_c)-1)) # eq 4.16, page 102
        epsilon_u = 0.004+0.25*f_emin/(C*f_c)
    else:
        epsilon_cc = epsilon_0
        epsilon_u = 0.004


    # epsilon_c = np.linspace(0, epsilon_u, 100)
    epsilon_c = e_c
    r = E_c/(E_c-f_cc/epsilon_cc)
    
    x = epsilon_c/epsilon_cc
    fc = f_cc*x*r/(r-1+x**r)
    return fc

def Mander84(e_s, strainHardening=True): # page 46, Moehle 2014
    """Returns an array of values along the stress-strain curve for steel
    reinforcement according to the relation proposed by 
    Mander et al. in 1984.

    """
    Grade = 'A706'
    Gr = {
        'A706': {
            'fy' :69000,
            'fsu':95000,
            'Esh':1000000, # fig 5.9
        },
    }
    E = steel[Grade]['E']
    f_ey  = steel[Grade]['f_ey']
    f_su = Gr[Grade]['fsu']
    E_sh = Gr[Grade]['Esh']
    epsilon_0 = 0.002
    epsilon_su = 0.14

    # epsilon_s = np.linspace(0,0.023,100)
    # epsilon_sh = 2*epsilon_0 # fig 5.7
    # epsilon_sh = 10*f_ey/E
    epsilon_sh = 0.016
    if e_s < f_ey/E:
        f_s = E*e_s
    elif e_s < epsilon_sh:
        f_s = f_ey
    elif e_s > epsilon_sh:
        f_s = f_ey
    elif e_s > epsilon_sh and strainHardening == True:
        f_s = f_su+(f_ey-f_su)*((epsilon_su-e_s)/(epsilon_su-epsilon_sh))**(E_sh*((epsilon_su-epsilon_sh)/(f_su-f_ey)))
    
    return f_s

