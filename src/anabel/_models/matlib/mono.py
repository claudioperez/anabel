import jax
import jax.numpy as jnp
from jax import lax
import collections


def bi_lin_pos_state(E, H0, C, e0=0.0, ep0=0.0, sigma0=0.0, q0=0.0, Y0=0.0, AD=False, H=None,init=False,linearize=False, **kwds):
    if H is None: H = lambda Dep: H0

    def g(Dep, sigma_trial, q_i, Y_i):
        return abs(sigma_trial - q_i) - (E + C)*Dep - Y_i - H( sigma_trial - (E + C)*Dep )*Dep
        
    if linearize:
        def solve_g(sigma_trial, q_i, Y_i):
            return (abs(sigma_trial-q_i) - Y_i )/(E+C+H0)
    else:
        def solve_g(sigma_trial, q_i, Y_i):
            return fsolve(g, 0., args=(sigma_trial, q_i, Y_i))[0]
    
    def step(e_i, ep_i, sigma_i, q_i, Y_i, *, De, **args):  # De, e_i, sigma_i, q_i, Y_i, ep_i 
        e_n = e_i + De                    # New total strain at the end of this step
        sigma_trial = sigma_i + E * De    # Trial stress
        xi_trial = sigma_trial  - q_i
        f_trial = abs(xi_trial) - Y_i

        if f_trial < 0.0: # Elastic step
            sigma_n = sigma_trial
            Et = E
            Y_n  = Y_i
            ep_n = ep_i
            q_n  = q_i
            
        else:
            Dep = solve_g(sigma_trial, q_i, Y_i)
            ep_n = ep_i + Dep * jnp.sign(xi_trial)
            sigma_n = E*(e_n - ep_n)
            Y_n =  Y_i + H(Y_i) * Dep
            Et = E*(C + H(Y_i))/( E + C + H(Y_i) )
            q_n = q_i + C*Dep*jnp.sign(xi_trial)

        return e_n, ep_n, sigma_n, q_n, Y_n, Et
    
    def Df (de, st):
        if f_trial < 0.0: # Elastic step
            Df = E
        else:
            Df = E*(C + H(Y_i))/( E + C + H(Y_i) )
        return Df, st

    if init:
        State = collections.namedtuple('State', 'e, ep, sigma, q, Y, Et', defaults=(e0, ep0, sigma0, q0, Y0, E))
        return step, State
    return step


def bi_lin(E, H0, C, e0=0.0, ep0=0.0, sigma0=0.0, q0=0.0, Y0=0.0, AD=False, H=None,init=False,linearize=False, **kwds):
    if H is None: H = lambda Dep: H0

    def g(Dep, sigma_trial, q_i, Y_i):
        return abs(sigma_trial - q_i) - (E + C)*Dep - Y_i - H( sigma_trial - (E + C)*Dep )*Dep
        
    if linearize:
        def solve_g(sigma_trial, q_i, Y_i):
            return (abs(sigma_trial-q_i) - Y_i )/(E+C+H0)
    else:
        def solve_g(sigma_trial, q_i, Y_i):
            return fsolve(g, 0., args=(sigma_trial, q_i, Y_i))[0]
    
    def step(De, e_i, ep_i, sigma_i, q_i, Y_i,*args):  # De, e_i, sigma_i, q_i, Y_i, ep_i 
        e_n = e_i + De                    # New total strain at the end of this step
        sigma_trial = sigma_i + E * De    # Trial stress
        xi_trial = sigma_trial  - q_i
        f_trial = abs(xi_trial) - Y_i

        if f_trial < 0.0: # Elastic step
            sigma_n = sigma_trial
            Et = E
            Y_n  = Y_i
            ep_n = ep_i
            q_n  = q_i
            
        else:
            Dep = solve_g(sigma_trial, q_i, Y_i)
            ep_n = ep_i + Dep * jnp.sign(xi_trial)
            sigma_n = E*(e_n - ep_n)
            Y_n =  Y_i + H(Y_i) * Dep
            Et = E*(C + H(Y_i))/( E + C + H(Y_i) )
            q_n = q_i + C*Dep*jnp.sign(xi_trial)

        return e_n, ep_n, sigma_n, q_n, Y_n, Et
    
    def Df (de, st):
        if f_trial < 0.0: # Elastic step
            Df = E
        else:
            Df = E*(C + H(Y_i))/( E + C + H(Y_i) )
        return Df, st

    if init:
        State = collections.namedtuple('State', 'e, ep, sigma, q, Y, Et', defaults=(e0, ep0, sigma0, q0, Y0, E))
        return step, State
    return step

def tri_lin(Y_1,Y_2,e_1,e_2, Hr2=1e-6,init=True, **kwds):

    E = Y_1/e_1 
    E1 = (Y_2-Y_1)/(e_2-e_1)
    E2 = Hr2*E
    Ei = [E,E1,E2]

    def s1(e_n): return E*e_n 
    def s2(e_n): return (Y_1-E1*e_1) + E1*e_n 
    def s3(e_n): return (Y_2-E2*e_2) + E2*e_n 
    
    def step(De, e_i, ep_i, sigma_i, q_i, Y_i,*args):  # De, e_i, sigma_i, q_i, Y_i, ep_i 
        e_n = e_i + De                    # New total strain at the end of this step
        sgn = jnp.sign(e_n)
        ea = abs(e_n)
        sigma_trial = [s1(ea),s2(ea),s3(ea)]
        sigma_n = min(sigma_trial)
        Et = Ei[sigma_trial.index(sigma_n)]
        sigma_n *= sgn
        ep_n = ep_i ; q_n = q_i ; Y_n = Y_i

        return e_n, ep_n, sigma_n, q_n, Y_n, Et

    if init:
        State = collections.namedtuple('State', 'e, ep, sigma, q, Y, Et', defaults=(0.0, 0.0, 0.0, None, None, E))
        return step, State
    return step

def gmp_std(fy,E,Eh,r,init=True, **kwds):

    # E = Y_1/e_1 
    # E1 = (Y_2-Y_1)/(e_2-e_1)
    # E2 = Hr2*E
    # Ei = [E,E1,E2]
    
    @jax.jit
    def _sig(eps):
        xi = eps/(fy/E)
        b = Eh/E
        return fy*(xi*b+(1-b)*xi/(1+abs(xi)**r)**(1/r))
    @jax.jit
    def _Et(eps):
        xi = eps/(fy/E)
        b = Eh/E
        return E*(b+(1-b)/(1+abs(xi)**r)**(1+1/r))
    
    def step(De, e_i, ep_i, sigma_i, q_i, Y_i,*args):  # De, e_i, sigma_i, q_i, Y_i, ep_i 
        e_n = e_i + De                    # New total strain at the end of this step
        sigma_n = _sig(e_n)
        Et = _Et(e_n)
        ep_n = ep_i ; q_n = q_i ; Y_n = Y_i

        return e_n, ep_n, sigma_n, q_n, Y_n, Et

    if init:
        State = collections.namedtuple('State', 'e, ep, sigma, q, Y, Et', defaults=(0.0, 0.0, 0.0, None, None, E))
        return step, State
    return step


def GMP(fy,E,Eh,r):
    @jax.jit
    def sig(eps):
        xi = eps/(fy/E)
        b = Eh/E
        return fy*(xi*b+(1-b)*xi/(1+abs(xi)**r)**(1/r))

    @jax.jit
    def Et(eps):
        xi = eps/(fy/E)
        b = Eh/E
        return E*(b+(1-b)/(1+abs(xi)**r)**(1+1/r))
    
    return sig, Et

def GMP2(fy,E,Eh,r,a1,a2,state=False):
    ey = fy/E
    b = Eh/E
    # sig, sig_i, e_i, e_max

    def R(e_max):
        return r-(a1+e_max)/(a2+e_max)


    def sig(e, sig_i, e_i, e_max):

        if e_i is None:
            xi = e/(fy/E)
            sig_n = fy*(xi*b+(1-b)*xi/(1+abs(xi)**R(e_max))**(1/R(e_max)))
        else:
            xi = (e-e_i)/(2*ey)
            sig_n = sig_i + 2*fy*(xi*b+(1-b)*xi/(1+abs(xi)**R(e_max))**(1/R(e_max)))
        e_max = max(e, e_max)

        return sig_n, sig_i, e_i, e_max

    def Et(e, sig_i, e_i, e_max):
        xi = eps/(fy/E)
        return E*(b+(1-b)/(1+abs(xi)**r)**(1+1/r))
    
    if state:
        State = collections.namedtuple('State', 'e, sig_i, e_i, e_max', defaults=(0, ep0, sigma0, q0, Y0))
        return step, State
    return sig, Et

def BiLin(epsy=None,E=None,fy=None,Eh=0.0,r=None,activ=None, MatData=True):
    if epsy is None: epsy = fy/E 
    if fy is None: fy = epsy*E
    epspmax = 1.0

    # @jax.jit
    def sig(eps):
        ex = lax.clamp(-epsy,eps,epsy)
        sgn = lax.sign(ex)
        exp = lax.clamp(-1.0, (eps-ex)*sgn, 1.0)*sgn
        return ex*E + exp*Eh

    # @jax.jit
    def Et(eps):
        return (E,Eh)[int(jax.lax.ge(abs(eps),epsy))]

    if MatData:
        MatData = {'sig':sig,'Et':Et,'epsy':epsy,'fy':fy}
        return sig, Et, MatData

    return sig, Et


def BiLin3(epsy=None,E=None,fy=None,Eh=0.0,r=None,activ=None, MatData=True,emax=1.0):
    if epsy is None: epsy = fy/E 
    if fy is None: fy = epsy*E

    @jax.jit
    def sig(eps,**kwds):
        ex = lax.clamp(-epsy,eps,epsy)
        sgn = lax.sign(ex)
        exp = lax.clamp(-1.0, (eps-ex)*sgn, 1.0)*sgn
        return ex*E + exp*Eh

    # @jax.jit
    # Et = jax.grad(sig)
    def Et(eps,e_i,**kwds):
        if jnp.sign(eps-e_i)==jnp.sign(e_i):
            return (E,Eh)[int(jax.lax.ge(abs(eps),epsy))]
        else: return E

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

