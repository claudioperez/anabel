import collections
import jax
from jax import lax
import jax.numpy as np
from scipy.optimize import fsolve
from functools import partial
# from .roots import path_solve

def return_map(E, H0, C, e0=0.0, ep0=0.0, sigma0=0.0, q0=0.0, Y0=0.0, AD=False, H=None,init=False,linearize=False, **kwds):
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
            ep_n = ep_i + Dep * np.sign(xi_trial)
            sigma_n = E*(e_n - ep_n)
            Y_n =  Y_i + H(Y_i) * Dep
            Et = E*(C + H(Y_i))/( E + C + H(Y_i) )
            q_n = q_i + C*Dep*np.sign(xi_trial)

        return e_n, ep_n, sigma_n, q_n, Y_n, Et

    # step = jax.jit(step, static_argnums=(0,2,1,3,4,5))

    if init:
        State = collections.namedtuple('State', 'e, ep, sigma, q, Y, Et', defaults=(e0, ep0, sigma0, q0, Y0, E))
        return step, State
    return step

