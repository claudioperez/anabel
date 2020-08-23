from scipy.optimize import fsolve 
import numpy as np

def return_map(E,H0,C,AD=False,H=None):
    if H is None:
        H = lambda Δep: H0

    def g(Δep, sigma_trial, q_i, Y_i):
        return abs(sigma_trial - q_i) - (E + C)*Δep - Y_i - H( sigma_trial - (E + C)*Δep )*Δep
    
    if AD:
        Δg = jax.grad(g)
    else:
        Δg = None
        
    def solve_g(sigma_trial, q_i, Y_i):
        return fsolve(g, 0, args=(sigma_trial, q_i, Y_i))[0]
 
    def step(Δe, e_i, ep_i, sigma_i, q_i, Y_i):  # Δe, e_i, sigma_i, q_i, Y_i, ep_i 
        e_n = e_i + Δe                           # New total strain at the end of this step
        sigma_trial = sigma_i + E * Δe           # Trial stress
        xi_trial = sigma_trial  - q_i
        f_trial = abs(xi_trial) - Y_i

        if f_trial < 0: # Elastic step
            sigma_n = sigma_trial
            Y_n = Y_i
            ep_n = ep_i
            q_n = q_i
            
        else: 
            Δep = solve_g(sigma_trial, q_i, Y_i)

            # Update ep_n, sigma_n, q_n, Y_n
            ep_n = ep_i + Δep * np.sign(xi_trial)
            sigma_n = E*(e_n - ep_n)
            Y_n =  Y_i + H(Y_i) * Δep

            q_n = q_i + C*Δep*np.sign(xi_trial)

        return e_n, ep_n, sigma_n, q_n, Y_n 
    return step