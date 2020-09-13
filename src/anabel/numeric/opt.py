import numpy as onp
import warnings
from collections import defaultdict

import scipy.optimize

# import anabel.ops as anp
anp = onp

import jax
# try:
#     import jax
#     import jax.numpy as np
# except:
#     np=onp
    
def _haukaas06(f):
    c1 = lambda state: abs(state['Gi'][0]/state['G0'])
    def c2(state):
        x = state['ui'][0]
        normx = jax.lax.max(1.0, f(x))
        return f(x/normx - (state['alphai']@x)/normx*state['alphai'].T)
    # return [c1,c2]
    return jax.jit(c1), jax.jit(c2)

_CRITERIA = {
    'default': lambda f: [
        lambda state: abs(state['Gi'][0]/state['G0']),
        lambda state: f(state['ui'][0] - (state['alphai']@state['ui'][0])*state['alphai'].T)],
    'haukaas-06':lambda f: _haukaas06(f),
}

def minimize(f, gradf, g, grad_G, u0, hess_g = None, args=(), loss=None, 
                callback=None, tol1=1.0e-4, tol2=1.0e-4, maxiter=20,
                verbose=False, soptions={}, doptions={}, ioptions={},
                step = None, dirn=None, incr=None, criteria='default',**kwargs):
    """Constrained minimization.
    
    Parameters.
    - f
        Minimized function
        
    - gradf
        **Unimplemented** - Gradient of minimized function
        
    - g
        Constraint function.
        
    - grad_G
        Gradient of constraint function.
        
    - u0
        Initial value.

    - hess_g = None
        **Unimplemented** - Hessian of constraint function
        
    - args=()
        
    - loss=None
        **Untested** - provide a different loss function
        
    - callback=None
        placeholder
        
    - tol1=1.0e-4
        Tolerance of 
        
    - tol2=1.0e-4
        
    - maxiter=20
        Maximum number of iterations.
        
    - verbose=False
        Flag for procedure verbosity. If `True`, intermediate results are printed to the console.
        
    - soptions={}
        Dictionary of options for stepper function.
        
    - doptions={}
        Dictionary of options for direction function.
        
    - ioptions={}
        Dictionary of options for incrementation function.
        
    - step = None
        String option or callable (**Untested**) indicating stepping function.
        
    - dirn=None
        String option or callable (**Untested**) indicating direction function.
    
    - incr=None
        String option or callable (**Untested**) indicating incrementing function.
        
    - **kwargs
        
    - criteria=None
        String indicating which set of convergence criteria functions.
        - `default`:
        - 'haukaas-06':
 
    """
    
    ndim = len(u0)
    ##########################################################################
    ## initialize strategy
    ##########################################################################
    
    criteria = _CRITERIA[criteria](f)
    state = {}
    
    
    if callable(dirn):pass
    elif dirn in {'dirn_init'}:
        pass
    else: # set default direction
        if not doptions: # direction func options is empty, set defaults
            doptions.setdefault('verbose', verbose)
            
        dirn = dirn_init(**doptions)
            
    if callable(incr):pass
    if incr =='minimize':
        ioptions.setdefault('c0',      10.)
        ioptions.setdefault('verbose', verbose)
        incr = _init_minimize_step(f, g, grad_G, **ioptions)
    if incr=='armijo':
        incr = incr_init_armijo(f, g, grad_G, **ioptions)
        
    if incr=='armijo-dkh':
        state.setdefault('mi',0)
        incr = incr_init_armijo_dkh(f, g, grad_G, **ioptions)
    else:
        ioptions.setdefault('c0',      10.)
        ioptions.setdefault('maxiter',  20)
        ioptions.setdefault('lamda0',  1.0)
        ioptions.setdefault('tol1',   tol1)
        ioptions.setdefault('tol2',   tol2)
        ioptions.setdefault('verbose', verbose)
            
        incr = incr_init_basic(f, g, grad_G, **ioptions)
        
    if callable(step): pass
    else:
        if not soptions: # soptions is empty
            soptions.setdefault('verbose', verbose)
            
        step = step_init(f, g, grad_G, dirn, incr, **soptions)
        
        
    ##########################################################################
    ## Initialize state
    ##########################################################################
    state.setdefault('G0',     0.0)
    state.setdefault('Gi',     [None, None])
    state.setdefault('ui',     [anp.zeros(ndim)]*2)
    state.setdefault('di',     anp.zeros((ndim,1)))
    state.setdefault('res',    [1.0, 0.0])
    state.setdefault('gradGi', anp.zeros(ndim))
    state.setdefault('alphai', anp.zeros((1,ndim)))
    state.setdefault('lamdai', 1.0)
    state.setdefault('mi',     [1.0, 1.0])
    state.setdefault('niter',  0.0)
    state.setdefault('siter',  [])
    state.setdefault('step',   {})

    if verbose:
        print('\niHL-RF Algorithm (Zhang and Der Kiureghian 1995)***************',
              '\n-------------------------------------------------------------------------'
              '\nInitialize iHL-RF: \n',
              'u0: ', anp.around(u0.T,4), '\n')

    state['G0'] = g(u0) # scale parameter
    state['Gi'][0] = state['G0']
    if verbose: print(' G0: ', anp.around(state['G0'],4),'\n')

    state['ui'][0] = u0[None,:].T

    state['GradGi'] = grad_G(state['ui'][0][:,0].T)
    if verbose: print(' GradGi: ', state['GradGi'], '\n',)

    state['alphai'] = -(state['GradGi'] / anp.linalg.norm(state['GradGi']))[None,:]
    state['niter']  = 0
    state['res'][0] = 1.0 # abs(state['Gi'][0] / G0)
    state['res'][1] = f( state['ui'][0] - (state['alphai']@state['ui'][0])*state['alphai'].T )

    if verbose:
        print('alphai: ', anp.around(state['alphai'],4),'\n',
              'res1: ' , anp.around(state['res'][0],4),'\n',
              'res2: ' , state['res'][1],'\n',)

    ##########################################################################
    ## Run
    ##########################################################################
    def run(state,step):
        
        while not(state['res'][0] < tol1 and state['res'][1] < tol2):
            state = step(state)
            
            state['res'][0] = criteria[0](state)
            state['res'][1] = criteria[1](state)
            state['niter'] += 1
            
            if (state['niter'] > maxiter):
                msg = 'Warning: Optimization failed to converge after {} iterations.'.format(state['niter']) 
                warnings.warn(msg)
                break
        return state
    
    return run(state,step)


def incr_init_armijo(f, g, grad_G, c0=10., a=0.5, b=0.5, lamda0=1.0, maxiter=5, verbose=False,**kwargs):
    if verbose: print('c: ', c0)
    
    def m(y,ci): return 0.5*f(y)**2 + ci*abs(g(y[:,0].T))
    def grad_m(y,gi,ci): return y[:,0].T + ci*anp.sign(gi)*grad_G(y[:,0].T)
    
    def incr(state):
        k = 0
        ci = c0
        lamda0 = b**k
        y0 = state['ui'][0]
        gi = g(y0[:,0].T)
        y1 = y0 + lamda0*state['di']
        state['mi'][0] = m(y0,ci)
        state['mi'][1] = m(y1,ci)
        state['siter'].append(0)
        state['lamdai'] = lamda0
        print('**********',grad_m(y1,gi,ci)@state['di'])
        while (state['mi'][1] > state['mi'][0] - a*state['lamdai']*grad_m(y1,gi,ci)@state['di']) :
            if verbose: print('Armijo iteration: ', state['siter'][state['niter']])
            k +=1
            state['lamdai'] = b**k
            state['step']['ci'] = f(state['ui'][0]) / anp.linalg.norm(state['GradGi']) + c0
            if verbose: print(' lamdai: ', state['lamdai'])
            y1 = y0 + state['lamdai'] * state['di']
            if verbose: print('ui1: ',y1.T)
            state['mi'][1] = m(y1,ci)
            
            state['siter'][state['niter']] += 1
            if (state['siter'][state['niter']] >= maxiter): 
                msg = 'Warning: Increment size at iteration {} failed to converge after {} iterations'.format( state['niter'], state['siter'][state['niter']])
                warnings.warn(msg)
                break
 
        state['ui'][1] = y1
        return state
    return incr

def incr_init_armijo_dkh(f, g, grad_G, c0=10., a=0.5, b0=0.5, mi=3,b=0.5, lamda0=2.0, maxiter=5, verbose=False,**kwargs):
    if verbose: print('c: ', c0)
    
    def m(y,ci): return 0.5*f(y)**2 + ci*abs(g(y[:,0].T))
    def grad_m(y,gi,ci): return y[:,0].T + ci*anp.sign(gi)*grad_G(y[:,0].T)
    
    def incr(state):
        k = 0
        ci = c0
        b0i = [b0]
        lamda0 = b0i(b**k)
        y0 = state['ui'][0]
        gi = g(y0[:,0].T)
        y1 = y0 + lamda0*state['di']
        state['mi'][0] = m(y0,ci)
        state['mi'][1] = m(y1,ci)
        state['siter'].append(0)
        state['lamdai'] = lamda0
        print('**********',grad_m(y1,gi,ci)@state['di'])
        while (state['mi'][1] > state['mi'][0] - a*state['lamdai']*grad_m(y1,gi,ci)@state['di']) :
            if verbose: print('Armijo iteration: ', state['siter'][state['niter']])
            k +=1
            state['lamdai'] = b**k
            state['step']['ci'] = f(state['ui'][0]) / anp.linalg.norm(state['GradGi']) + c0
            if verbose: print(' lamdai: ', state['lamdai'])
            y1 = y0 + state['lamdai'] * state['di']
            if verbose: print('ui1: ',y1.T)
            state['mi'][1] = m(y1,ci)
            
            state['siter'][state['niter']] += 1
            if (state['siter'][state['niter']] >= maxiter): 
                msg = 'Warning: Increment size at iteration {} failed to converge after {} iterations'.format( state['niter'], state['siter'][state['niter']])
                warnings.warn(msg)
                break
                
        state['ui'][1] = y1
        return state
    return incr

def incr_init_basic(f, g, grad_G, c0=10., lamda0=1.0, maxiter=15, verbose=False,**kwargs):
    if verbose: print('c: ', c0)
    def incr(state):
        state['step']['ci'] = f(state['ui'][0]) / anp.linalg.norm(state['GradGi']) + c0
        if verbose: print(' ci: ',state['step']['ci'])
        state['mi'][0] = 0.5*f(state['ui'][0])**2 + state['step']['ci']*abs(state['Gi'][0])
        state['mi'][1] = 0.5*f(state['ui'][1])**2 + state['step']['ci']*abs(state['Gi'][1])
        state['siter'].append(0)
        state['lamdai'] = lamda0
        
        while (state['mi'][1] >= state['mi'][0]) :
            state['lamdai'] = 0.5*state['lamdai']
            if verbose: print(' lamda:', state['lamdai'])
            state['ui'][1] = state['ui'][0] + state['lamdai'] * state['di']
            if verbose: print('ui1: ',state['ui'][1].T)
            state['Gi'][1] = g(state['ui'][1][:,0].T)
            state['mi'][1] = 0.5*anp.linalg.norm(state['ui'][1])**2 + state['step']['ci']*abs(state['Gi'][1])
            state['siter'][state['niter']] += 1
            if (state['siter'][state['niter']] >= maxiter): 
                msg = 'Warning: Increment size at iteration {} failed to converge after {} iterations'.format( state['niter'], state['siter'][state['niter']])
                warnings.warn(msg)
                break
        return state
    return incr

def dirn_init(verbose=False,**kwargs):
    
    def dirn(state):
        state['di'] = (state['Gi'][0]/anp.linalg.norm(state['GradGi']) + state['alphai']@state['ui'][0])*state['alphai'].T - state['ui'][0]
        # if verbose: print(' di: ',state['di'].T)
        return state
    
    return dirn

def step_init(f, g, grad_G,dirn,incr,loss=None,verbose=False,**kwargs):
    
    def step(state):
        G0 = state['G0']
        state = dirn(state)

        state['ui'][1] = state['ui'][0] + state['lamdai'] * state['di']
        # if verbose: print(' ui1: ',anp.around(state['ui'][1][:,0].T,4))

        state['Gi'][1] = g(state['ui'][1][:,0].T)
        # if verbose: print(f' Gi[1]: {state["Gi"][1]}')
        state = incr(state)

        state['ui'][0] = state['ui'][1]  
        state['Gi'][0] = g(state['ui'][0][:,0].T)
        state['GradGi'] = grad_G(state['ui'][0][:,0].T)
        state['alphai'] = -(state['GradGi'] / anp.linalg.norm(state['GradGi']))[None,:]

        if verbose:
            print('\niHL-RF step: {}'.format(state['niter']))
            print(' ui: ',     anp.around(state['ui'][0].T,4), '\n',
                  'Gi: ',      anp.around(state['Gi'][0],  4), '\n',
                  'GradGi: ',  anp.around(state['GradGi'], 4), '\n',
                  'alphai: ',  anp.around(state['alphai'], 4), '\n',
                  'res1: ' ,   anp.around(state['res'][0], 4), '\n',
                  'res2: ' ,   anp.around(state['res'][1], 4), '\n',)
        return state

    return step
