import warnings 
from collections import defaultdict
from functools import partial

import numpy as onp
try: 
    import jax
    import jax.numpy as np
except: 
    np=onp



def integrate(f, gradf, hessf=None, ff=None, nr=None,verbose=False, 
            stepper=None, soptions={}, iterate=None, itoptions={},
            init_solve=None, inoptions={}):

    ##########################################################################
    ## initialize strategy
    ##########################################################################

    if nr is not None: nr = -nr

    ## State Handler
    if callable(init_solve):
        pass 
    else:
        inoptions.setdefault('verbose',verbose)
        inoptions.setdefault('nr',nr)
        init_solve = _init_solve(f,gradf,**inoptions)

    ## Iterator 
    if callable(iterate): 
        pass
    else:
        itoptions.setdefault('nr',nr)
        itoptions.setdefault('verbose',verbose)
        iterate = iterate_init(f, gradf, hessf, **itoptions)

    ## Stepper
    if callable(stepper): pass
    else:
        if not soptions: # soptions is empty
            pass
        stepper = lambda state0, **opts:  [state0]
    

    ##########################################################################
    ## Solver
    ##########################################################################
    def solve(x0, params={}, ff=None):
        a = params
        x, a, state0 = init_solve(x0, a, ff)

        for step,state in enumerate(stepper(state0)):

            if verbose: print( 'step: ', step + 1)
            
            x, _ = iterate(x, a, state)

        return x

    return solve



def _init_solve(f, gradf, verbose=False,nr=None,**kwargs):

    def init_solve(x0, a={}, ff=None):
        if verbose: print('Initializing iterative solve.')
        state = {}
        x = x0

        f0 = f(x0, **a)[:nr]
        state['fi'] = f0
        if verbose: print('\nf0: ',np.around(f0,3))

        if ff is None: ff = np.zeros(f0.shape)
        elif callable(ff): ff = ff(x0)[:nr]
        if verbose: print('\nff: ',np.around(ff,3))

        state.setdefault('df',  ff-f0)
        state.setdefault('ff',     ff)
        state.setdefault('nr',     nr)
        return x, a, state
    
    return init_solve

# @jax.custom_jvp
def iterate_init(f, gradf, hessf=None, nr=None, maxiter=20,
                 tol=1e-3,loss=None,verbose=False,jit=True,**kwargs):
    if loss is None:
        loss = np.linalg.norm
    
    def update(x, a, state):
        if verbose: print('x: ', x)
        df = state['df']
        if verbose: print('df: ',df.T)

        gradfi = gradf(x,**a)[:nr,:nr]
        if verbose: print('gradf: \n',gradfi)

        dx = np.linalg.solve(gradfi, df)[:,0]

        if nr is not None:
            dx = np.pad(dx, (0,-nr), 'constant') 
        if verbose: print('dx: ',dx.T)
        x = x + dx

        state['fi'] = f(x, **a)[:nr]
        # state['df'] = df
        return x, state

    if jit:
        try:
            update = jax.jit(update)
            print('Iteration updater jit compilation successful.')
        except:
            pass
    
    def iterate(x0, a, state):
        x = x0
        for itr in range(maxiter):
            if verbose: print('iteration: ',itr)
            state['df'] = state['ff'] - state['fi']
            
            if loss(state['df']) <= tol: 
                return x, state

            x, state = update(x, a, state)

        if verbose:
            msg = ("Failed to converge after %d iterations, value is %s."
                % (itr + 1, loss(state['df'])))
            raise RuntimeError(msg)
        

    return iterate
