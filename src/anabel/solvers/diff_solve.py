import warnings 
from collections import defaultdict
from functools import partial

import anabel.ops as anp

def diff_solve(f, gradf, hessf=None, ff=None, nr=None,verbose=False, 
            step=None, soptions={}, iterate=None, itoptions={},
            solve=None, svoptions={},init=None, inoptions={}):

    ##########################################################################
    ## initialize strategy
    ##########################################################################

    if nr is not None: nr = -nr

    ## State Handler
    if callable(init):
        pass 
    else:
        inoptions.setdefault('verbose',verbose)
        inoptions.setdefault('nr',nr)
        init = _init_init_basic(f,gradf,**inoptions)

    ## Iterator 
    if callable(iterate): 
        pass
    else:
        itoptions.setdefault('nr',nr)
        itoptions.setdefault('verbose',verbose)
        iterate = iterate_init(f, gradf, hessf, **itoptions)

    ## Stepper
    if callable(step): pass
    else:
        if not soptions: # soptions is empty
            pass
        step = lambda state0, **opts:  [state0]

    ## Solver
    if callable(solve): pass
    else:
        if not soptions: # soptions is empty
            pass
        solve = init_solve_basic(init,step,iterate,False)

    ##########################################################################
    ## Solver
    ##########################################################################
    # def solve(x0, params={}, ff=None):
    #     a = params
    #     x, a, state0 = init_solve(x0, a, ff)

    #     for step,state in enumerate(stepper(state0)):

    #         if verbose: print( 'step: ', step + 1)

    #         x, _ , _ = iterate(x, a, state)

    #     return x
    # svoptions = {}
    # _solve = solve(init, step, iterate,**svoptions)

    return solve

def init_solve_basic(init,incr,iterate,verbose=False):

    def solve(x0, params={}, ff=None):
        a = params
        x, a, state0 = init(x0, a, ff)

        for i,state in enumerate(incr(state0)):

            if verbose: print( 'step: ', i + 1)
            
            x, _ , _ = iterate(x, a, state)

        return x

    return solve


def _init_init_basic(f, gradf, verbose=False,nr=None,**kwargs):

    def init(x0, a={}, ff=None):
        if verbose: print('Initializing iterative solve.')
        state = {}
        x = x0

        f0 = f(x0, **a)[:nr]
        state['fi'] = f0
        if verbose: print('\nf0: ',anp.around(f0,3))

        if ff is None: ff = anp.zeros(f0.shape)
        elif callable(ff): ff = ff(x0)[:nr]
        if verbose: print('\nff: ',anp.around(ff,3))

        state.setdefault('df',  ff-f0)
        state.setdefault('ff',     ff)
        state.setdefault('nr',     nr)
        return x, a, state
    
    return init

def iterate_init(f, gradf, hessf=None, nr=None, maxiter=20,
                 tol=1e-3,loss=None,verbose=False,jit=True,**kwargs):
    
    if loss is None:
        loss = anp.linalg.norm
    
    def update(x, a, state):
        if verbose: print('x: ', x)
        df = state['df']
        if verbose: print('df: ',df.T)

        gradfi = gradf(x,**a)[:nr,:nr]
        if verbose: print('gradf: \n',gradfi)

        dx = anp.linalg.solve(gradfi, df)[:,0]

        if nr is not None:
            dx = anp.pad(dx, (0,-nr), 'constant') 
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
                return x, None, state

            x,  state = update(x, a, state)

        if verbose:
            msg = ("Failed to converge after %d iterations, value is %s."
                % (itr + 1, loss(state['df'])))
            raise RuntimeError(msg)
        
    return iterate

