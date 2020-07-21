import warnings 
from collections import defaultdict
from functools import partial

import numpy as onp
try: 
    import jax
    import jax.numpy as np
except: 
    np=onp
    warnings.warn('Jax import failed.')



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


def path_solve(f, gradf, ff=None, nr=None, verbose=False, nosteps=1,
                soptions={}, itoptions={},svoptions={},inoptions={}):

    if nr is not None: nr = -nr

    ## State Handler
    init = _init_init_path(f,gradf,verbose=verbose,**itoptions)

    ## Iterator 
    iterate = _init_iter_path(f, gradf,verbose=verbose, **itoptions)

    ## Stepper
    step = lambda state0, **opts:  [state0]
    
    step = _init_incr_path(nosteps,verbose=verbose)


    ## Solver
    solve = _init_solv_path(init,step,iterate,verbose=verbose)

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

    def init_init(x0, a={}, ff=None):
        if verbose: print('Initializing iterative solve.')
        _state = {}
        x = x0

        f0 = f(x0, **a)[:nr]
        _state['fi'] = f0
        if verbose: print('\nf0: ',np.around(f0,3))

        if ff is None: ff = np.zeros(f0.shape)
        elif callable(ff): ff = ff(x0)[:nr]
        if verbose: print('\nff: ',np.around(ff,3))

        _state.setdefault('df',  ff-f0)
        _state.setdefault('ff',     ff)
        _state.setdefault('nr',     nr)
        return x, a, _state
    
    return init_init

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
        update = jax.jit(update)
        print('Iteration updater jit compilation successful.')
    
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



def _init_init_path(f, gradf, verbose=False,nr=None,**kwargs):

    def init_solve(x0, a=(), ff=None):
        if verbose: print('Initializing iterative solve.')
        state = {}
        x = x0

        f0, a = f(x0, *a)
        f0 = f0[:nr]
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

def _init_iter_path(f, gradf, hessf=None, nr=None, maxiter=20,
                    tol=1e-3,loss=None,verbose=False,jit=True,**kwargs):
    
    if loss is None:
        loss = np.linalg.norm
    
    def update(x, a, state):
        if verbose: print('x: ', x)
        df = state['df']
        if verbose: print('df: ',df.T)

        gradfi, _ = gradf(x,*a)
        gradfi = gradfi[:nr,:nr]
        if verbose: print('gradf: \n',gradfi)

        dx = np.linalg.solve(gradfi, df)

        if nr is not None:
            dx = np.pad(dx, (0,-nr), 'constant')

        if verbose: print('dx: ',dx.T)
        x = x + dx

        fi, a = f(x, *a)

        state['fi'] = fi[:nr]
        return x, a, state

    # if jit:
    #     try:
    #         update = jax.jit(update)
    #     except:
    #         pass
    
    def iterate(x0, a, state):
        x = x0
        for itr in range(maxiter):
            if verbose: print('iteration: ',itr)
            state['df'] = state['ff'] - state['fi']

            if loss(state['df']) <= tol: 
                return x, a, state

            x, a, state = update(x, a, state)

        msg = ("Failed to converge after %d iterations, value is %s."
            % (itr + 1, loss(state['df'])))
        # raise RuntimeError(msg)
        warnings.warn(msg)
        return x, a, state
        

    return iterate

def _init_incr_path(_,**kwds):

    def step(state0,nosteps):
        ff = state0['ff']
        df = state0['df']
        state = state0.copy()

        for i in range(1,abs(int(nosteps))+1):
            # state.update(ff=i*ff/nosteps)
            # state.update(df=i*df/nosteps)
            yield state


    return step

def _init_solv_path(init,incr,iterate,verbose=False):

    def solve(x0, params={}, ff=None,nostep=1):
        a = params
        x, a, state0 = init(x0, a, ff)

        for i,state in enumerate(incr(state0,nostep)):

            if verbose: 
                print( 'step: ', i + 1)
                # print( 'ff : ', state['ff'])
                # print( 'df : ', state['df'])
            # print(state['ff'], x)
            x, a , _ = iterate(x, a, state)
            # print(a)

        return x, a

    return solve

