import warnings
from collections import defaultdict
from functools import partial

import anabel.ops as anp

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



def _init_init_path(f, gradf, verbose=False,nr=None,**kwargs):

    def init_init(x0, a=(), ff=None):
        if verbose: print('Initializing iterative solve.')
        state = {}
        x = x0

        f0, a = f(x0, *a)
        f0 = f0[:nr]
        state['fi'] = f0
        if verbose: print('\nf0: ',anp.around(f0,3))

        if ff is None: ff = anp.zeros(f0.shape)
        elif callable(ff): ff = ff(x0)[:nr]
        if verbose: print('\nff: ',anp.around(ff,3))

        state.setdefault('df',  ff-f0)
        state.setdefault('ff',     ff)
        state.setdefault('nr',     nr)

        return x, a, state

    return init_init

def _init_iter_path(f, gradf, hessf=None, nr=None, maxiter=20,
                    tol=1e-3,loss=None,verbose=False,jit=True,**kwargs):

    if loss is None:
        loss = anp.linalg.norm

    def update(x, a, state):
        if verbose: 
            print('x: ', x)
            print('a: ', a)
        df = state['df']
        if verbose: print('df: ',df.T)

        gradfi, _ = gradf(x,*a)
        gradfi = gradfi[:nr,:nr]
        if verbose: print('gradf: \n',gradfi)

        dx = anp.linalg.solve(gradfi, df)

        if nr is not None:
            dx = anp.pad(dx, (0,-nr), 'constant')

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

