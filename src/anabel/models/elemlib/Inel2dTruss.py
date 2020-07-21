import jax 

class Element: pass

def Inel2dTruss(ElemSpace, _ridx, _return, bind):
    """
    `bind`: (>>=) :: State s a -> (a -> State s b) -> State s b
    
    state :: (s -> (a, s)) -> State s a

    return :: a -> State s a
    return x = state ( \ s -> (x, s) )

    compose f g = \s0 -> let (a1, s1) = f s0 in (g a1) s1 

    evalState :: State s a -> s -> a

    execState :: State s a -> s -> s
    """

    @jax.jit
    def f(x):
        S_a = ElemSpace.f(x) # f: a -> (s,a)
        S_b = bind(S_a, _response) # S_b = S_a >> _response
        return S_b
    
    return f

def _response(v): 
    return v