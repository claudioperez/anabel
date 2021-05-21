# external imports
import jax
# internal imports
import anon
import anon.atom as anp
from anon import quad

@anon.dual.generator(5)
def elem_0001(f=None,a1=1.0, a2=0.0,order=4):
    """
    Fourth order 1D Lagrange finite element with uniformly spaced nodes.
    
    Parameters
    ----------
    f: Callable
        element loading.
    a1: float
        Stiffness coefficient
    a2: float
        Mass coefficient
    """
    state = {}
    if f is None: f = lambda x: 0.0

    def transf(xi: float,x_nodes)->float:
        return ( x_nodes[0]*( + xi/6) 
               + x_nodes[1]*( - 4*xi/3) 
               + x_nodes[2]*( + 1) 
               + x_nodes[3]*( + 4*xi/3) 
               + x_nodes[4]*( - xi/6)
        )
    def grad_transf(xi,x_nodes):
        return abs(x_nodes[-1] - x_nodes[0])/2
    
    quad_points = quad.quad_points(n=order+1,rule="gauss-legendre")

    @jax.jit
    def jacx(u=None,y=None,state=None,xyz=None, a1=a1, a2=a2):
        x_nodes = anp.linspace(xyz[0][0],xyz[-1][0],5)
        grad = grad_transf(0,x_nodes)
        return a1*anp.array([
            [985/378, -3424/945, 508/315, -736/945, 347/1890],
            [-3424/945, 1664/189, -2368/315, 2944/945, -736/945],
            [508/315, -2368/315, 248/21, -2368/315, 508/315],
            [-736/945, 2944/945, -2368/315, 1664/189, -3424/945],
            [347/1890, -736/945, 508/315, -3424/945, 985/378],
        ]) / grad + a2*anp.array([
            [292/2835, 296/2835, -58/945, 8/405, -29/2835],
            [296/2835, 256/405, -128/945, 256/2835, 8/405],
            [-58/945, -128/945, 208/315, -128/945, -58/945],
            [8/405, 256/2835, -128/945, 256/405, 296/2835],
            [-29/2835, 8/405, -58/945, 296/2835, 292/2835],
        ])*grad


    @jax.jit
    def main(u,_,state,xyz,a1=a1,a2=a2):
        x_nodes = anp.linspace(xyz[0][0],xyz[-1][0],5)
        external_term = sum(
              anp.array([
                [f(transf(xi,x_nodes))*(2*xi**4/3 - 2*xi**3/3 - xi**2/6 + xi/6)],
                [f(transf(xi,x_nodes))*(-8*xi**4/3 + 4*xi**3/3 + 8*xi**2/3 - 4*xi/3)],
                [f(transf(xi,x_nodes))*(4*xi**4 - 5*xi**2 + 1)],
                [f(transf(xi,x_nodes))*(-8*xi**4/3 - 4*xi**3/3 + 8*xi**2/3 + 4*xi/3)],
                [f(transf(xi,x_nodes))*(2*xi**4/3 + 2*xi**3/3 - xi**2/6 - xi/6)],
              ]
            )*weight * grad_transf(xi,x_nodes) for xi, weight in zip(*quad_points)
        )
        resp = jacx(u,_,state,xyz,a1=a1,a2=a2)@u + external_term
        return u, resp, state
    return locals()
