import jax
import anon
import anon.atom as anp
from anon import quad

@anon.dual.generator(5)
def elem_0001(f=None,a1=1.0, a2=0.0,order=4):
    state = {}
    if f is None:
        f = lambda x: 0.0

    def transf(xi: float,x_nodes)->float:
        return ( x_nodes[0]*( + xi/6) 
               + x_nodes[1]*( - 4*xi/3) 
               + x_nodes[2]*( + 1) 
               + x_nodes[3]*( + 4*xi/3) 
               + x_nodes[4]*( - xi/6)
        )
    def grad_transf(xi,x_nodes):
        return abs(x_nodes[-1] - x_nodes[0])/2
#         return (x_nodes[0]*( + 1/6) 
#               + x_nodes[1]*( - 4/3) 
#               + x_nodes[2]*(   0.0) 
#               + x_nodes[3]*( + 4/3) 
#               + x_nodes[4]*( - 1/6)
#         )
    
    quad_points = quad.quad_points(n=order+1,rule="gauss-legendre")
#     quad_points = quad.quad_points(n=5,rule="mid")
    @jax.jit
    def jacx(u=None,y=None,state=None,xyz=None, a1=a1, a2=a2):
        x_nodes = anp.linspace(xyz[0][0],xyz[-1][0],5)
        return anp.array(
            [
                [
                985*a1/378,
                -3424*a1/945,
                508*a1/315,
                -736*a1/945,
                347*a1/1890,
              ],
              [
                -3424*a1/945,
                1664*a1/189,
                -2368*a1/315,
                2944*a1/945,
                -736*a1/945,
              ],
              [
                508*a1/315,
                -2368*a1/315,
                248*a1/21,
                -2368*a1/315,
                508*a1/315,
              ],
              [
                -736*a1/945,
                2944*a1/945,
                -2368*a1/315,
                1664*a1/189,
                -3424*a1/945,
              ],
              [
                347*a1/1890,
                -736*a1/945,
                508*a1/315,
                -3424*a1/945,
                985*a1/378,
              ],
#               [
#                 985*a1/378 + 292*a2/2835,
#                 -3424*a1/945 + 296*a2/2835,
#                 508*a1/315 - 58*a2/945,
#                 -736*a1/945 + 8*a2/405,
#                 347*a1/1890 - 29*a2/2835,
#               ],
#               [
#                 -3424*a1/945 + 296*a2/2835,
#                 1664*a1/189 + 256*a2/405,
#                 -2368*a1/315 - 128*a2/945,
#                 2944*a1/945 + 256*a2/2835,
#                 -736*a1/945 + 8*a2/405,
#               ],
#               [
#                 508*a1/315 - 58*a2/945,
#                 -2368*a1/315 - 128*a2/945,
#                 248*a1/21 + 208*a2/315,
#                 -2368*a1/315 - 128*a2/945,
#                 508*a1/315 - 58*a2/945,
#               ],
#               [
#                 -736*a1/945 + 8*a2/405,
#                 2944*a1/945 + 256*a2/2835,
#                 -2368*a1/315 - 128*a2/945,
#                 1664*a1/189 + 256*a2/405,
#                 -3424*a1/945 + 296*a2/2835,
#               ],
#               [
#                 347*a1/1890 - 29*a2/2835,
#                 -736*a1/945 + 8*a2/405,
#                 508*a1/315 - 58*a2/945,
#                 -3424*a1/945 + 296*a2/2835,
#                 985*a1/378 + 292*a2/2835,
#               ],

            ]
          ) / grad_transf(0,x_nodes)

#     @jax.jit
#     def main(u,_,state,xyz,a1=a1,a2=a2):
#         M = jacx(xyz=xyz,a1=0,a2=1)
#         x_nodes = anp.linspace(xyz[0][0],xyz[-1][0],5)
#         fi = anp.array([f(x) for x in x_nodes])[:,None]
#         external_term = M@fi
#         resp = jacx(u,_,state,xyz,a1=a1,a2=a2)@u + external_term
#         return u, resp, state
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