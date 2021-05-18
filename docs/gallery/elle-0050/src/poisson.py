# Claudio Perez
# May 2021
import jax
import anon.diff as diff
from anabel.template import template
import anabel.backend as anp


@template(6)
def poisson2(transf, test, trial, f=lambda x: 0.0, ndim=2, points=None, weights=None, thickness=1.0, **kwds):
    """
    Parameters
    ----------
    test, trial : Callable
        test and trial interpolants over the reference element.
    thickness : float
    
    http://people.inf.ethz.ch/arbenz/FEM17/pdfs/0-19-852868-X.pdf
    """
    state = {}
    
    det = anp.linalg.det
    slv = anp.linalg.solve
    
    jacn_test = diff.jacx(test)
    jacn_trial = diff.jacx(trial)
    
    def transf(xi, xyz):
        return test(xi)@xyz
    
    def jacn_transf(xi,xyz):
        return jacn_test(xi)@xyz
    
    def jacx_test(xi,xyz): 
        return slv(jacn_transf(xi,xyz), jacn_test(xi))
    
    def dvol(xi, xyz): 
        return 0.5*thickness*(abs(det(jacn_transf(xi,xyz))))
    
    def stif(u,xyz,xi,wght,**kwds):
        dNdx = jacx_test(xi,xyz)
        return (dNdx.T@dNdx)*dvol(xi,xyz)*wght
    
    fj = jax.vmap(f,0)
    
    def resp(u,xyz,xi,wght,**kwds):
        dNdx = jacx_test(xi,xyz)
        N = test(xi)[:,None]
        p = (dNdx.T@dNdx)@u*dvol(xi,xyz)*wght - (N@N.T)@fj(xyz)[:,None]*dvol(xi,xyz)*wght
        return p
    
    integral = jax.vmap(resp,(None,None,0,0))
    jac_integral = jax.vmap(stif,(None, None, 0, 0))
    
    def jacx(u,__,___,xyz,points,weights):
        return sum(jac_integral(u,xyz,points,weights))

    def main(u,__,___,xyz,points,weights):
        return sum(integral(u,xyz,points,weights))
    return locals()

@template(1)
def L2(transf,test,trial,u,quad_point=None, thickness=1.0):
    state = None
    det = anp.linalg.det
    slv = anp.linalg.solve
    du = lambda x: diff.jacfwd(u)(x)[:,None]
    jacn_test = diff.jacx(test)
    jacn_trial = diff.jacx(trial)

    def transf(xi, xyz):
        return test(xi)@xyz
    
    def jacn_transf(xi,xyz):
        return jacn_test(xi)@xyz

    dvol = lambda xi, xyz: 0.5*thickness*abs(det(jacn_transf(xi,xyz)))
     
    def resp(U,xyz,xi, wght):
        N = test(xi)[:,None]
        tmp = u(transf(xi,xyz)) - N.T@U
        q =  tmp.T@tmp * dvol(xi,xyz) * wght
        return q
    
    integral = jax.vmap(resp,(None,None,0,0))

    def main(u,__,___,xyz,points,weights):
        return sum(integral(u,xyz,points,weights))
    
    return locals()


@template(1)
def H1_v1(transf,test,trial,u,quad_point=None, thickness=1.0):
    state = None
    det = anp.linalg.det
    slv = anp.linalg.solve
    du = lambda x: diff.jacfwd(u)(x)[:,None]
    jacn_test = diff.jacx(test)
    jacn_trial = diff.jacx(trial)

    def transf(xi, xyz):
        return test(xi)@xyz
    
    def jacn_transf(xi,xyz):
        return jacn_test(xi)@xyz

    jacx_test = lambda xi,xyz: slv(jacn_transf(xi,xyz), jacn_test(xi))
    dvol = lambda xi, xyz: 0.5*thickness*abs(det(jacn_transf(xi,xyz)))
    
    
    def resp(U,xyz,xi, wght):
        tmp = du(transf(xi,xyz)) - jacx_test(xi,xyz)@U
        q = tmp.T@tmp * dvol(xi,xyz) * wght
        return q

    integral = jax.vmap(resp,(None,None,0,0))

    def main(u,__,___,xyz,points,weights):
        return sum(integral(u,xyz,points,weights))
    
    return locals()

@template(1)
def H1(transf,test,trial,u,quad_point=None, thickness=1.0):
    state = None
    det = anp.linalg.det
    slv = anp.linalg.solve
    du = lambda x: diff.jacfwd(u)(x)[:,None]
    jacn_test = diff.jacx(test)
    jacn_trial = diff.jacx(trial)

    def transf(xi, xyz):
        return test(xi)@xyz
    
    def jacn_transf(xi,xyz):
        return jacn_test(xi)@xyz

    jacx_test = lambda xi,xyz: slv(jacn_transf(xi,xyz), jacn_test(xi))
    dvol = lambda xi, xyz: 0.5*thickness*abs(det(jacn_transf(xi,xyz)))

    def resp(U,xyz,xi, wght):
        tmp = jacx_test(xi,xyz)@(U - u(transf(xi,xyz)))
        q = tmp.T@tmp * dvol(xi,xyz) * wght
        return q

    integral = jax.vmap(resp,(None,None,0,0))

    def main(u,__,___,xyz,points,weights):
        return sum(integral(u,xyz,points,weights))
    
    return locals()


