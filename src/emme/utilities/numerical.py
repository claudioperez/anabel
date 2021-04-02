import numpy as np 
from scipy.integrate import solve_ivp
try: import jax
except: pass


class NewtonRaphson:
    def __new__(cls, Model, U0 = None, tol=1.0e-3, loss=None, verbose=False,disp_dof=0):
        nf = Model.nf
        nt = Model.nt
        if U0 is None: U0 = np.zeros(nt)
        
        # create pure function Pr = Pr_U( U ) from function Pr = Pr_vector(U, Model)
        Pr_U = partial(emx.Pr_vector, Model )

        # use 'forward mode' algorithmic differentiation 
        # to obtain jacobian of Pr_U(U), Kt(U)
        Jac = jax.jacfwd(Pr_U)
        Kt = lambda Ui: np.squeeze(Jac(Ui))[:nf,:nf]
    
        @jax.jit
        def _dPu_(U, Pf):
            Pr = Pr_U(U)[:nf]
            Pu = Pf - Pr
            Kf = Kt(U)
            DUf = np.squeeze(np.linalg.solve(Kf,Pu))

            # reshaping
            DU = np.pad(DUf, (0, Model.nr), 'constant') 

            # update displacement
            Unew = U + DU
            return Unew, Pu
        
        
        def solve(Pf, verbose=False, maxiter=100):
            r = tol+1
            count = 1
            U = U0
            while jax.lax.gt(r , tol ):
                U, Pu = _dPu_(U,Pf)
                
                r = np.linalg.norm(Pu)

                if jax.lax.gt(count , maxiter): break
                if verbose: print('{}: {:.3f} {:.3f}'.format(count, r, U[disp_dof,...]))
                count+=1
            return U
            
        return solve

def ElcentroRHA(zeta, omega):
    g=386.4
    ground_motion = np.genfromtxt('ElcentroNS.csv', delimiter=',')[:750,:]
    ground_motion[0,0]=0.0
    time = ground_motion[:,0]
    t_span = (time[0], time[-1])
    u0 = (0.0, 0.0)

    def EOM(t, y):
        i = np.argmin(np.abs(time - t))
        D, dDdt = y
        dydt = [dDdt, -ground_motion[i,1]*g-omega**2*D-2*zeta*omega*dDdt]
        return dydt

    rtol=atol=1e-5
    sol = solve_ivp(EOM, y0=u0, t_span=t_span, args=(zeta, omega), rtol=rtol, atol=atol)
    D = sol.y[0,:]
    t = sol.t
    return t, D
   
def ExcitationInterpolation(p, Dt, T, u0, v0, k, Tn, zeta=0):
    n  = int(T/Dt) + 1
    c  = zeta*k*Tn/np.pi
    m  = k/(2*np.pi/Tn)**2
    wn = 2*np.pi/Tn
    wd = wn*np.sqrt(1 - zeta**2)
    
    # Initialize arrays
    u = np.zeros((n, 1))
    u[0] = v0
    vel = np.zeros((n,1))
    vel[0] = v0
    
    A = np.exp(-zeta*wn*Dt)*(zeta/np.sqrt(1-zeta**2)*np.sin(wd*Dt)+np.cos(wd*Dt))
    B = np.exp(-zeta*wn*Dt)*(1/wd*np.sin(wd*Dt))
    C = 1/k*(2*zeta/(wn*Dt)+np.exp(-zeta*wn*Dt)*(((1-2*zeta**2)/(wd*Dt)-zeta/np.sqrt(1-zeta**2))*np.sin(wd*Dt)-(1+2*zeta/(wn*Dt))*np.cos(wd*Dt)))
    D = 1/k*(1-2*zeta/(wn*Dt)+np.exp(-zeta*wn*Dt)*(2*zeta**2-1)/(wn*Dt)*np.sin(wd*Dt)+2*zeta/(wn*Dt)*np.cos(wd*Dt))
    A1 = -np.exp(-zeta*wn*Dt)*(wn/np.sqrt(1-zeta**2)*np.sin(wd*Dt))
    B1 =  np.exp(-zeta*wn*Dt)*(np.cos(wd*Dt)-zeta/np.sqrt(1-zeta**2)*np.sin(wd*Dt))
    C1 = 1/k*(-1/Dt+np.exp(-zeta*wn*Dt)*((wn/(np.sqrt(1-zeta**2))+zeta/(Dt*np.sqrt(1-zeta**2)))*np.sin(wd*Dt)+1/Dt*np.cos(wd*Dt)))
    D1 = 1/(k*Dt)*(1-np.exp(-zeta*wn*Dt)*(zeta/np.sqrt(1-zeta**2)*np.sin(wd*Dt)+np.cos(wd*Dt)))
    
 #     print(A,B,C,D,A1,B1,C1,D1)
    for i in range(n-1):
        u[i+1] = A*u[i]+B*vel[i]+C*p(Dt*i)+D*p(Dt*(i+1))
        vel[i+1] = A1*u[i]+B1*vel[i]+C1*p(Dt*i)+D1*p(Dt*(i+1))
    # u due to free vibration
    return u, vel

def CentralDifference(p, Dt, T, u0, v0, k, Tn, zeta=0):
    n    = int(T/Dt)+1
    c    = zeta*k*Tn/np.pi
    m    = k/(2*np.pi/Tn)**2
    
    a0 = (p(0.0)-c*v0-k*u0)/m
    k_hat = m/Dt**2+c/(2*Dt)
    alpha = m/Dt**2-c/(2*Dt)
    beta = k - 2*m/Dt**2
    ui = np.zeros(n)
    uip1 = np.zeros(n)
    uim1 = np.zeros(n)
    p_hat = np.zeros(n)
    
    for i in range(1,n):
        t = (i-1)*Dt
        if i==1:
            ui[i]=u0
            uim1[i] = u0-(Dt)*v0+0.5*Dt**2*a0
        else:
            ui[i] = uip1[i-1]
            uim1[i] = ui[i-1]
            
        p_hat[i] =  p(t) - alpha*uim1[i] - beta*ui[i]
        uip1[i] = p_hat[i]/k_hat
    
    return uip1

def Newmark(p, Dt, T, u0, v0, k, Tn, zeta=0):
    n = int(T/Dt)+1
    c = zeta*k*Tn/np.pi
    m = k/(2*np.pi/Tn)**2
    
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)
    p_hat = np.zeros(n)
    
    u[0] = u0
    v[0] = v0
    a[0] = (p(0.0) - c*v0 - k*u0)/m
    a1 = 4/Dt**2*m + 2/Dt*c
    a2 = 4/Dt*m + c
    a3 = m
    k_hat = k + a1
    
    for i in range(1, n-1):
        t = (i-1)*Dt
        p_hat[i+1] = p(t) +a1*u[i]+a2*v[i]+a3*a[i]
        u[i+1] = p_hat[i+1]/k_hat
        v[i+1] = 2/Dt*(u[i+1]-u[i])-v[i]
        a[i+1] = 4/Dt**2*(u[i+1]-u[i])-4/Dt*v[i]-a[i]

    return u, v, a

def Herrmann_Peterson(E, tau, epsilon, tot_time, N):
    dt = tot_time/N  
    time = np.zeros((N,1))
    sigma = np.zeros((N,1))
    h1 = np.zeros((N,1))
    h2 = np.zeros((N,1))
    h3 = np.zeros((N,1))

    for i in range(1, N):
        time[i] = time[i-1]+dt

        de = epsilon(time[i]) - epsilon(time[i-1])
        h1[i] = np.exp(-dt/tau[1])*h1[i-1] + E[1]*(tau[1]/dt)*(1- np.exp(-dt/tau[1]))*de 
        h2[i] = np.exp(-dt/tau[2])*h2[i-1] + E[2]*(tau[2]/dt)*(1- np.exp(-dt/tau[2]))*de 
        h3[i] = np.exp(-dt/tau[3])*h2[i-1] + E[3]*(tau[3]/dt)*(1- np.exp(-dt/tau[3]))*de 
        sigma[i] = E[0]*epsilon(time[i]) + h1[i] + h2[i] + h3[i] 
        
    return time, sigma

def _armijo_haukaas_init(b0: float=0.2, b: float = 0.5, m: int =4, a: float=0.5):
    
    def _armijo_haukaas():
        pass 
    try: 
        _armijo_haukaas = jax.jit(_armijo_haukaas)
        print('Jit Successful')
    except: pass
    return _armijo_haukaas

def _merit_init():
    pass

class iHLRF:
    def __init__(self, f, gradf, u0, loss=None, 
                    tol1=1.0e-4, tol2=1.0e-4, maxiter=20, maxiter_step=10,
                    step_opt={'c':10.},**kwargs):
        self.u0 = u0
        self.f = f 
        self.gradf = gradf
        self.tol1 = tol1
        self.tol2 = tol2
        self.maxiter = maxiter
        self.maxiter_step = maxiter_step
        self.loss = loss
        self.c = step_opt['c']
        if loss is None: 
            self.loss = np.linalg.norm

        # self.init()

    def init(self,verbose=False):
        if verbose:
            print('\niHL-RF Algorithm (Zhang and Der Kiureghian 1995)***************',
                  '\nInitialize iHL-RF: \n',
                  'u0: ', np.around(self.u0.T,4), '\n')

        self.G0 = self.f(self.u0) # scale parameter
        if verbose: print('G0: ',np.around(self.G0,4),'\n')

        self.ui = self.u0[None,:].T

        self.Gi = self.G0
        if verbose: print('Gi: ', np.around(self.Gi,4), '\n',)

        self.GradGi = self.gradf(self.ui[:,0].T)
        # if verbose: print('GradGi: ', np.around(self.GradGi,4), '\n',)
        if verbose: print('GradGi: ', self.GradGi, '\n',)

        self.alphai = -(self.GradGi / np.linalg.norm(self.GradGi))[None,:]
        self.count = 0
        self.res1 = abs(self.Gi/self.G0)
        self.res2 = self.loss(self.ui - (self.alphai@self.ui)*self.alphai.T )

        if verbose:
            print(
                # '\niHL-RF Algorithm (Zhang and Der Kiureghian 1995)***************',
                #   '\nInitialize iHL-RF: \n',
                #   'u0: ', np.around(self.u0.T,4), '\n',
                #   'G0: ',np.around(self.G0,4),'\n'
                #   'ui: ', np.around(self.ui.T,4), '\n',
                #   'Gi: ', np.around(self.Gi,4), '\n',
                #   'GradGi: ', np.around(self.GradGi,4),'\n',
                  'alphai: ', np.around(self.alphai,4),'\n',
                  'res1: ' , np.around(self.res1,4),'\n',
                #   'res2: ' , np.around(self.res2,4),'\n',
                  'res2: ' , self.res2,'\n',)
    def merit():
        pass

    def incr(self,method='adk', verbose=False):
        if method == 'basic':
            self.ui1 = self.ui + self.lamda * self.di
            return self.ui1 
        # c = 10.0
        self.ci = self.loss(self.ui) / np.linalg.norm(self.GradGi) + self.c
        self.mi = 0.5*self.loss(self.ui)**2 + self.ci*abs(self.Gi)
        self.mi1 = 0.5*self.loss(self.ui1)**2 + self.ci*abs(self.Gi1)

        self.count_step = 0
        while (self.mi1 >= self.mi) :
            self.lamda = self.lamda/2
            if verbose: print('lamda:', self.lamda)
            self.ui1 = self.ui + self.lamda * self.di
            if verbose: print('ui1: ',self.ui1)
            self.Gi1 = self.f(self.ui1[:,0].T)
            self.mi1 = 0.5*np.linalg.norm(self.ui1)**2 + self.ci*abs(self.Gi1)
            self.count_step += 1
            if (self.count_step >= self.maxiter_step): break
        return self.ui1

    
    
    def dirn(self):
        self.di = (self.Gi/np.linalg.norm(self.GradGi) + self.alphai@self.ui)*self.alphai.T - self.ui
        return self.di

    def step(self,verbose=False,basic=False):
        # self.di = (self.Gi/np.linalg.norm(self.GradGi) + self.alphai@self.ui)*self.alphai.T - self.ui
        di = self.dirn()
        if verbose: print('di: ',self.di)

        self.lamda = 1.0
        # self.lamda = 0.05

        self.ui1 = self.ui + self.lamda * di
        if verbose: print('ui1: ',self.ui1,self.ui1[:,0].T)

        self.Gi1 = self.f(self.ui1[:,0].T)

        self.incr(basic)

        self.ui = self.ui1  
        self.Gi = self.f(self.ui[:,0].T)
        self.GradGi = self.gradf(self.ui[:,0].T)
        self.alphai = -(self.GradGi / np.linalg.norm(self.GradGi))[None,:]

        self.res1 = abs(self.Gi/self.G0)
        self.res2 = self.loss(self.ui - (self.alphai@self.ui)*self.alphai.T)
        self.count += 1


        if verbose: print('\niHL-RF step: {}'.format(self.count))

        if verbose:
            print('ui: ',       np.around(self.ui,4), '\n',
                  'Gi: ',       np.around(self.Gi,4), '\n',
                  'GradGi: ', np.around(self.GradGi,4),'\n',
                  'alphai: ', np.around(self.alphai,4),'\n',
                  'res1: ' , np.around(self.res1,4),'\n',
                  'res2: ' , np.around(self.res2,4),'\n',)
        return self.ui
    
    def run(self,verbose=False, steps=None):
        self.init(verbose=verbose)
        if steps is not None: self.maxiter = steps
        while not(self.res1 < self.tol1 and self.res2 < self.tol2):
            self.step(verbose=verbose)

            if (self.count > self.maxiter): break

        return self.ui
