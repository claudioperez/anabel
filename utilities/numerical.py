import numpy as np 
# from numba import njit, jit
from scipy.integrate import solve_ivp


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


