pi = anp.pi
sin = anp.sin
cos = anp.cos
exp = anp.exp
def cn(t):
    return [
        pi**2*(pi*alpha*sin(pi*t)/(pi**3*alpha**2 + pi) 
            - cos(pi*t)/(pi**3*alpha**2 + pi))/100 
            + pi**2/(100*(pi**3*alpha**2*exp(pi**2*alpha*t) 
            + pi*exp(pi**2*alpha*t))), 
        pi**2*(9*pi*alpha*sin(pi*t)/(81*pi**3*alpha**2 + pi) 
            - cos(pi*t)/(81*pi**3*alpha**2 + pi))/100
            + pi**2/(100*(81*pi**3*alpha**2*exp(9*pi**2*alpha*t) 
            + pi*exp(9*pi**2*alpha*t))), 
        pi**2*(25*pi*alpha*sin(pi*t)/(625*pi**3*alpha**2 + pi) 
            - cos(pi*t)/(625*pi**3*alpha**2 + pi))/100 
            + pi**2/(100*(625*pi**3*alpha**2*exp(25*pi**2*alpha*t) 
            + pi*exp(25*pi**2*alpha*t))), 
        pi**2*(49*pi*alpha*sin(pi*t)/(2401*pi**3*alpha**2 + pi) 
            - cos(pi*t)/(2401*pi**3*alpha**2 + pi))/100 
            + pi**2/(100*(2401*pi**3*alpha**2*exp(49*pi**2*alpha*t) 
            + pi*exp(49*pi**2*alpha*t))), 
        pi**2*(81*pi*alpha*sin(pi*t)/(6561*pi**3*alpha**2 + pi) 
            - cos(pi*t)/(6561*pi**3*alpha**2 + pi))/100 
            + pi**2/(100*(6561*pi**3*alpha**2*exp(81*pi**2*alpha*t) 
            + pi*exp(81*pi**2*alpha*t)))
     ]

def u(x,t):
    c = cn(t)
    return sum( 
        c[n] * anp.sin(xi*x)
            for n,xi in enumerate([(2*k+1)*anp.pi for k in range(5)])
    ) 
