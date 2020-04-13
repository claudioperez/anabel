# Distributions

$$J_{\left(\mu, \sigma\right),\left(\mu, \delta\right)}=\left[\begin{array}{cc}
1 & 0 \\
\delta & \mu
\end{array}\right]$$

## Normal (2007 Exam)

## Lognormal

(see 2007 Exam for joint lognorm)

    s = np.sqrt(np.log(std[i]**2/mean[i]**2+1))
    scale = mean[i]**2/np.sqrt(std[i]**2+mean[i]**2)
    X[i] = scipy.stats.lognorm(s=s, scale=scale)

    J_t_tf[0,0] = (1+2*del1**2)/(mean1*(1+del1**2))   # dlam1 wrt mean1
    J_t_tf[0,1] = -del1/(mean1*(1+del1**2))           # dlam1 wrt std1    
    J_t_tf[1,0] = -del1**2/(zeta*mean1*(1+del1**2))   # dzeta wrt mean1
    J_t_tf[1,1] =  del1/(zeta*mean1*(1+del1**2))      # dzeta wrt std1

$$\begin{array}{l}
\mathrm{J}_{(\lambda \zeta), (\mu \sigma) }:=\frac{1}{\mu \left(1+\delta_{1}^{2}\right)} \left(\begin{array}{cc}
1+2  \delta_{1}^{2} & -\delta_{1} \\
\frac{-\delta_{1}^{2}}{\zeta} & \frac{\delta_{1}}{\zeta}
\end{array}\right)
\end{array}$$

## Gumbel (Ex 6.7, )

    scale = np.sqrt(6)/np.pi * std[i]
    loc = mean[i] - np.euler_gamma*scale
    X[i] = scipy.stats.gumbel_r(loc=loc, scale=scale)

$$\begin{aligned}
&v=\mu_{3}+\frac{\gamma \sqrt{6} \sigma_{3}}{\pi}\\
&\alpha=\frac{\pi}{\sqrt{6} \sigma_{3}}
\end{aligned}$$

$$\mathbf{J}_{(\alpha, v),\left(\mu, \sigma\right)}=\left[\begin{array}{cc}
1 & \frac{\gamma \sqrt{6}}{\pi} \\
0 & -\frac{\pi}{\sqrt{6} \sigma^{2}}
\end{array}\right]$$

## Gamma

    lam = mean[i]/std[i]**2
    k = (mean[i]/std[i])**2
    X[i] = scipy.stats.gamma(a=k, scale=1/lam);

$$\mathrm{J}_{\mathrm{(\lambda k}), (\mu  \sigma) } =\left(\begin{array}{cc}
\frac{1}{\sigma^{2}} & \frac{-2  \mu}{\sigma^{3}} \\
\frac{2  \mu}{\sigma^{2}} & \frac{-2 \mu^{2}}{\sigma^{3}}
\end{array}\right)$$

## Weibull (Ex 6.7)