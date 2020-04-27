# Numerical Methods

## Optimization

- Hasofer/Lind-Rackwitz/Fiessler (HL-RF)
- Modified and Improved HL-RF (mHL-RF and iHL-RF)
- Polak-He
- Abdo-Rackwitz
- Sequential Quadratic Programming (SQP)
- Gradient-Projection method

### iHL-RF

> - Initialize
>   - Set iteration number: $i=0$
>   - Set tolerances: $\varepsilon_{1}$ and $\varepsilon_{2}\left(\mathrm{e} . \mathrm{g} ., \varepsilon_{1}=\varepsilon_{2}=10^{-3}\right)$
>   - Set starting point $\left.\mathbf{x}_{0} \text { (e.g., } \mathbf{x}_{0}=\mathbf{M}\right), \mathbf{u}_{0}=\mathbf{u}\left(\mathbf{x}_{0}\right)$
>   - Set scale parameter $\left.G_{0}, \text { (e.g., } G_{0}=g(\mathbf{M})\right)$
> - Iterate while $\left|G\left(\mathbf{u}_{i}\right) / G_{0}\right| \leq \varepsilon_{1}$ `AND` $\left\|\mathbf{u}_{i}-\boldsymbol{\alpha}_{i} \mathbf{u}_{i} \boldsymbol{\alpha}_{i}^{\mathrm{T}}\right\| \leq \varepsilon_{2}$
>   $$\begin{aligned}
>     &\mathbf{x}_{i}=\mathbf{x}\left(\mathbf{u}_{i}\right)(\text { skip this step for } i=0)\\
>     &\mathbf{J}_{\mathbf{u}, \mathbf{x}}, \mathbf{J}_{\mathbf{x}, \mathbf{u}}=\mathbf{J}_{\mathbf{u}, \mathbf{x}}^{-1} \text {at } \mathbf{x}=\mathbf{x}_{i}\\
>     &G\left(\mathbf{u}_{i}\right)=g\left(\mathbf{x}_{i}\right)\\
>     &\nabla G\left(\mathbf{u}_{i}\right)=\nabla g\left(\mathbf{x}_{i}\right) \mathbf{J}_{\mathbf{x}, \mathbf{u}}\\
>     &\boldsymbol{\alpha}_{i}=-\nabla G\left(\mathbf{u}_{i}\right) /\left\|\nabla G\left(\mathbf{u}_{i}\right)\right\|\\
>     &\text { select } c_{i} \text { to satisfy } c_{i}>\left\|\mathbf{u}_{i}\right\| /\left\|\nabla G\left(\mathbf{u}_{i}\right)\right\|\\
>     &m\left(\mathbf{u}_{i}\right)=0.5\left\|\mathbf{u}_{i}\right\|^{2}+c_{i}\left|G\left(\mathbf{u}_{\mathbf{i}}\right)\right|
>     \end{aligned}$$
> - Determine search direction
>   $$\mathbf{d}_{i}=\left[\frac{G\left(\mathbf{u}_{i}\right)}{\left\|\nabla G\left(\mathbf{u}_{i}\right)\right\|}+\boldsymbol{\alpha}_{i} \mathbf{u}_{i}\right] \boldsymbol{\alpha}_{i}^{\mathrm{T}}-\mathbf{u}_{i}$$

## Interpolation

-------------

### Lagrange

$$P_{n}(x)=\sum_{i=0}^{n} f\left(x_{i}\right) l_{i}(x), \qquad l_{i}(x)=\prod_{j=0 \atop j \neq i}^{n} \frac{x-x_{j}}{x_{i}-x_{j}}, \quad 0 \leqslant i \leqslant n$$

## Quadrature
--------


$$\int_{\Omega_{e}} f(\mathbf{x}) \mathrm{d} \Omega=\int_{\square} f(\mathbf{x}(\boldsymbol{\xi})) j(\boldsymbol{\xi}) \mathrm{d} \boldsymbol{\xi}$$

### Change of Variables

$$
\begin{array}{l}
x=\phi(u, v) \\
y=\psi(u, v)
\end{array}
$$
It will be assumed that $\phi$ and $\psi$ have continuous partial derivatives and that the Jacobian
$$
J(u, v)=\left|\begin{array}{ll}
\partial \phi / \partial u & \partial \phi / c v \\
\partial \psi / \partial u & \partial \psi / \partial v
\end{array}\right|
$$
does not vanish in $B$. Suppose, further, that
$$
\iint_{\boldsymbol{\rho}} h(u, v) d u d v \approx \sum_{k=1}^{n} w_{k} h\left(u_{k}, v_{k}\right), \quad\left(u_{k}, v_{k}\right) \in B^{\prime}
$$
is a rule of approximate integration over $B^{\prime} .$ Now we have
$$
\begin{aligned}
\iint_{\boldsymbol{s}} f(x, y) d x d y &=\iint_{\boldsymbol{s}^{*}} f(\phi(u, v), \psi(u, v))|J(u, v)| d u d v \\
& \approx \sum_{k=1}^{n} w_{k} f\left(\phi\left(u_{k}, v_{k}\right), \psi\left(u_{k}, v_{k}\right)\right)\left|J\left(u_{k}, v_{k}\right)\right| \\
&=\sum_{k=1}^{n} W_{k} f\left(x_{k}, y_{k}\right)
\end{aligned}
$$
where
$$
x_{k}=\phi\left(u_{k}, v_{k}\right), \quad y_{k}=\psi\left(u_{k}, v_{k}\right), \quad \text { and } \quad W_{k}=w_{k}\left|J\left(u_{k}, v_{k}\right)\right|
$$
Thus
$$
\iint_{B} f(x, y) d x d y \approx \sum_{k=1}^{n} W_{k} f\left(x_{k}, y_{k}\right)
$$


$$\begin{aligned}
&\mathbf{J}(\xi)=\frac{\partial \mathbf{x}}{\partial \xi}\\
&j(\xi)=\operatorname{det} \mathbf{J}(\xi)
\end{aligned}$$

### Open Newton-Cotes

#### Midpoint

$$I_{M}=\frac{L_{i}}{n} \sum_{i=1}^{n I P} g_{i}$$

### Closed Newton-Cotes

$$\int_{a}^{b} f(x) d x \approx \int_{a}^{b} L(x) d x=\int_{a}^{b}\left(\sum_{i=0}^{n} f\left(x_{i}\right) l_{i}(x)\right) d x=\sum_{i=0}^{n} f\left(x_{i}\right) \int_{a}^{b} l_{i}(x) d x$$

let $x_{i}=a+i \frac{b-a}{n}=a+i h,$ and the notation $f_{i}$ be a shorthand for $f\left(x_{i}\right)$

#### Trapazoidal Rule

$$I_{T}=\frac{L_{i}}{2 n}\left(g_{1}+2 g_{2}+\cdots+2 g_{n}+g_{n+1}\right) \quad w_i = \frac{2}{n}, i \not ={1, n+1}$$

#### Simpson's Rule

$n$: odd

$$I_{S}=\frac{L_{i}}{3 n}\left(g_{1}+4 g_{2}+2 g_{3}+\cdots+2 g_{n-1}+4 g_{n}+g_{n+1}\right)$$

### Gaussian

[source](https://keisan.casio.com/exec/system/1330940731)

$$\int_{-1}^{1} x^{k} d x=\sum_{i=1}^{n} w_{i}^{(n)}\left(x_{i}^{(n)}\right)^{k}$$

$$\int_{a}^{b} w(x) f(x) d x \simeq \sum_{i=1}^{n} w_{i} f\left(x_{i}\right)$$

|Quadrature | interval | $w(x)$ | polynomials| order |
|:---:|:---:|:---:|:---:|:---:|
|(1) Legendre |$[-1,1]$| $1$ | $P_{n}(x)$| $p=2n-1$ |
|(2) Chebyshev 1st|  $(-1,1)$ | $\frac{1}{\sqrt{1-x^{2}}}$ | $T_{n}(x)$
|(3) Chebyshev 2nd| $[-1,1]$| $\sqrt{1-x^{2}}$ | $U_{n}(x)$|
|(4) Laguerre| $\left[0, \infty \right)$ | $x^{\alpha} e^{-x}$ | $L_{n}^{\alpha}(x)$ |
|(5) Hermite| $(-\infty, \infty)$ | $e^{-x^{2}}$ | $H_{n}(x)$ |
|(6) Jacobi | $(-1,1)$ | $(1-x)^{\alpha}(1+x)^{\beta}$ | $J_{n}^{\alpha, \beta}(x)$ |
|(7) Lobatto | $[-1,1]$ | $1$ | $\quad P_{n-1}^{\prime}(x)$ | $p=2n-3$
|(8) Kronrod | $[-1,1]$ | $1$ | $P_{n}(x)$ |


## Integration of ODEs

### First-order ODEs


### Second-oder ODEs

Acceleration Methods
).1.1 Constant Acceleration Method
).102 Timoshenko's Modified Acceleration Method.
).1.3 Newmark's Linear and Parabolic Acceleration Methods. 0 0 I6
Newmark's ~-Methodso .
)02 Methods of Finite Differences
).201 Levy's Method ..• 0 •
).2.2 Salvadori 's Method 0 •
Houbolt's Method 0 Q
3.3 Numerical Solution of Differential Equations
303.1 Euler's and Modified Euler Method .. 0 •
)03.2 Runge, Heun and Kutta's Third Order Rule .
30303 Kutta's Fourth Order Rules