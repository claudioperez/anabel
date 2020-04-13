# Numerical Methods

## Optimization

- Hasofer/Lind-Rackwitz/Fiessler (HL-RF)
- Modified and Improved HL-RF (mHL-RF and iHL-RF)
- Polak-He
- Abdo-Rackwitz
- Sequential Quadratic Programming (SQP)
- Gradient-Projection method

### iHL-RF

- Initialize
  - Set iteration number: $i=0$
  - Set tolerances: $\varepsilon_{1}$ and $\varepsilon_{2}\left(\mathrm{e} . \mathrm{g} ., \varepsilon_{1}=\varepsilon_{2}=10^{-3}\right)$
  - Set starting point $\left.\mathbf{x}_{0} \text { (e.g., } \mathbf{x}_{0}=\mathbf{M}\right), \mathbf{u}_{0}=\mathbf{u}\left(\mathbf{x}_{0}\right)$
  - Set scale parameter $\left.G_{0}, \text { (e.g., } G_{0}=g(\mathbf{M})\right)$
- Iterate while $\left|G\left(\mathbf{u}_{i}\right) / G_{0}\right| \leq \varepsilon_{1}$ `AND` $\left\|\mathbf{u}_{i}-\boldsymbol{\alpha}_{i} \mathbf{u}_{i} \boldsymbol{\alpha}_{i}^{\mathrm{T}}\right\| \leq \varepsilon_{2}$
  $$\begin{aligned}
    &\mathbf{x}_{i}=\mathbf{x}\left(\mathbf{u}_{i}\right)(\text { skip this step for } i=0)\\
    &\mathbf{J}_{\mathbf{u}, \mathbf{x}}, \mathbf{J}_{\mathbf{x}, \mathbf{u}}=\mathbf{J}_{\mathbf{u}, \mathbf{x}}^{-1} \text {at } \mathbf{x}=\mathbf{x}_{i}\\
    &G\left(\mathbf{u}_{i}\right)=g\left(\mathbf{x}_{i}\right)\\
    &\nabla G\left(\mathbf{u}_{i}\right)=\nabla g\left(\mathbf{x}_{i}\right) \mathbf{J}_{\mathbf{x}, \mathbf{u}}\\
    &\boldsymbol{\alpha}_{i}=-\nabla G\left(\mathbf{u}_{i}\right) /\left\|\nabla G\left(\mathbf{u}_{i}\right)\right\|\\
    &\text { select } c_{i} \text { to satisfy } c_{i}>\left\|\mathbf{u}_{i}\right\| /\left\|\nabla G\left(\mathbf{u}_{i}\right)\right\|\\
    &m\left(\mathbf{u}_{i}\right)=0.5\left\|\mathbf{u}_{i}\right\|^{2}+c_{i}\left|G\left(\mathbf{u}_{\mathbf{i}}\right)\right|
    \end{aligned}$$
- Determine search direction
  $$\mathbf{d}_{i}=\left[\frac{G\left(\mathbf{u}_{i}\right)}{\left\|\nabla G\left(\mathbf{u}_{i}\right)\right\|}+\boldsymbol{\alpha}_{i} \mathbf{u}_{i}\right] \boldsymbol{\alpha}_{i}^{\mathrm{T}}-\mathbf{u}_{i}$$

## Quadrature

### Midpoint

$$I_{M}=\frac{L_{i}}{n} \sum_{i=1}^{n I P} g_{i}$$

### Newton-Cotes

#### Trapazoidal Rule

$$I_{T}=\frac{L_{i}}{2 n}\left(g_{1}+2 g_{2}+\cdots+2 g_{n}+g_{n+1}\right) \quad w_i = \frac{2}{n}, i \not ={1, n+1}$$

#### Simpson's Rule

$n$: odd

$$I_{S}=\frac{L_{i}}{3 n}\left(g_{1}+4 g_{2}+2 g_{3}+\cdots+2 g_{n-1}+4 g_{n}+g_{n+1}\right)$$

### Gaussian

[source](https://keisan.casio.com/exec/system/1330940731)

$$\int_{a}^{b} w(x) f(x) d x \simeq \sum_{i=1}^{n} w_{i} f\left(x_{i}\right)$$

|Quadrature | interval | $w(x)$ | polynomials|
|:---:|:---:|:---:|:---:|
|(1) Legendre |$[-1,1]$| $1$ | $P_{n}(x)$|
|(2) Chebyshev 1st|  $(-1,1)$ | $\frac{1}{\sqrt{1-x^{2}}}$ | $T_{n}(x)$
|(3) Chebyshev 2nd| $[-1,1]$| $\sqrt{1-x^{2}}$ | $U_{n}(x)$|
|(4) Laguerre| $\left[0, \infty \right)$ | $x^{\alpha} e^{-x}$ | $L_{n}^{\alpha}(x)$
|(5) Hermite| $(-\infty, \infty)$ | $e^{-x^{2}}$ | $H_{n}(x)$ |
|(6) Jacobi | $(-1,1)$ | $(1-x)^{\alpha}(1+x)^{\beta}$ | $J_{n}^{\alpha, \beta}(x)$ |
|(7) Lobatto | $[-1,1]$ | $1$ | $\quad P_{n-1}^{\prime}(x)$
|(8) Kronrod | $[-1,1]$ | $1$ | $P_{n}(x)$