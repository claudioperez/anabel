# IV Linear Elasticity

$I=\int_{A} y^{2} d A$

##### Compatibility:

$\operatorname{curl}(\operatorname{curl} \epsilon)=0, \quad e_{i p q} e_{j r s} \epsilon_{q s, r p}=0$

## 7 Linear Constitutive Equations


### 7.1 Free Energy: Elasticity

$$\sigma=\mathbb{C} \epsilon, \quad \sigma_{i j}=C_{i j k l} \epsilon_{k l}$$
Voight Notation:

$$\left(\begin{array}{l}{\sigma_{11}} \\ {\sigma_{22}} \\ {\sigma_{33}} \\ {\sigma_{23}} \\ {\sigma_{13}} \\ {\sigma_{12}}\end{array}\right)=\left(\begin{array}{cccccc}{C_{1111}} & {C_{1122}} & {C_{1133}} & {C_{1123}} & {C_{1113}} & {C_{112}} \\ {C_{211}} & {C_{222}} & {C_{233}} & {C_{223}} & {C_{2213}} & {C_{2212}} \\ {C_{331}} & {C_{332}} & {C_{333}} & {C_{333}} & {C_{313}} & {C_{2312}} \\ {C_{1311}} & {C_{1322}} & {C_{1333}} & {C_{1323}} & {C_{1313}} & {C_{1312}} \\ {C_{1211}} & {C_{1222}} & {C_{1233}} & {C_{1223}} & {C_{1213}} & {C_{1212}}\end{array}\right)\left(\begin{array}{c}{\epsilon_{11}} \\ {\epsilon_{22}} \\ {\epsilon_{33}} \\ {2 \epsilon_{23}} \\ {2 \epsilon_{13}} \\ {2 \epsilon_{12}}\end{array}\right)$$

**Triclinic, 21**

**Monoclinic 13:**
$\left\{\mathcal{C}_{11}, \mathcal{C}_{22}, \mathcal{C}_{33}, \mathcal{C}_{44}, \mathcal{C}_{55}, \mathcal{C}_{66}, \mathcal{C}_{12}, \mathcal{C}_{13}, \mathcal{C}_{16}, \mathcal{C}_{23}, \mathcal{C}_{26}, \mathcal{C}_{36}, \mathcal{C}_{45}\right\}$

**Orthotropic,  9:**
$\left\{\mathcal{C}_{11}, \mathcal{C}_{22}, \mathcal{C}_{33}, \mathcal{C}_{44}, \mathcal{C}_{55}, \mathcal{C}_{66}, \mathcal{C}_{12}, \mathcal{C}_{13}, \mathcal{C}_{23}\right\}$

**Tetragonal, 6:**
$\left\{\mathcal{C}_{11}, \mathcal{C}_{33}, \mathcal{C}_{44}, \mathcal{C}_{66}, \mathcal{C}_{12}, \mathcal{C}_{13}\right\}$

**Transversely isotropic, 5:**
$\left\{\mathcal{C}_{11}, \mathcal{C}_{33}, \mathcal{C}_{44}, \mathcal{C}_{12}, \mathcal{C}_{13}\right\}$

**Cubic, 3:**
$\left\{\mathcal{C}_{11}, \mathcal{C}_{12}, \mathcal{C}_{44}\right\}$

### 7.5 Isotropic Relations

$\boldsymbol{\sigma}=\mathbb{C} \boldsymbol{\epsilon}=2 \mu \boldsymbol{\epsilon}+\lambda(\operatorname{tr} \boldsymbol{\epsilon}) \mathbf{1}$

$\boldsymbol{\sigma}=2 \mu \boldsymbol{\epsilon}^{\prime}+\kappa(\operatorname{tr} \boldsymbol{\epsilon}) \mathbf{1},
\quad \sigma_{i j}=2 \mu \epsilon_{i j}^{\prime}+\kappa\left(\epsilon_{k k}\right) \delta_{i j}$

$\mathbb{C}=2 \mu^{\mathrm{sym}}+\lambda \mathbf{1} \otimes \mathbf{1}, \quad C_{i j k l}=\mu\left(\delta_{i k} \delta_{j l}+\delta_{i l} \delta_{j k}\right)+\lambda \delta_{i j} \delta_{k l}$


$\epsilon=\frac{1}{2 \mu} \sigma^{\prime}+\frac{1}{9 \kappa}(\operatorname{tr} \sigma) 1,
\quad \epsilon_{i j}=\frac{1}{2 \mu} \sigma_{i j}^{\prime}+\frac{1}{9 \kappa}\left(\sigma_{k k}\right) \delta_{i j}$

$E \equiv \frac{9 \kappa \mu}{3 \kappa+\mu}, \quad \nu \equiv \frac{1}{2}\left[\frac{3 \kappa-2 \mu}{3 \kappa+\mu}\right]$

$-1<\nu<\frac{1}{2}$

$\kappa=\lambda+\frac{2}{3} \mu$

### 7.5.4

$$
\sigma=\frac{E}{(1+\nu)}\left[\epsilon+\frac{\nu}{(1-2 \nu)}(\operatorname{tr} \epsilon) 1\right], \quad \sigma_{i j}=\frac{E}{(1+\nu)}\left[\epsilon_{i j}+\frac{\nu}{(1-2 \nu)}\left(\epsilon_{k k}\right) \delta_{i j}\right]
$$

$$
\epsilon=\frac{1}{E}[(1+\nu) \sigma-\nu(\operatorname{tr} \sigma) 1] . \quad \epsilon_{i j}=\frac{1}{E}\left[(1+\nu) \sigma_{i j}-\nu\left(\sigma_{k k}\right) \delta_{i j}\right]
$$


## 8 Elastostatics

### Displacement Formulation (Navier)

$C_{i j k l} u_{k, l j}+b_{i}=0$

#### Isotropic

$\mu \triangle \mathbf{u}+(\lambda+\mu) \nabla \operatorname{div} \mathbf{u}+\mathbf{b}=0\\$  $
\mu u_{i, j j}+(\lambda+\mu) u_{j, j i}+b_{i}=0\\$

$(\lambda+2 \mu) \nabla \operatorname{div} \mathbf{u}-\mu \text { curl curl } \mathbf{u}+\mathbf{b}=\mathbf{0}\\$
 $(\lambda+2 \mu) u_{j, j i}-\mu e_{i j k} e_{k l m} u_{m, l j}+b_{i}=0$

##### Boundary:

$$\left.\begin{array}{r}{\mathbf{u}=\hat{\mathbf{u}} \text { on } \mathcal{S}_{1},} \\ {\left(\mu\left(\nabla \mathbf{u}+(\nabla \mathbf{u})^{\top}\right)+\lambda(\operatorname{div} \mathbf{u}) \mathbf{1}\right) \mathbf{n}=\hat{\mathbf{t}} \quad \text { on } \mathcal{S}_{2}}\end{array}\right\}$$

### Stress Formulation (Beltrami-Mitchell)

##### Compatibility:

$\sigma_{i j, k k}+\frac{1}{1+\nu} \sigma_{k k, i j}=-\frac{\nu}{1-\nu} b_{k, k} \delta_{i j}-b_{i, j}-b_{j, i}\\$
$\Delta \sigma_{k k}=-\frac{1+\nu}{1-\nu} b_{k, k}\\$


### 8.9.1 Plane Strain

$u_{\alpha}=u_{\alpha}\left(x_{1}, x_{2}\right), \quad u_{3}=0$

$\epsilon_{\alpha \beta}=\frac{1}{2}\left(u_{\alpha, \beta}+u_{\beta, \alpha}\right)$

$\epsilon_{13}=\epsilon_{23}=\epsilon_{33}=0$

$\sigma_{\alpha \beta}=\frac{E}{(1+\nu)}\left(\epsilon_{\alpha \beta}+\frac{\nu}{(1-2 \nu)}\left(\epsilon_{\gamma \gamma}\right) \delta_{\alpha \beta}\right)$

$\sigma_{33}=\nu \sigma_{\alpha \alpha}$

$\epsilon_{\alpha \beta}=\frac{1+\nu}{E}\left(\sigma_{\alpha \beta}-\nu\left(\sigma_{\gamma \gamma}\right) \delta_{\alpha \beta}\right)$

$\sigma_{\alpha \beta, \beta}+b_{\alpha}=0$

### Plane Stress

$\sigma_{\alpha \beta}=\sigma_{\alpha \beta}\left(x_{1}, x_{2}\right), \quad \sigma_{33}=\sigma_{13}=\sigma_{23}=0$

### Plane Stress/Strain

$\epsilon_{13}=\epsilon_{23}= \sigma_{13}=\sigma_{23}=0$

|                 |                                    Plane Stress                                   |                              Plane Strain                             |   |   |
|:---------------:|:---------------------------------------------------------------------------------:|:---------------------------------------------------------------------:|:-:|:-:|
|                 | $\sigma _{\alpha  \beta }=\sigma _{\alpha  \beta}\left(x_{1}, x_{2}\right)$ | $u_{\alpha  }=u_{\alpha }\left(x_{1}, x_{2}\right)\\$ $u_{3}=0$ |   |   |
|        s        |                                $\frac {1}{1+\nu }$                                |                                $1-\nu$                                |   |   |
|  $\sigma_{33}$  |                                        $0$                                        |                    $\nu  \sigma _{\alpha  \alpha }$                   |   |   |
| $\epsilon_{33}$ |                $-\frac {\nu }{1-\nu } \epsilon _{\alpha  \alpha }$                |                                  $0$                                  |   |   |
|                 |                                                                                   |                                                                       |   |   |



$$
\text{Navier}:\left\{\begin{array}{cl}{\left(\frac{E}{2(1+\nu)}\right) u_{\alpha, \beta \beta}+\left(\frac{E}{2(1+\nu)(1-2 \nu)}\right) u_{\beta, \beta \alpha}+b_{\alpha}=0} & {\text { for plane strain }} \\ {\left(\frac{E}{2(1+\nu)}\right) u_{\alpha, \beta \beta}+\frac{E}{2(1-\nu)} u_{\beta, \beta \alpha}+b_{\alpha}=0} & {\text { for plane stress }}\end{array}\right.
$$
<!-- 
$$
\sigma_{33}=\left\{\begin{array}{ll}{\nu \sigma_{\alpha \alpha}} & {\text { for plane strain }} \\ {0} & {\text { for plane stress. }}\end{array}\right.
$$
$$
s \stackrel{\text { def }}{=}\left\{\begin{array}{ll}{1-\nu} & {\text { for plane strain }} \\ {\frac{1}{1+\nu}} & {\text { for plane stress }}\end{array}\right.
$$ -->

**Constitutive Relation**
$$
\sigma_{\alpha \beta}=\frac{E}{(1+\nu)}\left(\epsilon_{\alpha \beta}+\left(\frac{1-s}{2 s-1}\right)\left(\epsilon_{\gamma \gamma}\right) \delta_{\alpha \beta}\right)
$$
$$
\epsilon_{\alpha \beta}=\frac{(1+\nu)}{E}\left(\sigma_{\alpha \beta}-(1-s)\left(\sigma_{\gamma \gamma}\right) \delta_{\alpha \beta}\right)
$$
**Equilibrium**
$$
\sigma_{\alpha \beta, \beta}+b_{\alpha}=0
$$

**Compatibility**
$$
\Delta\left(\sigma_{\alpha \alpha}\right)=\left(\sigma_{11}+\sigma_{22}\right),_{11}+\left(\sigma_{11}+\sigma_{22}\right),_{22}=-\frac{1}{s} b_{\alpha, \alpha}
$$


## Airy Stress Function

$\sigma_{11}=\varphi,_{22}, \quad \sigma_{22}=\varphi_{, 11}, \quad \sigma_{12}=-\varphi_{, 12}$

**Compatibility**

$\Delta \Delta \varphi=\varphi, 1111+2 \varphi, 1122+\varphi, 2222=0$

**Displacements**

$u_{1}=\frac{(1+\nu)}{E}(-\varphi, 1+s \psi, 2)+w_{1}\\$
$u_{2}=\frac{(1+\nu)}{E}\left(-\varphi, 2+s \psi,_{1}\right)+w_{2}\\$
where:
$\Delta \psi =0$ and $\psi_{, 12} =\Delta \varphi$

and w is a plane rigid displacement:\
$w_{1,1}=0, \quad w_{2,2}=0, \quad w_{1,2}+w_{2,1}=0$

**Polar form** (9.4.16):\
$\begin{aligned} \sigma_{r r} &=\frac{1}{r} \frac{\partial \varphi}{\partial r}+\frac{1}{r^{2}} \frac{\partial^{2} \varphi}{\partial \theta^{2}} \\ \sigma_{\theta \theta} &=\frac{\partial^{2} \varphi}{\partial r^{2}} \\ \sigma_{r \theta} &=-\frac{\partial}{\partial r}\left(\frac{1}{r} \frac{\partial \varphi}{\partial \theta}\right) \end{aligned}$



## Torsion

$\begin{array}{l}{u_{1}(\mathrm{x})\approx-\alpha x_{2} x_{3}} \\ {u_{2}(\mathrm{x})\approx\alpha x_{1} x_{3}} \\ u_{3}(\mathbf{x})=\alpha \varphi\left(x_{1}, x_{2}\right)\end{array}$

$\alpha=\frac{T}{\mu \bar{J}}$

$T=\int_{\mathcal{S}_{L}}\left(x_{1} \sigma_{23}-x_{2} \sigma_{13}\right) d a$

$\bar{J} \stackrel{\text { def }}{=} \int_{\Omega}\left(x_{1}^{2}+x_{2}^{2}+x_{1} \varphi_{, 2}-x_{2} \varphi,_{1}\right) d a$

For open sections:
$$
\mu \bar{J} =\mu\left(b_{1}+b_{2}+b_{3}\right) t^{3} / 3
$$

### Displacement Formulations

${\epsilon_{11}(\mathrm{x})=\epsilon_{22}(\mathrm{x})=\epsilon_{33}(\mathrm{x})=\epsilon_{12}(\mathrm{x})=0}\\$
$\epsilon_{13}(\mathrm{x})=\frac{1}{2}\left(\frac{\partial \varphi}{\partial x_{1}}-x_{2}\right) \alpha\\$
$\epsilon_{23}(\mathrm{x})=\frac{1}{2}\left(\frac{\partial \varphi}{\partial x_{2}}+x_{1}\right) \alpha\\$

$\sigma_{11}(\mathrm{x})=\sigma_{22}(\mathrm{x})=\sigma_{33}(\mathrm{x})=\sigma_{12}(\mathrm{x})=0 \\$
$\sigma_{13}(\mathrm{x})=\mu \alpha\left(\frac{\partial \varphi}{\partial x_{1}}-x_{2}\right)\\$
$\sigma_{23}(\mathrm{x})=\mu \alpha\left(\frac{\partial \varphi}{\partial x_{2}}+x_{1}\right)$

##### Equilibrium: 

$\sigma_{13,1}+\sigma_{23,2}=0 \quad \text { in } \Omega$

##### Boundary: 
$\Delta \varphi=0 \quad \text { in } \Omega\\$
$\frac{\partial \varphi}{\partial n}=x_{2} n_{1}-x_{1} n_{2} \quad \text { on } \Gamma$ 

### Stress Formulation

##### Compatibility:

$\epsilon_{13,2}-\epsilon_{23,1}=-\alpha \quad \text { in } \Omega\\$
$\Delta \Psi=\Psi_{, 11}+\Psi_{, 22} = -2 \mu \alpha \quad \text{ in  } \Omega$
$\text { subject to } \quad \Psi=0 \quad \text { on } \Gamma\\$
for $\sigma_{13}=\frac{\partial \Psi}{\partial x_{2}}, \quad \sigma_{23}=-\frac{\partial \Psi}{\partial x_{1}}$

$T=2 \int_{\Omega} \Psi d a$

## 9 Solutions

### Crack Tip

#### Mode III: Anti-plane tearing

$\Delta u_{z}=0 = \frac{\partial^{2} u_{z}}{\partial r^{2}}+\frac{1}{r^{2}} \frac{\partial^{2} u_{z}}{\partial \theta^{2}}+\frac{1}{r} \frac{\partial u_{z}}{\partial r}$

$\delta_{3} \stackrel{\text { def }}{=} u_{3}(r,+\pi)-u_{3}(r,-\pi)=\frac{4}{\mu} K_{\mathrm{III}} \sqrt{\frac{r}{2 \pi}}$

$\left(\begin{array}{c}{\sigma_{13}} \\ {\sigma_{23}}\end{array}\right)=\frac{K_{\mathrm{III}}}{\sqrt{2 \pi r}}\left(\begin{array}{c}{-\sin \left(\frac{\theta}{2}\right)} \\ {\cos \left(\frac{\theta}{2}\right)}\end{array}\right)+\text{ bounded terms}$

$\sigma_{11}=\sigma_{22}=\sigma_{33}=\sigma_{12}=0$

$\left(u_{3}\right)=\frac{K_{\mathrm{M}}}{2 \mu} \sqrt{\frac{r}{2 \pi}}\left(4 \sin \left(\frac{\theta}{2}\right)\right)+$ rigid displacenent.