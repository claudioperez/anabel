# IIX Plasticity

## 19 Limits & Failure Criteria

----------------------------------
Shear flow strength:
$k = Y/\sqrt{3}$

#### Invariants

1. Mean normal pressure.

   $\bar{p}=-\frac{1}{3} \operatorname{tr} \sigma=-\frac{1}{3}\left(\sigma_{k k}\right)\\$

   $\bar{p}=-\frac{1}{3}\left(\sigma_{11}+\sigma_{22}+\sigma_{33}\right)$

2. Equivalent shear/tensile stress.

   $\bar{\tau}=\frac{1}{\sqrt{2}}\left|\sigma^{\prime}\right|=\sqrt{\frac{1}{2} \operatorname{tr}\left(\sigma^{\prime 2}\right)}=\sqrt{\frac{1}{2} \sigma_{i j}^{\prime} \sigma_{i j}^{\prime}}\\$

   $\bar{\tau}=\left[\frac{1}{6}\left(\left(\sigma_{11}-\sigma_{22}\right)^{2}+\left(\sigma_{22}-\sigma_{33}\right)^{2}+\left(\sigma_{33}-\sigma_{11}\right)^{2}\right)+\left(\sigma_{12}^{2}+\sigma_{23}^{2}+\sigma_{31}^{2}\right)\right]^{1 / 2}\\$

   $\bar{\sigma}=\sqrt{\frac{3}{2}}\left|\sigma^{\prime}\right|=\sqrt{\frac{3}{2} \operatorname{tr}\left(\sigma^{\prime 2}\right)}=\sqrt{\frac{3}{2} \sigma_{i j}^{\prime} \sigma_{i j}^{\prime}}\\$

   $\bar{\sigma}=\left[\frac{1}{2}\left(\left(\sigma_{11}-\sigma_{22}\right)^{2}+\left(\sigma_{22}-\sigma_{33}\right)^{2}+\left(\sigma_{33}-\sigma_{11}\right)^{2}\right)+3\left(\sigma_{12}^{2}+\sigma_{23}^{2}+\sigma_{31}^{2}\right)\right]^{1 / 2}\\$

   $\bar{\sigma}=\sqrt{3} \bar{\tau}$

3. Third stress invariant.

    $\bar{r}=\left(\frac{9}{2} \operatorname{tr}\left(\sigma^{\prime 3}\right)\right)^{\frac{1}{3}}=\left(\frac{9}{2} \sigma_{i k}^{\prime} \sigma_{k j}^{\prime} \sigma_{j k}^{\prime}\right)^{\frac{1}{3}}$

### Failure Criteria

1. **Von Mises**

   $f = \bar{\sigma} - Y \leq 0$

2. **Tresca**

   $f = \left(\sigma_{1}-\sigma_{3}\right) \leq Y$\
   $f = \tau_{\max } -\tau_{y, \text { Tresca }} \leq 0$

   $\tau_{\max } \stackrel{\text { def }}{=} \frac{1}{2}\left(\sigma_{1}-\sigma_{3}\right) \geq 0$\
   $\tau_{y, \text { Tresca }} \stackrel{\text { def }}{=} \frac{1}{2} Y$

3. **Mohr-Coulomb**

   $f = \frac{1}{2}\left(\sigma_{1}-\sigma_{3}\right)+\frac{1}{2}\left(\sigma_{1}+\sigma_{3}\right) \sin \phi-c \cos \phi \leq 0$

   $\mu=\tan \phi$

4. **Drucker-Prager**

   $f(\bar{\tau}, \bar{p}, S)=\bar{\tau}-(S+\alpha \bar{p})\leq 0 \\$

## 20 One-Dimensional Plasticity

----------------------------------

### 20.1 Elastic-Plastic Response

#### Isotropic hardening

#### Kinematic hardening

$\sigma_{b}=\frac{1}{2}\left(\sigma_{f}+\sigma_{r}\right)\\$
$\left|\sigma-\sigma_{b}\right| \leq Y_{0}$

### 20.2 Isotropic rate-independent theory

1. Kinematic decomposition

    $\epsilon=\epsilon^{e}+\epsilon^{p}$

2. Constitutive relation

    $\sigma=E \epsilon^{e}=E\left(\epsilon-\epsilon^{p}\right)$

3. Yield condition

   $f=|\sigma|-Y\left(\bar{\epsilon}^{\mathrm{p}}\right) \leq 0$

4. Flow rule

    $\dot{\epsilon}^{\mathrm{p}}=\chi \beta \dot{\epsilon}\\$
    $\beta=\dfrac{E}{E+H\left(\bar{\epsilon}^{\mathrm{P}}\right)}>0 \quad \text { (by hypothesis) }\\$

    $H\left(\bar{\epsilon}^{\mathrm{P}}\right)=\dfrac{d Y\left(\bar{\epsilon}^{\mathrm{P}}\right)}{d \bar{\epsilon}^{\mathrm{P}}}$

5. Kuhn-Tucker condition

    $\chi=0 \quad {\text { if } f<0, \text { or if } f=0 \text { and } n^{p} \dot{\epsilon}<0, \quad \text { where } \quad n^{p}=\operatorname{sign}(\sigma)}\\$

6. Consistency condition

    $\chi=1 \quad {\text { if } f=0 \quad \text { and } \quad n^{p} \dot{\epsilon}>0}$

$\dot{\sigma}=E[1-\chi \beta] \dot{\epsilon} \\$

$\dot{\sigma}=E_{\mathrm{tan}} \dot{\epsilon} \\$

${E_{\mathrm{tan}}=\left\{\begin{array}{ll}{E} & {\text { if } \chi=0} \\ {\frac{E H\left(\bar{\epsilon}^{\mathrm{P}}\right)}{E+H\left(\bar{\epsilon}^{\mathrm{P}}\right)}} & {\text { if } \chi=1}\end{array}\right.}$

## 21 3D plasticity with isotropic hardening

----------------------------------

### 21.1 Introduction

### 21.2 Basic equations

### 21.3 Kinematical assumptions

### 21.4 Separability hypothesis

### 21.5 Constitutive characterization of elastic response

### 21.6 Constitutive equations for plastic response

### 21.7 Summary of Mises-Hill Theory

1. Strain decomposition, (21.7.1)

    $$
    \epsilon=\epsilon^{\mathrm{e}}+\epsilon^{\mathrm{P}}
    $$

2. Constitutive relation (21.7.2)

    $$\sigma=2 \mu\left(\epsilon-\epsilon^{\mathrm{p}}\right)+(\kappa-(2 / 3) \mu)(\operatorname{tr} \epsilon) \mathbf{1}$$
    with $\mu$ > 0 and $\kappa$ > 0 the elastic shear and bulk moduli.

3. Yield condition (21.7.3)

    $$
    f=\bar{\sigma}-Y\left(\bar{\epsilon}^{\mathrm{P}}\right) \leq 0
    $$

4. Evolution equations, (21.7.5)
    $$
    \begin{aligned} \dot{\epsilon}^{\mathrm{p}} &=\chi \beta\left(\bar{\epsilon}^{\mathrm{p}}\right)\left(\mathbf{n}^{\mathrm{p}}: \dot{\epsilon}\right) \mathbf{n}^{\mathrm{p}}, \quad \mathbf{n}^{\mathrm{p}}=\sqrt{3 / 2} \frac{\sigma^{\prime}}{\bar{\sigma}} \\ \dot{\epsilon}^{\mathrm{p}} &=\sqrt{2 / 3}\left|\dot{\epsilon}^{\mathrm{p}}\right| \end{aligned}
    $$

    with\
    **Stiffness ratio:**
    $$
    \beta\left(\bar{\epsilon}^{\mathrm{p}}\right)=\frac{3 \mu}{3 \mu+H\left(\bar{\epsilon}^{\mathrm{p}}\right)}>0
    $$
    **Hardening modulus** (21.7.6):
    $$
    H\left(\bar{\epsilon}^{\mathrm{p}}\right)=\frac{d Y\left(\bar{\epsilon}^{\mathrm{p}}\right)}{d \bar{\epsilon}^{\mathrm{p}}}
    $$
    **Switching parameter** (21.7.7)
    $$
    \chi=\left\{\begin{array}{ll}{0} & {\text { if } f<0, \text { or if } f=0 \text { and } n^{\mathrm{p}}: \dot{\epsilon} \leq 0} \\ {1} & {\text { if } f=0 \text { and } n^{\mathrm{p}}: \dot{\epsilon}>0}\end{array}\right.
    $$
    and typical **initial conditions:**
    $$
    \epsilon(\mathrm{x}, 0)=\epsilon^{\mathrm{p}}(\mathrm{x}, 0)=0, \quad \text { and } \quad \bar{\epsilon}^{\mathrm{p}}(\mathrm{x}, 0)=0
    $$

Also note:
$$
d \epsilon_{i j}=\underbrace{\frac{1+\nu}{E} d \sigma_{i j}-\frac{\nu}{E}\left(d \sigma_{k k}\right) \delta_{i j}}_{d \epsilon_{i j}^{e}}+\underbrace{(3 / 2) d \bar{\epsilon}^{\mathrm{p}} \frac{\sigma_{i j}^{\prime}}{\bar{\sigma}}}_{d \epsilon_{i j}^{\mathrm{p}}}
$$



## 24 Classical problems in rate-independent plasticity

----------------------------------

### 24.1 Elastic-plastic torsion of a cylindrical bar 387

#### 24.1.1 Kinematics 387

#### 24.1.2 Elastic constitutive equation 388

#### 24.1.3 Equilibrium 388

#### 24.1.4 Resultant torque: Elastic case 388

##### Shear stress in terms of applied torque and geometry 389

#### 24.1.5 Total twist of a shaft 390

#### 24.1.6 Elastic-plastic torsion 390

##### Onset of yield 390

##### Torque-twist relation 392

##### Spring-back 394

##### Residual stress 395

### 24.2 Spherical pressure vessel 397

#### 24.2.1 Elastic analysis 397

#### 24.2.2 Elastic-plastic analysis 400

##### Onset of yield 400

##### Partly-plastic spherical shell 400

##### Fully-plastic spherical shell 403

##### Residual stresses upon unloading

## 25 Rigid-perfectly-plastic materials. Two extremum principles

----------------------------------

## 25.1 Mixed boundary value problem for a rigid-perfectly-plastic solid

### 25.3 Limit Analysis

### 25.4 Lower bound theorem

**Statically admissible** stress field, with respect to to traction field $\mathbf{t}^{*}$ satisfies:

1. Equilibrium, $\operatorname{div} \boldsymbol{\sigma}^{*}+\mathbf{b}^{*}=\mathbf{0} \quad$ on $\quad \mathcal{B}$

2. Traction B.Cs, $\sigma^{*} \mathbf{n}=\mathbf{t}^{*} \quad$ on $\quad \partial \mathcal{B}$

3. Yield condition: $f\left(\sigma^{*}, Y\right)=\sqrt{3 / 2}\left|\sigma^{* \prime}\right|-Y \leq 0 \quad$ in $\quad \mathcal{B}$

$Q = Q_{pl}$

### 25.5 Upper bound theorem

**Kinematically admissible** velocity field, $\mathbf{v}^{*}$, satisfies:

1. Stretching-velocity relation, $\dot{\epsilon}^{*}=\frac{1}{2}\left(\left(\nabla \mathbf{v}^{*}\right)+\left(\nabla \mathbf{v}^{*}\right)^{\top}\right)$;

2. gives no volume change, $\operatorname{tr} \dot{\epsilon}^{*}=0$

3. Satisfies velocity B.Cs, $\mathbf{v}^{*}=\hat{\mathbf{v}} \quad$ on $\quad \mathcal{S}_{1}$

**Upper bound:**
$$
\Phi\left\{\mathbf{v}^{*}\right\}=\mathcal{D}_{\mathrm{int}}\left\{\mathbf{v}^{*}\right\}-\mathcal{W}_{\mathrm{ext}}\left\{\mathbf{v}^{*}\right\} \ge\underbrace{\Phi\{\mathbf{v}\}}_{=0}
$$
$$
\beta_{U}=\frac{\sum_{\mathcal{S}_{\tilde{d}}} A_{\mathcal{S}_{d}^{*}} k\left[v^{*}\right]}{\int_{\mathcal{B}} \tilde{\mathbf{b}} \cdot \mathrm{v}^{*} d v+\int_{\partial \mathcal{B}} \tilde{\mathrm{t}} \cdot \mathrm{v}^{*} d a}=\frac{\sum_{\mathcal{S}_{\dot{A}}} A_{\mathcal{S}_{\dot{d}}} \cdot k\left[v^{*}\right]}{\int_{\mathcal{B}} \tilde{\mathbf{b}} \cdot \mathrm{v}^{*} d v+\int_{\partial \mathcal{B}} \tilde{\mathrm{t}} \cdot \mathrm{v}^{*} d a}
$$


#### 25.5.1 Block-sliding velocity fields


#### 25.6.2 Hodograph

#### 25.6.3 Plane strain frictionless extrusion

### 25.7 Plane-strain indentation of a semi-infinite solid with a flat punch

