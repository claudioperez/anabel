
# I Math

## Calculus

$\frac{d}{d x} \int_{a}^{x} f(t) d t=f(x)$

## Calculus of Variations

*First variation* of a functional, $I$:
$\left.\delta I\{u ; w\} \stackrel{\text { def }}{=} \frac{d}{d \zeta} I\{u(x)+\zeta w(x)\}\right|_{\zeta=0}$

The condition:
$\delta I\{u ; w\}=0$
for all admissible $w$ is a necessary condition for $u(x)$ to be a minimizer of $I$.

**Euler-Lagrance Equation**
$$\frac{\partial F}{\partial u}-\frac{d}{d x}\left(\frac{\partial F}{\partial u^{\prime}}\right)=0 \quad \forall x \in\left(x_{0}, x_{1}\right)$$

**Taking variations**

$\delta I=\left.\frac{d}{d \zeta} I(\Psi+\zeta \delta \Psi)\right|_{\zeta=0}$

$\delta F=\left(\frac{\partial F}{\partial u}\right) \delta u+\left(\frac{\partial F}{\partial u^{\prime}}\right) \delta u^{\prime}$

$\begin{aligned} \delta I & \equiv \int_{x_{0}}^{x_{1}} \delta F\left(x, u, u^{\prime}\right) d x \\ &=\int_{x_{0}}^{x_{1}}\left(\frac{\partial F}{\partial u} \delta u+\frac{\partial F}{\partial u^{\prime}} \delta u^{\prime}\right) d x \\ &=\left.\frac{\partial F}{\partial u^{\prime}} \delta u\right|_{x_{0}} ^{x_{1}}+\int_{x_{0}}^{x_{1}}\left[\frac{\partial F}{\partial u}-\frac{d}{d x}\left(\frac{\partial F}{\partial u^{\prime}}\right)\right] \delta u d x \end{aligned}$

$\left.\delta H \equiv \delta\left(u^{\prime}\right) \stackrel{\text { def }}{=} \frac{d}{d \zeta} H\{u+\zeta \delta u\}\right|_{\zeta=0}=\left.\frac{d}{d \zeta}\left(u^{\prime}+\zeta(\delta u)^{\prime}\right)\right|_{\zeta=0}=(\delta u)^{\prime}$

## Tensors

### 1 Algebra

#### 1.2 Tensors

$S_{i j} \stackrel{\text { def }}{=} \mathbf{e}_{i} \cdot \mathbf{S} \mathbf{e}_{j}$

$C_{i j k l} \stackrel{\text { def }}{=}\left(\mathbf{e}_{i} \otimes \mathbf{e}_{j}\right): \mathbb{C}\left(\mathbf{e}_{k} \otimes \mathbf{e}_{l}\right)\\$

$\mathbf{S}=S_{i j} \mathbf{e}_{i} \otimes \mathbf{e}_{j}$


$v_i =S_{i j} u_{j}$

$\mathbb{I}^{\mathrm{sym}} \rightarrow \frac{1}{2}\left(\delta_{i k} \delta_{j l}+\delta_{i l} \delta_{j k}\right)$

##### Products

###### 1.2.3 Outer Product

$$(\mathbf{u} \otimes \mathbf{v}) \mathbf{w}=(\mathbf{v} \cdot \mathbf{w}) \mathbf{u}$$

###### 1.2.8

$$(\mathbf{S T})_{i j}=S_{i k} T_{k j}$$

###### 1.2.11 Inner / Dot Products

$$\mathbf{u} \cdot \mathbf{v}=\mathbf{v} \cdot \mathbf{u}=u_{i} v_{j} \delta_{ij} =u_{i} v_{i}\\\mathbf{S}: \mathbf{T}=\mathbf{T}: \mathbf{S}=S_{i j} T_{i j}\\\mathbf{S}\cdot \mathbf{T}=S_{i j} T_{j l}\mathrm{e}_{i}\mathrm{e}_{l}\neq \mathbf{T}\cdot \mathbf{S}$$

###### 1.2.1 Cross Product

$$(\mathbf{u} \times \mathbf{v})_{i}=e_{i j k} u_{j} v_{k}$$

##### 6 Symmetric - Skew components

$$\begin{array}{c}{\mathbf{S}=\mathbf{S}^{\top}, \quad S_{i j}=S_{j i}} \\ {\mathbf{\Omega}=-\mathbf{\Omega}^{\top}, \quad \Omega_{i j}=-\Omega_{j i}}\end{array}$$

$$\begin{array}{l}{(\operatorname{sym} \mathbf{T})_{i j}=\frac{1}{2}\left(T_{i j}+T_{j i}\right)} \\ {(\operatorname{skw} \mathbf{T})_{i j}=\frac{1}{2}\left(T_{i j}-T_{j i}\right)}\end{array}$$


##### 11 - Inner Product & Norm


$$|\mathbf{u}|=\sqrt{\mathbf{u} \cdot \mathbf{u}}=\sqrt{u_{i} u_{i}}\\|\mathbf{S}|=\sqrt{\mathbf{S}: \mathbf{S}}=\sqrt{S_{i j} S_{i j}}$$

##### 16 - Orthogonal tensors

$\begin{array}{c}{\mathbf{Q}^{\top} \mathbf{Q}=\mathbf{Q} \mathbf{Q}^{\top}=\mathbf{1}} \\ {\operatorname{det} \mathbf{Q}=\pm 1}\end{array}$

##### 17 - Transformations

$$v_{i}^{*}=\mathrm{e}_{i}^{*} \cdot \mathrm{v} \quad \text { and } \quad S_{i j}^{*}=\mathrm{e}_{i}^{*} \cdot \mathrm{Se}_{j}^{*}$$

$$\mathrm{Q} \stackrel{\text { def }}{=}  \mathrm{e}_{k} \otimes \mathrm{e}_{k}^{*}$$

$$v_{i}^{*}=Q_{i j} v_{j}\\S_{i j}^{*}=Q_{i k} Q_{j l} S_{k l}
\\C_{i j k l}^{*}=Q_{i p} Q_{j q} Q_{k r} Q_{l s} C_{p q r s}
$$

##### 18 - Eigenvalues

$$\omega^{3}-I_{1}(\mathbf{S}) \omega^{2}+I_{2}(\mathbf{S}) \omega-I_{3}(\mathbf{S})=0$$
Principal Invariants:
$$\begin{array}{l}{I_{1}(\mathbf{S})=\operatorname{tr} \mathbf{S}}  = \omega_1 + \omega_2 + \omega_3 \\ {I_{2}(\mathbf{S})=\frac{1}{2}\left[(\operatorname{tr}(\mathbf{S}))^{2}-\operatorname{tr}\left(\mathbf{S}^{2}\right)\right]}  = \omega_1 \omega_2 + \omega_2 \omega_3+ \omega_3 \omega_1 \\ {I_{3}(\mathbf{S})=\operatorname{det} \mathbf{S}} = \omega_1 \omega_2 \omega_3 \end{array}$$

### 2 Analysis

#### Derivatives

In the case of a vector field, the directional derivative is also a vector each of whose components gives the rate of change of the corresponding component of v in the direction of h. The gradient in this case will be a tensor field (that when applied to h gives the directional derivative of v in the direction of h).

Given a [[coordinate system]] {{math|''x''<sup>''i''</sup>}} for {{math|''i'' {{=}} 1, 2, …, ''n''}} on an {{math|''n''}}-manifold {{math|''M''}}, the [[tangent space|tangent vectors]]
:<math>\mathbf{e}_i = \frac{\partial}{\partial x^i} = \partial_i,\quad i = 1,\, 2,\, \dots,\, n</math>

define what is referred to as the local [[basis of a vector space|basis]] of the tangent space to {{math|''M''}} at each point of its domain. These can be used to define the [[metric tensor]]:
:<math>g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j</math>

and its inverse:

:<math>g^{ij} = \left( g^{-1} \right)_{ij}</math>

which can in turn be used to define the dual basis:

$$\mathbf{e}^i = \mathbf{e}_j g^{ji},\quad i = 1,\, 2,\, \dots,\, n$$

Some texts write <math>\mathbf{g}_i</math> for <math>\mathbf{e}_i</math>, so that the metric tensor takes the particularly beguiling form <math>g_{ij} = \mathbf{g}_i \cdot \mathbf{g}_j</math>. This convention also leaves use of the symbol <math>e_i</math> unambiguously for the [[vierbein]].


##### Gradient

$$ \begin{array}{c}{\mathbf{e}_{i} \cdot \operatorname{grad} \varphi(\mathbf{x})=[\operatorname{grad} \varphi(\mathbf{x})]_{i}=\frac{\partial \varphi(\mathbf{x})}{\partial x_{i}}}\\{\mathbf{e}_{i} \cdot \operatorname{grad} \mathbf{v}(\mathbf{x}) \mathbf{e}_{j}=[\operatorname{grad} \mathbf{v}(\mathbf{x})]_{i j}=\frac{\partial v_{i}(\mathbf{x})}{\partial x_{j}}}\end{array}$$

##### Jacobian

In general, the *derivative* of a function $f : \mathbb{R}^n \to \mathbb{R}^m$ at a point $p \in \mathbb{R}^n$, if it exists, is the unique linear transformation $Df(p) \in L(\mathbb{R}^n,\mathbb{R}^m)$ such that
$$
 \lim_{h \to 0} \frac{\|f(p+h)-f(p)-Df(p)h\|}{\|h\|} = 0;
$$
the matrix of $Df(p)$ with respect to the standard orthonormal bases of $\mathbb{R}^n$ and $\mathbb{R}^m$, called the *Jacobian matrix* of $f$ at $p$, therefore lies in $M_{m \times n}(\mathbb{R})$.

Now, suppose that $m=1$, so that $f : \mathbb{R}^n \to \mathbb{R}$. Then if $f$ is differentiable at $p$, $Df(p) \in L(\mathbb{R}^n,\mathbb{R}) = (\mathbb{R}^n)^\ast$ is a functional, and hence the Jacobian matrix, as you point out, lies in $M_{1 \times n}(\mathbb{R})$, i.e., is a row vector. However, by the Riesz representation theorem, $\mathbb{R}^n \cong (\mathbb{R}^n)^\ast$ via the map that sends a vector $x \in \mathbb{R}^n$ to the functional $y \mapsto \left\langle y,x \right\rangle$. Hence, if $f$ is differentiable at $p$, then the *gradient* of $f$ at $p$ is the unique (column!) vector $\nabla f(p) \in \mathbb{R}^n$ such that
$$
 \forall h \in \mathbb{R}^n, \quad Df(p)h = \left\langle \nabla f(p),h\right\rangle;
$$
in particular, if you unpack definitions, you'll find that the Jacobian matrix of $f$ at $p$ is precisely $\nabla f(p)^T$.

The Jacobian determinant can be viewed as the ratio of an infinitesimal change in the variables of one coordinate system to another. This requires that the function $f(x)$ maps $\mathbb{R}^n→\mathbb{R}^n$, which produces an $n×n$ square matrix for the Jacobian. For example:
$$\iiint_{R} f(x, y, z) d x d y d z=\iiint_{S} f(x(u, v, w), y(u, v, w), z(u, v, w))\left|\frac{\partial(x, y, z)}{\partial(u, v, w)}\right| d u d v d w$$

##### Directional Derivative

$$\operatorname{grad} \varphi(\mathbf{x})[\mathbf{h}]=\left.\frac{d}{d \alpha} \varphi(\mathbf{x}+\alpha \mathbf{h})\right|_{\alpha=0}$$

##### Divergence - Curl- Laplacian

$$\begin{aligned} \operatorname{div} \mathbf{v}=\operatorname{tr}[\operatorname{grad} \mathbf{v}] &=\frac{\partial v_{i}}{\partial x_{i}} \\(\operatorname{div} \mathbf{T})_{i} &=\frac{\partial T_{i j}}{\partial x_{j}} \\(\operatorname{curl} \mathbf{v})_{i} &=e_{i j k} \frac{\partial v_{k}}{\partial x_{j}} \\(\operatorname{curl} \mathbf{T})_{i j} &=e_{i p q} \frac{\partial T_{j q}}{\partial x_{p}} \end{aligned}$$

$$\Delta \mathbf{v}=\operatorname{div} \operatorname{grad} \mathbf{v}, \quad \Delta v_{i}=\frac{\partial^{2} v_{i}}{\partial x_{j} \partial x_{j}}\\$$
$$\Delta T_{i j}=\frac{\partial T_{i j}}{\partial x_{k} \partial x_{k}}$$

#### Integration

##### Integration by parts

If $u = u(x)$ and $du = u'(x) dx$, while $v = v(x)$ and $dv = v'(x) dx$, then integration by parts states that:

$$\begin{aligned}
\int_{a}^{b} u(x) v^{\prime}(x) d x &=[u(x) v(x)]_{a}^{b}-\int_{a}^{b} u^{\prime}(x) v(x) d x \\
&=u(b) v(b)-u(a) v(a)-\int_{a}^{b} u^{\prime}(x) v(x) d x
\end{aligned}$$

##### Divergence Theorem

$\begin{array}{rl}{\int \varphi n_{i} d a} & {=\int_{R} \frac{\partial \varphi}{\partial x_{i}} d v} \\ {\int_{\partial R} v_{i} n_{i} d a} & {=\int_{R} \frac{\partial v_{i}}{\partial x_{i}} d v} \\ {\int_{\partial R} T_{i j} n_{j} d a} & {=\int_{R} \frac{\partial T_{i j}}{\partial x_{j}} d v} \\ {\partial R} & {R}\end{array}$

## Step functions

### Heaviside Step Function, $h(t)$

$$h(t)=\left\{\begin{array}{l}{0 \text { for } t \leq 0} \\ {1 \text { for } t>0}\end{array}\right.$$

### Dirac Delta, $\delta$

$$\delta(t) \equiv \dot{h}(t)$$

$$\delta(t)=\left\{\begin{array}{ll}{0} & {\text { for } t \neq 0,} \\ {\infty} & {\text { for } t=0,}\end{array} \quad \text { where } \quad \int_{-\infty}^{\infty} \delta(t) d t=\int_{0^{-}}^{0^{+}} \delta(t) d t=1\right.$$

For any function continuous at $t=0$ :

$$\int_{-\infty}^{\infty} g(t) \delta(t) d t=\int_{0^{-}}^{0^{+}} g(t) \delta(t) d t=g(0)$$

## Convolution

$$
\int_{0^{-}}^{t} f(t-\tau) g(\tau) d \tau \equiv(f * g)(t)
$$

## Laplace Transformations

$$
L[f(t)]=\int_{0^{-}}^{\infty} e^{-s t} f(t) d t \equiv \bar{f}(s)
$$

The laplace transformation has the following properties:

 1. $L[(f * g)(t)]=L[f(t)] L[g(t)]=\bar{f}(s) \bar{g}(s)$

 2. $L[\dot{f}(t)]=s L[f(t)]-f\left(0^{-}\right)=s \bar{f}(s)-f\left(0^{-}\right)$