
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

In the case of a vector field, the directional derivative is also a vector each of whose components gives the rate of change of the corresponding component of v in the direction of h. The gradient in this case will be a tensor field (that when applied to h gives the directional derivative of v in the direction of h).

##### Gradient

$$ \begin{array}{c}{\mathbf{e}_{i} \cdot \operatorname{grad} \varphi(\mathbf{x})=[\operatorname{grad} \varphi(\mathbf{x})]_{i}=\frac{\partial \varphi(\mathbf{x})}{\partial x_{i}}}\\{\mathbf{e}_{i} \cdot \operatorname{grad} \mathbf{v}(\mathbf{x}) \mathbf{e}_{j}=[\operatorname{grad} \mathbf{v}(\mathbf{x})]_{i j}=\frac{\partial v_{i}(\mathbf{x})}{\partial x_{j}}}\end{array}$$

##### Directional Derivative

$$\operatorname{grad} \varphi(\mathbf{x})[\mathbf{h}]=\left.\frac{d}{d \alpha} \varphi(\mathbf{x}+\alpha \mathbf{h})\right|_{\alpha=0}$$

##### Divergence - Curl- Laplacian

$$\begin{aligned} \operatorname{div} \mathbf{v}=\operatorname{tr}[\operatorname{grad} \mathbf{v}] &=\frac{\partial v_{i}}{\partial x_{i}} \\(\operatorname{div} \mathbf{T})_{i} &=\frac{\partial T_{i j}}{\partial x_{j}} \\(\operatorname{curl} \mathbf{v})_{i} &=e_{i j k} \frac{\partial v_{k}}{\partial x_{j}} \\(\operatorname{curl} \mathbf{T})_{i j} &=e_{i p q} \frac{\partial T_{j q}}{\partial x_{p}} \end{aligned}$$

$$\Delta \mathbf{v}=\operatorname{div} \operatorname{grad} \mathbf{v}, \quad \Delta v_{i}=\frac{\partial^{2} v_{i}}{\partial x_{j} \partial x_{j}}\\$$
$$\Delta T_{i j}=\frac{\partial T_{i j}}{\partial x_{k} \partial x_{k}}$$

#### Integration

Divergence Theorem

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