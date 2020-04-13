# Calculus

## Fundamentals

$\frac{d}{d x} \int_{a}^{x} f(t) d t=f(x)$

## Differentiation

In the case of a vector field, the directional derivative is also a vector each of whose components gives the rate of change of the corresponding component of v in the direction of h. The gradient in this case will be a tensor field (that when applied to h gives the directional derivative of v in the direction of h).

<!-- Given a [[coordinate system]] $x^i$ for $i= 1, 2, ..., n$ on an $n$-manifold $M$, the [[tangent space|tangent vectors]] -->

$$\mathbf{e}_i = \frac{\partial}{\partial x^i} = \partial_i,\quad i = 1,\, 2,\, \dots,\, n$$

define what is referred to as the local [[basis of a vector space|basis]] of the tangent space to {{math|''M''}} at each point of its domain. These can be used to define the [[metric tensor]]:
$$g_{ij} = \mathbf{e}_i \cdot \mathbf{e}_j$$

and its inverse:

$$g^{ij} = \left( g^{-1} \right)_{ij}$$

which can in turn be used to define the dual basis:

$$\mathbf{e}^i = \mathbf{e}_j g^{ji},\quad i = 1,\, 2,\, \dots,\, n$$

Some texts write $\mathbf{g}_i$ for $\mathbf{e}_i$, so that the metric tensor takes the particularly beguiling form $g_{ij} = \mathbf{g}_i \cdot \mathbf{g}_j$. This convention also leaves use of the symbol $e_i$ unambiguously for the [[vierbein]].

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

## Integration

##### Integration by parts

If $u = u(x)$ and $du = u'(x) dx$, while $v = v(x)$ and $dv = v'(x) dx$, then integration by parts states that:

$$\begin{aligned}
\int_{a}^{b} u(x) v^{\prime}(x) d x &=[u(x) v(x)]_{a}^{b}-\int_{a}^{b} u^{\prime}(x) v(x) d x \\
&=u(b) v(b)-u(a) v(a)-\int_{a}^{b} u^{\prime}(x) v(x) d x
\end{aligned}$$

$$\int u \, d v=u v-\int v \, d u$$

$$\int_{a}^{b} u d v=\left.u v\right|_{a} ^{b}-\int_{a}^{b} v d u$$

Green's Theorem 

$$\int \int_{\text {Area }}\left(\frac{\partial Q}{\partial x}-\frac{\partial P}{\partial y}\right) d x d y=\oint_{C} P d x+\oint_{C} Q d y$$

Divergence Theorem

$$\int_{D} \nabla \cdot \mathbf{Q} d V=\int_{\partial D} \hat{\mathbf{n}} \cdot \mathbf{Q} d S$$

##### Divergence Theorem

$\begin{array}{rl}{\int \varphi n_{i} d a} & {=\int_{R} \frac{\partial \varphi}{\partial x_{i}} d v} \\ {\int_{\partial R} v_{i} n_{i} d a} & {=\int_{R} \frac{\partial v_{i}}{\partial x_{i}} d v} \\ {\int_{\partial R} T_{i j} n_{j} d a} & {=\int_{R} \frac{\partial T_{i j}}{\partial x_{j}} d v} \\ {\partial R} & {R}\end{array}$
