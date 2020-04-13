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
