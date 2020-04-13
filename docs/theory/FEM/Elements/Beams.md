# Mindlin Beam

$$\begin{array}{lcc}
\hline & \text { Bernoulli-Navier [7, kap. II.2] } & \text { Mindlin } \\
\hline \text { Valid for } & h / L<1 / 10 & h / L<1 / 3 \\
\text { Cross-section } & \text { planar, perpendicular } & \text { planar } \\
\gamma_{z x} & 0 & \neq 0 \text { (shear effects) } \\
\text { Unknowns } & w(x) & w(x), \varphi_{y}(x) \\
& \varphi_{y}(x)=-\frac{\mathrm{d} w(x)}{\mathrm{d} x} & \text { independent } \\
\hline
\end{array}$$

$$\begin{array}{lcc}
\hline & \text {Bernoulli-Navier} & \text {Mindlin} \\
\hline \text { Constitutive eqs: } \tau=G \gamma & 0 & \text { constant } \\
\text { Equilibrium eqs } & \text { quadratic } & ? \\
& [7, \text { kap. } \text { II. } 2.5] & \\
\hline
\end{array}$$

Kinematics

$$u_{1}(x, y)=u(x)-y \theta(x) \quad \text { and } \quad u_{2}=w(x)$$

$$\begin{aligned}
&\epsilon_{1}=\frac{\partial u}{\partial x}-y \frac{\partial \theta}{\partial x}=\epsilon(x)-y \chi(x)\\
&\gamma_{12}=\frac{\partial w}{\partial x}-\theta=\gamma(x)
\end{aligned}$$

$$\kappa=\frac{\mathrm{y}^{\prime \prime}}{\left[1+\left(\mathrm{y}^{\prime}\right)^{2}\right]^{3 / 2}} \approx-\frac{\mathrm{d}^{2} w}{\mathrm{d} x^{2}}$$

Equilibrium
$$\frac{\partial N}{\partial x}+b_{x}=0 \qquad N\left(\epsilon_{1}\right)=\int_{A} \sigma_{1}(\epsilon) \mathrm{d} A$$


$$\frac{\partial V}{\partial x}+b_{y}=0 \qquad V=\int_{A} \tau \mathrm{d} A$$


$$\frac{\partial M}{\partial x}+V=0 \qquad M\left(\epsilon_{1}\right)=-\int_{A} y \sigma_{1}(\epsilon) \mathrm{d} A$$

Governing Eqn

$$\frac{d^{2}}{d x^{2}}\left[E I(x) \frac{d^{2} u_{y}}{d x^{2}}-q_{1} u_{y}(x)\right]=0$$

## Concentrated Plasticity

### Truss/Brace

### EPP Beam

### Two-Component Parallel Model

### One-Component Series Model

## Distributed Inelasticity