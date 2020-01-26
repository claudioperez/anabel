# V Variational Formulation

## 10

### 10.1

#### Traction Equilibrium Condition: Weak Form

Virtual power balance,

$$\int_{\mathcal{B}}(\operatorname{sym} \nabla \mathbf{w}): \mathbb{C}(\operatorname{sym} \nabla \mathbf{u}) d v-\int_{\mathcal{S}_{2}} \hat{\mathbf{t}} \cdot \mathbf{w} d a-\int_{\mathcal{B}} \mathbf{b} \cdot \mathbf{w} d v=0 \quad \forall \text{ admissible w.}$$

is satisfied if and only if $\operatorname{div} \sigma+\mathrm{b}=0 \text{ in } \mathcal{B}$ and $\sigma \mathbf{n}=\hat{\mathbf{t}}$ on $\mathcal{S}_{2}$

Where $\mathbf{w}$ is a virtual velocity field such that $\mathbf{w}=\mathbf{0}$ on $\mathcal{S}_{1}$.

### 10.2

#### Elastostatic Displacement Problem: Weak Form

Given $\mathbb{C}$, $\mathbf{b}$, and boundary data $\mathbf{\hat{u}}$ and $\mathbf{\hat{t}}$, find a displacement field $\mathbf{u}$ equal to $\mathbf{\hat{u}}$ on $\mathcal{S}_{1}$ such that:

$$\int_{\mathcal{B}}(\operatorname{sym} \nabla \mathbf{w}): \mathbb{C}(\operatorname{sym} \nabla \mathbf{u}) d v-\int_{\mathcal{S}_{2}} \hat{\mathbf{t}} \cdot \mathbf{w} d a-\int_{\mathcal{B}} \mathbf{b} \cdot \mathbf{w} d v=0 \quad \forall \text{ admissible w.}$$