# 1 Plastic Analysis

## Upper Bound

$\dot{V}_{h p}=\mathbf{A}_{f} \dot{U}_{f} \quad \dot{U}_{f}=\mathbf{A}_{c p} \dot{U}_{c}$

$\dot{V}_{h p}=\mathbf{A}_{f} \mathbf{A}_{c p} \dot{U}_{c}=\mathbf{A}_{m p} \dot{U}_{c}$

$\begin{array}{c}{\dot{\mathcal{W}}_{e}=\mathcal{D}_{p}} \\ {\dot{U}_{f}^{T}\left(\lambda P_{r e f}+P_{c f}\right)=\left(\dot{V}_{h p}^{+}\right)^{T} Q_{p l}^{+}+\left(\dot{V}_{h p}^{-}\right)^{T} Q_{p l}^{-}}\end{array}$

$\begin{aligned} \dot{\boldsymbol{V}}_{h p}^{+} &=\left\{\begin{array}{ll}{\dot{\boldsymbol{V}}_{h p}} & {\text { if } \dot{\boldsymbol{V}}_{h p}>0} \\ {0} & {\text { otherwise }}\end{array}\right.\\ \dot{\boldsymbol{V}}_{h p}^{-}=\left\{\begin{array}{cc}{-\dot{\boldsymbol{V}}_{h p}} & {\text { if } \dot{\boldsymbol{V}}_{h p}<0} \\ {0} & {\text { otherwise }}\end{array}\right.\end{aligned}$

$\dot{V}_{h p}^{+}-\dot{V}_{h p}^{-}=\mathbf{A}_{f} \dot{U}_{f}$

$\lambda=\frac{\dot{V}_{h p}^{T} Q_{p l}}{\dot{U}_{f}^{T} P_{r e f}}=\frac{\mathbf{A}_{m p}^{T} Q_{p l}}{\mathbf{A}_{c p}^{T} P_{r e f}}$

## Lower Bound

## 4 Advanced

### 4.2 P-M Interaction

### 4.3 Deformation at Collapse

assumptions:

1) The applied loading consists only of the reference load $P_{r e f},$ which is increased monotonically by incrementing the load factor $\lambda$.

2) Plastic hinge deformations increase monotonically under the monotonically increasing reference load.
    $$
    \boldsymbol{V}_{\boldsymbol{\varepsilon}}=\mathbf{F}, \boldsymbol{Q}_{c}+\boldsymbol{V}_{0}
    $$

3) Select any hinge of the collapse mechanism as last to form and solve the kinematic relations for the corresponding displacements $U_{f}^{t}$, where the superseript tr stands for trial result.

4) Determine the plastic hinge deformations $V_{h p}^{\text {tr }}$ corresponding to $U_{f}^{\text {tr }}$ with
    $$
    \boldsymbol{V}=\boldsymbol{V}_{\varepsilon}+\boldsymbol{V}_{h p}^{t r}=\mathbf{A}_{f} \boldsymbol{U}_{f}^{t r}
    $$

5) If the sign of each plastic deformation matches the sign of the corresponding basic force $Q_{c}$ from the equilibrium equations, the last hinge location is correct. The trial displacements and plastic deformations from steps (1) and (2) give the free dof displacements $U_{f}$ and the plastic deformations $\boldsymbol{V}_{h p}$ at incipient collapse.

6) If the sign of one or more plastic deformations does not match the sign of the corresponding basic force, correct the free dof displacements and plastic deformations of Step (1) and (2) in a single step as described in the following.

If the sign of one or more plastic deformations does not match the sign of the corresponding basic