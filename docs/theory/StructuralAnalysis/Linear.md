# Linear Structural Analysis


## [1 Static Relations]()

$\mathbf{P}_f=\mathbf{B}_f\mathbf{Q}+\mathbf{P}_{wf}$

$\mathbf{Q} = \mathbf{B}^{-1}_i\left(\mathbf{P}_f-\mathbf{P}_{wf}\right)+\mathbf{B}^{-1}_x\mathbf{Q}_x$

## 2 Kinematic Relations

$\mathbf{V}_{\epsilon c}= \mathbf{A}_{f}\mathbf{U}_f+\mathbf{V}_{dc}$

$\mathbf{V} = \mathbf{V}_{\epsilon}+\mathbf{V}_{h} = \mathbf{\breve{A}}_f\mathbf{U}_f+\mathbf{V}_d \quad (5.46)$

## 3 Constitutive Relations

### Flexibility

$\mathbf{V}_{\epsilon c}= \mathbf{F}_{s}\mathbf{Q}+\mathbf{V}_{0}\quad$ $\boldsymbol{v} = \mathbf{fq} + v_0$


$$\mathbf{f}=\left[\begin{array}{l}{\frac{L}{E A}} & {0} & {0} \\ {0} & {\frac{L}{3 E I}} & {-\frac{L}{6 E I}} \\ {0} & {-\frac{L}{6 E I}} & {\frac{L}{3 E I}}\end{array}\right] \quad \boldsymbol{v}_{0}=\left(\begin{array}{c}{\varepsilon_{0} L+\Delta L_{0}} \\ {-\frac{\kappa_{0} L}{2}} \\ {\frac{\kappa_{0} L}{2}}\end{array}\right)+\left(\begin{array}{l}{0} \\ {-\frac{w L^{3}}{24 E I}} \\ {\frac{w L^{3}}{24 E I}}\end{array}\right)$$

### Stiffness

$\boldsymbol{q} = \mathbf{k}\boldsymbol{v} + \boldsymbol{q}_0$



## Virtual Work


## 8 Linear Elastic Structures

$\mathbf{U}_f = \mathbf{F}\mathbf{P}_f+\mathbf{U}_{0}$

## 9 Force Method

$$\begin{aligned} Q_{x}=&-\mathbf{F}_{x x}^{-1} \overline{\mathbf{B}}_{x}^{T} \underbrace{\left(\mathbf{F}_{s} Q_{p}+V_{0}\right)}_{V} \\ & \mathbf{F}_{x x}=\overline{\mathbf{B}}_{x}^{T} \mathbf{F}_{s} \overline{\mathbf{B}}_{x} \end{aligned}$$

$$\begin{array}{c}{Q=\overline{\mathbf{B}}\left(P_{f}-P_{w f}\right)+\overline{\mathbf{B}}_{v}\left(V_{0}-V_{d}\right)} \\ {\begin{aligned} \overline{\mathbf{B}}=\overline{\mathbf{B}}_{i}-\overline{\mathbf{B}}_{x} \mathbf{F}_{x x}^{-1} \mathbf{F}_{x i} & \\ \overline{\mathbf{B}}_{v}=&-\overline{\mathbf{B}}_{x} \mathbf{F}_{x x}^{-1} \overline{\mathbf{B}}_{x}^{T} \end{aligned}}\end{array}$$

## 10 Displacement Method

1. Set up kinematic matrix $A_f$

2. Set up initial nodal force vector P0 from the equivalent nodal force vector Pwf and the initial basic force vector $Q_0$:
   $$P_0 = P_{wf} + A^T_f Q_0$$

3. Set up stiffness matrix $K$ for the independent free dofs. Note that the stiffness coeffcient $K_{ij}$ represents the force at dof i due to a unit displacement at dof j with the displacements at all other independent free dofs equal to zero.
   $$K = A^T_f K_s A_f$$

4. Use: $P_f = K U_f + P_0$ to solve for the independent free dof displacements $U_f$

5. Determine the element deformations from the kinematic relations:
   $$V = A_fU_f$$

6. Determine the basic forces $Q$ of each frame element using the slope-deflection equations:
   $$q_i = \dfrac{2EI}{L}(2 \theta_i + \theta_j - 3\beta) + q_0$$
   $$q_j = \dfrac{2EI}{L}(2 \theta_j + \theta_i - 3\beta) + q_0$$

   or, for an element with moment release:
   $$q_i = \dfrac{3EI}{L}(\theta_i - \beta) + q_0$$

    Alternatively, the basic forces can be computed as:

    $$Q = K_s A_f U_f + Q_0$$

    The axial basic forces are determined from nodal equilibrium.

7. With the basic forces $Q$, determine the support reactions and check global equilibrium.

## 11 Linear Mechanisms

pass

## 12 Plastic Analysis

### Lower Bound 

#### Alternate Approach

### Upper Bound

$$\lambda=\frac{\dot{V}_{h p}^{T} Q_{p l}}{\dot{U}_{f}^{T} P_{r e f}}=\frac{\mathbf{A}_{m p}^{T} Q_{p l}}{\mathbf{A}_{c p}^{T} P_{r e f}}$$

$\dot{V}_{h p}=\mathbf{A}_{f} \dot{U}_{f}$

$\dot{U}_{f}=\mathbf{A}_{c p} \dot{U}_{c}$

$\dot{V}_{h p}=\mathbf{A}_{f} \mathbf{A}_{c p} \dot{U}_{c}=\mathbf{A}_{m p} \dot{U}_{c}$

$\begin{array}{c}{\dot{\mathcal{W}}_{e}=\mathcal{D}_{p}} \\ {\dot{U}_{f}^{T}\left(\lambda P_{r e f}+P_{c f}\right)=\left(\dot{V}_{h p}^{+}\right)^{T} Q_{p l}^{+}+\left(\dot{V}_{h p}^{-}\right)^{T} Q_{p l}^{-}}\end{array}$

$\begin{aligned} \dot{\boldsymbol{V}}_{h p}^{+} &=\left\{\begin{array}{ll}{\dot{\boldsymbol{V}}_{h p}} & {\text { if } \dot{\boldsymbol{V}}_{h p}>0} \\ {0} & {\text { otherwise }}\end{array}\right.\end{aligned}\\$

$\begin{aligned}\dot{\boldsymbol{V}}_{h p}^{-}=\left\{\begin{array}{cc}{-\dot{\boldsymbol{V}}_{h p}} & {\text { if } \dot{\boldsymbol{V}}_{h p}<0} \\ {0} & {\text { otherwise }}\end{array}\right.\end{aligned}$

$\dot{V}_{h p}^{+}-\dot{V}_{h p}^{-}=\mathbf{A}_{f} \dot{U}_{f}$



$\lambda=\frac{\mathbf{A}_{m p}^{T} Q_{p l}}{\mathbf{A}_{c p}^{T} P_{r e f}}$
