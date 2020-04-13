## Linear-Elastic Family of Frames

### (a) Kinematics

$$v=\hat{v}(\boldsymbol{u})$$

### (b) Force-deformation relation

$$\left(\begin{array}{l}
\boldsymbol{q}_{1} \\
\hline \boldsymbol{q}_{2} \\
\boldsymbol{q}_{3}
\end{array}\right)=\boldsymbol{q}=\mathbf{k} \boldsymbol{v} +\boldsymbol{q}_0=\left[\begin{array}{c|c}
\mathbf{k}_{a} & \mathbf{0} \\
\hline \mathbf{0} & \mathbf{k}_{b}
\end{array}\right]\left(\begin{array}{l}
\boldsymbol{v}_{1} \\
\hline \boldsymbol{v}_{2} \\
\boldsymbol{v}_{3}
\end{array}\right)+\boldsymbol{q}_0$$


$$\mathbf{k}_{a}=\frac{E A}{L}$$

$$\mathbf{k}_{b}=\frac{E I}{L}\left[\begin{array}{ll}
A & B \\
B & A
\end{array}\right] \quad \text { or } \quad \mathbf{k}_{b}=\frac{E I}{L}\left[\begin{array}{ll}
C & 0 \\
0 & 0
\end{array}\right] \quad \text { or } \quad \mathbf{k}_{b}=\frac{E I}{L}\left[\begin{array}{ll}
0 & 0 \\
0 & C
\end{array}\right]$$

### (c) Statics

$$\delta \boldsymbol{u}^{T} \boldsymbol{p}=\delta v^{T} q \qquad \boldsymbol{p}=\frac{\partial v}{\partial u}\boldsymbol{q}=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \boldsymbol{q}$$


### (d) Stiffness $\mathbf{k}_{e}=\mathbf{k}_{m}+\mathbf{k}_{g}$

- Material stiffness matrix

    $$\mathbf{k}_{m}=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \frac{\partial \boldsymbol{q}}{\partial v} \mathbf{a}_{g}(\boldsymbol{u})$$

- Geometric stiffness matrix $\mathbf{k}_{g}$
    $$
    \mathbf{k}_{g}=\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u})\right]}{\partial \boldsymbol{u}} q=\frac{q}{L}\left[\begin{array}{rrrr}
    1 & 0 & -1 & 0 \\
    0 & 1 & 0 & -1 \\
    -1 & 0 & 1 & 0 \\
    0 & -1 & 0 & 1
    \end{array}\right]
    $$


-----------------------------


<!-- $$q_{2}=\frac{E l}{L} \frac{\psi(\sin \psi-\psi \cos \psi)}{2-2 \cos \psi-\psi \sin \psi} v_{2}+\frac{E l}{L} \frac{\psi(\psi-\sin \psi)}{2-2 \cos \psi-\psi \sin \psi} v_{3}$$

$$q_{3}=\frac{E l}{L} \frac{\psi(\psi-\sin \psi)}{2-2 \cos \psi-\psi \sin \psi} v_{2}+\frac{E l}{L} \frac{\psi(\sin \psi-\psi \cos \psi)}{2-2 \cos \psi-\psi \sin \psi} v_{3}$$ -->

$$\psi=\bar{\psi} L=\sqrt{\frac{\left|q_{1}\right|}{E I}} L=\sqrt{\frac{\left|q_{1}\right| L^{2}}{E I}}$$

#### 1st Order: $A, B, C = 4, 2, 3$

#### 2nd Order Approximate Coefficients

<!-- for $q_i < 0$:
$$\begin{array}{l}
A=4-\frac{2}{15} \psi^{2} \\
B=2+1 / 30 \psi^{2} \\
C=3-1 / 5 \psi^{2}
\end{array}$$ -->

$$A=4+ \operatorname{sgn}(q_1) \frac{2}{15} \psi^{2}$$
$$B=2- \operatorname{sgn}(q_1) \frac{1}{30} \psi^{2}$$
$$C=3+ \operatorname{sgn}(q_1) \frac{1}{5 } \psi^{2}$$

#### Exact

$$A=\frac{\psi(\sin \psi-\psi \cos \psi)}{2-2 \cos \psi-\psi \sin \psi} $$
$$ B=\frac{\psi(\psi-\sin \psi)}{2-2 \cos \psi-\psi \sin \psi}$$

$$C=\frac{\psi^{2} \sin \psi}{\sin \psi-\psi \cos \psi}$$

#### Case A  $0.5<\psi \leq 2,$

<!-- $k$ consists of two contributions:

1. a contribution $k_{b l}$ equal to the basic stiffness under linear geometry, and

2. a contribution that accounts for the effect of axial basic force $q_{1}$ denoted with subscript $P \delta,$ noting that it is the second order approximation of the nonlinear intra-element geometry effect, which is linear in $q_{1}$ -->

$$\mathbf{k}_{b}=\mathbf{k}_{b l}+\mathbf{k}_{P \delta}$$

$$\mathbf{k}_{b}=\frac{E I}{L}\left[\begin{array}{ll}
4 & 2 \\
2 & 4
\end{array}\right]+\frac{q_{1} L}{30}\left[\begin{array}{rr}
4 & -1 \\
-1 & 4
\end{array}\right]=\mathbf{k}_{b l}+\mathbf{k}_{P \delta}$$
For a frame element with a moment release at end $j$ or $i$ the matrix $\mathbf{k}_{b}$ becomes

$\mathbf{k}_{b}=\mathbf{k}_{b l}+\mathbf{k}_{P \delta}$
with $\quad \mathbf{k}_{b l}=\frac{3 E l}{L}\left[\begin{array}{ll}1 & 0 \\ 0 & 0\end{array}\right] \quad$ or $\quad \frac{3 E l}{L}\left[\begin{array}{ll}0 & 0 \\ 0 & 1\end{array}\right]$
and $\quad k_{P \delta}=\frac{q_{1} L}{5}\left[\begin{array}{ll}1 & 0 \\ 0 & 0\end{array}\right] \quad$ or $\quad \frac{q_{1} L}{5}\left[\begin{array}{ll}0 & 0 \\ 0 & 1\end{array}\right]$

#### Case B $\psi > 2,$

If  then we subdivide the original element in the structural model into two elements of length $L / 2,$ thus halving the $\psi$ value of each new element, allowing us to use the stiffness coefficients of the second order approximation of the nonlinear intra-element geometry effect on the preceding page.


### Lecture 15

Projection matrix:
$$\begin{aligned}
&\mathbf{k}_{g a}=\frac{q_{1}}{L_{n}}\left[\begin{array}{cccc}
\left(\mathbf{1}-i_{d} i_{d}^{T}\right) & 0 & -\left(1-i_{d} i_{d}^{T}\right) & 0 \\
0 & 0 & 0 & 0 \\
-\left(1-i_{d} i_{d}^{T}\right) & 0 & \left(1-i_{d} i_{d}^{T}\right) & 0 \\
0 & 0 & 0 & 0
\end{array}\right]\\

&\mathbf{k}_{g b}=\frac{q_{2}+q_{3}}{L_{n}^{2}}\left[\begin{array}{cccc}
\left(i_{d} n_{d}^{T}+n_{d} i_{d}^{T}\right) & 0 & -\left(i_{d} n_{d}^{T}+n_{d} i_{d}^{T}\right) & 0 \\
-\left(i_{d} n_{d}^{T}+n_{d} i_{d}^{T}\right) & 0 & \left(i_{d} n_{d}^{T}+n_{d} i_{d}^{T}\right) & 0 \\
0 & 0 & 0 & 0
\end{array}\right]
\end{aligned}$$

## Inelastic Frames
FBE and DBE can not be modeled in the same way as they inherently differ from each other •Accuracy of the solution can be improved: –for FBE, by either increasing the NIPs (preferable from a computational standpoint) or the number of elements, –for DBE, only by increasing the number of elements. •In case of FBE, both local and global quantities converge fast  with increasing NIPs. •In case of DBE, higher derivatives converge slower to the exact solution and thus, accurate determination of local response quantities requires a finer finite-element mesh than the accurate determination of global response quantities. •Although computationally more expensive, FBE generally improves global and local response without mesh refinement. •To accurately capture local response of elements whose plastic hinges locations and lengths can be estimated, NIPs of a FBE has to be chosen such that integration weights at locations of plastic hinges match the plastic hinge lengths. 
### Displacement Formulation

PVD is used to formulate equilibrium between s(x) and q. This is “weak equilibrium” that leads to error in force boundary conditions. Thus, internal forces are not in equilibrium with element basic forces.  

$$\mathbf{q}=\int_{0}^{L} \mathbf{B}^{T}(x) \mathbf{s}(x) d x \approx \sum_{t=1}^{N_{p}} \mathbf{B}^{T}\left(x_{t}\right) \mathbf{s}\left(x_{t}\right) w_{t}$$

$$\mathbf{k} \equiv \frac{\partial \mathbf{q}}{\partial \mathbf{v}}=\int_{0}^{L} \mathbf{B}^{T}(x) \mathbf{k}_{s}(x) \mathbf{B}(x) d x \approx \sum_{t=1}^{N_{p}} \mathbf{B}^{T}\left(x_{t}\right) \mathbf{k}_{s}\left(x_{t}\right) \mathbf{B}\left(x_{t}\right) w_{t}$$

$$\mathbf{k}_{s}(x) \equiv \frac{\partial \mathbf{s}}{\partial \mathbf{e}}=\int_{A} \mathbf{a}_{s}^{T} \mathbf{E}_{T} \mathbf{a}_{s} d A=\sum_{t=1}^{N_{f t h e r}} \mathbf{a}_{s_{t}}^{T} \mathbf{E}_{T_{1}} \mathbf{a}_{s_{t}} A_{t}$$

$$\mathbf{E}_{T} \equiv \partial \boldsymbol{\sigma} / \partial \boldsymbol{\varepsilon}$$



