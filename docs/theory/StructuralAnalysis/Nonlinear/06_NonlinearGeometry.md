# 6 Nonlinear Geometry

$v:$ Intra-node deformations

## (a) Structural Kinematics

$$v=\hat{v}(\boldsymbol{u})$$

## (b) Force-deformation relation

$$q=\frac{E A}{L} v+q_{0}$$

## (c) Structural Statics

$$\delta \boldsymbol{u}^{T} \boldsymbol{p}=\delta v^{T} q \qquad \boldsymbol{p}=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \boldsymbol{q}$$


$$
a_{g}(u)=\frac{\partial v}{\partial u}
$$

## (d) Stiffness

$$
\mathbf{k}_{e}=\frac{\partial \boldsymbol{p}}{\partial \boldsymbol{u}}=\mathbf{k}_{m}+\mathbf{k}_{g}
$$

- Material stiffness matrix

$$\mathbf{k}_{m}=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \frac{\partial \boldsymbol{q}}{\partial v} \mathbf{a}_{g}(\boldsymbol{u})=\frac{E A}{L}\left[\begin{array}{cc}
\mathbf{N} & -\mathbf{N} \\
-\mathbf{N} & \mathbf{N}
\end{array}\right]$$

$$\mathbf{N}=\left[\begin{array}{cc}
\left(\frac{\Delta X+\Delta u_{x}}{L}\right)^{2} & \left(\frac{\Delta X+\Delta u_{x}}{L}\right)\left(\frac{\Delta Y+\Delta u_{y}}{L}\right) \\
\left(\frac{\Delta X+\Delta u_{x}}{L}\right)\left(\frac{\Delta Y+\Delta u_{y}}{L}\right) & \left(\frac{\Delta Y+\Delta u_{y}}{L}\right)^{2}
\end{array}\right]$$

- Geometric stiffness matrix $\mathbf{k}_{g}$
$$
\mathbf{k}_{g}=\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u})\right]}{\partial \boldsymbol{u}} q=\frac{q}{L}\left[\begin{array}{rrrr}
1 & 0 & -1 & 0 \\
0 & 1 & 0 & -1 \\
-1 & 0 & 1 & 0 \\
0 & -1 & 0 & 1
\end{array}\right]
$$

-------------

$$\delta v=\frac{\partial v}{\partial u} \delta u=\mathrm{a}_{g}(u) \delta u$$

$$\begin{aligned}
\mathbf{k}_{e}=\frac{\partial \boldsymbol{p}}{\partial \boldsymbol{u}}=\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u}) q\right]}{\partial \boldsymbol{u}} &=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \frac{\partial q}{\partial \boldsymbol{u}}+\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u})\right]}{\partial \boldsymbol{u}} q \\
&=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \frac{\partial q}{\partial v} \frac{\partial v}{\partial \boldsymbol{u}}+\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u})\right]}{\partial \boldsymbol{u}} q \\
&=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \frac{\partial q}{\partial v} \mathbf{a}_{g}(\boldsymbol{u})+\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u})\right]}{\partial \boldsymbol{u}} q \\
&=\mathbf{k}_{m} \quad+\mathbf{k}_{g}
\end{aligned}$$

## Felippa

$$e=\mathbf{B}_{0} \mathbf{u}+\frac{1}{2} \mathbf{u}^{T} \mathbf{H} \mathbf{u}$$
$$\mathbf{K}=A_{0} L_{0} \mathbf{B}^{T} \frac{\partial s}{\partial \mathbf{u}}+A_{0} L_{0} s \frac{\partial \mathbf{B}^{T}}{\partial \mathbf{u}}=\mathbf{K}_{M}+\mathbf{K}_{G}$$
$$\mathbf{K}=\frac{\partial \mathbf{p}}{\partial \mathbf{u}}=\frac{\partial}{\partial \mathbf{u}}\left(V_{0} s \frac{\partial e}{\partial \mathbf{u}}\right)=V_{0} E \frac{\partial e}{\partial \mathbf{u}} \otimes \frac{\partial e}{\partial \mathbf{u}}+V_{0} s \frac{\partial^{2} e}{\partial \mathbf{u} \partial \mathbf{u}}=\mathbf{K}_{M}+\mathbf{K}_{G}$$

## 6.2 Nonlinear Truss
<!-- ## 6.3 SDF Truss

| Deformation | Variation | Statics | Basic force |  
|:----------------------------------------------:|:------------------------------------------------:|:--------------------------------------:|:------------------------------------------------------------------------------:|
| $v_{L}=\frac{\Delta Y}{L} U$ | $\delta v_{L}=\frac{\Delta Y}{L} \delta U$ | $P_{r}=\frac{\Delta Y}{L} q_{L}$ | $q_{L}=\frac{E A}{L}\left(\frac{\Delta Y}{L} U\right)+q_{0}$ |  
| $v_{G L}=\frac{\Delta Y}{L} U+\frac{U^{2}}{2 L}$ | $\delta v_{G L}=\frac{\Delta Y+U}{L} \delta U$ | $P_{r}=\frac{\Delta Y+U}{L} q_{G L}$ | $q_{G L}=\frac{E A}{L}\left(\frac{\Delta Y}{L} U+\frac{U^{2}}{2 L}\right)+q_{0}$ |  
| $v_{R E}=L_{n}-L$ | $\delta v_{R E}=\frac{\Delta Y+U}{L_{n}} \delta U$ | $P_{r}=\frac{\Delta Y+U}{L_{n}} q_{R E}$ | $q_{R E}=\frac{E A}{L}\left(L_{n}-L\right)+q_{0}$ |

### 6.4 Structure Stiffness

$$K_t = \dfrac{EA}{L^3}(\Delta Y + U)^2 + \dfrac{EA}{L^3}\left( \Delta Y + \dfrac{U}{2}\right)U$$

$$K_t = \dfrac{EA}{L^3}(\Delta Y + U)^2 + \dfrac{EA}{L^3}\left( \Delta Y + \dfrac{U}{2}\right)U$$ -->





## (a) Kinematics

### Green-Lagrange Strain

$$v_{G L}=\frac{\Delta X}{L} \Delta \boldsymbol{u}_{x}+\frac{\Delta Y}{L} \Delta \boldsymbol{u}_{y}+\frac{\left(\Delta \boldsymbol{u}_{x}\right)^{2}}{2 L}+\frac{\left(\Delta \boldsymbol{u}_{y}\right)^{2}}{2 L}$$

## (b) Force-deformation relation

$$q=\frac{E A}{L} v+q_{0}$$

## (c) Statics

$$\delta \boldsymbol{u}^{T} \boldsymbol{p}=\delta v^{T} q \qquad \boldsymbol{p}=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \boldsymbol{q}$$


$$
a_{g}(u)=\frac{\partial v}{\partial u}=\left[-\frac{\Delta X+\Delta u_{x}}{L} \quad-\frac{\Delta Y+\Delta u_{y}}{L} \quad \frac{\Delta X+\Delta u_{x}}{L} \quad \frac{\Delta Y+\Delta u_{y}}{L}\right]
$$

## (d) Stiffness

$$
\mathbf{k}_{e}=\frac{\partial \boldsymbol{p}}{\partial \boldsymbol{u}}=\mathbf{k}_{m}+\mathbf{k}_{g}
$$

- Material stiffness matrix

$$\mathbf{k}_{m}=\mathbf{a}_{g}^{T}(\boldsymbol{u}) \frac{\partial \boldsymbol{q}}{\partial v} \mathbf{a}_{g}(\boldsymbol{u})=\frac{E A}{L}\left[\begin{array}{cc}
\mathbf{N} & -\mathbf{N} \\
-\mathbf{N} & \mathbf{N}
\end{array}\right]$$

$$\mathbf{N}=\left[\begin{array}{cc}
\left(\frac{\Delta X+\Delta u_{x}}{L}\right)^{2} & \left(\frac{\Delta X+\Delta u_{x}}{L}\right)\left(\frac{\Delta Y+\Delta u_{y}}{L}\right) \\
\left(\frac{\Delta X+\Delta u_{x}}{L}\right)\left(\frac{\Delta Y+\Delta u_{y}}{L}\right) & \left(\frac{\Delta Y+\Delta u_{y}}{L}\right)^{2}
\end{array}\right]$$

- Geometric stiffness matrix $\mathbf{k}_{g}$
$$
\mathbf{k}_{g}=\frac{\partial\left[\mathbf{a}_{g}^{T}(\boldsymbol{u})\right]}{\partial \boldsymbol{u}} q=\frac{q}{L}\left[\begin{array}{rrrr}
1 & 0 & -1 & 0 \\
0 & 1 & 0 & -1 \\
-1 & 0 & 1 & 0 \\
0 & -1 & 0 & 1
\end{array}\right]
$$

## Prestressed Cable

$$U = -\dfrac{q_0}{k_s + \frac{EA}{L}} \qquad q_r = k_sv_s=-k_sU$$