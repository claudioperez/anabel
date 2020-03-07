# Govindjee

## Construction

First, the solution space and the weighting function space are replaced by finite dimensional subspaces. Second, these subspaces are employed in the weak form to generate matrix equations which are then used to solve for the unknown displacements at the nodes.

The space of admissible variations is defined as
$$
\mathcal{V}^{h}=\left\{\boldsymbol{v}^{h}(\boldsymbol{x}) | v_{i}^{h}(\boldsymbol{x})=\sum_{A \in \eta-\eta_{u}} N_{A}(\boldsymbol{x}) v_{i A}^{h}\right\}
$$
where $\eta$ is the set of all node numbers, $\eta_{u}$ is the set of node numbers where the displacements are prescribed, and $v_{i A}^{h}$ refers to the vector $v_{i}^{h}$ at node $A .$ 

$$
\left(\begin{array}{l}
v_{1}(\boldsymbol{x}) \\
v_{2}(\boldsymbol{x})
\end{array}\right)=\sum_{A \in \eta-\eta_{u}} N_{A}(\boldsymbol{x}) v_{i A}=\left[\begin{array}{ccccccc}
N_{1} & 0 & N_{2} & 0 & \cdots & N_{n} & 0 \\
0 & N_{1} & 0 & N_{2} & \cdots & 0 & N_{n}
\end{array}\right]\left(\begin{array}{c}
v_{11} \\
v_{21} \\
v_{12} \\
v_{22} \\
\vdots \\
v_{1 n} \\
v_{2 n}
\end{array}\right)=N \boldsymbol{v}
$$
where $n$ is the number of nodes in $\eta-\eta_{u} .$ The $N$ matrix is of great utility when working with the FEM equations; as given above, it is strictly for $2-\mathrm{D}$ problems but is trivially extended
to 3 -D. This type of notation also extends to the symmetric gradients of functions; viz. in $2-\mathrm{D}$ we have that
$$
\left(\begin{array}{c}
v_{1,1} \\
v_{2,2} \\
v_{1,2}+v_{2,1}
\end{array}\right)=\left[\begin{array}{ccccccc}
N_{1,1} & 0 & N_{2,1} & 0 & \ldots & N_{n, 1} & 0 \\
0 & N_{1,2} & 0 & N_{2,2} & \ldots & 0 & N_{n, 2} \\
N_{1,2} & N_{1,1} & N_{2,2} & N_{2,1} & \ldots & N_{n, 2} & N_{n, 1}
\end{array}\right]\left(\begin{array}{c}
v_{11} \\
v_{21} \\
v_{12} \\
v_{22} \\
\vdots \\
v_{1 n} \\
v_{2 n}
\end{array}\right)=B v
$$

## Fundamentals

$$k_{a b}^{e}=\int_{\Omega_{e}} \nabla N_{a}^{e} \cdot \kappa \nabla N_{b}^{e}=\int_{\Omega_{e}} N_{a, i}^{e} \kappa_{i j} N_{b, j}^{e} =\int_{\Omega_{e}} \boldsymbol{B}_{a}^{T} \boldsymbol{\kappa} \boldsymbol{B}_{b} \quad a, b \in\left\{1,2, \cdots, N_{e n}\right\}$$

$N_{en}$ is the number of element nodes.

The vector $B_{a}$ has dimensions of $N_{s d}$ by $1,$ where $N_{s d}$ stands for number of spatial dimensions.

## Isoparametric Formulations

$$\int_{\Omega_{e}} f(\boldsymbol{x}) d \boldsymbol{x} \longmapsto \int f(\boldsymbol{x}(\boldsymbol{\xi})) \operatorname{det}\left[\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{\xi}}\right] d \boldsymbol{\xi} \approx \sum_{\ell=1}^{N_{i n t}} W_{\ell} f\left(\boldsymbol{x}\left(\overline{\boldsymbol{\xi}}_{\ell}\right)\right) \operatorname{det}\left[\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{\xi}}\left(\overline{\boldsymbol{\xi}}_{\ell}\right)\right]$$
where $\bar{\xi}_{\ell}$ represent integration points and $W_{\ell}$ are the weights.