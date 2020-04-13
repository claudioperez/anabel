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

$$\int_{\Omega}(\boldsymbol{B} \boldsymbol{v})^{T} \boldsymbol{D} \boldsymbol{B} \boldsymbol{u}=\int_{\Omega}(\boldsymbol{N} \boldsymbol{v})^{T} \boldsymbol{b}+\int_{\Gamma_{t}}(\boldsymbol{N} \boldsymbol{v})^{T} \bar{t}$$

$$k_{a b}^{e}=\int_{\Omega_{e}} \nabla N_{a}^{e} \cdot \kappa \nabla N_{b}^{e}=\int_{\Omega_{e}} N_{a, i}^{e} \kappa_{i j} N_{b, j}^{e} =\int_{\Omega_{e}} \boldsymbol{B}_{a}^{T} \boldsymbol{\kappa} \boldsymbol{B}_{b} \quad a, b \in\left\{1,2, \cdots, N_{e n}\right\}$$

$N_{en}$ is the number of element nodes.

The vector $B_{a}$ has dimensions of $N_{s d}$ by $1,$ where $N_{s d}$ stands for number of spatial dimensions.


Integrals are usually restricted to individual elements and then the contributions are added together through an assembly operation like that of direct stiffness. In this case, the element stiffness matrix is given by
$$
k^{e}=\int_{\Omega_{e}} B^{T} D B
$$
where the dimension of $B$ is now $3 \times 2 N_{e n}$ for 2 -D elasticity problems - $N_{e n}$ being the number of element nodes. In 3 -D the dimension of $B$ becomes $6 \times 3 N_{e n}$. The dimension of $k^{e}$ itself
is either $2 N_{e n} \times 2 N_{e n}$ or $3 N_{e n} \times 3 N_{e n} .$ 

### Nodal Forces

The element force vector is given by the following expression:
$$
f^{e}=\int_{\Omega_{e}} N^{T} b+\int_{\Gamma_{i}^{e}} N^{T} \bar{t}
$$
$$\boldsymbol{F}=\int_{\Omega} \boldsymbol{N}^{T} \boldsymbol{b} d \Omega+\int_{\Gamma_{t}} \boldsymbol{N}^{T} \overline{\boldsymbol{t}} d \Gamma_{t}$$
where $N$ will have dimensions $2 \times 2 N_{e n}$ and $3 \times 3 N_{e n}$ for $2-\mathrm{D}$ and $3-\mathrm{D}$ problems, respectively. [Note that the last integral in ( 18.9 ) is a surface integral.]

For a 2D triangle:
$$\boldsymbol{f}^{e}=\left(\begin{array}{c}
f_{11} \\
f_{21} \\
f_{12} \\
f_{22} \\
f_{13} \\
f_{23}
\end{array}\right)=\int_{\Gamma_{t}^{e}}\left[\begin{array}{cc}
N_{1}^{e} & 0 \\
0 & N_{1}^{e} \\
N_{2}^{e} & 0 \\
0 & N_{2}^{e} \\
N_{3}^{e} & 0 \\
0 & N_{3}^{e}
\end{array}\right]\left(\begin{array}{l}
0 \\
q
\end{array}\right) d \Gamma_{t}^{e}=\int_{\Gamma_{t}^{e}}\left(\begin{array}{c}
0 \\
0 \\
0 \\
N_{2}^{e} q \\
0 \\
N_{3}^{e} q
\end{array}\right) d \Gamma_{t}^{e}$$

An alternative and equivalent formula for computing such integrals that is often used in continuum mechanics is
$$
\int_{\Gamma} f(\boldsymbol{x}) \boldsymbol{n} d \Gamma=\int_{\gamma} f(\boldsymbol{x}(\boldsymbol{X})) \operatorname{det}\left[\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{X}}\right]\left[\frac{\partial \boldsymbol{X}}{\partial \boldsymbol{x}}\right]^{T} \boldsymbol{n}_{X} d \gamma
$$
where $\boldsymbol{x}$ and $\boldsymbol{X}$ are different parameterizations of the same edge. $\boldsymbol{x}$ are the physical coordinates along the edge $\Gamma$ and $X$ are an alternative set of coordinates along the edge $\gamma=\boldsymbol{X}(\Gamma) ;$ in continuum mechanics the coordinates $\boldsymbol{X}$ are usually the Lagrangian coordinates of the "material points"; in our case they will be the isoparametric coordinates. $n$ is the normal to the edge in the physical coordinates and $n_{X}$ is the normal
to the edge in the $X$ coordinates. For the case explored above this relation reduces to
$$
\int_{\Gamma} f(\boldsymbol{x}) \boldsymbol{n} d \Gamma=\int_{0}^{1} f\left(\boldsymbol{x}\left(t_{1}, 0,1-t_{1}\right)\right) \operatorname{det}\left[\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{\xi}}\left(t_{1}, 0,1-t_{1}\right)\right]\left[\frac{\partial \boldsymbol{\xi}}{\partial \boldsymbol{x}}\left(t_{1}, 0,1-t_{1}\right)\right]^{T} \boldsymbol{n}_{X} d t_{1}
$$
where $\boldsymbol{n}_{X}$ is the normal to the edge in the parent domain; in this case $\boldsymbol{n}_{X}=(0,-1)^{T}$

## Isoparametric Formulations

$$\int_{\Omega_{e}} f(\boldsymbol{x}) d \boldsymbol{x} \longmapsto \int f(\boldsymbol{x}(\boldsymbol{\xi})) \operatorname{det}\left[\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{\xi}}\right] d \boldsymbol{\xi} \approx \sum_{\ell=1}^{N_{i n t}} W_{\ell} f\left(\boldsymbol{x}\left(\overline{\boldsymbol{\xi}}_{\ell}\right)\right) \operatorname{det}\left[\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{\xi}}\left(\overline{\boldsymbol{\xi}}_{\ell}\right)\right]$$
where $\bar{\xi}_{\ell}$ represent integration points and $W_{\ell}$ are the weights.