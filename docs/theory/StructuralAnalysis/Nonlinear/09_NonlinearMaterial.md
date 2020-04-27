# Path-Independent Material

$$E_{t}=\frac{d \sigma}{d \varepsilon}$$

$$s_r(x)=\int_{A} \mathbf{a}_{s}(y, z)^{T} \sigma_{m} d A=\left(\begin{array}{l}
N(x) \\
M_{z}(x) \\
M_{y}(x)
\end{array}\right)$$

$$\mathbf{k}_{s}=\frac{\partial s_r}{\partial e}=\int_{A} \mathbf{a}_{s}(y, z)^{T} \frac{d \sigma_{m}}{d \varepsilon_{m}} \frac{\partial \varepsilon_{m}}{\partial e} d A=\int_{A} \mathbf{a}_{s}(y, z)^{T} \frac{d \sigma_{m}}{d \varepsilon_{m}} \mathbf{a}_{s}(y, z) d A$$

$$s=\left(\begin{array}{l}
N \\
M
\end{array}\right)=\sum_{i=1}^{n I P}\left(\begin{array}{r}
1 \\
-y_{i}
\end{array}\right) \sigma_{i} b_{i} w_{i}$$

$$\mathbf{k}_{s}=\sum_{i=1}^{n I P}\left[\begin{array}{rr}
1 & -y_{i} \\
-y_{i} & y_{i}^{2}
\end{array}\right] E_{i} b_{i} w_{i}$$

## 1) Given $e(x)$ determine $s(x)$

1) The determination of the failure surface for the interaction of section force resultants under assumed section deformations at failure.
2) The determination of the section forces $s(x)$ for section deformations $e(x)$ resulting from element deformations $\boldsymbol{v}$ through displacement interpolation functions.

> .
> 1. For $i=1 \ldots n I P$ determine the strain at integration point i
>    $$
>    \varepsilon_{i}=\left[\begin{array}{ll}
>    1 & -y_{i}
>    \end{array}\right]\left(\begin{array}{l}
>    \varepsilon_{a} \\
>    \kappa
>    \end{array}\right)
>    $$
> 2. Use the material model to determine the stress $\sigma_{i}=\sigma\left(\varepsilon_{i}\right)$
> 3. Determine the section forces $s$ from
>    $$
>    s=s_{r}(e)=\left(\begin{array}{c}
>    N_{r} \\
>    M_{r}
>    \end{array}\right)=\sum_{i=1}^{n I P}\left(\begin{array}{r}
>    1 \\
>    -y_{i}
>    \end{array}\right) \sigma_{i} b_{i} w_{i}
>    $$
>    where $b_{i}=b\left(y_{i}\right)$ and $w_{i}$ is the integration weight at $i$
>    If required, determine the section stiffness $\mathbf{k}_{s}$ from
>    $$
>    \mathbf{k}_{s}=\sum_{i=1}^{n I P}\left[\begin{array}{rr}
>    1 & -y_{i} \\
>    -y_{i} & y_{i}^{2}
>    \end{array}\right] E_{i} b_{i} w_{i}
>    $$
> .

## 2) Given $s(x)$ determine $e(x)$

1) The determination of the deformation state of statically determinate structures under given nodal forces (e.g. RC cantilever column under eccentric axial force).
2) The determination of the failure surface for the interaction of section force resultants under incremental analysis to failure.
3) The determination of the section deformations $e(x)$ for given section forces $s(x)$ resulting from the element basic forces $q$ through force interpolation functions.

> 1. Given the nonlinear equation $s_{u}(e)=0$ and a guess for the solution $e_{0}$
> 2. For $i=0 \ldots n$ determine the function value $s_{u}\left(e_{i}\right)$ and the derivative $\mathbf{k}_{s}\left(e_{i}\right)$
> 3. Determine the correction to the previous solution estimate $\Delta e_{i}=\mathbf{k}_{s}\left(e_{i}\right)\left\langle s_{u}\left(e_{i}\right)\right.$
> 4. Update the solution estimate $e_{i+1}=e_{i}+\Delta e_{i}$ Return to step 2 until the error norm is smaller than a specified tolerance.
> 5. On convergence determine the section forces for the final deformations.

## 3) Given $e_i$, $s_j$ find $e_j$, $s_i$.

## - Given eccentricity, find $s$.