# Damping in Structures

## Classical Damping

### Rayleigh Damping

$\mathbf{c}=a_{0} \mathbf{m}+a_{1} \mathbf{k}$

$$\zeta_{n}=\frac{a_{0}}{2} \frac{1}{\omega_{n}}+\frac{a_{1}}{2} \omega_{n}$$
$$\frac{1}{2}\left[\begin{array}{cc}{1 / \omega_{i}} & {\omega_{i}} \\ {1 / \omega_{j}} & {\omega_{j}}\end{array}\right]\left\{\begin{array}{l}{a_{0}} \\ {a_{1}}\end{array}\right\}=\left\{\begin{array}{l}{\zeta_{i}} \\ {\zeta_{j}}\end{array}\right\}$$

for $\zeta_i=\zeta_j$
$$a_{0}=\zeta \frac{2 \omega_{i} \omega_{j}}{\omega_{i}+\omega_{j}} \quad a_{1}=\zeta \frac{2}{\omega_{i}+\omega_{j}}$$

### Caughey Damping

$$\mathbf{c}=\mathbf{m} \sum_{l=0}^{N-1} a_{l}\left[\mathbf{m}^{-1} \mathbf{k}\right]^{l}$$