# 5 - Nonlinear SDOF

## 5.1 Nonlinear RSA (ASCE 7)

## 5.2 Equivalent Linearization

Iterative process where $\Delta_{max}$ is used to find $[K,\zeta]_{eff}$.

Used for base-isolated structures.

1. Apply $P$, obtain $V_{base}$ (Pushover analysis)
2. Use $T_0$ and $\zeta$  find $\Delta_i = Sa(T_0,\zeta)T_0$

## 5.3 Capacity Spectrum

1. Guess $\zeta$, find 

## 5.4 Coefficient Method (ASCE 41)

$\delta_{t}=C_{0} C_{1} C_{2} S_{a} \frac{T_{e}^{2}}{4 \pi^{2}} g$\

- $C_{o}(\Phi_n\Gamma)$: converts SDOF spectral displacement to MDOF roof displacement
- $C_{1}(~\gamma)$: amplification for bilinear response
- $C_{2}$: amplification for pinched hysteresis, stiffness degradation, and strength deterioration
- $T_{\text {e}}=T_{\text {initial }} \sqrt{\mathrm{k}_{\mathrm{i}} / \mathrm{ke}}$

$\mu_{\text {strength}}=\frac{S_{a}^{e}}{V_{y} / w} C_{m}$\
$R_{\max }=\mu_{\max }=\frac{\Delta_{d}}{\Delta_{y}}+\frac{\left|\alpha_{e}\right|^{-h}}{4}$\
$h =1+0.15 \ln T_{e}$\
$\alpha_{e} =\alpha_{P-\Delta}+\lambda\left(\alpha_{2}-\alpha_{P-\Delta}\right)$\
$\lambda = \begin{aligned}
&0.8 \text { if } S_{X 1} \geq 0.6\\
&0.2 \text { if } S_{X 1} \leq 0.6
\end{aligned}$
