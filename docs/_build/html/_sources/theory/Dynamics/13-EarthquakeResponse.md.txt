<!-- ---
title: MDF Earthquake Response
template: default
permalink: /Dynamics/13-EarthquakeResponse
--- -->

# Earthquake Response

## Part A: Response History

### 13.1 Modal Analysis

$$\begin{array}{c}{\mathrm{mi}+\mathrm{c} \dot{\mathrm{u}}+\mathrm{k} \mathrm{u}=\mathrm{p}_{\mathrm{eff}}(t)} \\ {\mathrm{p}_{\mathrm{eff}}(t)=-\mathrm{m} \iota \ddot{u}_{g}(t)}\end{array}$$

#### 13.1.2 Modal Expansion of Displacements and Forces

$$\mathbf{m} \iota=\sum_{n=1}^{N} \mathbf{s}_{n}=\sum_{n=1}^{N} \Gamma_{n} \mathbf{m} \phi_{n}\\$$

$$\Gamma_{n}=\frac{L_{n}}{M_{n}} \quad L_{n}=\phi_{n}^{T} \mathbf{m} \iota \quad M_{n}=\phi_{n}^{T} \mathbf{m} \phi_{n}$$

- $\iota \quad$: *Influence vector*.
  - Represents the displacements of the masses resulting from static application of a unit ground displacement.

- $\Gamma_{n}$: *Modal participation factor*.

#### 13.1.3 Modal equations

$$\ddot{q}_{n}+2 \zeta_{n} \omega_{n} \dot{q}_{n}+\omega_{n}^{2} q_{n}=-\Gamma_{n} \ddot{u}_{g}(t)$$

$$\ddot{D}_{n}+2 \zeta_{n} \omega_{n} \dot{D}_{n}+\omega_{n}^{2} D_{n}=-\ddot{u}_{g}(t)$$

#### 13.3.4 Modal Response

##### Pseudo-acceleration response of the nth-mode SDF system to $\ddot{u}_{g}(t)$:

$A_{n}(t)=\omega_{n}^{2} D_{n}(t)$

##### Equivalent static force:

$\mathbf{f}_{n}(t)=\mathbf{s}_{n} A_{n}(t)$

##### Modal Displacements

$\mathbf{u}_{n}(t)=\phi_{n} q_{n}(t)=\Gamma_{n} \phi_{n} D_{n}(t)$

$$\mathbf{u}_{n}(t)=\underbrace{\dfrac{\Gamma_{n}}{\omega_{n}^{2}} \phi_{n}}_{\mathbf{u^{st}_n}} A_{n}(t)$$

##### General response quantities

$$r(t)=\sum_{n=1}^{N} r_{n}(t)=\sum_{n=1}^{N} r_{n}^{\mathrm{st}} A_{n}(t)$$

### 3.2 Symmetric plan, multi-story buildings

1. Define the ground acceleration ¨ ug(t) numerically at every time step t.

2. Define the structural properties.

    1. Determine the mass matrix m and lateral stiffness matrix k (Section 9.4).
    2. Estimate the modal damping ratios ζn (Chapter 11).

3. Determine the natural frequencies ωn (natural periods Tn = 2π/ωn) and natural modes φn of vibration (Chapter 10).

4. Determine the modal components sn [Eq. (13.2.4)] of the effective earthquake force distribution.

5. Compute the response contribution of the nth mode by the following steps, which are repeated for all modes, n = 1, 2, . . . , N:
    1. Perform static analysis of the building subjected to lateral forces sn to determine r st n , the modal static response for each desired response quantity r (Table 13.2.1).
    2. Determine the pseudo-acceleration response An(t) of the nth-mode SDF system to ¨ ug(t), using numerical time-stepping methods (Chapter 5).
    3. Determine rn(t) from Eq. (13.2.8).
6. Combine the modal contributions rn(t) to determine the total response using
Eq. (13.2.10).