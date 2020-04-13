# 6 - MDOF Analysis

## 1. Linear Static - ELF (ASCE 7)

- minimum $C_s$ value is important for long-period structures near faults.
- $k$: Approximates higher-mode effects.
- $R$: determined from past EQs via FEMA P695
  - **hybrid systems** (2 systems in 1 dir): $R = min(R_1,R_2)$

## 2. Linear Dynamic - Modal RSA

1. Reduce spectrum by $\frac{I_e}{R}$
2. Conduct modal analysis subject to $V_b\geq V_{ELF}$

## 3. Nonlinear Static

1. Nonlinear RSA
2. Equivalent Linearization
3. Capacity Spectrum
4. Coefficient Method (ASCE 41)

### MDOF -> SDOF

1. Bilinearize
2. plot $V/M^*_n$ vs $D_1=\dfrac{u_{roof}}{\Gamma \Phi_{roof, n}}$
3. $\omega = \sqrt{\dfrac{V_{yn}}{M^*_{n}D_{yn}}}$

## 