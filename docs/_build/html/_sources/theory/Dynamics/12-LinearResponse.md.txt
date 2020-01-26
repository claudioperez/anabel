# Linear Response

Linear response of multiple degree-of-freedom systems to dynamic excitation


## Part B: Modal Analysis

### 12.7 - Summary

let $n$ denote mode numbers, and $i, j, k...$ denote dofs.

The dynamic response of an MDF system to external force vector $\mathbf{p}(t)$ can be computed by modal analysis, summarized next as a sequence of steps:

1. Define the structural properties.

    a. Determine the mass matrix $\mathbf{m}$ and stiffness matrix $\mathbf{k}$ (Chapter 9).

    b. Estimate the modal damping ratios $\zeta_n$ (Chapter 11).

2. Determine the natural frequencies $\omega_n$ and modes $\phi_n$ (Chapter 10).

3. Compute the response in each mode by the following steps:

    a. Set up Eq. (12.4.5) or (12.4.6) and solve for $q_n(t)$.
    $$M_{n} \ddot{q}_{n}+C_{n} \dot{q}_{n}+K_{n} q_{n}=P_{n}(t)\quad (12.4.5)$$
    $$\ddot{q}_{n}+2 \zeta_{n} \omega_{n} \dot{q}_{n}+\omega_{n}^{2} q_{n}=\frac{P_{n}(t)}{M_{n}}\quad (12.4.6)$$
    where $M_{n} = \phi_n^T\mathbf{m} \phi_n$, $K_{n} = \phi_n^T\mathbf{k} \phi_n$, and $P_{n} = \phi_n^T\mathbf{p}(t)$

    b. Compute the nodal displacements $\mathbf{u}_{n}(t)$ from Eq. (12.5.1).

    $$\mathbf{u}_{n}(t)=\phi_{n} q_{n}(t) \quad(12.5.1)$$
    c. Compute the element forces associated with the nodal displacements $\mathbf{u}_n(t)$ by implementing one of the two methods described in Section 12.6 for the desired values of $t$ and the element forces of interest.

4. Combine the contributions of all the modes to determine the total response. In particular, the nodal displacements $\mathbf{u}(t)$ are given by Eq. (12.5.2) and element forces by Eq. (12.6.1).

    a. 
    $$\mathbf{u}(t)=\sum_{n=1}^{N} \mathbf{u}_{n}(t)=\sum_{n=1}^{N} \phi_{n} q_{n}(t)\quad (12.5.2)$$

    b. Element forces.
    1. Modal summation
        $$
        r(t)=\sum_{n=1}^{N} r_{n}(t)
        $$
    2. Equivalent static forces.
        $$\mathbf{f}_{n}(t)=\omega_{n}^{2} \mathbf{m} \phi_{n} q_{n}(t)$$