# 5 Event-to-Event

1) Set up the current stiffness of the structural model with basic force releases at all $p_{k}$ locations where plastic hinges appeared in the events through $k$
    $$
    \mathbf{K}^{(k)}=\mathbf{A}_{f}^{T} \mathbf{K}_{s}^{(k)} \mathbf{A}_{f}
    $$
2) Solve for the free dof displacements $U_{f}^{\prime}$ and the basic forces $Q^{\prime}$ under reference load
    $$
    \begin{array}{l}
    U_{f}^{(k)}=\mathbf{K}^{(k)} | P_{\text {ref}} \\
    Q^{(k)}=\mathbf{K}_{s}^{(k)}\left[\mathbf{A}_{f} U_{f}^{(k)}\right]
    \end{array}
    $$
3) Determine the DC ratio under the reference load at locations $m$ without plastic hinge
    $$
    D C_{m}^{\prime}=\frac{Q_{m}^{(k)}}{\operatorname{sgn} Q_{p, i, m}^{\operatorname{en}}-Q_{m}^{(k)}}
    $$

4) Determine the load factor increment $\Delta \lambda^{(k)}$ to next event
    $$
    \Delta \lambda^{(k)}=\frac{1}{\max \left(D C_{m}^{\prime}\right)}
    $$

5) Update the load factor, the free dof displacements and the basic forces to next event
    $$
    \begin{aligned}
    \lambda^{(k+1)} &=\lambda^{(k)}+\Delta \lambda^{(k)} \\
    U_{f}^{(k+1)} &=U_{f}^{(k)}+\Delta \lambda^{(k)} U_{f}^{(k)} \\
    Q^{(k+1)} &=Q^{(k)}+\Delta \lambda^{(k)} Q^{(k)}
    \end{aligned}
    $$

6) Determine the plastic deformations at next event
    $$
    \boldsymbol{V}_{h p}^{(k+1)}=\mathbf{A}_{f} \boldsymbol{U}_{f}^{(k+1)}-\mathbf{F}_{s}^{(0)} \boldsymbol{Q}^{(k+1)}
    $$
    where $\mathbf{F}_{a}^{(0)}$ is the collection of initial element flexibility matrices.