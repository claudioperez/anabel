# 7 Solution Strategies

1. **Load Incrementation**: $P_{f}^{(k)}=P_{f}^{(k-1)}+\Delta \lambda P_{\text {ref }}$
2. Use the solution at $k-1$ as initial guess $U_{0}^{(k)}=U^{(k-1)}$
3. Structure State Determination: $P_{r}^{(k)}=P_{r}\left(U_{0}^{(k)}\right)$ and $K_{t}^{(k)}=K_{t}\left(U_{0}^{(k)}\right)$
4. Determine $P_{u}^{(k)}=P_{f}^{(k)}-P_{r}^{(k)}$
5. Determine $\Delta U_{0}^{(k)}=K_{t}^{(k)} \backslash P_{u}^{(k)}$
6. Update the solution $U_{1}^{(k)}=U_{0}^{(k)}+\Delta U_{0}^{(k)}$
7. **Equilibrium Iterations**

   For $i=1 \ldots n$ and constant $k$ without superscript ( $k$ ) except for $P_{f}$.
   - State Determination: $P_{r}\left(U_{i}\right)$ and $K_{t}\left(U_{i}\right)$
   - Nodal force unbalance $P_{u}\left(U_{i}\right)=P_{f}^{(k)}-P_{r}\left(U_{i}\right)$
   - Solution correction $\Delta U_{i}=K_{t}\left(U_{i}\right) \backslash P_{u}\left(U_{i}\right)$
   - Update estimate $U_{i+1}=U_{i}+\Delta U_{i}$

    Repeat steps (a)-(d) until the error norm satisfies the specified tolerance.

8. On convergence of the equilibrium iterations determine the resisting forces for the final $U^{(k)}$