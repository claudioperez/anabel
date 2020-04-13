## Total Lagrange

### Kirchhoff

$$\begin{array}{l}
\frac{\mathrm{d} u_{\ell}}{\mathrm{d} x}=\mathbf{b}_{u}^{\mathrm{T}} \mathbf{a}\\
\frac{\mathrm{d} w}{\mathrm{d} x}=\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}\\
\chi=-\frac{\mathrm{d}^{2} w}{\mathrm{d} x^{2}}=\mathbf{c}^{\mathrm{T}} \mathbf{w}
\end{array}$$

$$\begin{array}{l}
\mathbf{b}_{u}=\frac{1}{\ell_{0}}(-1,+1,-4 \xi) \\
\mathbf{b}_{w}=\frac{1}{4 \ell_{0}}\left(6\left(\xi^{2}-1\right), \ell_{0}\left(3 \xi^{2}-2 \xi-1\right),-6\left(\xi^{2}-1\right), \ell_{0}\left(3 \xi^{2}+2 \xi-1\right)\right) \\
\mathbf{c}=-\frac{1}{\ell_{0}^{2}}\left(6 \xi, \ell_{0}(3 \xi-1),-6 \xi, \ell_{0}(3 \xi+1)\right)
\end{array}$$

$$\begin{array}{l}
\dot{N}=\underbrace{\left(\int_{-h / 2}^{+h / 2} b\left(z_{\ell}\right) E_{\tan \left(z_{\ell}\right) \mathrm{d} z_{\ell}}\right)}_{\overline{E A}} \dot{\epsilon}_{\ell}+\underbrace{\left(\int_{-h / 2}^{+h / 2} b\left(z_{\ell}\right) E_{\tan \left(z_{\ell}\right) z_{\ell} \mathrm{d} z_{\ell}}\right)}_{E X} \dot{x} \\
\dot{M}=\underbrace{\left(\int_{-h / 2}^{+h / 2} b\left(z_{\ell}\right) E_{\tan }\left(z_{\ell}\right) z_{\ell} \mathrm{d} z_{\ell}\right)}_{E X} \dot{\epsilon}_{\ell}+\underbrace{\left(\int_{-h / 2}^{+h / 2} b\left(z_{\ell}\right) E_{\tan \left(z_{\ell}\right) z_{\ell}^{2} \mathrm{d} z_{\ell}}\right)}_{E I} \dot{x}
\end{array}$$

$$\begin{aligned}
\mathbf{K}_{\mathrm{aa}} &=\frac{\partial \mathbf{f}_{\mathrm{int}}^{\mathrm{a}}}{\partial \mathbf{a}}=\int_{\ell_{0}} E A \mathbf{b}_{u} \mathbf{b}_{u}^{\mathrm{T}} \mathrm{d} x \\
\mathbf{K}_{\mathrm{aw}} &=\frac{\partial \mathbf{f}_{\mathrm{int}}^{\mathrm{a}}}{\partial \mathbf{w}}=\int_{\ell_{0}} E A\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right) \mathbf{b}_{u} \mathbf{b}_{w}^{\mathrm{T}} \mathrm{d} x \\
\mathbf{K}_{\mathrm{ww}} &=\frac{\partial \mathbf{f}_{\mathrm{int}}^{\mathrm{w}}}{\partial \mathbf{w}}=\int_{\ell_{0}}\left(E I \mathbf{c} \mathbf{c}^{\mathrm{T}}+E A\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right)^{2} \mathbf{b}_{w} \mathbf{b}_{w}^{\mathrm{T}}+N \mathbf{b}_{w} \mathbf{b}_{w}^{\mathrm{T}}\right) \mathrm{d} x
\end{aligned}$$

$$\begin{array}{l}
\mathbf{K}_{\mathrm{aa}}=\int_{\ell_{0}} \overline{E A} \mathbf{b}_{u} \mathbf{b}_{u}^{\mathrm{T}} \mathrm{d} x \\
\mathbf{K}_{\mathrm{aw}}=\int_{\ell_{0}}\left(\overline{E A}\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right) \mathbf{b}_{u} \mathbf{b}_{w}^{\mathrm{T}}+\overline{E X} \mathbf{b}_{u} \mathbf{c}^{\mathrm{T}}\right) \mathrm{d} x \\
\mathbf{K}_{\mathrm{ww}}=\int_{\ell_{0}}\left(\overline{E I} \mathbf{c} \mathbf{c}^{\mathrm{T}}+\overline{E A}\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right)^{2} \mathbf{b}_{w} \mathbf{b}_{w}^{\mathrm{T}}+N \mathbf{b}_{w} \mathbf{b}_{w}^{\mathrm{T}}\right. \\
\quad+\overline{E X}\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right)\left(\mathbf{b}_{w} \mathbf{c}^{\mathrm{T}}+\mathbf{c} \mathbf{b}_{w}^{\mathrm{T}}\right) \mathrm{d} x
\end{array}$$


### Timoshenko

$$\begin{aligned}
&\mathbf{f}_{\mathrm{int}}^{\mathrm{a}}=\int_{\ell_{0}} N \mathbf{b}_{u} \mathrm{d} x\\
&\mathbf{f}_{\mathrm{int}}^{\mathrm{w}}=\int_{\ell_{0}}\left(N\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right) \mathbf{b}_{w}+Q \mathbf{b}_{w}\right) \mathrm{d} x\\
&\mathbf{f}_{\mathrm{int}}^{\theta}=\int_{\ell_{0}}\left(M \mathbf{b}_{\theta}+Q \mathbf{h}_{\theta}\right) \mathrm{d} x
\end{aligned}$$

$$\left[\begin{array}{ccc}
\mathbf{K}_{\text {aa }} & \mathbf{K}_{\text {aw }} & \mathbf{K}_{\mathbf{a} \theta} \\
\mathbf{K}_{\text {aw }}^{\mathrm{T}} & \mathbf{K}_{\mathbf{W W}} & \mathbf{K}_{\mathbf{W} \theta} \\
\mathbf{K}_{\mathbf{a} \theta}^{\mathrm{T}} & \mathbf{K}_{\mathbf{w} \theta}^{\mathrm{T}} & \mathbf{K}_{\theta \theta}
\end{array}\right]\left(\begin{array}{c}
\mathrm{d} \mathbf{a} \\
\mathrm{d} \mathbf{w} \\
\mathrm{d} \boldsymbol{\theta}
\end{array}\right)=\left(\begin{array}{c}
\mathbf{f}_{\mathrm{ext}}^{\mathrm{a}}-\mathbf{f}_{\mathrm{int}}^{\mathrm{a}} \\
\mathbf{f}_{\mathrm{ext}}^{\mathrm{w}}-\mathbf{f}_{\mathrm{int}}^{\mathrm{w}} \\
\mathbf{f}_{\mathrm{ext}}^{\theta}-\mathbf{f}_{\mathrm{int}}^{\theta}
\end{array}\right)$$

$$\begin{aligned}
&\mathbf{K}_{\mathrm{aa}}=\int_{\ell_{0}} \overline{E A} \mathbf{b}_{u} \mathbf{b}_{u}^{\mathrm{T}} \mathrm{d} x\\
&\mathbf{K}_{\mathrm{aw}}=\int_{\ell_{0}} \overline{E A}\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right) \mathbf{b}_{u} \mathbf{b}_{w}^{\mathrm{T}} \mathrm{d} x\\
&\mathbf{K}_{\mathrm{a} \theta}=\int_{\ell_{0}} \overline{E X} \mathbf{b}_{u} \mathbf{b}_{\theta}^{\mathrm{T}} \mathrm{d} x\\
&\mathbf{K}_{\mathrm{ww}}=\int_{\ell_{0}}\left(\overline{E A}\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right)^{2}+\overline{G A}+N\right) \mathbf{b}_{w} \mathbf{b}_{w}^{\mathrm{T}} \mathrm{d} x\\
&\mathbf{K}_{\mathrm{w} \theta}=\int_{\ell_{0}}\left(\overline{E X}\left(\mathbf{b}_{w}^{\mathrm{T}} \mathbf{w}^{\prime}\right) \mathbf{b}_{w} \mathbf{b}_{\theta}^{\mathrm{T}}+\overline{G A} \mathbf{b}_{w} \mathbf{h}_{\theta}^{\mathrm{T}}\right) \mathrm{d} x\\
&\mathbf{K}_{\theta \theta}=\int_{\ell_{0}}\left(\overline{E I} \mathbf{b}_{\theta} \mathbf{b}_{\theta}^{\mathrm{T}}+\overline{G A} \mathbf{h}_{\theta} \mathbf{h}_{\theta}^{\mathrm{T}}\right) \mathrm{d} x
\end{aligned}$$