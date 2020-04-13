# Energy

## PVD

A structure is in equilibrium under a system of loads and initial strains if for any
admissible virtual displacement, the internal virtual work equals the external virtual work.

$$\begin{aligned}
&W_{I_{e}}=\int_{0}^{L_{e}}(\delta \varepsilon) E A \frac{d \bar{u}}{d x} d x=\left\{\delta u_{e}\right\}^{T} \int_{0}^{L_{e}}\{B\} E A\{B\}^{T} d x_{e}\left\{u_{e}\right\}=\left\{\delta u_{e}\right\}^{T}\left[k_{e}\right]\left\{u_{e}\right\}\\
&W_{E_{e}}=\int_{0}^{L_{e}}(\delta u) p(x) d x=\left\{\delta u_{e}\right\}^{T} \int_{0}^{L_{s}}\{N\} p\left(x_{e}\right) d x_{e}=-\left\{\delta u_{e}\right\}^{T}\left\{p_{0_{e}}\right\}
\end{aligned}$$

$$\int_{\Omega} \delta \varepsilon_{i j} \sigma_{j i} d \Omega=\int_{\Omega} \delta u_{i} b_{i} d \Omega+\int_{\Gamma} \delta u_{i} t_{i} d \Gamma$$

$$\sum_{e}\left(\int_{\Omega_{e}}\{\delta \varepsilon\}^{T}\{\sigma\} d \Omega_{e}-\int_{\Omega_{e}}\{\delta u\}^{T}\{b\} d \Omega_{e}-\int_{\Gamma_{t}}\{\delta u\}^{T}\{t\} d \Gamma_{e}\right)=0$$

## Minimum Potential Energy

For an elastic body, the potential energy is given by
$$
\Pi(\boldsymbol{u})=\int_{\Omega} W(\boldsymbol{\varepsilon}(u))-\int_{\Omega} \boldsymbol{b} \cdot \boldsymbol{u}-\int_{\Gamma_{t}} \bar{t} \cdot \boldsymbol{u}
$$

$$\left.\frac{d}{d \alpha}\right|_{\alpha=0} \Pi(\boldsymbol{u}+\alpha \boldsymbol{v})=\int_{\Omega} \varepsilon_{i j}(v) C_{i j k l} \varepsilon_{k l}(u)-\int_{\Omega} b_{i} v_{i}-\int_{\Gamma_{t}} \bar{t}_{i} v_{i}=0$$