# FEM

$$\{\sigma\}=[D]\{[B][u\}-\{\alpha\} T\}+\left\{\sigma_{0}\right\}$$

$$\{p\}=[k]\{u\}+\left\{p_{0}\right\}$$

$$[k]=\int_{\Omega_{e}}[B]^{T}[D][B] d \Omega_{e}$$

$$\left\{p_{0}\right\}=\underbrace{-\int_{\Omega_{e}}[N]^{T}\{b\} d \Omega_{e}}_{\text {body force }}
\underbrace{-\int_{\Gamma_{t}}[N]^{T}\{t\} d \Gamma_{e}}_{\text {boundary traction}}
\underbrace{-\int_{\Omega_{e}}[B]^{T}[D]\left\{\varepsilon_{0}\right\} d \Omega_{e}}_{\text {initial strain}}
\underbrace{+\int_{\Omega_{e}}[B]^{T}\left\{\sigma_{0}\right\} d \Omega_{e}}_{\text {initial stress}}$$
