# Strain

## 1D Strain

$$\begin{aligned}
&\epsilon_E \stackrel{\text { def }}{=} \frac{l-l_{0}}{l_{0}} \text { Engineering Strain }\\
&\epsilon \stackrel{\text { def }}{=} \frac{1}{2} \frac{l^{2}-l_{0}^{2}}{l^{2}} \text { Cauchy Strain }\\
&\epsilon_L \stackrel{\text { def }}{=} \int_{l_0}^{l} \frac{d l}{l}=\ln \frac{l}{l_{0}} \text { Logarithmic Strain }
\end{aligned}$$

$$\epsilon_{L}=\ln \left(\frac{L+\Delta L}{L}\right)=\ln \left(1+\frac{\Delta L}{L}\right)=\ln \left(1+\epsilon_{E}\right)$$
Green-Lagrange
$$\varepsilon_{G}:=\frac{1}{2}\left(\frac{l^{2}-L^{2}}{L^{2}}\right)=\frac{1}{2}\left(\frac{l^{2}}{L^{2}}-1\right)$$

Almansi-Hamel (Eulerian) strain:
$$\varepsilon_{A}:=\frac{1}{2}\left(\frac{l^{2}-L^{2}}{l^{2}}\right)=\frac{1}{2}\left(1-\frac{L^{2}}{l^{2}}\right)$$

## 1-D Stress

Engineering/Nominal stress:
$$P=\sigma_{E}:=\frac{T}{A}$$
Cauchy/True stress:
$$\sigma=\sigma_{T}:=\frac{T}{a}$$
Relation between engineering and true stress (no volume change):
$$\sigma_{T}=\frac{T}{a}=\frac{T l}{A L}=\sigma_{E}\left(\frac{L+\Delta L}{L}\right)=\sigma_{E}\left(1+\varepsilon_{E}\right)$$

1-D stress-strain relations

- True stress - Green strain:
    $$\sigma_{T}=E_{T G}\left(\frac{l^{2}-L^{2}}{2 L^{2}}\right)$$
- True stress - True strain:
    $$\sigma_{T}=E_{T T} \ln \left(\frac{l}{L}\right)$$