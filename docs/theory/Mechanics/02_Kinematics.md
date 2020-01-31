# II Kinematics

Displacement:

$\mathbf{u}(\mathrm{X}, t)=\chi(\mathrm{X}, t)-\mathrm{X}, \quad u_{i}\left(X_{1}, X_{2}, X_{3}\right)=\chi_{i}\left(X_{1}, X_{2}, X_{3}\right)-X_{i}$

Velocity/Acceleration

$\begin{aligned} \dot{\mathbf{u}}(\mathbf{X}, t) &=\frac{\partial \chi(\mathbf{X}, t)}{\partial t}(3.8.3)\\ \ddot{\mathbf{u}}(\mathbf{X}, t) &=\frac{\partial^{2} \chi(\mathbf{X}, t)}{\partial t^{2}} (3.8.4)\end{aligned}$

### 3.2 Deformation/Displacement Gradient

$\begin{aligned} \mathbf{F}(\mathbf{X}, t)=\frac{\partial}{\partial \mathbf{X}} \chi(\mathbf{X}, t),\quad & F_{i j}=\frac{\partial}{\partial X_{j}} \chi_{i}\left(X_{1}, X_{2}, X_{3}, t\right), \quad \operatorname{det} \mathbf{F}(\mathbf{X}, t)>0 \\ \mathbf{H}(\mathbf{X}, t)=\frac{\partial}{\partial \mathbf{X}} \mathbf{u}(\mathbf{X}, t), \quad & H_{i j} =\frac{\partial}{\partial X_{j}} u_{i}\left(X_{1}, X_{2}, X_{3}, t\right) \\ \mathbf{H}(\mathbf{X}, t)=\mathbf{F}(\mathbf{X}, t)-1,\quad & H_{i j}=F_{i j}-\delta_{i j} \end{aligned}$

$$J \equiv \operatorname{det}\left(\frac{\partial \chi}{\partial \mathbf{X}}\right)=\operatorname{det} \mathbf{F} =\frac{d v}{d v_{\mathrm{R}}}\neq 0$$

### 3.3 Stretch & Rotation

**Polar Decomposition:** $\mathbf{F}=\mathbf{R} \mathbf{U}=\mathbf{V} \mathbf{R}$
$$
\begin{array}{ll}{\mathbf{C}=\mathbf{U}^{2}=\mathbf{F}^{\mathrm{T}} \mathbf{F},} & {C_{i j}=F_{k i} F_{k j}=\frac{\partial \chi_{k}}{\partial X_{i}} \frac{\partial \chi_{k}}{\partial X_{j}}} \\ {\mathbf{B}=\mathbf{V}^{2}=\mathbf{F F}^{\top},} & {B_{i j}=F_{i k} F_{j k}=\frac{\partial \chi_{i}}{\partial X_{k}} \frac{\partial \chi_{j}}{\partial X_{k}}}\end{array}
$$

$\lambda \stackrel{\text {def}}{=} \frac{d s}{d S}=|\mathbf{U} \mathbf{e}|= \sqrt{\mathbf{e} \cdot \mathbf{C}(\mathbf{X}) \mathbf{e}}$

where $d S=|\mathrm{dX}|, d s=|\mathrm{dx}|, \mathrm{e}=\frac{d \mathrm{X}}{|d \mathrm{X}|}$

**Engineering shear:** $\gamma =\sin ^{-1}\left[\frac{\mathbf{e}^{(1)} \cdot \mathbf{C} \mathbf{e}^{(2)}}{\lambda\left(\mathbf{e}^{(1)}\right) \lambda\left(\mathbf{e}^{(2)}\right)}\right]$

### 3.4 Strain

**Green strain:**
$\mathrm{E} \stackrel{\text { def }}{=} \frac{1}{2}\left(\mathrm{F}^{\top} \mathrm{F}-1\right)=\frac{1}{2}\left(\mathbf{H}+\mathbf{H}^{\top}+\mathbf{H}^{\top} \mathbf{H} .\right)$

**Hencky's Log strain:**
$\ln \mathbf{U} \stackrel{\text { def }}{=} \sum_{i=1}^{3}\left(\ln \lambda_{i}\right) \mathbf{r}_{i} \otimes \mathbf{r}_{i}$

3.5.2 Infinitesimal Strain

$\epsilon'$: distortion
$\epsilon_M\delta_{ij}$: dilation
$$
\begin{aligned} \epsilon &=\frac{1}{2}\left[\mathbf{H}+\mathbf{H}^{\top}\right], & \epsilon &=\epsilon^{\top}, \quad|\mathbf{H}| \ll 1 \\ \epsilon_{i j}=& \frac{1}{2}\left[\frac{\partial u_{i}}{\partial X_{j}}+\frac{\partial u_{j}}{\partial X_{i}}\right], & \epsilon_{j i}=\epsilon_{i j}, &\left|\frac{\partial u_{i}}{\partial X_{j}}\right| \ll 1 \end{aligned}
$$

### 3.A Linearization

$\operatorname{lin} \mathrm{Y}_{o} f(\mathbf{Y})=f\left(\mathbf{Y}_{o}\right)+\left.\frac{d}{d \alpha} f\left(\mathbf{Y}_{o}+\alpha\left(\mathbf{Y}-\mathbf{Y}_{o}\right)\right)\right|_{\alpha=0}$
$\operatorname{lin}_{0} f(\mathbf{H})=f(0)+\left.\frac{d}{d \alpha} f(\alpha \mathbf{H})\right|_{\alpha=0}$

### 3.B Compatibility