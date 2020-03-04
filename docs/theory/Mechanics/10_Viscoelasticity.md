# X Viscoelasticity
$\sigma(t)=\eta \dot{\epsilon}(t)$
## 29 Linear Viscoelasticity

### 29.2 Stress relaxation & Creep

##### Stress (Relaxation test), given $\epsilon(t)=\epsilon_{0} h(t), \quad \dot{\epsilon}(t)=\epsilon_{0} \delta(t)$

$\sigma(t)=\sum_{i=1}^{N} h\left(t-t_{i}\right) \Delta \sigma_{i}\\$

$\sigma(t)=\int_{0^{-}}^{t} E_{r}(t-\tau) \frac{d \epsilon(\tau)}{d \tau} d \tau =\left(E_{r} * \dot{\epsilon}\right)(t)$

##### Strain (Creep test), given $\sigma(t)=\sigma_{0} h(t)$

$\epsilon(t)=\sum_{i=1}^{N} J_{c}\left(t-t_{i}\right) \Delta \sigma_{i}\quad(p497)\\$

$\epsilon(t)=\int_{0-}^{t} J_{c}(t-\tau) \frac{d \sigma(\tau)}{d \tau} d \tau\quad(29.2.6)$

$J_{c}(t) \stackrel{\text { def }}{=} \frac{\epsilon(t)}{\sigma_{0}}$

### 29.5 Correspondence Principle (1D)

$$
\bar{\sigma}(s)=\bar{E}_{r}^{*}(s) \bar{\epsilon}(s)
$$
where

$$
\bar{E}_{r}^{*}(s)=s \bar{E}_{r}(s)
$$

$$
\bar{J}_{c}(s) \bar{E}_{r}(s)=\frac{1}{s^{2}}
$$

### 29.7 Oscillatory Response

$J^{*} E^{*}=1$

#### Applied stress, $\sigma(t)=\sigma_{0} \cos (\omega t)$

$\begin{aligned} \epsilon(t) &=\epsilon_{0} \cos (\omega t-\delta) \\ &=\epsilon_{0} \cos (\delta) \cos (\omega t)-\epsilon_{0} \sin (\delta) \sin (\omega t) \\ &=\sigma_{0}\left(J^{\prime} \cos (\omega t)+J^{\prime \prime} \sin (\omega t)\right) \end{aligned}$

Storage compliance, $J^{\prime} \stackrel{\text { def }}{=} \frac{\epsilon_{0}}{\sigma_{0}} \cos (\delta)$.

Loss compliance, $J^{\prime \prime} \stackrel{\text { def }}{=} \frac{\epsilon_{0}}{\sigma_{0}} \sin (\delta)$.

#### Applied strain, $\epsilon(t)=\epsilon_{0} \cos (\omega t)$

$\begin{aligned} \sigma(t) &=\sigma_{0} \cos (\omega t+\delta) \\ &=\sigma_{0} \cos (\delta) \cos (\omega t)-\sigma_{0} \sin (\delta) \sin (\omega t) \\ & =\epsilon_{0}\left(E^{\prime} \cos (\omega t)-E^{\prime \prime} \sin (\omega t)\right)\end{aligned}$

 Storage modulus, $E^{\prime} \stackrel{\text { def }}{=} \frac{\sigma_{0}}{\epsilon_{0}} \cos (\delta)$.

 Loss modulus, $E^{\prime \prime} \stackrel{\text { def }}{=} \frac{\sigma_{0}}{\epsilon_{0}} \sin (\delta)$.

### 29.8 Complex formulation of oscillatory response

#### Applied stress, $\sigma(t)=\sigma_{0} e^{i \omega t}$

$\begin{aligned} \epsilon(t) &=\epsilon_{0} e^{i(\omega t-\delta)} \\ &=\sigma_{0}\left[J^{\prime}-i J^{\prime \prime}\right] e^{i \omega t}\\ & =J^{*} \sigma(t)\end{aligned}$

Complex compliance, $J^{*}(\omega) \stackrel{\text { def }}{=} \frac{\epsilon(t)}{\sigma_{0} e^{i \omega t}}=J^{\prime}-i J^{\prime \prime}$

#### Applied strain, $\epsilon(t)=\epsilon_{0} e^{i \omega t}$

$\begin{aligned} \sigma(t) &=\sigma_{0} e^{i(\omega t+\delta)} \\ &=\epsilon_{0}\left[E^{\prime}+i E^{\prime \prime}\right] e^{i \omega t} \\ & =E^{*} \epsilon(t)\end{aligned}$

Complex modulus, $E^{*}(\omega) \stackrel{\text { def }}{=} \frac{\sigma(t)}{\epsilon_{0} e^{i \omega t}}=E^{\prime}+i E^{\prime \prime}$.

#### 29.8.1 Energy dissipation under oscillatory conditions

### 29.9 More on complex variable representation

|  | Kelvin-Voight | Maxwell | Std | Gen |
|:---------------------------:|:--------------------------------:|:-----------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------:|
| $E^{\prime}(\omega)$ | $E$ | $\frac{\tau^{2} \omega^{2}}{\tau^{2} \omega^{2}+1} E$ | $\frac{E_{r e}+E_{r g}\left(\tau_{R}^{(1)} \omega\right)^{2}}{1+\left(\tau_{R}^{(1)} \omega\right)^{2}}$ | $E^{(0)}+\sum_{\alpha} \frac{E^{(\alpha)}\left(\omega \tau_{R}^{(\alpha)}\right)^{2}}{1+\left(\omega \tau_{R}^{(\alpha)}\right)^{2}}$ |
| $E^{\prime \prime}(\omega)$ | $\eta \omega =E \tau_{R} \omega$ | $\frac{\tau \omega}{\tau^{2} \omega^{2}+1} E$ | $\frac{\left(E_{r g}-E_{r e}\right)\left(\tau_{R}^{(1)} \omega\right)}{1+\left(\tau_{R}^{(1)} \omega\right)^{2}}$ | $\sum_{\alpha} \frac{E^{(\alpha)}\left(\omega \tau_{R}^{(\alpha)}\right)}{1+\left(\omega \tau_{R}^{(\alpha)}\right)^{2}}$ |
| $\tan \delta(\omega)$ |  |  | $\frac{\left(E_{r g}-E_{r e}\right)\left(\tau_{R}^{(1)} \omega\right)}{E_{r e}+E_{r g}\left(\tau_{R}^{(1)} \omega\right)^{2}}$ | $\frac{E^{\prime \prime}(\omega)}{E^{\prime}(\omega)}$ |
|  |  |  |  |  |

### 29.10 Time-integration

### 29.11 3D Constitutive equation

#### 29.11.1 BVP for isotropic linear viscoelasticity

#### 29.11.2 Correspondence principle in three dimensions