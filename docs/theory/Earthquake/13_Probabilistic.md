# 13 Seismic Hazard Analysis

## Ground Motion Models

$$\begin{aligned}
\text { GMM: } X & \rightarrow Y \\
X: & M_W, R_{jb}, V_{s30}, \text{etc} \\
Y: & \operatorname{Sa}(T), PGA, \text{etc}\\
\end{aligned}$$

- Wells and Coppersmith (1994): $\mathrm{M}_{\mathrm{w}}=4.38+1.49 \log_{10} \mathrm{L} \pm \sigma$
  - $L$: Fault length [km]
  - $\sigma \approx 0.26$
- Boore, Joyner, Fumal (1997): $M, V_{s30}, T, h\rightarrow \log (\operatorname{Sa}( \_, \zeta=5\%))$
- NGA West 2 (2014) (Shallow Crustal)
- NGA East
- NGA Subduction

**Lowest usable freq.** (L 13 slide 14)

**GMPEs are for RotD50 with 5% damping**

$a_{\mathrm{ROT}}(t, \theta)=\stackrel{\mathrm{H}_{1}}{a_{1}}(t) \cos (\theta)+\stackrel{\mathrm{H}_2}{a_{2}}(t) \sin (\theta)$

- For each angle $(0-180)$ compute spectral ordinate
- RotD00 : $-\min$, RotD100 $-\max$
- RotD50 50 percentile (median)

### General GMM Assumptions

- Response quantities ($Y$) are lognormally distributed.
- $\ln (Y) = \hat{f}(M_W)$
- Relation influenced by site
- Relation influenced by source (fault)

### Recap - GMM

- Why do we need ground motion models?
  - Predict local GM parameters
- Are west coast ground motions use to determine east coast ground motions models?
  - NO
- What distribution do we assume for our ground motion parameters?
  - lognormal
- How is uncertainty included in the models?
  - $\ln{y_i} = \ln{y_i} + \sigma_i$
- Is the prediction equation the same for Sa(T_1) and Sa(T_2)
  - No. Form may/may not be same. Coefficients will be different.

## Deterministic Analysis

> In CA, deterministic analysis is used in near-fault situations to find limit on hazard for code response spectra.

1. Identification and characterization of all EQ sources capable of producing significant ground motion at the site
   1. types
   2. characteristic mag (based on $M_?$)
   3. recent history, geological evidence, historda
2. Estimate the earthquake demands at the site for each source
   - Use GMMs
   - Assume minimum rupture distances
   - Use local soil properties
   - Median $+ 1.0 \sigma$
3. Ground motion parameters selected based on the controlling hazard
   - May need to use RS envelope if no grnd mtn clearly controls, but not prefered

## Probabilistic Analysis

1. Now factor in (a) probability of different rupture locations, and (b) recurrence relationships for site.
2. Now include uncertainty in GMMs.
3. Combine all sources of hazard to predict probability of an intensity measure being exceeded.

### 1 

#### a)

   - Rupture distance pdf is a function of fault geometry.
  
#### b) recurrence relationship

- Gutenberg-Richter: $\log \lambda_{m}=a-b$ [ref](GMT20200319-1:12:50)
  - $\mathrm{P}\left[M\lt m | M \gt m_{0}\right]=\frac{\lambda_{m_{0}}-\lambda_{m}}{\lambda_{m_{0}}}=\frac{P(A \cup B)}{P(B)}$ ([ref](GMT20200331-09:43))
- Truncated exponential
  - $\mathrm{P}\left[M \lt m | m_{0}\lt M<m_{\max }\right]=\frac{\lambda_{m_{0}}-\lambda_{m}}{\lambda_{m_{0}}-\lambda_{m_{\max }}}$([ref](GMT20200331-11:55))

Probability of mag $m$ or greater ([ref](GMT20200331-09:15)):

### 2 Uncertainty in GMM

- $\mathrm{MCER}_{\text {prob }}=\mathrm{PSHA} *$ risk factor maximum direction factor
- $\mathrm{MCER}_{\text {determ }}=$ maximum of: DSHA* maximum direction factor and deterministic lower limit
- site Specific $=$ minimum of $\mathrm{MCER}_{\text {prob }}$ and MCER determ

$$\begin{aligned}
&\begin{array}{c}
S_{a}=S_{D S}\left(0.4+0.6 \frac{T}{T_{0}}\right) \text { for } T<T_{0} \\
S_{a}=S_{D S} \text { for } T_{0}<T<T_{S} \\
S_{a}=\frac{s_{D 1}}{T} \text { for } T_{S}<T<T_{L}
\end{array}\\
&S_{a}=\frac{s_{D 1} \cdot T_{L}}{T^{2}} \text { for } T>T_{L}\\
\end{aligned}$$

#### [Earthquake Temporal Occurance](GMT20200331-000520-00:31:42)

$\text { Poisson distribution: } \mathrm{P}_{m}(N \geq 1)=1-e^{-\lambda_{m} t}$\
$\lambda_m = -\frac{1}{t}\log{(1-P_m)}$\
$RI=\frac{1}{\lambda}$