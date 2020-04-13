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
- Boore, Joyner, Fumal (1997): $M, V_{s30}, T, h\rightarrow \ln (\operatorname{Sa}( \_, \zeta=5\%))$
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
   - Median $+1.0 \sigma$
3. Ground motion parameters selected based on the controlling hazard
   - May need to use RS envelope if no grnd mtn clearly controls, but not prefered

## Probabilistic Analysis

1. Now factor in (a) probability of different rupture locations, and (b) recurrence relationships for site.
   - Rupture distance pdf is a function of fault geometry.
2. Now include uncertainty in GMMs.
3. Combine all sources of hazard to predict probability of an intensity measure being exceeded.


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

- $PSA$: Pseudo-absolute acceleration response spectrum (g)
- $PGA$: Peak ground acceleration (g)
- $PGV$: Peak ground velocity (cm/s)
- $S_d$ : Relative displacement response spectrum (cm)
- $M_{w}$ : Moment magnitude
- $R_{RUP}$ : Closest distance to coseismic rupture (km), used in ASK13, CB13 and CY13. See Figures a, b and c for illustation
- $R_{JB}$ : Closest distance to surface projection of coseismic rupture (km). See Figures a, b and c for illustation
- $R_{X}$ : Horizontal distance from top of rupture measured perpendicular to fault strike (km). See Figures a, b and c for illustation
- $R_{y0}$ : The horizontal distance off the end of the rupture measured parallel to strike (km)
- $V_{S30}$ : The average shear-wave velocity (m/s) over a subsurface depth of 30 m
- $U$ : Unspecified-mechanism factor:  1 for unspecified; 0 otherwise
- $FRV$ : Reverse-faulting factor:  0 for strike slip, normal, normal-oblique; 1 for reverse, reverse-oblique and thrust
- $FNM$ : Normal-faulting factor:  0 for strike slip, reverse, reverse-oblique, thrust and normal-oblique; 1 for normal
- $FHW$ : Hanging-wall factor:  1 for site on down-dip side of top of rupture; 0 otherwise
- $Dip$ :  Average dip of rupture plane (degrees)
- $ZTOR$ : Depth to top of coseismic rupture (km)
- $ZHYP$ : Hypocentral depth from the earthquake
- $Z1.0$ : Depth to Vs=1 km/sec
- $Z2.5$ : Depth to Vs=2.5 km/sec
- $W$ :   Fault rupture width (km)
- $FAS$ :    0 for mainshock; 1 for aftershock
- $DDPP$ :   Directivity term, direct point parameter; uses 0 for median predictions
- $PGAr (g)$ :  Peak ground acceleration on rock (g), this specific cell is updated in the cell for BSSA14 and CB14, for others it is taken account for in the macros
- $ZBOT (km)$ : The depth to the bottom of the seismogenic crust
- $ZBOR(km)$ : The depth to the bottom of the rupture plane
