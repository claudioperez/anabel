# Guest Lecture - GM Selection

Choose a ground motion that is less than worst case, high enough to ensure safety, but not so rare that it's unreasonable.

Outline

- [Example Site](#example-site)
- [DSHA Magnitude Selection]()
- [Soil Sites]()
- [ASCE7/SEI 7-16]()
- [Not all PSHA are equal]()
- [Time History Selection]()
- [Time History Modification]()
- [Details]()

## Example Site

### DSHA

- Earthquake Scenario:
  - $\mathrm{M_w} = 8.2$
  - $\operatorname{R_{rup}}=5.1 \mathrm{km}$
  - $V_{s 30}=760 \mathrm{m} / \mathrm{s}$
- $\mapsto$ Median $\pm \sigma$ R. Spectrum

### PSHA

- Code: **Haz45**
- SSC: **UCERF3**
- GMC: **NGA West2**
- $\mapsto$ Uniform hazard spectrum
  - PSA vs T, at 2% in 50 years

--------

- $MCER_{prob}=\mathrm{PSHA}^{*}$ risk factor $^{*}$ max direction factor
- $MCER_{determ}$ = max of: DSHA* max direction factor and deterministic lower limit
- Site Specific = min $MCER_{prob}$ , $MCER_{determ}$

[Design Spectrum](00:27:00)

- Prelim $=\frac{2}{3} \mathrm{MCER}$
- $80 \% \mathrm{DRS}=80 \%$ mapped based
- Design spectrum $=$ max of prelim and $80 \%$ DRS

## DSHA 

### [Magnitude Selection, SSC](00:36:00)

### [Soil Sites](00:58:20)

- Vs30 is not a fundamental physical parameter, it's an index for the full Vs profile
- Vs30 is intended to provide a clear definition of the "rock" or input motion
- Vs30 is not meant to replace site-specific analysis

The most appropriate process is to run a "rock" site (or as stiff as you can get your profile), then add a site response on top. [[time]](1:05:00)

#### [Basins](1:06:42)

Site response analysis needs at least half a wavelength of thickness to pick up the response to a given period.

- $\mathrm{T}=4 \mathrm{H} / \mathrm{Vs}<-$ thickness of full wavelength
- $\mathrm{H}=\mathrm{T}^{*} \mathrm{Vs} / 2<-$ soil column thickness req'd for half wavelength
- e.g. $\mathrm{H}=2 * 400 / 2=400 \mathrm{m}$

Site response can be performed with "RVT" (Random vibration theory), for example, [Strata](https://www.geoengineer.org/software/strata)

### [Site Coefficients - ASCE7/SEI 7-16](1:11:00)

## PSHA

### [PSHA Quality & Cost](1:16:30)

- Design spectra/PSHA can vary in cost from 5 thousand dollars to 5 million dollars
- Large Regional Models (CEUS SSHAC): millions
- Nuclear facilities: $\$ 200 k-\$ 400 k$
- Regions that need SSC and/or GMC development: $\$150\mathrm{k}-\$250\mathrm{k}$
- Atypical regions (e.g. Hawaii): $\$75\mathrm{k}-\$150\mathrm{k}$
- California: $\$ 15 \mathrm{k}-\$ 30 \mathrm{k}$

Ground Motion Characterization

- California: NGA West 2
- CEUS: NGA East
- PNW (Pacific Northwest?): NGA Subduction

Seismic Source Characterization

- UCERF3
- CEUS: EPRI CEUS model

### [Time History Selection](1:27:16)

- Selection is based on **magnitude**, **distance** and **site class** of the records.
- The deterministic scenario controls, so we use the $\mathrm{M}, \mathrm{R}$ from the deterministic scenario.

Use mag & dist from whichever controls between DSHA and PSHA.

Select motions with low $Sa$ variability in target range, then scale to the intensity of the target spectra.

>"Shallow crustal" more important than mechanism (i.e. strike-slip, reverse oblique) [[time]](1:33:00)

>$R_{rup}$, $V_{s30}$ values may be skewed towards larger values due to tendency towards lognormal distributions.

- Parameters that may impact structural response:
  - duration
  - number of cycles,
  - arias intensity,
  - instantaneous power,
  - Etc.
- All of these parameters are correlated to magnitude, distance and
$V_{s30}$.
- The hope, is that by selecting ground motions with similar $\mathrm{M}, \mathrm{R}$ to the target, we get the "right" parameters, and you get the "right"structural response.

>Don't use too many records from one earthquake



### Time History Modification

[Scaling vs Spectral Matching](1:40:52)

- Spectral matching suppresses the variability of the ground motion (Desired).
- So does scaling, we usually select and scale the records so they closely match the target in the period range of interest.
- If you want the real variability of the structural response, you need more records -- 7 or 11 records can only get you an estimate of the average response.

## Details

### [Directivity](1:45:18)

- Somerville et al. (1997) performed a study that showed:
  - FN component is larger than the FP
  - There are large FN pulses
- **BUT** It's not that simple

### [Rotation](1:53:50)

## Index

- [Direction/Directivity of Ground motion](1:45:18)

- PSHA
  - [Quality & Cost](1:16:30)
- [Rotation of Ground Motion](1:53:50)
- Scale factors
  - [acceptable ranges](1:35:00)
- [Spectral matching](1:40:52)
- [Time History Selection](1:27:16)

## Accronyms

- CEUS - Central and Eastern United States
- SSC - Seismic source characterization (e.g. UCERF3)
- GMC - Ground motion characterization (e.g. NGA West)
