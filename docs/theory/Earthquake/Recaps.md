## 2. Performance Objectives

## 3. Linear SDOF Response

### Main topics

1) What is a linear elastic response spectrum
- How to find it
- What does it look like
- How does it describe the predicted performance
of an elastic structure?
2) How does damping change elastic behaivor
and the response spectrum?
3) Briefly: what response spectrum is used in
design?

### Recap!

1) What is a linear elastic response spectrum?
2) What is pseudo acceleration?
3) Do you use relative or absolute acceleration? Relative or absolute displacement?
4) How does viscous damping change elastic behavior and the response spectrum?
5) What is the relationship between the design basis DBE and maximum considered $\mathrm{MCE}_{\mathrm{R}}$ demands?
6) For ASCE $7,$ what is roughly the probability of exceedance in 50 years for the $\mathrm{MCE}_{\mathrm{R}}$

## 4. Nonlinear SDOF Response

### Overview

1. Definitions for comparison of nonlinear
behavior.
2. Generation of nonlinear spectra.
3. Trends in nonlinear behavior.
4. Code based R factor.
5. Effects of nonlinear shape on response

### Recap!

- What are the definitions of R, μ , γ
- For what period range are inelastic displacements significantly
amplified?
- For what period range are inelastic displacements close to elastic
displacements?
- How is nonlinear response presented in the code?
- How do the following affect nonlinear response:
  - Stiffness softening?
  - Strength degradation?
  - P Delta?


## 5. Simple SDOF Analysis

### Overview: Linear vs. nonlinear methods for SDOF

* Linear --> response spectrum analysis
* Nonlinear --> how can we do a response
spectrum analysis?

- Nonlinear response spectrum
- Equivalent linearization
- Capacity spectrum
- Coefficient method

### Summary

- Nonlinear response spectrum
- Equivalent linearization
- Capacity spectrum
- Coefficient method

## 6. Simple MDOF Analysis

## 7. Lateral Systems

## 8. Preliminary Design

### Recap

- How might you find a "minimum stiffness" before design
- Outline steps for moment frame preliminary design
- Outline steps for braced frame preliminary design

## 9. Irregular Systems

### Recap

- examples of vertical irregularities
  - offsets in later-load system
  - mass irregularities
  - stiffness irregularities

- examples of horizontal irregularities
  - **Torsional irregularities** <- focus
  - Re-entrant corners
  - Diaphragm eccentricity/ cut-outs
  - Non-parallel lateral load systems
  - Out-of-plane offsets

- What causes torsional irregularities?
  - eccentricity between COM and COR

- Why do short columns perform poorly?
  - small $h$ creates large shear

- Quick strength check for soft story:
  - see eqn above

## 10. Code Procedures

### [Recap](2020312-28:00:00)

- If a structure is designed using a linear analysis method, does that mean it will stay in the linear behavior range during an earthquake?
  - No.
- What are the top two considerations when deciding if it is acceptable to use a linear static design method?
  1. Predeminant first mode
  2. Predictable nonlinear response - clear failure mechanism
- A higher value of R results in what?
  - Smaller design base shear, $V_{base} = V_{el}/R$
- A higher importance factor results in what
  - Larger base shear
- For ELF or linear dynamic analyses, do you need to multiple my $C_d$ to find final displacements?
  - **Yes.** Multiply by $\frac{C_d}{I_e}$
- What constitutes a torsional irregularity?
  - $\frac{\delta_{max}}{\delta_{avg}}\gt 1.2$
- How do you account for accidental torsion in an analysis?
  - ELF: offset forces from COM
  - Dyn: offset COM $\pm 5 \%$
- At what hazard level is *nonlinear* time history analysis. Is this different than for other analysis methods?
  - Typically, $MCE_R$
  - Yes. Other methods based on $\frac{2}{3}MCE_R$
- For NLTH do you need to multiple by $C_d$ to find displacements?
  - No. Because you have not multiplied by...

## 11. Intro to Seismology

### [Overview](GMT20200312-00:39:00)

- Measuring Earthquakes

### Recap

- Are all seismic waves of the same type?
- Do we expect permanent ground deformation after earthquakes?
- Do motion records from one type of seismic zone (say subduction) represent motions from other zones?
- In a normal or reverse fault, will the hanging wall or foot wall experience more damage?
- What is the name of a fault that does not rupture all the way to the surface?

## 12. Ground Motion Quantificaiton

### Recap

- Is it possible to measure a 9.0 using the Richter scale?
  - No, because the richter scale saturates around $M_L ~ 7.0$
- What magnitude measurement do we typically use now?
1. Moment magnitude, $M_w$
2. $\times 31.6$
3. Sig. dur is less subjective
4. No. PGA lacks PGD, duration, freq. content, etc.
5. Key fault parameters
   - slip rate
   - Characheristic magnitue, based on $M_o$ (i.e what is the fault capable of)
   - Fault type
6. Table

|  | Intensity | Sig. Duration | Freq Content |
|---------------------|:--------------:|:---------------:|:--------------:|
| Fwd Directivity | $\uparrow$ | $\downarrow$ | $\downarrow$ |
| $V_{s30}\downarrow$ | - | $\uparrow$ | $\uparrow$ |
| $M \uparrow$ | $\uparrow$ | $\uparrow$ | $\uparrow$ |
| Near Fault | $\uparrow$ | $\downarrow$ | $\uparrow$ |
| Distance $\uparrow$ | $\downarrow$ | $\uparrow$ | $\uparrow$ |

## 13. Ground Motion Models

## Recap - GMM

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

## 14. PBEE
