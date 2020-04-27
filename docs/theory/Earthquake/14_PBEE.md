# 14 PBEE

$$\begin{array}{ccccc}
\hline \text { Seismic  } & \text { Engineering } & \text { Damage } & \text { Demand } & \text { Performance } \\
\text { Hazard  }& \text { Demand } & \text { Measures } & \text { Variables } & \text { Targets} \\
\hline
\text { Hazard analysis, } & \text { Structural } & \text { Fragilities for } & \text { - \$ Losses } & Pr_f < \%  \\
\text { Ground } & \text { performance } & \text { failure states } & \text { - Collapse } & \text { - Losses < \$ } \\
\text { motions } & \text { analysis } & \text { -Structural } & \text { - Length of } & \text { -Downtime <t } \\
\hline \lambda \text { (iM) } & \text { G(EDP|IM) } & \text { G(DM|EDP) } & \text { G(DV|DM) }  & \lambda(DV)
\end{array}$$

**Mean annual rate of exceedance:**

$$\lambda(D V)=\iiint G(D V | D M) d G \,(D M | E D P) d G \,(E D P | I M)|d \lambda(I M)|$$

## Seismic input

### Scenario types

<!-- 4/02/20 21:00 -->

#### A. Intensity based

- Assess damage related to a particular **target spectrum**

#### B. Scenario based

- Based on a DSHA particular M, r

#### C. Time based

- Assess damage over specified period of time
- Used to find mean annual rate of exceeding some $-amount of losses.

- Choose intensity measure (IIM) $-\mathrm{PGA}, \mathrm{Sa}(\mathrm{T}),$ entire spectrum, etc
$\cdot \lambda_{y^{*}}(\text { mean annual rate of exceedance })$ from Probabilistic Seismic Hazard Analysis $-$ GMPEs $\rightarrow \mathrm{P}\left(\mathrm{Y}>\mathrm{y}^{*} | \mathrm{m}, \mathrm{r}\right)$
- Probability that a magnitude m occurs
- Probability that EQ occurs at distance $\underline{r}$
\cdot Select hazard levels at which analysis will be done
$-50 \%$ prob of exceedance in $50 \mathrm{yrs}$ $-10 \%$ in 50 yrs $-$ Etc

**Hazard Curve** - Annual freq. of exceedence vs. IM

## [Response Analysis](43:18)

### EDP Selection

- **Drift**
- Stress (Not common)
- **Floor accel.**
  - Ceiling tiles, anchored equipment, etc.
- Floor vel. - rocking/sliding equipment.

#### EDP Matrix

$$\begin{array}{lcccc}
\hline \text { Filename } & \text {edp 1} & \text {edp 2} & \cdots & \text {edp N} \\
\hline \text { GM 1 } & 2.02 & 0.75 & \cdots & 1.05 \\
\text { GM 2 } & 0.98 & 0.27 & \cdots & 0.35 \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
\text { GM M } & 1.44 & 0.88 & \cdots & 1.15 \\
\hline
\end{array}$$

Assume EDPs are **lognormally** distributed.
<!-- Location in video lecture: GMT20200402 - 00:49:07 -->

Select IM for low EDP dispersion - well correlated.

High dispersion will require running more ground motions.
<!-- Location in video lecture: GMT20200402 - 00:50:20 -->

## [Damage Analysis](GMT20200402-52:55)

Approximate inventory of building.

FEMA P-58 provides normative quantities

Each component has:

- A relation: EDP $\mapsto$ Damage
- Damage states
- A relation: Damage $\mapsto$ Repair level

### [Damage States](GMT20200402-1:05:00)

<!-- 1:07:50 -->
1. Sequential - progressively worse damage.
2. Mutually exclusive (e.g. shear or flexural)
3. Simultaneous (e.g. elevator: motor, and/or cable)

### [Fragility Curves](GMT20200402-1:07:50)

$P(D>DS | EPD)$ vs. $EDP$\
$P(g_i<0 | EPD)$ vs. $EDP$

Generated from lab testing, EQ data, analysis, or judgement. see P-58

Dispersion:
$$\beta=\sqrt{\beta_{r}^{2}+\beta_{u}^{2}}$$

- $\beta_r$: aleatory
- $\beta_u$: epistemic

## [Loss Analysis (Monetary)]()


## [Additional Topics](GMT20200414-10:02)

### Incorporating residual drift, collapse

### Measuring death and downtime

### Using linear analysis to find engineering demand parameters

## Aside: Topics in Probability-Stats

$\operatorname{Var}(\mathbf{\bar{x}})=\frac{\sigma^2}{n}$