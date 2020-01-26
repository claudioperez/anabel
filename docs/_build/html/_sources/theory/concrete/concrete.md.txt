# 11 SMRF

- Joint Shear
  
- Anchorage

- Column Shear

- $\sum M_{nc} \ge \frac{6}{5}\sum M_{nb}$ (Check one floor below roof)

## Columns

### SEE JOINT SECTION

## Beams

#### Shear 

$V_u = f(M_pr)$

$V_c = 0$ in plastic hinge regions.

<div class="page"/>

# 12 Walls

$\delta_c$ Wall displacement capacity at top of wall

## Resistance Factors

Axial/Moment:  
$\phi = 0.65-0.9$

Shear:\
$\phi = 0.75$, For $V_u$ amplified by $\Omega_v$\
$\phi = 0.60$ if $V_n < V(M_n)$\
for squat walls, take $\phi=0.6$

Coupling beam shear:\
$\phi = 0.85$ diagonally reinforced\
$\phi = 0.75$ otherwise

## 13.4 Behavioral Observations

### Slender vs Squat

------------------

### Flexural Response

------------------

### Lateral stability

------------------


$\lambda_c = \frac{c}{b}\frac{l_w}{b} > 40$

$b \geq \sqrt{0.025cl_w}$ **for SBEs**.

### Dynamic response

-----------------------------



## Preliminary Proportioning for $V_b$

assume $l_{be}=0.2l_w$

## Detailing 

long. bar fracture:\
$A_sf_y \ge A_{be}f_r \therefore \rho_l \ge 6\sqrt{f'_c}/f_y$

Cut-offs:\
$0.8l_w$ or $l_d$ above next floor

No-splice zone:\
min($20'$, $h_{1^{st}floor}$)

## Slender Walls w/ C. Section

### Distributed rebar

$\rho_l, \quad \rho_t=\frac{2A_{si}}{ts}\ge 0.0025$ if $V_u \ge \lambda A_{cv}\sqrt{f'_c}$ (18.10.2.1)

### Boundary Elements

SEE NOTES PAGE 163 B.E. FRACTURE

### Shear

$V_u=\frac{M_n[P_u]}{M_u}\omega_vV_{code}$

### SBE

---------------------------

Required if \
$c \ge\dfrac{l_w}{900(\delta_{u}/h_w)} \quad \text{or} \quad \sigma \ge \dfrac{f'_c}{5}$

use given graph to determine $c$.

for min. height of $h_{be} = \text{max}(l_w, \frac{M_{u,cs}}{4V_{u,cs}})$

- $A_{sh} \ge 0.09 s b_c \frac{f'_c}{f_y} \ge0.3\left(\frac{A_g}{A_{ch}}-1\right)sb_c \frac{f'_c}{f_y}$

- 

## 13.9 Walls w/o C. Section

## 13.10.1 Conventional Squat Walls

## 13.16 - Openings

Tie region: $A_s=T_u/\phi f_y$\
Strut region: $P_u \leq \phi P_o$

$\phi = 0.65$ in wall piers\
$\phi = 0.60$ for wall shear\
$\phi = 0.75$ otherwise

## 13.12 Coupled Walls

----------------------

### Coupling beams

$$
V_{n}=2 A_{v d} f_{y} \sin \alpha \leq 10 \sqrt{f_{c}^{\prime}} A_{c w}
$$

- $\phi$ for $A_{tr}$ is $0.75$


### Wall Piers

$P_u=1.2P_D+0.5P_L+n_sP_E$


# 13 Gravity Framing

## Columns

### Confinement

if $P_u \ge 0.35 A_gf'_c$:

- Support all bars with $135^o$ hook.

- $\frac{A_{sh}}{sb_c}\ge 0.3..., 0.09..., 0.2...$

### Shear

$V_u = f(M_{pr}(P_u))$, often $\frac{2M_{pr}}{l_w}$

## Beams

FIGURE FROM NOTES PAGE 200

## Slabs

$V_n = 4\lambda_s\sqrt{f'_c}b_o d$

DRIFT CAPACITY

<div class="page"/>

# 14 Diaphragms

### Moment 

$M_u =\dfrac{wl^2}{8}\le \phi M_n = 0.9(A_sf_y0.9d)$

$T_u=C_u=\frac{M_u}{jd}$

$T_u \le Tn = 0.9A_sfy$

$C_u \le \phi \alpha P_{no} = (0.65)(0.8)(A_sf_y+0.85A_cf'_c)$

### Collectors

#### Shear friction

$\Omega_oV_u\le \underbrace{\phi}_{0.75} \underbrace{\mu}_{1.4} A_{sf}f_y$

- Give $A_{sf}$ as $in^2/ft$


### Openings

#### Comp. Zones

- Confine if $\frac{C}{A}\ge 0.2f'_c$