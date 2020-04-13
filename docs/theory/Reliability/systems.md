# Systems

component failure: $p_i = 1-E[S_i]$\
system failure: $p_{sys} = 1-E[\phi(\mathbf{S})]$

Definitions:

- Cut Set : any set of components whose joint failure constitutes failure of
the system
- Min Cut Set : a cut set with minimum set of components (if any component
is removed from the set, the remainder is not a cut set)
- Disjoint Cut Sets: Mutually exclusive cut sets
- Link Set : any set of components whose joint survival constitutes survival
of the system
- Min Link Set : a link set with minimum set of components (if any component is removed from the set, the remainder is not a link set)
- Disjoint Link Sets: Mutually exclusive link sets

A general two state system can be represented as:

- A series system of parallel subsystems, each representing a min cut set, or
- A parallel system of series subsystems, each representing a min link set.

### cut-set formulation

$S=\Pi_{j} c_{j}(\mathbf{s})=\Pi_{j}\left[1-\prod_{i \in C_{j}}\left(1-s_{i}\right)\right] \quad$

$c_{j}(\mathbf{s})=1-\prod_{i \in C_{j}}\left(1-s_{i}\right)$ system function for $j$ th cut set

$S=1-\prod_{j}\left[1-l_{j}(\mathbf{s})\right]=1-\prod_{j}\left(1-\prod_{i \in L_{j}} s_{i}\right) \quad$ link-set formulation
$l_{j}(\mathbf{s})=\prod_{i \in L_{j}} s_{i} \quad$ system functions for $j$ th link set

#### Series systems:
$\begin{aligned} P_{\text {sys}} &=1-\mathrm{E}\left[s_{1} s_{2} \cdots s_{N}\right]=1-\prod_{i=1}^{N} \mathrm{E}\left[s_{i}\right] \\ &=1-\prod_{i=1}^{N}\left(1-p_{i}\right) \end{aligned}$
#### Parallel systems:
$P_{s y s}=1-\mathrm{E}\left[1-\prod_{i=1}^{N}\left(1-s_{i}\right)\right]$
$$
=\prod_{i=1}^{N} p_{i}
$$
#### General systems:
$P_{s y s}=1-\mathrm{E}\left\{\Pi_{j}\left[1-\prod_{i \in C_{j}}\left(1-s_{i}\right)\right]\right\}$ cut-set formulation
$P_{s y s}=1-\mathrm{E}\left\{1-\Pi_{j}\left(1-\Pi_{i \in L_{j}} s_{i}\right)\right\}$
$=\mathrm{E}\left\{\Pi_{j}\left(1-\prod_{i \in L_{j}} s_{i}\right)\right\} \quad$ link-set formulation

## 8.3 

### 8.3.2

#### Minimum cut sets

#### Minimum link sets

#### Disjoint cut sets

#### Disjoint link sets

## 8.4 Bounds on Series Systems

### Unimodal bound

(8.30,31)

### Bimodal bound

(8.35,38)

### Trimodal bound

(8.39,40)

## 8.5 Bounds by Linear Programming

(8.46)

### Advantages of LP

- Unified approach for series, parallel and general systems.
- May incorporate marginal, joint, or conditional component probabilities
- Results in the narrowest possible bounds for the given information.

### Disadvantages

- Problem size is $2^N$

## 8.6 Matrix-based System Reliability

## 8.7 Structural (Physics-based) System Reliability