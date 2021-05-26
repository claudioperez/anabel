# Inelasticity in Frames

The `elle-002x` series of studies are centered around a 2D reinforced concrete portal frame.


## Contents

- `section`
- `linear-frame`
- `corot-frame`
- `reliability`

- [Static Analysis of a Linear Frame](elle-0020) This notebook uses a beam element from the `elle.beam2d` library.
- [`elle-0020-oo`](elle-0020-oo) This notebook uses the default beam element applied by the `emme.SkeletalModel` generator.

- [Static Analysis of a Geometrically Nonlinear Frame](elle-0021)


<!-- - [`opsy-0020`]() -->



## Model Description

Structure
: The structure is depicted in @fig:frame

![](img/frame.svg){#fig:frame}

Girders
: T-girders with the following properties:

![](tee.svg)

Columns
: Square $30\times 30$ concrete columns fixed at the base.

