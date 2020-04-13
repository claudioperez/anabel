# Boundary Value Problems

## Weak form

1. Multiply both sides of the DE or PDE by an arbitrary function
   1. The function must be homogeneous (= 0) where displacement BC’s are specified.
   2. The function must have sufficient continuity for differentiation.
2. Integrate over the domain, e.g. length of the rod
3. Integrate by parts using Green’s theorem to reduce derivatives to their minimum order.
4. Replace the boundary conditions by an appropriate construction.

Note:

1. Weak form is formulated in terms of axial force (or equivalently axial stress as in general mechanics problems). This only involves equilibrium
2. Compatibility relationship is based on infinitesimal strain, .
3. We did not make statements @ stress strain relationships , so the relationship between the strong and weak forms of equilibrium are true for linear & nonlinear materials
4. The weak form (or the structural analysis interpretation as the PVD) provides a framework for finding an approximate solution
5. For the exact solution, we need to look at all possible trail functions (e.g. describing the axial force and VD), which is a formidable task.
6. Rayleigh’s method (and its extension to use superposition of several functions, i.e. the Rayleigh Ritz procedure) provides a convenient way to limit the number of functions that we are examining and since the limited functions may not include the exact solution, the obtained solution will be an approximation
7. We would like to pick simple functions for easy integration & provide a set of algebraic equations that can be solved efficiently. The FEM provides a systematic way for this.