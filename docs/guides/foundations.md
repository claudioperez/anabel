---
abstract:  A novel sofware protocol for finite element modeling is
  presented that draws on concepts which have recently revolutionized
  the fields of deep learning and artificial intelligence. The
  presentation is broken into two independent sections. In the first
  section, a high-level picture is painted of a novel programming
  paradigm which one might argue is largely responsible for modern
  developments in the field of deep learning. The neural network
  abstraction is presented as a model for implementing computations
  which admit mathematical promises. This section concludes with a
  illustrative implementation of forward propagation developed in the
  C++ programming language. The second section builds on the
  illustration of the first to propose a prototcol for model composition
  which is compatible with the design of state-of-the-art systems like
  OpenSees, as well as modern deep learning libraries. Focus is placed
  on a unique style of finite element modeling which is particularly
  powerful for problems of reliability and extreme hazards. The concepts
  of composition and compile-time programming which were presented
  earlier through C/C++ are expanded upon.
author:
- Claudio Perez
date: Spring, 2021
geometry:
- margin=1truein
institute: University of California, Berkeley
keywords:
- nonlinear finite element analysis
- reliability
- resilience
- inelasticity
- algorithmic differentiation
subtitle: (subtitle)
title: Integrable Composition of Finite Element Networks
---



```{=html}
<!-- >*"The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."* - Edsger Dijkstra-->
```
## Preface

Engineers have become quite good at approximating how much a body with
precisely *these* properties and exactly *that* composition might
deflect, if this particular load acted on it in exactly *that* way. The
trouble is we don't really know what we're modeling. The earthquake will
come from an angle we didn't consider,
`<!-- the trouble is we've still got no idea what loads will act where, and  -->`{=html}
and exactly how long the foreman`<!--contractor's foreman-->`{=html} let
the garden hose run after the concrete showed up late, is a secret
buried deeper than the inspector's forgotten Coke bottle, laid to rest
for eternity when the footing was poured. It's an uncertain world.
Exciting strides are being taken to characterize this uncertainty and
researchers have made powerful links between the axioms of probability
and finite element (FE)
methods`<!-- [@derkiureghianStructural1986]-->`{=html}. But unlike the
FE method which was quickly decomposed into linear algebraic operations
and put into FORTRAN to be disseminated across the globe
[@clough1960finite; @wilson1970sap; @wilson1975etabs], these models
don't lend themselves so directly to traditional paradigms of scientific
computing.

Problems like this can be considered alongside a broader class of
"next-generation" modeling problems for which a FE model will not
provide the end result itself, but instead play a role inside an
encompassing analysis procedure. Several problems of great relevance to
infrastructure resilience entail these demands including regional-scale
modeling, parameter calibration, system reliability and design/topology
optimization. Modern attempts to tackle such problems often incur large
computational overheads and quickly grow in implementation complexity,
making remotely modular and reusable solutions exceedingly rare.
Furthermore, these solutions typically demand additional information
from components of the FE model to be hard-coded into the framework in
order to obtain exact or approximate gradients.
`<!-- Although typically applied to problems with uncertainty, programs that are thought of abstractly as artificial neural networks (ANNs) are not nebulous black boxes; they are precise mathematical expressions.-->`{=html}
`<!-- $\null\nobreak\ensuremath{\blacksquare}$ -->`{=html}

```{=html}
<!-- This project will result in publications presenting a novel formulation of the finite element method  -->
```
This project proposes a modeling paradigm and a reference programming
framework that mirrors the graph-like structure of artificial neural
networks (ANNs)
`<!-- to allow the same rigorous deductions to be made about nonlinear and path-dependent finite element (FE) models.  -->`{=html}
so that elements of a FE model, as well as the model itself, can be
algorithmically differentiated and parallelized, with the modularity of
a node in a computational graph.

The resulting framework exposes a convenient high-level interface
through the Python programming language which can be used to state,
compose, and efficiently deploy complex physical models.
`<!--, and algorithmically derive *analytic* response gradients as they develop new finite elements-->`{=html}
`<!--, all without computing a single derivative by hand (should one be so careless)-->`{=html}.
Not only does this open the door for powerful new computational
procedures and inverse analyses, but it substantially simplifies how we
interact with FE software, and provides an accessible tool for
engineering education.

```{=tex}
\setcounter{chapter}{1}
```
# Part I: Composition of Expressions

```{=html}
<!-- Recently the field of deep learning has found its way into every day life and it seems as though the sorcerers at Google. The work horses beneath these monumental strides in technology are available to the public through tools like TensorFlow, Matlab, and PyTorch which have become household names in the realm of scientific computing.  -->
```
In @frostig2018compiling, researchers from the Google Brain team begin
their presentation of the revolutionary JAX project with a simple
observation; machine learning workloads tend to consist largely of
subroutines satisfying two simple properties:

1.  They are *pure* in the sense that their execution is free of side
    effects, and
2.  They are *statically composed* in the sense that they implement a
    computation which can be defined as a static data dependency on a
    graph of some set of *primitive* operations (This will be further
    explained).

The exploitation of these properties with techniques such as automatic
differentiation and just-in-time compilation has become a central
component of tools like TensorFlow, Matlab and PyTorch, which in the
recent decade have been at the forefront of an "explosion" in the field
of deep learning. The utility that these ground-breaking platforms
provide is that they present researchers and developers with a high
level platform for concisely staging mathematical computations with an
intuitive "neural network" paradigm. By their structure, these resulting
models admit analytic gradients which allows for the application of
powerful optimization algorithms.

```{=html}
<!-- An excellent example of such a machine learning model is the Python library [BRAILS] which provides users a means for building models which infer structural properties about buildings directly from images. Phrased differently, what such a library produces is a mapping $f: A \rightarrow B$ which will take a a set of pixels, $a$, and yield a classification $b \in B$. In order to build this function $f$, the BRAILS library creates a function $F: \Theta \rightarrow A \rightarrow B$ using primitives from the [TensorFlow] library (i.e., A function taking an element $\theta \in \Theta$) -->
```
Algorithmic differentiation (AD) is a class of computational methods
which leverage the chain rule in order to exactly evaluate[^1] the
gradient of a mathematical computation which is implemented as a
computer program. These methods are often further categorized into
*forward* and *reverse* modes, the later of which further specializes to
the *backpropagation* class of algorithms which has become ubiquitous in
the field of machine learning. These methods generally leverage either
(1) source code transformation techniques or (2) operator overloading so
that such a gradient computation can be deduced entirely from the
natural semantics of the programming language being used. Much of the
early literature which explores applications of AD to fields like
structural modeling [@ozaki1995higherorder; @martins2001connection]
employ the first approach, where prior to compilation, an pre-processing
step is performed on some source code which generates new code
implementing the derivative of an operation. This approach is generally
used with primitive languages like FORTRAN and C which lack built-in
abstractions like operator overloading and was likely selected for these
works because at the time, such languages were the standard in the
field. Although these compile-time methods allow for powerful
optimizations to be used, they can add significant complexity to a
project. Languages like C++, Python and Matlab allow user-defined types
to *overload* built-in primitive operators like `+` and `*` so that when
they are applied to certain data types they invoke a user-specified
procedure. These features allow for AD implementations which are
entirely contained within a programming language's standard ecosystem.

## Building a Differentiable Model

There are several approaches which may be followed to arrive at the same
formulation of forward mode automatic differentiation
[@hoffmann2016hitchhiker]. A particularly elegant entry point for the
purpose of illustration is that used by @pearlmutter2007lazy which
begins by considering the algebra of *dual numbers*. These numbers are
expressions of the form $a + b\varepsilon$, $a,b \quad \in \mathbb{R}$
for which addition and multiplication is like that of complex numbers,
save that the symbol $\varepsilon$ be defined to satisfy
$\varepsilon^2 = 0$.

Now consider an arbitrary analytic function
$f: \mathbb{R} \rightarrow \mathbb{R}$, admitting the following Taylor
series expansion:

$$
f(a+b \varepsilon)=\sum_{n=0}^{\infty} \frac{f^{(n)}(a) b^{n} \varepsilon^{n}}{n !} 
$$

By truncating the series at $n = 1$ and adhering to the operations
defined over dual number arithmetic, one obtains the expression

$$
f(a + b \varepsilon) = f(a) + bf^\prime (a) \varepsilon, 
$$

This result insinuates that by elevating such a function to perform on
dual numbers, it has effectively been transformed into a new function
which simultaneously evaluates both it's original value, and it's
derivative $f^\prime$.

For instance, consider the following pair of mappings:

$$
\begin{aligned}
f: & \mathbb{R} \rightarrow \mathbb{R} \\
   & x \mapsto (c - x)
\end{aligned}
$$

$$
\begin{aligned}
g: & \mathbb{R} \rightarrow \mathbb{R} \\
   & x \mapsto a x + b f(x)
\end{aligned}
$$ {\#eq:dual-area}

One can use the system of dual numbers to evaluate the gradient of $g$
at some point $x \in \mathbb{R}$, by evaluating $g$ at the dual number
$x + \varepsilon$, where it is noted that the dual coefficient $b = 1$
is in fact equal to the derivative of the dependent variable $x$. This
procedure may be thought of as *propagating* a differential
$\varepsilon$ throughout the computation.

$$
\begin{aligned}
g(x + \varepsilon)   & = a (x + \varepsilon) + b f(x + \varepsilon ) \\
                     & = a x + a \varepsilon + b ( c - x - \varepsilon) \\
                     & = (a - b) x + bc + (a - b) \varepsilon 
\end{aligned}
$$

The result of this expansion is a dual number whose real component is
the value of $g(x)$, and dual component, $a-b$, equal to the gradient of
$g$ evaluated at $x$.

Such a system lends itself well to implementation in programming
languages where an object might be defined to transparently act as a
scalar. For example, in C, one might create a simple data structure with
fields `real` and `dual` that store the real component, $a$ and *dual*
component, $b$ of such a number as show in @lst:struct-dual.

``` {#lst:struct-dual .cpp caption="Basic dual number data type."}
typedef struct dual {
    float real, dual;
} dual_t;
```

Then, using the operator overloading capabilities of C++, the
application of the `*` operator can be overwritten as follows so that
the property $\varepsilon = 0$ is enforced, as show in @lst:dual-mult.

``` {#lst:dual-mult .cpp caption="Implementation of dual-dual multiplilcation in C++"}
dual_t dual_t::operator * (dual_t a, dual_t b){
    return (dual_t) {
        .real = a.real * b.real,
        .dual = a.real * b.dual + b.real * a.dual
    };
}
```

Similar operations are defined for variations of the `*`, `+`, and `-`
operators in a single-file library in the appendix. With this complete
arithmetic interface implemented, forward propagation may be performed
simply with statements like that of @lst:dual-expr.

``` {#lst:dual-expr .cpp caption="Dual number arithmetic in C++ through operator overloading."}
int main(int argc, char **argv){
  dual_t x(6.0,1.0);
  dual_t dg = a * x + (c - x) * b;
  printd(dA);
}
```

## Template Expressions

The simple `dual_t` numeric type developed in the previous section
allows both an expression, and its gradient to be evaluated
simultaneously with only the minor penalty that each value stores an
associated `dual` value. However, if such a penalty was imposed on every
value carried in a modeling framework, both memory and computation
demands would scale very poorly. In order to eliminate this burden, the
concept of a *template expression* is introduced. This is again
illustrated using the C++ programming language in order to maintain
transparency to the corresponding machine instructions, but as expanded
in the following part, this technique becomes extremely powerful when
implemented in a context where access to the template expression in a
processed form, such as an abstract syntax tree, can be readily
manipulated. Examples of this might be through a higher-level language
offering introspective capabilities (e.g., Python or most notably Lisp),
or should one be so brave, via compiler extensions (an option which is
becoming increasingly viable with recent developments and investment in
projects like LLVM [@lattner2004llvm]).

In @lst:dual-template, the function $g$ has been defined as a *template*
on generic types `TA`, `TB`, `TC` and `TX`. In basic use cases, like
that shown in @lst:dual-template, a modern C++ compiler will deduce the
types of the arguments passed in a call to `G` so that dual operations
will only be used when arguments of type `dual_t`{.c} are passed.
Furthermore, such an implementation will produce a derivative of `G`
with respect to whichever parameter is of type `dual_t`{.c}. For
example, in the assignment to variable `dg` in @lst:dual-template, a
`dual_t`{.c} value is created from variable `x` within the call to `G`,
which automatically causes the compiler to create a function with
signature `dual_t (*)(dual_t, real_t, real_t, real_t)`{.c} which
propagates derivatives in `x`.

``` {#lst:dual-template .cpp caption="Forward propagation using template metaprogramming"}
template<typename TX,typename TB, typename TA, typename TC>
auto G(TX x, TB b, TA a, TC c){
  printd(b*x);
  printd(c - x);
  return a * x + (c - x) *b;
}

int main(int argc, char **argv){
  real_t a = 60.0, c=18.0, b=18.0;
  real_t x = 6.0;
  real_t g = G(x,b,a,c);
  dual_t dg = G(dual_t(x,1),b,a,c);
  printd(dg);
}
```

In this minimal example, the set of operations for which we have
explicitly defined corresponding dual operations constitute a set of
*primitive* operations which may be composed arbitrarily in expressions
which will implicitly propagate machine precision first derivatives.
These compositions may include arbitrary use of control structures like
`while`{.c} / `for`{.c} loops, branching via `if`{.c} statements, and by
composition of such operations, numerical schemes involving indefinite
iteration.

A thorough treatment of AD for iterative procedures such as the use of
the Newton-Raphson method for solving nonlinear equations is presented
by @beck1994automatic, @gilbert1992automatic and
@griewank1993derivative. Historic accounts of the development of AD can
be found in @iri2009automatic. Such treatments generally begin with the
work of Wengert (e.g. @wengert1964simple, ).
`<!-- Substantial contributions to the field have been made by A. Griewank [@griewank1991automatic; @griewank1989automatic]. -->`{=html}
A more recent work, @elliott2009beautiful develops an elegant
presentation and implementation of forward-mode AD in the context of
purely functional programming.

```{=tex}
\setcounter{chapter}{2}
```
# Part II: The Finite Element Method

```{=html}
<!-- ## Differentiable Modeling -->
```
The finite element method has become the standard method for modeling
PDEs in countless fields of research, and this is due in no small part
to the natural modularity that arises in the equations it produces.
Countless software applications and programming frameworks have been
developed which manifest this modularity, each with a unique flavor.

```{=html}
<!-- [DEALii] [MOOSE] -->
```
A particularly powerful dialect of FE methods for problems in resilience
is the composable approach developed through works like
@spacone1996fibre where the classical beam element is generalized as a
composition of arbitrary section models, which may in turn be comprised
of several material models. Platforms like [FEDEASLab] and [OpenSees]
are designed from the ground up to natively support such compositions
[@mckenna2010nonlinear]. This study builds on this composable
architecture by defining a single protocol for model components which
can be used to nest or compose elements indefinitely.

In order to establish an analogous structure to that of ANNs which is
sufficiently general to model geometric and material nonlinearities, we
first assert that such a FE program will act as a mapping,
$\Phi: \Theta\times\mathcal{S}\times\mathcal{U}\rightarrow\mathcal{S}\times\mathcal{U}$,
which walks a function,
$f: \Theta\times\mathcal{S}\times\mathcal{U}\rightarrow\mathcal{S}\times\mathcal{R}$
(e.g. by applying the Newton-Raphson method),
`<!--and its derivative $D_U f: (\Theta, \mathcal{S},\mathcal{U})\rightarrow\mathcal{U}\rightarrow (\mathcal{S},\mathcal{R})$,-->`{=html}
until arriving at a solution, $(S^*,U^*)=\Phi(\Theta,S_0,U_0)$ such that
$f(\Theta,S_0,U^*) = (S^*, \mathbf{0})$ in the region of $U_0$. In
problems of mechanics, $f$ may be thought of as a potential energy
gradient or equilibrium statement, and the elements of $\mathcal{U}$
typically represent a displacement vector, $\mathcal{R}$ a normed force
residual, $\mathcal{S}$ a set of path-dependent state variables, and
$\Theta$ a set of parameters which an analyst may wish to optimize
(e.g. the locations of nodes, member properties, parameters, etc.). The
function $f$ is composed from a set of $i$ local functions
$g_i: \theta\times s\times u \rightarrow s\times r$ that model the
response and dissipation of elements over regions of the
`<!--discretized-->`{=html} domain, which in turn may be comprised of a
substructure or material function, until ultimately resolving through
elementary operations, $h_j: \mathbb{R}^n\rightarrow\mathbb{R}^m$
(e.g. Vector products, trigonometric functions, etc.).

In a mathematical sense, for a given solution routine, all of the
information that is required to specify a computation which evaluates
$\Phi$, and some gradient, $D_\Theta \Phi$, of this model with respect
to locally well behaved parameters $\Theta$ is often defined entirely by
the combination of (1) a set of element response functions $g_i$, and
(2) a graph-like data structure, indicating their connectivity (i.e.,
the static data dependency of @frostig2018compiling) . For such
problems, within a well-structured framework, the procedures of deriving
analytic gradient functions and handling low-level machine instructions
can be lifted entirely from the analyst through abstraction, who should
only be concerned with defining the trial functions, $g_i$, and
establishing their connectivity.

## The `anabel` Library

An experimental library has been written in a combination of Python, C
and C++ which provides functional abstractions that can be used to
compose complex physical models. These abstractions isolate elements of
a model which can be considered *pure and statically composed* (PSC),
and provides an interface for their composition in a manner which allows
for parameterized functions and gradients to be concisely composed. In
this case, Python rarely serves as more than a configuration language,
through which users simply annotate the entry and exit points of PSC
operations. Once a complete operation is specified, it is compiled to a
low-level representation which then may be executed entirely independent
of the Python interpreter's runtime.

Ideally such a library will be used to interface existing suites like
FEDEASLab and OpenSees, but for the time being all experiments have been
carried out with finite element models packaged under the `elle`
namespace. Of particular interest, as expanded on in the conclusion of
this report, would be developing a wrapper around the standard element
API of OpenSees to interface with XLA, the back-end used by the `anabel`
library. The `anabel` package is available on [PyPi.org] and can be
`pip` installed, as explained in the online documentation.

Functions in the `anabel` library operate on objects, or collections of
objects, which implement an interface which can be represented as
$(S^*,U^*)=\Phi(\Theta,S_0,U_0)$, and generally can be classified as
pertaining to either (1) composition or (2) assembly related tasks.
Composition operations involve those which create new objects from
elementary objects in a manner analogous to mathematical composition.
These operations include the \[anabel.template\] decorator for creating
composable operations.

Assembly operations involve more conventional model-management utilities
such as tracking and assigning DOFs, nodes and element properties. These
are primarily handled by objects which subclass the [anabel.Assembler]
through model building methods. These operations are generally required
for the creation of OpenSees-styled models where, for example, a model
may contain several types of frame elements, each configured using
different section, material, and integration rules.

## Examples

Two examples are presented which illustrate the composition of a
physical model from a collection of PSC routines. In these examples, we
consider a function, $F: \mathbb{R}^9 \rightarrow \mathbb{R}^9$, that is
the finite element model representation of the structure shown in
@fig:frame. For illustrative purposes, all model components are defined
only by a single direct mapping, and properties such as stiffness
matrices are implicitly extracted using forward mode AD. The example
[@elle-0050] in the online documentation illustrates how components may
define their own stiffness matrix operation, increasing model
computational efficiency.

![Basic frame]

Columns measure $30 \times 30$ inches and the girder is cast-in-place
monolithically with a slab as show in @fig:tee-dims. The dimensions
$b_w$ and $d - t_f$ are known with reasonable certainty to both measure
$18$ inches, but due to shear lag in the flange and imperfections in the
finishing of the slab surface, the dimensions $b_f$ and $t_f$ are only
known to linger in the vicinity of $60$ inches and $6$ inches,
respectively. Reinforcement is neglected and gross section properties
are used for all members. Concrete with a $28$-day compresive strength
of $f^\prime_c=4$ ksi is used, and the elastic modulus is taken at
$3600$.

![Standard T-shaped girder cross section.]

We are equipped with the Python function in @lst:beam-no1 which contains
in its closure a definition of the function `beam` implementing a
standard linear 2D beam as a map from a set of displacements, $v$, to
local forces $q$.

``` {#lst:beam-no1 .python caption="Classical beam element implementation."}
@anabel.template(3)
def beam2d_template(
    q0: array = None,
    E:  float = None,
    A:  float = None,
    I:  float = None,
    L:  float = None
):
    def beam(v, q, state=None, E=E,A=A,I=I,L=L):
        C0 = E*I/L
        C1 = 4.0*C0
        C2 = 2.0*C0
        k = anp.array([[E*A/L,0.0,0.0],[0.0,C1,C2],[0.0,C2,C1]])
        return v, k@v, state
    return locals()
```

### Reliability Analysis

Our first order of business is to quantify the uncertainty surrounding
the dimensions $t_f$ and $b_f$. We seek to approximate the probability
that a specified displacement, $u_{max}$, will be exceeded assuming the
dimensions $t_f$ and $b_f$ are distributed as given in @tbl:rvs and a
correlation coefficient of $0.5$. This criteria is quantified by the
limit state function @eq:limit-state

$$
g(t_f, b_f) = u_{max} - \Phi(t_f, b_f) 
$$ {\#eq:limit-state}

   Variable   Marginal Distribution   Mean   Standard Deviation
  ---------- ----------------------- ------ --------------------
    $t_f$           Lognormal         $6$          $0.6$
    $b_f$           Lognormal         $60$          $6$

  : Distributions of random variables. {\#tbl:rvs}

To this end, a first order reliability analysis (FORM) is carried out,
whereby the limit state function $g$ is approximated as a hyperplane
tangent to the limit surface at the so-called design point,
$\mathbf{y}^{*}$, by first-order truncation of the Taylor-series
expansion. This design point is the solution of the following
constrained minimization problem in a standard normal space, which if
found at a global extrema, ensures that the linearized limit-state
function is that with the largest possible failure domain (in the
transformed space):

$$\mathbf{y}^{*}=\operatorname{argmin}\{\|\mathbf{y}\| \quad | G(\mathbf{y})=0\}$$

where the function $G$ indicates the limit state function $g$ when
evaluated in the transformed space. This transformed space is
constructed using a Nataf transformation. At this point we find
ourselves in a quandary, as our limit state is defined in terms of
parameters $t_f$ and $b_f$ which our beam model in @lst:beam-no1 does
not accept. However, as luck would have it, our analysis of
@eq:dual-area might be repurposed as an expression for the area of a T
girder. @lst:anabel-expr illustrates how an expression for `A` may be
defined in terms of parameters `tf` and `bf`, and the instantiation of a
`beam_template` instance which is composed of this expressions (The
variable `I` is defined similarly. This has been omitted from the
listing for brevity but the complete model definition is included as an
appendix). In this study, the `anabel.backend` module, an interface to
the [JAX] numpy API, will take on the role of our minimal C++ forward
propagation library with additional support for arbitrary-order
differentiation and linear algebraic primitives.

``` {#lst:anabel-expr .python caption="Expression composition using `anabel`"}
import anabel

model = anabel.SkeletalModel(ndm=2, ndf=3)

bw, tw = 18, 18
tf, bf = model.param("tf","bf")
# define an expression for the area
area = lambda tf, bf: bf*tf + bw*tw
# create a model parameter from this expression
A  = model.expr(area, tf, bf)
...
# instantiate a `beam` in terms of this expression
girder = beam_template(A=A, I=I, E=3600.0)
```

Once the remainder of the model has been defined, the `compose` method
of the `anabel.SkeletalModel` class is used to build a function `F`
which takes the model parameters `tf` and `bf`, and returns the model's
displacement vector. A function which computes the gradient of the
function `F` can be generated using the automatic differentiation API of
the JAX library. This includes options for both forward and reverse mode
AD. Now equipped with a response and gradient function, the limit state
function can easily be defined for the FORM analysis. The FORM analysis
is carried out using the self-implemented library `aleatoire`, also
available on [PyPi.org][1]. The results of the reliability analysis are
displayed in @fig:reliability, where the probability of exceeding a
lateral displacement of $u_{max} = 0.2 \text{inches}$ is approximately
$3.6\%$.

![Design point obtained from first-order reliability analysis in the
physical and transformed spaces, respectively.]

### Cyclic Analysis

In the final leg of our journey, the basic beam element in @lst:beam-no1
is replaced with that in @lst:beam-no6. This template accepts a sequence
of functions $s: \mathbb{R}^2 \rightarrow \mathbb{R}^2$ representing
cross sectional response. The function `_B` in the closure of this
template is used to interpolate these section models according to an
imposed curvature distribution. Additionally, each function $s_i$ is
composed of material fiber elements which are integrated over the cross
section as depicted in @fig:tee-quad.

![Illustrative depiction of fiber section for T-shaped girder.
*Integration points enlarged to show detail*.]

``` {#lst:beam-no6 .python caption="Expression template for fiber frame element."}
@anabel.template(3)
def fiber_template(*sections, L=None, quad=None):
    state = [s.origin[2] for s in sections]
    params = {...: [anon.dual.get_unspecified_parameters(s) for s in sections]}
    locs, weights = quad_points(**quad)
    def _B(xi,L):
        x = L/2.0*(1.0+xi)
        return anp.array([[1/L,0.,0.], [0.0, (6*x/L-4)/L, (6*x/L-2)/L]])

    def main(v, q=None,state=state,L=L,params=params):
        B = [_B(xi,L) for xi in locs]
        S = [s(b@v, None, state=state[i], **params[...][i]) 
             for i,(b,s) in enumerate(zip(B, sections))]
        q = sum(L/2.0*w*b.T@s[1] for b,w,s in zip(B,weights,S))
        state = [s[2] for s in S]
        return v, q, state
    return locals()
```

The results of a simple cyclic load history analysis are depicted in
@fig:fiber-history.

![Cyclic response of portal frame with fiber girder.]

## Conclusion

An approach to finite element modeling was presented in which models are
composed from trees of expressions which map between well-defined sets
of inputs to outputs in a completely stateless manner. It was shown that
this statelessness admits arbitrary compositions while maintaining
mathematical properties/guarantees. These guarantees may be exploited
through JIT compilation to extract unexpected performance from otherwise
inherently slow programming languages/environments.

```{=html}
<!--
Additional examples illustrating the capabilities of the proposed protocol are listed below.

\pagebreak

```{=latex}
\begin{description}
  \item[\href{https://claudioperez.github.io/anabel/gallery/elle-0040}{1D Transient Poisson Equation}] \hfil \\ 
    \begin{minipage}[c]{0.35\linewidth} 
       \begin{flushleft}
    A study of the transient heat equation. The method of lines is used so that a solution in space is obtained using the finite element method, and time discretization is handled through an implicit Runge-Kutta solver accepting an arbitrary lower-diagonal Butcher tableau.
       \end{flushleft}
    \end{minipage}
    \begin{minipage}[c]{0.5\linewidth}
      \begin{flushright}
    %\adjustbox{valign=t}{%
        \includegraphics[width=0.9\linewidth,height=\textheight,keepaspectratio,valign=t]{img/pde-iso.pdf}
      \end{flushright}
    \end{minipage}
  \item[\href{https://claudioperez.github.io/anabel/gallery/elle-0050}{Poisson Equation in 2D}] \hfil \\ 
    \begin{minipage}[c]{0.35\linewidth} 
       \begin{flushleft}
    A parametric study of the 2D Poisson equation using both a classical sparse linear solve, and a conjugate gradient solve with algorithmic tangent. Element integration weights are parameterized across the domain and model assembly procedures are collected into a vectorized function. Highest resolution mesh includes over $12,000$ degrees of freedom and $13,000$ elements. 
       \end{flushleft}
    \end{minipage}
    \begin{minipage}[c]{0.5\linewidth}
      \begin{flushright}
    %\adjustbox{valign=t}{%
        \includegraphics[width=0.9\linewidth,height=\textheight,keepaspectratio,valign=t]{img/mesh4-gauss02.png}
      \end{flushright}
    \end{minipage}
\end{description}
```
-->
```
# Appendix

## Forward-mode AD Library for C++

``` {.cpp include="/home/claudio/pkgs/anabel/src/libelle/ad.cc"}
```



## Example Using Forward-mode AD in C++

``` {.cpp include="/home/claudio/pkgs/anabel/src/libelle/tee.cc"}
```



## Simple Portal Frame Composition

``` {.python include="/home/claudio/stdy/elle-0020/src/simple_portal.py"}
```

[^1]: Within machine precision

  [FEDEASLab]: FEDEASLab {citeprgm="filippou2004fedeaslab"}
  [OpenSees]: https://github.com/OpenSees/OpenSees
  {citeprgm="derkiureghian2006structural"}
  [PyPi.org]: https://pypi.org/project/anabel
  [anabel.Assembler]: https://claudioperez.github.io/anabel/api/assembler.html#Assembler
  [Basic frame]: img/frame.pdf {#fig:frame width="50%"}
  [Standard T-shaped girder cross section.]: img/tee-plain.pdf
  {#fig:tee-dims width="60%"}
  [JAX]: https://github.com/google/jax {citeprgm="frostig2018compiling"}
  [1]: PiPi.org/projects/aleatoire
  [Design point obtained from first-order reliability analysis in the physical and transformed spaces, respectively.]:
    img/reliability.pdf {#fig:reliability width="80%"}
  [Illustrative depiction of fiber section for T-shaped girder. *Integration points enlarged to show detail*.]:
    img/tee.pdf {#fig:tee-quad width="80%"}
  [Cyclic response of portal frame with fiber girder.]: img/fiber-cycle.pdf
  {#fig:fiber-history width="80%"}
