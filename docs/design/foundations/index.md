<h1 id="part-i-composition-of-expressions">Part I: Composition of Expressions</h1>
<!-- Recently the field of deep learning has found its way into every day life and it seems as though the sorcerers at Google. The work horses beneath these monumental strides in technology are available to the public through tools like TensorFlow, Matlab, and PyTorch which have become household names in the realm of scientific computing.  -->
<p>In <span class="citation" data-cites="frostig2018compiling">[@frostig2018compiling]</span>, researchers from the Google Brain team begin their presentation of the revolutionary JAX project with a simple observation; machine learning workloads tend to consist largely of subroutines satisfying two simple properties:</p>
<ol type="1">
<li>They are <em>pure</em> in the sense that their execution is free of side effects, and</li>
<li>They are <em>statically composed</em> in the sense that they implement a computation which can be defined as a static data dependency on a graph of some set of <em>primitive</em> operations (This will be further explained).</li>
</ol>
<p>The exploitation of these properties with techniques such as automatic differentiation and just-in-time compilation has become a central component of tools like TensorFlow, Matlab and PyTorch, which in the recent decade have been at the forefront of an “explosion” in the field of deep learning. The utility that these ground-breaking platforms provide is that they present researchers and developers with a high level platform for concisely staging mathematical computations with an intuitive “neural network” paradigm. By their structure, these resulting models admit analytic gradients which allows for the application of powerful optimization algorithms.</p>
<!-- An excellent example of such a machine learning model is the Python library [BRAILS] which provides users a means for building models which infer structural properties about buildings directly from images. Phrased differently, what such a library produces is a mapping $f: A \rightarrow B$ which will take a a set of pixels, $a$, and yield a classification $b \in B$. In order to build this function $f$, the BRAILS library creates a function $F: \Theta \rightarrow A \rightarrow B$ using primitives from the [TensorFlow] library (i.e., A function taking an element $\theta \in \Theta$) -->
<p>Algorithmic differentiation (AD) is a class of computational methods which leverage the chain rule in order to exactly evaluate<a href="#fn1" class="footnote-ref" id="fnref1" role="doc-noteref"><sup>1</sup></a> the gradient of a mathematical computation which is implemented as a computer program. These methods are often further categorized into <em>forward</em> and <em>reverse</em> modes, the later of which further specializes to the <em>backpropagation</em> class of algorithms which has become ubiquitous in the field of machine learning. These methods generally leverage either (1) source code transformation techniques or (2) operator overloading so that such a gradient computation can be deduced entirely from the natural semantics of the programming language being used. Much of the early literature which explores applications of AD to fields like structural modeling <span class="citation" data-cites="ozaki1995higherorder martins2001connection">[@ozaki1995higherorder; @martins2001connection]</span> employ the first approach, where prior to compilation, an pre-processing step is performed on some source code which generates new code implementing the derivative of an operation. This approach is generally used with primitive languages like FORTRAN and C which lack built-in abstractions like operator overloading and was likely selected for these works because at the time, such languages were the standard in the field. Although these compile-time methods allow for powerful optimizations to be used, they can add significant complexity to a project. Languages like C++, Python and Matlab allow user-defined types to <em>overload</em> built-in primitive operators like <code>+</code> and <code>*</code> so that when they are applied to certain data types they invoke a user-specified procedure. These features allow for AD implementations which are entirely contained within a programming language’s standard ecosystem.</p>
<h2 id="building-a-differentiable-model">Building a Differentiable Model</h2>
<p>There are several approaches which may be followed to arrive at the same formulation of forward mode automatic differentiation <span class="citation" data-cites="hoffmann2016hitchhiker">[@hoffmann2016hitchhiker]</span>. A particularly elegant entry point for the purpose of illustration is that used by <span class="citation" data-cites="pearlmutter2007lazy">[@pearlmutter2007lazy]</span> which begins by considering the algebra of <em>dual numbers</em>. These numbers are expressions of the form <span class="math inline">\(a + b\varepsilon\)</span>, <span class="math inline">\(a,b \quad \in \mathbb{R}\)</span> for which addition and multiplication is like that of complex numbers, save that the symbol <span class="math inline">\(\varepsilon\)</span> be defined to satisfy <span class="math inline">\(\varepsilon^2 = 0\)</span>.</p>
<p>Now consider an arbitrary analytic function <span class="math inline">\(f: \mathbb{R} \rightarrow \mathbb{R}\)</span>, admitting the following Taylor series expansion:</p>
<p><span><span class="math display">\[
f(a+b \varepsilon)=\sum_{n=0}^{\infty} \frac{f^{(n)}(a) b^{n} \varepsilon^{n}}{n !} 
\qquad(1)\]</span></span></p>
<p>By truncating the series at <span class="math inline">\(n = 1\)</span> and adhering to the operations defined over dual number arithmetic, one obtains the expression</p>
<p><span><span class="math display">\[
f(a + b \varepsilon) = f(a) + bf^\prime (a) \varepsilon, 
\qquad(2)\]</span></span></p>
<p>This result insinuates that by elevating such a function to perform on dual numbers, it has effectively been transformed into a new function which simultaneously evaluates both it’s original value, and it’s derivative <span class="math inline">\(f^\prime\)</span>.</p>
<p>For instance, consider the following pair of mappings:</p>
<p><span><span class="math display">\[
\begin{aligned}
f: &amp; \mathbb{R} \rightarrow \mathbb{R} \\
   &amp; x \mapsto (c - x)
\end{aligned}
\qquad(3)\]</span></span></p>
<p><span id="eq:dual-area"><span class="math display">\[
\begin{aligned}
g: &amp; \mathbb{R} \rightarrow \mathbb{R} \\
   &amp; x \mapsto a x + b f(x)
\end{aligned}
\qquad(4)\]</span></span></p>
<p>One can use the system of dual numbers to evaluate the gradient of <span class="math inline">\(g\)</span> at some point <span class="math inline">\(x \in \mathbb{R}\)</span>, by evaluating <span class="math inline">\(g\)</span> at the dual number <span class="math inline">\(x + \varepsilon\)</span>, where it is noted that the dual coefficient <span class="math inline">\(b = 1\)</span> is in fact equal to the derivative of the dependent variable <span class="math inline">\(x\)</span>. This procedure may be thought of as <em>propagating</em> a differential <span class="math inline">\(\varepsilon\)</span> throughout the computation.</p>
<p><span><span class="math display">\[
\begin{aligned}
g(x + \varepsilon)   &amp; = a (x + \varepsilon) + b f(x + \varepsilon ) \\
                     &amp; = a x + a \varepsilon + b ( c - x - \varepsilon) \\
                     &amp; = (a - b) x + bc + (a - b) \varepsilon 
\end{aligned}
\qquad(5)\]</span></span></p>
<p>The result of this expansion is a dual number whose real component is the value of <span class="math inline">\(g(x)\)</span>, and dual component, <span class="math inline">\(a-b\)</span>, equal to the gradient of <span class="math inline">\(g\)</span> evaluated at <span class="math inline">\(x\)</span>.</p>
<p>Such a system lends itself well to implementation in programming languages where an object might be defined to transparently act as a scalar. For example, in C, one might create a simple data structure with fields <code>real</code> and <code>dual</code> that store the real component, <span class="math inline">\(a\)</span> and <em>dual</em> component, <span class="math inline">\(b\)</span> of such a number as show in lst. 1.</p>
<div id="lst:struct-dual" class="listing cpp">
<p>Listing 1: Basic dual number data type.</p>
<div class="sourceCode" id="cb1"><pre class="sourceCode cpp"><code class="sourceCode cpp"><span id="cb1-1"><a href="#cb1-1" aria-hidden="true" tabindex="-1"></a><span class="kw">typedef</span> <span class="kw">struct</span> dual <span class="op">{</span></span>
<span id="cb1-2"><a href="#cb1-2" aria-hidden="true" tabindex="-1"></a>    <span class="dt">float</span> real<span class="op">,</span> dual<span class="op">;</span></span>
<span id="cb1-3"><a href="#cb1-3" aria-hidden="true" tabindex="-1"></a><span class="op">}</span> <span class="dt">dual_t</span><span class="op">;</span></span></code></pre></div>
</div>
<p>Then, using the operator overloading capabilities of C++, the application of the <code>*</code> operator can be overwritten as follows so that the property <span class="math inline">\(\varepsilon = 0\)</span> is enforced, as show in lst. 2.</p>
<div id="lst:dual-mult" class="listing cpp">
<p>Listing 2: Implementation of dual-dual multiplilcation in C++</p>
<div class="sourceCode" id="cb2"><pre class="sourceCode cpp"><code class="sourceCode cpp"><span id="cb2-1"><a href="#cb2-1" aria-hidden="true" tabindex="-1"></a><span class="dt">dual_t</span> <span class="dt">dual_t</span><span class="op">::</span><span class="kw">operator</span> <span class="op">*</span> <span class="op">(</span><span class="dt">dual_t</span> a<span class="op">,</span> <span class="dt">dual_t</span> b<span class="op">){</span></span>
<span id="cb2-2"><a href="#cb2-2" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> <span class="op">(</span><span class="dt">dual_t</span><span class="op">)</span> <span class="op">{</span></span>
<span id="cb2-3"><a href="#cb2-3" aria-hidden="true" tabindex="-1"></a>        <span class="op">.</span>real <span class="op">=</span> a<span class="op">.</span>real <span class="op">*</span> b<span class="op">.</span>real<span class="op">,</span></span>
<span id="cb2-4"><a href="#cb2-4" aria-hidden="true" tabindex="-1"></a>        <span class="op">.</span>dual <span class="op">=</span> a<span class="op">.</span>real <span class="op">*</span> b<span class="op">.</span>dual <span class="op">+</span> b<span class="op">.</span>real <span class="op">*</span> a<span class="op">.</span>dual</span>
<span id="cb2-5"><a href="#cb2-5" aria-hidden="true" tabindex="-1"></a>    <span class="op">};</span></span>
<span id="cb2-6"><a href="#cb2-6" aria-hidden="true" tabindex="-1"></a><span class="op">}</span></span></code></pre></div>
</div>
<p>Similar operations are defined for variations of the <code>*</code>, <code>+</code>, and <code>-</code> operators in a single-file library in the appendix. With this complete arithmetic interface implemented, forward propagation may be performed simply with statements like that of lst. 3.</p>
<div id="lst:dual-expr" class="listing cpp">
<p>Listing 3: Dual number arithmetic in C++ through operator overloading.</p>
<div class="sourceCode" id="cb3"><pre class="sourceCode cpp"><code class="sourceCode cpp"><span id="cb3-1"><a href="#cb3-1" aria-hidden="true" tabindex="-1"></a><span class="dt">int</span> main<span class="op">(</span><span class="dt">int</span> argc<span class="op">,</span> <span class="dt">char</span> <span class="op">**</span>argv<span class="op">){</span></span>
<span id="cb3-2"><a href="#cb3-2" aria-hidden="true" tabindex="-1"></a>  <span class="dt">dual_t</span> x<span class="op">(</span><span class="fl">6.0</span><span class="op">,</span><span class="fl">1.0</span><span class="op">);</span></span>
<span id="cb3-3"><a href="#cb3-3" aria-hidden="true" tabindex="-1"></a>  <span class="dt">dual_t</span> dg <span class="op">=</span> a <span class="op">*</span> x <span class="op">+</span> <span class="op">(</span>c <span class="op">-</span> x<span class="op">)</span> <span class="op">*</span> b<span class="op">;</span></span>
<span id="cb3-4"><a href="#cb3-4" aria-hidden="true" tabindex="-1"></a>  printd<span class="op">(</span>dA<span class="op">);</span></span>
<span id="cb3-5"><a href="#cb3-5" aria-hidden="true" tabindex="-1"></a><span class="op">}</span></span></code></pre></div>
</div>
<h2 id="template-expressions">Template Expressions</h2>
<p>The simple <code>dual_t</code> numeric type developed in the previous section allows both an expression, and its gradient to be evaluated simultaneously with only the minor penalty that each value stores an associated <code>dual</code> value. However, if such a penalty was imposed on every value carried in a modeling framework, both memory and computation demands would scale very poorly. In order to eliminate this burden, the concept of a <em>template expression</em> is introduced. This is again illustrated using the C++ programming language in order to maintain transparency to the corresponding machine instructions, but as expanded in the following part, this technique becomes extremely powerful when implemented in a context where access to the template expression in a processed form, such as an abstract syntax tree, can be readily manipulated. Examples of this might be through a higher-level language offering introspective capabilities (e.g., Python or most notably Lisp), or should one be so brave, via compiler extensions (an option which is becoming increasingly viable with recent developments and investment in projects like LLVM <span class="citation" data-cites="lattner2004llvm">[@lattner2004llvm]</span>).</p>
<p>In lst. 4, the function <span class="math inline">\(g\)</span> has been defined as a <em>template</em> on generic types <code>TA</code>, <code>TB</code>, <code>TC</code> and <code>TX</code>. In basic use cases, like that shown in lst. 4, a modern C++ compiler will deduce the types of the arguments passed in a call to <code>G</code> so that dual operations will only be used when arguments of type <code class="sourceCode c">dual_t</code> are passed. Furthermore, such an implementation will produce a derivative of <code>G</code> with respect to whichever parameter is of type <code class="sourceCode c">dual_t</code>. For example, in the assignment to variable <code>dg</code> in lst. 4, a <code class="sourceCode c">dual_t</code> value is created from variable <code>x</code> within the call to <code>G</code>, which automatically causes the compiler to create a function with signature <code class="sourceCode c">dual_t <span class="op">(*)(</span>dual_t<span class="op">,</span> real_t<span class="op">,</span> real_t<span class="op">,</span> real_t<span class="op">)</span></code> which propagates derivatives in <code>x</code>.</p>
<div id="lst:dual-template" class="listing cpp">
<p>Listing 4: Forward propagation using template metaprogramming</p>
<div class="sourceCode" id="cb4"><pre class="sourceCode cpp"><code class="sourceCode cpp"><span id="cb4-1"><a href="#cb4-1" aria-hidden="true" tabindex="-1"></a><span class="kw">template</span><span class="op">&lt;</span><span class="kw">typename</span> TX<span class="op">,</span><span class="kw">typename</span> TB<span class="op">,</span> <span class="kw">typename</span> TA<span class="op">,</span> <span class="kw">typename</span> TC<span class="op">&gt;</span></span>
<span id="cb4-2"><a href="#cb4-2" aria-hidden="true" tabindex="-1"></a><span class="kw">auto</span> G<span class="op">(</span>TX x<span class="op">,</span> TB b<span class="op">,</span> TA a<span class="op">,</span> TC c<span class="op">){</span></span>
<span id="cb4-3"><a href="#cb4-3" aria-hidden="true" tabindex="-1"></a>  printd<span class="op">(</span>b<span class="op">*</span>x<span class="op">);</span></span>
<span id="cb4-4"><a href="#cb4-4" aria-hidden="true" tabindex="-1"></a>  printd<span class="op">(</span>c <span class="op">-</span> x<span class="op">);</span></span>
<span id="cb4-5"><a href="#cb4-5" aria-hidden="true" tabindex="-1"></a>  <span class="cf">return</span> a <span class="op">*</span> x <span class="op">+</span> <span class="op">(</span>c <span class="op">-</span> x<span class="op">)</span> <span class="op">*</span>b<span class="op">;</span></span>
<span id="cb4-6"><a href="#cb4-6" aria-hidden="true" tabindex="-1"></a><span class="op">}</span></span>
<span id="cb4-7"><a href="#cb4-7" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb4-8"><a href="#cb4-8" aria-hidden="true" tabindex="-1"></a><span class="dt">int</span> main<span class="op">(</span><span class="dt">int</span> argc<span class="op">,</span> <span class="dt">char</span> <span class="op">**</span>argv<span class="op">){</span></span>
<span id="cb4-9"><a href="#cb4-9" aria-hidden="true" tabindex="-1"></a>  <span class="dt">real_t</span> a <span class="op">=</span> <span class="fl">60.0</span><span class="op">,</span> c<span class="op">=</span><span class="fl">18.0</span><span class="op">,</span> b<span class="op">=</span><span class="fl">18.0</span><span class="op">;</span></span>
<span id="cb4-10"><a href="#cb4-10" aria-hidden="true" tabindex="-1"></a>  <span class="dt">real_t</span> x <span class="op">=</span> <span class="fl">6.0</span><span class="op">;</span></span>
<span id="cb4-11"><a href="#cb4-11" aria-hidden="true" tabindex="-1"></a>  <span class="dt">real_t</span> g <span class="op">=</span> G<span class="op">(</span>x<span class="op">,</span>b<span class="op">,</span>a<span class="op">,</span>c<span class="op">);</span></span>
<span id="cb4-12"><a href="#cb4-12" aria-hidden="true" tabindex="-1"></a>  <span class="dt">dual_t</span> dg <span class="op">=</span> G<span class="op">(</span><span class="dt">dual_t</span><span class="op">(</span>x<span class="op">,</span><span class="dv">1</span><span class="op">),</span>b<span class="op">,</span>a<span class="op">,</span>c<span class="op">);</span></span>
<span id="cb4-13"><a href="#cb4-13" aria-hidden="true" tabindex="-1"></a>  printd<span class="op">(</span>dg<span class="op">);</span></span>
<span id="cb4-14"><a href="#cb4-14" aria-hidden="true" tabindex="-1"></a><span class="op">}</span></span></code></pre></div>
</div>
<p>In this minimal example, the set of operations for which we have explicitly defined corresponding dual operations constitute a set of <em>primitive</em> operations which may be composed arbitrarily in expressions which will implicitly propagate machine precision first derivatives. These compositions may include arbitrary use of control structures like <code class="sourceCode c"><span class="cf">while</span></code> / <code class="sourceCode c"><span class="cf">for</span></code> loops, branching via <code class="sourceCode c"><span class="cf">if</span></code> statements, and by composition of such operations, numerical schemes involving indefinite iteration.</p>
<p>A thorough treatment of AD for iterative procedures such as the use of the Newton-Raphson method for solving nonlinear equations is presented by <span class="citation" data-cites="beck1994automatic">[@beck1994automatic]</span>, <span class="citation" data-cites="gilbert1992automatic">[@gilbert1992automatic]</span> and <span class="citation" data-cites="griewank1993derivative">[@griewank1993derivative]</span>. Historic accounts of the development of AD can be found in <span class="citation" data-cites="iri2009automatic">[@iri2009automatic]</span>. Such treatments generally begin with the work of Wengert (e.g. <span class="citation" data-cites="wengert1964simple">[@wengert1964simple]</span>, ). <!-- Substantial contributions to the field have been made by A. Griewank [@griewank1991automatic; @griewank1989automatic]. --> A more recent work, <span class="citation" data-cites="elliott2009beautiful">[@elliott2009beautiful]</span> develops an elegant presentation and implementation of forward-mode AD in the context of purely functional programming.</p>
<h1 id="part-ii-the-finite-element-method">Part II: The Finite Element Method</h1>
<!-- ## Differentiable Modeling -->
<p>The finite element method has become the standard method for modeling PDEs in countless fields of research, and this is due in no small part to the natural modularity that arises in the equations it produces. Countless software applications and programming frameworks have been developed which manifest this modularity, each with a unique flavor.</p>
<!-- [DEALii] [MOOSE] -->
<p>A particularly powerful dialect of FE methods for problems in resilience is the composable approach developed through works like <span class="citation" data-cites="spacone1996fibre">[@spacone1996fibre]</span> where the classical beam element is generalized as a composition of arbitrary section models, which may in turn be comprised of several material models. Platforms like <a href="FEDEASLab" data-citeprgm="filippou2004fedeaslab">FEDEASLab</a> and <a href="https://github.com/OpenSees/OpenSees" data-citeprgm="derkiureghian2006structural">OpenSees</a> are designed from the ground up to natively support such compositions <span class="citation" data-cites="mckenna2010nonlinear">[@mckenna2010nonlinear]</span>. This study builds on this composable architecture by defining a single protocol for model components which can be used to nest or compose elements indefinitely.</p>
<p>In order to establish an analogous structure to that of ANNs which is sufficiently general to model geometric and material nonlinearities, we first assert that such a FE program will act as a mapping, <span class="math inline">\(\Phi: \Theta\times\mathcal{S}\times\mathcal{U}\rightarrow\mathcal{S}\times\mathcal{U}\)</span>, which walks a function, <span class="math inline">\(f: \Theta\times\mathcal{S}\times\mathcal{U}\rightarrow\mathcal{S}\times\mathcal{R}\)</span> (e.g. by applying the Newton-Raphson method), <!--and its derivative $D_U f: (\Theta, \mathcal{S},\mathcal{U})\rightarrow\mathcal{U}\rightarrow (\mathcal{S},\mathcal{R})$,--> until arriving at a solution, <span class="math inline">\((S^*,U^*)=\Phi(\Theta,S_0,U_0)\)</span> such that <span class="math inline">\(f(\Theta,S_0,U^*) = (S^*, \mathbf{0})\)</span> in the region of <span class="math inline">\(U_0\)</span>. In problems of mechanics, <span class="math inline">\(f\)</span> may be thought of as a potential energy gradient or equilibrium statement, and the elements of <span class="math inline">\(\mathcal{U}\)</span> typically represent a displacement vector, <span class="math inline">\(\mathcal{R}\)</span> a normed force residual, <span class="math inline">\(\mathcal{S}\)</span> a set of path-dependent state variables, and <span class="math inline">\(\Theta\)</span> a set of parameters which an analyst may wish to optimize (e.g. the locations of nodes, member properties, parameters, etc.). The function <span class="math inline">\(f\)</span> is composed from a set of <span class="math inline">\(i\)</span> local functions <span class="math inline">\(g_i: \theta\times s\times u \rightarrow s\times r\)</span> that model the response and dissipation of elements over regions of the <!--discretized--> domain, which in turn may be comprised of a substructure or material function, until ultimately resolving through elementary operations, <span class="math inline">\(h_j: \mathbb{R}^n\rightarrow\mathbb{R}^m\)</span> (e.g. Vector products, trigonometric functions, etc.).</p>
<p>In a mathematical sense, for a given solution routine, all of the information that is required to specify a computation which evaluates <span class="math inline">\(\Phi\)</span>, and some gradient, <span class="math inline">\(D_\Theta \Phi\)</span>, of this model with respect to locally well behaved parameters <span class="math inline">\(\Theta\)</span> is often defined entirely by the combination of (1) a set of element response functions <span class="math inline">\(g_i\)</span>, and (2) a graph-like data structure, indicating their connectivity (i.e., the static data dependency of <span class="citation" data-cites="frostig2018compiling">[@frostig2018compiling]</span>) . For such problems, within a well-structured framework, the procedures of deriving analytic gradient functions and handling low-level machine instructions can be lifted entirely from the analyst through abstraction, who should only be concerned with defining the trial functions, <span class="math inline">\(g_i\)</span>, and establishing their connectivity.</p>
<h2 id="the-anabel-library">The <code>anabel</code> Library</h2>
<p>An experimental library has been written in a combination of Python, C and C++ which provides functional abstractions that can be used to compose complex physical models. These abstractions isolate elements of a model which can be considered <em>pure and statically composed</em> (PSC), and provides an interface for their composition in a manner which allows for parameterized functions and gradients to be concisely composed. In this case, Python rarely serves as more than a configuration language, through which users simply annotate the entry and exit points of PSC operations. Once a complete operation is specified, it is compiled to a low-level representation which then may be executed entirely independent of the Python interpreter’s runtime.</p>
<p>Ideally such a library will be used to interface existing suites like FEDEASLab and OpenSees, but for the time being all experiments have been carried out with finite element models packaged under the <code>elle</code> namespace. Of particular interest, as expanded on in the conclusion of this report, would be developing a wrapper around the standard element API of OpenSees to interface with XLA, the back-end used by the <code>anabel</code> library. The <code>anabel</code> package is available on <a href="https://pypi.org/project/anabel">PyPi.org</a> and can be <code>pip</code> installed, as explained in the online documentation.</p>
<p>Functions in the <code>anabel</code> library operate on objects, or collections of objects, which implement an interface which can be represented as <span class="math inline">\((S^*,U^*)=\Phi(\Theta,S_0,U_0)\)</span>, and generally can be classified as pertaining to either (1) composition or (2) assembly related tasks. Composition operations involve those which create new objects from elementary objects in a manner analogous to mathematical composition. These operations include the [anabel.template] decorator for creating composable operations.</p>
<p>Assembly operations involve more conventional model-management utilities such as tracking and assigning DOFs, nodes and element properties. These are primarily handled by objects which subclass the <a href="https://claudioperez.github.io/anabel/api/assembler.html#Assembler">anabel.Assembler</a> through model building methods. These operations are generally required for the creation of OpenSees-styled models where, for example, a model may contain several types of frame elements, each configured using different section, material, and integration rules.</p>
<h2 id="examples">Examples</h2>
<p>Two examples are presented which illustrate the composition of a physical model from a collection of PSC routines. In these examples, we consider a function, <span class="math inline">\(F: \mathbb{R}^9 \rightarrow \mathbb{R}^9\)</span>, that is the finite element model representation of the structure shown in fig. 1. For illustrative purposes, all model components are defined only by a single direct mapping, and properties such as stiffness matrices are implicitly extracted using forward mode AD. The example <span class="citation" data-cites="elle-0050">[@elle-0050]</span> in the online documentation illustrates how components may define their own stiffness matrix operation, increasing model computational efficiency.</p>
<figure>
<img src="img/frame.svg" id="fig:frame" style="margin:auto; display: block; max-width: 75%" style="width:50.0%" alt="Figure 1: Basic frame" /><figcaption aria-hidden="true">Figure 1: Basic frame</figcaption>
</figure>
<p>Columns measure <span class="math inline">\(30 \times 30\)</span> inches and the girder is cast-in-place monolithically with a slab as show in fig. 2. The dimensions <span class="math inline">\(b_w\)</span> and <span class="math inline">\(d - t_f\)</span> are known with reasonable certainty to both measure <span class="math inline">\(18\)</span> inches, but due to shear lag in the flange and imperfections in the finishing of the slab surface, the dimensions <span class="math inline">\(b_f\)</span> and <span class="math inline">\(t_f\)</span> are only known to linger in the vicinity of <span class="math inline">\(60\)</span> inches and <span class="math inline">\(6\)</span> inches, respectively. Reinforcement is neglected and gross section properties are used for all members. Concrete with a <span class="math inline">\(28\)</span>-day compressive strength of <span class="math inline">\(f^\prime_c=4\)</span> ksi is used, and the elastic modulus is taken at <span class="math inline">\(3600\)</span>.</p>
<figure>
<img src="img/tee-plain.svg" id="fig:tee-dims" style="margin:auto; display: block; max-width: 75%" style="width:60.0%" alt="Figure 2: Standard T-shaped girder cross section." /><figcaption aria-hidden="true">Figure 2: Standard T-shaped girder cross section.</figcaption>
</figure>
<p>We are equipped with the Python function in lst. 5 which contains in its closure a definition of the function <code>beam</code> implementing a standard linear 2D beam as a map from a set of displacements, <span class="math inline">\(v\)</span>, to local forces <span class="math inline">\(q\)</span>.</p>
<div id="lst:beam-no1" class="listing python">
<p>Listing 5: Classical beam element implementation.</p>
<div class="sourceCode" id="cb5"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb5-1"><a href="#cb5-1" aria-hidden="true" tabindex="-1"></a><span class="at">@anabel.template</span>(<span class="dv">3</span>)</span>
<span id="cb5-2"><a href="#cb5-2" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> beam2d_template(</span>
<span id="cb5-3"><a href="#cb5-3" aria-hidden="true" tabindex="-1"></a>    q0: array <span class="op">=</span> <span class="va">None</span>,</span>
<span id="cb5-4"><a href="#cb5-4" aria-hidden="true" tabindex="-1"></a>    E:  <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span>,</span>
<span id="cb5-5"><a href="#cb5-5" aria-hidden="true" tabindex="-1"></a>    A:  <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span>,</span>
<span id="cb5-6"><a href="#cb5-6" aria-hidden="true" tabindex="-1"></a>    I:  <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span>,</span>
<span id="cb5-7"><a href="#cb5-7" aria-hidden="true" tabindex="-1"></a>    L:  <span class="bu">float</span> <span class="op">=</span> <span class="va">None</span></span>
<span id="cb5-8"><a href="#cb5-8" aria-hidden="true" tabindex="-1"></a>):</span>
<span id="cb5-9"><a href="#cb5-9" aria-hidden="true" tabindex="-1"></a>    <span class="kw">def</span> beam(v, q, state<span class="op">=</span><span class="va">None</span>, E<span class="op">=</span>E,A<span class="op">=</span>A,I<span class="op">=</span>I,L<span class="op">=</span>L):</span>
<span id="cb5-10"><a href="#cb5-10" aria-hidden="true" tabindex="-1"></a>        C0 <span class="op">=</span> E<span class="op">*</span>I<span class="op">/</span>L</span>
<span id="cb5-11"><a href="#cb5-11" aria-hidden="true" tabindex="-1"></a>        C1 <span class="op">=</span> <span class="fl">4.0</span><span class="op">*</span>C0</span>
<span id="cb5-12"><a href="#cb5-12" aria-hidden="true" tabindex="-1"></a>        C2 <span class="op">=</span> <span class="fl">2.0</span><span class="op">*</span>C0</span>
<span id="cb5-13"><a href="#cb5-13" aria-hidden="true" tabindex="-1"></a>        k <span class="op">=</span> anp.array([[E<span class="op">*</span>A<span class="op">/</span>L,<span class="fl">0.0</span>,<span class="fl">0.0</span>],[<span class="fl">0.0</span>,C1,C2],[<span class="fl">0.0</span>,C2,C1]])</span>
<span id="cb5-14"><a href="#cb5-14" aria-hidden="true" tabindex="-1"></a>        <span class="cf">return</span> v, k<span class="op">@</span>v, state</span>
<span id="cb5-15"><a href="#cb5-15" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> <span class="bu">locals</span>()</span></code></pre></div>
</div>
<h3 id="reliability-analysis">Reliability Analysis</h3>
<p>Our first order of business is to quantify the uncertainty surrounding the dimensions <span class="math inline">\(t_f\)</span> and <span class="math inline">\(b_f\)</span>. We seek to approximate the probability that a specified displacement, <span class="math inline">\(u_{max}\)</span>, will be exceeded assuming the dimensions <span class="math inline">\(t_f\)</span> and <span class="math inline">\(b_f\)</span> are distributed as given in tbl. 1 and a correlation coefficient of <span class="math inline">\(0.5\)</span> through a first-order reliability (FORM) analysis. This criteria is quantified by the limit state function eq. 6</p>
<p><span id="eq:limit-state"><span class="math display">\[
g(t_f, b_f) = u_{max} - \Phi(t_f, b_f) 
\qquad(6)\]</span></span></p>
<div id="tbl:rvs">
<table>
<caption>Table 1: Distributions of random variables.</caption>
<thead>
<tr class="header">
<th style="text-align: center;">Variable</th>
<th style="text-align: center;">Marginal Distribution</th>
<th style="text-align: center;">Mean</th>
<th style="text-align: center;">Standard Deviation</th>
</tr>
</thead>
<tbody>
<tr class="odd">
<td style="text-align: center;"><span class="math inline">\(t_f\)</span></td>
<td style="text-align: center;">Lognormal</td>
<td style="text-align: center;"><span class="math inline">\(6\)</span></td>
<td style="text-align: center;"><span class="math inline">\(0.6\)</span></td>
</tr>
<tr class="even">
<td style="text-align: center;"><span class="math inline">\(b_f\)</span></td>
<td style="text-align: center;">Lognormal</td>
<td style="text-align: center;"><span class="math inline">\(60\)</span></td>
<td style="text-align: center;"><span class="math inline">\(6\)</span></td>
</tr>
</tbody>
</table>
</div>
<p>These probability distributions define a parameter space for the problem, the limit state function, <span class="math inline">\(g\)</span>, defines a <em>failure domain</em>, and our objective is to essentially count how many elements in our space of parameters will map into the failure domain. To this end, the limit state function <span class="math inline">\(g\)</span> is approximated as a hyperplane tangent to the limit surface at the so-called design point, <span class="math inline">\(\mathbf{y}^{*}\)</span>, by first-order truncation of the Taylor-series expansion. This design point is the solution of the following constrained minimization problem in a standard normal space, which if found at a global extrema, ensures that the linearized limit-state function is that with the largest possible failure domain (in the transformed space):</p>
<p><span><span class="math display">\[\mathbf{y}^{*}=\operatorname{argmin}\{\|\mathbf{y}\| \quad | G(\mathbf{y})=0\}\qquad(7)\]</span></span></p>
<p>where the function <span class="math inline">\(G\)</span> indicates the limit state function <span class="math inline">\(g\)</span> when evaluated in the transformed space. This transformed space is constructed using a Nataf transformation. At this point we find ourselves in a quandary, as our limit state is defined in terms of parameters <span class="math inline">\(t_f\)</span> and <span class="math inline">\(b_f\)</span> which our beam model in lst. 5 does not accept. However, as luck would have it, our analysis of eq. 4 might be repurposed as an expression for the area of a T girder. lst. 6 illustrates how an expression for <code>A</code> may be defined in terms of parameters <code>tf</code> and <code>bf</code>, and the instantiation of a <code>beam_template</code> instance which is composed of this expressions (The variable <code>I</code> is defined similarly. This has been omitted from the listing for brevity but the complete model definition is included as an appendix). In this study, the <code>anabel.backend</code> module, an interface to the <a href="https://github.com/google/jax" data-citeprgm="frostig2018compiling">JAX</a> numpy API, will take on the role of our minimal C++ forward propagation library with additional support for arbitrary-order differentiation and linear algebraic primitives.</p>
<div id="lst:anabel-expr" class="listing python">
<p>Listing 6: Expression composition using `anabel`</p>
<div class="sourceCode" id="cb6"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb6-1"><a href="#cb6-1" aria-hidden="true" tabindex="-1"></a><span class="im">import</span> anabel</span>
<span id="cb6-2"><a href="#cb6-2" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb6-3"><a href="#cb6-3" aria-hidden="true" tabindex="-1"></a>model <span class="op">=</span> anabel.SkeletalModel(ndm<span class="op">=</span><span class="dv">2</span>, ndf<span class="op">=</span><span class="dv">3</span>)</span>
<span id="cb6-4"><a href="#cb6-4" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb6-5"><a href="#cb6-5" aria-hidden="true" tabindex="-1"></a>bw, tw <span class="op">=</span> <span class="dv">18</span>, <span class="dv">18</span></span>
<span id="cb6-6"><a href="#cb6-6" aria-hidden="true" tabindex="-1"></a>tf, bf <span class="op">=</span> model.param(<span class="st">&quot;tf&quot;</span>,<span class="st">&quot;bf&quot;</span>)</span>
<span id="cb6-7"><a href="#cb6-7" aria-hidden="true" tabindex="-1"></a><span class="co"># define an expression for the area</span></span>
<span id="cb6-8"><a href="#cb6-8" aria-hidden="true" tabindex="-1"></a>area <span class="op">=</span> <span class="kw">lambda</span> tf, bf: bf<span class="op">*</span>tf <span class="op">+</span> bw<span class="op">*</span>tw</span>
<span id="cb6-9"><a href="#cb6-9" aria-hidden="true" tabindex="-1"></a><span class="co"># create a model parameter from this expression</span></span>
<span id="cb6-10"><a href="#cb6-10" aria-hidden="true" tabindex="-1"></a>A  <span class="op">=</span> model.expr(area, tf, bf)</span>
<span id="cb6-11"><a href="#cb6-11" aria-hidden="true" tabindex="-1"></a>...</span>
<span id="cb6-12"><a href="#cb6-12" aria-hidden="true" tabindex="-1"></a><span class="co"># instantiate a `beam` in terms of this expression</span></span>
<span id="cb6-13"><a href="#cb6-13" aria-hidden="true" tabindex="-1"></a>girder <span class="op">=</span> beam_template(A<span class="op">=</span>A, I<span class="op">=</span>I, E<span class="op">=</span><span class="fl">3600.0</span>)</span></code></pre></div>
</div>
<figure>
<img src="img/area.svg" id="fig:area-graph" style="margin:auto; display: block; max-width: 75%" alt="Figure 3: Computational graph for the area of a T-section." /><figcaption aria-hidden="true">Figure 3: Computational graph for the area of a T-section.</figcaption>
</figure>
<p>Once the remainder of the model has been defined, the <code>compose</code> method of the <code>anabel.SkeletalModel</code> class is used to build a function <code>F</code> which takes the model parameters <code>tf</code> and <code>bf</code>, and returns the model’s displacement vector. A function which computes the gradient of the function <code>F</code> can be generated using the automatic differentiation API of the JAX library. This includes options for both forward and reverse mode AD. Now equipped with a response and gradient function, the limit state function can easily be defined for the FORM analysis. The FORM analysis is carried out using the self-implemented library <code>aleatoire</code>, also available on <a href="PiPi.org/projects/aleatoire">PyPi.org</a>. The results of the reliability analysis are displayed in fig. 4, where the probability of exceeding a lateral displacement of <span class="math inline">\(u_{max} = 0.2 \text{inches}\)</span> is approximately <span class="math inline">\(3.6\%\)</span>.</p>
<figure>
<img src="img/reliability.svg" id="fig:reliability" style="margin:auto; display: block; max-width: 75%" style="width:80.0%" alt="Figure 4: Design point obtained from first-order reliability analysis in the physical and transformed spaces, respectively." /><figcaption aria-hidden="true">Figure 4: Design point obtained from first-order reliability analysis in the physical and transformed spaces, respectively.</figcaption>
</figure>
<h3 id="cyclic-analysis">Cyclic Analysis</h3>
<p>In the final leg of our journey, the basic beam element in lst. 5 is replaced with that in lst. 7. This template accepts a sequence of functions <span class="math inline">\(s: \mathbb{R}^2 \rightarrow \mathbb{R}^2\)</span> representing cross sectional response. The function <code>_B</code> in the closure of this template is used to interpolate these section models according to an imposed curvature distribution. Additionally, each function <span class="math inline">\(s_i\)</span> is composed of material fiber elements which are integrated over the cross section as depicted in fig. 5.</p>
<figure>
<img src="img/tee.svg" id="fig:tee-quad" style="margin:auto; display: block; max-width: 75%" style="width:80.0%" alt="Figure 5: Illustrative depiction of fiber section for T-shaped girder. Integration points enlarged to show detail." /><figcaption aria-hidden="true">Figure 5: Illustrative depiction of fiber section for T-shaped girder. <em>Integration points enlarged to show detail</em>.</figcaption>
</figure>
<div id="lst:beam-no6" class="listing python">
<p>Listing 7: Expression template for fiber frame element.</p>
<div class="sourceCode" id="cb7"><pre class="sourceCode python"><code class="sourceCode python"><span id="cb7-1"><a href="#cb7-1" aria-hidden="true" tabindex="-1"></a><span class="at">@anabel.template</span>(<span class="dv">3</span>)</span>
<span id="cb7-2"><a href="#cb7-2" aria-hidden="true" tabindex="-1"></a><span class="kw">def</span> fiber_template(<span class="op">*</span>sections, L<span class="op">=</span><span class="va">None</span>, quad<span class="op">=</span><span class="va">None</span>):</span>
<span id="cb7-3"><a href="#cb7-3" aria-hidden="true" tabindex="-1"></a>    state <span class="op">=</span> [s.origin[<span class="dv">2</span>] <span class="cf">for</span> s <span class="kw">in</span> sections]</span>
<span id="cb7-4"><a href="#cb7-4" aria-hidden="true" tabindex="-1"></a>    params <span class="op">=</span> {...: [anon.dual.get_unspecified_parameters(s) <span class="cf">for</span> s <span class="kw">in</span> sections]}</span>
<span id="cb7-5"><a href="#cb7-5" aria-hidden="true" tabindex="-1"></a>    locs, weights <span class="op">=</span> quad_points(<span class="op">**</span>quad)</span>
<span id="cb7-6"><a href="#cb7-6" aria-hidden="true" tabindex="-1"></a>    <span class="kw">def</span> _B(xi,L):</span>
<span id="cb7-7"><a href="#cb7-7" aria-hidden="true" tabindex="-1"></a>        x <span class="op">=</span> L<span class="op">/</span><span class="fl">2.0</span><span class="op">*</span>(<span class="fl">1.0</span><span class="op">+</span>xi)</span>
<span id="cb7-8"><a href="#cb7-8" aria-hidden="true" tabindex="-1"></a>        <span class="cf">return</span> anp.array([[<span class="dv">1</span><span class="op">/</span>L,<span class="fl">0.</span>,<span class="fl">0.</span>], [<span class="fl">0.0</span>, (<span class="dv">6</span><span class="op">*</span>x<span class="op">/</span>L<span class="op">-</span><span class="dv">4</span>)<span class="op">/</span>L, (<span class="dv">6</span><span class="op">*</span>x<span class="op">/</span>L<span class="op">-</span><span class="dv">2</span>)<span class="op">/</span>L]])</span>
<span id="cb7-9"><a href="#cb7-9" aria-hidden="true" tabindex="-1"></a></span>
<span id="cb7-10"><a href="#cb7-10" aria-hidden="true" tabindex="-1"></a>    <span class="kw">def</span> main(v, q<span class="op">=</span><span class="va">None</span>,state<span class="op">=</span>state,L<span class="op">=</span>L,params<span class="op">=</span>params):</span>
<span id="cb7-11"><a href="#cb7-11" aria-hidden="true" tabindex="-1"></a>        B <span class="op">=</span> [_B(xi,L) <span class="cf">for</span> xi <span class="kw">in</span> locs]</span>
<span id="cb7-12"><a href="#cb7-12" aria-hidden="true" tabindex="-1"></a>        S <span class="op">=</span> [s(b<span class="op">@</span>v, <span class="va">None</span>, state<span class="op">=</span>state[i], <span class="op">**</span>params[...][i]) </span>
<span id="cb7-13"><a href="#cb7-13" aria-hidden="true" tabindex="-1"></a>             <span class="cf">for</span> i,(b,s) <span class="kw">in</span> <span class="bu">enumerate</span>(<span class="bu">zip</span>(B, sections))]</span>
<span id="cb7-14"><a href="#cb7-14" aria-hidden="true" tabindex="-1"></a>        q <span class="op">=</span> <span class="bu">sum</span>(L<span class="op">/</span><span class="fl">2.0</span><span class="op">*</span>w<span class="op">*</span>b.T<span class="op">@</span>s[<span class="dv">1</span>] <span class="cf">for</span> b,w,s <span class="kw">in</span> <span class="bu">zip</span>(B,weights,S))</span>
<span id="cb7-15"><a href="#cb7-15" aria-hidden="true" tabindex="-1"></a>        state <span class="op">=</span> [s[<span class="dv">2</span>] <span class="cf">for</span> s <span class="kw">in</span> S]</span>
<span id="cb7-16"><a href="#cb7-16" aria-hidden="true" tabindex="-1"></a>        <span class="cf">return</span> v, q, state</span>
<span id="cb7-17"><a href="#cb7-17" aria-hidden="true" tabindex="-1"></a>    <span class="cf">return</span> <span class="bu">locals</span>()</span></code></pre></div>
</div>
<p>The results of a simple cyclic load history analysis are depicted in fig. 6.</p>
<figure>
<img src="img/fiber-cycle.svg" id="fig:fiber-history" style="margin:auto; display: block; max-width: 75%" style="width:80.0%" alt="Figure 6: Cyclic response of portal frame with fiber girder." /><figcaption aria-hidden="true">Figure 6: Cyclic response of portal frame with fiber girder.</figcaption>
</figure>
<h2 id="conclusion">Conclusion</h2>
<p>An approach to finite element modeling was presented in which models are composed from trees of expressions which map between well-defined sets of inputs to outputs in a completely stateless manner. It was shown that this statelessness admits arbitrary compositions while maintaining mathematical properties/guarantees. These guarantees may be exploited through JIT compilation to extract unexpected performance from otherwise inherently slow programming languages/environments.</p>
<h1 id="appendix">Appendix</h1>
<h2 id="forward-mode-ad-library-for-c">Forward-mode AD Library for C++</h2>
<div class="sourceCode" id="cb8" data-include="/home/claudio/pkgs/anabel/src/libelle/ad.cc"><pre class="sourceCode cpp"><code class="sourceCode cpp"></code></pre></div>
<h2 id="example-using-forward-mode-ad-in-c">Example Using Forward-mode AD in C++</h2>
<div class="sourceCode" id="cb9" data-include="/home/claudio/pkgs/anabel/src/libelle/tee.cc"><pre class="sourceCode cpp"><code class="sourceCode cpp"></code></pre></div>
<h2 id="simple-portal-frame-composition">Simple Portal Frame Composition</h2>
<div class="sourceCode" id="cb10" data-include="/home/claudio/stdy/elle-0020/src/simple_portal.py"><pre class="sourceCode python"><code class="sourceCode python"></code></pre></div>
<div id="refs" class="references csl-bib-body" role="doc-bibliography">
<div id="ref-frostig2018compiling" class="csl-entry" role="doc-biblioentry">
1. Frostig, R., Johnson, M. J., and Leary, C. <span>“<strong>Compiling Machine Learning Programs via High-Level Tracing</strong>”</span> (2018): Available at <a href="https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf">https://cs.stanford.edu/~rfrostig/pubs/jax-mlsys2018.pdf</a>
</div>
<div id="ref-ozaki1995higherorder" class="csl-entry" role="doc-biblioentry">
2. Ozaki, I., Kimura, F., and Berz, M. <span>“<strong>Higher-Order Sensitivity Analysis of Finite Element Method by Automatic Differentiation</strong>”</span> (1995): 223–234. doi:<a href="https://doi.org/cqq274">cqq274</a>
</div>
<div id="ref-martins2001connection" class="csl-entry" role="doc-biblioentry">
3. Martins, J., Sturdza, P., and Alonso, J. <span>“<strong>The Connection Between the Complex-Step Derivative Approximation and Algorithmic Differentiation</strong>”</span> (2001): doi:<a href="https://doi.org/10.2514/6.2001-921">10.2514/6.2001-921</a>
</div>
<div id="ref-hoffmann2016hitchhiker" class="csl-entry" role="doc-biblioentry">
4. Hoffmann, P. H. W. <span>“<strong>A <span>Hitchhiker</span>’s <span>Guide</span> to <span>Automatic Differentiation</span></strong>”</span> (2016): 775–811. doi:<a href="https://doi.org/ggsd72">ggsd72</a>, Available at <a href="https://arxiv.org/abs/1411.0583">https://arxiv.org/abs/1411.0583</a>
</div>
<div id="ref-pearlmutter2007lazy" class="csl-entry" role="doc-biblioentry">
5. Pearlmutter, B. A. and Siskind, J. M. <span>“<strong>Lazy Multivariate Higher-Order Forward-Mode <span>AD</span></strong>”</span> (2007): 155. doi:<a href="https://doi.org/bpjm8z">bpjm8z</a>
</div>
<div id="ref-lattner2004llvm" class="csl-entry" role="doc-biblioentry">
6. Lattner, C. and Adve, V. <span>“<strong><span>LLVM</span>: A Compilation Framework for Lifelong Program Analysis Transformation</strong>”</span> (2004): 75–86. doi:<a href="https://doi.org/d5brsd">d5brsd</a>
</div>
<div id="ref-beck1994automatic" class="csl-entry" role="doc-biblioentry">
7. Beck, T. <span>“<strong>Automatic Differentiation of Iterative Processes</strong>”</span> (1994): 109–118. doi:<a href="https://doi.org/10.1016/0377-0427(94)90293-3">10.1016/0377-0427(94)90293-3</a>
</div>
<div id="ref-gilbert1992automatic" class="csl-entry" role="doc-biblioentry">
8. Gilbert, J. C. <span>“<strong>Automatic Differentiation and Iterative Processes</strong>”</span> (1992): 13–21. doi:<a href="https://doi.org/10.1080/10556789208805503">10.1080/10556789208805503</a>
</div>
<div id="ref-griewank1993derivative" class="csl-entry" role="doc-biblioentry">
9. Griewank, A., Bischof, C., Corliss, G., Carle, A., and Williamson, K. <span>“<strong>Derivative Convergence for Iterative Equation Solvers</strong>”</span> (1993): 321–355. doi:<a href="https://doi.org/10.1080/10556789308805549">10.1080/10556789308805549</a>
</div>
<div id="ref-iri2009automatic" class="csl-entry" role="doc-biblioentry">
10. Iri, M. and Kubota, K. <span>“<strong>Automatic Differentiation: Introduction, History and Rounding Error <span class="nocase">estimationAutomatic Differentiation</span>: <span>Introduction</span>, <span>History</span> and <span>Rounding Error Estimation</span></strong>”</span> (2009): 153–159. doi:<a href="https://doi.org/10.1007/978-0-387-74759-0_26">10.1007/978-0-387-74759-0_26</a>
</div>
<div id="ref-wengert1964simple" class="csl-entry" role="doc-biblioentry">
11. Wengert, R. E. <span>“<strong>A Simple Automatic Derivative Evaluation Program</strong>”</span> (1964): 463–464. doi:<a href="https://doi.org/bwkd4g">bwkd4g</a>
</div>
<div id="ref-elliott2009beautiful" class="csl-entry" role="doc-biblioentry">
12. Elliott, C. M. <span>“<strong>Beautiful <span>Differentiation</span></strong>”</span> (2009): 12. doi:<a href="https://doi.org/10.1145/1631687.1596579">10.1145/1631687.1596579</a>
</div>
<div id="ref-spacone1996fibre" class="csl-entry" role="doc-biblioentry">
13. Spacone, E. and Filippou, F. C. <span>“<strong>Fibre Beam–Column Model for Non‐linear Analysis of <span>R</span>/<span>C</span> Frames: <span>Part I</span>. <span>Formulation</span></strong>”</span> (1996): 15.
</div>
<div id="ref-mckenna2010nonlinear" class="csl-entry" role="doc-biblioentry">
14. McKenna, F., Scott, M. H., and Fenves, G. L. <span>“<strong>Nonlinear <span>Finite</span>-<span>Element Analysis Software Architecture Using Object Composition</span></strong>”</span> (2010): 95–107. doi:<a href="https://doi.org/10.1061/(ASCE)CP.1943-5487.0000002">10.1061/(ASCE)CP.1943-5487.0000002</a>
</div>
<div id="ref-elle-0050" class="csl-entry" role="doc-biblioentry">
15. Perez, C. <span>“<strong>Parameterized Finite Element Analysis of a <span>Poisson</span> Problem in <span>2d</span></strong>”</span> (2021): Available at <a href="https://claudioperez.github.io/anabel/gallery/elle-0050/">https://claudioperez.github.io/anabel/gallery/elle-0050/</a>
</div>
</div>
<section class="footnotes" role="doc-endnotes">
<hr />
<ol>
<li id="fn1" role="doc-endnote"><p>Within machine precision<a href="#fnref1" class="footnote-back" role="doc-backlink">↩︎</a></p></li>
</ol>
</section>
