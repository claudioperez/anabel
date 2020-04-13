## Linear Algebra

#### Matrix Inverse

https://math.stackexchange.com/questions/805273/matrix-inverse-in-tensor-notation/2768414

Using a basis, you could use the common method of calculation given by
$$A^{-1} = \frac{\text{adj}(A)}{\det(A)} $$
where $\det$ is the determinant, and $\text{adj}$ is the adjugate, i.e. the transpose of the matrix of cofactors $\text{cof}(A)$.

One can show that
$$ {(\text{adj}(A))^{i}}_{j} = \frac{\partial \det(A)}{\partial {A^{j}}_{i}}. $$
In order to calculate this, we can use the expression for the determinant in terms of the generalized Kronecker delta
$$\det(A) = \frac{1}{n!}\delta^{a_1\dots a_n}_{b_1\dots b_n}{A^{b_1}}_{a_1}\cdots{A^{b_n}}_{a_n}$$
where $n$ is the dimension of the vector space.

Hence the adjugate is

$$\begin{aligned}
{(\text{adj}(A))^{a}}_{b}
&= \frac{\partial}{\partial {A^{b}}_{a}} \left( \frac{1}{n!}\delta^{c_1\dots c_n}_{d_1\dots d_n}{A^{d_1}}_{c_1}\cdots{A^{d_n}}_{c_n} \right)\\
&= \frac{1}{n!} \delta^{c_1\dots c_n}_{d_1\dots d_n} \frac{\partial}{\partial {A^{b}}_{a}} \left({A^{d_1}}_{c_1}\cdots{A^{d_n}}_{c_n} \right)
\end{aligned}$$

If we perform the calculation (for instance, in $n=3$) we get
$$\begin{aligned}
{(\text{adj}(A))^{a}}_{b}
&= \frac{1}{3!} \delta^{ijk}_{lmn}
\frac{\partial}{\partial {A^{b}}_{a}} \left( {A^{l}}_{i}{A^{m}}_{j}{A^{n}}_{k} \right)\\
&= \frac{1}{3!} \delta^{ijk}_{lmn}
\left(
\delta^{l}_{b}\delta^{a}_{i}{A^{m}}_{j}{A^{n}}_{k} +
\delta^{m}_{b}\delta^{a}_{j}{A^{l}}_{i}{A^{n}}_{k} +
\delta^{n}_{b}\delta^{a}_{k}{A^{l}}_{i}{A^{m}}_{j}
\right)\\
&= \frac{1}{3!}
\left(
\delta^{ijk}_{lmn} \delta^{l}_{b}\delta^{a}_{i}{A^{m}}_{j}{A^{n}}_{k} +
\delta^{ijk}_{lmn} \delta^{m}_{b}\delta^{a}_{j}{A^{l}}_{i}{A^{n}}_{k} +
\delta^{ijk}_{lmn} \delta^{n}_{b}\delta^{a}_{k}{A^{l}}_{i}{A^{m}}_{j}
\right)\\
&= \frac{1}{3!}
\left(
\delta^{ajk}_{bmn} {A^{m}}_{j}{A^{n}}_{k} +
\delta^{iak}_{lbn} {A^{l}}_{i}{A^{n}}_{k} +
\delta^{ija}_{lmb} {A^{l}}_{i}{A^{m}}_{j}
\right)\\
&= \frac{1}{2!}\delta^{aij}_{bmn} {A^{m}}_{i}{A^{n}}_{j}
\end{aligned}$$

hence the inverse (in the $n=3$ case) is

$${(A^{-1})^{a}}_{b} = \frac{{(\text{adj}(A))^{a}}_{b}}{\det(A)}
= \frac{\frac{1}{2!}\delta^{aij}_{bmn} {A^{m}}_{i}{A^{n}}_{j}}
{\frac{1}{3!} \delta^{cde}_{fgh}{A^{f}}_{c}{A^{g}}_{d}{A^{h}}_{e} }
= 3\frac{\delta^{aij}_{bmn} {A^{m}}_{i}{A^{n}}_{j}}
{\delta^{cde}_{fgh}{A^{f}}_{c}{A^{g}}_{d}{A^{h}}_{e} }$$

You can check that the $n$-dimensional adjugate is
$${(\text{adj}(A))^{a}}_{b} = \frac{1}{(n-1)!}\delta^{ac_1\dots c_{n-1}}_{bd_1\dots d_{n-1}} {A^{d_1}}_{c_1}\cdots{A^{d_{n-1}}}_{c_{n-1}}$$

and finally we get the $n$-dimensional inverse
$${(A^{-1})^{a}}_{b}
= \frac{{(\text{adj}(A))^{a}}_{b}}{\det(A)}
= \frac{\frac{1}{(n-1)!}\delta^{ac_1\dots c_{n-1}}_{bd_1\dots d_{n-1}} {A^{d_1}}_{c_1}\cdots{A^{d_{n-1}}}_{c_{n-1}}}
{\frac{1}{n!}\delta^{e_1\dots e_n}_{f_1\dots f_n}{A^{f_1}}_{e_1}\cdots{A^{f_n}}_{e_n}}
= n\frac{\delta^{ac_1\dots c_{n-1}}_{bd_1\dots d_{n-1}} {A^{d_1}}_{c_1}\cdots{A^{d_{n-1}}}_{c_{n-1}}}
{\delta^{e_1\dots e_n}_{f_1\dots f_n}{A^{f_1}}_{e_1}\cdots{A^{f_n}}_{e_n}}$$

Note that all expressions (apart from those with the partial derivative) contain exclusively components of tensors. In these cases we can safely assume that the whole expression is a tensor as well, and we can interpret the indices not as components but as abstract tensor indices.