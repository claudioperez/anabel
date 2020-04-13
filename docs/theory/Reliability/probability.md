# Probability

## Axioms

- Probability is bounded between $0$ and $1,$ i.e., 
  $$0 \leq \operatorname{Pr}(E) \leq 1$$
- Probability of the certain event equals $1,$ i.e., $\operatorname{Pr}(S)=1$
- Probability of the union of two mutually exclusive events is the sum of their probabilities, i.e., 
  $$\operatorname{Pr}\left(E_{1} \cup E_{2}\right)=\operatorname{Pr}\left(E_{1}\right)+\operatorname{Pr}\left(E_{2}\right),$$
  where $E_{1}$ and $E_{2}$ are mutually exclusive.

## Basic Rules

### Addition (Inclusion-Exclusion)

$$\begin{aligned}
\operatorname{Pr}\left(\bigcup_{i} E_{i}\right) &=\sum_{i} \operatorname{Pr}\left(E_{i}\right)-\sum_{i<j} \operatorname{Pr}\left(E_{i} E_{j}\right)+\sum_{i<j<k} \operatorname{Pr}\left(E_{i} E_{j} E_{k}\right)-\cdots \\
&+(-1)^{n-1} \operatorname{Pr}\left(E_{1} E_{2} \cdots E_{n}\right)
\end{aligned}$$

$\operatorname{Pr}\left(E_{1} \cup E_{2}\right)=\operatorname{Pr}\left(E_{1}\right)+\operatorname{Pr}\left(E_{2}\right)-\operatorname{Pr}\left(E_{1} E_{2}\right)$

$\operatorname{Pr}\left(\bigcup_{i} E_{i}\right)=1-\operatorname{Pr}\left(\bar{E}_{1} \bar{E}_{2} \cdots \bar{E}_{n}\right)$

### Conditional Probability

$\operatorname{Pr}\left(A | B\right)=\frac{\operatorname{Pr}\left(A B\right)}{P\left(B\right)}$

### Intersection (Multiplication) 

$\operatorname{Pr}\left(A B\right)=\operatorname{Pr}\left(A | B\right) \operatorname{Pr}\left(B\right)=\operatorname{Pr}\left(B | A\right) \operatorname{Pr}\left(A\right)$

$P(A B C)=P(A) P(B | A) P(C | A B)$

### Total Probability

$$\begin{aligned}
\operatorname{Pr}(A) &=\sum_{i=1}^{n} \operatorname{Pr}\left(A E_{i}\right) \\
&=\sum_{i=1}^{n} \operatorname{Pr}\left(A | E_{i}\right) \operatorname{Pr}\left(E_{i}\right)
\end{aligned}$$

Where events $E_i$ partition the sample space.

### Bayes's Rule

$$\operatorname{Pr}\left(E_{i} | A\right)=\frac{\operatorname{Pr}\left(A | E_{i}\right)}{\operatorname{Pr}(A)} \operatorname{Pr}\left(E_{i}\right)$$

- $\operatorname{Pr}\left(E_{i}\right)$ is known as the *prior* probability
- $\operatorname{Pr}\left(E_{i} | A\right)$ is known as the *posterior* probability
- $\operatorname{Pr}\left(A | E_{i}\right)$ is known as the *likelihood*
- $\operatorname{Pr}\left(A\right)$ is known as the *normalizing factor*

$$\operatorname{Pr}\left(E_{i} | A\right)=\frac{\operatorname{Pr}\left(A | E_{i}\right) \operatorname{Pr}\left(E_{i}\right)}{\sum_{j=1}^{n} \operatorname{Pr}\left(A | E_{j}\right) \operatorname{Pr}\left(E_{j}\right)}$$