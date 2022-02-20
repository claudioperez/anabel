---
title: Anabel
description: Finite element neural networks.
...

<h1><img src="img/main.svg" alt="" width=100></img>Anabel</h1>

Finite element neural networks.

------------------

[![Build Status][travis-image]][travis-link]
[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Commits since latest release][gh-image]][gh-link]

**Table of contents**

- [Composition, Combinators & Categories](#composition-combinators--categories)
- [Roadmap and Goals](#roadmap-and-goals)
  - [Non-goals](#non-goals)
- [Installation](#installation)

Anabel is a *functional* library for constructing finite element programs that leverage abstractions from the field of deep learning to develop new models for complex phenomena like inelasticity and geometric nonlinearity.

<!-- Explore Anabel's reference element library, [**Elle**](elle), to see examples.-->

<!--
## Composition, Combinators & Categories

```python
import anabel

f  = anabel.compose('graph.yml')
x0 = anabel.init_domain(f)
# attractor.f : X,A -> X
#               x => x0 - Df
x = anabel.fixed_point(f, x0)
```
-->

## Modules

### numeric

General numerical methods for:
    
    - quadrature
    - optimization

### autodiff

### roots

### ops

### core

<!--
## Roadmap and Goals

This project is a unification of a series of isolated studies into how machine learning abstractions might be useful in finite element analysis problems. Several of these studies proved very successful, and short-term developments are expected to focus on providing a concise set of tools that will allow the reproduction and extended development of the most promising of these studies.

The following is a partial list of some successful studies and the tools which were developed for them. Items with a check mark have been successfully ported from their original implementation to the unified Anabel package.

- [ ] Use of AD and JIT compilation for a FORM analysis of a geometrically nonlinear truss.
  - [x] Differentiable/JIT-able truss element with degenerate Green-Lagrange strain and parameter sensitivities (see notebook using **`elle.truss2d`** element).
  - [ ] Differentiable/JIT-able Newton-Raphson implementation.

### Non-goals

- The purpose of this project is not to apply *machine learning* to FEA.

-->

## Installation

The *base* Anabel package can be installed from a terminal with the following command:

```bash
$ pip install anabel
```

This installation includes basic tools for composing "neural network" -like models along with some convenient IO utilities. However, both automatic differentiation and JIT capabilities require Google's Jaxlib module which is currently in early development and only packaged for Ubuntu systems. On Windows systems this can be easily overcome by downloading the Ubuntu terminal emulator from Microsoft's app store and enabling the Windows Subsystem for Linux (WSL). The following extended command will install Anabel along with all necessary dependencies for automatic differentiation and JIT compilation:

```bash
$ pip install anabel[jax]
```

The in-development version can be installed the following command:

```bash
$ pip install https://github.com/claudioperez/anabel/archive/master.zip
```

[pypi-v-image]: https://img.shields.io/pypi/v/anabel.svg
[pypi-v-link]: https://pypi.org/project/anabel/

[travis-image]: https://api.travis-ci.org/claudioperez/anabel.svg?branch=master
[travis-link]: https://travis-ci.org/claudioperez/anabel

[gh-link]: https://github.com/claudioperez/anabel/compare/v0.0.0...master
[gh-image]: https://img.shields.io/github/commits-since/claudioperez/anabel/v0.0.0.svg

