---
title: Anabel
description: Finite element neural networks.
...

<h1><img src="img/main.svg" alt="" width=100></img>Anabel</h1>

Finite element neural networks.

[![Build Status][travis-image]][travis-link]
[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Commits since latest release][gh-image]][gh-link]

**Table of contents**

- [Composition, Combinators & Categories](#composition-combinators--categories)
- [Roadmap and Goals](#roadmap-and-goals)
  - [Non-goals](#non-goals)
- [Installation](#installation)

Anabel is a *functional* library for constructing complex finite element models that leverage abstractions from the field of deep learning.

Explore Anabel's reference element library, [**Elle**](elle), to see more examples.

## Composition, Combinators & Categories

```python
import anabel

model = anabel.compose('graph.yml')
# attractor.f : X,A -> X
#               x => x0 - Df
x = anabel.fixed_point(model.f, model.x0)
```

## Roadmap and Goals

This project began as a series of isolated studies into how machine learning abstractions might be useful in finite element analysis problems. Several of these studies proved very successful, and short-term developments are expected to focus on providing a concise set of tools that will allow researchers to reproduce and build upon the most promising of these.

The following is a list of these studies and the tools which were developed for them. Items with a check mark have been successfully ported to the Anabel library.

- [ ] JIT-able truss model with algorithmic stiffness.
- [ ] Use of AD for a FORM analysis of a geometrically nonlinear truss.
  - [x] Differentiable/JIT-able truss element with degenerate Green-Lagrange strain (see notebook using **`elle.truss2d`** element).
  - [ ] Differentiable/JIT-able Newton-Raphson implementation.

### Non-goals

- The purpose of this project is not to apply *machine learning* to FEA.


## Installation

The *basic* Anabel library can be installed from a terminal with the following command:

```bash
pip install anabel
```

This installation includes basic tools for composing neural networks along with some convenient IO utilities. However, both automatic differentiation and JIT capabilities require Google's Jaxlib module which is currently in early development and only packaged for Ubuntu systems. On Windows systems this can be easily overcome by downloading the Ubuntu terminal emulator from Microsoft's app store and enabling the Windows Subsystem for Linux (WSL). The following extended command will install Anabel along with all necessary dependencies for automatic differentiation and JIT compilation:

```bash
pip install anabel[jax]
```

The in-development version can be installed the following command:

    pip install https://github.com/claudioperez/anabel/archive/master.zip

[pypi-v-image]: https://img.shields.io/pypi/v/anabel.svg
[pypi-v-link]: https://pypi.org/project/anabel/

[travis-image]: https://api.travis-ci.org/claudioperez/anabel.svg?branch=master
[travis-link]: https://travis-ci.org/claudioperez/anabel

[gh-link]: https://github.com/claudioperez/anabel/compare/v0.0.0...master
[gh-image]: https://img.shields.io/github/commits-since/claudioperez/anabel/v0.0.0.svg

