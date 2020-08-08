---
title: Anabel
description: Anonymous finite elements with automatic derivatives.
...

<h1><img src="img/emtec-4.png" alt="" width=100></img>Anabel</h1>

[![Build Status][travis-image]][travis-link]
[![PyPI Version][pypi-v-image]][pypi-v-link]
[![Commits since latest release][gh-image]][gh-link]

Anonymous finite elements with automatic derivatives.

**Table of contents**

- [Anonymous Elements](#anonymous-elements)
- [Composition, Combinators & Categories](#composition-combinators--categories)
- [Automatic Differentiation](#automatic-differentiation)
- [Installation](#installation)

Anabel is a *functional* library for constructing complex finite element models that leverage several abstractions from the field of deep learning.

## Anonymous Elements

Finite element programs constructed with Anabel are more than modular; they're *anonymous*.

```python
def truss(xyz,E,A):
    return [[]]
```

Explore Anabel's reference element library, [**Elle**](elle), to see more examples.

## Composition, Combinators & Categories

```python
import anabel

model = anabel.compose('graph.yml')
# attractor.f : X,A -> X
#               x => x0 - Df
x = anabel.fixed_point(model.f, model.x0)
```

## Automatic Differentiation

```python
import anabel

model = anabel.compose('graph.yml')

grad_f = anabel.autodiff.jacfwd(model.f, 0)
```

## Installation

    pip install anabel

You can also install the in-development version with:

    pip install https://github.com/claudioperez/anabel/archive/master.zip



[pypi-v-image]: https://img.shields.io/pypi/v/anabel.svg
[pypi-v-link]: https://pypi.org/project/anabel/

[travis-image]: https://api.travis-ci.org/claudioperez/anabel.svg?branch=master
[travis-link]: https://travis-ci.org/claudioperez/anabel

[gh-link]: https://github.com/claudioperez/anabel/compare/v0.0.0...master
[gh-image]: https://img.shields.io/github/commits-since/claudioperez/anabel/v0.0.0.svg
