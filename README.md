<h1><img src="images/emtec-4.png" alt="logo" width=100></img>anabel</h1>

[![Travis-CI Build Status](https://api.travis-ci.org/claudioperez/anabel.svg?branch=master)](https://travis-ci.org/claudioperez/anabel)
[![Commits since latest release](https://img.shields.io/github/commits-since/claudioperez/anabel/v0.0.0.svg)](https://github.com/claudioperez/anabel/compare/v0.0.0...master)

Anonymous finite elements with analytic derivatives.

**Table of contents**
- [Installation](#installation)
- [Documentation](#documentation)
- [Development](#development)

## Installation

> `pip install anabel`

You can also install the in-development version with:

> `pip install https://github.com/claudioperez/anabel/archive/master.zip`

## Documentation

$$\operatorname{fix}g(x,a) = x-Df(x,a)^{-1}$$

To use the project:

```python
    import anabel

    attractor = anabel.compose('graph.yml')
    # attractor.f : X,A -> X
    #               x => x0 - Df
    x = anabel.fixed_point(model.f, model.x0)
```

## Development

To run the all tests run:

> `$ tox`

<!-- Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox -->
