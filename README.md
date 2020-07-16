<div align="center">
<img src="https://raw.githubusercontent.com/claudioperez/anabel/master/images/emtec-3.jpg" alt="logo"></img>
</div>

# anabel

[![Travis-CI Build Status](https://api.travis-ci.org/claudioperez/anabel.svg?branch=master)](https://travis-ci.org/claudioperez/anabel)
[![Commits since latest release](https://img.shields.io/github/commits-since/claudioperez/anabel/v0.0.0.svg)](https://github.com/claudioperez/anabel/compare/v0.0.0...master)

Anonymous finite elements with analytic derivatives.

- [anabel](#anabel)
  - [Installation](#installation)
  - [Documentation](#documentation)
  - [Development](#development)

## Installation

> `pip install anabel`

You can also install the in-development version with::

> `pip install https://github.com/claudioperez/anabel/archive/master.zip`


## Documentation

To use the project:

```python
    import anabel
    anabel.unimplemented_function()
```

## Development

To run the all tests run:

> `$ tox`

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
