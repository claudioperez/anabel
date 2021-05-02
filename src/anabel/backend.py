"""Atomic operations implemented through the JAX Numpy API.

[`jax.numpy`](https://jax.readthedocs.io/en/latest/jax.numpy.html)

"""
from functools import wraps
import numpy as np
# from anon.conf import _BACKEND 

# backend = _BACKEND['ops']
backend = ["jax"]
try:
    from jax.config import config
    config.update("jax_enable_x64", True)
    from jax.numpy import *
except:
    from numpy import *

@wraps(np.genfromtxt)
def genfromtxt(*args, **kwds):
    return asarray(np.genfromtxt(*args,**kwds))

@wraps(np.empty)
def empty(*args,**kwds):
    return np.empty(*args,**kwds)
