"""Automatic differentiation API
"""

try:
    from anon.diff import jacx, taylor
    from jax import jacfwd, jacrev, grad
except:
    pass


