"""Automatic differentiation API
"""

try:
    from anon.diff import jacx
    from jax import jacfwd, jacrev, grad
except:
    pass


