
from functools import partial
from fractions import Fraction

import jax.numpy as jnp



def lagrange(i=None,n=None,p=None,x=None):
    r"""
    Lagrange polynomial interpolation.
    Given two 1-D arrays `x` and `w,` returns the Lagrange interpolating
    polynomial through the points ``(x, w)``.
    Warning: This implementation is numerically unstable. Do not expect to
    be able to use more than about 20 points even if they are chosen optimally.
    Parameters
    ----------
    x : array_like
        `x` represents the x-coordinates of a set of datapoints.
    w : array_like
        `w` represents the y-coordinates of a set of datapoints, i.e. f(`x`).
    Returns
    -------
    lagrange : `numpy.poly1d` instance
        The Lagrange interpolating polynomial.
    Examples
    --------
    Interpolate :math:`f(x) = x^3` by 3 points.
    >>> from scipy.interpolate import lagrange
    >>> x = np.array([0, 1, 2])
    >>> y = x**3
    >>> poly = lagrange(x, y)
    Since there are only 3 points, Lagrange polynomial has degree 2. Explicitly,
    it is given by
    .. math::
        \begin{aligned}
            L(x) &= 1\times \frac{x (x - 2)}{-1} + 8\times \frac{x (x-1)}{2} \\
                 &= x (-2 + 3x)
        \end{aligned}
    >>> from numpy.polynomial.polynomial import Polynomial
    >>> Polynomial(poly).coef
    array([ 3., -2.,  0.])

    """

    if x is not None: 
        M = len(x)
    else:
        M = n 
        x = jnp.linspace(-1,1,n)
    cf = jnp.ones(1)
    for k in range(M):
        if k == i: continue
        fac = x[i]-x[k]
        cf = jnp.convolve(cf,   jnp.array([1.0, -x[k]])/fac)
    return cf
