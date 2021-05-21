"""
# Transient

General transformations and utilities for transient modeling.
"""
from anabel.template import template

@template(dim="shape", order="new_order")
def linear_hot(f, df):
    """Add a linear higher order term to a function

    Examples
    --------
    First define some matrices
    ```python
    >>A = jnp.array([[1.0, 0.0, 0.0],
                     [0.0, 4.0, 2.0],
                     [0.0, 2.0, 4.0]])
    >>B = jnp.array([[-0.23939017,  0.58743526, -0.77305379],
                     [ 0.81921268, -0.30515101, -0.48556508],
                     [-0.52113619, -0.74953498, -0.40818426]])
    >>C = A@B
    ```

    Next define a linear function, `f: x -> Ax`:
    ```python
    >>f = anon.dual.wrap(
        lambda x,*args,**kwds: x, A@x, {},
        dim=3
    )
    ```

    Create a new function with a linear higher order term (`ff: x,dx -> f(x) + Bdx`)
    ```python
    >>ff = linear_hot(f, B)

    >>x = dx = jnp.ones((3,1))

    >>f(x)
    (DeviceArray([[1.],
                [1.],
                [1.]], dtype=float32),
    DeviceArray([[1.],
                [6.],
                [6.]], dtype=float32),
    {})

    >>ff((x, dx))
    ((DeviceArray([[1.],
                [1.],
                [1.]], dtype=float32),
    DeviceArray([[1.],
                [1.],
                [1.]], dtype=float32)),
    DeviceArray([[0.57499135],
                [6.0284967 ],
                [4.3211446 ]], dtype=float32),
    {})
    ```

    Studies
    -------
    [elle-0008](/stdy/elle-0008)

    """
    new_order = f.order+1

    yshape = f.shape[1]

    if f.order == 0:
        if isinstance(f.shape[0],tuple):
            xshape = (2, *f.shape[0])
        else:
            xshape = (2, f.shape[0])
        def main(x, *args,**kwds):
            _x, fx, state= f(x[0], *args, **kwds)
            return x, fx + df@x[1], state
    else:
        xshape = (new_order+1, *f.shape[0][1:])
        def main(x, *args, **kwds):
            _x, fx, state = f(x[:-1], *args, **kwds)
            return x, fx + df@x[-1], state
    
    shape = (xshape, yshape)
    state = f.origin[2]
    params = f.params
    return locals()


