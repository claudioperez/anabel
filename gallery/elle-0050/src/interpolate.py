# Claudio Perez
# May 2021
import anabel.backend as anp

def lagrange_t6():
    """
    Quadratic Lagrange polynomial interpolation over a triangle.
    
    Parameters
    ----------
    r,s : float
         coordinates in the natural 
        element.
    
    ```
    s
    ^
    |
    | 
    | 
    ------------> r
    ```
    """
    def main(x,*args): 
        r,s = x.flatten()
        return anp.array([
             (1 - r - s) - 2*r*(1 - r - s) - 2*s*(1 - r - s),
             r - 2*r*(1 - r - s) - 2*r*s,
             s - 2*r*s - 2*s*(1-r-s),
             4*r*(1 - r - s), #6
             4*r*s,
             4*s*(1 - r - s )
        ])
    def jacx(x,*args): 
        r,s = x
        return anp.array([
        [4*r + 4*s - 3, 4*r - 1, 0, -8*r - 4*s + 4, 4*s, -4*s],
        [4*r + 4*s - 3, 0, 4*s - 1, -4*r, 4*r, -4*r - 8*s + 4]])
    main.jacx = jacx
    return main