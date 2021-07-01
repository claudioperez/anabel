# Claudio Perez
from typing import Tuple

import anabel.backend as anp
from anabel.abstract import SectionPatch

__all__ = ["RectangularPatch"]

class CircularPatch(SectionPatch):
    def __init__(self, origin, radius=None, diameter=None, **kwds):
        """
        diameter [diam] (optional): float
            Circular diameter
        radius (optional): float
            Circular diameter

        """
        if radius is None and diameter is not None:
            radius = 2.0 * diameter

        self.radius = radius
        self.diameter = diameter

        self.area = anp.pi*radius**2
        self.centroid = anp.asarray(origin)

    @property
    def Ix(self):
        """Moment of inertia"""
        return 0.25 * self.radius**4 * anp.pi
    
    @property
    def Iy(self):
        """Moment of inertia"""
        return 0.25 * self.radius**4 * anp.pi

    @property
    def J(self):
        return 0.5 * self.area * self.radius ** 2

    def to_mpl(self):
        import matplotlib.patches
        return matplotlib.patches.Circle(self.origin, self.radius)


class RectangularPatch(SectionPatch):
    """
    ```
      ^ y
      |
    1 +-------------+ 2
      |             | 
      |      +      |-----
      |             |    ^ centroid[1]
    0 +-------------+ 3 ---> x
      |----->|

    centroid[0]
    ```

    """
    no_divs: Tuple[int, int]
    def __init__(self, coord_ll, coord_ur, **kwds):
        super().__init__(**kwds)
        self.no_divs = (0,0)

        self.origin = anp.asarray(coord_ll)

        self.vertices = anp.array([
            coord_ll, 
            (coord_ll[0], coord_ur[1]),
            coord_ur,
            (coord_ll[0], coord_ur[1]),
        ])

        self.centroid = anp.array([
            coord_ur[0]/2 + coord_ll[0]/2,
            coord_ur[1]/2 + coord_ll[1]/2,
        ])

        self.area = (coord_ur[0] - coord_ll[0]) * (coord_ur[1] - coord_ll[1])
    
    @property
    def Ix(self):
        width, height = self.vertices[2] - self.vertices[0]
        return 1/12 * width * height ** 3 
    
    @property
    def Iy(self):
        width, height = self.vertices[2] - self.vertices[0]
        return 1/12 * height * width ** 3

    def dump_opensees(self, **kwds)->str:
        if self.material:
            matTag = self.material.tag
        else:
            matTag = 0
        numSubDivY, numSubDivZ = self.no_divs
        return f"patch rect {matTag} {numSubDivY}"


    def to_mpl(self):
        import matplotlib.patches
        props = {}
        if self.color:
            props.update({"color": self.color})
        return matplotlib.patches.Rectangle(
                self.vertices[0], *(self.vertices[2] - self.vertices[0]), **props
        )



