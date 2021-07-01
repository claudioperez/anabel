
import anabel.backend as anp
from anabel.abstract import ModelComponent, SectionPatch, SectionFiber, SectionLayer
from .patches import RectangularPatch

__all__ = ["PatchedSection"]


class PatchedSection(SectionPatch):
    def __init__(self, *args, origin=None, **kwds):
        ModelComponent.__init__(self, **kwds)
        if origin is None:
            origin = anp.zeros(2)
        self.origin = origin
        self.patches = list(filter(lambda x: isinstance(x,SectionPatch), args))
        self.fibers = list(filter(lambda x: isinstance(x,SectionFiber), args))
        self.layers = list(filter(lambda x: isinstance(x,SectionLayer), args))

    @property
    def area(self):
        return sum(i.area for i in self.patches)
    
    @property
    def centroid(self):
        return sum(i.centroid * i.area for i in self.patches) / self.area

    @property
    def Ix(self):
        centroid = self.centroid[1]
        return sum(p.Ix + p.area * (p.centroid[1] - centroid)**2  for p in self.patches)

    @property
    def Iy(self):
        centroid = self.centroid[0]
        return sum(p.Iy + p.area * (p.centroid[0] - centroid)**2  for p in self.patches)

    def shift(self, shift):
        pass

    def dump_opensees(self, indent=4, depth=0, **kwds):
        tag = self.tag if self.tag else 0
        tab = ' ' * indent * (depth + 1)
        children = tab + f"\n{tab}".join((
                *map(lambda x: x.dump_opensees(), self.patches),
                *map(lambda x: x.dump_opensees(), self.fibers),
                *map(lambda x: x.dump_opensees(), self.layers)
        ))
        return f"""section fiberSec {tag} {{
{children}
}}
""".replace("\n\n","\n")

    def rectangle(self, coord_ll, coord_ur, **kwds):
        """Add a rectangular patch to the section"""
        patch = RectangularPatch(coord_ll, coord_ur, parent=self, **kwds)
        self.patches.append(patch)
        return patch

    rect = rectangle

    def fiber(self, area):
        pass

    def ring(self):
        pass

    def circ(self):
        pass

    def patch(self, patch):
        self.patches.append(patch)
        return patch



    def to_mpl(self, **kwds):
        import matplotlib.collections
        return matplotlib.collections.PatchCollection(
            [p.to_mpl() for p in self.patches],
            facecolor="grey",
            edgecolor="grey",
            alpha=0.3
        )

    def plot(self, ax=None, fig=None, **kwds):
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.set_autoscale_on(True)
            ax.set_aspect(1)
        ax.add_collection(self.to_mpl(**kwds))
        ax.set_xlim(-200, 200)
        ax.set_ylim(-50, 100)
        return fig, ax


